"""Gaussian-splat reconstruction: fit a set of 3D Gaussians to the keyframes.

Left unimplemented by design -- this is where the reconstruction itself is built:
the Gaussian representation and its reparametrization (means, log-scales,
logit-opacities, normalized wxyz quaternions, colours; exp / sigmoid / normalize),
initialization from the metric seed cloud, the optimization loop (per-parameter-group
Adam and its schedule), the gsplat ``rasterization(...)`` calls (``render_mode="RGB+ED"``
for the expected-depth ground constraint), the se(3) pose retraction on the view
matrices, and the losses (photometric L1 + D-SSIM, and the ground/wheel anchor on the
rendered depth) with their weighting.

The surrounding pipeline (``scripts/run_gs.py``) prepares the inputs and calls
``train_gaussians``, then logs and scores the result through the visualization and
evaluation helpers. Only the entry-point signature is fixed here so that pipeline has
something to call.
"""
from __future__ import annotations

import numpy as np
import torch
import gsplat
import torchvision.transforms.functional as F

from nuslam.pointcloud import nn_distances

from nuslam.eval import render

def train_gaussians(
    init_xyz: np.ndarray,       # (M, 3) metric seed points -> Gaussian means
    init_rgb: np.ndarray,       # (M, 3) uint8 seed colours
    poses: np.ndarray,          # (N, 4, 4) metric camera->world (view matrices = inv(poses))
    K: np.ndarray,              # (3, 3) intrinsics (shared across views)
    images,                     # sequence of (H, W, 3) images, one per view
    train_idx: np.ndarray,      # views to optimize on (the rest are held out for novel-view scoring)
    lr_for: dict,               # set of LRs for the different params
    num_iters: int = 10000,     # number of iterations to optimize 3DGS for
    batch_size: int = 1,         # batch size for 3DGS optimization
    structural_lambda: float = 0.5, # the weight of the structural loss in 3DGS optimization
    ssim_kernel_size: int = 11, # the size of the kernel used for SSIM
    ssim_sigma: float = 1.5,    # the sigma used for the SSIM kernel
    ssim_k1: float = 0.01,      # the K1 used for SSIM
    ssim_k2: float = 0.03,      # the K2 used for SSIM
    ssim_l: float = 1.0,        # the L used for SSIM
    log_every: int = 0,         # call on_log every this many steps (0 = never)
    on_log=None,                # observability callback: on_log(step, loss, photo, dssim, gaussians, render)
    init_gaussians: dict | None = None,  # warm start: if given, seed params from this checkpoint
                                         # ({means, scales, quats, opacities, sh}) instead of the
                                         # point cloud -- invert the activations (log_scales=log(scales),
                                         # opacities_logits=logit(opacities), colors=sh, quats as-is).
    clip_scales_to_knn: bool = False,    # warm start only: clip oversized checkpoint scales down to
                                         # the local kNN spacing (salvages a blurry big-scale checkpoint).
    masks=None,                          # optional per-view (H,W) bool keep-masks (True=use pixel);
                                         # sky pixels set False so the loss ignores them and no sky
                                         # Gaussians are grown to reconstruct sky.
    optimize_poses: bool = False,        # refine the camera poses jointly through the rasterizer,
                                         # optimizing a raw quaternion + translation per view (quats
                                         # normalized at use, like the Gaussian quats; no Lie retraction).
                                         # SE(3) only for input/output. Off by default: poses stay
                                         # frozen at the recovered solution. Parametrization is core
                                         # geometry -- seam below.
):
    """Fit the Gaussians to the images and return the optimized set.

    The optimization is developed by extending this code as capabilities are added
    (poses frozen at the recovered solution first, then freed through the rasterizer,
    then a ground/wheel anchor on the rendered surface). Return the optimized set
    (means, scales, quaternions, opacities, colours) together with a render(view)
    method for scoring; the exact shape follows the representation.
    """
    # specifying the parameters to be optimized. scales are sized to the local point
    # spacing (mean kNN distance) so Gaussians start ~as big as their neighbourhood,
    # not an arbitrary fixed size -- a zeros log-scale would be exp(0)=1 m per Gaussian.
    def _cu(a): return torch.tensor(np.asarray(a), dtype=torch.float32, device="cuda")
    if init_gaussians is not None:
        # warm start: rebuild the params from a saved checkpoint, inverting the stored
        # activations (log_scales=log(scales), opacities_logits=logit(opacities), colors=sh).
        scales = np.asarray(init_gaussians["scales"])
        if clip_scales_to_knn:  # cap oversized blobs at the local kNN spacing
            knn = np.clip(nn_distances(init_gaussians["means"]), 1e-4, None)      # (N,)
            scales = np.minimum(scales, knn[:, None])
        means = torch.nn.Parameter(_cu(init_gaussians["means"]))
        log_scales = torch.nn.Parameter(torch.log(_cu(scales).clamp_min(1e-12)))
        opacities_logits = torch.nn.Parameter(torch.logit(_cu(init_gaussians["opacities"]).clamp(1e-6, 1 - 1e-6)))
        quats = torch.nn.Parameter(_cu(init_gaussians["quats"]))
        colors_sh_coeff = torch.nn.Parameter(_cu(init_gaussians["sh"]))
    else:
        knn = np.clip(nn_distances(init_xyz), 1e-4, None)                         # (M,) local spacing
        means = torch.nn.Parameter(torch.tensor(init_xyz, dtype=torch.float32).to("cuda"))
        quats = torch.nn.Parameter(torch.stack([torch.ones((init_xyz.shape[0],), dtype=torch.float32),
                                                torch.zeros((init_xyz.shape[0],), dtype=torch.float32),
                                                torch.zeros((init_xyz.shape[0],), dtype=torch.float32),
                                                torch.zeros((init_xyz.shape[0],), dtype=torch.float32)], dim=1).to("cuda"))
        log_scales = torch.nn.Parameter(_cu(np.log(knn)[:, None].repeat(3, 1)))   # isotropic, ~neighbour spacing
        opacities_logits = torch.nn.Parameter(torch.zeros((init_xyz.shape[0],), dtype=torch.float32).to("cuda"))
        colors_sh_coeff = torch.nn.Parameter(torch.zeros((init_xyz.shape[0], 4, 3), dtype=torch.float32).to("cuda")) # degrees 0,1 -> 4 coeffs
        colors_sh_coeff.data[:,0,:] = torch.tensor((init_rgb.astype(np.float32)/255.0-0.5)*2*np.sqrt(np.pi)) # initialize degree 0 (homogeneous color) with the rgb
    viewmats = torch.nn.Parameter(torch.linalg.inv(torch.tensor(poses[train_idx], dtype=torch.float32).to("cuda")))
    Ks = torch.tensor([K]*len(train_idx), dtype=torch.float32).to("cuda")
    width, height = images[0].shape[1], images[0].shape[0]
    gt_img = torch.tensor(images)[train_idx].to("cuda").permute(0,3,1,2).to(torch.float32)/255.0 # (N,3,H,W)
    keep_img = None if masks is None else \
        torch.tensor(np.stack(masks))[train_idx].to("cuda").float()[:, None]  # (N,1,H,W), 1=non-sky

    params = torch.nn.ParameterDict({
        "means": means,
        "quats": quats,
        "scales": log_scales,
        "opacities": opacities_logits,
        "colors": colors_sh_coeff,
        # "viewmats": viewmats, # frozen for now
    })

    optimizers = {k: torch.optim.Adam([p], lr=lr_for[k]) for k, p in params.items()}

    if optimize_poses:
        # SEAM (core geometry, to implement): free the camera poses through the rasterizer.
        # Parametrize each view's pose as a raw quaternion q_i + translation t_i (normalize q_i at
        # use, as with the Gaussian quats -- no Lie-algebra delta / exp retraction), initialized from
        # the input SE(3) `viewmats`. Build the viewmats from (q_i, t_i) each step for the
        # rasterization call; gradient path (q, t) -> viewmats -> rasterization.
        #   Keep q_i, t_i in a SEPARATE torch.optim.Adam (lr from lr_for["pose_quats"/"pose_trans"]),
        #   NOT in `params`/`optimizers` above: MCMCStrategy densification (step_post_backward)
        #   indexes every param in that dict by Gaussian id and would resize/corrupt per-view poses.
        #   Step this optimizer manually in the loop (zero_grad before, step after backward).
        # Read the refined poses back out as SE(3) via the return. Left unimplemented by design.
        raise NotImplementedError(
            "optimize_poses: pose refinement through the rasterizer is not implemented yet "
            "(raw per-view quaternion + translation in a separate optimizer, viewmats rebuilt each step)."
        )

    # auto-densification strategy
    strategy = gsplat.MCMCStrategy(cap_max=5_000_000)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    def render(pose: np.ndarray, K: np.ndarray):
        with torch.no_grad():
            pose = torch.tensor(pose, dtype=torch.float32, device="cuda")
            K = torch.tensor(K, dtype=torch.float32, device="cuda")
            render_colors, render_alphas, info = gsplat.rasterization(params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), params["colors"], torch.linalg.inv(pose).reshape(1,4,4), K.reshape(1,3,3), width, height, render_mode="RGB", sh_degree=1)
        return render_colors[0].clamp(0,1).cpu().numpy()

    for step in range(num_iters):
        # pick a random subset of views
        step_batch = torch.randperm(len(train_idx))[:min(len(train_idx), batch_size)]
        batch_gt = gt_img[step_batch]
        mk = keep_img[step_batch] if keep_img is not None else None   # (b,1,H,W) non-sky, or None

        # rasterization forward pass
        render_colors_e_depths, render_alphas, info = gsplat.rasterization(params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), params["colors"], viewmats[step_batch], Ks[step_batch], width, height, render_mode="RGB+ED", sh_degree=1, packed=True)
        
        render_colors = render_colors_e_depths[..., :3].permute(0,3,1,2) # (N,3,H,W)
        render_depths = render_colors_e_depths[..., 3:] # (N.H,W,3)

        # pre-backward densification strategy step
        strategy.step_pre_backward(params, optimizers, state, step, info, packed=True)
        
        # photometric loss (masked: only non-sky pixels contribute)
        if mk is not None:
            photometric_loss = ((render_colors - batch_gt).abs() * mk).sum() / (mk.sum() * 3 + 1e-8)
        else:
            photometric_loss = (render_colors - batch_gt).abs().mean() # mean over pixels

        # DSSIM = 1 - SSIM, structural loss. Zero sky in both so it doesn't reward sky structure.
        rc = render_colors if mk is None else render_colors * mk
        gt = batch_gt if mk is None else batch_gt * mk
        m_x = F.gaussian_blur(rc, ssim_kernel_size, ssim_sigma)
        m_y = F.gaussian_blur(gt, ssim_kernel_size, ssim_sigma)
        s_x = F.gaussian_blur((rc-m_x)**2, ssim_kernel_size, ssim_sigma)
        s_y = F.gaussian_blur((gt-m_y)**2, ssim_kernel_size, ssim_sigma)
        s_xy = F.gaussian_blur((rc-m_x)*(gt-m_y), ssim_kernel_size, ssim_sigma)

        C1 = (ssim_k1*ssim_l)**2
        C2 = (ssim_k2*ssim_l)**2

        dssim = 1 - (((2*m_x*m_y+C1)*(2*s_xy+C2))/((m_x**2+m_y**2+C1)*(s_x+s_y+C2))).mean()

        # total loss
        loss = (1-structural_lambda)*photometric_loss + structural_lambda*dssim

        loss.backward() # backprop

        strategy.step_post_backward(params, optimizers, state, step, info, lr=lr_for["means"])

        for o in optimizers.values():
            o.step()
            o.zero_grad()

        # observability (viz / logging) -- handled by the caller's callback
        if on_log is not None and log_every and step % log_every == 0:
            gaussians = (params["means"], torch.exp(params["scales"]), params["quats"],
                         torch.sigmoid(params["opacities"]), params["colors"])
            on_log(step, loss.item(), photometric_loss.item(), dssim.item(), gaussians, render)

    return (params["means"], torch.exp(params["scales"]), params["quats"], torch.sigmoid(params["opacities"]), params["colors"]), render