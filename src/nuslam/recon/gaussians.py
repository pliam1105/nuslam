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
from gsplat.strategy import ops as _gs_ops   # prune Gaussians + their optimizer/strategy state in lockstep
import torchvision.transforms.functional as F

from nuslam.pointcloud import nn_distances

from nuslam.eval import render

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

@torch.no_grad()
def _offscene_prune_mask(means, viewmats, Ks, empty_img, width, height, prune_range):
    """(N,) bool, True = prune. A Gaussian centroid is off-scene if it is:
      (a) invisible in EVERY view (behind the camera or projecting out of bounds everywhere);
      (b) farther than ``prune_range`` metres from EVERY camera centre (skipped if prune_range<=0);
      (c) projecting onto a should-be-empty pixel (``empty_img``: sky OR far-range, NOT vehicle) in
          ANY view -- those regions are supposed to be empty/black, so a Gaussian landing there even
          once is a floater in confirmed-empty space. (Vehicles are excluded: real content sits
          behind the mask, so a Gaussian projecting onto a vehicle may be legitimate in other views.)
    Projection is looped over views to keep memory flat at ~5M Gaussians."""
    N, V = means.shape[0], viewmats.shape[0]
    dev = means.device
    seen = torch.zeros(N, device=dev)
    hit_empty = torch.zeros(N, dtype=torch.bool, device=dev)
    mind = torch.full((N,), float("inf"), device=dev)
    centres = torch.linalg.inv(viewmats)[:, :3, 3]                          # (V,3) world camera centres
    for v in range(V):
        Xc = means @ viewmats[v, :3, :3].T + viewmats[v, :3, 3]             # (N,3) camera coords
        uvw = Xc @ Ks[v].T                                                  # (N,3)
        u = uvw[:, 0] / uvw[:, 2]; y = uvw[:, 1] / uvw[:, 2]
        inb = (Xc[:, 2] > 0) & (u >= 0) & (u < width) & (y >= 0) & (y < height)
        seen += inb.float()
        if empty_img is not None:
            # sanitize before indexing: z~0 gives nan/inf u,y -> .long() would be out of bounds (CUDA
            # assert). Those points are ~inb=False anyway, so index them at 0 and they get masked out.
            ui = torch.nan_to_num(u, 0.0, 0.0, 0.0).clamp(0, width - 1).long()
            yi = torch.nan_to_num(y, 0.0, 0.0, 0.0).clamp(0, height - 1).long()
            is_empty = empty_img[v, 0, yi, ui] > 0.5                        # projects onto sky/far here?
            hit_empty |= inb & is_empty                                     # ANY view
        if prune_range > 0:
            mind = torch.minimum(mind, torch.linalg.norm(means - centres[v], dim=1))
    prune = seen == 0                                                       # (a)
    if empty_img is not None:
        prune = prune | hit_empty                                          # (c)
    if prune_range > 0:
        prune = prune | (mind > prune_range)                              # (b)
    return prune

def viewmats_from_qt(translations: torch.tensor, quaternions: torch.tensor):
    viewmats = torch.zeros((translations.shape[0], 4, 4), dtype=torch.float32, device="cuda")
    viewmats[:, :3, :3] = quaternion_to_matrix(quaternions)
    viewmats[:, :3, 3] = translations
    viewmats[:, 3, 3] = 1.0
    return viewmats

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
    pose_trans_reg: float = 0.0,         # weight for a pose-translation regularizer (e.g. deviation from
                                         # the init translation); loss term author-added at the seam. 0=off.
    pose_quat_reg: float = 0.0,          # weight for a pose-rotation regularizer on the NORMALIZED
                                         # quaternion; loss term author-added at the seam. 0=off.
    first_pose_trans_reg: float = 0.0,   # weight for a HARD prior pinning the FIRST view's translation to
                                         # its init (anchors the global gauge); loss author-added. 0=off.
    first_pose_quat_reg: float = 0.0,    # weight for a HARD prior pinning the FIRST view's rotation
                                         # (normalized quaternion) to its init; loss author-added. 0=off.
    opacity_reg: float = 0.0,            # MCMC opacity-sparsity weight lambda_o (L1 on sigmoid(opacities)).
    scale_reg: float = 0.0,              # MCMC covariance weight lambda_Sigma (L1 on exp(scales)).
                                         # Loss terms are author-added at the "total loss" seam; ~0.01 typical.
    depths=None,                         # optional per-view (H,W) metric depth (DA3, same units as means)
                                         # to supervise the rendered expected depth. Aligned with images.
    depth_lambda: float = 0.0,           # weight of the expected-depth loss (author-added at the seam).
    sky_masks=None,                      # optional per-view (H,W) bool sky masks: supervise those pixels
                                         # toward black (sky has nothing behind it -> the black background).
    sky_lambda: float = 0.0,             # weight of the sky->black loss (author-added at the seam).
    prune_offscene: bool = False,        # every prune_every steps, remove Gaussians whose centroids are
                                         # off-scene (invisible in all views, beyond prune_range from every
                                         # camera, or projecting into the non-kept/black region in most
                                         # views). MCMC's refill-to-cap re-adds the freed budget in-scene.
    prune_range: float = 0.0,            # metres: prune centroids farther than this from EVERY camera
                                         # (<=0 disables the range criterion; visibility/black still apply).
    prune_every: int = 100,              # cadence (steps) of the off-scene prune. No warmup: starts at step 0.
):
    """Fit the Gaussians to the images and return the optimized set.

    The optimization is developed by extending this code as capabilities are added
    (poses frozen at the recovered solution first, then freed through the rasterizer,
    then a ground/wheel anchor on the rendered surface). Return the optimized set
    (means, scales, quaternions, opacities, colours) together with a render(view)
    method for scoring; the exact shape follows the representation.
    """
    # ---- recenter the world origin at the scene (train-camera centroid). In the global nuScenes
    # frame the scene sits ~1 km from origin, so the world->cam translation |t_wc| ~ 1 km and a
    # sub-degree rotation error swings the camera centre (C = -R^T t_wc) by metres -- badly
    # conditioned for pose optimization. Rendering is translation-equivariant, so shifting all means
    # and all camera centres by one offset changes nothing but the numbers; we optimize in this
    # local frame and transform means + refined poses back to the global frame on return/at the seams.
    scene_center = np.asarray(poses[train_idx][:, :3, 3].mean(0), dtype=np.float32)   # (3,) global offset
    poses = poses.copy(); poses[:, :3, 3] -= scene_center
    if init_gaussians is not None:
        init_gaussians = {**init_gaussians,
                          "means": np.asarray(init_gaussians["means"], np.float32) - scene_center}
    else:
        init_xyz = np.asarray(init_xyz, np.float32) - scene_center
    center_t = torch.tensor(scene_center, dtype=torch.float32, device="cuda")         # local->global on the seams

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
    Ks = torch.tensor([K]*len(train_idx), dtype=torch.float32).to("cuda")
    width, height = images[0].shape[1], images[0].shape[0]
    gt_img = torch.tensor(images)[train_idx].to("cuda").permute(0,3,1,2).to(torch.float32)/255.0 # (N,3,H,W)
    keep_img = None if masks is None else \
        torch.tensor(np.stack(masks))[train_idx].to("cuda").float()[:, None]  # (N,1,H,W), 1=non-sky
    gt_depth = None if depths is None else \
        torch.tensor(np.stack(depths))[train_idx].to("cuda").to(torch.float32)[:, None]  # (N,1,H,W) metric
    sky_img = None if sky_masks is None else \
        torch.tensor(np.stack(sky_masks))[train_idx].to("cuda").float()[:, None]  # (N,1,H,W), 1=sky

    params = torch.nn.ParameterDict({
        "means": means,
        "quats": quats,
        "scales": log_scales,
        "opacities": opacities_logits,
        "colors": colors_sh_coeff,
    })

    optimizers = {k: torch.optim.Adam([p], lr=lr_for[k]) for k, p in params.items()}

    viewmats = torch.linalg.inv(torch.tensor(poses[train_idx], dtype=torch.float32).to("cuda"))
    viewmats /= viewmats[:,3:,3:]
    init_translations = viewmats[:, :3, 3].clone()
    translations = init_translations.clone()
    init_quaternions = matrix_to_quaternion(viewmats[:, :3, :3]).clone()
    quaternions = init_quaternions.clone()

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
        translations = torch.nn.Parameter(translations)
        quaternions = torch.nn.Parameter(quaternions)

        optim_translations = torch.optim.Adam([translations], lr=lr_for["pose_trans"])
        optim_quaternions = torch.optim.Adam([quaternions], lr=lr_for["pose_quats"])

    # auto-densification strategy
    strategy = gsplat.MCMCStrategy(cap_max=5_000_000)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    def render(pose: np.ndarray, K: np.ndarray):
        # callers pass a GLOBAL camera->world pose; shift it into the local (recentered) frame the
        # means live in before rendering.
        with torch.no_grad():
            pose = torch.tensor(pose, dtype=torch.float32, device="cuda").clone()
            pose[:3, 3] -= center_t
            K = torch.tensor(K, dtype=torch.float32, device="cuda")
            render_colors, render_alphas, info = gsplat.rasterization(params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), params["colors"], torch.linalg.inv(pose).reshape(1,4,4), K.reshape(1,3,3), width, height, render_mode="RGB", sh_degree=1)
        return render_colors[0].clamp(0,1).cpu().numpy()

    for step in range(num_iters):
        # pick a random subset of views
        step_batch = torch.randperm(len(train_idx))[:min(len(train_idx), batch_size)]
        batch_gt = gt_img[step_batch]
        mk = keep_img[step_batch] if keep_img is not None else None   # (b,1,H,W) non-sky, or None

        # rasterization forward pass
        render_colors_e_depths, render_alphas, info = gsplat.rasterization(params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), params["colors"], viewmats_from_qt(translations[step_batch], quaternions[step_batch]), Ks[step_batch], width, height, render_mode="RGB+ED", sh_degree=1, packed=True)
        
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
        # SEAM (author, §3d): add the MCMC regularizers here using the supplied weights, e.g.
        #   loss += opacity_reg * torch.sigmoid(params["opacities"]).mean()   # opacity L1 (sparsity)
        #   loss += scale_reg   * torch.exp(params["scales"]).mean()          # covariance L1 (sqrt-eig = scales)
        # opacity_reg / scale_reg default 0.0 (no-op) until wired.
        loss = (1-structural_lambda)*photometric_loss + structural_lambda*dssim + \
            opacity_reg*torch.sigmoid(params["opacities"]).mean() + \
            scale_reg*torch.exp(params["scales"]).mean() + \
            pose_trans_reg*torch.square(translations - init_translations).mean() + \
            pose_quat_reg*torch.square(quaternions/quaternions.norm(keepdim=True, dim=1) - init_quaternions/init_quaternions.norm(keepdim=True, dim=1)).mean() + \
            first_pose_trans_reg*torch.square(translations[0]-init_translations[0]).mean() + \
            first_pose_quat_reg*torch.square(quaternions[0]/quaternions[0].norm(keepdim=True) - init_quaternions[0]/init_quaternions[0].norm(keepdim=True)).mean()

        # depth-supervision seam (author, §3d): supervise the rendered expected depth toward the DA3
        # metric depth where valid + non-masked. render_depths is (b,H,W,1), gt_depth[step_batch] is
        # (b,1,H,W). e.g. masked L1 (also gate out zero/invalid DA3 depth as needed):
        #   rd = render_depths.permute(0, 3, 1, 2)
        #   w  = mk if mk is not None else torch.ones_like(rd)
        #   loss += depth_lambda * ((rd - gt_depth[step_batch]).abs() * w).sum() / (w.sum() + 1e-8)
        if depth_lambda > 0 and gt_depth is not None:
            rd = render_depths.permute(0, 3, 1, 2)
            w  = mk if mk is not None else torch.ones_like(rd)
            w = w * (gt_depth[step_batch] > 0)
            loss += depth_lambda * ((rd - gt_depth[step_batch]).abs() * w).sum() / (w.sum() + 1e-8)

        # sky->black seam (author, §3d): push the render toward black on sky pixels (sky has nothing
        # behind it, so black = the background). sky_img[step_batch] is (b,1,H,W), render_colors (b,3,H,W).
        # e.g. L1 to black over sky pixels:
        #   sb = sky_img[step_batch]
        #   loss += sky_lambda * (render_colors.abs() * sb).sum() / (sb.sum() * 3 + 1e-8)

        if sky_lambda > 0 and sky_img is not None:
            sb = sky_img[step_batch]
            loss += sky_lambda * (render_colors.abs() * sb).sum() / (sb.sum() * 3 + 1e-8)

        loss.backward() # backprop

        strategy.step_post_backward(params, optimizers, state, step, info, lr=lr_for["means"])

        for o in optimizers.values():
            o.step()
            o.zero_grad()

        if optimize_poses:   # the pose optimizers only exist when poses are being refined
            optim_translations.step()
            optim_quaternions.step()
            optim_translations.zero_grad()
            optim_quaternions.zero_grad()

        # off-scene prune: drop Gaussians whose centroids are invisible / out of range / only on
        # non-kept pixels, in lockstep across params + optimizer + strategy state. No warmup (from
        # step 0); MCMC's step_post_backward refills to cap_max next refine, re-adding the freed
        # budget by sampling in-scene Gaussians -- so this doubles as relocation.
        if prune_offscene and step % prune_every == 0:
            pm = _offscene_prune_mask(params["means"].detach(), viewmats_from_qt(translations, quaternions), Ks,
                                      sky_img, width, height, prune_range)   # sky_img = sky ∪ far (black region)
            if bool(pm.any()):
                # prune params + their Adam states in lockstep. NOT gsplat.ops.remove: that also
                # indexes the strategy `state` per Gaussian, but MCMC's state is only a fixed (51,51)
                # `binoms` table (not per-Gaussian) -> out-of-bounds. MCMC itself prunes via this same
                # _update_param_with_optimizer and never touches state per Gaussian.
                sel = torch.where(~pm)[0]                              # indices to keep
                _gs_ops._update_param_with_optimizer(
                    lambda name, p: torch.nn.Parameter(p[sel], requires_grad=p.requires_grad),
                    lambda name, v: v[sel], params, optimizers)

        # observability (viz / logging) -- handled by the caller's callback. Means are un-recentered
        # (+ center_t) so snapshots / rrd land in the global frame the caller works in.
        if on_log is not None and log_every and step % log_every == 0:
            gaussians = (params["means"] + center_t, torch.exp(params["scales"]), params["quats"],
                         torch.sigmoid(params["opacities"]), params["colors"])
            # current train-view poses in the GLOBAL frame (camera->world), for live drift + snapshots;
            # None when poses are frozen.
            cur_poses = None
            if optimize_poses:
                with torch.no_grad():
                    cp = torch.linalg.inv(viewmats_from_qt(translations, quaternions))
                    cp[:, :3, 3] += center_t
                    cur_poses = cp.cpu().numpy()
            on_log(step, loss.item(), photometric_loss.item(), dssim.item(), gaussians, render, cur_poses)

    # refined camera->world poses for the train views (None when poses were frozen). translations/
    # quaternions parametrize world->cam (viewmats) in the LOCAL frame, so invert and add back the
    # scene offset to return the global camera->world convention of the input `poses`.
    refined_poses = None
    if optimize_poses:
        with torch.no_grad():
            refined_poses = torch.linalg.inv(viewmats_from_qt(translations, quaternions))
            refined_poses[:, :3, 3] += center_t
            refined_poses = refined_poses.cpu().numpy()
    # un-recenter the returned means back to the global frame (everything else is frame-invariant).
    return (params["means"] + center_t, torch.exp(params["scales"]), params["quats"],
            torch.sigmoid(params["opacities"]), params["colors"]), render, refined_poses