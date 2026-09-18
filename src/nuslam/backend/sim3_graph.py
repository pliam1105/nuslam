"""§14 — unifying COLMAP and DA3 in one Sim(3) factor graph.

    ############################################################################
    #  The metric-anchor FACTOR DESIGN is core estimator substance (CLAUDE.md  #
    #  §3 + §8): the custom ground-anchor and ego-on-ground residuals and their #
    #  JACOBIANS (the scale-resolution geometry) are DERIVED AND WRITTEN BY THE  #
    #  AUTHOR -- NOT generated. The generic factors (gauge, COLMAP-relative,     #
    #  depth-ratio) are STOCK gtsam factors; this file wires them + the noise    #
    #  knobs, provides the input contracts + prep, and leaves the two custom     #
    #  anchor factors (backend.factors) as stubs.                               #
    ############################################################################

gtsam>=4.3 wraps ``Similarity3`` as a first-class Values-storable / optimizable variable AND the
templated ``PriorFactorSimilarity3`` / ``BetweenFactorSimilarity3`` (the 4.2 RELEASE wheel does NOT --
requirements.txt pins the 4.3 pre-release). So the generic §14.4 factors are STOCK, and only the
metric anchor is custom. Variable layout (§14.2/§14.3):

    H_i = Similarity3(R_i, t_i, s_i)   per keyframe   -- s_i: COLMAP units -> metres
    rho = log r_m                      scalar(s)      -- r_m: DA3 depth -> COLMAP units (one per submap)

    gauge prior      -> PriorFactorSimilarity3(H_0, init)                        (stock)
    COLMAP relative  -> BetweenFactorSimilarity3(H_{i-1}, H_i, (rel_R,rel_t,1))  (stock; the Sim(3)
                        between H_{i-1}^{-1}H_i carries the scale-coupled translation + scale
                        constancy s_i=s_{i-1} natively)
    depth-ratio      -> PriorFactorDouble(rho, log r_hat)                        (stock; sigma from MAD)
    ground anchor    -> GroundAnchorSim3Factor(H_i, rho)    \  CustomFactor -- the §14 novelty,
    ego-on-ground    -> EgoOnGroundFactor(H_i)              /  author core (backend.factors)

World point (§14.3):  X = s_i R_i ( e^{rho} d K^{-1} u ) + t_i ,  camera centre C_i = t_i.

Submap variant (§14 "DA3 runs per window"): overlapping windows, each with its OWN rho_m; adjacent
windows share cameras (shared H_i) that anchor them; the incremental solve is iSAM2.
`Sim3GraphInputs`/`prepare_sim3_inputs`/`build_sim3_graph` are single-window;
`SubmapGraphInputs`/`prepare_submap_inputs`/`build_submap_graph`/`solve_incremental` are the submap
version. Both wire the stock backbone; the ground-anchor/ego CustomFactors are author stubs (OFF in
the anchor-free submap graph).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- input contract
@dataclass
class Sim3GraphInputs:
    """Factor-ready arrays for the §14 Sim(3) graph. All plumbing; produced by
    :func:`prepare_sim3_inputs` from the existing pipeline caches."""
    tokens: list                      # per-frame sample tokens, in graph order
    K: np.ndarray                     # (3, 3) true intrinsics K_true
    sensor2ego: np.ndarray            # (4, 4) camera->ego extrinsic (constant per channel)
    cam_height: float                 # h = sensor2ego[2, 3], metres (ego-on-ground target)

    # --- initial variable values (from the current closed-form estimate) ---
    init_R: np.ndarray                # (N, 3, 3) initial rotation of H_i (camera->world)
    init_t: np.ndarray                # (N, 3)    initial translation of H_i (camera centre, metres)
    init_s: np.ndarray                # (N,)      initial scale s_i (COLMAP units -> metres)
    init_log_r: float                 # initial rho = log r_m (DA3 depth -> COLMAP units)

    # --- COLMAP relative-pose measurements, consecutive frames (§14.4 relative-pose factor) ---
    colmap_rel_R: np.ndarray          # (N-1, 3, 3) R_{i-1}^T R_i in COLMAP's own frame
    colmap_rel_t: np.ndarray          # (N-1, 3)    R_{i-1}^T (c_i - c_{i-1}), COLMAP LOCAL units

    # --- ground-anchor observations per frame: road-pixel rays + DA3 depths (§14.4 ground anchor) ---
    ground_rays: list                 # per-frame (Nk, 3): K^{-1} [u, v, 1] for road pixels
    ground_depths: list               # per-frame (Nk,):   DA3 depth d at those pixels (DA3 units)

    # --- depth-ratio unary on rho (§14.3): median + MAD of per-point z_colmap / z_da3 ---
    depth_ratio_median: float         # r_hat_m  (the median the current pipeline fixes)
    depth_ratio_mad: float            # MAD of the ratios -> sigma_rho (honest noise model, §5.6)
    depth_ratio_n: int                # number of point-observations behind the median

    @property
    def n_frames(self) -> int: return len(self.tokens)

    def save(self, path):
        np.savez(str(path),
                 tokens=np.array(self.tokens), K=self.K, sensor2ego=self.sensor2ego, cam_height=self.cam_height,
                 init_R=self.init_R, init_t=self.init_t, init_s=self.init_s, init_log_r=self.init_log_r,
                 colmap_rel_R=self.colmap_rel_R, colmap_rel_t=self.colmap_rel_t,
                 ground_rays=np.array(self.ground_rays, dtype=object), ground_depths=np.array(self.ground_depths, dtype=object),
                 depth_ratio_median=self.depth_ratio_median, depth_ratio_mad=self.depth_ratio_mad, depth_ratio_n=self.depth_ratio_n)


# ----------------------------------------------------------------------------- submap input contract (§14 "DA3 per window")
@dataclass
class SubmapRecon:
    """One INDEPENDENTLY-reconstructed window (VGGT-SLAM 2.0 style). COLMAP and DA3 both run per
    window, so this submap is in its OWN up-to-scale frame. Its frames are its OWN variables H^m_j
    (an overlap frame is reconstructed in BOTH submaps -> two variables, tied by a submap-alignment
    factor). Local index j indexes into frame_idx; global index = frame_idx[j]."""
    id: int
    frame_idx: np.ndarray             # (n,) global frame indices in this window
    tokens: list                      # (n,) sample tokens

    # --- this window's COLMAP relatives (up-to-scale), for the within-submap §14.4 relative factors ---
    rel_R: np.ndarray                 # (n-1,3,3) R_{j-1}^T R_j
    rel_t: np.ndarray                 # (n-1,3)   R_{j-1}^T (c_j - c_{j-1}), submap-local COLMAP units

    # --- init WORLD placement of this submap's per-frame H^m_j (from the submap-chain init) ---
    init_R: np.ndarray                # (n,3,3) camera->world rotation
    init_t: np.ndarray                # (n,3)   camera centre
    init_s: np.ndarray                # (n,)    scale s init (constant within a submap; author sets convention)

    # --- rho_m depth-ratio (§14.3): log median(z_colmap / z_da3) within this window + robust sigma ---
    log_r: float                      # rho_m init = log median(z_colmap / z_da3)
    log_r_sigma: float                # 1.4826 * MAD(log ratios)  -> sigma for the rho_m unary (log-space)
    n_ratio: int                      # number of point-obs behind it

    # --- DA3 metric point-cloud material per frame (strided full frame): rays + DA3 depths ---
    da3_rays: list                    # per-frame (Nk,3): K^{-1}[u,v,1]
    da3_depths: list                  # per-frame (Nk,):  DA3 depth (metric-up-to-scale via this window's DAQ)

    # --- ROAD-masked subset for the ground anchor (§14.4): road minus sky/veh, subsampled per frame ---
    ground_rays: list                 # per-frame (Mk,3): K^{-1}[u,v,1] for road pixels only
    ground_depths: list               # per-frame (Mk,):  this window's DA3 depth at those road pixels


@dataclass
class SubmapOverlap:
    """The shared cameras between two adjacent submaps -> the §14.4 submap-alignment factor:
    a pure-scaling Sim(3) between-factor (s_hat, I, 0) on the shared H variables, s_hat from the
    DA3(n)/DA3(m) depth ratio at the overlap (log-space, robust sigma)."""
    m: int                            # first submap id
    n: int                            # second submap id (adjacent; usually m+1)
    shared_frame_idx: np.ndarray      # (k,) global frame indices shared by both
    m_local: np.ndarray               # (k,) local indices of the shared frames within submap m
    n_local: np.ndarray               # (k,) local indices within submap n
    log_s: float                      # log median DA3(n)/DA3(m) at the overlap  (the pure-scaling measurement)
    log_s_sigma: float                # 1.4826 * MAD(log ratios)  -> sigma for the alignment factor


@dataclass
class SubmapGraphInputs:
    """Factor-ready inputs for the SUBMAP §14 graph (VGGT-SLAM 2.0 style): a list of independently-
    reconstructed submaps (each with its own per-frame H variables + rho_m) and the overlap graph that
    aligns them by pure scaling. Absolute metric scale is left UNCONSTRAINED (no ground anchor here;
    §14.5/§14.6) -- verify the log-scale gauge freedom afterwards. §4 plumbing; from
    :func:`prepare_submap_inputs`."""
    tokens: list                      # all N distinct frames, graph order (for reference/eval)
    K: np.ndarray                     # (3,3) true intrinsics
    sensor2ego: np.ndarray            # (4,4) camera->ego extrinsic
    cam_height: float                 # sensor2ego[2,3] (for the ego anchor later)
    submaps: list                     # list[SubmapRecon]
    overlaps: list                    # list[SubmapOverlap]
    gt_c2w: np.ndarray = None         # (N,4,4) GT camera->global (ego2global_gt @ sensor2ego), eval only

    @property
    def n_frames(self) -> int: return len(self.tokens)
    @property
    def n_submaps(self) -> int: return len(self.submaps)

    def save(self, path):
        sm = np.array([dict(id=s.id, frame_idx=s.frame_idx, tokens=np.array(s.tokens),
                            rel_R=s.rel_R, rel_t=s.rel_t, init_R=s.init_R, init_t=s.init_t, init_s=s.init_s,
                            log_r=s.log_r, log_r_sigma=s.log_r_sigma, n_ratio=s.n_ratio,
                            da3_rays=np.array(s.da3_rays, dtype=object), da3_depths=np.array(s.da3_depths, dtype=object),
                            ground_rays=np.array(s.ground_rays, dtype=object),
                            ground_depths=np.array(s.ground_depths, dtype=object))
                       for s in self.submaps], dtype=object)
        ov = np.array([dict(m=o.m, n=o.n, shared_frame_idx=o.shared_frame_idx, m_local=o.m_local,
                            n_local=o.n_local, log_s=o.log_s, log_s_sigma=o.log_s_sigma)
                       for o in self.overlaps], dtype=object)
        np.savez(str(path), tokens=np.array(self.tokens), K=self.K, sensor2ego=self.sensor2ego,
                 cam_height=self.cam_height, submaps=sm, overlaps=ov,
                 gt_c2w=(self.gt_c2w if self.gt_c2w is not None else np.zeros(0)))

    @classmethod
    def load(cls, path) -> "SubmapGraphInputs":
        """Inverse of :func:`save` -- rebuild the dataclasses from the npz (pure numpy, no nuScenes)."""
        d = np.load(str(path), allow_pickle=True)
        submaps = [SubmapRecon(id=int(s["id"]), frame_idx=s["frame_idx"], tokens=list(s["tokens"]),
                               rel_R=s["rel_R"], rel_t=s["rel_t"], init_R=s["init_R"], init_t=s["init_t"],
                               init_s=s["init_s"], log_r=float(s["log_r"]), log_r_sigma=float(s["log_r_sigma"]),
                               n_ratio=int(s["n_ratio"]), da3_rays=list(s["da3_rays"]), da3_depths=list(s["da3_depths"]),
                               ground_rays=list(s.get("ground_rays", [])) if hasattr(s, "get") else list(s["ground_rays"]),
                               ground_depths=list(s.get("ground_depths", [])) if hasattr(s, "get") else list(s["ground_depths"]))
                   for s in d["submaps"]]
        overlaps = [SubmapOverlap(m=int(o["m"]), n=int(o["n"]), shared_frame_idx=o["shared_frame_idx"],
                                  m_local=o["m_local"], n_local=o["n_local"], log_s=float(o["log_s"]),
                                  log_s_sigma=float(o["log_s_sigma"])) for o in d["overlaps"]]
        gt = d["gt_c2w"] if "gt_c2w" in d.files else None
        if gt is not None and gt.size == 0: gt = None
        return cls(tokens=list(d["tokens"]), K=d["K"], sensor2ego=d["sensor2ego"],
                   cam_height=float(d["cam_height"]), submaps=submaps, overlaps=overlaps, gt_c2w=gt)


# ----------------------------------------------------------------------------- input prep (§4 plumbing, complete)
def prepare_sim3_inputs(scene: str = "scene-0061", *, cache_root: str = "out/frontend_cache",
                        dataroot: str = "data/nuscenes", version: str = "v1.0-mini",
                        colmap_sparse: str = "out/colmap/sparse/0",
                        road_tag: str = "sam3-road-0.35", sky_tag: str = "sam3-sky-0.5",
                        veh_tag: str = "sam3-moving-vehicle-0.5", stride: int = 8) -> Sim3GraphInputs:
    """Assemble the §14 factor inputs from the existing caches. Pure plumbing: it
    reads the DA3 metric upgrade, the DA3 depth, the aligned COLMAP poses, the raw
    COLMAP sparse reconstruction, and the road/sky/vehicle masks; it computes the
    COLMAP relative poses, the per-frame road-pixel rays + DA3 depths, and the
    per-point DA3/COLMAP depth-ratio statistics. No estimation happens here."""
    import pycolmap
    from ..data import NuScenesMonoSource
    from ..frontend import cache

    src = NuScenesMonoSource(dataroot, version, camera="CAM_FRONT"); kfs = src.load_scene(scene)
    mu = cache.load_metric_upgrade(cache_root, scene); dep = cache.load_depth(cache_root, scene)
    pidx = {t: i for i, t in enumerate(mu.tokens)}
    frames = [kf for kf in kfs if kf.token in dep and kf.token in pidx]
    tokens = [kf.token for kf in frames]
    K = np.asarray(mu.K_true, np.float64); Kinv = np.linalg.inv(K)
    s2e = frames[0].calib.sensor2ego.matrix(); h = float(s2e[2, 3])

    # aligned COLMAP poses (camera->world, metric global frame) for init H_i + relative measurements
    cg = np.load(f"{cache_root}/{scene}/colmap_poses_global.npz", allow_pickle=True)
    cmap = {str(t): P for t, P in zip(cg["tokens"], cg["poses"])}
    colmap_pose = np.stack([cmap[t] for t in tokens]).astype(np.float64)     # (N,4,4) camera->world
    init_R = colmap_pose[:, :3, :3].copy(); init_t = colmap_pose[:, :3, 3].copy()
    init_s = np.ones(len(frames))                                            # author sets the true s_i init convention
    rel_R = np.stack([colmap_pose[i - 1, :3, :3].T @ colmap_pose[i, :3, :3] for i in range(1, len(frames))])
    rel_t = np.stack([colmap_pose[i - 1, :3, :3].T @ (colmap_pose[i, :3, 3] - colmap_pose[i - 1, :3, 3]) for i in range(1, len(frames))])

    # per-frame road-pixel rays + DA3 depths (road minus sky/veh), strided
    road = cache.load_seg_masks(cache_root, scene, road_tag)
    sky = cache.load_seg_masks(cache_root, scene, sky_tag); veh = cache.load_seg_masks(cache_root, scene, veh_tag)
    ground_rays, ground_depths = [], []
    for kf in frames:
        d = dep[kf.token].depth; H, W = d.shape
        keep = road[kf.token] & ~sky[kf.token] & ~veh[kf.token] & (d > 0)
        vv, uu = np.where(keep); sub = slice(None, None, max(1, len(vv) // 4000)) if len(vv) else slice(0, 0)
        vv, uu = vv[sub][::stride], uu[sub][::stride]
        rays = (Kinv @ np.stack([uu, vv, np.ones_like(uu)], 0)).T             # (Nk,3) K^-1 [u,v,1]
        ground_rays.append(rays.astype(np.float32)); ground_depths.append(d[vv, uu].astype(np.float32))

    # per-point depth ratio r_m = z_colmap / z_da3 from the RAW COLMAP sparse vs DA3 depth
    rec = pycolmap.Reconstruction(colmap_sparse)
    name2tok = {im.name: im.name for im in rec.images.values()}              # placeholder; resolve below
    # map each COLMAP image to a frame token: raw image names are f"{token}.png" or index-based
    imgs = list(rec.images.values())
    def img_tok(im):
        n = im.name.rsplit(".", 1)[0]
        return n if n in pidx else (tokens[int(n[1:])] if n[1:].isdigit() and int(n[1:]) < len(tokens) else None)
    ratios = []
    for p in rec.points3D.values():
        X = np.append(p.xyz, 1.0)
        for el in p.track.elements:
            im = rec.images[el.image_id]; tok = img_tok(im)
            if tok is None or tok not in dep: continue
            rig = im.cam_from_world; rig = rig() if callable(rig) else rig
            M = np.asarray(rig.matrix()) if hasattr(rig, "matrix") else np.hstack([np.asarray(rig.rotation.matrix()), np.asarray(rig.translation)[:, None]])
            zc = (M @ X)[2]                                                   # COLMAP raw depth of the point
            u, v = im.points2D[el.point2D_idx].xy; dd = dep[tok].depth
            if 0 <= int(v) < dd.shape[0] and 0 <= int(u) < dd.shape[1]:
                zg = dd[int(v), int(u)]                                       # DA3 depth at the pixel
                if zc > 1e-6 and zg > 1e-6: ratios.append(zc / zg)
    ratios = np.asarray(ratios)
    med = float(np.median(ratios)) if len(ratios) else float("nan")
    mad = float(np.median(np.abs(ratios - med)) * 1.4826) if len(ratios) else float("nan")

    return Sim3GraphInputs(
        tokens=tokens, K=K, sensor2ego=s2e, cam_height=h,
        init_R=init_R, init_t=init_t, init_s=init_s, init_log_r=float(np.log(med)) if med > 0 else 0.0,
        colmap_rel_R=rel_R, colmap_rel_t=rel_t, ground_rays=ground_rays, ground_depths=ground_depths,
        depth_ratio_median=med, depth_ratio_mad=mad, depth_ratio_n=int(len(ratios)))


# ----------------------------------------------------------------------------- submap input prep (§4 plumbing, complete)
def _windows(n: int, window: int, step: int) -> list:
    """Overlapping frame windows covering [0, n): starts at 0, step, 2*step, ... clamped to end."""
    starts = list(range(0, max(1, n - window + 1), step))
    if not starts or starts[-1] + window < n: starts.append(max(0, n - window))
    return [np.arange(a, min(a + window, n)) for a in sorted(set(starts))]


def _log_ratio_stats(ratios) -> tuple:
    """(log-median, robust log-sigma, n) for positive ratios: sigma = 1.4826*MAD(log r) (§14.3, log-space)."""
    r = np.asarray([x for x in ratios if x > 0], np.float64)
    if r.size == 0: return 0.0, 1.0, 0
    lr = np.log(r); med = float(np.median(lr))
    sig = float(1.4826 * np.median(np.abs(lr - med)))
    return med, max(sig, 1e-3), int(r.size)


def run_submap_colmap(kfs_window, sky, veh, K, work_dir):
    """Masked COLMAP SfM on ONE window -- no GPS/global alignment, so the window is in its OWN
    up-to-scale frame (VGGT-SLAM style independence). Returns {local_j: cam->world (4x4, up-to-scale)}
    for the registered frames + the pycolmap Reconstruction. This is the SLOW per-submap step; call it
    when ready (short forward-driving windows can under-register -- see the endpoint windows)."""
    import pycolmap
    from PIL import Image
    work = Path(work_dir); (work / "images").mkdir(parents=True, exist_ok=True); (work / "masks").mkdir(exist_ok=True)
    for j, kf in enumerate(kfs_window):
        Image.fromarray(np.asarray(kf.image())).save(work / "images" / f"{j:03d}.png")
        keep = ~(sky[kf.token] | veh[kf.token])
        Image.fromarray((keep.astype(np.uint8) * 255)).save(work / "masks" / f"{j:03d}.png.png")
    db = work / "database.db"
    if db.exists(): db.unlink()
    ropts = pycolmap.ImageReaderOptions(); ropts.mask_path = str(work / "masks")
    ropts.camera_model = "PINHOLE"; ropts.camera_params = f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}"
    eopts = pycolmap.FeatureExtractionOptions(); eopts.sift.max_num_features = 12000
    pycolmap.extract_features(db, work / "images", camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ropts, extraction_options=eopts)
    pycolmap.match_exhaustive(db)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = opts.ba_refine_principal_point = opts.ba_refine_extra_params = False
    # tuned for short FORWARD-DRIVING windows: the defaults reject every init pair because motion is
    # ~100% forward (init_max_forward_motion 0.95) and parallax is low (init_min_tri_angle 16, inliers 100).
    opts.mapper.init_max_forward_motion = 1.0      # allow fully-forward init pairs (the key fix)
    opts.mapper.init_min_num_inliers = 30          # short masked window -> fewer inliers
    opts.mapper.init_min_tri_angle = 1.0           # low forward-motion parallax
    opts.mapper.init_max_error = 8.0               # looser epipolar for the init pair
    opts.mapper.init_max_reg_trials = 5
    opts.mapper.abs_pose_min_num_inliers = 15
    (work / "sparse").mkdir(exist_ok=True)
    maps = pycolmap.incremental_mapping(db, work / "images", work / "sparse", options=opts)
    if not maps: return {}, None
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    c2w = {}
    for im in rec.images.values():
        j = int(im.name.rsplit(".", 1)[0]); T = np.eye(4); T[:3, :4] = im.cam_from_world().inverse().matrix()
        c2w[j] = T
    return c2w, rec


def submap_daq(dep_window, tokens_window, K_true, *, m_weight: float = 1.0):
    """Per-window DAQ metric upgrade (author core, called as infra): DA3 (K_da3, R, t) per frame ->
    metric-up-to-scale cam->world poses. Mirrors scripts/run_metric_upgrade for one window."""
    from ..recon import (normalized_projective_cameras, build_daq_system, solve_daq,
                         plane_at_infinity, rectifying_homography, metric_cameras, decompose_metric_camera)
    K_da3 = np.stack([dep_window[t].intrinsic for t in tokens_window])
    R = np.stack([dep_window[t].extrinsic[:3, :3] for t in tokens_window])
    t = np.stack([dep_window[t].extrinsic[:3, 3] for t in tokens_window])
    M = np.linalg.inv(K_true) @ K_da3[0]                              # reference-camera mismatch
    cams = normalized_projective_cameras(K_true, K_da3, R, t)
    A = build_daq_system(cams, m_prior=(M if m_weight > 0 else None), m_weight=m_weight)
    H = rectifying_homography(plane_at_infinity(solve_daq(A)), M)
    poses = {}
    for j, P in enumerate(metric_cameras(cams, H)):
        Kj, Rj, tj = decompose_metric_camera(P); T = np.eye(4); T[:3, :3] = Rj.T; T[:3, 3] = -Rj.T @ tj
        poses[j] = T
    return poses


def prepare_submap_inputs(scene: str = "scene-0061", *, cache_root: str = "out/frontend_cache",
                          dataroot: str = "data/nuscenes", version: str = "v1.0-mini",
                          submap_cache: str = "out/submap_cache",
                          road_tag: str = "sam3-road-0.35", sky_tag: str = "sam3-sky-0.5",
                          veh_tag: str = "sam3-moving-vehicle-0.5", ground_stride: int = 4,
                          window: int = 11, step: int = 7) -> SubmapGraphInputs:
    """Assemble the SUBMAP §14 inputs (VGGT-SLAM 2.0) from the REAL per-window reconstructions.

    Requires the per-window DA3+DAQ+COLMAP cache ``{submap_cache}/sm{m}.npz`` (one INDEPENDENT
    reconstruction per window: re-run DA3-Base, DAQ metric upgrade, tuned masked COLMAP). There is NO
    global-COLMAP fallback -- if a window is missing/under-registered this raises. Each submap keeps its
    own up-to-scale frame; they are chain-PLACED into a common world by composing ONE shared frame's full
    6-DOF pose (SE(3), no scale fit; init only). The overlap ``log_s`` is the genuine median DA3(n)/DA3(m)
    ratio at the shared frames.
    Pure plumbing, no estimation."""
    from ..data import NuScenesMonoSource
    from ..frontend import cache as fcache

    src = NuScenesMonoSource(dataroot, version, camera="CAM_FRONT"); kfs = src.load_scene(scene)
    dep = fcache.load_depth(cache_root, scene)
    frames = [kf for kf in kfs if kf.token in dep]; tokens = [kf.token for kf in frames]; N = len(frames)
    K = np.asarray(frames[0].calib.intrinsic, np.float64)
    s2e = frames[0].calib.sensor2ego.matrix(); h = float(s2e[2, 3])
    gt_c2w = np.stack([kf.ego2global_gt.matrix() @ kf.calib.sensor2ego.matrix() for kf in frames])  # (N,4,4) GT cam->global

    # road masks for the ground anchor (road minus sky/veh); a helper selects road DA3 rays for a frame
    road = fcache.load_seg_masks(cache_root, scene, road_tag)
    sky = fcache.load_seg_masks(cache_root, scene, sky_tag); veh = fcache.load_seg_masks(cache_root, scene, veh_tag)
    def _road_select(rays, depths, tok):
        rays = np.asarray(rays, float); depths = np.asarray(depths, float)
        uv = (K @ rays.T)                                        # rays = K^-1[u,v,1] -> uv = [u,v,1]
        u = np.round(uv[0]).astype(int); v = np.round(uv[1]).astype(int)
        Hm, Wm = road[tok].shape
        ib = (u >= 0) & (u < Wm) & (v >= 0) & (v < Hm)
        keep = np.zeros(len(rays), bool)
        keep[ib] = (road[tok][v[ib], u[ib]] & ~sky[tok][v[ib], u[ib]] & ~veh[tok][v[ib], u[ib]] & (depths[ib] > 0))
        idx = np.where(keep)[0][::ground_stride]
        return rays[idx].astype(np.float32), depths[idx].astype(np.float32)

    wins = _windows(N, window, step); sc = Path(submap_cache)
    missing = [m for m in range(len(wins)) if not (sc / f"sm{m}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"per-window recon cache missing for submaps {missing} under {sc} -- run the per-submap "
            f"DA3+DAQ+COLMAP pass first (scratchpad/submap_recon.py). No global-COLMAP fallback.")
    recons = []
    for m in range(len(wins)):
        d = np.load(sc / f"sm{m}.npz", allow_pickle=True)
        c2w = d["colmap_c2w"].astype(np.float64)
        if not np.isfinite(c2w).all():
            raise ValueError(f"submap {m}: COLMAP under-registered ({int(d['nreg'])}/{len(d['tokens'])}) -- "
                             f"NaN poses. Re-run that window's COLMAP; no fallback.")
        recons.append(dict(idx=np.asarray(d["frame_idx"]), toks=list(d["tokens"]), c2w=c2w,
                           da3_rays=list(d["da3_rays"]), da3_depths=list(d["da3_depths"]), ratios=list(d["ratios"])))

    # chain-place each submap into a common world by composing ONE shared frame's full 6-DOF (SE(3)) pose.
    # World = submap-0's own COLMAP frame. Placement is rotation+translation only (no scale fit): the SE(3)
    # that carries that one shared camera's local pose onto its already-placed world pose, applied rigidly to
    # the whole submap. init_s stays 1 -- the per-submap relative scale is NOT fitted here; it is resolved by
    # the submap-alignment factor (via rho + log_s). Init only; does not affect the optimum.
    world_pose = {}; placed = []
    for m, rc in enumerate(recons):
        c2w = rc["c2w"]; idx = list(rc["idx"])
        if m == 0:
            R_a, t_a = np.eye(3), np.zeros(3)
        else:
            g = next(int(gg) for gg in idx if int(gg) in world_pose)   # one shared frame (the first)
            jl = idx.index(g); Rl, Cl = c2w[jl, :3, :3], c2w[jl, :3, 3]
            Rw_g, Cw_g = world_pose[g]                                  # its full world pose already placed
            R_a = Rw_g @ Rl.T                                           # SE(3) compose: world = R_a @ local + t_a
            t_a = Cw_g - R_a @ Cl
        Rw = R_a[None] @ c2w[:, :3, :3]
        Cw = c2w[:, :3, 3] @ R_a.T + t_a                                # s_a = 1
        placed.append((Rw, Cw, np.ones(len(idx))))
        for j, g in enumerate(idx): world_pose.setdefault(int(g), (Rw[j], Cw[j]))

    submaps = []
    for m, rc in enumerate(recons):
        idx = rc["idx"]; c2w = rc["c2w"]; Rw, Cw, sw = placed[m]
        rel_R = np.stack([c2w[j - 1, :3, :3].T @ c2w[j, :3, :3] for j in range(1, len(idx))])
        rel_t = np.stack([c2w[j - 1, :3, :3].T @ (c2w[j, :3, 3] - c2w[j - 1, :3, 3]) for j in range(1, len(idx))])
        log_r, sig, nr = _log_ratio_stats(rc["ratios"])
        g_rays, g_depths = [], []                                        # road-masked ground rays per frame
        for j, tok in enumerate(rc["toks"]):
            gr, gd = _road_select(rc["da3_rays"][j], rc["da3_depths"][j], tok)
            g_rays.append(gr); g_depths.append(gd)
        submaps.append(SubmapRecon(id=m, frame_idx=idx, tokens=list(rc["toks"]), rel_R=rel_R, rel_t=rel_t,
                                   init_R=Rw, init_t=Cw, init_s=sw, log_r=log_r, log_r_sigma=sig, n_ratio=nr,
                                   da3_rays=rc["da3_rays"], da3_depths=rc["da3_depths"],
                                   ground_rays=g_rays, ground_depths=g_depths))

    # overlaps -> submap-alignment. log_s = median DA3(n)/DA3(m) at shared frames (pixel-aligned strided depths).
    overlaps = []
    for a in range(len(submaps)):
        for b in range(a + 1, len(submaps)):
            sh = np.intersect1d(submaps[a].frame_idx, submaps[b].frame_idx)
            if not sh.size: continue
            ma = {int(g): j for j, g in enumerate(submaps[a].frame_idx)}
            mb = {int(g): j for j, g in enumerate(submaps[b].frame_idx)}
            ratios = []
            for g in sh:
                zm = submaps[a].da3_depths[ma[int(g)]]; zn = submaps[b].da3_depths[mb[int(g)]]
                msk = (zm > 0) & (zn > 0); ratios.extend((zn[msk] / zm[msk]).tolist())    # DA3(n)/DA3(m)
            log_s, ssig, _ = _log_ratio_stats(ratios)
            overlaps.append(SubmapOverlap(m=a, n=b, shared_frame_idx=sh,
                                          m_local=np.array([ma[int(g)] for g in sh]),
                                          n_local=np.array([mb[int(g)] for g in sh]),
                                          log_s=log_s, log_s_sigma=ssig))

    return SubmapGraphInputs(tokens=tokens, K=K, sensor2ego=s2e, cam_height=h, submaps=submaps,
                             overlaps=overlaps, gt_c2w=gt_c2w)


# ----------------------------------------------------------------------------- graph build harness
def _H(i: int) -> int:
    import gtsam
    return int(gtsam.symbol("h", i))          # Similarity3 key for frame i

def _RHO() -> int:
    import gtsam
    return int(gtsam.symbol("r", 0))          # scalar rho key

def _sim3(R, C, s):
    """gtsam Similarity3 whose CAMERA CENTRE is C, i.e. transformFrom(0) = C, with rotation R, scale s.
    gtsam's action is p -> s(R p + t) so centre = s*t; the §14.3 math uses X = sRp + t with centre = t.
    To place the centre at C we must store t = C/s (identity for s=1, so single-window init_s=ones is unaffected)."""
    import gtsam
    return gtsam.Similarity3(gtsam.Rot3(np.asarray(R, float)), np.asarray(C, float).reshape(3) / float(s), float(s))

def _rho_sigma(median: float, mad: float) -> float:
    """Data-driven 1-sigma on rho = log r from the ratio MAD (§5.6 honest noise); floored."""
    s = abs(mad / median) if median else 1.0     # d(log r) ~ dr / r
    return float(max(s, 1e-3))

def _gauge_noise(scale_sigma: float = 1e-3):
    """Gauge prior noise on the first pose. GTSAM's sim(3) tangent = [omega(3), rho(3), lambda(1)] with
    lambda = log s LAST, and scale couples into translation (GetV), so this must be a Diagonal on the
    genuine 7-DoF variable -- not a zeroed column.

    §14.5 says leave scale FREE (sigma 1e6) so the anchors determine it. But with NO anchor yet, that
    leaves the system underconstrained in log-scale (the marginal is indeterminate). So FOR NOW we also
    gauge-fix scale with a small sigma (default 1e-3) to keep it well-posed; the recovered global scale
    is then arbitrary (= init), pending the anchors.

    LATER, once the ground/ego anchors are added: pass scale_sigma=1e6 (free, §14.5), AND replace this
    whole gauge prior with a HORIZONTAL-translation + rotation-only prior -- the ego anchor pins t_z and
    the anchors pin scale, so only (x, y, yaw/rotation) remain gauge. That is a CUSTOM factor, since scale
    couples into translation on Sim(3) and cannot be a zeroed column."""
    import gtsam
    return gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6 + [scale_sigma]))

def build_sim3_graph(inputs: Sim3GraphInputs, *, rel_sigma: float = 1e-2):
    """Build the single-window §14 graph + initial Values from `inputs`.

    Wires the STOCK backbone (§14.4): the §14.5 gauge prior on H_0 (SE(3)-tight, scale-FREE), the
    per-consecutive COLMAP ``BetweenFactorSimilarity3`` relatives, and the ``PriorFactorDouble`` depth-
    ratio unary on rho. The metric-anchor CustomFactors (``GroundAnchorSim3Factor`` / ``EgoOnGroundFactor``)
    are the author's core, added once written. ``rel_sigma`` + the rho sigma (from the MAD) are noise knobs.

    Absolute scale: §14.5 leaves it free for the anchors to fix, but with no anchor yet that is
    underconstrained -- so the gauge prior gauge-FIXES scale for now (``_gauge_noise`` default), giving an
    arbitrary (= init) global scale. Pass a free-scale gauge + the anchors later (see ``_gauge_noise``).

    Returns ``(graph, values)`` ready for :func:`solve`.
    """
    import gtsam

    graph = gtsam.NonlinearFactorGraph(); values = gtsam.Values()
    # --- variables + initial values: per-frame Similarity3 H_i + scalar rho ---
    for i in range(inputs.n_frames):
        values.insert(_H(i), _sim3(inputs.init_R[i], inputs.init_t[i], inputs.init_s[i]))
    values.insert(_RHO(), float(inputs.init_log_r))

    # --- STOCK backbone (§14.4) ---
    rel_noise = gtsam.noiseModel.Isotropic.Sigma(7, rel_sigma)       # COLMAP relative confidence (author knob)
    rho_noise = gtsam.noiseModel.Isotropic.Sigma(1, _rho_sigma(inputs.depth_ratio_median, inputs.depth_ratio_mad))
    graph.add(gtsam.PriorFactorSimilarity3(                          # §14.5 gauge: frame only, scale FREE
        _H(0), _sim3(inputs.init_R[0], inputs.init_t[0], inputs.init_s[0]), _gauge_noise()))
    for i in range(1, inputs.n_frames):
        meas = _sim3(inputs.colmap_rel_R[i - 1], inputs.colmap_rel_t[i - 1], 1.0)   # (1, rel_R, rel_t) = H_{i-1}^{-1}H_i
        graph.add(gtsam.BetweenFactorSimilarity3(_H(i - 1), _H(i), meas, rel_noise))
    graph.add(gtsam.PriorFactorDouble(_RHO(), float(inputs.init_log_r), rho_noise))

    # ======================================================================== §3 AUTHOR: metric-anchor CustomFactors
    # The backbone pins the FRAME gauge (not scale), chains the COLMAP relatives, and priors rho. What
    # makes it metric-FROM-SEMANTICS is the anchor -- author core (backend.factors), the only hand-written
    # Jacobians (§14.8): GroundAnchorSim3Factor(_H(i), _RHO()) + EgoOnGroundFactor(_H(i)). With the anchors
    # present every scale is observable (§14.6). NOTE (author): once anchors are on, the §14.5 gauge prior
    # is replaced by a horizontal-translation + rotation-only prior (ego anchor already pins t_z, the
    # anchors pin scale) -- a CUSTOM factor, since scale couples into translation on Sim(3).
    return graph, values


def solve(graph, values, *, max_iters: int = 100, verbose: bool = False):
    """Levenberg–Marquardt driver (plumbing). Runs once the graph has factors."""
    import gtsam
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iters)
    if verbose: params.setVerbosityLM("SUMMARY")
    return gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()


# --------------------------------------------------------------------- submap graph build harness
def _HM(m: int, j: int) -> int:
    import gtsam
    return int(gtsam.symbol("h", m * 1000 + j))   # Sim3 for submap m's LOCAL frame j (overlap frame -> two variables)

def _RHO_M(m: int) -> int:
    import gtsam
    return int(gtsam.symbol("r", m))              # per-submap depth-ratio scalar rho_m = log r_m (§14, DA3 per window)

def _align_rot(a, b) -> np.ndarray:
    """Rotation mapping unit vector a onto unit vector b (Rodrigues from the cross product)."""
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    b = np.asarray(b, float); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); n = np.linalg.norm(v)
    if n < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def metric_init(inputs, values, *, down_axis=(0., 1., 0.)) -> tuple:
    """ONE-TIME GT-free metric init of the anchored graph's initial `values` (§14.5): (1) rotate the whole
    world so the FIRST camera is upright (its down axis -> world -z), then (2) set the global scale so the
    camera floats cam_height above the road, from the median (camera_z - ground_point_z) over all frames.
    Places the init in the metric basin so the solve does not slide into the s->0 collapse. Applies the
    global similarity (alpha, R_align, 0) to every H_i in place; returns (R_align, alpha)."""
    s0 = inputs.submaps[0]
    R0 = np.asarray(values.atSimilarity3(_HM(s0.id, 0)).rotation().matrix())
    R_align = _align_rot(R0 @ np.asarray(down_axis, float), np.array([0., 0., -1.]))   # first cam down -> world down
    diffs = []                                                    # up-to-scale camera height above the road
    for sm in inputs.submaps:
        for j in range(len(sm.frame_idx)):
            S = values.atSimilarity3(_HM(sm.id, j))
            C = R_align @ np.asarray(S.transformFrom(np.zeros(3))); R = R_align @ np.asarray(S.rotation().matrix())
            rays = np.asarray(sm.ground_rays[j], float); depths = np.asarray(sm.ground_depths[j], float)
            if len(rays) == 0: continue
            gp = (R @ (np.exp(sm.log_r) * depths[:, None] * rays).T).T + C            # ground points in world (s=1)
            diffs.append(C[2] - gp[:, 2])
    dz = float(np.median(np.concatenate(diffs))) if diffs else 1.0
    alpha = inputs.cam_height / dz if abs(dz) > 1e-6 else 1.0
    for sm in inputs.submaps:                                     # apply global similarity (alpha, R_align, 0)
        for j in range(len(sm.frame_idx)):
            S = values.atSimilarity3(_HM(sm.id, j))
            C = np.asarray(S.transformFrom(np.zeros(3))); R = np.asarray(S.rotation().matrix())
            values.update(_HM(sm.id, j), _sim3(R_align @ R, alpha * (R_align @ C), alpha * S.scale()))
    return R_align, alpha


def _ransac_ground_inliers(P: np.ndarray, *, thresh: float = 0.10, iters: int = 200, seed: int = 0):
    """RANSAC plane fit to camera-frame road points P (n,3); returns an inlier boolean mask. Frame-local
    and gauge-independent -- rejects mask bleed (curbs, low objects) before they poison the ground anchor
    (CLAUDE.md guardrail: RANSAC, not least-squares). Returns all-False if n < 3."""
    n = len(P)
    if n < 3:
        return np.zeros(n, bool)
    rng = np.random.default_rng(seed); best = np.zeros(n, bool)
    for _ in range(iters):
        i, j, k = rng.choice(n, 3, replace=False)
        nrm = np.cross(P[j] - P[i], P[k] - P[i]); nn = np.linalg.norm(nrm)
        if nn < 1e-9: continue
        nrm /= nn; d = -nrm @ P[i]
        inl = np.abs(P @ nrm + d) < thresh
        if inl.sum() > best.sum(): best = inl
    return best


def build_submap_graph(inputs: SubmapGraphInputs, *, rel_sigma: float = 1e-2, align_pose_sigma: float = 3e-2,
                       use_anchors: bool = False, ground_sigma: float = 0.6, ego_sigma: float = 0.20,
                       horiz_pos_sigma: float = 1e-2, horiz_head_sigma: float = 1e-3,
                       first_pose_rot_sigma: float = None, apply_metric_init: bool = True,
                       max_ground_per_frame: int = 100, ground_ransac_thresh: float = 0.10):
    """Build the SUBMAP §14 graph (VGGT-SLAM 2.0 style) + initial Values from `inputs`.

    Each submap is INDEPENDENT: its frames are its OWN ``Similarity3`` variables ``H^m_j``, so an
    overlap frame has TWO variables (one per submap). Factors (all STOCK; §14.4):

      * gauge prior on H^0_0 -- fixes the frame gauge (and scale, for now; see :func:`_gauge_noise`);
      * within-submap COLMAP relatives ``BetweenFactorSimilarity3(H^m_{j-1}, H^m_j, (1, rel_R, rel_t))``;
      * per-submap depth-ratio ``PriorFactorDouble(rho_m, log r_hat_m)``, sigma = 1.4826*MAD(log r);
      * SUBMAP-ALIGNMENT ``SubmapAlignmentFactor(H^m_a, H^n_b, rho_m, rho_n, log_s)`` on each shared camera --
        the author's §3 CustomFactor (backend.factors): a PURE-SCALING residual (§14.2) tight on the 6 SE(3)
        dims (the shared camera coincides), whose s_hat = exp((rho_m - rho_n) - log_s) is formed from the rho
        VARIABLES so the two-scale coupling is EXACT (not the stock-between init-rho approximation).

    The metric-anchor CustomFactors (per frame) are the author's core; off here. Returns ``(graph, values)``.
    """
    import gtsam

    graph = gtsam.NonlinearFactorGraph(); values = gtsam.Values()
    # --- variables: per-(submap, frame) Similarity3 H^m_j (overlap frame -> two) + per-submap rho_m ---
    for sm in inputs.submaps:
        for j in range(len(sm.frame_idx)):
            values.insert(_HM(sm.id, j), _sim3(sm.init_R[j], sm.init_t[j], sm.init_s[j]))
        values.insert(_RHO_M(sm.id), float(sm.log_r))
    if use_anchors and apply_metric_init:                        # place the init in the metric basin (GT-free)
        metric_init(inputs, values)

    from nuslam.backend.factors import (SubmapAlignmentFactor, HorizontalGaugeFactor, EgoOnGroundFactor,
                                         GroundAnchorSim3Factor, UprightGaugeFactor)
    rel_noise = gtsam.noiseModel.Isotropic.Sigma(7, rel_sigma)
    s0 = inputs.submaps[0]                                            # gauge on the very first pose
    if use_anchors:
        # anchors present -> horizontal-translation + heading gauge only (x,y,yaw); ground sets z/roll/pitch,
        # ego sets scale. Replaces the scale-fixing prior, which would fight the ground for z/roll/pitch.
        horiz_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([horiz_pos_sigma, horiz_pos_sigma, horiz_head_sigma, horiz_head_sigma]))
        graph.add(HorizontalGaugeFactor(_HM(s0.id, 0), horiz_noise).as_custom_factor())
        # soft UPRIGHT constraint on the first pose's DOWN axis (roll/pitch only). Removes the zero-cost
        # collapse minimum: at s->0 the ground factors vanish and free the global roll/pitch, letting
        # rotation null the ego lever. Pinning the down axis toward vertical denies that freedom, soft
        # enough not to fight the ground's roll/pitch at the true (near-upright) solution. Scale-, yaw- and
        # position-invariant, so it never touches what the other anchors resolve.
        if first_pose_rot_sigma is not None:   # OFF by default: the metric init handles the collapse basin,
            # so the upright factor is redundant (identical ATE with/without). Kept as an option for robustness.
            graph.add(UprightGaugeFactor(_HM(s0.id, 0),
                                         gtsam.noiseModel.Isotropic.Sigma(2, first_pose_rot_sigma)).as_custom_factor())
    else:                                                            # anchor-free: full scale-fixing gauge prior
        graph.add(gtsam.PriorFactorSimilarity3(_HM(s0.id, 0), _sim3(s0.init_R[0], s0.init_t[0], s0.init_s[0]), _gauge_noise()))
    for sm in inputs.submaps:
        for j in range(1, len(sm.frame_idx)):                        # within-submap COLMAP relatives
            meas = _sim3(sm.rel_R[j - 1], sm.rel_t[j - 1], 1.0)      # (1, rel_R, rel_t) = H^m_{j-1}^{-1} H^m_j
            graph.add(gtsam.BetweenFactorSimilarity3(_HM(sm.id, j - 1), _HM(sm.id, j), meas, rel_noise))
        graph.add(gtsam.PriorFactorDouble(_RHO_M(sm.id), float(sm.log_r),      # per-window depth-ratio (log-space MAD)
                                          gtsam.noiseModel.Isotropic.Sigma(1, sm.log_r_sigma)))
    for o in inputs.overlaps:                                        # submap-alignment: pure scaling, EXACT
        # The measurement s_hat = s_n/s_m at the shared camera is NOT the raw DA3 ratio: the two-scale model
        # (§14.3) makes s_m e^{rho_m} z_da3_m = s_n e^{rho_n} z_da3_n at a shared point, so s_n/s_m =
        # e^{rho_m-rho_n}*(z_da3_m/z_da3_n) = exp((rho_m - rho_n) - log_s). SubmapAlignmentFactor forms that
        # s_hat from the rho_m/rho_n VARIABLES inside its residual (so the coupling is exact, not the init-rho
        # approximation of a stock BetweenFactorSimilarity3); we pass only the constant log_s = log DA3(n)/DA3(m).
        align_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([align_pose_sigma] * 6 + [max(o.log_s_sigma, 1e-3)]))
        for ml, nl in zip(o.m_local, o.n_local):
            graph.add(SubmapAlignmentFactor(_HM(o.m, int(ml)), _HM(o.n, int(nl)),
                                            _RHO_M(o.m), _RHO_M(o.n), o.log_s, align_noise).as_custom_factor())

    # ==== metric anchors, per submap+frame (§14.4): ego-height (scale) + RANSAC-gated ground (z/roll/pitch).
    if use_anchors:
        ego_noise = gtsam.noiseModel.Isotropic.Sigma(1, ego_sigma)
        huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)   # robust against residual road-mask bleed
        s2e = inputs.sensor2ego
        for sm in inputs.submaps:
            for j in range(len(sm.frame_idx)):
                graph.add(EgoOnGroundFactor(_HM(sm.id, j), s2e, ego_noise).as_custom_factor())
                rays = np.asarray(sm.ground_rays[j], float); depths = np.asarray(sm.ground_depths[j], float)
                if len(rays) < 3: continue                       # skip weakly-supported frames
                P = depths[:, None] * rays                       # camera-frame road points (metric up to window scale)
                inl = _ransac_ground_inliers(P, thresh=ground_ransac_thresh)
                idx = np.where(inl)[0]
                if len(idx) > max_ground_per_frame:              # subsample the inliers to keep the graph light
                    idx = idx[np.linspace(0, len(idx) - 1, max_ground_per_frame).astype(int)]
                if len(idx) == 0: continue
                # NORMALIZE by point count: scale sigma by sqrt(N) so the frame's N ground factors carry a
                # FIXED total information (= one constraint at ground_sigma), independent of how many road
                # pixels survived. Without this the ground weight grows with N and drowns the per-frame ego.
                gsig = ground_sigma * np.sqrt(len(idx))
                gnoise = gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Isotropic.Sigma(1, gsig))
                for k in idx:
                    graph.add(GroundAnchorSim3Factor(_HM(sm.id, j), _RHO_M(sm.id),
                                                     rays[k], float(depths[k]), gnoise).as_custom_factor())
    return graph, values


def submap_world_poses(inputs: SubmapGraphInputs, values) -> dict:
    """Per-submap solved world poses {submap_id: (n,4,4)} from a Values (each submap's own H^m_j)."""
    out = {}
    for sm in inputs.submaps:
        Ps = []
        for j in range(len(sm.frame_idx)):
            S = values.atSimilarity3(_HM(sm.id, j)); T = np.eye(4)
            T[:3, :3] = np.asarray(S.rotation().matrix())
            T[:3, 3] = np.asarray(S.transformFrom(np.zeros(3)))    # actual camera centre = s*t (see _sim3)
            Ps.append(T)
        out[sm.id] = np.stack(Ps)
    return out


def solve_incremental(inputs: SubmapGraphInputs, *, relinearize_skip: int = 1):
    """iSAM2 driver stub for the INCREMENTAL submap solve (§14 "incremental SAM").

    HARNESS ONLY: feed the graph submap-by-submap. For each new submap m, build a
    ``gtsam.NonlinearFactorGraph`` + ``Values`` with its NEW variables (its H^m_j + rho_m) and factors
    from :func:`build_submap_graph` restricted to that window -- the gauge only for submap 0, the
    within-submap relatives, the depth-ratio prior, and the submap-alignment factors to the PREVIOUS
    submap's overlap variables (already in iSAM) -- then ``isam.update(new_graph, new_values)``. The
    incremental factor set + any metric-anchor CustomFactors are author core (§8); this stub only
    constructs the ISAM2 object."""
    import gtsam
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    _ = relinearize_skip
    raise NotImplementedError(
        "incremental submap solve is author core (§8): drive `isam.update(new_graph, new_values)` per "
        "submap with the stock backbone + submap-alignment factors. This stub only constructs ISAM2.")
