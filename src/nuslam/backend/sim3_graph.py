"""§14 — unifying COLMAP and DA3 in one Sim(3) factor graph.

    ############################################################################
    #  The FACTOR DESIGN is core estimator substance (CLAUDE.md §3 + §8):      #
    #  the Sim(3) variables + rho scalar, the custom ground-anchor and         #
    #  ego-on-ground residuals and their JACOBIANS, the COLMAP-relative and    #
    #  depth-ratio measurements, and the graph connectivity are DERIVED AND    #
    #  WRITTEN BY THE AUTHOR here -- NOT generated. This file provides only:    #
    #    (a) the input contract `Sim3GraphInputs` and its plumbing prep, and    #
    #    (b) the graph-build / solve HARNESS with the factors left as stubs.    #
    ############################################################################

GTSAM in this build exposes ``Similarity3`` and ``CustomFactor`` but NOT the
templated ``BetweenFactorSimilarity3`` / ``PriorFactorSimilarity3`` -- so every
§14.4 factor (gauge prior, COLMAP-relative, depth-ratio, ground-anchor,
ego-on-ground) is written as a ``CustomFactor`` (residual + Jacobians into the
passed ``H`` list). Variable layout follows §14.2/§14.3:

    H_i = Similarity3(s_i, R_i, t_i)   per keyframe   -- s_i: COLMAP units -> metres
    rho = log r_m                      one scalar     -- r_m: DA3 depth -> COLMAP units

World point (§14.3):  X = s_i R_i ( e^{rho} d K^{-1} u ) + t_i ,  camera centre C_i = t_i.

Submap variant (§14 "DA3 runs per window"): the scene is split into overlapping windows, each
with its OWN depth ratio rho_m; adjacent windows share cameras (shared H_i) that anchor them, and
the incremental solve is iSAM2. `Sim3GraphInputs`/`prepare_sim3_inputs`/`build_sim3_graph` are the
single-window version; `SubmapGraphInputs`/`prepare_submap_inputs`/`build_submap_graph`/
`solve_incremental` are the submap version. Both keep the factors as author §3 stubs.
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
class Submap:
    """One window of the scene. DA3 runs per window (§14), so each submap carries its OWN
    COLMAP->DA3 depth-ratio r_m (its metric-up-to-scale factor) and its own slice of the DA3
    point-cloud material. Frames are referenced by GLOBAL index into `SubmapGraphInputs.tokens`,
    so an overlap frame is simply a frame that appears in two submaps (a SHARED camera)."""
    id: int
    frame_idx: np.ndarray             # (n,) global frame indices into SubmapGraphInputs.tokens
    tokens: list                      # (n,) the corresponding sample tokens

    # --- this submap's COLMAP->DA3 depth ratio (the per-window rho_m; §14.3 depth-ratio unary) ---
    depth_ratio_median: float         # r_m = median(z_colmap / z_da3) over THIS window's COLMAP point-obs
    depth_ratio_mad: float            # 1.4826*MAD of those ratios -> sigma for the rho_m unary
    depth_ratio_n: int                # number of point-obs behind this window's median

    # --- DA3 metric point-cloud material per frame (strided full frame): rays + DA3 depths (the map) ---
    da3_rays: list                    # per-frame (Nk, 3): K^{-1} [u, v, 1]
    da3_depths: list                  # per-frame (Nk,):   DA3 depth d (DA3 units; * r_m -> COLMAP, then H_i -> world)


@dataclass
class SubmapGraphInputs:
    """Factor-ready inputs for the SUBMAP §14 graph: a global per-frame initialization + global
    consecutive COLMAP relatives (one per pair — a shared frame's relative is NOT double-counted),
    plus the submap partition, its per-window depth ratios, and the overlap (shared-camera) bookkeeping
    that anchors adjacent submaps. Absolute metric scale is left as the free gauge (no ground anchor in
    this part). All §4 plumbing; produced by :func:`prepare_submap_inputs`."""
    tokens: list                      # all N frames, graph order
    K: np.ndarray                     # (3,3) true intrinsics
    sensor2ego: np.ndarray            # (4,4) camera->ego extrinsic
    cam_height: float                 # sensor2ego[2,3] (carried for later; unused while ground anchor is off)

    # --- global per-frame init for H_i (from the aligned COLMAP poses; consistent starting point) ---
    init_R: np.ndarray                # (N,3,3) camera->world rotation
    init_t: np.ndarray                # (N,3)   camera centre
    init_s: np.ndarray                # (N,)    scale s_i init (ones; author sets the convention, as in Sim3GraphInputs)

    # --- global consecutive COLMAP relatives (§14.4 relative-pose factor; one per (i-1,i) pair) ---
    colmap_rel_R: np.ndarray          # (N-1,3,3) R_{i-1}^T R_i
    colmap_rel_t: np.ndarray          # (N-1,3)   R_{i-1}^T (c_i - c_{i-1})

    # --- the submap decomposition + overlap graph ---
    submaps: list                     # list[Submap], in order; each frame's rho is its submap's depth_ratio
    frame_submaps: list               # per-frame list of submap ids the frame belongs to (2 => a shared camera)
    overlaps: list                    # list of (m, n, shared_frame_idx (k,) global) for each overlapping submap pair

    @property
    def n_frames(self) -> int: return len(self.tokens)
    @property
    def n_submaps(self) -> int: return len(self.submaps)

    def save(self, path):
        sm = np.array([dict(id=s.id, frame_idx=s.frame_idx, tokens=np.array(s.tokens),
                            depth_ratio_median=s.depth_ratio_median, depth_ratio_mad=s.depth_ratio_mad,
                            depth_ratio_n=s.depth_ratio_n,
                            da3_rays=np.array(s.da3_rays, dtype=object), da3_depths=np.array(s.da3_depths, dtype=object))
                       for s in self.submaps], dtype=object)
        ov = np.array([(m, n, idx) for (m, n, idx) in self.overlaps], dtype=object)
        np.savez(str(path), tokens=np.array(self.tokens), K=self.K, sensor2ego=self.sensor2ego,
                 cam_height=self.cam_height, init_R=self.init_R, init_t=self.init_t, init_s=self.init_s,
                 colmap_rel_R=self.colmap_rel_R, colmap_rel_t=self.colmap_rel_t, submaps=sm,
                 frame_submaps=np.array(self.frame_submaps, dtype=object), overlaps=ov)


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


def prepare_submap_inputs(scene: str = "scene-0061", *, cache_root: str = "out/frontend_cache",
                          dataroot: str = "data/nuscenes", version: str = "v1.0-mini",
                          colmap_sparse: str = "out/colmap/sparse/0",
                          window: int = 11, step: int = 7, stride: int = 8) -> SubmapGraphInputs:
    """Assemble the SUBMAP §14 inputs from the existing caches (pure plumbing, no estimation).

    Partitions the scene into overlapping windows, and for each window computes its OWN
    COLMAP->DA3 median depth ratio (median of z_colmap / z_da3 over that window's COLMAP
    point-observations) and gathers its strided full-frame DA3 point-cloud material. The
    global per-frame init and consecutive COLMAP relatives come from the aligned COLMAP
    poses; the overlaps are the shared cameras that anchor adjacent submaps. No ground
    anchor here -- absolute metric scale is the free gauge the graph does not pin."""
    import pycolmap
    from ..data import NuScenesMonoSource
    from ..frontend import cache

    src = NuScenesMonoSource(dataroot, version, camera="CAM_FRONT"); kfs = src.load_scene(scene)
    mu = cache.load_metric_upgrade(cache_root, scene); dep = cache.load_depth(cache_root, scene)
    pidx = {t: i for i, t in enumerate(mu.tokens)}
    frames = [kf for kf in kfs if kf.token in dep and kf.token in pidx]
    tokens = [kf.token for kf in frames]; tset = set(tokens); N = len(frames)
    K = np.asarray(mu.K_true, np.float64); Kinv = np.linalg.inv(K)
    s2e = frames[0].calib.sensor2ego.matrix(); h = float(s2e[2, 3])

    # global per-frame init + consecutive relatives from the aligned COLMAP poses
    cg = np.load(f"{cache_root}/{scene}/colmap_poses_global.npz", allow_pickle=True)
    cmap = {str(t): P for t, P in zip(cg["tokens"], cg["poses"])}
    cpose = np.stack([cmap[t] for t in tokens]).astype(np.float64)
    init_R = cpose[:, :3, :3].copy(); init_t = cpose[:, :3, 3].copy(); init_s = np.ones(N)
    rel_R = np.stack([cpose[i - 1, :3, :3].T @ cpose[i, :3, :3] for i in range(1, N)])
    rel_t = np.stack([cpose[i - 1, :3, :3].T @ (cpose[i, :3, 3] - cpose[i - 1, :3, 3]) for i in range(1, N)])

    # per-token COLMAP->DA3 depth ratios (z_colmap / z_da3), from the RAW sparse vs DA3 depth
    rec = pycolmap.Reconstruction(colmap_sparse)
    def img_tok(im):
        n = im.name.rsplit(".", 1)[0]
        return n if n in pidx else (tokens[int(n[1:])] if n[1:].isdigit() and int(n[1:]) < N else None)
    tok_ratios = {t: [] for t in tokens}
    for p in rec.points3D.values():
        X = np.append(p.xyz, 1.0)
        for el in p.track.elements:
            im = rec.images[el.image_id]; tok = img_tok(im)
            if tok is None or tok not in tset: continue
            rig = im.cam_from_world; rig = rig() if callable(rig) else rig
            M = np.asarray(rig.matrix()) if hasattr(rig, "matrix") else np.hstack([np.asarray(rig.rotation.matrix()), np.asarray(rig.translation)[:, None]])
            zc = (M @ X)[2]; u, v = im.points2D[el.point2D_idx].xy; dd = dep[tok].depth
            if zc > 1e-6 and 0 <= int(v) < dd.shape[0] and 0 <= int(u) < dd.shape[1]:
                zg = dd[int(v), int(u)]
                if zg > 1e-6: tok_ratios[tok].append(zc / zg)

    # strided full-frame DA3 rays + depths per frame (the DA3 point-cloud material)
    da3_rays_all, da3_depths_all = {}, {}
    for kf in frames:
        d = dep[kf.token].depth; H, W = d.shape
        vv, uu = np.mgrid[0:H:stride, 0:W:stride].reshape(2, -1); z = d[vv, uu]
        keep = z > 0; vv, uu, z = vv[keep], uu[keep], z[keep]
        da3_rays_all[kf.token] = ((Kinv @ np.stack([uu, vv, np.ones_like(uu)], 0)).T).astype(np.float32)
        da3_depths_all[kf.token] = z.astype(np.float32)

    # build submaps over overlapping windows
    submaps, frame_submaps = [], [[] for _ in range(N)]
    for m, idx in enumerate(_windows(N, window, step)):
        toks = [tokens[i] for i in idx]
        r = np.array([x for t in toks for x in tok_ratios[t]])
        med = float(np.median(r)) if r.size else float("nan")
        mad = float(np.median(np.abs(r - med)) * 1.4826) if r.size else float("nan")
        submaps.append(Submap(id=m, frame_idx=idx, tokens=toks, depth_ratio_median=med,
                              depth_ratio_mad=mad, depth_ratio_n=int(r.size),
                              da3_rays=[da3_rays_all[t] for t in toks],
                              da3_depths=[da3_depths_all[t] for t in toks]))
        for i in idx: frame_submaps[i].append(m)

    # overlap (shared-camera) bookkeeping for every pair of submaps that share frames
    overlaps = []
    for a in range(len(submaps)):
        for b in range(a + 1, len(submaps)):
            sh = np.intersect1d(submaps[a].frame_idx, submaps[b].frame_idx)
            if sh.size: overlaps.append((a, b, sh))

    return SubmapGraphInputs(
        tokens=tokens, K=K, sensor2ego=s2e, cam_height=h, init_R=init_R, init_t=init_t, init_s=init_s,
        colmap_rel_R=rel_R, colmap_rel_t=rel_t, submaps=submaps, frame_submaps=frame_submaps, overlaps=overlaps)


# ----------------------------------------------------------------------------- graph build harness (factors are §3 stubs)
def _H(i: int) -> int:
    import gtsam
    return int(gtsam.symbol("h", i))          # Similarity3 key for frame i

def _RHO() -> int:
    import gtsam
    return int(gtsam.symbol("r", 0))          # scalar rho key

def build_sim3_graph(inputs: Sim3GraphInputs):
    """Build the §14 factor graph and its initial Values from `inputs`.

    HARNESS ONLY. The variables + initial values are inserted here (plumbing); the
    FACTORS are the author's core work and are left as stubs. Fill each stub with a
    ``gtsam.CustomFactor(noise, keys, error_func)`` whose ``error_func`` returns the
    residual and writes the analytic Jacobians into the passed ``H`` list.

    Returns ``(graph, values)`` ready for :func:`solve` once the factors are added.
    """
    import gtsam

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    # --- variables + initial values (plumbing) ---
    # NOTE (this GTSAM build): Similarity3 is a MATH helper only -- it is NOT Values-storable
    # (no atSimilarity3 / insert overload), and there is no insertDouble (use insert(key, float)).
    # The scalar rho IS storable and is inserted here. The per-frame Sim(3) VARIABLE representation
    # is author core (§3): choose a storable form -- Pose3 H_i + Double s_i (7-DoF split), or a
    # Vector7 in sim(3) coords -- init values are in inputs.init_R/init_t/init_s. Insert them under
    # that representation and wire the CustomFactor keys to match.
    values.insert(_RHO(), float(inputs.init_log_r))

    # ======================================================================== §3 AUTHOR: VARIABLES + FACTORS (§14.4)
    # First insert the per-frame Sim(3) variable H_i under a Values-storable representation (see the
    # note above -- Similarity3 is not storable here), initialized from inputs.init_R/init_t/init_s.
    # Then each factor is a gtsam.CustomFactor(noise_model, [keys], error_func); the factor DEFINITIONS
    # (design notes) live in backend.factors -- derive the residuals + Jacobians (§14.2/§14.3/§5.4) there
    # and add instances here, in order:
    #
    #   1. Sim3GaugePriorFactor(H_0)          -- tight prior to init; fixes the 7-DoF gauge (incl. scale).
    #   2. ColmapRelativeSim3Factor(H_{i-1},H_i) -- (rel_R[i], rel_t[i]); rotation + local-unit translation +
    #                                            scale constancy s_i = s_{i-1}, in one factor.
    #   3. DepthRatioPriorFactor(rho)         -- residual rho - log(depth_ratio_median); sigma = |MAD|.
    #   4. GroundAnchorSim3Factor(H_i, rho)   -- per road obs k: e_z^T( s_i R_i (e^{rho} depth_k ray_k) + t_i ) -> 0.
    #   5. EgoOnGroundFactor(H_i)             -- e_z^T ego centre via H_i, sensor2ego -> cam_height;
    #                                            Jacobian sparsity d/ds = 0, d/drho = 0 (the §5.4 thesis).
    #
    raise NotImplementedError(
        "§14 variable representation + factors are author core (CLAUDE.md §3/§8): pick a Values-storable "
        "Sim(3) form (Similarity3 is not storable in this GTSAM build), insert H_i from inputs.init_*, add "
        "the CustomFactors above, then remove this guard. rho is already inserted.")
    return graph, values


def solve(graph, values, *, max_iters: int = 100, verbose: bool = False):
    """Levenberg–Marquardt driver (plumbing). Runs once the graph has factors."""
    import gtsam
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iters)
    if verbose: params.setVerbosityLM("SUMMARY")
    return gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()


# --------------------------------------------------------------------- submap graph build harness (factors are §3 stubs)
def _RHO_M(m: int) -> int:
    import gtsam
    return int(gtsam.symbol("r", m))          # per-submap depth-ratio scalar rho_m = log r_m (§14, DA3 per window)

def build_submap_graph(inputs: SubmapGraphInputs):
    """Build the SUBMAP §14 graph + initial Values from `inputs`.

    HARNESS ONLY (plumbing): inserts one scalar ``rho_m`` per submap with its initial value. The
    per-frame Sim(3) variable H_i and all FACTORS are the author's core work (§3/§8) — see the note
    in :func:`build_sim3_graph` (Similarity3 is a math helper only in this GTSAM build, not
    Values-storable, so H_i needs a storable representation the author picks). Frames in an overlap
    are SHARED variables — both submaps' factors reference the same H_i, which is what anchors
    adjacent submaps (§14.2); the overlaps list is provided for the per-submap scale cross-check /
    any explicit submap-alignment factor.

    Returns ``(graph, values)`` ready for :func:`solve` (batch) once H_i + the factors are added.
    """
    import gtsam

    graph = gtsam.NonlinearFactorGraph(); values = gtsam.Values()
    # --- storable variables + initial values (plumbing): one rho_m per submap ---
    for sm in inputs.submaps:
        r = sm.depth_ratio_median
        values.insert(_RHO_M(sm.id), float(np.log(r)) if r and r > 0 else 0.0)   # insert(key, float): no insertDouble here

    # ======================================================================== §3 AUTHOR: H_i VARIABLE + FACTORS (§14.4)
    # Insert per-frame H_i under a Values-storable Sim(3) representation (init from inputs.init_*), then add:
    #   1. Gauge prior on H_0                 -- fixes the 7-DoF gauge (absolute scale stays free w/o a ground anchor).
    #   2. COLMAP relative (i-1,i)            -- inputs.colmap_rel_R/T[i-1]; rotation + local-unit translation + scale
    #                                            constancy s_i = s_{i-1}, one CustomFactor per consecutive pair.
    #   3. Depth-ratio unary on rho_m         -- per submap: rho_m - log(depth_ratio_median); sigma from |MAD|.
    #   4. (optional) submap-alignment / cross-check on the overlaps -- each submap's scale should agree; a shared
    #                                            camera H_i already couples the two windows, so this is a soft check.
    # The absolute-scale factors (ground anchor, ego-on-ground) are intentionally OFF for this part.
    raise NotImplementedError(
        "§14 submap H_i representation + factors are author core (CLAUDE.md §3/§8): insert H_i, add the "
        "CustomFactors above, then remove this guard. Per-submap rho_m + initial values are already inserted.")
    return graph, values


def solve_incremental(inputs: SubmapGraphInputs, *, relinearize_skip: int = 1):
    """iSAM2 driver stub for the INCREMENTAL submap solve (§14 "incremental SAM").

    HARNESS ONLY: the author feeds the graph submap-by-submap — for each submap, add its new
    H_i / rho_m variables + factors (from :func:`build_submap_graph`'s factor set restricted to
    that submap, plus the shared-camera factors linking it to the previous submap) and call
    ``isam.update(new_factors, new_values)``. The factor construction is §3 author core."""
    import gtsam
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    _ = relinearize_skip
    raise NotImplementedError(
        "incremental submap solve is author core (§8): drive `isam.update(...)` per submap with the "
        "author's factors. This stub only constructs the ISAM2 object.")
