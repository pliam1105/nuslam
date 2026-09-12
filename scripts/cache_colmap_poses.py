"""Cache COLMAP camera poses in the nuScenes global (GT/GPS) metric frame, per keyframe token.

DA3's poses are corrupted by dynamic objects, so the training cameras come from a masked COLMAP
structure-from-motion solve instead. This script runs that solve on the scene's keyframes with the
cached sky + vehicle masks excluded from feature extraction, conditions it for forward driving
(known PINHOLE intrinsics fixed, low minimum triangulation angle), aligns the full trajectory to
GPS once with the same iterated lever-arm Umeyama used for DA3 (a far more reliable scale/rotation
than a short subset), and saves the per-token global-frame poses so run_gs can load and subset them
by token (mirroring the cached metric-upgrade poses). The sparse SfM points are discarded.

    python scripts/cache_colmap_poses.py --scene scene-0061

Requires the cached sky/vehicle SAM3 masks (run run_gs once with --mask-sky --mask-vehicles, or
seed the seg cache) and pycolmap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource, gps_positions_at, load_proprio_streams  # noqa: E402
from nuslam.frontend import cache  # noqa: E402
from nuslam.recon import resolve_scale_gps  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default="scene-0061")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--work", type=Path, default=Path("out/colmap"), help="COLMAP working dir (images/masks/db/sparse)")
    p.add_argument("--sky-tag", default="sam3-sky-0.5", help="cached sky mask tag")
    p.add_argument("--vehicle-tag", default="sam3-moving vehicle-0.5", help="cached vehicle mask tag")
    p.add_argument("--max-features", type=int, default=12000)
    p.add_argument("--min-tri-angle", type=float, default=2.0, help="low, for forward-driving low parallax")
    p.add_argument("--reuse-sparse", action="store_true", help="skip SfM, reuse an existing sparse/0 reconstruction")
    return p.parse_args()


def run_sfm(kfs, sky, veh, K, work, args):
    """Masked COLMAP SfM tuned for forward driving; writes work/sparse/0. Returns nothing."""
    (work / "images").mkdir(parents=True, exist_ok=True)
    (work / "masks").mkdir(exist_ok=True)
    for i, kf in enumerate(kfs):
        name = f"{i:03d}.png"
        Image.fromarray(np.asarray(kf.image())).save(work / "images" / name)
        keep = ~(sky[kf.token] | veh[kf.token])               # 255 = use for features, 0 = ignore
        Image.fromarray((keep.astype(np.uint8) * 255)).save(work / "masks" / f"{name}.png")
    print(f"wrote {len(kfs)} images + masks; fx={K[0, 0]:.0f}", flush=True)

    db = work / "database.db"
    if db.exists():
        db.unlink()
    ropts = pycolmap.ImageReaderOptions()
    ropts.mask_path = str(work / "masks")
    ropts.camera_model = "PINHOLE"
    ropts.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"   # known intrinsics, held fixed
    eopts = pycolmap.FeatureExtractionOptions()
    eopts.sift.max_num_features = args.max_features
    pycolmap.extract_features(db, work / "images", camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ropts, extraction_options=eopts)
    pycolmap.match_exhaustive(db)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False          # intrinsics known -> do not refine
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.mapper.init_min_tri_angle = args.min_tri_angle
    opts.mapper.abs_pose_min_num_inliers = 15
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    maps = pycolmap.incremental_mapping(db, work / "images", sparse, options=opts)
    for idx, rec in maps.items():
        print(f"  model {idx}: {rec.num_reg_images()}/{len(kfs)} images registered, "
              f"{rec.num_points3D()} sparse points", flush=True)


def main() -> int:
    args = parse_args()
    src = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    kfs = src.load_scene(args.scene)
    scene = kfs[0].scene_name
    sky = cache.load_seg_masks(args.cache_root, scene, args.sky_tag)
    veh = cache.load_seg_masks(args.cache_root, scene, args.vehicle_tag)
    if sky is None or veh is None:
        print(f"need cached sky/vehicle masks ('{args.sky_tag}', '{args.vehicle_tag}') under "
              f"{args.cache_root}/{scene} -- run run_gs.py --mask-sky --mask-vehicles once to build them")
        return 1
    K = kfs[0].calib.intrinsic

    if not args.reuse_sparse:
        run_sfm(kfs, sky, veh, K, args.work, args)

    rec = pycolmap.Reconstruction(str(args.work / "sparse" / "0"))
    name2c2w = {}
    for img in rec.images.values():
        T = np.eye(4); T[:3, :4] = img.cam_from_world().inverse().matrix()   # camera->world (up-to-scale)
        name2c2w[img.name] = T
    reg = [i for i in range(len(kfs)) if f"{i:03d}.png" in name2c2w]          # registered frames, my order
    c2w = np.stack([name2c2w[f"{i:03d}.png"] for i in reg])
    tokens = [kfs[i].token for i in reg]

    streams = load_proprio_streams(args.dataroot, scene, version=args.version)
    gps_xy, gps_valid = gps_positions_at(streams, [kfs[i].timestamp_us for i in reg])
    s2e = kfs[0].calib.sensor2ego.matrix()
    res = resolve_scale_gps(c2w, gps_xy, s2e, gps_valid)       # recon -> global Sim(3), same fit as DA3
    R_align = res.T[:3, :3] / res.scale

    poses = np.zeros((len(reg), 4, 4))
    for k in range(len(reg)):
        R = R_align @ c2w[k][:3, :3]
        U, _, Vt = np.linalg.svd(R)                            # re-orthonormalize the rotation block
        if np.linalg.det(U @ Vt) < 0:
            U[:, -1] *= -1
        poses[k, :3, :3] = U @ Vt
        poses[k, :3, 3] = (res.T @ np.append(c2w[k][:3, 3], 1.0))[:3]
        poses[k, 3, 3] = 1.0

    out = args.cache_root / scene / "colmap_poses_global.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tokens=np.array(tokens), poses=poses, scale=res.scale, residual=res.residual)
    print(f"cached {len(reg)}/{len(kfs)} COLMAP global poses -> {out}  "
          f"(scale={res.scale:.4f}, GPS residual={res.residual:.3f} m)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
