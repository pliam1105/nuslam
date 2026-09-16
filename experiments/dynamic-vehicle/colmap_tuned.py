"""COLMAP on the van, retuned like the global scene: known K held fixed, high feature count, low SIFT
peak threshold (low-texture van), relaxed mapper thresholds. Reuses the images/masks already written."""
import numpy as np, pycolmap
from pathlib import Path
SP = "out/vehicle_cache"
work = Path(SP) / "colmap_van"
from nuslam.data import NuScenesMonoSource
K = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT").load_scene("scene-0061")[0].calib.intrinsic
nimg = len(list((work / "images").glob("*.png")))
for sub in ("sparse", "sparse_tuned"):
    pass
db = work / "db_tuned.db"
if db.exists(): db.unlink()
(work / "sparse_tuned").mkdir(exist_ok=True)

ropts = pycolmap.ImageReaderOptions(); ropts.mask_path = str(work / "masks"); ropts.camera_model = "PINHOLE"
ropts.camera_params = f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]}"
eopts = pycolmap.FeatureExtractionOptions()
eopts.sift.max_num_features = 20000
eopts.sift.peak_threshold = 0.001      # default 0.0067 -> many more low-contrast features (smooth van)
eopts.sift.edge_threshold = 15.0       # default 10 -> keep more edge-like features
pycolmap.extract_features(db, work / "images", camera_mode=pycolmap.CameraMode.SINGLE,
                          reader_options=ropts, extraction_options=eopts)
mopts = pycolmap.FeatureMatchingOptions()
try: mopts.sift.max_ratio = 0.9; mopts.sift.max_distance = 0.9   # relax matching for low-texture
except Exception: pass
pycolmap.match_exhaustive(db, matching_options=mopts)
opts = pycolmap.IncrementalPipelineOptions()
opts.ba_refine_focal_length = False; opts.ba_refine_principal_point = False; opts.ba_refine_extra_params = False
opts.min_num_matches = 8
opts.mapper.init_min_tri_angle = 1.0        # tiny baseline between adjacent van views
opts.mapper.abs_pose_min_num_inliers = 8
opts.mapper.filter_max_reproj_error = 6.0
maps = pycolmap.incremental_mapping(db, work / "images", work / "sparse_tuned", options=opts)
if not maps:
    print(f"COLMAP tuned: STILL no reconstruction ({nimg} imgs)")
else:
    best = max(maps.values(), key=lambda r: r.num_reg_images())
    errs = [p.error for p in best.points3D.values()]
    print(f"COLMAP tuned: {len(maps)} model(s); best {best.num_reg_images()}/{nimg} images, "
          f"{best.num_points3D()} points, mean reproj {np.mean(errs):.2f}px")
    print("  per-model reg:", [r.num_reg_images() for r in maps.values()])
