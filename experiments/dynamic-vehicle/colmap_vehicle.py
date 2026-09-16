"""Experiment A: COLMAP on the tracked van. Full frames + a VAN-ONLY mask (keep=van) + the real single
camera K (van is rigid -> COLMAP treats it as static with the camera orbiting). Inverse of the static
COLMAP (which excludes vehicles). Reports how much of the van COLMAP can register."""
import numpy as np, pycolmap, shutil
from pathlib import Path
from PIL import Image
from nuslam.data import NuScenesMonoSource
from nuslam.frontend import cache

SP = "out/vehicle_cache"
MIN_AREA = 6000
work = Path(SP) / "colmap_van";
if work.exists(): shutil.rmtree(work)
(work / "images").mkdir(parents=True); (work / "masks").mkdir(); (work / "sparse").mkdir()
src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT"); kfs = src.load_scene("scene-0061")
K = kfs[0].calib.intrinsic
bt = dict((int(f), m) for f, m in np.load(f"{SP}/best_track.npy", allow_pickle=True))

n = 0
for fi in sorted(bt):
    m = bt[fi]
    if m.sum() < MIN_AREA: continue
    Image.fromarray(np.asarray(kfs[fi].image())).save(work / "images" / f"f{fi:02d}.png")
    Image.fromarray((m.astype(np.uint8) * 255)).save(work / "masks" / f"f{fi:02d}.png")  # keep=van
    n += 1
print(f"wrote {n} full frames + van masks; fx={K[0,0]:.0f}", flush=True)

db = work / "db.db"
ropts = pycolmap.ImageReaderOptions(); ropts.mask_path = str(work / "masks"); ropts.camera_model = "PINHOLE"
ropts.camera_params = f"{K[0,0]:.4f},{K[1,1]:.4f},{K[0,2]:.4f},{K[1,2]:.4f}"
pycolmap.extract_features(db, work / "images", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ropts)
pycolmap.match_exhaustive(db)
opts = pycolmap.IncrementalPipelineOptions(); opts.min_num_matches = 8
maps = pycolmap.incremental_mapping(db, work / "images", work / "sparse", options=opts)

if not maps:
    print("COLMAP: NO reconstruction (van not registerable) -- expected for low-texture/reflective vehicles")
else:
    rec = maps[0] if isinstance(maps, list) else maps[0]
    r = pycolmap.Reconstruction(str(work / "sparse" / "0")) if (work / "sparse" / "0").exists() else rec
    errs = [p.error for p in r.points3D.values()]
    print(f"COLMAP: registered {r.num_reg_images()}/{n} images, {len(r.points3D)} 3D points, "
          f"mean reproj err {np.mean(errs):.2f}px" if errs else f"registered {r.num_reg_images()}/{n}, 0 points")
