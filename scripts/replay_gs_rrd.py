"""Replay a finished GS run's saved snapshots + training curves into a .rrd (viz only).

Loads the `gs_snapshots/*.npz` a run wrote and logs them with the existing viz
helper (rrlog.log_gaussians / log_scalar) onto the `iter` timeline, so you can
scrub the Gaussian evolution and the loss/PSNR/N curves after the fact. Gaussians
are downsampled per frame and positional outliers clipped so the file stays
viewable (a full multi-million x many-snapshot recording would be enormous).

Examples
--------
    # by run name under a scene's runs/ dir
    python scripts/replay_gs_rrd.py --scene scene-0061 --run run4
    # or point straight at any dir that has gs_snapshots/ + train_log.csv
    python scripts/replay_gs_rrd.py --run-dir out/frontend_cache/scene-0061
    # then: rerun out/<run>.rrd
"""
import argparse
import csv
import glob
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src")
from nuslam.viz import rerun_logging as rrlog  # noqa: E402

C0 = 0.28209479177387814  # SH DC basis; display colour = DC*C0 + 0.5


def rgba(sh: np.ndarray, opac: np.ndarray) -> np.ndarray:
    rgb01 = np.clip(sh[:, 0, :] * C0 + 0.5, 0.0, 1.0)
    return (np.concatenate([rgb01, opac.reshape(-1, 1)], axis=1) * 255).astype(np.uint8)


def load_curves(csv_path: Path):
    if not csv_path.exists():
        return
    raw = csv_path.read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore")
    cols = [("loss", "train/loss"), ("photometric", "train/photometric"),
            ("dssim", "train/dssim"), ("num_gaussians", "train/num_gaussians"),
            ("heldout_psnr", "eval/heldout_psnr")]
    for r in csv.DictReader(io.StringIO(raw)):
        try:
            step = int(r["step"])
        except (KeyError, ValueError):
            continue
        for col, name in cols:
            if r.get(col):
                rrlog.log_scalar(name, float(r[col]), step=step)


def _load_img(path: Path, max_w: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if max_w and im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)))
    return np.asarray(im)


def load_renders(run_dir: Path, stem: str, every: int, max_w: int):
    """Log GT (static) + estimated renders (on the iter timeline) for train and heldout."""
    rdir = run_dir / "renders"
    if not rdir.is_dir():
        return
    for split in ("train", "heldout"):
        gt = rdir / f"{split}_gt.png"
        if gt.exists():
            rrlog.log_image(f"render/{split}/gt", _load_img(gt, max_w), static=True)
        ests = sorted(rdir.glob(f"{split}_est_*.png"))
        for f in ests[:: max(1, every)]:
            step = int(f.stem.split("_")[-1])
            rrlog.set_step(step)
            rrlog.log_image(f"render/{split}/est", _load_img(f, max_w))
        if ests:
            print(f"  {split}: logged {len(ests[::max(1, every)])}/{len(ests)} renders + gt")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", help="scene id; with --run resolves out/frontend_cache/<scene>/runs/<run>")
    ap.add_argument("--run", help="run name under the scene's runs/ dir")
    ap.add_argument("--run-dir", help="explicit dir holding gs_snapshots/ and train_log.csv (overrides --scene/--run)")
    ap.add_argument("--cache", default="out/frontend_cache", help="frontend cache root")
    ap.add_argument("--out", help="output .rrd path (default out/<run>.rrd)")
    ap.add_argument("--max-pts", type=int, default=400_000, help="max Gaussians logged per frame (random subset)")
    ap.add_argument("--every", type=int, default=4, help="keep every Nth snapshot (the last is always kept)")
    ap.add_argument("--clip-m", type=float, default=200.0, help="drop Gaussians farther than this from the frame median centre")
    ap.add_argument("--render-every", type=int, default=5, help="keep every Nth est render (train & heldout)")
    ap.add_argument("--render-width", type=int, default=640, help="downscale renders to this max width (0 = full res)")
    ap.add_argument("--no-renders", action="store_true", help="skip the train/heldout image renders")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        stem = run_dir.name
    elif args.scene and args.run:
        run_dir = Path(args.cache) / args.scene / "runs" / args.run
        stem = args.run
    else:
        ap.error("give --run-dir, or both --scene and --run")

    snaps = sorted(glob.glob(str(run_dir / "gs_snapshots" / "gs_*.npz")))
    if not snaps:
        ap.error(f"no gs_snapshots/*.npz under {run_dir}")
    keep = snaps[::args.every]
    if snaps[-1] not in keep:
        keep.append(snaps[-1])

    out = args.out or f"out/{stem}.rrd"
    rng = np.random.default_rng(0)
    rrlog.init(f"{stem}-replay", save=out)
    load_curves(run_dir / "train_log.csv")
    if not args.no_renders:
        load_renders(run_dir, stem, args.render_every, args.render_width)

    for f in keep:
        step = int(Path(f).stem.split("_")[-1])
        d = np.load(f)
        means, scales, quats, opac, sh = d["means"], d["scales"], d["quats"], d["opacities"], d["sh"]
        ctr = np.median(means, axis=0)
        idx = np.flatnonzero(np.linalg.norm(means - ctr, axis=1) < args.clip_m)
        if idx.size > args.max_pts:
            idx = rng.choice(idx, args.max_pts, replace=False)
        rrlog.set_step(step)
        rrlog.log_gaussians(f"{stem}/gs", means[idx], scales[idx], quats[idx], rgba(sh, opac)[idx])
        print(f"  step {step:6d}: logged {idx.size}/{len(means)} gaussians")

    print(f"\nwrote {out}  ({len(keep)} gaussian frames, <= {args.max_pts} pts/frame)\n  view:  rerun {out}")


if __name__ == "__main__":
    main()
