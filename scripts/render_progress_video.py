"""Training-progress video from a run's SAVED render PNGs (viz only, no GPU, no re-render).

run_gs.py writes train_gt.png / heldout_gt.png once and train_est_<step>.png /
heldout_est_<step>.png every log step. This stitches them into a 2x2 panel
(GT | estimate, for the train view and the held-out view) across the run's steps, so
the estimate is shown sharpening over training. Because it uses the images the run
itself rendered (with the run's own cameras), it is correct for any pose source
(DA3 / GT / COLMAP) with no geometry setup.

    python scripts/render_progress_video.py --scene scene-0061 --run run4
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def fit_w(img, w):
    h = round(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def label(img, text, y=22):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def load(path, w):
    im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return fit_w(im, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene")
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache-root", default="out/frontend_cache")
    ap.add_argument("--out", default=None, help="default out/<run>_progress.mp4")
    ap.add_argument("--cell-w", type=int, default=420, help="per-image width (px)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth logged step")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    rdir = Path(args.cache_root) / (args.scene or "") / "runs" / args.run / "renders"
    if not rdir.is_dir():
        sys.exit(f"no renders dir under {rdir}")
    steps = sorted(int(Path(f).stem.split("_")[-1])
                   for f in glob.glob(str(rdir / "heldout_est_*.png")))
    if not steps:
        sys.exit(f"no heldout_est_*.png under {rdir}")
    steps = steps[:: max(1, args.stride)]
    w = args.cell_w
    gts = {}
    for split in ("train", "heldout"):
        g = rdir / f"{split}_gt.png"
        gts[split] = load(g, w) if g.exists() else None

    out = args.out or f"out/{args.run}_progress.mp4"
    pad = lambda n: np.full((6, w, 3), 30, np.uint8)
    with imageio.get_writer(out, fps=args.fps, macro_block_size=None) as wr:
        for step in steps:
            cols = []
            for split in ("train", "heldout"):
                est_p = rdir / f"{split}_est_{step:06d}.png"
                if not est_p.exists():
                    continue
                est = label(load(est_p, w).copy(), f"{split} est")
                gt = gts[split]
                top = label(gt.copy(), f"{split} GT") if gt is not None else np.full_like(est, 30)
                # match heights (gt/est identical size already)
                cols.append(np.vstack([top, np.full((6, w, 3), 30, np.uint8), est]))
            if not cols:
                continue
            body = np.hstack([np.hstack([c, np.full((c.shape[0], 6, 3), 30, np.uint8)]) for c in cols])
            bar = np.full((34, body.shape[1], 3), 20, np.uint8)
            label(bar, f"{args.run}  iter {step}", y=24)
            wr.append_data(np.vstack([bar, body]))
    print(f"wrote {out}  ({len(steps)} steps from saved renders)")


if __name__ == "__main__":
    main()
