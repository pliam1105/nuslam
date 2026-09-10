"""Generate the README result figures for a run (viz only): a GT-vs-render montage,
the training curves, and a top-down GT-vs-recon trajectory. Writes into docs/.

    python scripts/make_readme_figures.py --scene scene-0061 --run run4
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
DOCS = REPO / "docs"
DOCS.mkdir(exist_ok=True)


def render_montage(run_dir, out):
    r = run_dir / "renders"
    est_t = sorted(r.glob("train_est_*.png"))[-1]
    est_h = sorted(r.glob("heldout_est_*.png"))[-1]
    rows = [("train", r / "train_gt.png", est_t), ("held-out", r / "heldout_gt.png", est_h)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6))
    for i, (name, gt, est) in enumerate(rows):
        for j, (title, p) in enumerate([(f"{name}  GT", gt), (f"{name}  render", est)]):
            axes[i, j].imshow(np.asarray(Image.open(p)))
            axes[i, j].set_title(title, fontsize=11)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def training_curves(run_dir, out):
    raw = (run_dir / "train_log.csv").read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore")
    rows = [r for r in csv.DictReader(io.StringIO(raw)) if r.get("num_gaussians")]
    step = np.array([int(r["step"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    psnr = np.array([float(r["heldout_psnr"]) for r in rows])
    n = np.array([int(r["num_gaussians"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(step, loss, color="#c1440e", lw=1.5)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("training loss"); ax[0].set_title("Loss")
    ax[0].grid(alpha=0.25)
    axr = ax[1].twinx()
    ax[1].plot(step, psnr, color="#1f6feb", lw=1.5, label="held-out PSNR")
    axr.plot(step, n / 1e6, color="#6e7681", lw=1.2, ls="--", label="Gaussians (M)")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("held-out PSNR (dB)", color="#1f6feb")
    axr.set_ylabel("Gaussians (millions)", color="#6e7681")
    ax[1].set_title("Held-out PSNR vs. Gaussian count")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def trajectory(args, out):
    from render_gs_video import load_geometry
    geom = load_geometry(args)
    poses_m, gt = geom[1], geom[6]
    est_c = poses_m[:, :3, 3]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(est_c[:, 0], est_c[:, 1], "-o", ms=3, color="#1f6feb", label="DA3 / metric-upgrade estimate")
    if gt is not None:
        gt_c = gt[:, :3, 3]
        ax.plot(gt_c[:, 0], gt_c[:, 1], "-o", ms=3, color="#2da44e", label="nuScenes GT")
        d = np.linalg.norm(est_c - gt_c, axis=1)
        for a, b in zip(est_c, gt_c):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#d0d7de", lw=0.6, zorder=0)
        ax.set_title(f"Camera trajectory in the metric-recon frame  —  GT offset: mean {d.mean():.2f} m, max {d.max():.2f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")   # 1 m in x == 1 m in y, so the shape is undistorted
    ax.margins(0.05); ax.grid(alpha=0.25); ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene-0061")
    ap.add_argument("--run", default="run4")
    ap.add_argument("--cache-root", default="out/frontend_cache")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--camera", default="CAM_FRONT")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--only", choices=["montage", "curves", "trajectory"], help="just one figure")
    args = ap.parse_args()

    run_dir = Path(args.cache_root) / args.scene / "runs" / args.run
    if args.only in (None, "montage"):
        render_montage(run_dir, DOCS / "render_compare.png")
    if args.only in (None, "curves"):
        training_curves(run_dir, DOCS / "training_curves.png")
    if args.only in (None, "trajectory"):
        trajectory(args, DOCS / "trajectory.png")


if __name__ == "__main__":
    main()
