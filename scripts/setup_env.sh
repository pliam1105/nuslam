#!/usr/bin/env bash
# Set up the nuslam Python environment and link the nuScenes data.
#
#   ./scripts/setup_env.sh
#
# Reuses the system torch/CUDA build via --system-site-packages, installs the
# rest into .venv, and (if not already present) symlinks data/nuscenes to the
# mini split. The mini split itself is fetched by scripts/download_data.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d .venv ]; then
  echo "creating .venv (reusing system site-packages for torch/cuda)"
  python3 -m venv --system-site-packages .venv
fi

echo "installing requirements"
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt

echo
echo "verifying key imports"
.venv/bin/python - <<'PY'
import importlib
for m in ["gtsam", "nuscenes", "transformers", "foxglove_websocket", "torch"]:
    mod = importlib.import_module(m)
    print(f"  ok  {m:20} {getattr(mod, '__version__', '?')}")
import torch
print(f"  cuda available: {torch.cuda.is_available()}")
PY

echo
echo "done. next:"
echo "  ./scripts/download_data.sh            # nuScenes v1.0-mini (~4.2 GB), if not linked"
echo "  .venv/bin/python scripts/inspect_sample.py --out out/calib_check.png"
