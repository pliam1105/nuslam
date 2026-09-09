"""Per-rung metric log for the staged Gaussian ladder (Stage 3).

The plan measures every rung against the Rung-3.1 baseline: photometric quality on
held-out views (PSNR/SSIM/L1), trajectory error vs GT (ATE/RPE), depth-vs-lidar,
and each loss term. This records those scalars, one entry per rung, to a single
JSON so a later rung is compared to earlier ones without re-running them -- the
"record the baseline before any pose freedom" step made durable.

Plumbing: it stores and formats numbers; it computes none of them (the render /
depth / trajectory scorers live in ``nuslam.eval``; the loss terms come from the
author's training loop).
"""
from __future__ import annotations

import json
from pathlib import Path


def record_rung(path: Path | str, rung: str, metrics: dict, *, notes: str = "") -> Path:
    """Record ``metrics`` (a flat ``{name: number}`` dict) for ``rung`` into the JSON
    at ``path``, replacing any prior entry for that rung. Returns the path."""
    path = Path(path)
    data = load_rungs(path)
    entry = {k: float(v) for k, v in metrics.items()}
    if notes:
        entry["_notes"] = notes
    data[str(rung)] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def load_rungs(path: Path | str) -> dict:
    """Return the ``{rung: {metric: value}}`` log, or ``{}`` if the file is absent."""
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def format_rungs(data: dict, *, baseline: str | None = None) -> str:
    """Render the log as a text table, rungs as columns, metrics as rows.

    If ``baseline`` names a rung present in ``data``, each cell also shows the signed
    delta from that rung's value, so drift from the baseline is visible at a glance.
    """
    if not data:
        return "(no rungs recorded)"
    rungs = list(data.keys())
    metrics = sorted({k for e in data.values() for k in e if not k.startswith("_")})
    w = max([len(m) for m in metrics] + [6])
    head = "metric".ljust(w) + "".join(f"  {r:>14}" for r in rungs)
    lines = [head, "-" * len(head)]
    base = data.get(baseline, {}) if baseline else {}
    for m in metrics:
        row = m.ljust(w)
        for r in rungs:
            v = data[r].get(m)
            if v is None:
                row += f"  {'-':>14}"
            elif baseline and r != baseline and m in base:
                row += f"  {v:>8.4g}{('%+.3g' % (v - base[m])):>6}"
            else:
                row += f"  {v:>14.4g}"
        lines.append(row)
    return "\n".join(lines)
