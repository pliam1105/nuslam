"""A small on-disk log of per-stage scalar metrics.

Records each stage's scalar metrics -- photometric quality on held-out views
(PSNR/SSIM/L1), trajectory error vs GT (ATE/RPE), depth-vs-lidar, and any loss terms
-- to a single JSON, one entry per named stage, so a later stage is compared to an
earlier one without recomputing it. Baseline metrics are captured once and drift is
read against them.

Plumbing: this stores and formats numbers; it computes none of them (the render /
depth / trajectory scorers live in ``nuslam.eval``; loss terms come from the
optimization).
"""
from __future__ import annotations

import json
from pathlib import Path


def record_metrics(path: Path | str, stage: str, metrics: dict, *, notes: str = "") -> Path:
    """Record ``metrics`` (a flat ``{name: number}`` dict) for ``stage`` into the JSON
    at ``path``, replacing any prior entry for that stage. Returns the path."""
    path = Path(path)
    data = load_metrics_log(path)
    entry = {k: float(v) for k, v in metrics.items()}
    if notes:
        entry["_notes"] = notes
    data[str(stage)] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def load_metrics_log(path: Path | str) -> dict:
    """Return the ``{stage: {metric: value}}`` log, or ``{}`` if the file is absent."""
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def format_metrics_log(data: dict, *, baseline: str | None = None) -> str:
    """Render the log as a text table, stages as columns, metrics as rows.

    If ``baseline`` names a stage present in ``data``, each cell also shows the signed
    delta from that stage's value, so drift from the baseline is visible at a glance.
    """
    if not data:
        return "(no metrics recorded)"
    stages = list(data.keys())
    metrics = sorted({k for e in data.values() for k in e if not k.startswith("_")})
    w = max([len(m) for m in metrics] + [6])
    head = "metric".ljust(w) + "".join(f"  {s:>14}" for s in stages)
    lines = [head, "-" * len(head)]
    base = data.get(baseline, {}) if baseline else {}
    for m in metrics:
        row = m.ljust(w)
        for s in stages:
            v = data[s].get(m)
            if v is None:
                row += f"  {'-':>14}"
            elif baseline and s != baseline and m in base:
                row += f"  {v:>8.4g}{('%+.3g' % (v - base[m])):>6}"
            else:
                row += f"  {v:>14.4g}"
        lines.append(row)
    return "\n".join(lines)
