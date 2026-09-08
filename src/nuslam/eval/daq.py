"""DAQ solve diagnostics -- how well and how consistently the metric upgrade fit.

Infrastructure (§4): these score the author's DAQ solve; they make no estimation
decision. The metrics, and what a good value looks like:

  * ``a_null_ratio``   -- sigma_min/sigma_max of the DLT matrix ``A``. Near 0 means
    ``vec(Omega*)`` is a clean null vector (the "proportional to I" system is
    (nearly) exactly solvable).
  * ``a_gap``          -- sigma_{-2}/sigma_{-1} of ``A``. Large (>>1) means the null
    space is 1-D, i.e. the solution is unique / well-separated.
  * ``omega_rank3_gap``-- |lambda_3|/|lambda_4| of the (pre-rank-3) ``Omega*``.
    Large means ``Omega*`` is cleanly rank-3 and the plane at infinity is
    well-determined; small means it is barely constrained (shaky upgrade).
  * ``fit_resid``      -- per camera, ||omega_i/scale - I||_F with
    omega_i = P~_i Omega* P~_i^T. This is exactly what the DLT minimized; small
    everywhere = clean fit, a few large = bad frames to gate.
  * ``metric_resid``   -- per camera, ||A3/s (A3/s)^T - I||_F where A3 is the left
    3x3 of the recovered metric camera ``P_metric``. This is "how Euclidean" the
    recovered cameras are (subsumes recovered-K anisotropy + skew).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DAQDiagnostics:
    n_cameras: int
    a_null_ratio: float
    a_gap: float
    omega_rank3_gap: float
    fit_resid_mean: float
    fit_resid_median: float
    fit_resid_max: float
    metric_resid_mean: float
    metric_resid_median: float
    metric_resid_max: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def __str__(self) -> str:  # pragma: no cover - reporting
        return (
            f"DAQ[{self.n_cameras}]  A_null={self.a_null_ratio:.2e} A_gap={self.a_gap:.1f}  "
            f"Omega*_rank3_gap={self.omega_rank3_gap:.1f}\n"
            f"  fit(prop.I) resid  mean={self.fit_resid_mean:.4f} "
            f"median={self.fit_resid_median:.4f} max={self.fit_resid_max:.4f}\n"
            f"  metric(Euclid) resid mean={self.metric_resid_mean:.4f} "
            f"median={self.metric_resid_median:.4f} max={self.metric_resid_max:.4f}"
        )


def _dev_from_identity(m: np.ndarray) -> float:
    """||m/scale - I||_F after normalizing the average diagonal to 1 (kills the gauge scale)."""
    d = np.trace(m) / m.shape[0]
    if abs(d) < 1e-12:
        return float("nan")
    return float(np.linalg.norm(m / d - np.eye(m.shape[0])))


def evaluate_daq(cameras: np.ndarray, A: np.ndarray, omega_star: np.ndarray,
                 metric_cameras: np.ndarray | None = None) -> DAQDiagnostics:
    """Diagnostics for a DAQ solve. See module docstring for what each number means.

    Args:
        cameras:        (N, 3, 4) normalized projective cameras ``P~_i``.
        A:              (5N, 16) DLT matrix from ``build_daq_system``.
        omega_star:     (4, 4) solved dual absolute quadric (rank-3 enforced).
        metric_cameras: (N, 3, 4) recovered ``P_metric`` (for the Euclidean residual).
    """
    cameras = np.asarray(cameras, float)
    n = len(cameras)

    sv = np.linalg.svd(A, compute_uv=False)
    a_null = float(sv[-1] / sv[0]) if sv[0] > 0 else float("nan")
    a_gap = float(sv[-2] / sv[-1]) if sv[-1] > 0 else float("inf")

    # Pre-rank-3 Omega* (smallest right singular vector, symmetrized) -> eigenvalue gap.
    Vt = np.linalg.svd(A)[2]
    omega_raw = Vt[-1].reshape(4, 4)
    omega_raw = (omega_raw + omega_raw.T) / 2.0
    mags = np.sort(np.abs(np.linalg.eigvalsh(omega_raw)))
    omega_gap = float(mags[1] / mags[0]) if mags[0] > 0 else float("inf")

    fit = np.array([_dev_from_identity(cameras[i] @ omega_star @ cameras[i].T) for i in range(n)])

    if metric_cameras is not None:
        mc = np.asarray(metric_cameras, float)
        mr = []
        for i in range(len(mc)):
            A3 = mc[i][:, :3]
            s = abs(np.linalg.det(A3)) ** (1.0 / 3.0)
            if s > 1e-12:
                An = A3 / s
                mr.append(float(np.linalg.norm(An @ An.T - np.eye(3))))
        mr = np.array(mr) if mr else np.array([np.nan])
    else:
        mr = np.array([np.nan])

    return DAQDiagnostics(
        n_cameras=n,
        a_null_ratio=a_null, a_gap=a_gap, omega_rank3_gap=omega_gap,
        fit_resid_mean=float(np.nanmean(fit)), fit_resid_median=float(np.nanmedian(fit)),
        fit_resid_max=float(np.nanmax(fit)),
        metric_resid_mean=float(np.nanmean(mr)), metric_resid_median=float(np.nanmedian(mr)),
        metric_resid_max=float(np.nanmax(mr)),
    )
