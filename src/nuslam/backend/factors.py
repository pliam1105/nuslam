"""§14 metric-anchor custom factors -- left unimplemented by design.

    ############################################################################
    #  Core estimator substance. Intentionally left unimplemented.             #
    #  The GROUND-ANCHOR and EGO-ON-GROUND residuals and their JACOBIANS -- the #
    #  scale-resolution geometry, where monocular scale resolves -- are the     #
    #  heart of the metric-scale mechanism and are derived and written here,    #
    #  not generated.                                                          #
    ############################################################################

This module holds only the CUSTOM §14 factors -- the ones no stock gtsam factor expresses:
``GroundAnchorSim3Factor`` and ``EgoOnGroundFactor`` (the ground/ego-height scale anchor,
CLAUDE.md rung 2), and ``SubmapAlignmentFactor`` (the submap-alignment that couples the two rho
scales, §14.3). The GENERIC §14.4 factors are stock and are wired in ``sim3_graph``:

  * gauge prior     -> ``gtsam.PriorFactorSimilarity3(H_0, init)``
  * COLMAP relative -> ``gtsam.BetweenFactorSimilarity3(H_{i-1}, H_i, (rel_R, rel_t, s=1))``
  * depth-ratio     -> ``gtsam.PriorFactorDouble(rho, log r_hat)``
  * (optional) §14.2 cross-check -> ``gtsam.BetweenFactorDouble(rho_m, rho_n, 0)``

(gtsam>=4.3 makes ``Similarity3`` a first-class variable + ships those factors; see requirements.txt.)

The two factors here couple the per-frame ``Similarity3`` variable H_i = (R_i, t_i, s_i) with the
depth-ratio scalar rho through a bespoke geometric residual -- that coupling is why they must be
custom. They are stubs: they name the seam and hold the design notes; the residual + Jacobians are
derived and written by the author (§3/§8).

GTSAM offers two routes for a custom factor; the choice (and why) is a design decision:

  * ``gtsam.CustomFactor(noise_model, keys, error_func)`` -- Python error callback, Jacobians filled
    numerically or explicitly into the passed ``H`` list.
  * subclass a ``NoiseModelFactor`` in C++/pybind -- analytic, faster.

Design decisions to pin down before writing a line:
  - Ground anchor: which road pixels feed it, inlier gating, and when it is skipped (it is OFF
    entirely for the anchor-free submap graph).
  - Jacobians: analytic or numerical? The §5.4 sparsity of the ego-height residual (d/ds_i = 0) is
    the claim to exploit and verify.
  - Noise model + robustifier (Huber?) and its threshold, with justification.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- shared base + callback contract
class _Sim3Factor:
    """Base for the custom §14 factors: stores the keys + noise model and wires the error
    callback. Subclasses store their measurement in ``__init__`` (the INPUTS), set ``DIM``
    (the residual length, i.e. the OUTPUT), and IMPLEMENT ``error`` -- the one method that
    is author core.

    ``error`` is the ``gtsam.CustomFactor`` callback (the "eval"):

        error(values, H=None) -> np.ndarray of shape (DIM,)

    Given the current ``values``, it returns the residual, and when ``H`` is not None it
    writes each variable's analytic Jacobian into ``H[k]`` -- one (DIM, tangent_dim) block
    per key, in ``self.keys`` order (tangent_dim = 7 for a ``Similarity3`` H_i, 1 for a rho
    scalar). The noise model whitens it; leave whitening to gtsam. ``as_custom_factor`` (plumbing)
    turns this into the graph-ready factor.
    """
    DIM: int = 0

    def __init__(self, keys, noise) -> None:
        self.keys = [int(k) for k in keys]   # gtsam variable keys this factor touches, in residual order
        self.noise = noise                    # gtsam noise model (author picks the sigmas / robustifier)

    def error(self, values, H=None) -> np.ndarray:
        """AUTHOR CORE (§14): residual (DIM,) + Jacobians into H. Left unimplemented."""
        raise NotImplementedError(
            f"{type(self).__name__}.error is author core (§14): implement the residual + Jacobians.")

    def as_custom_factor(self):
        """Plumbing: build ``gtsam.CustomFactor(noise, keys, callback)`` from this spec."""
        import gtsam
        return gtsam.CustomFactor(self.noise, self.keys, lambda this, v, H: self.error(v, H))


# ----------------------------------------------------------------------------- §14.4 metric-anchor factors (custom)
class GroundAnchorSim3Factor(_Sim3Factor):
    """Binary on (H_i, rho): a road-pixel back-projection constrained to the ground plane.

    INPUTS:  ``hi_key`` (a ``Similarity3``), ``rho_key`` (a Double); a road observation ``ray`` (3,) =
             K^{-1}[u,v,1] and its DA3 ``depth`` (scalar); ``noise`` (1-DoF). (One factor per obs; batch
             to DIM = Nk if preferred -- then ``ray``/``depth`` are stacked and the Jacobians gain rows.)
    OUTPUT:  residual (1,) = e_z^T( s_i R_i ( e^{rho} · depth · ray ) + t_i )  -> 0
             (the DA3-depth point lies on the z = 0 ground).
    IMPLEMENT ``error``: Jacobians H[0] (1, 7) on the Sim(3) tangent and H[1] (1, 1) = d/drho.
             INTENTIONALLY OFF for the anchor-free submap graph; part of the full single-window graph.
    """
    DIM = 1

    def __init__(self, hi_key, rho_key, ray, depth, noise) -> None:
        super().__init__([hi_key, rho_key], noise)
        self.ray = np.asarray(ray, float)         # (3,) K^{-1}[u,v,1]
        self.depth = float(depth)                 # DA3 depth at that pixel


class EgoOnGroundFactor(_Sim3Factor):
    """Unary on H_i: the ego camera centre sits one camera-height above the ground.

    INPUTS:  ``hi_key`` (a ``Similarity3``); ``sensor2ego`` (4,4) camera->ego extrinsic; ``cam_height``
             scalar (= sensor2ego[2,3]); ``noise`` (1-DoF).
    OUTPUT:  residual (1,) = e_z^T( ego centre expressed via H_i and sensor2ego ) - cam_height.
    IMPLEMENT ``error``: Jacobian H[0] is (1, 7) with the §5.4 SPARSITY -- d/ds_i = 0 (the height
             residual does not see the per-frame scale), which is why it fixes the absolute scale
             without fighting the depth-ratio. (rho is not a key here.)
    """
    DIM = 1

    def __init__(self, hi_key, sensor2ego, cam_height, noise) -> None:
        super().__init__([hi_key], noise)
        self.sensor2ego = np.asarray(sensor2ego, float)   # (4,4)
        self.cam_height = float(cam_height)


class SubmapAlignmentFactor(_Sim3Factor):
    """Submap-alignment on a shared camera (§14.2/§14.4), coupling BOTH scales -- the reason it must be
    custom, not a stock ``BetweenFactorSimilarity3``.

    The H variables are on each submap's COLMAP frame, but the measurable overlap quantity is the DA3
    depth ratio. The two-scale consistency (§14.3) makes the true H-scale ratio depend on the rho
    VARIABLES: s_n/s_m = e^{rho_m - rho_n} * (z_da3_m/z_da3_n) = exp((rho_m - rho_n) - log_s). Because it
    depends on rho_m, rho_n (optimized), a fixed-measurement between-factor is only an approximation
    (build_submap_graph uses one with the init rho); the exact factor is here.

    INPUTS:  ``hm_key``, ``hn_key`` (the two submaps' shared-camera ``Similarity3``), ``rho_m_key``,
             ``rho_n_key`` (Doubles); ``log_s`` = log median(z_da3_n/z_da3_m) at the overlap; ``noise``
             (7-DoF Diagonal: tight on the 6 SE(3) dims -- the shared camera coincides -- and the DA3
             log-MAD on the scale row).
    OUTPUT:  residual (7,) = Logmap( meas^{-1} . (H^m^{-1} H^n) ), meas = Similarity3(I, 0, s_hat) with
             s_hat = exp((rho_m - rho_n) - log_s). Only the scale (lambda) component depends on rho.
    IMPLEMENT ``error``: Jacobians H[0] (7,7) on H^m, H[1] (7,7) on H^n, H[2] (7,1) and H[3] (7,1) on
             rho_m/rho_n -- nonzero only in the lambda row (d s_hat/d rho_m = s_hat, d/d rho_n = -s_hat).
    """
    DIM = 7

    def __init__(self, hm_key, hn_key, rho_m_key, rho_n_key, log_s, noise) -> None:
        super().__init__([hm_key, hn_key, rho_m_key, rho_n_key], noise)
        self.log_s = float(log_s)                         # log median DA3(n)/DA3(m) at the overlap
