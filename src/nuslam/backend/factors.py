"""§14 Sim(3) custom factors -- left unimplemented by design.

    ############################################################################
    #  Core estimator substance. Intentionally left unimplemented.             #
    #  The Sim(3) factors that unify COLMAP + DA3 into a metric reconstruction  #
    #  -- their residuals, variable connectivity, and Jacobians -- are the      #
    #  heart of the metric-scale mechanism and are derived and written here,    #
    #  not generated.                                                          #
    ############################################################################

This module is the single home for the project's custom-factor DEFINITIONS. All are
the §14 Sim(3) unification factors, consumed by ``sim3_graph.build_sim3_graph`` /
``build_submap_graph``: ``Sim3GaugePriorFactor``, ``ColmapRelativeSim3Factor``,
``DepthRatioPriorFactor``, ``GroundAnchorSim3Factor``, ``EgoOnGroundFactor``,
``SubmapScaleCrossCheckFactor``. ``GroundAnchorSim3Factor`` + ``EgoOnGroundFactor``
are the current graph form of the ground/ego-height scale anchor (CLAUDE.md rung 2 --
where monocular scale resolves).

They act on the per-frame Similarity3 variable H_i = (s_i, R_i, t_i) and the depth-
ratio scalar rho (single-window) / rho_m (one per submap). Every one is a
``gtsam.CustomFactor`` -- this GTSAM build has no templated ``BetweenFactorSimilarity3``,
and ``Similarity3`` is not even Values-storable here, so the variable representation
(Pose3 + Double, or a Vector7) is itself a design choice made alongside these residuals.

All are stubs: they name the seam and hold the design notes; the residuals, variable
connectivity, and Jacobians are derived and written by the author (§3/§8).

GTSAM offers two routes for a custom factor; the choice (and why) is a design decision:

  * ``gtsam.CustomFactor(noise_model, keys, error_func)`` -- Python error callback,
    Jacobians filled numerically or explicitly into the passed ``H`` list.
  * subclass a ``NoiseModelFactor`` in C++/pybind -- analytic, faster.

Design decisions to pin down before writing a line:
  - The Values-storable Sim(3) representation (Pose3 + Double s_i, or a Vector7).
  - Ground anchor: which road pixels feed it, inlier gating, and when it is skipped
    (it is OFF entirely for the anchor-free submap graph).
  - Jacobians: analytic or numerical? The §5.4 sparsity of the ego-height residual
    (d/ds = 0, d/drho = 0) is the claim to exploit and verify.
  - Noise model + robustifier (Huber?) and its threshold, with justification.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- shared base + callback contract
class _Sim3Factor:
    """Base for the §14 CustomFactors: stores the keys + noise model and wires the error
    callback. Subclasses store their measurement in ``__init__`` (the INPUTS), set ``DIM``
    (the residual length, i.e. the OUTPUT), and IMPLEMENT ``error`` -- the one method that
    is author core.

    ``error`` is the ``gtsam.CustomFactor`` callback (the "eval"):

        error(values, H=None) -> np.ndarray of shape (DIM,)

    Given the current ``values``, it returns the residual, and when ``H`` is not None it
    writes each variable's analytic Jacobian into ``H[k]`` -- one (DIM, tangent_dim) block
    per key, in ``self.keys`` order (tangent_dim = 7 for a Sim(3) H_i under the chosen
    representation, 1 for a rho scalar). The noise model whitens it; leave whitening to gtsam.
    ``as_custom_factor`` (plumbing) turns this into the graph-ready factor.
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


# ----------------------------------------------------------------------------- §14.4 factors
class Sim3GaugePriorFactor(_Sim3Factor):
    """Unary prior on H_0 that pins the graph's 7-DoF gauge to the initialization.

    INPUTS:  ``h0_key`` (the H_0 variable); ``init`` -- the target Sim(3) to prior toward
             (from ``Sim3GraphInputs.init_R/init_t/init_s[0]``); ``noise`` (7-DoF).
    OUTPUT:  residual (7,) = Logmap(H_0^{-1} · init) in the representation's tangent.
    IMPLEMENT ``error``: Jacobian H[0] is (7, 7). Fixes rotation + translation (6 DoF); with
             NO ground/ego anchor the absolute SCALE is also a gauge freedom pinned here
             (tight sigma on s_0) -> metric-up-to-one-global-scale. The tangent + sigma split
             (pose vs scale) are the design choices.
    """
    DIM = 7

    def __init__(self, h0_key, init, noise) -> None:
        super().__init__([h0_key], noise)
        self.init = init                      # target H_0 value (Similarity3 or (R, t, s))


class ColmapRelativeSim3Factor(_Sim3Factor):
    """Binary factor on (H_{i-1}, H_i) from a consecutive COLMAP relative measurement.

    INPUTS:  ``prev_key``, ``cur_key`` (H_{i-1}, H_i); measurement ``rel_R`` (3,3) and
             ``rel_t`` (3,) in COLMAP LOCAL units (from ``Sim3GraphInputs.colmap_rel_R/t``);
             ``noise`` (7-DoF).
    OUTPUT:  residual (7,) = [ rotation error (3) | translation-in-local-units error (3) |
             log(s_i / s_{i-1}) (1) ] -- the last term is the scale CONSTANCY s_i = s_{i-1}
             (COLMAP is one rigid reconstruction, so its unit is constant along the chain).
    IMPLEMENT ``error``: Jacobians H[0], H[1] each (7, 7).
    """
    DIM = 7

    def __init__(self, prev_key, cur_key, rel_R, rel_t, noise) -> None:
        super().__init__([prev_key, cur_key], noise)
        self.rel_R = np.asarray(rel_R, float)     # (3,3) R_{i-1}^T R_i
        self.rel_t = np.asarray(rel_t, float)     # (3,)  R_{i-1}^T (c_i - c_{i-1}), local units


class DepthRatioPriorFactor(_Sim3Factor):
    """Unary on the depth-ratio scalar rho (single-window) or rho_m (per submap).

    INPUTS:  ``rho_key``; ``log_ratio_median`` = log(depth_ratio_median) for this window;
             ``noise`` (1-DoF, sigma from the ratio MAD -- §5.6 honest noise model).
    OUTPUT:  residual (1,) = rho - log_ratio_median.  Ties DA3 depth units to COLMAP units
             (e^rho: DA3 -> COLMAP); one per window in the submap graph.
    IMPLEMENT ``error``: Jacobian H[0] is (1, 1) = [[1.0]].
    """
    DIM = 1

    def __init__(self, rho_key, log_ratio_median, noise) -> None:
        super().__init__([rho_key], noise)
        self.log_ratio_median = float(log_ratio_median)


class GroundAnchorSim3Factor(_Sim3Factor):
    """Binary on (H_i, rho): a road-pixel back-projection constrained to the ground plane.

    INPUTS:  ``hi_key``, ``rho_key``; a road observation ``ray`` (3,) = K^{-1}[u,v,1] and its
             DA3 ``depth`` (scalar); ``noise`` (1-DoF). (One factor per obs; batch to DIM = Nk
             if preferred -- then ``ray``/``depth`` are stacked and the Jacobians gain rows.)
    OUTPUT:  residual (1,) = e_z^T( s_i R_i ( e^{rho} · depth · ray ) + t_i )  -> 0
             (the DA3-depth point lies on the z = 0 ground).
    IMPLEMENT ``error``: Jacobians H[0] (1, 7) and H[1] (1, 1) = d/drho. INTENTIONALLY OFF
             for the anchor-free submap graph; part of the full single-window §14 graph.
    """
    DIM = 1

    def __init__(self, hi_key, rho_key, ray, depth, noise) -> None:
        super().__init__([hi_key, rho_key], noise)
        self.ray = np.asarray(ray, float)         # (3,) K^{-1}[u,v,1]
        self.depth = float(depth)                 # DA3 depth at that pixel


class EgoOnGroundFactor(_Sim3Factor):
    """Unary on H_i: the ego camera centre sits one camera-height above the ground.

    INPUTS:  ``hi_key``; ``sensor2ego`` (4,4) camera->ego extrinsic; ``cam_height`` scalar
             (= sensor2ego[2,3]); ``noise`` (1-DoF).
    OUTPUT:  residual (1,) = e_z^T( ego centre expressed via H_i and sensor2ego ) - cam_height.
    IMPLEMENT ``error``: Jacobian H[0] is (1, 7) with the §5.4 SPARSITY -- d/ds_i = 0 (the
             height residual does not see the per-frame scale), which is why it fixes the
             absolute scale without fighting the depth-ratio. (rho is not a key here.)
    """
    DIM = 1

    def __init__(self, hi_key, sensor2ego, cam_height, noise) -> None:
        super().__init__([hi_key], noise)
        self.sensor2ego = np.asarray(sensor2ego, float)   # (4,4)
        self.cam_height = float(cam_height)


class SubmapScaleCrossCheckFactor(_Sim3Factor):
    """OPTIONAL soft factor encoding the §14.2 cross-check between two adjacent submaps.

    Adjacent submaps already share the same H_i on their overlap frames (a shared variable),
    which is what anchors them -- so this factor is NOT needed for graph connectivity. It only
    encodes the consistency check as a soft residual so a disagreement is penalized/visible.

    INPUTS:  ``rho_m_key``, ``rho_n_key`` (the two windows' depth-ratio scalars); ``noise``
             (1-DoF, loose). (The author may instead formulate it on the shared-camera scales.)
    OUTPUT:  residual (1,) = rho_m - rho_n  (the two windows' DA3->COLMAP ratios should agree;
             a large value localizes a bad window, e.g. submap 0 here).
    IMPLEMENT ``error``: Jacobians H[0] (1,1) = [[1]], H[1] (1,1) = [[-1]]. Optional -- the
             author decides whether to include it and its exact residual.
    """
    DIM = 1

    def __init__(self, rho_m_key, rho_n_key, noise) -> None:
        super().__init__([rho_m_key, rho_n_key], noise)
