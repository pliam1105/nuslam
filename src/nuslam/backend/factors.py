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
CLAUDE.md rung 2), ``HorizontalGaugeFactor`` (the horizontal-translation + heading first-pose gauge,
§14.5, replacing the scale-fixing prior), and ``SubmapAlignmentFactor`` (the submap-alignment that
couples the two rho scales, §14.3). The GENERIC §14.4 factors are stock and are wired in ``sim3_graph``:

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
import gtsam
from nuslam.backend import sim3_lie as Lie

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

    def error(self, values: gtsam.Values, H : list[np.ndarray] = None) -> np.ndarray:
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
    
    def error(self, values, H=None):
      rho = values.atDouble(self.keys[1])
      p_local = float(np.exp(rho))*self.depth*self.ray

      Hi = values.atSimilarity3(self.keys[0])
      Jq = Lie.point_transform_jacobian(Hi, p_local)
      s, R = Hi.scale(), Hi.rotation().matrix()
      
      if H is not None:
        H[0] = Jq[2:3, :]
        H[1] = np.array([[ (s*R@p_local)[2] ]])

      return np.array([(s*R@p_local)[2]+s*Hi.translation()[2]])

class EgoOnGroundFactor(_Sim3Factor):
    """Unary on H_i: the ego origin sits ON the ground plane (z = 0).

    INPUTS:  ``hi_key`` (a ``Similarity3``); ``sensor2ego`` (4,4) camera->ego extrinsic; ``noise`` (1-DoF).
    OUTPUT:  residual (1,) = e_z^T( X_ego ), where the ego origin in the metric world is
             X_ego = s_i t_i (the metric camera CENTRE) + R_i p_local (the metric sensor->ego offset,
             UNSCALED -- p_local = the ego origin in the camera frame is already in metres). Target is 0:
             the ego origin lies on the road plane, so the camera floats at its (metric) mounting height
             above it. NOTE: this is NOT e_z^T transformFrom(p_local) (that would double-scale the metric
             offset), and there is NO cam_height term -- the metric height lives in p_local, not a target.
    WHY IT BREAKS SCALE (not scale-degenerate): the lever arm R_i p_local is metric and UNSCALED, so as
             s_i -> 0 the residual tends to (R_i p_local)_z != 0 -- the collapse is penalised. That fixed
             metric offset is exactly what pins the absolute scale; the road-plane (z=0) factors alone are
             scale-free and would let s collapse, so this anchor must carry the metric weight.
    IMPLEMENT ``error``: Jacobian H[0] is (1, 7). In GTSAM's [omega | rho | lambda] tangent it is
             [ (-R_i[p_local]_x)[z,:] | (s_i R_i)[z,:] | 0 ]:
               - omega: only R_i p_local rotates (the centre's omega-derivative is 0);
               - rho:   only the centre s_i t_i moves (d centre/d rho = s_i R_i; R_i p_local has no rho);
               - lambda = 0: R_i p_local has no scale, and the centre's lambda-derivative vanishes.
             Depends on s_i through the centre (that scale-sensitivity pins metric scale); not on rho.
    """
    DIM = 1

    def __init__(self, hi_key, sensor2ego, noise) -> None:
        super().__init__([hi_key], noise)
        self.sensor2ego = np.asarray(sensor2ego, float)   # (4,4)
        self.p_local = np.linalg.inv(self.sensor2ego)[:3, 3]   # ego origin in camera frame (metric)

    def error(self, values, H=None):
      Hi = values.atSimilarity3(self.keys[0])
      s, R = Hi.scale(), Hi.rotation().matrix()

      if H is not None:
        H[0] = np.hstack([ (-R @ Lie.skew(self.p_local))[2:3,:], (s*R)[2:3,:], [[0.0]] ])

      return np.array([(R@self.p_local)[2]+s*Hi.translation()[2]])


class HorizontalGaugeFactor(_Sim3Factor):
    """Unary on the FIRST pose H_0: the horizontal-translation + heading gauge (§14.5) that REPLACES the
    scale-fixing gauge prior. It fixes 3 of the 6 SE(3) gauge DoF -- horizontal position (x, y) and yaw --
    and leaves z, roll, pitch to the ground plane and SCALE FREE (resolved by the ego anchor). So the full
    7-DoF Sim(3) gauge is pinned jointly: this factor (x, y, yaw) + ground plane (z, roll, pitch) + ego
    (scale). Both residual components are scale-invariant by construction, so it does NOT touch the gauge
    that the anchors exist to determine.

    INPUTS:  ``hi_key`` (the first pose's ``Similarity3`` H_0); ``noise`` (4-DoF). Probe points are the
             camera-frame origin ``p0 = (0,0,0)`` and forward axis ``p1 = (0,0,1)``; targets are the world
             horizontal origin ``(0,0)`` and heading ``+x = (1,0)`` (stored for clarity/overridability).
    OUTPUT:  residual (4,):
               r[0:2] = ( H_0 . p0 )_{xy}              - (0,0)      # centre's xy -> horizontal position
               r[2:4] = normalize( ( H_0 . p1 )_{xy} ) - (1,0)     # forward axis projected to ground -> yaw
             Component 1 is scale-free at the solution (its zero-set (t_xy = 0) is independent of s); the
             normalization in component 2 removes the scale factor entirely (a unit direction).
    IMPLEMENT ``error``: Jacobian H[0] is (4, 7), both rows built from :func:`sim3_lie.point_transform_jacobian`:
               - rows 0:2 : take rows [0,1] (the xy rows) of point_transform_jacobian(H_0, p0);
               - rows 2:4 : with u = (H_0 . p1)_{xy}, n = u/||u||, chain the unit-normalization Jacobian
                            (I2 - n n^T)/||u||  onto rows [0,1] of point_transform_jacobian(H_0, p1).
             The normalization projector kills the radial (scale) direction -> the yaw residual is rank-1,
             so the 4-vector is naturally rank-3 (x, y, yaw), matching the DoF it fixes.
    """
    DIM = 4

    def __init__(self, hi_key, noise, *, p0=(0., 0., 0.), p1=(0., 0., 1.),
                 target_xy=(0., 0.), target_heading=(1., 0.)) -> None:
        super().__init__([hi_key], noise)
        self.p0 = np.asarray(p0, float)                   # camera-frame origin -> horizontal position probe
        self.p1 = np.asarray(p1, float)                   # camera-frame forward axis -> heading probe
        self.target_xy = np.asarray(target_xy, float)     # world horizontal position target (0,0)
        self.target_heading = np.asarray(target_heading, float)   # world heading target +x (1,0)
    
    def error(self, values, H=None):
      Hi = values.atSimilarity3(self.keys[0])
      
      s, R, t = Hi.scale(), Hi.rotation().matrix(), Hi.translation()

      q_0 = s*(R@self.p0+t)
      q_front = s*R@(self.p1-self.p0)

      Jq_0 = Lie.point_transform_jacobian(Hi, self.p0)
      Jq_1 = Lie.point_transform_jacobian(Hi, self.p1)

      r_pos = q_0[:2]-self.target_xy
      n = q_front[:2]/np.linalg.norm(q_front[:2])
      r_heading = n -self.target_heading
      
      if H is not None:
        H[0] = np.concatenate([Jq_0[:2, :], ((np.eye(2)-np.outer(n,n))/np.linalg.norm(q_front[:2]))@(Jq_1-Jq_0)[:2, :]], axis=0)

      return np.concatenate([r_pos, r_heading], axis=0)


class UprightGaugeFactor(_Sim3Factor):
    """Unary on the FIRST pose H_0: a SOFT upright constraint on the down direction (roll, pitch) -- the
    same rotated-axis-normalization construction as HorizontalGaugeFactor, but on the camera's DOWN axis
    instead of the forward axis. It exists to remove the zero-cost collapse minimum: at s -> 0 the ground
    factors vanish and free the global roll/pitch, letting rotation null the ego lever; pinning the first
    pose's down axis toward vertical denies that freedom. Soft ("not too hard"): it must not fight the
    ground, which sets roll/pitch metrically at the true (near-upright) solution -- it only breaks the
    degeneracy. Constrains the DOWN direction, NOT the heading (that is HorizontalGaugeFactor's yaw) and
    NOT position/scale.

    INPUTS:  ``hi_key`` (H_0 ``Similarity3``); ``noise`` (2-DoF); ``down_axis`` = camera-frame down (nuScenes
             / OpenCV convention +y = (0,1,0)); ``target_xy`` = (0,0) (a vertical world axis has zero xy).
    OUTPUT:  residual (2,) = normalize( (s R a_down) )_{xy} - (0,0). The world down axis is the (unit)
             rotated body down axis; its horizontal (xy) component is zero iff the camera is upright, so
             this drives roll+pitch toward vertical. Yaw-, position-, and scale-invariant (the residual
             vector rotates with yaw but its zero-set does not; normalization removes scale; no translation
             enters).
    IMPLEMENT ``error``: with v = s R a_down, d = v/||v||, the Jacobian is the xy rows of
             (I3 - d d^T)/||v|| @ [ -sR[a_down]_x | 0 | +sR a_down ]  (the rotated-axis Jacobian). The
             projector kills the radial (scale) direction so the lambda column is 0; the rho block is 0
             (no translation). Rank 2 (roll, pitch).
    """
    DIM = 2

    def __init__(self, hi_key, noise, *, down_axis=(0., 1., 0.), target_xy=(0., 0.)) -> None:
        super().__init__([hi_key], noise)
        self.a = np.asarray(down_axis, float)             # camera-frame down axis
        self.target_xy = np.asarray(target_xy, float)     # world horizontal component of a vertical axis = (0,0)

    def error(self, values, H=None):
      Hi = values.atSimilarity3(self.keys[0])
      s, R = Hi.scale(), Hi.rotation().matrix()
      v = s * R @ self.a                                  # world down axis (scaled); normalize -> unit direction
      nv = np.linalg.norm(v); d = v / nv

      if H is not None:
        Jv = np.hstack([-s * R @ Lie.skew(self.a), np.zeros((3, 3)), (s * R @ self.a).reshape(3, 1)])
        H[0] = (((np.eye(3) - np.outer(d, d)) / nv) @ Jv)[:2, :]

      return d[:2] - self.target_xy


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
    
    def error(self, values, H=None):
      Hm = values.atSimilarity3(self.keys[0])
      Hn = values.atSimilarity3(self.keys[1])
      rho_m = values.atDouble(self.keys[2])
      rho_n = values.atDouble(self.keys[3])

      Log = gtsam.Similarity3.Logmap
      rel = Hm.between(Hn)
      s_hat = float(np.exp((rho_m - rho_n) - self.log_s))
      meas = gtsam.Similarity3(gtsam.Rot3(), np.zeros(3), s_hat)
      r = Log(meas.inverse().compose(rel))
      Jri = Lie.right_jacobian_inv(r)
      Jli = Lie.left_jacobian_inv(r)
      e_lam = np.zeros((7,1)); e_lam[6,0] = 1.0

      if H is not None:
        H[0] = Jri @ (-Lie.adjoint(rel.inverse()))
        H[1] = Jri
        H[2] = -Jli @ e_lam
        H[3] = Jli @ e_lam

      return r