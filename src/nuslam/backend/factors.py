"""Custom factors -- AUTHOR WRITES THIS FILE (CLAUDE.md s3).

    ############################################################################
    #  This is core backend substance. It is intentionally left unimplemented. #
    #  The ground-plane fit and the wheel-contact-to-plane factor -- and their  #
    #  residuals and Jacobians -- are the heart of the interview deep-dive and  #
    #  MUST be the author's own work. Do not let an assistant fill these in.    #
    ############################################################################

These stubs exist only to name the seam and hold your notes. Nothing about the
residual arguments, the variable connectivity, or whether the plane is a graph
variable vs. a RANSAC-estimated constant is decided here -- those are exactly the
choices you must be able to derive and defend.

gtsam gives you two routes for a custom factor; the choice (and why) is yours:

  * ``gtsam.CustomFactor(noise_model, keys, error_func)`` -- Python error
    callback, Jacobians filled numerically or by you into the passed ``H`` list.
  * subclass a ``NoiseModelFactor`` in C++/pybind -- analytic, faster.

Design decisions to pin down before writing a line (each is deep-dive fodder):
  - Ground plane parametrization: (n, d)? a point+normal? an OrientedPlane3?
    Is it a graph variable or fit by RANSAC outside the graph each keyframe?
  - Wheel-contact factor: residual = signed distance of each wheel-contact point
    to the plane? In which frame is the contact point expressed? Which variables
    does it touch (pose only, or pose + plane)?
  - Jacobians: analytic or numerical? If analytic, derive d(residual)/d(pose)
    on the SE(3) tangent and be ready to reproduce it at a whiteboard.
  - Noise model + robustifier (Huber?) and its threshold, with justification.
  - Inlier gating: when is the ground assumption invalid and the factor skipped?
"""
from __future__ import annotations


class GroundPlaneFactor:
    """RANSAC-supported local ground-plane constraint. AUTHOR WRITES.

    See module banner. Left unimplemented on purpose.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "GroundPlaneFactor is core backend substance (CLAUDE.md s3) -- the author "
            "designs the parametrization, residual, and Jacobian. Not to be autofilled."
        )


class WheelContactToPlaneFactor:
    """Constrains the vehicle's wheel-contact points to the ground plane. AUTHOR WRITES.

    This factor is where monocular scale resolves (build ladder rung 2). Its
    residual and Jacobian are the centerpiece of the deep-dive -- see module banner.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "WheelContactToPlaneFactor is core backend substance (CLAUDE.md s3) -- the "
            "author designs the residual (contact-point-to-plane distance), its frame, "
            "and its Jacobian. Not to be autofilled."
        )
