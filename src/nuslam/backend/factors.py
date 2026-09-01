"""Custom factors -- left unimplemented by design.

    ############################################################################
    #  Core estimator substance. Intentionally left unimplemented.             #
    #  The ground-plane fit and the wheel-contact-to-plane factor -- and their  #
    #  residuals and Jacobians -- are the heart of the metric-scale mechanism   #
    #  and are derived and written here, not generated.                         #
    ############################################################################

These stubs name the seam and hold the design notes. Nothing about the residual
arguments, the variable connectivity, or whether the plane is a graph variable
vs. a RANSAC-estimated constant is decided here -- those are the choices to be
derived and defended.

GTSAM offers two routes for a custom factor; the choice (and why) is a design
decision:

  * ``gtsam.CustomFactor(noise_model, keys, error_func)`` -- Python error
    callback, Jacobians filled numerically or explicitly into the passed ``H`` list.
  * subclass a ``NoiseModelFactor`` in C++/pybind -- analytic, faster.

Design decisions to pin down before writing a line:
  - Ground plane parametrization: (n, d)? a point+normal? an OrientedPlane3?
    Is it a graph variable or fit by RANSAC outside the graph each keyframe?
  - Wheel-contact factor: residual = signed distance of each wheel-contact point
    to the plane? In which frame is the contact point expressed? Which variables
    does it touch (pose only, or pose + plane)?
  - Jacobians: analytic or numerical? If analytic, derive d(residual)/d(pose)
    on the SE(3) tangent.
  - Noise model + robustifier (Huber?) and its threshold, with justification.
  - Inlier gating: when is the ground assumption invalid and the factor skipped?
"""
from __future__ import annotations


class GroundPlaneFactor:
    """RANSAC-supported local ground-plane constraint. Left unimplemented by design.

    See module banner. The parametrization, residual, and Jacobian are derived
    and written here.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "GroundPlaneFactor is core estimator substance, left unimplemented by design. "
            "Its parametrization, residual, and Jacobian are derived and written here."
        )


class WheelContactToPlaneFactor:
    """Constrains the vehicle's wheel-contact points to the ground plane.

    This factor is where monocular scale resolves (build ladder rung 2). Its
    residual and Jacobian are the core of the metric-anchor mechanism -- see
    module banner. Left unimplemented by design.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "WheelContactToPlaneFactor is core estimator substance, left unimplemented by "
            "design. Its residual (contact-point-to-plane distance), its frame, and its "
            "Jacobian are derived and written here."
        )
