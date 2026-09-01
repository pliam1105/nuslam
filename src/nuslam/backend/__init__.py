"""Estimator backend -- core, left unimplemented by design.

This package holds only the *seam*: the input/output contract (:class:`SlamInputs`,
:class:`SlamEstimate`), the estimator entry point (:class:`MonocularSLAM`, body
unimplemented), and the custom-factor stubs (:mod:`.factors`). The factor-graph
design, the custom factors' residuals/Jacobians, the scale-resolution logic and
the sensor fusion are built here. Everything around this package (data, frontend,
viz, eval) is plumbing and is complete.
"""
from .graph import MonocularSLAM, SlamEstimate, SlamInputs

__all__ = ["MonocularSLAM", "SlamInputs", "SlamEstimate"]
