"""Estimator backend -- the §14 Sim(3) factor graph (core estimator substance).

Holds the factor-graph work for unifying COLMAP + DA3 into a metric reconstruction:

  * :mod:`.sim3_graph` -- the input contracts (:class:`Sim3GraphInputs` single-window,
    :class:`SubmapGraphInputs` / :class:`Submap` per-window) + their §4 input-prep, and
    the build/solve HARNESS (:func:`build_sim3_graph`, :func:`build_submap_graph`,
    :func:`solve`, :func:`solve_incremental`) with the factors left as author stubs.
  * :mod:`.factors` -- the custom-factor DEFINITIONS (design notes; residuals + Jacobians
    written by the author, §3/§8).

The Sim(3) variable representation, the custom factors, and the scale resolution are the
author's core work. Everything around this package (data, frontend, recon, viz, eval) is
plumbing and is complete.
"""
from .sim3_graph import (
    Sim3GraphInputs,
    Submap,
    SubmapGraphInputs,
    build_sim3_graph,
    build_submap_graph,
    prepare_sim3_inputs,
    prepare_submap_inputs,
    solve,
    solve_incremental,
)

__all__ = [
    "Sim3GraphInputs",
    "prepare_sim3_inputs",
    "build_sim3_graph",
    "Submap",
    "SubmapGraphInputs",
    "prepare_submap_inputs",
    "build_submap_graph",
    "solve",
    "solve_incremental",
]
