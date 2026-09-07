"""Reconstruction core -- written by hand (representation, init, optimization, loss).

This package holds the Gaussian-splat reconstruction: the representation and its
initialization, the gsplat calls, the pose refinement, the losses (photometric +
metric anchor), and the scale resolution. It is core substance, built here, not
generated. Only the depth back-projection seam is stubbed so the surrounding viz
and evaluation plumbing have something to call.
"""
from .depth_init import unproject_depth_to_world

__all__ = ["unproject_depth_to_world"]
