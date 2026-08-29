"""Delegated frontend: offline point tracking + prompt-based road segmentation.

Two tracker backends (CoTracker learned, KLT classical) share Shi-Tomasi seeding
and grid replenishment; both are run offline and cached (see ``cache.py``). Model
imports are lazy at call time so importing this package pulls neither torch nor
transformers.
"""
from . import cache, refine, seeding
from .refine import RefineConfig, refine_tracks_subpixel
from .segmentation import RoadSegmenter, SegConfig
from .seeding import SeedConfig
from .tracking import CoTrackerFrontend, KLTFrontend, TrackConfig, make_tracker

__all__ = [
    "CoTrackerFrontend",
    "KLTFrontend",
    "make_tracker",
    "TrackConfig",
    "SeedConfig",
    "RefineConfig",
    "refine_tracks_subpixel",
    "RoadSegmenter",
    "SegConfig",
    "cache",
    "seeding",
    "refine",
]
