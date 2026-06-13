"""Stage B — object segmentation, tracking, and extrusion selection."""

from dimendia.segmentation.base import ObjectTracker, build_tracker
from dimendia.segmentation.flow import OpticalFlow
from dimendia.segmentation.projectile_selector import ProjectileSelector

__all__ = ["ObjectTracker", "build_tracker", "OpticalFlow", "ProjectileSelector"]
