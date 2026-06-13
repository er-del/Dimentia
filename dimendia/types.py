"""Shared array conventions and lightweight data containers.

Conventions used across the whole pipeline:

* **Frame** — ``np.uint8`` array of shape ``(H, W, 3)`` in **RGB** order.
* **DepthMap** — ``np.float32`` array of shape ``(H, W)`` normalized to ``[0, 1]``
  where **1.0 == nearest** to the camera and **0.0 == farthest**. This is a
  disparity-like ("inverse depth") convention, matching Depth Anything's output
  and making the per-object *proximity* term a direct mean of the mask region.
* **Mask** — boolean array of shape ``(H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Type aliases (documentation only; numpy has no static dtype/shape generics here).
Frame = np.ndarray
DepthMap = np.ndarray
Mask = np.ndarray


@dataclass
class TrackedObject:
    """A segmented, tracked object with the cues used for extrusion scoring."""

    object_id: int
    mask: Mask
    centroid: tuple[float, float]  # (x, y) in pixels
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) px/frame
    mean_depth: float = 0.0  # 0..1, higher == nearer
    area_ratio: float = 0.0  # mask area / frame area
    saliency: float = 0.0  # 0..1
    proximity: float = 0.0  # 0..1, normalized depth cue
    velocity_mag: float = 0.0  # 0..1, normalized speed cue
    framing: float = 0.0  # 0..1, how centered the object is
    score: float = 0.0  # final weighted extrusion score


@dataclass
class FrameResult:
    """Per-frame artifacts produced by the pipeline (useful for previews/debug)."""

    index: int
    frame: Frame
    depth: DepthMap
    objects: list[TrackedObject] = field(default_factory=list)
    primary_id: int | None = None
    rendered: Frame | None = None
