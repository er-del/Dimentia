"""Saliency-driven extrusion scoring — the core selection heuristic (Stage B).

Implements the spec's formula exactly::

    score = depth_w * proximity
          + motion_w * velocity
          + saliency_w * attention
          + center_w * framing

The highest-scoring tracked object becomes the **primary extrusion candidate**
(LDI Layer 0). All four cues are normalized to ``[0, 1]`` so the weights are
directly interpretable.
"""

from __future__ import annotations

import numpy as np

from dimendia.config import ScoringWeights
from dimendia.types import TrackedObject


class ProjectileSelector:
    def __init__(self, weights: ScoringWeights | None = None):
        self.weights = weights or ScoringWeights()

    def select(self, objects: list[TrackedObject], frame_shape: tuple[int, int]) -> int | None:
        """Score objects in place and return the id of the primary candidate."""
        if not objects:
            return None
        h, w = frame_shape
        diag = float(np.hypot(h, w))
        half_diag = diag / 2.0
        cx, cy = w / 2.0, h / 2.0
        vel_ref = max(1.0, 0.05 * diag)  # ~5% of diagonal/frame == "fast"
        wgt = self.weights

        for obj in objects:
            proximity = float(np.clip(obj.mean_depth, 0.0, 1.0))
            vmag_raw = float(np.hypot(obj.velocity[0], obj.velocity[1]))
            velocity = float(np.clip(vmag_raw / vel_ref, 0.0, 1.0))
            dist = float(np.hypot(obj.centroid[0] - cx, obj.centroid[1] - cy))
            framing = float(np.clip(1.0 - dist / half_diag, 0.0, 1.0))
            attention = float(np.clip(obj.saliency, 0.0, 1.0))

            obj.proximity = proximity
            obj.velocity_mag = velocity
            obj.framing = framing
            obj.score = (
                wgt.depth * proximity
                + wgt.motion * velocity
                + wgt.saliency * attention
                + wgt.center * framing
            )

        primary = max(objects, key=lambda o: o.score)
        return primary.object_id
