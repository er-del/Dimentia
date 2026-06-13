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

from typing import Any

import cv2
import numpy as np

from dimendia.config import ScoringWeights, SemanticBackend
from dimendia.types import Frame, TrackedObject


class ProjectileSelector:
    def __init__(
        self,
        weights: ScoringWeights | None = None,
        semantic_backend: SemanticBackend = SemanticBackend.HAAR,
    ):
        self.weights = weights or ScoringWeights()
        self.semantic_backend = semantic_backend
        self._cascade: Any = None

    def _face_cascade(self) -> Any:
        if self._cascade is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
            self._cascade = cv2.CascadeClassifier(path)
        return self._cascade

    def _semantic_scores(self, objects: list[TrackedObject], frame: Frame) -> dict[int, float]:
        """Return id -> 1.0 when the object's mask overlaps a detected face >30%."""
        cascade = self._face_cascade()
        if cascade.empty():
            return {}
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        scores: dict[int, float] = {}
        for obj in objects:
            area = float(obj.mask.sum())
            hit = 0.0
            if area > 0:
                for fx, fy, fw, fh in faces:
                    inside = float(obj.mask[fy : fy + fh, fx : fx + fw].sum())
                    if inside / area > 0.30:
                        hit = 1.0
                        break
            scores[obj.object_id] = hit
        return scores

    def select(
        self,
        objects: list[TrackedObject],
        frame_shape: tuple[int, int],
        frame: Frame | None = None,
    ) -> int | None:
        """Score objects in place and return the id of the primary candidate."""
        if not objects:
            return None
        h, w = frame_shape
        diag = float(np.hypot(h, w))
        half_diag = diag / 2.0
        cx, cy = w / 2.0, h / 2.0
        vel_ref = max(1.0, 0.05 * diag)  # ~5% of diagonal/frame == "fast"
        wgt = self.weights

        semantic_scores: dict[int, float] = {}
        if wgt.semantic > 0 and frame is not None and self.semantic_backend != SemanticBackend.NONE:
            semantic_scores = self._semantic_scores(objects, frame)

        for obj in objects:
            proximity = float(np.clip(obj.mean_depth, 0.0, 1.0))
            vmag_raw = float(np.hypot(obj.velocity[0], obj.velocity[1]))
            velocity = float(np.clip(vmag_raw / vel_ref, 0.0, 1.0))
            dist = float(np.hypot(obj.centroid[0] - cx, obj.centroid[1] - cy))
            framing = float(np.clip(1.0 - dist / half_diag, 0.0, 1.0))
            attention = float(np.clip(obj.saliency, 0.0, 1.0))
            semantic = semantic_scores.get(obj.object_id, 0.0)

            obj.proximity = proximity
            obj.velocity_mag = velocity
            obj.framing = framing
            obj.score = (
                wgt.depth * proximity
                + wgt.motion * velocity
                + wgt.saliency * attention
                + wgt.center * framing
                + wgt.semantic * semantic
            )

        primary = max(objects, key=lambda o: o.score)
        return primary.object_id
