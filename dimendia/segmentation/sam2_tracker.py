"""SAM2 segmentation backend (used when ``sam2`` + weights are installed).

Strategy: our cue map (motion + saliency + depth) supplies positive **prompt
points** at foreground peaks; SAM2's image predictor turns each into a
high-quality mask; identity/velocity are assigned by the same IoU association used
by the classical tracker. This keeps SAM2 focused on what it is best at (masks)
while reusing DIMENDIA's selection logic.

Requires environment variables ``SAM2_CHECKPOINT`` (path to weights) and
optionally ``SAM2_CONFIG`` (model config name). Raises on missing deps/config so
the factory degrades to the classical tracker.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from dimendia.config import PipelineConfig
from dimendia.segmentation.base import ObjectTracker
from dimendia.segmentation.classical_tracker import ClassicalTracker, _mask_iou
from dimendia.segmentation.saliency import spectral_residual_saliency
from dimendia.types import DepthMap, Frame, TrackedObject


class SAM2Tracker(ObjectTracker):
    name = "sam2"

    def __init__(self, config: PipelineConfig, device: str | None = None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = os.environ.get("SAM2_CHECKPOINT")
        if not checkpoint or not os.path.exists(checkpoint):
            raise RuntimeError("SAM2_CHECKPOINT env var must point to SAM2 weights")
        model_cfg = os.environ.get("SAM2_CONFIG", "sam2_hiera_l.yaml")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = build_sam2(model_cfg, checkpoint, device=device)
        self._predictor = SAM2ImagePredictor(model)
        self._cues = ClassicalTracker(config)  # cue map + id association
        self.max_objects = self._cues.max_objects

    def reset(self) -> None:
        self._cues.reset()

    def track(self, frame: Frame, depth: DepthMap, flow: np.ndarray | None) -> list[TrackedObject]:
        fg = self._cues._foreground_probability(frame, depth, flow)
        seeds = self._seed_points(fg)
        if not seeds:
            return []

        self._predictor.set_image(frame)
        saliency = spectral_residual_saliency(frame)
        objects: list[TrackedObject] = []
        for x, y in seeds:
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            mask = masks[int(np.argmax(scores))].astype(bool)
            obj = self._cues._describe(mask, depth, saliency, frame.shape[:2])
            if obj is not None:
                objects.append(obj)

        objects = _dedupe(objects, iou_thresh=0.6)[: self.max_objects]
        return self._cues._associate(objects)

    def _seed_points(self, fg: np.ndarray) -> list[tuple[int, int]]:
        """Pick up to ``max_objects`` well-separated peaks of the cue map."""
        blurred = cv2.GaussianBlur(fg, (0, 0), sigmaX=5.0)
        h, w = blurred.shape
        min_dist = int(0.08 * max(h, w))
        peaks: list[tuple[int, int]] = []
        work = blurred.copy()
        for _ in range(self.max_objects):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            if max_val < 0.35:
                break
            x, y = max_loc
            peaks.append((int(x), int(y)))
            cv2.circle(work, (x, y), min_dist, 0.0, thickness=-1)
        return peaks


def _dedupe(objects: list[TrackedObject], iou_thresh: float) -> list[TrackedObject]:
    kept: list[TrackedObject] = []
    for obj in objects:
        if all(_mask_iou(obj.mask, k.mask) < iou_thresh for k in kept):
            kept.append(obj)
    return kept
