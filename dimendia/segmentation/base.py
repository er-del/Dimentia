"""Object tracker interface and backend factory (Stage B).

A tracker turns a frame (plus its depth map and optical flow) into a list of
:class:`~dimendia.types.TrackedObject` with **stable ids across frames**. SAM2 is
used when installed; otherwise a classical motion/saliency/depth tracker runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dimendia.config import PipelineConfig, SegmentationBackend
from dimendia.logging import get_logger
from dimendia.types import DepthMap, Frame, TrackedObject

log = get_logger(__name__)


class ObjectTracker(ABC):
    name: str = "base"

    @abstractmethod
    def track(self, frame: Frame, depth: DepthMap, flow: np.ndarray | None) -> list[TrackedObject]:
        """Return tracked objects for this frame with stable ids."""
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027 - optional hook, concrete no-op by design
        """Clear per-clip tracking state."""


def build_tracker(config: PipelineConfig, *, device: str | None = None) -> ObjectTracker:
    from dimendia.segmentation.classical_tracker import ClassicalTracker

    backend = config.segmentation_backend

    def classical() -> ObjectTracker:
        log.info("segmentation backend: classical (motion + saliency + depth)")
        return ClassicalTracker(config)

    if backend == SegmentationBackend.CLASSICAL:
        return classical()

    if backend in (SegmentationBackend.AUTO, SegmentationBackend.SAM2):
        try:
            from dimendia.segmentation.sam2_tracker import SAM2Tracker

            tracker = SAM2Tracker(config, device=device)
            log.info("segmentation backend: SAM2")
            return tracker
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            log.warning("SAM2 unavailable (%s)", exc)
            if backend == SegmentationBackend.SAM2:
                return classical()

    return classical()
