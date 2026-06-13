"""Depth estimator interface and backend factory.

All estimators return a ``float32`` depth map normalized to ``[0, 1]`` with the
convention **1.0 == nearest** (see :mod:`dimendia.types`). ``AUTO`` resolves to
the best backend that successfully loads, always falling back to the
weight-free :class:`~dimendia.depth.classical.ClassicalDepthEstimator` so the
pipeline runs anywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dimendia.config import DepthBackend
from dimendia.logging import get_logger
from dimendia.types import DepthMap, Frame

log = get_logger(__name__)


class DepthEstimator(ABC):
    """Estimate per-frame relative depth from a single RGB frame."""

    name: str = "base"

    @abstractmethod
    def estimate(self, frame: Frame) -> DepthMap:
        """Return a ``(H, W)`` float32 depth map in ``[0, 1]`` (1 == near)."""
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027 - optional hook, concrete no-op by design
        """Hook for stateful estimators; no-op by default."""


def build_depth_estimator(
    backend: DepthBackend = DepthBackend.AUTO,
    *,
    device: str | None = None,
) -> DepthEstimator:
    """Instantiate a depth backend, degrading gracefully to the classical one."""
    from dimendia.depth.classical import ClassicalDepthEstimator

    def classical() -> DepthEstimator:
        log.info("depth backend: classical (CPU, no weights)")
        return ClassicalDepthEstimator()

    if backend == DepthBackend.CLASSICAL:
        return classical()

    if backend in (DepthBackend.AUTO, DepthBackend.DEPTH_ANYTHING_V2):
        try:
            from dimendia.depth.depth_anything import DepthAnythingV2Estimator

            est = DepthAnythingV2Estimator(device=device)
            log.info("depth backend: Depth Anything V2 (%s)", est.device)
            return est
        except Exception as exc:  # noqa: BLE001 - intentional graceful fallback
            log.warning("Depth Anything V2 unavailable (%s)", exc)
            if backend == DepthBackend.DEPTH_ANYTHING_V2:
                return classical()

    if backend in (DepthBackend.AUTO, DepthBackend.MIDAS):
        try:
            from dimendia.depth.midas import MiDaSEstimator

            est = MiDaSEstimator(device=device)  # type: ignore[assignment]
            log.info("depth backend: MiDaS (%s)", est.device)
            return est
        except Exception as exc:  # noqa: BLE001
            log.warning("MiDaS unavailable (%s)", exc)

    return classical()


def _as_near_is_high(depth: np.ndarray, near_is_high: bool) -> DepthMap:
    """Normalize to ``[0, 1]`` with the package convention (1 == near)."""
    from dimendia.imageops import normalize01

    d = normalize01(depth)
    if not near_is_high:
        d = 1.0 - d
    return d.astype(np.float32)
