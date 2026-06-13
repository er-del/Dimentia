"""Weight-free monocular depth estimator (CPU).

Not a learned model — it fuses classic monocular cues into a *plausible* relative
depth so the entire pipeline (LDI, inpainting, pop-out rendering) runs end-to-end
without a GPU or any downloads. When ``dimendia[models]`` and weights are present,
:class:`~dimendia.depth.depth_anything.DepthAnythingV2Estimator` replaces this.

Cues fused (all pushed toward "1 == near"):

* **Defocus / local contrast** — in-focus, high-frequency regions are usually the
  subject and thus nearer.
* **Vertical (ground-plane) prior** — content lower in the frame tends to be nearer.
* **Center bias** — central content tends to be the foreground subject.

The fused estimate is snapped to image edges with a guided filter.
"""

from __future__ import annotations

import cv2
import numpy as np

from dimendia.depth.base import DepthEstimator
from dimendia.imageops import guided_filter, normalize01, to_gray_f32
from dimendia.types import DepthMap, Frame


class ClassicalDepthEstimator(DepthEstimator):
    name = "classical"

    def __init__(
        self,
        contrast_weight: float = 0.5,
        vertical_weight: float = 0.3,
        center_weight: float = 0.2,
    ):
        self.contrast_weight = contrast_weight
        self.vertical_weight = vertical_weight
        self.center_weight = center_weight

    def estimate(self, frame: Frame) -> DepthMap:
        gray = to_gray_f32(frame)
        h, w = gray.shape

        # Defocus proxy: multi-resolution high-frequency energy. Computing the
        # residual at several Gaussian scales captures both fine and coarse focus
        # cues; the normalized maps are averaged so no single scale dominates.
        contrast = self._multiscale_contrast(gray)

        # Vertical ground-plane prior: bottom of frame == nearer.
        vprior = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        vprior = np.repeat(vprior, w, axis=1)

        # Center bias: distance from frame center (nearer at center).
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        radial = np.sqrt(xx * xx + yy * yy) / np.sqrt(2.0)
        center = 1.0 - radial

        depth = (
            self.contrast_weight * contrast
            + self.vertical_weight * vprior
            + self.center_weight * center
        )
        depth = normalize01(depth)

        # Edge-aware refinement so depth boundaries track image boundaries.
        radius = max(2, int(round(min(h, w) * 0.02)))
        depth = guided_filter(gray, depth, radius=radius, eps=1e-3)
        return normalize01(depth)

    @staticmethod
    def _multiscale_contrast(gray: np.ndarray) -> np.ndarray:
        """Average normalized high-frequency contrast over three Gaussian scales."""
        maps = []
        for sigma in (5.0, 9.0, 18.0):
            low = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
            high = np.abs(gray - low)
            pooled = cv2.GaussianBlur(high, (0, 0), sigmaX=sigma)
            maps.append(normalize01(pooled))
        return normalize01(np.mean(maps, axis=0).astype(np.float32))
