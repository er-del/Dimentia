"""Weight-free disocclusion inpainting (CPU): Telea fill + temporal propagation.

Per frame, holes are filled with OpenCV's Telea inpainting. To suppress the
flicker/texture-popping that frame-independent inpainting causes, the previous
filled result is flow-warped into the current frame and blended inside the hole.
This is the default inpainter and runs everywhere.
"""

from __future__ import annotations

import cv2
import numpy as np

from dimendia.inpainting.base import Inpainter
from dimendia.types import Frame, Mask


class ClassicalInpainter(Inpainter):
    name = "classical"

    def __init__(self, radius: int = 3, temporal_alpha: float = 0.5, dilate: int = 2):
        self.radius = radius
        self.temporal_alpha = temporal_alpha  # weight on the fresh per-frame fill
        self.dilate = dilate
        self._prev_filled: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_filled = None
        self._prev_gray = None

    def inpaint(self, frame: Frame, mask: Mask) -> Frame:
        h, w = frame.shape[:2]
        hole: np.ndarray = mask.astype(np.uint8)
        if self.dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate * 2 + 1,) * 2)
            hole = cv2.dilate(hole, kernel)
        hole_bool = hole.astype(bool)

        filled = cv2.inpaint(frame, hole * 255, self.radius, cv2.INPAINT_TELEA)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self._prev_filled is not None and self._prev_filled.shape == filled.shape:
            warped = self._warp_prev(gray)
            a = self.temporal_alpha
            blended = a * filled.astype(np.float32) + (1.0 - a) * warped.astype(np.float32)
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            out = frame.copy()
            out[hole_bool] = blended[hole_bool]
        else:
            out = frame.copy()
            out[hole_bool] = filled[hole_bool]

        self._prev_filled = out
        self._prev_gray = gray
        return out

    def _warp_prev(self, cur_gray: np.ndarray) -> np.ndarray:
        assert self._prev_filled is not None and self._prev_gray is not None
        flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
            cur_gray, self._prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = cur_gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        return cv2.remap(
            self._prev_filled, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
