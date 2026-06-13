"""Weight-free disocclusion inpainting (CPU): Telea fill + temporal propagation.

Per frame, holes are filled with OpenCV's Telea inpainting. To suppress the
flicker/texture-popping that frame-independent inpainting causes, the previous
filled result is flow-warped into the current frame and blended inside the hole.
This is the default inpainter and runs everywhere.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from dimendia.inpainting.base import Inpainter
from dimendia.logging import get_logger
from dimendia.segmentation.flow import compute_occlusion_mask
from dimendia.types import DepthMap, Frame, Mask

log = get_logger(__name__)

_INPAINT_TIMEOUT = 30  # seconds


def _inpaint_telea(frame: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Run cv2.inpaint with a timeout so large holes don't hang the pipeline."""
    result: list[np.ndarray] = [frame]
    exc: list[BaseException | None] = [None]

    def _target() -> None:
        try:
            result[0] = cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=_INPAINT_TIMEOUT)
    if t.is_alive():
        log.warning(
            "cv2.inpaint timed out after %ds (hole ratio %.0f%%); using original frame",
            _INPAINT_TIMEOUT,
            (mask > 0).sum() / mask.size * 100,
        )
        return frame
    if exc[0] is not None:
        raise exc[0]  # type: ignore[misc]
    return result[0]


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

    def inpaint(self, frame: Frame, mask: Mask, depth: DepthMap | None = None) -> Frame:
        h, w = frame.shape[:2]
        hole: np.ndarray = mask.astype(np.uint8)
        if self.dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate * 2 + 1,) * 2)
            hole = cv2.dilate(hole, kernel)
        hole_bool = hole.astype(bool)

        # Depth-context: exclude foreground pixels from the Telea source so only
        # background-colored pixels diffuse into the holes.
        fill_mask = hole
        if depth is not None and depth.shape[:2] == (h, w):
            fg = depth > float(np.percentile(depth, 60))
            fill_mask = (hole_bool | fg).astype(np.uint8)

        filled = _inpaint_telea(frame, fill_mask * 255, 5)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self._prev_filled is not None and self._prev_filled.shape == filled.shape:
            warped, occluded = self._warp_prev(gray)
            a = self.temporal_alpha
            blended = a * filled.astype(np.float32) + (1.0 - a) * warped.astype(np.float32)
            # Drop the warped contribution where the previous frame had no valid
            # correspondence (newly disoccluded pixels).
            blended[occluded] = filled.astype(np.float32)[occluded]
            filled = np.clip(blended, 0, 255).astype(np.uint8)

        out = frame.copy()

        # Color-aware boundary blending (feathering) for seamless transitions.
        # Create a soft alpha matte from the dilated hole mask.
        alpha_hole: np.ndarray = hole.astype(np.float32)
        alpha_hole = cv2.GaussianBlur(alpha_hole, (7, 7), 0)
        alpha_hole = np.clip(alpha_hole, 0.0, 1.0)[..., None]

        # Blend the filled content over the original frame using the soft matte.
        out = filled.astype(np.float32) * alpha_hole + frame.astype(np.float32) * (1.0 - alpha_hole)
        out = np.clip(out, 0, 255).astype(np.uint8)

        self._prev_filled = out
        self._prev_gray = gray
        return out

    def _warp_prev(self, cur_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._prev_filled is not None and self._prev_gray is not None
        flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
            cur_gray, self._prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = cur_gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(
            self._prev_filled, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        back = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
            self._prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        occluded = compute_occlusion_mask(flow.astype(np.float32), back.astype(np.float32))
        return warped, occluded
