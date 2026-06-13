"""Temporal stabilization of a per-frame depth sequence (Stage A, part 2).

Per-frame monocular depth is normalized independently, so raw sequences exhibit
*scale flicker* and *boundary swimming*. This stabilizer removes both:

1. **Flow warping** — the previous stabilized depth is warped into the current
   frame using optical flow, giving a temporal prior aligned to current content.
2. **Scale alignment** — the current depth is fit (affine: ``a*d + b``) onto that
   warped prior via least squares, locking the depth scale across frames.
3. **EMA blending** — current and prior are blended; motion-aware blending trusts
   the current frame more where flow magnitude is high (prevents ghosting).
"""

from __future__ import annotations

import cv2
import numpy as np

from dimendia.config import TemporalDepthConfig
from dimendia.imageops import normalize01
from dimendia.types import DepthMap, Frame


class TemporalDepthStabilizer:
    def __init__(self, config: TemporalDepthConfig | None = None):
        self.config = config if config is not None else TemporalDepthConfig()  # type: ignore[call-arg]
        self._prev_stable: np.ndarray | None = None
        self._prev_gray_u8: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_stable = None
        self._prev_gray_u8 = None

    def stabilize(self, depth: DepthMap, frame: Frame) -> DepthMap:
        gray_u8 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        depth = depth.astype(np.float32)

        if self._prev_stable is None or self._prev_stable.shape != depth.shape:
            result = depth
        else:
            warped_prev, flow_mag = self._warp_prev(gray_u8)
            aligned = depth
            if self.config.scale_align:
                aligned = self._affine_align(depth, warped_prev)
            alpha = self._alpha_map(depth.shape, flow_mag)
            result = alpha * aligned + (1.0 - alpha) * warped_prev

        result = normalize01(result)
        self._prev_stable = result
        self._prev_gray_u8 = gray_u8
        return result

    # -- internals -----------------------------------------------------------

    def _warp_prev(self, cur_gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Warp the previous stabilized depth into the current frame.

        Uses backward flow (current -> previous) so each current pixel samples its
        source in the previous frame.
        """
        assert self._prev_stable is not None and self._prev_gray_u8 is not None
        flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
            cur_gray_u8, self._prev_gray_u8, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = cur_gray_u8.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(
            self._prev_stable, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
        return warped.astype(np.float32), flow_mag

    @staticmethod
    def _affine_align(cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Least-squares fit ``a*cur + b ~= ref`` to put ``cur`` on ``ref``'s scale."""
        x = cur.reshape(-1)
        y = ref.reshape(-1)
        a_mat = np.stack([x, np.ones_like(x)], axis=1)
        sol, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if not np.isfinite(a) or not np.isfinite(b) or a <= 1e-3:
            return cur
        return a * cur + b

    def _alpha_map(self, shape: tuple[int, ...], flow_mag: np.ndarray) -> np.ndarray:
        """Per-pixel EMA weight on the current frame (higher == trust current)."""
        base = float(self.config.ema_alpha)
        if not self.config.motion_aware:
            return np.full(shape, base, dtype=np.float32)
        thr = max(1e-3, self.config.motion_threshold)
        motion = np.clip(flow_mag / (2.0 * thr), 0.0, 1.0)
        return (base + (1.0 - base) * motion).astype(np.float32)
