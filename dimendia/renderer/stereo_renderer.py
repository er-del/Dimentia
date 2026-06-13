"""Stereoscopic + VR renderers (Modes 3 & 4) via depth-image-based rendering.

Two virtual eyes are synthesized by shifting each LDI layer by a disparity
proportional to ``(layer_depth - screen_plane)``: content nearer than the screen
plane gets crossed disparity (pops out), farther content gets uncrossed disparity
(sinks in). Layer-granular shifts keep each eye hole-free.

* ``STEREO`` → full side-by-side (Left | Right), each at source resolution.
* ``VR`` → half-width side-by-side (each eye squeezed to W/2), the format VR
  players expect.
"""

from __future__ import annotations

import cv2
import numpy as np

from dimendia.config import PipelineConfig, RenderMode
from dimendia.ldi.layered_depth_image import LayeredDepthImage
from dimendia.renderer.compositor import (
    composite_back_to_front,
    to_uint8,
    warp_layer_depth,
    warp_layer_mesh,
)
from dimendia.types import Frame

_SCREEN_PLANE = 0.5  # depth that renders exactly at the display surface


class StereoRenderer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._warp = warp_layer_mesh if config.mesh_warp else warp_layer_depth

    def render(
        self, ldi: LayeredDepthImage, frame_index: int, extrusion: float | None = None
    ) -> Frame:
        ex = self.config.extrusion if extrusion is None else extrusion
        left = self._render_eye(ldi, +1.0, ex)
        right = self._render_eye(ldi, -1.0, ex)
        if self.config.render_mode == RenderMode.ANAGLYPH:
            return make_anaglyph(left, right)
        if self.config.render_mode == RenderMode.VR:
            return self._side_by_side(left, right, half_width=True)
        return self._side_by_side(left, right, half_width=False)

    def _render_eye(self, ldi: LayeredDepthImage, sign: float, extrusion: float) -> Frame:
        h, w = ldi.height, ldi.width
        baseline_px = self.config.stereo_baseline * w * (extrusion / 100.0)
        center = (w / 2.0, h / 2.0)
        warped: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in ldi.layers:  # near -> far
            # Per-pixel depth warp for genuine stereo disparity.
            dx_max = sign * baseline_px * 0.5
            solid = layer.name == "background"
            warped.append(
                self._warp(
                    layer.color,
                    layer.alpha,
                    layer.depth,
                    dx_scale=dx_max,
                    dy_scale=0.0,
                    scale=1.0,
                    center=center,
                    solid=solid,
                    depth_ref=_SCREEN_PLANE,
                )
            )
        composed = composite_back_to_front(list(reversed(warped)), h, w)
        return to_uint8(composed)

    @staticmethod
    def _side_by_side(left: Frame, right: Frame, half_width: bool) -> Frame:
        h, w = left.shape[:2]
        if half_width:
            lw = cv2.resize(left, (w // 2, h), interpolation=cv2.INTER_AREA)
            rw = cv2.resize(right, (w // 2, h), interpolation=cv2.INTER_AREA)
            return np.concatenate([lw, rw], axis=1)
        return np.concatenate([left, right], axis=1)


def make_anaglyph(left: Frame, right: Frame) -> Frame:
    """Red-cyan anaglyph: red from left eye, green+blue from right eye."""
    out = right.copy()
    out[..., 0] = left[..., 0]
    return out
