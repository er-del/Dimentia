"""Pop-out cinematic + depth-parallax renderers (Modes 1 & 2).

**Pop-out cinematic** is the headline effect. A virtual screen plane is implied by
cinematic matte bars; background/foreground layers are clipped to the inner frame,
while the primary object is scaled up, given exaggerated parallax, and composited
*after* (on top of) the matte bars so it appears to break out of the frame toward
the viewer. A soft contact shadow on the bars sells the depth.

**Depth parallax** applies a gentle, depth-scaled viewpoint sway to all layers
equally — the "wiggle 3D" look — without matte bars or extrusion.
"""

from __future__ import annotations

import math

import numpy as np

from dimendia.config import PipelineConfig, RenderMode
from dimendia.ldi.layered_depth_image import LayeredDepthImage
from dimendia.renderer.compositor import (
    add_matte_bars,
    composite_back_to_front,
    drop_shadow,
    layer_center,
    over,
    to_uint8,
    warp_layer,
)
from dimendia.types import Frame


class FrameBreakRenderer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sway_period = 90.0  # frames per full left-right cycle

    def _viewpoint_dx(self, frame_index: int, width: int) -> float:
        amp = (self.config.extrusion / 100.0) * 0.05 * width
        return amp * math.sin(2.0 * math.pi * frame_index / self.sway_period)

    def render(self, ldi: LayeredDepthImage, frame_index: int) -> Frame:
        if self.config.render_mode == RenderMode.PARALLAX:
            return self._render_parallax(ldi, frame_index)
        return self._render_popout(ldi, frame_index)

    # -- Mode 2 --------------------------------------------------------------

    def _render_parallax(self, ldi: LayeredDepthImage, frame_index: int) -> Frame:
        h, w = ldi.height, ldi.width
        dx_view = self._viewpoint_dx(frame_index, w)
        center = (w / 2.0, h / 2.0)
        warped: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in ldi.layers:  # near -> far
            sx = -dx_view * layer.mean_depth
            solid = layer.name == "background"
            warped.append(warp_layer(layer.color, layer.alpha, sx, 0.0, 1.0, center, solid=solid))
        composed = composite_back_to_front(list(reversed(warped)), h, w)
        return to_uint8(composed)

    # -- Mode 1 --------------------------------------------------------------

    def _render_popout(self, ldi: LayeredDepthImage, frame_index: int) -> Frame:
        h, w = ldi.height, ldi.width
        dx_view = self._viewpoint_dx(frame_index, w)
        center = (w / 2.0, h / 2.0)
        strength = self.config.extrusion / 100.0

        background_layers: list[tuple[np.ndarray, np.ndarray]] = []
        primary: tuple[np.ndarray, np.ndarray] | None = None
        for layer in ldi.layers:  # near -> far
            if layer.index == 0:  # primary extrusion object
                obj_center = layer_center(layer.alpha)
                scale = 1.0 + 0.10 * strength
                sx = -dx_view * layer.mean_depth * 1.8  # exaggerated parallax
                primary = warp_layer(
                    layer.color, layer.alpha, sx, 0.0, scale, obj_center, solid=False
                )
            else:
                sx = -dx_view * layer.mean_depth
                solid = layer.name == "background"
                background_layers.append(
                    warp_layer(layer.color, layer.alpha, sx, 0.0, 1.0, center, solid=solid)
                )

        base = composite_back_to_front(list(reversed(background_layers)), h, w)
        base_u8, _ = add_matte_bars(to_uint8(base), self.config.matte_ratio)
        base = base_u8.astype(np.float32)

        if primary is not None:
            p_color, p_alpha = primary
            shadow = drop_shadow(p_alpha, offset=int(6 + 10 * strength), blur=int(4 + 6 * strength))
            base = base * (1.0 - 0.45 * shadow[..., None])  # contact shadow on bars
            base = over(base, p_color.astype(np.float32), p_alpha)

        return to_uint8(base)
