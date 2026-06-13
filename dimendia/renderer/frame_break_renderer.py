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
    warp_layer_depth,
    warp_layer_mesh,
)
from dimendia.types import Frame


class FrameBreakRenderer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sway_period = 90.0  # frames per full left-right cycle
        self._warp = warp_layer_mesh if config.mesh_warp else warp_layer_depth

    def _viewpoint_dx(self, frame_index: int, width: int, extrusion: float) -> float:
        # Reduced amplitude: 0.02 instead of 0.05 for subtler, more natural sway.
        amp = (extrusion / 100.0) * 0.02 * width
        return amp * math.sin(2.0 * math.pi * frame_index / self.sway_period)

    def render(
        self, ldi: LayeredDepthImage, frame_index: int, extrusion: float | None = None
    ) -> Frame:
        ex = self.config.extrusion if extrusion is None else extrusion
        if self.config.render_mode == RenderMode.PARALLAX:
            return self._render_parallax(ldi, frame_index, ex)
        return self._render_popout(ldi, frame_index, ex)

    # -- Mode 2 --------------------------------------------------------------

    def _render_parallax(self, ldi: LayeredDepthImage, frame_index: int, extrusion: float) -> Frame:
        h, w = ldi.height, ldi.width
        dx_view = self._viewpoint_dx(frame_index, w, extrusion)
        center = (w / 2.0, h / 2.0)
        warped: list[tuple[np.ndarray, np.ndarray]] = []
        for layer in ldi.layers:  # near -> far
            # Per-pixel depth warp for genuine parallax separation.
            dx_max = -dx_view * 2.0  # scale factor for depth-based displacement
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
                )
            )
        composed = composite_back_to_front(list(reversed(warped)), h, w)
        return to_uint8(composed)

    # -- Mode 1 --------------------------------------------------------------

    def _render_popout(self, ldi: LayeredDepthImage, frame_index: int, extrusion: float) -> Frame:
        h, w = ldi.height, ldi.width
        dx_view = self._viewpoint_dx(frame_index, w, extrusion)
        center = (w / 2.0, h / 2.0)
        strength = extrusion / 100.0

        background_layers: list[tuple[np.ndarray, np.ndarray]] = []
        primary: tuple[np.ndarray, np.ndarray] | None = None
        for layer in ldi.layers:  # near -> far
            if layer.index == 0:  # primary extrusion object
                obj_center = layer_center(layer.alpha)
                # Reduced scale: 3% max instead of 10%.
                scale = 1.0 + 0.03 * strength
                # Per-pixel depth warp with moderately exaggerated parallax.
                dx_max = -dx_view * 1.3 * 2.0
                primary = self._warp(
                    layer.color,
                    layer.alpha,
                    layer.depth,
                    dx_scale=dx_max,
                    dy_scale=0.0,
                    scale=scale,
                    center=obj_center,
                    solid=False,
                )
            else:
                dx_max = -dx_view * 2.0
                solid = layer.name == "background"
                background_layers.append(
                    self._warp(
                        layer.color,
                        layer.alpha,
                        layer.depth,
                        dx_scale=dx_max,
                        dy_scale=0.0,
                        scale=1.0,
                        center=center,
                        solid=solid,
                    )
                )

        base = composite_back_to_front(list(reversed(background_layers)), h, w)
        base_u8, _ = add_matte_bars(to_uint8(base), self.config.matte_ratio)
        base = base_u8.astype(np.float32)

        if primary is not None:
            p_color, p_alpha = primary
            shadow = drop_shadow(p_alpha, offset=int(4 + 6 * strength), blur=int(3 + 4 * strength))
            # Softer shadow: 0.25 instead of 0.45.
            base = base * (1.0 - 0.25 * shadow[..., None])
            base = over(base, p_color.astype(np.float32), p_alpha)

        return to_uint8(base)
