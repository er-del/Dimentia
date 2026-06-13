"""Layered Depth Image: a front-to-back stack of RGBA + depth layers.

DIMENDIA models each frame as three persistent layers (extensible to N):

* **Layer 0** — the primary extrusion object (projectile).
* **Layer 1** — remaining foreground (other near content).
* **Layer 2** — the background plate (holes behind nearer layers inpainted).

Layers are ordered nearest-first. The renderer parallax-shifts and composites them
back-to-front, which is what produces occlusion-correct pop-out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from dimendia.imageops import feather_mask, normalize01
from dimendia.types import DepthMap, Frame, Mask, TrackedObject

InpaintFn = Callable[[Frame, Mask], Frame]


@dataclass
class Layer:
    name: str
    index: int  # 0 == nearest
    color: np.ndarray  # (H, W, 3) uint8 RGB, full-frame
    alpha: np.ndarray  # (H, W) float32 in [0, 1]
    depth: np.ndarray  # (H, W) float32 in [0, 1], 1 == near
    object_id: int | None = None

    @property
    def mean_depth(self) -> float:
        cover = self.alpha > 0.05
        if not cover.any():
            return float(self.depth.mean())
        return float(self.depth[cover].mean())


class LayeredDepthImage:
    def __init__(self, layers: list[Layer], size_wh: tuple[int, int]):
        # Keep nearest-first ordering by representative depth (1 == near).
        self.layers = sorted(layers, key=lambda layer: layer.mean_depth, reverse=True)
        for i, layer in enumerate(self.layers):
            layer.index = i
        self.width, self.height = size_wh

    def flatten(self) -> Frame:
        """Composite all layers back-to-front (debug / preview)."""
        out = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for layer in reversed(self.layers):  # far -> near
            a = layer.alpha[..., None]
            out = layer.color.astype(np.float32) * a + out * (1.0 - a)
        return np.clip(out, 0, 255).astype(np.uint8)


class LDIBuilder:
    """Assemble a :class:`LayeredDepthImage` from a frame, depth, and tracked objects."""

    def __init__(self, foreground_percentile: float = 70.0, feather_radius: int = 3):
        self.foreground_percentile = foreground_percentile
        self.feather_radius = feather_radius

    def build(
        self,
        frame: Frame,
        depth: DepthMap,
        objects: list[TrackedObject],
        primary_id: int | None,
        inpaint_fn: InpaintFn | None = None,
    ) -> LayeredDepthImage:
        h, w = frame.shape[:2]
        primary = next((o for o in objects if o.object_id == primary_id), None)

        primary_mask = primary.mask if primary is not None else np.zeros((h, w), dtype=bool)

        # Foreground = other tracked objects plus generically near content.
        fg_mask = np.zeros((h, w), dtype=bool)
        for obj in objects:
            if obj.object_id != primary_id:
                fg_mask |= obj.mask
        thresh = float(np.percentile(depth, self.foreground_percentile))
        fg_mask |= depth >= thresh
        fg_mask &= ~primary_mask

        layers: list[Layer] = []

        # Layer 0 — primary extrusion object.
        layers.append(
            Layer(
                name="projectile",
                index=0,
                color=frame.copy(),
                alpha=feather_mask(primary_mask, self.feather_radius),
                depth=depth.copy(),
                object_id=primary_id,
            )
        )

        # Layer 1 — remaining foreground.
        layers.append(
            Layer(
                name="foreground",
                index=1,
                color=frame.copy(),
                alpha=feather_mask(fg_mask, self.feather_radius),
                depth=depth.copy(),
            )
        )

        # Layer 2 — background plate (holes behind nearer layers reconstructed).
        hole = primary_mask | fg_mask
        bg_color = frame.copy()
        bg_depth = depth.copy()
        if hole.any() and inpaint_fn is not None:
            bg_color = inpaint_fn(frame, hole)
            bg_depth = self._inpaint_depth(depth, hole)
        layers.append(
            Layer(
                name="background",
                index=2,
                color=bg_color,
                alpha=np.ones((h, w), dtype=np.float32),
                depth=normalize01(bg_depth) * 0.5,  # push background away
            )
        )

        return LayeredDepthImage(layers, (w, h))

    @staticmethod
    def _inpaint_depth(depth: DepthMap, hole: Mask) -> DepthMap:
        d8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        mask = (hole.astype(np.uint8)) * 255
        filled = cv2.inpaint(d8, mask, 3, cv2.INPAINT_TELEA)
        return filled.astype(np.float32) / 255.0
