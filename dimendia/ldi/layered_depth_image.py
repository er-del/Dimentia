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

from dimendia.imageops import feather_mask, matting_refine, normalize01
from dimendia.types import DepthMap, Frame, Mask, TrackedObject

InpaintFn = Callable[[Frame, Mask, DepthMap], Frame]


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

    def __init__(self, foreground_percentile: float = 70.0, feather_radius: int = 5):
        self.foreground_percentile = foreground_percentile
        self.feather_radius = feather_radius

    def build(
        self,
        frame: Frame,
        depth: DepthMap,
        objects: list[TrackedObject],
        primary_id: int | None,
        inpaint_fn: InpaintFn | None = None,
        num_layers: int = 3,
    ) -> LayeredDepthImage:
        h, w = frame.shape[:2]
        num_layers = max(3, int(num_layers))
        primary = next((o for o in objects if o.object_id == primary_id), None)

        primary_mask = primary.mask if primary is not None else np.zeros((h, w), dtype=bool)

        def refined_alpha(mask: Mask) -> np.ndarray:
            coarse = feather_mask(mask, self.feather_radius)
            return matting_refine(coarse, frame, depth)

        layers: list[Layer] = []

        # Layer 0 — primary extrusion object (extraction unchanged).
        layers.append(
            Layer(
                name="projectile",
                index=0,
                color=frame.copy(),
                alpha=refined_alpha(primary_mask),
                depth=depth.copy(),
                object_id=primary_id,
            )
        )

        # Middle layers — quantize the depth range into (num_layers - 2) bands.
        # ``num_layers - 1`` percentile thresholds bound ``num_layers - 2`` bands.
        thresholds = np.sort(
            np.percentile(depth, np.linspace(30, 95, num_layers - 1)).astype(np.float32)
        )
        n_bands = num_layers - 2
        other_objs = np.zeros((h, w), dtype=bool)
        for obj in objects:
            if obj.object_id != primary_id:
                other_objs |= obj.mask

        band_union = np.zeros((h, w), dtype=bool)
        for i in range(n_bands):
            lo = float(thresholds[i])
            if i == n_bands - 1:  # nearest band extends to the very front
                band = depth >= lo
                band |= other_objs  # keep other tracked objects in the front band
            else:
                hi = float(thresholds[i + 1])
                band = (depth >= lo) & (depth < hi)
            band &= ~primary_mask
            band_union |= band
            name = "foreground" if num_layers == 3 else f"band{i}"
            layers.append(
                Layer(
                    name=name,
                    index=i + 1,
                    color=frame.copy(),
                    alpha=refined_alpha(band),
                    depth=depth.copy(),
                )
            )

        # Far layer — background plate (holes behind nearer layers reconstructed).
        hole = primary_mask | band_union
        bg_color = frame.copy()
        bg_depth = depth.copy()
        if hole.any() and inpaint_fn is not None:
            bg_color = inpaint_fn(frame, hole, depth)
            bg_depth = self._inpaint_depth(depth, hole)
        layers.append(
            Layer(
                name="background",
                index=num_layers - 1,
                color=bg_color,
                alpha=np.ones((h, w), dtype=np.float32),
                # Push background away but maintain more natural depth variation.
                depth=normalize01(bg_depth) * 0.4 + 0.05,
            )
        )

        return LayeredDepthImage(layers, (w, h))

    @staticmethod
    def _inpaint_depth(depth: DepthMap, hole: Mask) -> DepthMap:
        import threading

        d8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        mask = (hole.astype(np.uint8)) * 255

        result: list[np.ndarray] = [d8]

        def _target() -> None:
            result[0] = cv2.inpaint(d8, mask, 3, cv2.INPAINT_TELEA)

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=15)
        if t.is_alive():
            return depth
        return result[0].astype(np.float32) / 255.0
