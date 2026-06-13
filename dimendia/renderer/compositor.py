"""Layer warping and compositing primitives shared by the renderers.

The renderers use a multiplane-style approximation: each LDI layer is shifted by a
scalar disparity derived from its representative depth (near layers move more),
which keeps every layer hole-free and is cheap enough for CPU. Compositing is the
standard back-to-front "over" operator using each layer's alpha.
"""

from __future__ import annotations

import cv2
import numpy as np

from dimendia.ldi.layered_depth_image import Layer, LayeredDepthImage
from dimendia.types import Frame


def warp_layer(
    color: np.ndarray,
    alpha: np.ndarray,
    dx: float,
    dy: float,
    scale: float,
    center: tuple[float, float],
    solid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate + scale a layer about ``center``.

    ``solid`` layers (a full-frame background) use edge replication so shifting
    doesn't expose black borders; partial layers use transparent borders.
    """
    h, w = alpha.shape
    m = cv2.getRotationMatrix2D(center, 0.0, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    color_border = cv2.BORDER_REPLICATE if solid else cv2.BORDER_CONSTANT
    out_color = cv2.warpAffine(color, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=color_border)
    out_alpha = cv2.warpAffine(
        alpha, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return out_color, out_alpha


def over(base: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Alpha-composite ``color`` over ``base`` (all float32 HxWx3 / HxW)."""
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    return color.astype(np.float32) * a + base * (1.0 - a)


def composite_back_to_front(
    layers: list[tuple[np.ndarray, np.ndarray]], height: int, width: int
) -> np.ndarray:
    """Composite ``(color, alpha)`` pairs given far-to-near; returns float32 HxWx3."""
    out = np.zeros((height, width, 3), dtype=np.float32)
    for color, alpha in layers:
        out = over(out, color.astype(np.float32), alpha)
    return out


def layer_center(alpha: np.ndarray) -> tuple[float, float]:
    """Centroid of a layer's coverage, falling back to the frame center."""
    h, w = alpha.shape
    cover = alpha > 0.5
    if not cover.any():
        return (w / 2.0, h / 2.0)
    ys, xs = np.where(cover)
    return (float(xs.mean()), float(ys.mean()))


def add_matte_bars(img: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Draw cinematic letterbox bars; return ``(image, bar_mask)`` (bar_mask True on bars)."""
    h, w = img.shape[:2]
    bar = int(round(ratio * h))
    bar_mask = np.zeros((h, w), dtype=bool)
    if bar <= 0:
        return img, bar_mask
    out = img.copy()
    out[:bar] = 0
    out[h - bar :] = 0
    bar_mask[:bar] = True
    bar_mask[h - bar :] = True
    return out, bar_mask


def drop_shadow(alpha: np.ndarray, offset: int, blur: int) -> np.ndarray:
    """Soft shadow alpha from a layer's alpha, offset down-right."""
    h, w = alpha.shape
    m = np.array([[1, 0, offset], [0, 1, offset]], dtype=np.float32)
    shifted = cv2.warpAffine(alpha, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    k = max(1, blur) * 2 + 1
    return cv2.GaussianBlur(shifted, (k, k), 0)


def to_uint8(img: np.ndarray) -> Frame:
    return np.clip(img, 0, 255).astype(np.uint8)


def near_layers_first(ldi: LayeredDepthImage) -> list[Layer]:
    """LDI layers ordered nearest-first (already maintained by LayeredDepthImage)."""
    return list(ldi.layers)
