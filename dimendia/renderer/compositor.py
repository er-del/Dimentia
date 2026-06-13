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


def warp_layer_depth(
    color: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    dx_scale: float,
    dy_scale: float,
    scale: float,
    center: tuple[float, float],
    solid: bool,
    depth_ref: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel depth-based warp: each pixel is displaced by its depth value.

    Near pixels (depth > depth_ref) move one direction, far pixels the other,
    creating genuine parallax. ``dx_scale`` / ``dy_scale`` set the maximum
    displacement in pixels at full depth contrast.
    """
    h, w = alpha.shape

    # Build per-pixel displacement from depth.
    d = depth.astype(np.float32)
    disp = d - depth_ref  # negative for far, positive for near

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Apply scale around center.
    cx, cy = center
    grid_x = cx + (grid_x - cx) / max(scale, 1e-6)
    grid_y = cy + (grid_y - cy) / max(scale, 1e-6)

    # Apply depth-based displacement (inverse mapping: where to sample FROM).
    map_x = grid_x - dx_scale * disp
    map_y = grid_y - dy_scale * disp

    color_border = cv2.BORDER_REPLICATE if solid else cv2.BORDER_CONSTANT
    out_color = cv2.remap(
        color,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=color_border,
        borderValue=(0, 0, 0),
    )
    out_alpha = cv2.remap(
        alpha,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out_color, out_alpha


def _warp_triangle(
    src_color: np.ndarray,
    src_alpha: np.ndarray,
    dst_color: np.ndarray,
    dst_alpha: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
    solid: bool,
) -> None:
    """Affine-warp one source triangle into ``dst_*`` (clipped to image bounds)."""
    h, w = dst_alpha.shape
    r1 = cv2.boundingRect(t_src.astype(np.float32))
    r2 = cv2.boundingRect(t_dst.astype(np.float32))
    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return
    t1 = (t_src - np.array([r1[0], r1[1]], dtype=np.float32)).astype(np.float32)
    t2 = (t_dst - np.array([r2[0], r2[1]], dtype=np.float32)).astype(np.float32)

    sx, sy, sw, sh = r1
    sx0, sy0 = max(sx, 0), max(sy, 0)
    src_patch = src_color[sy0 : sy + sh, sx0 : sx + sw]
    alpha_patch = src_alpha[sy0 : sy + sh, sx0 : sx + sw]
    if src_patch.size == 0:
        return
    # Re-base triangle coords if the source bbox was clipped at the top/left.
    t1 = t1 - np.array([sx0 - sx, sy0 - sy], dtype=np.float32)

    m = cv2.getAffineTransform(t1, t2)
    cborder = cv2.BORDER_REFLECT if solid else cv2.BORDER_CONSTANT
    warped_color = cv2.warpAffine(
        src_patch.astype(np.float32), m, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cborder
    )
    warped_alpha = cv2.warpAffine(
        alpha_patch.astype(np.float32),
        m,
        (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = np.zeros((r2[3], r2[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask, t2.astype(np.int32), 1.0, cv2.LINE_AA)

    rx, ry, rw, rh = r2
    dx0, dy0 = max(rx, 0), max(ry, 0)
    dx1, dy1 = min(rx + rw, w), min(ry + rh, h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    ox, oy = dx0 - rx, dy0 - ry
    sub_color = warped_color[oy : oy + (dy1 - dy0), ox : ox + (dx1 - dx0)]
    sub_alpha = (warped_alpha * mask)[oy : oy + (dy1 - dy0), ox : ox + (dx1 - dx0)]
    sub_mask = mask[oy : oy + (dy1 - dy0), ox : ox + (dx1 - dx0)][..., None]

    dst_color[dy0:dy1, dx0:dx1] = (
        dst_color[dy0:dy1, dx0:dx1] * (1.0 - sub_mask) + sub_color * sub_mask
    )
    region_a = dst_alpha[dy0:dy1, dx0:dx1]
    dst_alpha[dy0:dy1, dx0:dx1] = np.maximum(region_a, sub_alpha)


def warp_layer_mesh(
    color: np.ndarray,
    alpha: np.ndarray,
    depth: np.ndarray,
    dx_scale: float,
    dy_scale: float,
    scale: float,
    center: tuple[float, float],
    solid: bool,
    depth_ref: float = 0.5,
    grid_step: int = 16,
    tear_threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangle-mesh warp: a vertex grid is displaced by depth and rasterized.

    Triangles spanning a depth discontinuity larger than ``tear_threshold`` are
    *torn* (skipped) rather than stretched, so foreground and background separate
    cleanly at occlusion boundaries instead of producing rubber-sheet smearing.
    """
    h, w = alpha.shape
    cx, cy = center
    out_color = np.zeros((h, w, 3), dtype=np.float32)
    out_alpha = np.zeros((h, w), dtype=np.float32)

    xs = list(range(0, w - 1, grid_step)) + [w - 1]
    ys = list(range(0, h - 1, grid_step)) + [h - 1]

    def depth_at(x: int, y: int) -> float:
        return float(depth[min(y, h - 1), min(x, w - 1)])

    def dest(x: int, y: int) -> tuple[float, float]:
        disp = depth_at(x, y) - depth_ref
        return (cx + (x - cx) * scale + dx_scale * disp, cy + (y - cy) * scale + dy_scale * disp)

    color_f = color.astype(np.float32)
    alpha_f = alpha.astype(np.float32)
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            x0, x1 = xs[xi], xs[xi + 1]
            y0, y1 = ys[yi], ys[yi + 1]
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            d = [depth_at(px, py) for px, py in corners]
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                tri = (d[a], d[b], d[c])
                if max(tri) - min(tri) > tear_threshold:
                    continue  # tear at depth discontinuity
                src_tri = np.array([corners[a], corners[b], corners[c]], dtype=np.float32)
                dst_tri = np.array(
                    [dest(*corners[a]), dest(*corners[b]), dest(*corners[c])], dtype=np.float32
                )
                _warp_triangle(color_f, alpha_f, out_color, out_alpha, src_tri, dst_tri, solid)
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
