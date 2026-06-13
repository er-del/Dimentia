"""Small, dependency-light image utilities shared across pipeline stages."""

from __future__ import annotations

import cv2
import numpy as np


def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Min-max normalize an array to ``[0, 1]`` as float32."""
    x = x.astype(np.float32)
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _align8(n: int) -> int:
    """Round *n* up to the nearest multiple of 8."""
    return max(8, (n + 7) & ~7)


def resize_long_edge(frame: np.ndarray, long_edge: int) -> np.ndarray:
    """Resize so the longer side equals ``long_edge`` (no upscaling).

    Both dimensions are rounded to the nearest multiple of 8 so that models
    like RAFT (which require H/W divisible by 8) work correctly.
    """
    h, w = frame.shape[:2]
    cur = max(h, w)
    if cur <= long_edge:
        return frame
    scale = long_edge / float(cur)
    new_w = _align8(max(1, int(round(w * scale))))
    new_h = _align8(max(1, int(round(h * scale))))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_to(arr: np.ndarray, size_wh: tuple[int, int], align: bool = False) -> np.ndarray:
    """Resize an array to ``(width, height)`` choosing a sensible interpolation."""
    w = _align8(size_wh[0]) if align else size_wh[0]
    h = _align8(size_wh[1]) if align else size_wh[1]
    if arr.shape[1] == w and arr.shape[0] == h:
        return arr
    up = w * h > arr.shape[1] * arr.shape[0]
    interp = cv2.INTER_CUBIC if up else cv2.INTER_AREA
    if arr.dtype == bool:
        out = cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return out.astype(bool)
    return cv2.resize(arr, (w, h), interpolation=interp)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving guided filter (He et al. 2010), single-channel guide.

    Used to snap a coarse depth/alpha estimate to image edges without the opencv
    contrib (``ximgproc``) dependency.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, ddepth=-1, ksize=ksize, normalize=True)

    mean_g = box(guide)
    mean_s = box(src)
    corr_g = box(guide * guide)
    corr_gs = box(guide * src)
    var_g = corr_g - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = box(a)
    mean_b = box(b)
    return mean_a * guide + mean_b


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` depth map (1 == near) to an RGB visualization."""
    d = np.clip(depth, 0.0, 1.0)
    u8 = (d * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)  # BGR
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def to_gray_f32(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 -> single-channel float32 in ``[0, 1]``."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def feather_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Return a soft ``[0, 1]`` alpha from a boolean mask (anti-aliased edges)."""
    m = mask.astype(np.float32)
    if radius <= 0:
        return m
    k = 2 * radius + 1
    return cv2.GaussianBlur(m, (k, k), 0)


def matting_refine(
    alpha: np.ndarray,
    frame: np.ndarray,
    depth: np.ndarray,
    *,
    edge_dilate: int = 5,
    window: int = 15,
) -> np.ndarray:
    """Refine a coarse alpha matte along depth discontinuities.

    A trimap's *uncertain* band is built from dilated depth edges (``cv2.Canny``).
    Inside that band each pixel's alpha is re-estimated from local color
    statistics: the foreground/background colour means within a ``window``x
    ``window`` neighbourhood are compared and alpha is set to the relative colour
    distance. Pixels outside the band keep their coarse alpha.
    """
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    d8 = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
    edges = cv2.Canny(d8, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_dilate * 2 + 1,) * 2)
    uncertain = cv2.dilate(edges, k) > 0
    if not uncertain.any():
        return a

    fg = (a > 0.6).astype(np.float32)
    bg = (a < 0.4).astype(np.float32)

    ksize = (window, window)

    def win_sum(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, ddepth=-1, ksize=ksize, normalize=False)

    rgb = frame.astype(np.float32)
    eps = 1e-3
    fg_count = win_sum(fg) + eps
    bg_count = win_sum(bg) + eps
    mean_fg = np.stack([win_sum(rgb[..., c] * fg) / fg_count for c in range(3)], axis=-1)
    mean_bg = np.stack([win_sum(rgb[..., c] * bg) / bg_count for c in range(3)], axis=-1)

    dist_fg = np.sqrt(((rgb - mean_fg) ** 2).sum(axis=-1)) + eps
    dist_bg = np.sqrt(((rgb - mean_bg) ** 2).sum(axis=-1)) + eps
    refined_band = dist_bg / (dist_fg + dist_bg)

    out = a.copy()
    out[uncertain] = refined_band[uncertain].astype(np.float32)
    k_soft = 2 * (edge_dilate // 2) + 1
    return np.clip(cv2.GaussianBlur(out, (k_soft, k_soft), 0), 0.0, 1.0).astype(np.float32)
