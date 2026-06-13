"""Generate a tiny synthetic clip — a moving/approaching ball over a textured
background. Useful for the CLI self-test and quick local verification without any
sample media.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dimendia.io import VideoWriter


def make_synthetic_clip(
    path: str | Path,
    n_frames: int = 24,
    width: int = 320,
    height: int = 240,
    fps: float = 24.0,
    seed: int = 0,
) -> str:
    """Write a synthetic mp4 with a foreground ball crossing the frame."""
    rng = np.random.default_rng(seed)

    # Static textured background (gradient + low-frequency noise).
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    grad = (xx / width * 120 + yy / height * 80).astype(np.float32)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 8)
    base = np.clip(grad + 40 + 25 * noise, 0, 255)
    background = np.stack([base, base * 0.8 + 20, base * 0.6 + 40], axis=-1).astype(np.uint8)

    writer = VideoWriter(path, fps)
    try:
        for i in range(n_frames):
            frame = background.copy()
            t = i / max(1, n_frames - 1)
            cx = int(width * (0.1 + 0.8 * t))
            cy = int(height * (0.6 - 0.2 * np.sin(np.pi * t)))
            radius = int(min(width, height) * (0.08 + 0.10 * t))  # grows == approaching
            cv2.circle(frame, (cx, cy), radius, (40, 220, 255), thickness=-1)
            cv2.circle(frame, (cx, cy), radius, (10, 120, 160), thickness=2)
            writer.append(frame)
    finally:
        writer.close()
    return str(path)
