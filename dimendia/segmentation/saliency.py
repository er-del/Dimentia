"""Visual saliency ("attention") via the spectral residual method.

Self-contained NumPy implementation of Hou & Zhang (2007) so it works with
``opencv-python-headless`` (no contrib ``cv2.saliency`` module required).
"""

from __future__ import annotations

import cv2
import numpy as np

from dimendia.imageops import normalize01
from dimendia.types import Frame


def spectral_residual_saliency(frame: Frame, resize: int = 256) -> np.ndarray:
    """Return a ``[0, 1]`` saliency map at the input frame's resolution."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    scale = resize / max(h, w)
    small = cv2.resize(
        gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )

    fft = np.fft.fft2(small)
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    avg = cv2.blur(log_amp, (3, 3))
    spectral_residual = log_amp - avg
    combined = np.exp(spectral_residual + 1j * phase)
    saliency = np.abs(np.fft.ifft2(combined)) ** 2
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=2.5)
    saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
    return normalize01(saliency)
