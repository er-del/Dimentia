"""Depth Anything V2 backend (default when GPU + weights are available).

Wraps the Hugging Face ``transformers`` depth-estimation pipeline. Imported
lazily so that the base package stays installable without torch. Depth Anything
outputs a disparity-like map (larger == nearer), matching our convention.
"""

from __future__ import annotations

import numpy as np

from dimendia.depth.base import DepthEstimator, _as_near_is_high
from dimendia.imageops import resize_to
from dimendia.types import DepthMap, Frame

_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


class DepthAnythingV2Estimator(DepthEstimator):
    name = "depth_anything_v2"

    def __init__(self, model_id: str = _DEFAULT_MODEL, device: str | None = None):
        import torch  # noqa: F401  (validates the extra is installed)
        from transformers import pipeline

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_device = 0 if device == "cuda" else -1
        self._pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=torch_device,
        )

    def estimate(self, frame: Frame) -> DepthMap:
        from PIL import Image

        h, w = frame.shape[:2]
        out = self._pipe(Image.fromarray(frame))
        predicted = out["predicted_depth"]  # torch.Tensor, larger == nearer
        depth = predicted.squeeze().detach().cpu().numpy().astype(np.float32)
        if depth.shape != (h, w):
            depth = resize_to(depth, (w, h))
        return _as_near_is_high(depth, near_is_high=True)
