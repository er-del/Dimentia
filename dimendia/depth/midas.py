"""MiDaS depth backend (fallback model when Depth Anything is unavailable).

Loaded from torch hub (``intel-isl/MiDaS``). MiDaS predicts inverse depth, so
larger values are nearer — consistent with our convention.
"""

from __future__ import annotations

import numpy as np

from dimendia.depth.base import DepthEstimator, _as_near_is_high
from dimendia.types import DepthMap, Frame


class MiDaSEstimator(DepthEstimator):
    name = "midas"

    def __init__(self, model_type: str = "DPT_Hybrid", device: str | None = None):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self._model = torch.hub.load("intel-isl/MiDaS", model_type)
        self._model.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "DPT" in model_type:
            self._transform = transforms.dpt_transform
        else:
            self._transform = transforms.small_transform

    def estimate(self, frame: Frame) -> DepthMap:
        torch = self._torch
        h, w = frame.shape[:2]
        batch = self._transform(frame).to(self.device)
        with torch.no_grad():
            pred = self._model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze()
        depth = pred.detach().cpu().numpy().astype(np.float32)
        return _as_near_is_high(depth, near_is_high=True)
