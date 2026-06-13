"""Optical flow with a learned backend (RAFT) and a classical fallback (Farneback).

Returns dense **forward** flow (previous -> current) as a ``(H, W, 2)`` float32
array of ``(dx, dy)`` displacements. RAFT is used when torchvision and its
weights are available; otherwise OpenCV Farneback runs on CPU.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dimendia.logging import get_logger
from dimendia.types import Frame

log = get_logger(__name__)


def _farneback(prev_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray:
    """Thin wrapper so the ``None`` flow-init is isolated behind a type: ignore."""
    return cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
        prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )


class OpticalFlow:
    def __init__(self, prefer_raft: bool = True, device: str | None = None):
        self.backend = "farneback"
        self._raft: Any = None
        self._transforms: Any = None
        self._torch: Any = None
        self.device = device or "cpu"
        if prefer_raft:
            self._try_init_raft(device)

    def _try_init_raft(self, device: str | None) -> None:
        try:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            if dev == "cpu":
                raise RuntimeError("CPU device detected; falling back to Farneback for performance")
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=False).to(dev).eval()
            self._raft = model
            self._transforms = weights.transforms()
            self._torch = torch
            self.device = dev
            self.backend = "raft"
            log.info("optical flow backend: RAFT (%s)", dev)
        except Exception as exc:  # noqa: BLE001 - graceful fallback
            log.info("optical flow backend: Farneback (RAFT unavailable: %s)", exc)

    def flow(self, prev: Frame, cur: Frame) -> np.ndarray:
        if self.backend == "raft":
            return self._raft_flow(prev, cur)
        return self._farneback_flow(prev, cur)

    @staticmethod
    def _farneback_flow(prev: Frame, cur: Frame) -> np.ndarray:
        pg = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        cg = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
        return _farneback(pg, cg).astype(np.float32)

    def _raft_flow(self, prev: Frame, cur: Frame) -> np.ndarray:
        torch = self._torch
        h, w = prev.shape[:2]

        def to_tensor(img: Frame) -> Any:
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0)

        b1, b2 = self._transforms(to_tensor(prev), to_tensor(cur))
        with torch.no_grad():
            flows = self._raft(b1.to(self.device), b2.to(self.device))
        flow = flows[-1][0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        if flow.shape[:2] != (h, w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        return flow


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
