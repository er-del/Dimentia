"""ProPainter backend for video-consistent disocclusion inpainting.

ProPainter (Zhou et al., ICCV 2023) operates on a whole clip rather than a single
frame. To fit DIMENDIA's streaming inpainter interface we keep a bounded temporal
**window** of recent frames/masks and run ProPainter over that window, returning
the fill for the current frame. This trades compute for temporal consistency and
is intended for the offline ``quality`` mode.

Integration is done through ProPainter's official CLI (``inference_propainter.py``)
so we don't depend on an unstable Python API:

* Set ``PROPAINTER_DIR`` to a checkout of https://github.com/sczhou/ProPainter
  (with weights downloaded as per its README).
* Optionally set ``PROPAINTER_WINDOW`` (default 20) and ``PROPAINTER_PYTHON``
  (defaults to the current interpreter).

If the directory/script is missing the constructor raises, so the factory falls
back to the classical inpainter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from dimendia.inpainting.base import Inpainter
from dimendia.logging import get_logger
from dimendia.types import Frame, Mask

log = get_logger(__name__)


class ProPainterAdapter(Inpainter):
    name = "propainter"

    def __init__(self) -> None:
        self.dir = os.environ.get("PROPAINTER_DIR")
        if not self.dir:
            raise RuntimeError("PROPAINTER_DIR not set")
        self.script = Path(self.dir) / "inference_propainter.py"
        if not self.script.exists():
            raise RuntimeError(f"inference_propainter.py not found in {self.dir}")
        self.window = int(os.environ.get("PROPAINTER_WINDOW", "20"))
        self.python = os.environ.get("PROPAINTER_PYTHON", sys.executable)
        self._frames: list[Frame] = []
        self._masks: list[Mask] = []

    def reset(self) -> None:
        self._frames.clear()
        self._masks.clear()

    def inpaint(self, frame: Frame, mask: Mask) -> Frame:
        self._frames.append(frame)
        self._masks.append(mask.astype(bool))
        if len(self._frames) > self.window:
            self._frames.pop(0)
            self._masks.pop(0)
        if not self._masks[-1].any():
            return frame
        filled = self._run_window()
        return filled if filled is not None else frame

    def _run_window(self) -> Frame | None:
        with tempfile.TemporaryDirectory(prefix="dimendia_pp_") as tmp:
            tmp_path = Path(tmp)
            fdir = tmp_path / "frames"
            mdir = tmp_path / "masks"
            odir = tmp_path / "out"
            fdir.mkdir()
            mdir.mkdir()
            for i, (fr, mk) in enumerate(zip(self._frames, self._masks, strict=False)):
                cv2.imwrite(str(fdir / f"{i:05d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mdir / f"{i:05d}.png"), (mk.astype(np.uint8) * 255))
            cmd = [
                self.python,
                str(self.script),
                "--video",
                str(fdir),
                "--mask",
                str(mdir),
                "--output",
                str(odir),
                "--save_frames",
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, cwd=self.dir, timeout=300)
            except subprocess.TimeoutExpired:
                log.warning("ProPainter inference timed out after 300s; using raw frame")
                return None
            except subprocess.CalledProcessError as exc:  # pragma: no cover
                log.warning("ProPainter inference failed (%s); using raw frame", exc.returncode)
                return None
            results = sorted(odir.rglob("*.png")) + sorted(odir.rglob("*.jpg"))
            if not results:
                return None
            out = cv2.imread(str(results[-1]))
            if out is None:
                return None
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
