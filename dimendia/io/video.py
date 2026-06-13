"""Streaming video read/write backed by ffmpeg (via imageio-ffmpeg).

Frames are always surfaced as ``uint8`` RGB arrays. The writer encodes H.264 by
default; :func:`mux_audio` copies the original soundtrack back afterwards since
the raw frame writer is video-only.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio
import numpy as np

from dimendia.logging import get_logger

log = get_logger(__name__)


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int | None  # may be None if the container doesn't report it


def _ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # fall back to the bundled binary from imageio-ffmpeg
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - environment dependent
        return None


class VideoReader:
    """Iterate decoded RGB frames from a video file.

    Parameters
    ----------
    path:
        Path to the input video.
    start_frame:
        Frame index to seek to (0-based). Frames before this are skipped.
    end_frame:
        Exclusive upper bound — iteration stops *before* this frame index.
        ``None`` means read to the end.
    """

    def __init__(
        self,
        path: str | Path,
        start_frame: int = 0,
        end_frame: int | None = None,
    ):
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Input video not found: {self.path}")
        self._reader = iio.get_reader(self.path)
        meta = self._reader.get_meta_data()
        size = meta.get("size", (0, 0))
        n = meta.get("nframes")
        # imageio sometimes reports inf for streamed sources.
        total_frames = int(n) if isinstance(n, (int, float)) and np.isfinite(n) else None
        self.meta = VideoMeta(
            fps=float(meta.get("fps", 30.0)) or 30.0,
            width=int(size[0]),
            height=int(size[1]),
            n_frames=total_frames,
        )

        self._start = max(0, start_frame)
        self._end = end_frame
        if total_frames is not None and self._end is not None:
            self._end = min(self._end, total_frames)
        self._yielded = 0
        self._limit = (self._end - self._start) if self._end is not None else None

    def __iter__(self) -> Iterator[np.ndarray]:
        for i, frame in enumerate(self._reader):  # type: ignore[attr-defined]
            if i < self._start:
                continue
            if self._limit is not None and self._yielded >= self._limit:
                return
            arr = np.asarray(frame)
            if arr.ndim == 2:  # grayscale -> RGB
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 4:  # drop alpha
                arr = arr[..., :3]
            self._yielded += 1
            yield np.ascontiguousarray(arr[..., :3]).astype(np.uint8)

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoWriter:
    """Encode RGB frames to an H.264 mp4 (video only)."""

    def __init__(self, path: str | Path, fps: float, quality: int = 8):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = iio.get_writer(
            self.path,
            fps=max(1.0, float(fps)),
            codec="libx264",
            quality=quality,
            macro_block_size=None,  # allow odd dimensions
            ffmpeg_log_level="error",
            pixelformat="yuv420p",
        )

    def append(self, frame: np.ndarray) -> None:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self._writer.append_data(arr)

    def close(self, timeout: int = 60) -> None:
        import threading

        done = threading.Event()
        excHolder: list[BaseException | None] = [None]

        def _target() -> None:
            try:
                self._writer.close()
            except BaseException as exc:
                excHolder[0] = exc
            finally:
                done.set()

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        if not done.wait(timeout=timeout):
            log.warning("VideoWriter.close() timed out after %ds; forcing", timeout)
            try:
                self._writer._writer.kill()  # type: ignore[attr-defined]
            except Exception:
                pass
            return
        if excHolder[0] is not None:
            raise excHolder[0]

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def mux_audio(source_video: str | Path, video_only: str | Path, output: str | Path) -> bool:
    """Copy the audio track from ``source_video`` onto ``video_only`` -> ``output``.

    Returns ``True`` on success. Silently returns ``False`` (leaving the
    video-only file to be used directly) when ffmpeg is unavailable or the source
    has no audio — audio is a nicety, not a hard requirement.
    """
    exe = _ffmpeg_exe()
    if exe is None:
        log.warning("ffmpeg not found; skipping audio mux")
        return False
    cmd = [
        exe,
        "-y",
        "-i",
        str(video_only),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",  # optional audio
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True
    except subprocess.TimeoutExpired:
        log.warning("audio mux timed out after 120s; using video-only output")
        return False
    except subprocess.CalledProcessError as exc:  # pragma: no cover - ffmpeg dependent
        log.warning("audio mux failed (%s); using video-only output", exc.returncode)
        return False
