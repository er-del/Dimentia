"""End-to-end orchestration of Stages A→E.

Frames are streamed from the input video, processed at a capped *work resolution*
for speed, then the rendered result is scaled back to the source resolution and
encoded. A progress callback is emitted per frame so UIs/CLIs can report status.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dimendia.config import Mode, PipelineConfig, RenderMode
from dimendia.depth import TemporalDepthStabilizer, build_depth_estimator
from dimendia.imageops import colorize_depth, normalize01, resize_long_edge, resize_to
from dimendia.inpainting import build_inpainter
from dimendia.io import VideoReader, VideoWriter, mux_audio
from dimendia.ldi import LDIBuilder
from dimendia.logging import get_logger
from dimendia.renderer import build_renderer
from dimendia.segmentation import OpticalFlow, ProjectileSelector, build_tracker
from dimendia.types import DepthMap, Frame

log = get_logger(__name__)


@dataclass
class ProgressEvent:
    stage: str
    frame_index: int
    total: int | None
    message: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class ConversionResult:
    output_path: str
    depth_preview_path: str | None
    frames: int
    fps: float
    width: int
    height: int


class DimendiaPipeline:
    def __init__(self, config: PipelineConfig | None = None, *, device: str | None = None):
        self.config = config or PipelineConfig()
        self.device = device
        self.depth = build_depth_estimator(self.config.depth_backend, device=device)
        self.temporal = TemporalDepthStabilizer(self.config.temporal)
        self.tracker = build_tracker(self.config, device=device)
        self.selector = ProjectileSelector(
            self.config.weights,
            semantic_backend=self.config.semantic_backend,
            config=self.config,
        )
        self.ldi_builder = LDIBuilder()
        self.flow = OpticalFlow(prefer_raft=device != "cpu", device=device)
        self.inpainter = (
            build_inpainter(self.config.inpainting_backend) if self.config.inpaint else None
        )
        self.renderer = build_renderer(self.config)
        self._prev_hist: np.ndarray | None = None

    def reset(self) -> None:
        self.temporal.reset()
        self.tracker.reset()
        if self.inpainter is not None:
            self.inpainter.reset()
        self._prev_hist = None

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        progress: ProgressCallback | None = None,
        depth_preview_path: str | Path | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> ConversionResult:
        self.reset()
        reader = VideoReader(input_path, start_frame=start_frame, end_frame=end_frame)
        meta = reader.meta
        out_w, out_h = meta.width, meta.height

        target_fps = float(self.config.output_fps) if self.config.output_fps else meta.fps

        log.info(
            "input %sx%s @ %.2ffps (%s frames) mode=%s render=%s -> output %.2ffps",
            out_w,
            out_h,
            meta.fps,
            meta.n_frames,
            self.config.mode.value,
            self.config.render_mode.value,
            target_fps,
        )

        # Compute effective frame count for progress reporting.
        if start_frame > 0 or end_frame is not None:
            effective_total = (end_frame or meta.n_frames or 0) - start_frame
            if effective_total < 0:
                effective_total = 0
        else:
            effective_total = meta.n_frames or 0

        if self.config.write_depth_preview and depth_preview_path is None:
            depth_preview_path = str(
                Path(output_path).with_name(Path(output_path).stem + "_depth.mp4")
            )

        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_video.close()
        writer = VideoWriter(tmp_video.name, target_fps)
        depth_writer: VideoWriter | None = None
        if depth_preview_path is not None:
            depth_writer = VideoWriter(depth_preview_path, target_fps)

        prev_work: Frame | None = None
        count = 0

        # Keep track of fractional frame accumulation to handle arbitrary FPS conversions.
        frame_accumulator = 0.0
        fps_ratio = target_fps / meta.fps

        # Bidirectional depth requires a look-ahead pass, so materialize the clip
        # and precompute a forward+backward stabilized depth per frame up front.
        merged_depths: list[DepthMap] | None = None
        frame_source: Iterator[tuple[int, Frame]]
        if self.config.bidirectional_depth:
            materialized = list(reader)
            works_all = [resize_long_edge(f, self.config.work_long_edge) for f in materialized]
            merged_depths = self._bidirectional_depths(works_all)
            frame_source = enumerate(materialized)
        else:
            frame_source = enumerate(reader)

        try:
            for i, frame in frame_source:
                frame_accumulator += fps_ratio

                # If accumulator < 1.0, we skip this frame (downsampling)
                if frame_accumulator < 1.0:
                    if progress is not None:
                        progress(ProgressEvent("render", i, effective_total))
                    continue

                # Process the frame
                work = resize_long_edge(frame, self.config.work_long_edge)
                depth_override = merged_depths[i] if merged_depths is not None else None
                out_frame, depth_vis = self._process_frame(
                    work, prev_work, i, (out_w, out_h), depth_override=depth_override
                )

                # Write frame one or more times (duplication for upsampling)
                while frame_accumulator >= 1.0:
                    writer.append(out_frame)
                    if depth_writer is not None and depth_vis is not None:
                        depth_writer.append(depth_vis)
                    count += 1
                    frame_accumulator -= 1.0

                prev_work = work
                if progress is not None:
                    progress(ProgressEvent("render", i, effective_total))
        finally:
            log.info("frame loop finished; closing writer (%d frames written)", count)
            writer.close(timeout=120)
            if depth_writer is not None:
                depth_writer.close(timeout=120)
            reader.close()

        import shutil

        log.info("muxing audio into final output...")
        if mux_audio(input_path, tmp_video.name, output_path):
            Path(tmp_video.name).unlink(missing_ok=True)
        else:
            try:
                shutil.move(tmp_video.name, output_path)
            except Exception as e:
                log.error("Failed to move temporary video to output path: %s", e)
                shutil.copy2(tmp_video.name, output_path)
                Path(tmp_video.name).unlink(missing_ok=True)

        log.info("wrote %s (%s frames at %.2ffps)", output_path, count, target_fps)
        return ConversionResult(
            output_path=str(output_path),
            depth_preview_path=str(depth_preview_path) if depth_preview_path else None,
            frames=count,
            fps=target_fps,
            width=out_w,
            height=out_h,
        )

    def _process_frame(
        self,
        work: Frame,
        prev_work: Frame | None,
        index: int,
        out_size: tuple[int, int],
        depth_override: DepthMap | None = None,
    ) -> tuple[Frame, Frame | None]:
        # Hard scene cuts invalidate all temporal state; reset before processing.
        self._detect_scene_cut(work)

        if depth_override is not None:
            depth = depth_override
        else:
            depth_raw = self.depth.estimate(work)
            depth = self.temporal.stabilize(depth_raw, work)
        flow = self.flow.flow(prev_work, work) if prev_work is not None else None
        objects = self.tracker.track(work, depth, flow)
        primary_id = self.selector.select(objects, work.shape[:2], work)

        effective_extrusion = self._effective_extrusion(depth)

        inpaint_fn = self.inpainter.inpaint if self.inpainter is not None else None
        ldi = self.ldi_builder.build(
            work,
            depth,
            objects,
            primary_id,
            inpaint_fn=inpaint_fn,
            num_layers=self.config.num_layers,
        )
        rendered = self.renderer.render(ldi, index, extrusion=effective_extrusion)

        # Scale rendered output to source resolution, preserving stereo aspect.
        out_w, out_h = out_size
        scale = out_h / rendered.shape[0]
        target_w = int(round(rendered.shape[1] * scale))
        target_h = out_h
        # H.264 requires even dimensions; round down to nearest even number.
        if target_w % 2 != 0:
            target_w -= 1
        if target_h % 2 != 0:
            target_h -= 1
        target = (target_w, target_h)
        out_frame = resize_to(rendered, target)

        depth_vis: Frame | None = None
        if self.config.write_depth_preview:
            depth_vis = resize_to(colorize_depth(depth), (out_w, out_h))
        return out_frame, depth_vis

    def _detect_scene_cut(self, work: Frame) -> None:
        """Reset temporal state when the HSV histogram changes abruptly."""
        hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if self._prev_hist is not None:
            dist = cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if dist > self.config.scene_cut_threshold:
                log.info("scene cut detected (hist dist %.3f); resetting temporal state", dist)
                self.temporal.reset()
                self.tracker.reset()
                if self.inpainter is not None:
                    self.inpainter.reset()
        self._prev_hist = hist

    def _effective_extrusion(self, depth: DepthMap) -> float:
        """Scale extrusion by the frame's depth range (clamped to [0.3x, 1.5x])."""
        if not self.config.adaptive_extrusion:
            return self.config.extrusion
        depth_range = float(np.percentile(depth, 95) - np.percentile(depth, 5))
        scaled = self.config.extrusion * depth_range / 0.7
        lo = 0.3 * self.config.extrusion
        hi = 1.5 * self.config.extrusion
        return float(np.clip(scaled, lo, hi))

    def _bidirectional_depths(self, works: list[Frame]) -> list[DepthMap]:
        """Forward + backward stabilization of the whole clip, blended 0.5/0.5."""
        raw = [self.depth.estimate(w) for w in works]
        n = len(works)

        self.temporal.reset()
        forward = [self.temporal.stabilize(raw[i], works[i]) for i in range(n)]

        self.temporal.reset()
        backward: list[DepthMap] = [forward[i] for i in range(n)]
        for i in range(n - 1, -1, -1):
            backward[i] = self.temporal.stabilize(raw[i], works[i])

        self.temporal.reset()
        return [normalize01(0.5 * forward[i] + 0.5 * backward[i]) for i in range(n)]


def convert_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mode: Mode | str = Mode.BALANCED,
    render_mode: RenderMode | str = RenderMode.POPOUT,
    extrusion: float = 100.0,
    depth_preview: bool = False,
    device: str | None = None,
    progress: ProgressCallback | None = None,
    **overrides: object,
) -> ConversionResult:
    """Convenience wrapper: build a config from a mode preset and run the pipeline."""
    mode = Mode(mode)
    render_mode = RenderMode(render_mode)
    config = PipelineConfig.from_mode(
        mode,
        render_mode=render_mode,
        extrusion=extrusion,
        write_depth_preview=depth_preview,
        **overrides,
    )
    pipeline = DimendiaPipeline(config, device=device)
    return pipeline.convert(input_path, output_path, progress=progress)
