"""End-to-end orchestration of Stages A→E.

Frames are streamed from the input video, processed at a capped *work resolution*
for speed, then the rendered result is scaled back to the source resolution and
encoded. A progress callback is emitted per frame so UIs/CLIs can report status.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dimendia.config import Mode, PipelineConfig, RenderMode
from dimendia.depth import TemporalDepthStabilizer, build_depth_estimator
from dimendia.imageops import colorize_depth, resize_long_edge, resize_to
from dimendia.inpainting import build_inpainter
from dimendia.io import VideoReader, VideoWriter, mux_audio
from dimendia.ldi import LDIBuilder
from dimendia.logging import get_logger
from dimendia.renderer import build_renderer
from dimendia.segmentation import OpticalFlow, ProjectileSelector, build_tracker
from dimendia.types import Frame

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
        self.selector = ProjectileSelector(self.config.weights)
        self.ldi_builder = LDIBuilder()
        self.flow = OpticalFlow(prefer_raft=device != "cpu", device=device)
        self.inpainter = (
            build_inpainter(self.config.inpainting_backend) if self.config.inpaint else None
        )
        self.renderer = build_renderer(self.config)

    def reset(self) -> None:
        self.temporal.reset()
        self.tracker.reset()
        if self.inpainter is not None:
            self.inpainter.reset()

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
            effective_total = meta.n_frames

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
        
        try:
            for i, frame in enumerate(reader):
                frame_accumulator += fps_ratio
                
                # If accumulator < 1.0, we skip this frame (downsampling)
                if frame_accumulator < 1.0:
                    if progress is not None:
                        progress(ProgressEvent("render", i, effective_total))
                    continue
                    
                # Process the frame
                work = resize_long_edge(frame, self.config.work_long_edge)
                out_frame, depth_vis = self._process_frame(work, prev_work, i, (out_w, out_h))
                
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
    ) -> tuple[Frame, Frame | None]:
        depth_raw = self.depth.estimate(work)
        depth = self.temporal.stabilize(depth_raw, work)
        flow = self.flow.flow(prev_work, work) if prev_work is not None else None
        objects = self.tracker.track(work, depth, flow)
        primary_id = self.selector.select(objects, work.shape[:2])

        inpaint_fn = self.inpainter.inpaint if self.inpainter is not None else None
        ldi = self.ldi_builder.build(work, depth, objects, primary_id, inpaint_fn=inpaint_fn)
        rendered = self.renderer.render(ldi, index)

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
