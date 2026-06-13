"""``dimendia`` command-line entrypoint.

Commands:

* ``dimendia convert INPUT OUTPUT`` — run the full 2D→3D pipeline.
* ``dimendia selftest`` — generate a synthetic clip and convert it end-to-end
  (used by CI as a smoke test; needs no GPU or model downloads).
* ``dimendia info`` — print the resolved backends for the current environment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import click
from tqdm import tqdm

from dimendia.config import (
    DepthBackend,
    InpaintingBackend,
    Mode,
    PipelineConfig,
    RenderMode,
    SegmentationBackend,
)
from dimendia.pipeline import DimendiaPipeline, ProgressEvent, convert_video


def _parse_timestamp(value: str) -> float:
    """Parse a timestamp string into seconds.

    Accepted formats: ``HH:MM:SS``, ``MM:SS``, ``SS``, or a raw float.
    """
    parts = value.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 1:
        return float(parts[0])
    raise click.BadParameter(f"Invalid timestamp format: {value!r}")


@click.group()
@click.version_option(package_name="dimendia")
def main() -> None:
    """DIMENDIA — turn ordinary video into cinematic pop-out 3D."""


def _resolve_render_mode(render_mode: str, stereo: bool, vr: bool, anaglyph: bool) -> RenderMode:
    if anaglyph:
        return RenderMode.ANAGLYPH
    if vr:
        return RenderMode.VR
    if stereo:
        return RenderMode.STEREO
    return RenderMode(render_mode)


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice([m.value for m in Mode]),
    default=Mode.BALANCED.value,
    show_default=True,
    help="Speed/quality preset.",
)
@click.option(
    "--render-mode",
    type=click.Choice([r.value for r in RenderMode]),
    default=RenderMode.POPOUT.value,
    show_default=True,
    help="Rendering style (overridden by --stereo/--vr).",
)
@click.option(
    "--extrusion", type=float, default=100.0, show_default=True, help="Pop-out strength (0-200)."
)
@click.option("--stereo", is_flag=True, help="Side-by-side stereoscopic output.")
@click.option("--vr", is_flag=True, help="Half-width SBS VR output.")
@click.option("--anaglyph", is_flag=True, help="Red-cyan anaglyph output.")
@click.option("--depth-preview", is_flag=True, help="Also write a depth visualization video.")
@click.option("--device", type=str, default=None, help="Force compute device (e.g. cpu, cuda).")
@click.option(
    "--depth-backend",
    type=click.Choice([b.value for b in DepthBackend]),
    default=DepthBackend.AUTO.value,
    show_default=True,
)
@click.option(
    "--segmentation-backend",
    type=click.Choice([b.value for b in SegmentationBackend]),
    default=SegmentationBackend.AUTO.value,
    show_default=True,
)
@click.option(
    "--inpainting-backend",
    type=click.Choice([b.value for b in InpaintingBackend]),
    default=InpaintingBackend.AUTO.value,
    show_default=True,
)
@click.option(
    "--start",
    type=str,
    default=None,
    help="Start timestamp (HH:MM:SS, MM:SS, or seconds). Only process from this point.",
)
@click.option(
    "--end",
    type=str,
    default=None,
    help="End timestamp (HH:MM:SS, MM:SS, or seconds). Stop processing at this point.",
)
@click.option(
    "--matte-ratio",
    type=float,
    default=0.12,
    show_default=True,
    help="Cinematic matte bar thickness as fraction of frame height.",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Target output frames per second (e.g. 24, 30). Default matches input.",
)
def convert(
    input_path: str,
    output_path: str,
    mode: str,
    render_mode: str,
    extrusion: float,
    matte_ratio: float,
    stereo: bool,
    vr: bool,
    anaglyph: bool,
    depth_preview: bool,
    device: str | None,
    depth_backend: str,
    segmentation_backend: str,
    inpainting_backend: str,
    start: str | None,
    end: str | None,
    fps: float | None,
) -> None:
    """Convert INPUT_PATH (mp4/mov/avi) into a pop-out 3D video at OUTPUT_PATH."""
    resolved_render = _resolve_render_mode(render_mode, stereo, vr, anaglyph)
    config = PipelineConfig.from_mode(
        Mode(mode),
        render_mode=resolved_render,
        extrusion=extrusion,
        matte_ratio=matte_ratio,
        depth_backend=DepthBackend(depth_backend),
        segmentation_backend=SegmentationBackend(segmentation_backend),
        inpainting_backend=InpaintingBackend(inpainting_backend),
        write_depth_preview=depth_preview,
        output_fps=fps,
    )
    pipeline = DimendiaPipeline(config, device=device)

    # Resolve timestamps to frame indices (need fps first).
    start_frame = 0
    end_frame = None
    if start is not None or end is not None:
        from dimendia.io import VideoReader as _VR

        _probe = _VR(input_path)
        fps = _probe.meta.fps
        _probe.close()
        if start is not None:
            start_frame = int(round(_parse_timestamp(start) * fps))
        if end is not None:
            end_frame = int(round(_parse_timestamp(end) * fps))

    bar = tqdm(desc="DIMENDIA", unit="f")

    def on_progress(ev: ProgressEvent) -> None:
        if bar.total is None and ev.total:
            bar.total = ev.total
        bar.update(1)

    result = pipeline.convert(
        input_path,
        output_path,
        progress=on_progress,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    bar.close()
    click.echo(
        f"Wrote {result.output_path} ({result.frames} frames, {result.width}x{result.height})"
    )
    if result.depth_preview_path:
        click.echo(f"Depth preview: {result.depth_preview_path}")


@main.command()
@click.option("--mode", type=click.Choice([m.value for m in Mode]), default=Mode.FAST.value)
@click.option(
    "--render-mode",
    type=click.Choice([r.value for r in RenderMode]),
    default=RenderMode.POPOUT.value,
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Keep the rendered output at this path instead of a temp file.",
)
def selftest(mode: str, render_mode: str, output: str | None) -> None:
    """Generate a synthetic clip and run the full pipeline (CPU, no downloads)."""
    from dimendia.synthetic import make_synthetic_clip

    workdir = tempfile.mkdtemp(prefix="dimendia_selftest_")
    src = make_synthetic_clip(Path(workdir) / "input.mp4", n_frames=16, width=256, height=192)
    out = output or str(Path(workdir) / "output.mp4")
    result = convert_video(src, out, mode=mode, render_mode=render_mode, depth_preview=True)
    assert Path(result.output_path).exists() and Path(result.output_path).stat().st_size > 0
    click.echo(
        f"selftest OK -> {result.output_path} "
        f"({result.frames} frames, {result.width}x{result.height})"
    )


@main.command()
def info() -> None:
    """Print which backends resolve in the current environment."""
    from dimendia.depth import build_depth_estimator
    from dimendia.inpainting import build_inpainter
    from dimendia.segmentation import build_tracker

    cfg = PipelineConfig()
    depth = build_depth_estimator(cfg.depth_backend)
    tracker = build_tracker(cfg)
    inpainter = build_inpainter(cfg.inpainting_backend)
    click.echo(f"depth backend       : {depth.name}")
    click.echo(f"segmentation backend: {tracker.name}")
    click.echo(f"inpainting backend  : {inpainter.name}")


if __name__ == "__main__":
    main()
