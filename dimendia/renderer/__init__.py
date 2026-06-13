"""Stage E — pop-out / parallax / stereo / VR rendering."""

from dimendia.config import PipelineConfig, RenderMode
from dimendia.renderer.frame_break_renderer import FrameBreakRenderer
from dimendia.renderer.stereo_renderer import StereoRenderer


def build_renderer(config: PipelineConfig):
    """Return the renderer implementing ``config.render_mode``."""
    if config.render_mode in (RenderMode.STEREO, RenderMode.VR, RenderMode.ANAGLYPH):
        return StereoRenderer(config)
    return FrameBreakRenderer(config)


__all__ = ["FrameBreakRenderer", "StereoRenderer", "build_renderer"]
