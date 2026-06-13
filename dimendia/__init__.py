"""DIMENDIA — monocular video to cinematic pop-out 3D.

The public surface is intentionally small: most users either call the
:func:`dimendia.convert_video` helper or use the ``dimendia`` CLI.
"""

from __future__ import annotations

from dimendia.config import Mode, PipelineConfig, RenderMode
from dimendia.pipeline import DimendiaPipeline, convert_video

__version__ = "0.1.0"

__all__ = [
    "Mode",
    "RenderMode",
    "PipelineConfig",
    "DimendiaPipeline",
    "convert_video",
    "__version__",
]
