"""Configuration: quality modes, render modes, scoring weights, and presets.

A :class:`PipelineConfig` fully determines a run. The three named modes
(``fast`` / ``balanced`` / ``quality``) are convenience presets that trade speed
for fidelity; every field can still be overridden individually.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """Speed/quality preset."""

    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class RenderMode(str, Enum):
    """Output rendering style (maps to the four renderer modes in the spec)."""

    POPOUT = "popout"  # Mode 1 — Pop-Out Cinematic (frame-break)
    PARALLAX = "parallax"  # Mode 2 — Depth Parallax
    STEREO = "stereo"  # Mode 3 — Stereo Left/Right (anaglyph or SBS)
    VR = "vr"  # Mode 4 — VR Export (side-by-side / top-bottom)
    ANAGLYPH = "anaglyph"  # Red-cyan anaglyph (single-image stereo)


class DepthBackend(str, Enum):
    AUTO = "auto"  # prefer Depth Anything V2, fall back automatically
    DEPTH_ANYTHING_V2 = "depth_anything_v2"
    MIDAS = "midas"
    CLASSICAL = "classical"  # CPU, no weights


class SegmentationBackend(str, Enum):
    AUTO = "auto"
    SAM2 = "sam2"
    CLASSICAL = "classical"  # saliency + motion + GrabCut


class SemanticBackend(str, Enum):
    """Optional semantic prior used to bias extrusion-object selection."""

    NONE = "none"
    HAAR = "haar"  # OpenCV Haar cascade (e.g. frontal face detection)


class InpaintingBackend(str, Enum):
    AUTO = "auto"
    PROPAINTER = "propainter"
    CLASSICAL = "classical"  # cv2.inpaint + temporal propagation


class ScoringWeights(BaseModel):
    """Weights for the saliency-driven extrusion score.

    ``score = depth*proximity + motion*velocity + saliency*attention + center*framing``
    """

    depth: float = 0.40
    motion: float = 0.25
    saliency: float = 0.20
    center: float = 0.15
    semantic: float = 0.0


class TemporalDepthConfig(BaseModel):
    ema_alpha: float = Field(0.6, ge=0.0, le=1.0)  # weight on the new frame
    scale_align: bool = True  # least-squares scale+shift to previous frame
    motion_aware: bool = True  # relax smoothing where flow magnitude is high
    motion_threshold: float = 2.0  # px/frame above which smoothing is reduced


class PipelineConfig(BaseModel):
    """Top-level configuration for a single conversion run."""

    mode: Mode = Mode.BALANCED
    render_mode: RenderMode = RenderMode.POPOUT

    # Backends (AUTO resolves to the best available at runtime).
    depth_backend: DepthBackend = DepthBackend.AUTO
    segmentation_backend: SegmentationBackend = SegmentationBackend.AUTO
    inpainting_backend: InpaintingBackend = InpaintingBackend.AUTO

    # Processing resolution: long edge is capped here for speed; output is
    # rescaled back to the source resolution.
    work_long_edge: int = 720

    # Rendering knobs.
    extrusion: float = 100.0  # subjective strength 0..200, scales parallax/pop
    matte_ratio: float = 0.12  # cinematic bar thickness as fraction of height
    popout_scale: float = 0.10  # additional size boost for the primary object in popout mode
    bar_glow: float = 0.22  # subtle highlight strength where the object crosses the matte bars
    stereo_baseline: float = 0.04  # virtual interaxial as fraction of width
    mesh_warp: bool = False  # use triangle-mesh warping instead of scalar/per-pixel

    # Segmentation / selection.
    selection_stability: float = 0.18  # reward for keeping the same primary object across frames
    selection_switch_threshold: float = 0.08  # required score gap to switch primary objects
    selection_size_penalty: float = 0.16  # penalize overly large object masks so background doesn't dominate
    morphology_kernel: int = 7  # tracker mask cleanup kernel size

    # Scene model.
    num_layers: int = 3  # projectile + (N-2) depth bands + background

    # Stability / quality toggles.
    scene_cut_threshold: float = 0.6  # Bhattacharyya hist distance that marks a cut
    adaptive_extrusion: bool = True  # scale extrusion by per-frame depth range
    bidirectional_depth: bool = False  # forward+backward depth stabilization pass
    semantic_backend: SemanticBackend = SemanticBackend.HAAR

    # Stage configs.
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    temporal: TemporalDepthConfig = Field(default_factory=lambda: TemporalDepthConfig())  # type: ignore[arg-type,call-arg]

    # Toggles.
    inpaint: bool = True
    write_depth_preview: bool = False

    # Target output framerate. If None, matches input fps.
    output_fps: float | None = None

    @classmethod
    def from_mode(cls, mode: Mode, **overrides: object) -> PipelineConfig:
        """Build a config from a named preset, then apply field overrides."""
        preset = _MODE_PRESETS[mode]
        merged = {**preset, "mode": mode, **overrides}
        return cls.model_validate(merged)


# Preset field overrides per mode. Kept as plain dicts so ``from_mode`` can merge
# them with user overrides cleanly.
_MODE_PRESETS: dict[Mode, dict[str, object]] = {
    Mode.FAST: {
        "work_long_edge": 540,
        "inpaint": False,
        "temporal": TemporalDepthConfig(ema_alpha=0.7, scale_align=True, motion_aware=False),
    },
    Mode.BALANCED: {
        "work_long_edge": 720,
        "inpaint": True,
        "temporal": TemporalDepthConfig(ema_alpha=0.6),
    },
    Mode.QUALITY: {
        "work_long_edge": 1080,
        "inpaint": True,
        "temporal": TemporalDepthConfig(ema_alpha=0.5, motion_threshold=1.5),
    },
}
