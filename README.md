# DIMENDIA

**Convert ordinary monocular videos into cinematic pop-out 3D** — objects visually
emerge from the screen via depth-aware layered scene reconstruction and frame-break
rendering. No depth sensors, no stereo cameras, no manual masking required.

## Pipeline

| Stage | Description | Default Backend | GPU Backend |
|-------|------------|-----------------|-------------|
| **A** — Depth | Temporally stable monocular depth (scale-aligned EMA) | Classical (gradient + center + vertical cues) | Depth Anything V2 Large / MiDaS |
| **B** — Segmentation | Multi-object tracking + extrusion scoring | Motion + saliency + depth + GrabCut | SAM2 |
| **C** — LDI | 3-layer depth image (projectile / foreground / background) | — | — |
| **D** — Inpainting | Disocclusion fill behind extruded objects | Telea + temporal propagation | ProPainter |
| **E** — Render | Pop-out / parallax / stereo / VR output | — | — |

> **CPU fallbacks** run the full pipeline end-to-end without a GPU or model
> downloads. Install `dimendia[models]` and weights for high-quality depth/segmentation.

## Install

```bash
# Core (CPU, no model downloads)
pip install -e .

# With GPU model backends (Depth Anything V2, RAFT flow)
pip install -e ".[models]"

# Development
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, ffmpeg.

## Quick Start

```bash
# Convert a video with pop-out 3D (default mode)
python -m dimendia.cli.convert convert input.mp4 output.mp4

# Fast mode, depth preview
python -m dimendia.cli.convert convert input.mp4 output.mp4 --mode fast --depth-preview

# Quality mode, increased pop-out
python -m dimendia.cli.convert convert input.mp4 output.mp4 --mode quality --extrusion 150

# Stereoscopic side-by-side output
python -m dimendia.cli.convert convert input.mp4 output_sbs.mp4 --stereo

# VR (half-SBS for headsets)
python -m dimendia.cli.convert convert input.mp4 output_vr.mp4 --vr

# Check which backends resolved
python -m dimendia.cli.convert info

# End-to-end self-test (synthetic clip, CPU, no downloads)
python -m dimendia.cli.convert selftest
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--mode {fast,balanced,quality}` | Speed/quality preset (default: balanced) |
| `--render-mode {popout,parallax,stereo,vr}` | Rendering style (default: popout) |
| `--extrusion FLOAT` | Pop-out strength 0–200 (default: 100) |
| `--stereo` | Side-by-side stereoscopic output |
| `--vr` | Half-width SBS for VR |
| `--depth-preview` | Also write a depth visualization video |
| `--depth-backend {auto,depth_anything_v2,midas,classical}` | Force depth backend |
| `--segmentation-backend {auto,sam2,classical}` | Force segmentation backend |
| `--inpainting-backend {auto,propainter,classical}` | Force inpainting backend |
| `--device` | Force compute device (cpu, cuda) |

## Render Modes

1. **Pop-Out Cinematic** (`popout`) — cinematic matte bars; the primary object
   breaks out of the frame toward the viewer.
2. **Depth Parallax** (`parallax`) — gentle viewpoint sway scaled by depth.
3. **Stereo L/R** (`stereo`) — full side-by-side for 3D displays/glasses.
4. **VR Export** (`vr`) — half-width side-by-side for VR headsets.

## Extrusion Scoring

The system automatically selects the pop-out object:

```
score = depth_weight × proximity
      + motion_weight × velocity
      + saliency_weight × attention
      + center_weight × framing
```

Weights are configurable in `PipelineConfig.weights`.

## Architecture

```
dimendia/
├── config.py              # Mode presets, scoring weights, pipeline config
├── pipeline.py            # Stage A→E orchestrator
├── io/video.py            # ffmpeg frame read/write, audio passthrough
├── depth/                 # Stage A
│   ├── depth_anything.py  # Depth Anything V2 (GPU)
│   ├── midas.py           # MiDaS fallback (GPU)
│   ├── classical.py       # Weight-free CPU depth
│   └── temporal_depth.py  # Scale alignment + EMA + motion-aware smoothing
├── segmentation/          # Stage B
│   ├── sam2_tracker.py    # SAM2 adapter (GPU)
│   ├── classical_tracker.py  # Motion + saliency + GrabCut (CPU)
│   ├── flow.py            # RAFT / Farneback optical flow
│   ├── saliency.py        # Spectral residual attention
│   └── projectile_selector.py  # Extrusion scoring
├── ldi/                   # Stage C
│   └── layered_depth_image.py  # 3-layer LDI + builder
├── inpainting/            # Stage D
│   ├── propainter_adapter.py   # ProPainter CLI integration
│   └── classical.py       # Telea + temporal propagation (CPU)
├── renderer/              # Stage E
│   ├── frame_break_renderer.py  # Pop-out cinematic + parallax
│   ├── stereo_renderer.py # DIBR stereo / VR
│   └── compositor.py      # Layer warp + composite primitives
└── cli/convert.py         # `dimendia` CLI entrypoint
```

## GPU Model Setup

### Depth Anything V2

```bash
pip install dimendia[models]
# Weights auto-download via HuggingFace on first run
```

### SAM2

```bash
pip install sam2  # or clone https://github.com/facebookresearch/sam2
export SAM2_CHECKPOINT=/path/to/sam2_hiera_large.pt
export SAM2_CONFIG=sam2_hiera_l.yaml
```

### ProPainter

```bash
git clone https://github.com/sczhou/ProPainter /opt/ProPainter
# Download weights per ProPainter README
export PROPAINTER_DIR=/opt/ProPainter
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install

# Lint & format
ruff check dimendia
black dimendia
mypy dimendia

# Self-test (CPU, no downloads)
dimendia selftest
```

## License

Apache 2.0
