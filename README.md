# DIMENDIA

**Convert ordinary monocular videos into cinematic pop-out 3D** — objects visually
emerge from the screen via depth-aware layered scene reconstruction and frame-break
rendering. No depth sensors, no stereo cameras, no manual masking required.

DIMENDIA runs the **entire pipeline on CPU** with zero model downloads (classical
fallbacks), and transparently upgrades to GPU model backends when they're installed.

---

## How it works

A monocular frame is turned into 3D through five ordered stages (A→E):

| Stage | Description | Default Backend (CPU) | Optional GPU Backend |
|-------|------------|-----------------------|----------------------|
| **A** — Depth | Temporally stable monocular depth (scale-aligned EMA, optionally bidirectional) | Classical (multi-scale contrast + center + vertical cues) | Depth Anything V2 / MiDaS |
| **B** — Segmentation | Multi-object tracking (Kalman persistence) + extrusion scoring | Motion + saliency + depth + GrabCut | SAM2 |
| **C** — LDI | Layered depth image: projectile + N−2 depth bands + background (configurable) | — | — |
| **D** — Inpainting | Depth-context disocclusion fill behind extruded objects | Telea + temporal propagation | ProPainter |
| **E** — Render | Pop-out / parallax / stereo / VR / anaglyph output | — | — |

---

## Tutorial: from zero to a 3D video

This walks through a complete first run end-to-end. Every step here works on a
laptop CPU — no GPU required.

### 1. Prerequisites

- **Python ≥ 3.10**
- **ffmpeg** (used for reading/writing video and copying the audio track)

```bash
python --version        # 3.10 or newer
ffmpeg -version          # any recent build
```

If `ffmpeg` is missing: `apt-get install ffmpeg` (Debian/Ubuntu),
`brew install ffmpeg` (macOS), or download a static build for Windows.

### 2. Install

```bash
git clone https://github.com/er-del/Dimentia.git
cd Dimentia
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Core install — CPU-only, no model downloads
pip install -e .
```

This installs the `dimendia` command. Confirm it's available:

```bash
dimendia --help
```

> Prefer not to install? Every command below also works as a module:
> `python -m dimendia.cli.convert <subcommand> ...`

### 3. Verify your setup

Check which backends resolved in your environment (CPU classical backends are
always available; GPU ones appear only if their weights/packages are installed):

```bash
dimendia info
```

Then run the built-in self-test. It synthesizes a tiny clip and pushes it through
the full A→E pipeline — a great smoke test that needs no input video and no
downloads:

```bash
dimendia selftest                       # uses a temp file
dimendia selftest --output selftest.mp4 # keep the result to inspect
```

You should see `selftest OK -> ... (N frames, WxH)`.

### 4. Your first conversion

Point it at any short `.mp4` / `.mov` / `.avi`. Start with `--mode fast` so your
first run finishes quickly:

```bash
dimendia convert input.mp4 output.mp4 --mode fast
```

The output keeps the source resolution and audio. The primary "pop-out" object is
selected automatically (see [Extrusion scoring](#extrusion-scoring)).

> No clip handy? Grab the self-test output from step 3 and feed it back in:
> `dimendia convert selftest.mp4 output.mp4 --mode fast`.

### 5. See what the depth looks like

To understand *why* objects pop the way they do, also write a depth visualization
video alongside the result:

```bash
dimendia convert input.mp4 output.mp4 --mode fast --depth-preview
# writes output.mp4 and output_depth.mp4
```

### 6. Pick a quality mode

| Mode | Processing resolution | Inpainting | Use for |
|------|----------------------|-----------|---------|
| `fast` | 540p | off | quick previews, iteration |
| `balanced` *(default)* | 720p | on | most footage |
| `quality` | 1080p | on | final renders |

```bash
dimendia convert input.mp4 output.mp4 --mode quality
```

### 7. Pick a render (output) mode

| Render mode | Flag | What you get |
|-------------|------|--------------|
| Pop-out cinematic *(default)* | `--render-mode popout` | matte bars + object breaking out of the frame |
| Depth parallax | `--render-mode parallax` | gentle viewpoint sway scaled by depth |
| Stereo L/R | `--stereo` | full side-by-side for 3D displays/glasses |
| VR | `--vr` | half-width side-by-side for headsets |
| Anaglyph | `--anaglyph` | single-image red-cyan 3D (works with cheap glasses) |

```bash
dimendia convert input.mp4 out_popout.mp4                       # popout
dimendia convert input.mp4 out_parallax.mp4 --render-mode parallax
dimendia convert input.mp4 out_sbs.mp4 --stereo
dimendia convert input.mp4 out_vr.mp4 --vr
dimendia convert input.mp4 out_anaglyph.mp4 --anaglyph
```

### 8. Tune the pop-out strength

`--extrusion` scales the effect (0–200, default 100). By default DIMENDIA also
*adapts* extrusion per frame based on the scene's depth range, so flat scenes don't
over-pop and deep scenes don't under-pop.

```bash
dimendia convert input.mp4 output.mp4 --extrusion 150   # stronger
dimendia convert input.mp4 output.mp4 --extrusion 60    # subtler
```

### 9. Trim and reframe

Process only part of a clip and/or retime the output:

```bash
# Only the segment from 5s to 12s
dimendia convert input.mp4 clip.mp4 --start 00:00:05 --end 00:00:12

# Force 24 fps output
dimendia convert input.mp4 output.mp4 --fps 24
```

`--start` / `--end` accept `HH:MM:SS`, `MM:SS`, or raw seconds.

### 10. Advanced features (Python API)

A few capabilities aren't exposed as CLI flags yet. Reach them through the
`convert_video` helper, which forwards any extra keyword to `PipelineConfig`:

```python
from dimendia import convert_video
from dimendia.config import ScoringWeights

convert_video(
    "input.mp4", "output.mp4",
    mode="quality",
    render_mode="anaglyph",
    extrusion=120,

    # --- advanced toggles ---
    num_layers=5,              # more depth bands -> smoother parallax (default 3)
    mesh_warp=True,            # triangle-mesh warp that tears at depth edges
    adaptive_extrusion=True,   # scale pop-out by per-frame depth range (default on)
    bidirectional_depth=True,  # forward+backward depth stabilization (steadier)
    scene_cut_threshold=0.6,   # reset temporal state on hard cuts (HSV hist dist)

    # bias the auto-selected pop-out object toward detected faces
    weights=ScoringWeights(depth=0.35, motion=0.2, saliency=0.15,
                           center=0.1, semantic=0.2),
)
```

- **`num_layers`** — the scene is split into `projectile + (num_layers − 2)` depth
  bands `+ background`. Higher = smoother depth transitions, slower.
- **`mesh_warp`** — warps each layer as a displaced triangle mesh, tearing
  triangles across large depth discontinuities (cleaner silhouettes than per-pixel
  warping).
- **`bidirectional_depth`** — runs depth stabilization both forward and backward
  over the clip and blends them; reduces flicker at the cost of buffering frames.
- **`weights.semantic`** — when `> 0`, a Haar face detector boosts objects whose
  mask overlaps a detected face, so people are preferred as the pop-out subject.

For full control, build the config explicitly:

```python
from dimendia import DimendiaPipeline, PipelineConfig, RenderMode

config = PipelineConfig.from_mode("balanced", render_mode=RenderMode.STEREO, num_layers=4)
pipeline = DimendiaPipeline(config, device="cpu")
result = pipeline.convert("input.mp4", "output.mp4")
print(result.output_path, result.frames, result.width, result.height)
```

### 11. Go GPU (optional)

For higher-fidelity depth and segmentation, install the model extras and the
backends will be picked up automatically (see [GPU model setup](#gpu-model-setup)):

```bash
pip install -e ".[models]"
dimendia convert input.mp4 output.mp4 --mode quality --device cuda
```

You're done — that's a full pipeline from install to a finished 3D video.

---

## CLI reference

```
dimendia convert INPUT_PATH OUTPUT_PATH [OPTIONS]
dimendia selftest [OPTIONS]
dimendia info
```

| Flag | Description |
|------|-------------|
| `--mode {fast,balanced,quality}` | Speed/quality preset (default: `balanced`) |
| `--render-mode {popout,parallax,stereo,vr,anaglyph}` | Rendering style (default: `popout`) |
| `--extrusion FLOAT` | Pop-out strength 0–200 (default: 100) |
| `--stereo` | Side-by-side stereoscopic output |
| `--vr` | Half-width SBS for VR headsets |
| `--anaglyph` | Red-cyan anaglyph output |
| `--depth-preview` | Also write a depth visualization video (`*_depth.mp4`) |
| `--start TEXT` | Start timestamp (`HH:MM:SS`, `MM:SS`, or seconds) |
| `--end TEXT` | End timestamp (`HH:MM:SS`, `MM:SS`, or seconds) |
| `--fps FLOAT` | Target output frame rate (default: match input) |
| `--depth-backend {auto,depth_anything_v2,midas,classical}` | Force depth backend |
| `--segmentation-backend {auto,sam2,classical}` | Force segmentation backend |
| `--inpainting-backend {auto,propainter,classical}` | Force inpainting backend |
| `--device TEXT` | Force compute device (`cpu`, `cuda`) |

> `--stereo`, `--vr`, and `--anaglyph` are shortcuts that override `--render-mode`.

---

## Python API reference

```python
convert_video(
    input_path, output_path, *,
    mode="balanced", render_mode="popout",
    extrusion=100.0, depth_preview=False, device=None,
    progress=None, **config_overrides,
) -> ConversionResult
```

`**config_overrides` are applied on top of the mode preset, so any
`PipelineConfig` field below can be set here.

### Key `PipelineConfig` fields

| Field | Default | Meaning |
|-------|---------|---------|
| `mode` | `balanced` | Speed/quality preset |
| `render_mode` | `popout` | Output style (`popout`/`parallax`/`stereo`/`vr`/`anaglyph`) |
| `extrusion` | `100.0` | Pop-out strength (0–200) |
| `work_long_edge` | `720` | Internal processing resolution cap |
| `num_layers` | `3` | LDI layers: `projectile + (N−2) bands + background` |
| `mesh_warp` | `False` | Triangle-mesh warping with depth-edge tearing |
| `adaptive_extrusion` | `True` | Scale extrusion by per-frame depth range |
| `bidirectional_depth` | `False` | Forward+backward depth stabilization pass |
| `scene_cut_threshold` | `0.6` | HSV-histogram (Bhattacharyya) distance that triggers a temporal reset |
| `stereo_baseline` | `0.04` | Virtual interaxial distance (fraction of width) |
| `semantic_backend` | `haar` | Semantic prior for object selection (`none`/`haar`) |
| `weights` | see below | Extrusion scoring weights |
| `inpaint` | `True` | Reconstruct background behind extruded objects |
| `output_fps` | `None` | Target output fps (default: match input) |

---

## Extrusion scoring

The pop-out object is chosen automatically by a weighted score:

```
score = depth_weight    × proximity      # how near the camera
      + motion_weight    × velocity       # how fast it moves
      + saliency_weight  × attention       # how visually salient
      + center_weight    × framing         # how centered
      + semantic_weight  × face_overlap     # overlaps a detected face (if enabled)
```

Defaults (`ScoringWeights`): `depth=0.40`, `motion=0.25`, `saliency=0.20`,
`center=0.15`, `semantic=0.0`. Set `semantic > 0` to prefer people.

---

## Architecture

```
dimendia/
├── config.py              # Mode presets, scoring weights, pipeline config
├── pipeline.py            # Stage A→E orchestrator
├── io/video.py            # ffmpeg frame read/write, audio passthrough
├── depth/                 # Stage A
│   ├── depth_anything.py  # Depth Anything V2 (GPU)
│   ├── midas.py           # MiDaS fallback (GPU)
│   ├── classical.py       # Weight-free CPU depth (multi-scale contrast)
│   └── temporal_depth.py  # Scale alignment + EMA + occlusion-aware smoothing
├── segmentation/          # Stage B
│   ├── sam2_tracker.py    # SAM2 adapter (GPU)
│   ├── classical_tracker.py  # Motion + saliency + GrabCut + Kalman (CPU)
│   ├── flow.py            # RAFT / Farneback flow + occlusion mask
│   ├── saliency.py        # Spectral residual attention
│   └── projectile_selector.py  # Extrusion scoring (+ optional Haar faces)
├── ldi/                   # Stage C
│   └── layered_depth_image.py  # N-layer LDI + depth-aware matting
├── inpainting/            # Stage D
│   ├── propainter_adapter.py   # ProPainter CLI integration
│   └── classical.py       # Telea + temporal propagation + depth context (CPU)
├── renderer/              # Stage E
│   ├── frame_break_renderer.py  # Pop-out cinematic + parallax
│   ├── stereo_renderer.py # DIBR stereo / VR / anaglyph
│   └── compositor.py      # Layer warp (per-pixel + mesh) + composite primitives
└── cli/convert.py         # `dimendia` CLI entrypoint
```

---

## GPU model setup

GPU backends are optional. With `pip install -e ".[models]"` and the weights below,
`--depth-backend auto` / `--segmentation-backend auto` will prefer them.

### Depth Anything V2

```bash
pip install -e ".[models]"
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
# Download weights per the ProPainter README
export PROPAINTER_DIR=/opt/ProPainter
```

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install

# Lint, format, type-check
ruff check dimendia
black dimendia
mypy dimendia

# Self-test (CPU, no downloads)
dimendia selftest
```

---

## Troubleshooting

- **`ffmpeg` not found / cannot open video** — install ffmpeg (step 1) and confirm
  `ffmpeg -version` works in the same shell/venv.
- **`dimendia: command not found`** — activate the venv, or run the module form
  `python -m dimendia.cli.convert ...`.
- **Conversion is slow** — use `--mode fast`, lower `--extrusion`, or trim with
  `--start`/`--end`. Resolution is capped internally by `work_long_edge`.
- **GPU backend not used** — run `dimendia info` to see what resolved; install
  `.[models]` and any required weights/env vars, then pass `--device cuda`.
- **Wrong object pops out** — adjust `ScoringWeights` (e.g. raise `center` or set
  `semantic > 0` to prefer faces) via the Python API.

---

## License

Apache 2.0
