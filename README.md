# D4RT: Dynamic 4D Reconstruction and Tracking

This is an implementation of Google DeepMind's [D4RT](https://d4rt-paper.github.io/) - a unified transformer model for 4D scene reconstruction from video.

## Overview

D4RT is a single model that can:
- **Track points** through video sequences (sparse and dense)
- **Reconstruct dense depth** maps (including sub-pixel resolution)
- **Generate point clouds** with colors and normals
- **Predict long-term motion** beyond video length
- **Estimate camera poses** between frames

## Architecture (Figure 7 - Paper Exact)

```
Input Video [B, T, C, H, W]              Original Aspect Ratio
(Resize to square 256×256)                      W/H
         │                                       │
         ▼                                       ▼
┌─────────────────────┐                  ┌──────────────┐
│     Tokenizer       │                  │      FC      │
│  (2×16×16 patches)  │                  │ (scalar→768) │
│  (NO normalization) │                  │              │
└─────────────────────┘                  └──────────────┘
         │                                       │
         ▼                                       │
┌─────────────────────┐                          │
│        PE           │                          │
│ (pos encoding)      │               (NO PE for AR token)
└─────────────────────┘                          │
         │                                       │
         ▼                                       ▼
   [B, 3072, 768]                          [B, 1, 768]
   (video + PE)                            (AR token)
         │                                       │
         └───────────── Concat ──────────────────┘
                          │
                          ▼
                   [B, 3073, 768]
                          │
         ┌────────────────┴────────────────┐
         │       N Encoder Blocks          │
         │  ┌───────────────────────────┐  │
         │  │ Per-Frame Self-Attention  │  │  ← Local (256 tokens/frame)
         │  │           ↓               │  │    (AR token in global only)
         │  │          MLP              │  │
         │  │           ↓               │  │
         │  │ Global Self-Attention     │  │  ← All 3073 tokens
         │  │           ↓               │  │
         │  │          MLP              │  │
         │  └───────────────────────────┘  │
         │         (× N layers)            │
         └─────────────────────────────────┘
                          │
                          ▼
                  Remove AR token
                          │
                          ▼
              Global Scene Representation F
                   [B, 3072, 768]
                          │
                          ▼
┌─────────────────────────────────────┐
│    Query Encoder                    │
│  - Fourier(u,v) → 40D               │
│  - Temporal embeddings → 768D       │
│  - 9×9 RGB patch CNN → 256D         │
│  → Concat + Project → 512D          │
└─────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────┐
│   Cross-Attention Decoder (8L)      │
│  Self-Attn → Cross-Attn → FFN       │
└─────────────────────────────────────┘
                          │
                          ▼
   Output Heads:
   - xyz: 3D position [B, N, 3]
   - uv: 2D coordinates [B, N, 2]
   - normals: surface normals [B, N, 3]
   - motion: displacement [B, N, 3]
   - visibility: occlusion [B, N, 1]
   - confidence: prediction quality [B, N, 1]
```

### Key Architectural Features

| Feature | Implementation |
|---------|---------------|
| **Encoder Blocks** | Each block has BOTH local + global attention (Figure 7) |
| **Aspect Ratio Token** | W/H → FC → separate token, NO positional encoding |
| **Patch Normalization** | Disabled by default (preserves brightness) |
| **Output Heads** | 13D total: xyz(3) + uv(2) + vis(1) + disp(3) + normal(3) + conf(1) |

## Capabilities

| Task | Query Pattern | Output |
|------|---------------|--------|
| Point Tracking | Fixed (u,v,t_src), varying t_tgt | 3D trajectories |
| Dense Tracking | All pixels, all frames | Full scene flow |
| Depth Map | All pixels, t_src=t_tgt=t_cam | Per-frame depth |
| Sub-Pixel Depth | Continuous (u,v) at higher resolution | High-res depth (Figure 9) |
| Point Cloud | All pixels, fixed t_cam | 3D reconstruction |
| Long-term Pred | Extend t_tgt beyond video | Future positions |

### Sub-Pixel Depth (Figure 9)

The model's continuous coordinate space (Fourier encoding of (u,v) ∈ [0,1]²) enables:
- Query at arbitrary positions, not tied to pixel grid
- Output resolution independent of input (256×256 input → 512×512 or higher output)
- Achieves sub-pixel accuracy by leveraging learned scene representation

## Loss Function (Paper-Exact)

```
L = (1/N) Σ [c·λ3D·L3D - λconf·log(c) + λ2D·L2D + λvis·Lvis + λdisp·Ldisp + λnormal·Lnormal]
```

Key features:
- **Confidence-weighted 3D loss**: `c·λ3D·L3D` - 3D error scaled by prediction confidence
- **Confidence penalty**: `-λconf·log(c)` - encourages high confidence predictions

**Weights from paper:**
| Weight | Value | Description |
|--------|-------|-------------|
| λ3D | 1.0 | Primary 3D supervision |
| λ2D | 0.1 | 2D coordinate loss |
| λvis | 0.1 | Visibility loss |
| λdisp | 0.1 | Motion displacement loss |
| λconf | 0.2 | Confidence penalty |
| λnormal | 0.5 | Surface normal loss |

## Project Structure

```
d4rt/
├── models/          # Model architectures
│   ├── encoder.py           # Spatio-temporal ViT encoder
│   ├── decoder.py           # Cross-attention decoder
│   └── components/          # Attention, embeddings, blocks
│       ├── aspect_ratio_token.py  # W/H → FC → token
│       ├── encoder_block.py       # Local + global attention
│       └── ...
├── losses/          # Loss functions
│   ├── composite_loss.py    # Paper-exact loss
│   └── ...
├── inference/       # Inference modules
│   ├── tracking.py          # Point tracking
│   ├── dense_tracking.py    # All-pixel tracking
│   ├── depth.py             # Depth reconstruction
│   ├── subpixel_depth.py    # Sub-pixel depth (Figure 9)
│   ├── point_cloud.py       # Point cloud export
│   └── long_term.py         # Future prediction
├── visualization/   # Paper-style visualizations
│   └── d4rt_visualizer.py
├── data/            # Dataset loaders
└── training/        # Training loop

configs/
├── model/
│   └── vit_b_d4rt.yaml      # Paper architecture config
└── training/
    └── train_d4rt.yaml      # Paper training config

tests/                       # Complete test suite
```

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/MasahiroOgawa/D4RT_MasImpl.git
cd D4RT_MasImpl

# Sync dependencies
uv sync

# Activate
source .venv/bin/activate
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### Training (Paper Architecture)

```bash
python scripts/train.py --config configs/training/train_d4rt.yaml
```

### Inference

```python
from d4rt.models import build_d4rt_model
from d4rt.inference import PointTracker, DensePixelTracker, SubPixelDepthReconstructor

# Build model
model = build_d4rt_model(config)

# Point tracking
tracker = PointTracker(model)
trajectories, visibility = tracker.track_points(video, points)

# Dense tracking
dense_tracker = DensePixelTracker(model)
result = dense_tracker.track_all_pixels(video, source_frame=0, stride=8)

# Sub-pixel depth (Figure 9)
depth_reconstructor = SubPixelDepthReconstructor(model)
depth = depth_reconstructor.reconstruct_depth(
    video, frame_idx=0, output_resolution=(512, 512)  # Higher than input
)
```

### Visualization

```python
from d4rt.visualization import D4RTVisualizer

# Depth map visualization
D4RTVisualizer.visualize_depth_map(depth, "output/depth.png", colormap='turbo')

# Dense track visualization
D4RTVisualizer.visualize_dense_tracks(video, trajectories_2d, visibility, "output/tracks.mp4")

# Point cloud export
from d4rt.inference import PointCloudReconstructor
reconstructor = PointCloudReconstructor(model)
result = reconstructor.reconstruct(video, frame_idx=0)
reconstructor.export_ply(result['points'], result['colors'], "output/scene.ply")
```

## VideoMAE Pretrained Weights

For best results, use VideoMAE pretrained weights:

```python
from d4rt.models.encoder import load_videomae_weights

encoder = build_vit_encoder(config)
load_videomae_weights(encoder, "path/to/videomae_vit_b.pth")
```

Download from: https://github.com/MCG-NJU/VideoMAE

## Model Configurations

| Model | Layers | Hidden Dim | Parameters | Config |
|-------|--------|------------|------------|--------|
| ViT-B | 12 | 768 | ~230M | `vit_b_d4rt.yaml` |
| ViT-L | 24 | 1024 | ~451M | `vit_l.yaml` |
| ViT-g | 40 | 1408 | ~1.144B | `vit_g.yaml` |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_encoder.py -v

# Run with coverage
pytest tests/ --cov=d4rt --cov-report=html
```

## References

- **Paper**: [D4RT: Unified, Fast 4D Scene Reconstruction & Tracking](https://arxiv.org/html/2512.08924v1)
- **Website**: https://d4rt-paper.github.io/
- **Original work**: Google DeepMind

## License

This project is licensed under the GNU General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.
