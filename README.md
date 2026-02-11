# D4RT: Dynamic 4D Reconstruction and Tracking

Implementation of Google DeepMind's [D4RT](https://d4rt-paper.github.io/) - a unified transformer model for 4D scene reconstruction from video.

## What This Can Do

D4RT is a single model that handles multiple 4D vision tasks:

| Task | Description |
|------|-------------|
| **Point Tracking** | Track sparse points through video in 3D |
| **Dense Tracking** | Track all pixels with full scene flow |
| **Depth Estimation** | Reconstruct per-frame depth maps |
| **Point Cloud** | Generate 3D point clouds with colors/normals |
| **Long-term Prediction** | Predict positions beyond video length |

## Installation

```bash
git clone https://github.com/MasahiroOgawa/D4RT_MasImpl.git
cd D4RT_MasImpl

# Using uv (recommended)
uv sync
source .venv/bin/activate

# Or using pip
pip install -e .
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/training/train_paper_arch.yaml
```

### Inference

```python
from d4rt.models import build_d4rt_model
from d4rt.inference import PointTracker, DepthReconstructor

# Load model
model = build_d4rt_model(config)
model.load_state_dict(torch.load("checkpoint.pth"))

# Track points
tracker = PointTracker(model)
trajectories = tracker.track_points(video, query_points)

# Reconstruct depth
depth_recon = DepthReconstructor(model)
depth = depth_recon.reconstruct(video, frame_idx=0)
```

### Evaluation

```bash
python scripts/evaluate.py \
    --config configs/model/vit_b_d4rt.yaml \
    --checkpoint checkpoints/checkpoint.pth \
    --data_dir data/kubric/val
```

## Project Structure

```
d4rt/
├── models/          # Model architectures
├── losses/          # Loss functions
├── inference/       # Tracking, depth, point cloud
├── data/            # Dataset loaders
└── training/        # Training loop

configs/             # Configuration files
scripts/             # Training and evaluation scripts
tests/               # Unit and integration tests
doc/                 # Detailed documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](doc/architecture.md) | Detailed model architecture (Figure 7) |
| [Training Guide](doc/training.md) | Training loop, loss functions, hyperparameters |
| [Inference Guide](doc/inference.md) | Point tracking, depth, pose estimation |
| [Implementation Notes](doc/implementation_notes.md) | Differences from paper, fixes, lessons learned |

## Loss Functions

This implementation combines the paper's loss formulation with practical improvements for depth learning.

### Loss Weights

| Loss | Weight | Description |
|------|--------|-------------|
| **3D L1** | 1.0 | Primary 3D position loss (DUSt3R-style normalization) |
| **2D L1** | 0.1 | Image-space coordinate loss |
| **Visibility** | 0.1 | Binary cross-entropy for occlusion |
| **Confidence** | 0.2 | Penalty term `-log(c)` for honest confidence |
| **Normal** | 0.5 | Surface normal cosine loss |
| **Motion** | 0.1 | Temporal motion consistency |
| **Depth** | 1.0 | Direct L1 depth loss (see below) |

### 3D Loss Normalization

Two normalization modes are available (configurable via `norm_mode`):

| Mode | Formula | Description |
|------|---------|-------------|
| `dust3r` (default) | Joint normalization by combined 3D distance | Both pred and GT normalized by same scale factor |
| `paper` | `pred / pred_mean`, `gt / gt_mean` + log transform | Paper's scale-invariant formulation |

### Direct Depth Loss (Key Addition)

To prevent **depth variance collapse** (model predicting near-constant depth), we add a direct L1 depth loss:

```
L_depth = λ_depth * |pred_z - gt_z|
```

| Property | Description |
|----------|-------------|
| **Weight** | 1.0 (configurable via `depth` in loss config) |
| **Not scale-invariant** | Provides absolute depth supervision |
| **Prevents variance collapse** | Penalizes when predictions cluster near mean |

This is critical because scale-invariant losses alone allow the model to minimize loss by predicting all depths near the mean value.

## Model Variants

| Model | Parameters | Config |
|-------|------------|--------|
| ViT-B | ~230M | `configs/model/vit_b_d4rt.yaml` |
| ViT-L | ~451M | `configs/model/vit_l.yaml` |
| ViT-g | ~1.1B | `configs/model/vit_g.yaml` |

## Testing

```bash
pytest tests/ -v
```

## References

- **Paper**: [D4RT: Unified, Fast 4D Scene Reconstruction & Tracking](https://arxiv.org/html/2512.08924v1)
- **Project Page**: https://d4rt-paper.github.io/
- **Original Authors**: Google DeepMind

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.
