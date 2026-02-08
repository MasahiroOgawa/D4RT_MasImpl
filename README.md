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

This implementation follows the paper's loss formulation exactly, with one additional auxiliary loss:

### Original Paper Losses (Unchanged)

| Loss | Formula | Weight |
|------|---------|--------|
| **3D L1** | `L1(pred_xyz / pred_mean, gt_xyz / gt_mean)` | 1.0 |
| **2D L2** | `L2(pred_xy, gt_xy)` | 0.1 |
| **Visibility** | Binary cross-entropy | 0.1 |
| **Confidence** | Huber penalty for low confidence | 0.2 |

The 3D loss uses scale-invariant normalization: predictions are divided by their mean depth, and ground truth by its mean depth, ensuring the loss is independent of absolute scale.

### Auxiliary Log-Depth Loss (Added)

To help the model learn positive depth values more effectively, we add an auxiliary log-depth loss:

```
L_depth_aux = |log(clamp(pred_z, min=0.1)) - log(clamp(gt_z, min=0.1))|
```

| Property | Description |
|----------|-------------|
| **Weight** | 1.0 (configurable via `depth_aux` in loss config) |
| **Scale-invariant** | Depends only on ratio `pred_z / gt_z`, not absolute values |
| **Strong gradient** | Gives large gradient when pred_z is near zero or negative |

This auxiliary loss complements the paper's loss by providing stronger supervision for depth learning.

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
