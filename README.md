# D4RT: Dynamic 4D Reconstruction and Tracking

This is an implementation of Google DeepMind's [D4RT](https://d4rt-paper.github.io/) - a unified transformer model for 4D scene reconstruction from video.

## Overview

D4RT is a single model that can:
- **Track points** through video sequences
- **Reconstruct dense depth** maps
- **Estimate camera poses** between frames
- **Perform novel view synthesis**

## Architecture

- **Encoder**: Spatio-temporal ViT (ViT-B/L/g) processes 48 video frames
- **Decoder**: 8-layer cross-attention transformer queries the scene representation
- **Query System**: 5-tuple queries (u, v, t_src, t_tgt, t_cam) enable flexible inference

### Differences from Original D4RT Paper

This implementation includes modifications to improve training stability when fine-tuning from scratch or with pretrained weights:

| Component | Original D4RT | This Implementation |
|-----------|---------------|---------------------|
| Encoder initialization | VideoMAE pretrained weights | Supports both pretrained and random init |
| Patch embedding normalization | None (relies on pretrained scale balance) | **LayerNorm after patch embedding** |
| Positional embedding | Learned, std=0.02 | Learned, std=0.5 (scaled for patch norm) |
| Context pooling | Not mentioned | Optional (3072 → 128 tokens) |
| Q/K LR multiplier | Not mentioned | Optional (10× for cross-attention) |
| Encoder LR multiplier | Not mentioned | Optional (for preserving pretrained weights) |

#### Why LayerNorm After Patch Embedding?

The original D4RT uses VideoMAE pretrained weights, where the patch embedding and positional embedding scales are already balanced from self-supervised pretraining. When training from scratch or fine-tuning:

1. **Problem**: Conv3d patch embedding produces features with large norm (~100), while positional embeddings are small (~0.6). This 170× scale mismatch causes:
   - Positional information overwhelmed by content features
   - All encoder tokens become nearly identical (99.97% cosine similarity)
   - Model cannot distinguish spatial locations → poor tracking

2. **Solution**: Adding LayerNorm after patch embedding normalizes features, and we scale up positional embedding initialization to match:

| Component | Before Fix | After Fix |
|-----------|------------|-----------|
| Patch features norm | ~100 | ~27.7 (LayerNorm) |
| Positional embed norm | ~0.6 (std=0.02) | ~13.9 (std=0.5) |
| Scale ratio | 170× | 2× |
| Token cosine similarity | 0.9997 | 0.012 |

```
Original D4RT:  PatchEmbed → + PosEmbed(std=0.02) → Transformer
This impl:      PatchEmbed → LayerNorm → + PosEmbed(std=0.5) → Transformer
```

To disable this (for strict paper reproduction with pretrained weights):
```yaml
encoder:
  use_patch_norm: false  # Disable LayerNorm, use std=0.02 for pos_embed
```

#### VideoMAE Pretrained Weights

To use VideoMAE pretrained weights (recommended for best results):

```python
from d4rt.models.encoder import load_videomae_weights

encoder = build_vit_encoder(config)
load_videomae_weights(encoder, "path/to/videomae_vit_b.pth")
```

Download pretrained weights from: https://github.com/MCG-NJU/VideoMAE

#### Encoder Learning Rate Multiplier

When using pretrained weights, use a lower learning rate for the encoder to preserve learned features:

```yaml
optimizer:
  lr: 1e-4
  encoder_lr_multiplier: 0.1  # Encoder LR = 1e-5
  qk_lr_multiplier: 10.0      # Cross-attention Q/K LR = 1e-3
```

This creates three parameter groups:
- **Encoder**: 0.1× base LR (preserves pretrained features)
- **Cross-attention Q/K**: 10× base LR (faster attention learning)
- **Other parameters**: 1× base LR (default)

## Project Structure

```
d4rt/
├── models/          # Model architectures (encoder, decoder, components)
├── data/            # Dataset loaders and preprocessing
├── losses/          # Loss functions
├── training/        # Training loop and utilities
├── inference/       # Inference modules (tracking, depth, pose)
└── utils/           # Helper functions

configs/             # Model and training configurations
scripts/             # Training and inference scripts
tests/               # Unit tests
doc/                 # Documentation and flowcharts
```

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/MasahiroOgawa/D4RT_MasImpl.git
cd D4RT_MasImpl

# Sync dependencies and create virtual environment (one command!)
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Install in development mode
uv pip install -e .
```

### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .
```

## Quick Start

### Training (ViT-B on Kubric)

```bash
python scripts/train.py --config configs/training/debug.yaml
```

### Inference

```bash
# Point tracking
python scripts/infer_tracking.py \
    --checkpoint checkpoints/model.pth \
    --video data/video.mp4

# Depth reconstruction
python scripts/infer_depth.py \
    --checkpoint checkpoints/model.pth \
    --video data/video.mp4 \
    --frame 0
```

## Documentation

- [Training Specifications](doc/training.md)
- [Inference Guide](doc/inference.md)
- [Architecture Flowchart](doc/flowchart_architecture.md)
- [Training Flowchart](doc/flowchart_training.md)
- [Inference Flowchart](doc/flowchart_inference.md)
- [Data Pipeline](doc/flowchart_data_pipeline.md)

## Model Configurations

| Model | Layers | Hidden Dim | Parameters | Training Time |
|-------|--------|------------|------------|---------------|
| ViT-B | 12 | 768 | ~230M | ~1 week |
| ViT-L | 24 | 1024 | ~451M | ~2 weeks |
| ViT-g | 40 | 1408 | ~1.144B | ~3-4 weeks |

## References

- **Paper**: [D4RT: Unified, Fast 4D Scene Reconstruction & Tracking](https://arxiv.org/html/2512.08924v1)
- **Website**: https://d4rt-paper.github.io/
- **Original work**: Google DeepMind


## License

This project is licensed under the GNU General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.
