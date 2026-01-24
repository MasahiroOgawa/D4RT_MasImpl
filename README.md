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

```bash
# Clone the repository
git clone https://github.com/MasahiroOgawa/D4RT_MasImpl.git
cd D4RT_MasImpl

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Install dependencies with uv
uv pip install -r requirements.txt

# Install the package in development mode
uv pip install -e .
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

MIT License - see LICENSE file for details.
