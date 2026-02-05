# D4RT Implementation Summary

Implementation of Google DeepMind's D4RT for 4D scene reconstruction from video.

## Architecture

| Component | Description | Parameters |
|-----------|-------------|------------|
| Encoder | Spatio-temporal ViT (3D Conv patching) | 174M |
| Query Encoder | Fourier + temporal + patch CNN | 0.7M |
| Decoder | 8-layer cross-attention transformer | 42M |
| **Total** | ViT-B configuration | **217M** |

## Key Features

- **Input**: 24 frames @ 256x256, queries (u, v, t_src, t_tgt, t_cam)
- **Output**: 3D position (xyz), visibility, confidence
- **Training**: AdamW, cosine LR with warmup, gradient checkpointing

## Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| L1 3D | 1.0 | Primary 3D supervision (mean depth norm + signed-log) |
| L2 2D | 0.1 | Reprojection error |
| Visibility | 0.1 | Occlusion BCE |
| Motion | 0.1 | Temporal consistency |
| Normal | 0.5 | Surface normal alignment |
| Confidence | 0.2 | Uncertainty estimation |

## Project Structure

```
d4rt/
├── models/       # Encoder, decoder, query encoder
├── losses/       # Multi-task loss functions
├── data/         # Dataset loaders (Kubric)
├── training/     # Trainer, optimizer, checkpointing
├── inference/    # Tracking, depth, pose estimation
└── evaluation/   # TAP-Vid metrics

scripts/          # Training, evaluation, debugging
configs/          # Model and training configs
doc/              # Documentation
```

## Usage

```bash
# Train
uv run python scripts/train.py --config-name=train_50k_movi_paper

# Evaluate
uv run python scripts/quick_eval.py --checkpoint checkpoints/checkpoint_latest.pth

# Inference
uv run python scripts/infer_tracking.py --checkpoint model.pth --video input.mp4
```

## Key Implementation Notes

See `doc/implementation_notes.md` for:
- Confidence warmup (prevents low-confidence exploitation)
- UV coordinate normalization fix
- Paper-exact 3D loss formula
