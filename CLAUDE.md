# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

D4RT (Dynamic 4D Reconstruction and Tracking) - Implementation of Google DeepMind's unified transformer model for 4D scene reconstruction from video. Single model handles point tracking, dense tracking, depth estimation, point clouds, and long-term prediction.

## Development Commands

```bash
# Environment setup (always use uv, not pip)
uv sync
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run single test file
pytest tests/unit/test_encoder.py -v

# Run specific test
pytest tests/unit/test_encoder.py::TestEncoder::test_forward -v

# Code formatting
black d4rt/ tests/
isort d4rt/ tests/

# Training (using Hydra config)
uv run python scripts/train.py --config-name train_50k_movi_paper

# Quick evaluation
uv run python scripts/quick_eval.py --checkpoint checkpoints/checkpoint_latest.pth

# Monitor training logs
tail -f outputs/training_*.log
```

## Architecture

**Model Pipeline:**
```
Video [B, T=24, 3, 256, 256]
    → Encoder (SpatioTemporalViT, 12 layers, 768D)
    → Global Scene Representation F [B, 3072, 768]
    → Query Encoder (spatial + temporal + appearance → 512D)
    → Decoder (CrossAttention, 8 layers)
    → Output Heads (xyz, uv, visibility, confidence, normals, motion)
```

**Query System:** 5-tuple `(u, v, t_src, t_tgt, t_cam)` enables multiple tasks:
- Point tracking: fix (u,v,t_src), vary t_tgt
- Depth reconstruction: vary (u,v) grid, fix all t
- Dense tracking: vary (u,v) and t_tgt

**Inference Design:** Encode once (expensive) → query many times (cheap)

## Loss Function

7-component composite loss with critical fixes:

| Loss | Weight | Notes |
|------|--------|-------|
| 3D L1 | 1.0 | DUSt3R-style joint normalization |
| 2D L1 | 0.1 | Image-space coordinates |
| Visibility | 0.1 | BCE for occlusion |
| Confidence | 0.2 | `-log(c)` penalty |
| Normal | 0.5 | Cosine similarity |
| Motion | 0.1 | Temporal consistency |
| Depth | 1.0 | Direct L1 (prevents variance collapse) |

**Confidence Warmup:** Linear interpolation over 25k steps to prevent model exploiting low confidence.

## Key Directories

- `d4rt/models/` - Encoder (VideoMAE-based ViT), Decoder, output heads
- `d4rt/losses/` - All loss components including L1_3D with normalization
- `d4rt/training/` - Trainer, checkpointing, optimizer setup
- `d4rt/inference/` - Point tracking, depth, pose estimation
- `configs/training/` - Hydra YAML configs (train_50k_movi_paper.yaml is main)
- `configs/model/` - Model architecture configs (vit_b_movi.yaml)
- `scripts/` - All executable scripts (train, evaluate, inference, debug)

## Repository Conventions

- **Keep top directory clean** - Never place scripts in root; always use `scripts/` subdirectory
- **Use uv** for all Python package management, not pip
- **Hydra configs** in `configs/` - override with `--config-name` or `key=value`
- **Checkpoints** saved to `checkpoints/` with symlinks for `_latest` and `_best`
- **Logs** written to `outputs/` with timestamps

## Current Training Status

Metrics being tracked:
- **AJ** (Average Jaccard): Overall tracking accuracy
- **APD3D**: 3D position accuracy
- **OA** (Occlusion Accuracy): Visibility prediction
- **Z correlation**: Depth prediction quality (known issue: variance collapse)

## Known Issues & Fixes

1. **Depth variance collapse**: Model predicts near-constant depth. Fixed with direct L1 depth loss (λ=1.0)
2. **Confidence exploitation**: Model outputs low confidence to reduce weighted loss. Fixed with confidence warmup over 25k steps
3. **3D loss scale invariance**: Paper's normalization allows trivial solutions. Using DUSt3R-style joint normalization
4. **Depth +1 initialization**: Per author feedback (Feb 2026), add +1 to predicted Z values - "initialization would otherwise start at 0, hindering training dynamics"

## Author Feedback (Feb 2026)

From Mehdi Sajjadi (Google DeepMind):
- Scale is normalized during training (Sec. 2.3) → model predicts normalized values at inference
- Evaluations use standard procedures: **either matching scales, or scale-invariant metrics**
- Some regression-to-the-mean is expected since predictions are deterministic
- **Key trick**: Add +1 to estimated depth values since initialization starts at 0

## Evaluation Protocol

Following official TAP-Vid-3D benchmark (https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d):

**Scale Alignment (Global Median Rescaling):**
```
scale = median(||P_gt||) / median(||P_pred||)
pred_aligned = pred * scale
```
Where `||P|| = sqrt(x² + y² + z²)` is 3D Euclidean norm from origin.

**Training vs Evaluation:**
| Aspect | Training (Paper) | Evaluation (TAP-Vid-3D) |
|--------|------------------|------------------------|
| Normalization | Independent Z-mean | Joint 3D-norm median |
| Formula | `xyz / mean(z)` | `xyz * scale` |
| Transform | Signed log | None |

Both are correlated: lower training loss → lower evaluation error (~5x ratio).
