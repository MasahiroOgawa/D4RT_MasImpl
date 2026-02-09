# D4RT Training

## Quick Start

```bash
# Start training with tmux (recommended)
./scripts/train_tmux.sh train_50k_movi_paper

# Or run directly
uv run python scripts/train.py --config-name train_50k_movi_paper --config-path ../configs/training
```

## Training Algorithm Overview

D4RT training teaches the model to predict 3D positions from video queries. The key insight is that by sampling diverse queries across space, time, and visibility conditions, the model learns a unified representation for all 4D vision tasks.

### Training Loop

Each training step follows this process:

1. **Encode Video** - The encoder processes the entire video once to create a global scene representation F
2. **Sample Queries** - Random queries are sampled with diverse spatial/temporal/visibility conditions
3. **Predict 3D** - The decoder predicts 3D positions for each query by attending to F
4. **Compute Loss** - Multiple losses supervise different aspects of the prediction
5. **Update Weights** - Backpropagate and update model parameters

### Query Sampling Strategy

Queries are sampled with a strategic distribution to ensure the model learns all scenarios:

| Query Type | Ratio | Purpose |
|------------|-------|---------|
| Visible points | 50% | Points with ground truth 3D - primary supervision |
| Occluded points | 25% | Points behind objects - teaches occlusion reasoning |
| Random points | 25% | Uniform sampling - ensures spatial coverage |

For each query, temporal coordinates (t_src, t_tgt, t_cam) are sampled uniformly across all frames.

## Loss Functions

The total loss combines multiple supervision signals:

| Loss | Weight | Description |
|------|--------|-------------|
| **L1 3D** | 1.0 | Primary 3D loss (normalized by mean depth, signed-log transform) |
| **L2 2D** | 0.1 | Reprojection error in image space |
| **Normal** | 0.5 | Cosine similarity of surface normals |
| **Motion** | 0.1 | Temporal consistency of motion |
| **Visibility** | 0.1 | Binary cross-entropy for occlusion prediction |
| **Confidence** | 0.2 | Penalty term `-log(c)` to encourage honest confidence |
| **Depth** | 10.0 | Direct L1 depth loss `|pred_z - gt_z|` |

### 3D Loss Normalization (Paper Formula)

The paper's 3D loss uses scale-invariant normalization:

1. **Mean depth normalization**: Divide positions by the mean ground truth depth
2. **Signed-log transform**: `sign(x) * log(1 + |x|)` to dampen influence of far points

This makes the loss scale-invariant and prevents distant points from dominating.

### Direct Depth Loss

To address depth variance collapse (where the model predicts all depths near the mean), a direct L1 depth loss is added:

```
L_depth = λ_depth * |pred_z - gt_z|
```

With `λ_depth = 10.0`, this strongly penalizes absolute depth errors and encourages the model to learn the full depth distribution.

## Training Configuration

Current configuration (`configs/training/train_50k_movi_paper.yaml`):

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.03 |
| Batch size | 1 (with gradient accumulation 4) |
| Effective batch size | 4 |
| Gradient clipping | Max L2-norm of 10 |
| Warmup steps | 2500 |
| Total steps | 50000 |
| Precision | FP32 |
| Gradient checkpointing | Enabled |

### Pretrained Weights

VideoMAE pretrained encoder is critical for performance (paper ablation Table 11):
- Random init: 0.738 depth error
- VideoMAE pretrained: 0.302 depth error

Pretrained weights: `pretrained/videomae_vit_b_k400_800e.pth`

## Checkpoints

Checkpoints are saved to `checkpoints/`:
- `checkpoint_step_XXXXXX.pth` - Every 5000 steps
- `checkpoint_latest.pth` - Symlink to latest checkpoint
- `best_metrics.json` - Best validation metrics

Each checkpoint contains:
- Model weights
- Optimizer state
- Scheduler state
- Current step
- Training metrics

## Monitoring

### Tmux Training (Recommended)

The `train_tmux.sh` script provides integrated monitoring:

```bash
./scripts/train_tmux.sh train_50k_movi_paper
```

Windows:
- **training** - Training process output
- **eval** - Background evaluation (every 1000 steps)
- **battery** - Battery monitor
- **status** - Quick status view

### Manual Monitoring

```bash
# Watch training log
tail -f outputs/training_*.log

# Watch evaluation results
tail -f outputs/eval_monitor_*.log

# Check latest checkpoint
ls -lt checkpoints/*.pth | head -3
```

### Key Metrics to Watch

| Metric | Healthy Range | Problem Sign |
|--------|---------------|--------------|
| pred_z_std / gt_z_std | > 0.5 | < 0.3 = variance collapse |
| Z correlation | > 0.5 | < 0.3 = poor depth learning |
| AJ (Average Jaccard) | increasing | stuck < 0.01 |
| Mean confidence | 0.3 - 0.8 | < 0.3 or > 0.95 |

## Tips

1. **Use VideoMAE pretrained weights** - Critical for good performance
2. **Start with ViT-B** - Faster iteration, good for debugging
3. **Use gradient checkpointing** - Enables larger batch sizes on limited VRAM
4. **Monitor depth variance** - `pred_z_std / gt_z_std` should be > 0.5
5. **Check Z correlation** - Should improve steadily during training
6. **Use tmux training** - Survives terminal close, includes monitoring
