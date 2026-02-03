# D4RT Training

## Quick Start

```bash
python scripts/train.py --config configs/training/train_paper_arch.yaml
```

## Training Algorithm Overview

D4RT training teaches the model to predict 3D positions from video queries. The key insight is that by sampling diverse queries across space, time, and visibility conditions, the model learns a unified representation for all 4D vision tasks.

### Training Loop

Each training step follows this process:

1. **Encode Video** → The encoder processes the entire video once to create a global scene representation F
2. **Sample Queries** → Random queries are sampled with diverse spatial/temporal/visibility conditions
3. **Predict 3D** → The decoder predicts 3D positions for each query by attending to F
4. **Compute Loss** → Multiple losses supervise different aspects of the prediction
5. **Update Weights** → Backpropagate and update model parameters

### Query Sampling Strategy

Queries are sampled with a strategic distribution to ensure the model learns all scenarios:

| Query Type | Ratio | Purpose |
|------------|-------|---------|
| Visible points | 50% | Points with ground truth 3D - primary supervision |
| Occluded points | 25% | Points behind objects - teaches occlusion reasoning |
| Random points | 25% | Uniform sampling - ensures spatial coverage |

For each query, temporal coordinates (t_src, t_tgt, t_cam) are sampled uniformly across all frames.

### Loss Functions

The total loss combines multiple supervision signals:

| Loss | Weight | Description |
|------|--------|-------------|
| **L1 3D** | 1.0 | Primary loss on 3D position (normalized by mean depth, log-transformed) |
| **L2 2D** | 0.1 | Reprojection error in image space |
| **Normal** | 0.5 | Cosine similarity of surface normals |
| **Motion** | 0.1 | Temporal consistency of motion |
| **Visibility** | 0.1 | Binary cross-entropy for occlusion prediction |
| **Confidence** | 0.2 | Penalty term `-log(c)` to encourage honest confidence |

**Confidence Weighting**: The 3D loss is weighted by predicted confidence `c`. This teaches the model to output high confidence for accurate predictions and low confidence for uncertain ones. See [Implementation Notes](implementation_notes.md#confidence-warmup) for details on the warmup schedule.

### 3D Loss Normalization

To handle varying scene scales and distances, the 3D loss applies two transforms:

1. **Mean depth normalization**: Divide positions by the mean ground truth depth
2. **Signed-log transform**: `sign(x) · log(1 + |x|)` to dampen influence of far points

This makes the loss scale-invariant and prevents distant points from dominating.

## Training Configuration

Key hyperparameters (from paper):

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.03 |
| Batch size | 1 (with gradient accumulation 8) |
| Gradient clipping | Max L2-norm of 10 |
| Warmup steps | 2500 |
| Total steps | 50000 |
| Mixed precision | FP16 |

## Checkpoints

Checkpoints are saved every 5000 steps to `checkpoints/`. Each checkpoint contains:
- Model weights
- Optimizer state
- Current step
- Training configuration

## Monitoring

Training progress can be monitored via:
- Console output (loss values every 100 steps)
- WandB logging (if enabled in config)
- Validation metrics (every 2500 steps)

## Tips

1. **Start with ViT-B**: Faster iteration, good for debugging
2. **Use gradient checkpointing**: Enables larger batch sizes
3. **Monitor confidence**: If mean confidence drops below 0.5, the model may be exploiting the loss (see implementation notes)
4. **Check scale ratio**: `pred_std / gt_std` should be > 0.2 for healthy learning
