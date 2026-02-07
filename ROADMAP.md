# D4RT Implementation Roadmap

**Goal**: Reproduce D4RT paper performance on TAPVid-3D benchmark.

## Target Metrics (Paper)

| Metric | Target |
|--------|--------|
| Average Jaccard (AJ) | 0.304 |
| APD3D | 0.410 |
| Occlusion Accuracy | 0.875 |

## Current Status

**Phase**: 14 in progress (Multi-dataset training)
**Training**: Step 70,487 / 100,000 (70% complete)
**Best Checkpoint**: Step 70,000

### Multi-Dataset Training Results (2026-02-07, Step 70k)

| Metric | Result | Target | % of Target |
|--------|--------|--------|-------------|
| **AJ** | 0.0298 | 0.304 | 9.8% |
| **APD3D** | 0.0323 | 0.410 | 7.9% |
| **OA** | 0.9242 | 0.875 | 106% |

**Observations**:
- Multi-dataset (Kubric 70% + PointOdyssey 30%) did NOT improve AJ
- AJ stuck at ~0.03 (same as Kubric-only at 50k)
- OA exceeds target, scale ratio OK (0.44)
- Loss ~0.06-0.10 (converged but not improving AJ)

### Previous 50k Results (Kubric-only)

| Metric | Result | Target | % of Target |
|--------|--------|--------|-------------|
| **AJ** | 0.0320 | 0.304 | 10.5% |
| **APD3D** | 0.0321 | 0.410 | 7.8% |
| **OA** | 0.9609 | 0.875 | 110% |

---

## Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1-8 | Core implementation (model, data, losses, training) | Done |
| 9-11 | Bug fixes (FP16 overflow, UV normalization) | Done |
| 12a | Download full MOVi-A (9,703 scenes) | Done |
| 12b | First 50k training (buggy GT) | Done |
| 13 | TAP-Vid metrics implementation | Done |
| 13.5 | Paper loss functions + confidence warmup | Done |
| GT Fix | Fix dataset to use tracks_3d directly | Done |
| 12c | Retrain 50k with fixed GT | Done |

---

## Current Phase

### Phase 14: Add Real-World Datasets (In Progress)

**Progress**:
- [x] Created `PointOdysseyDataset` loader (`d4rt/data/datasets/pointodyssey.py`)
- [x] Created `MultiDataset` with weighted sampling (`d4rt/data/datasets/multi_dataset.py`)
- [x] Updated training script to support multi-dataset
- [x] Created `train_multidataset.yaml` config
- [x] Downloaded PointOdyssey val split (14 scenes)
- [x] Training on Kubric + PointOdyssey val (step 64k → 70k+)
- [ ] Continue training to 100k steps (~8 hours remaining)
- [ ] Download PointOdyssey train split (~134GB)

---

## Investigation: Why AJ is Low (2026-02-07)

### Key Findings

#### 1. Model Fails to Learn Depth (Z coordinate)

Correlation between predictions and ground truth:
| Axis | Correlation |
|------|-------------|
| X | 0.834 (good) |
| Y | 0.461 (mediocre) |
| **Z (depth)** | **0.072** (almost random!) |

The model learns X/Y reasonably but **cannot predict depth**. This is the root cause of low AJ.

#### 2. Paper Uses Alignment During Evaluation

The D4RT paper applies **scale-and-shift alignment** before computing metrics:
> "determine a single global scale factor between the predicted and ground-truth depths"
> "first align the predicted and ground-truth point clouds via mean-shifting"

Our evaluation computes AJ on raw predictions without alignment.

#### 3. Our Loss Normalization Differs from Paper

| Aspect | Paper | Our Implementation |
|--------|-------|-------------------|
| 3D normalization | `pred / pred_mean`, `gt / gt_mean` | Both by `gt_mean` |
| Scale signal | None (fully scale-invariant) | Yes |

Paper formula: Both normalized by **their respective** mean depths (scale-invariant).

#### 4. AJ Metric Definition

```
AJ = TP / (TP + FP + FN)

where:
  error_i = ||pred_i - gt_i||_2  (Euclidean distance in meters)
  within_thresh_i = (error_i < 0.5m)

  TP = points predicted visible AND within 0.5m AND actually visible
  FP = points predicted visible but (not within 0.5m OR not visible)
  FN = points actually visible but not predicted visible
```

---

## Next Steps: Fix Evaluation & Loss

### Phase 14.1: Match Paper's Protocol (TODO)

1. **Fix loss normalization** (`d4rt/losses/composite_loss.py`):
   ```python
   # Current (wrong):
   pred_norm = pred_xyz / (gt_mean_depth + 1e-8)
   gt_norm = gt_xyz / (gt_mean_depth + 1e-8)

   # Paper (correct):
   pred_norm = pred_xyz / (pred_xyz[..., 2:3].mean() + 1e-8)
   gt_norm = gt_xyz / (gt_xyz[..., 2:3].mean() + 1e-8)
   ```

2. **Add alignment to evaluation** (`d4rt/evaluation/metrics.py`):
   - Scale alignment: `pred_aligned = pred * (gt_mean / pred_mean)`
   - Scale+shift: `pred_aligned = (pred - pred_mean) * scale + gt_mean`

3. **Investigate depth learning failure**:
   - Check encoder features for depth information
   - Verify query encoding includes depth cues
   - Check if decoder can map features to depth

### Phase 14.2: Debug Depth Prediction (COMPLETED)

**Root Cause Found**: Depth signal is lost in decoder layers and XYZ head.

#### Depth Correlation Through Network (Sample 0)

```
Layer         | Depth Correlation (PC1 vs GT_Z)
--------------|--------------------------------
query_proj    | +0.582  ← Good depth info exists!
layer_0-2     | ~0.54   ← Maintained
layer_3-4     | ~0.45   ← Starting to degrade
layer_5-7     | ~0.27   ← Lost 54% through cross-attention
xyz_head (Z)  | +0.106  ← Lost 61% more in linear layer
```

**Total: 82% of depth signal lost!**

#### Key Insights

1. **Query embedding HAS depth info** (corr=0.58) - Fourier encoding of (u,v) + patches capture some depth cues
2. **Decoder layers LOSE depth** (0.58 → 0.27) - Cross-attention dilutes depth-specific features
3. **XYZ head is MISALIGNED** (0.27 → 0.11) - Linear weights not extracting depth dimensions

#### Paper Query Encoding (Verified)

Paper confirms queries are 2D: `q = (u, v, t_src, t_tgt, t_cam)` - same as our implementation.
The model should infer depth from monocular cues, but decoder fails to preserve this signal.

#### Hypotheses for Fix

1. **Decoder layers**: Cross-attention may be mixing depth info with other features
2. **XYZ head**: Simple linear layer may need depth-specific architecture
3. **Loss signal**: Z gradient may be too weak compared to X/Y
4. **Scale mismatch**: Using paper's normalization (pred/pred_mean) may help

### Phase 14.3: Implementation Fixes (DONE - 2026-02-07)

1. **Match paper's loss normalization** ✓
   - File: `d4rt/losses/composite_loss.py`
   - Changed from: both normalized by `gt_mean_depth`
   - Changed to: `pred / pred_mean_depth`, `gt / gt_mean_depth`
   - Loss is now **scale-invariant** (matches paper)

2. **Add scale-and-shift alignment to evaluation** ✓
   - File: `d4rt/evaluation/metrics.py`
   - Added `align_predictions()` function with modes: none, scale, shift, scale_shift
   - Updated `compute_tapvid_metrics()` to use alignment (default: scale_shift)
   - File: `scripts/quick_eval.py` - added `--alignment` argument

**Test Results (checkpoint 70k, trained with OLD loss):**
| Alignment | AJ |
|-----------|-----|
| none | 0.030 |
| scale_shift | 0.014 (worse - no correct structure) |

**Next Step:** Retrain with new scale-invariant loss. The model should learn better relative structure when not penalized for scale mismatch.

### Phase 14.4: Retrain with Paper Loss (TODO)

1. Stop current training
2. Start fresh training from step 0 (or resume from earlier checkpoint before multi-dataset)
3. Use new scale-invariant loss
4. Evaluate with scale_shift alignment
5. Expected: Better AJ after alignment

**Architectural improvements** (if retraining doesn't help):
- Separate depth head with dedicated features
- Depth-aware attention mechanism
- Auxiliary depth loss

---

## Remaining Phases

### Phase 15: Large-Scale Training

- Train 100k-200k steps on multi-dataset mix
- Match paper's training recipe

### Phase 16: Final Evaluation

Run on official benchmarks:
- TAPVid-3D (DriveTrack, ADT, PStudio)
- TAP-Vid-DAVIS (2D tracking)

---

## Quick Reference

```bash
# Multi-dataset training (Kubric + PointOdyssey)
uv run python scripts/train.py --config-name=train_multidataset \
    +training.resume_from=checkpoints/checkpoint_step_0050000.pth

# Single-dataset training (Kubric only)
uv run python scripts/train.py --config-name=train_50k_movi_paper \
    +training.resume_from=checkpoints/checkpoint_latest.pth

# Quick evaluation (10 scenes)
uv run python scripts/quick_eval.py --checkpoint checkpoints/checkpoint_latest.pth

# Download PointOdyssey
uv run python scripts/download_pointodyssey.py --split val   # ~20GB
uv run python scripts/download_pointodyssey.py --split train # ~134GB
```

---

## Timeline Estimate

| Phase | Duration |
|-------|----------|
| 12c (current 50k training) | ~10 hours |
| 14 (add datasets) | 3-5 days |
| 15 (large-scale training) | 2-4 weeks |
| 16 (final evaluation) | 1-2 days |

**Total**: 4-6 weeks for full reproduction

---

## References

- [D4RT Paper](https://arxiv.org/html/2512.08924v1)
- [Project Page](https://d4rt-paper.github.io/)
- [TAPVid-3D Benchmark](https://tapvid3d.github.io/)
