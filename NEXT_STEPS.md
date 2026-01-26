# D4RT Implementation: Path to Paper-Level Performance

## 🎯 FINAL GOAL
**Reproduce D4RT paper performance on public benchmarks as closely as possible**

**Target Metrics (TAPVid-3D benchmark):**
- Average Jaccard (AJ): **0.304**
- Average Percent within Delta 3D (APD3D): **0.410**
- Occlusion Accuracy (OA): **0.875-0.897**
- Speed: **18-300× faster than prior methods**

**Current Status:**
- ✅ Phase 1-12b completed: 50k training on full MOVi-A (9,703 samples) finished
- 📊 Current performance: Validation loss improved 36%, tracking quality shows moderate improvements
- 📍 **We are here**: Ready for Phase 13 (TAP-Vid evaluation) or Phase 14-15 (more data/training)
- 🎯 Remaining: Phases 13-16 to reach paper performance

---

## Overall Progress Tracking

**Phases 1-11: Foundation & Bug Fixes** ✅ COMPLETED
- Build training pipeline
- Fix critical FP16 bug
- Fix 2D projection loss
- Verify all components working

**Phases 12-16: Scale to Paper Performance** 🚧 IN PROGRESS
- ✅ Scale up training (50k steps on full MOVi-A)
- 📋 Implement proper evaluation metrics (Phase 13)
- 📋 Add real-world datasets (Phase 14)
- 📋 Large-scale multi-dataset training (Phase 15)
- 📋 Match paper benchmarks (Phase 16)

---

## ✅ COMPLETED: Phase 11 - Fix 2D Projection Loss & Full Training

**Training completed successfully with all losses:**
- ✅ Fixed FP16 overflow bug (switched to FP32)
- ✅ Fixed 2D projection loss overflow (UV clamping)
- ✅ Fixed checkpointing symlink bug
- ✅ Trained 5000 steps with full loss function (3D + 2D + motion + visibility)
- ✅ Final validation loss: 2.75 (converged)
- ✅ No inf/nan issues throughout training

**Results:**
- Model is learning (losses decreasing)
- Tracking quality still poor (minimal movement ~0.08 units/frame)
- Need longer training (50k+ steps) and more diverse data

---

## ✅ COMPLETED: Phase 10 - Critical Bug Fix & First Successful Training

**MAJOR BREAKTHROUGH** - Identified and fixed critical FP16 bug preventing all learning!

### 🔴 Critical Bug Found: FP16 Overflow

**The Problem:**
- Previous 5k training was completely broken - model never learned
- FP16 mixed precision caused numerical overflow to inf/nan
- Gradients were NaN, model weights frozen
- Tracking was completely non-functional (all points in corner)

**The Fix:**
- Disabled FP16 → Use FP32
- Disabled problematic losses (2D projection, motion, visibility, normal)
- Reduced learning rate 10x (1e-4 → 1e-5)
- Stronger gradient clipping (1.0 → 0.5)

**Results After Fix:**
- ✅ Training loss: 146.07 → 2.43 (98.3% reduction!)
- ✅ Validation loss: 5.05 → 2.56 (stable convergence)
- ✅ No inf/nan throughout 5000 steps
- ✅ Model actually learning for the first time
- ⚠️ Tracking still poor (points don't follow objects)

### Current Status

**Training completed successfully:**
- Duration: 59 minutes
- Checkpoints: 6 saved (every 1000 steps)
- Final checkpoint: `checkpoints/checkpoint_step_0005000.pth`
- Log: `logs/train_fp32.log`

**Tracking quality: POOR**
- Points have high visibility (99.9%)
- But minimal movement (0.06-0.23 units)
- Do NOT track actual objects
- Need 2D supervision for proper tracking

---

## ✅ COMPLETED: Phase 12b - 50k Training on Full MOVi-A Dataset

**Training Summary:**
- ✅ Dataset: Full MOVi-A (9,703 training samples, 70 GB)
- ✅ Training completed: 50,000 steps in 14h 40min
- ✅ Final validation loss: **1.77** (down from 2.75 at 5k steps)
- ✅ Improvement: **36% reduction in validation loss**
- ✅ Training stability: No OOM errors, gradient checkpointing working

**Tracking Quality Comparison (5k vs 50k):**

Inference was run on the same validation video with both checkpoints:

| Metric | 5k Checkpoint | 50k Checkpoint | Change |
|--------|---------------|----------------|--------|
| Validation Loss | 2.75 | 1.77 | -36% ✅ |
| Point 0 Movement | 0.156 | 0.317 | +103.5% ✅ |
| Point 1 Movement | 0.036 | 0.059 | +63.2% ✅ |
| Point 2 Movement | 0.289 | 0.363 | +25.6% ✅ |
| Point 3 Movement | 0.047 | 0.042 | -10.6% ⚠️ |
| Point 4 Movement | 0.048 | 0.040 | -16.5% ⚠️ |
| Temporal Consistency | 0.0828 | 0.0249 | -69.9% ⚠️ |

**Assessment:**
- ✅ Clear learning signal: Validation loss decreased significantly
- ⚠️ Mixed tracking improvements: Some points track much better (+103%), others slightly worse
- ⚠️ Reduced temporal dynamics: Points move less between frames (smoother but potentially less responsive)
- 📊 Need proper benchmarking: TAP-Vid metrics required for objective evaluation

**Recommendations:**
1. **Phase 13** (Immediate): Evaluate with official TAP-Vid-3D metrics to get objective performance numbers
2. **Phase 14** (Next): Add real-world datasets (PointOdyssey, Co3Dv2) for better generalization
3. **Phase 15** (After): Continue training to 100k-200k steps with multi-dataset mix

**Files:**
- Checkpoint: `checkpoints/checkpoint_step_0050000.pth` (1.5 GB)
- Results: `results/tracking_50k.npz`
- Visualization: `results/tracking_50k.mp4`
- Log: `logs/train_50k_movi.log`

---

## 🎯 Phase 12c-16: Continue Path to Paper Performance (CURRENT)

**Goal:** Match D4RT paper performance on public benchmarks (TAPVid-3D).

**Paper targets:**
- TAPVid-3D AJ: 0.304
- APD3D: 0.410
- OA: 0.875-0.897

**See detailed roadmap:** `ROADMAP_TO_PAPER_PERFORMANCE.md`

**Next immediate steps:**
1. ✅ ~~Download full MOVi-A dataset (9,703 samples)~~ DONE
2. ✅ ~~Train for 50k steps on full dataset~~ DONE
3. 📋 **Phase 13**: Implement TAP-Vid evaluation metrics
4. 📋 **Phase 14**: Add real-world datasets (PointOdyssey, Co3Dv2, ScanNet)
5. 📋 **Phase 15**: Large-scale multi-dataset training (100k-200k steps)

### What's Working:

1. ✅ **uv sync workflow** - Dependency management with lockfile
2. ✅ **MOVi-A dataset downloaded** - 100 train + 20 val samples from `gs://kubric-public/tfds`
3. ✅ **Data conversion** - MOVi format → D4RT Kubric format converter working
4. ✅ **Training on real data** - 5000 steps completed successfully
5. ✅ **All components verified** - Data loading, model forward/backward, loss, validation
6. ✅ **Memory optimizations** - Gradient checkpointing enabled, no OOM errors

### Current Setup:

```bash
# Full 5k training (completed)
python scripts/train_simple.py \
    --model-config configs/model/vit_b_movi.yaml \
    --training-config configs/training/train_5k_movi.yaml \
    --data-dir data/kubric

# Current dataset: 100 MOVi-A training scenes (24 frames @ 256x256 each)
```

### Training Results (5000 steps):
- **Final validation loss**: 295.92
- **Checkpoint saved**: `checkpoints/checkpoint_step_0005000.pth` (503 MB)
- **Training speed**: ~2.4 it/s on GPU (11.6 GB VRAM)
- **Memory usage**: Stable with gradient checkpointing enabled
- **Issues**: Some numerical instability (inf losses) around step 4500, but recovered

---

## 🎯 Phase 10: Inference Testing & Validation

**Priority**: Test the trained model before scaling up further.

Now that we have a trained checkpoint, the next step is to **verify it actually works** for point tracking before investing time in larger-scale training.

### Phase 10 Tasks:

#### ✅ 10a. Inference Script Working (COMPLETED)
Successfully tested point tracking on validation video:

```bash
# Run inference on validation scene
source .venv/bin/activate
python scripts/infer_tracking.py \
    --checkpoint checkpoints/checkpoint_step_0005000.pth \
    --model-config configs/model/vit_b_movi.yaml \
    --video data/val_scene_00000.mp4 \
    --points "[[0.3,0.3],[0.5,0.5],[0.7,0.3],[0.3,0.7],[0.7,0.7]]" \
    --num-frames 24 \
    --resolution 256 \
    --output results/tracking_test.npz \
    --visualize results/tracking_test.mp4
```

**Results:**
- ✅ Model loads from checkpoint correctly
- ✅ Can process validation videos
- ✅ Generated trajectories: 5 points × 24 frames
- ✅ Visibility scores: 60% visible (threshold > 0.5)
- ✅ Visualization video created: `results/tracking_test.mp4`

**Fixes applied:**
- Added `--model-config` argument (checkpoint doesn't store full model config)
- Fixed batch dimension handling in tracker
- Removed unsupported `encoder_features` parameter

#### 10b. Quantitative Evaluation
Compare against validation set metrics:
- Average Position Error (APE)
- Occlusion accuracy
- Survival curves (what % of tracks survive over time)

#### 10c. Visual Inspection
Generate videos with overlaid tracks to qualitatively assess:
- Do tracks follow object motion?
- How well does it handle occlusions?
- Are there failure cases?

**Expected time**: 1-2 hours

---

## Alternative: Scale Up Training & Data (Path 1-3)

If you want to train longer before testing inference, you have 3 paths forward (in priority order):

---

### Path 1: Train ViT-B on Larger MOVi-A Dataset (⭐ Recommended First)

Download more MOVi-A samples and train for longer to verify the full pipeline.

```bash
# Step 1: Download full MOVi-A training set (9,703 samples)
# NOTE: This will take ~3-4 hours and require ~50GB disk space
python scripts/download_movi.py \
    --dataset movi_a \
    --split train \
    --output-dir data/movi_raw

# Step 2: Convert to D4RT format (will take ~3-4 hours)
python scripts/convert_movi_to_kubric.py \
    --dataset movi_a \
    --split train \
    --output-dir data/kubric

# Step 3: Download validation set (250 samples, ~5 min)
python scripts/download_movi.py \
    --dataset movi_a \
    --split validation \
    --output-dir data/movi_raw

python scripts/convert_movi_to_kubric.py \
    --dataset movi_a \
    --split validation \
    --output-dir data/kubric

# Step 4: Train ViT-B for 10k steps (~3-4 hours on single GPU)
python scripts/train_simple.py \
    --model-config configs/model/vit_b_movi.yaml \
    --training-config configs/training/debug.yaml \
    --data-dir data/kubric
```

**Expected results after 10k steps**:
- L1 3D loss: ~0.05-0.1m (should decrease from ~1.0)
- Validation tracking working
- Point tracks should visually follow objects

**Pros**:
- ✅ Verifies full pipeline at scale
- ✅ Creates checkpoint for inference testing
- ✅ No additional dataset setup needed

**Cons**:
- ⏱ Takes 3-4 hours for download/conversion + 3-4 hours training

---

### Path 2: Add More MOVi Variants (B, C, D, E)

MOVi has 5 variants with increasing complexity. Once MOVi-A training is stable, add more data:

```bash
# MOVi-B: More complex scenes
python scripts/download_movi.py --dataset movi_b --split train --num-samples 1000
python scripts/convert_movi_to_kubric.py --dataset movi_b --split train --output-dir data/kubric

# MOVi-C: Even more complex
python scripts/download_movi.py --dataset movi_c --split train --num-samples 1000
python scripts/convert_movi_to_kubric.py --dataset movi_c --split train --output-dir data/kubric

# MOVi-D: Indoor scenes
python scripts/download_movi.py --dataset movi_d --split train --num-samples 1000
python scripts/convert_movi_to_kubric.py --dataset movi_d --split train --output-dir data/kubric

# MOVi-E: Outdoor scenes (512x512 available)
python scripts/download_movi.py --dataset movi_e --split train --num-samples 1000 --resolution 512x512
python scripts/convert_movi_to_kubric.py --dataset movi_e --split train --output-dir data/kubric --resolution 512x512
```

**Dataset info**:
- MOVi-A: ~10k scenes, simple (2-3 objects)
- MOVi-B: ~10k scenes, more objects
- MOVi-C: ~10k scenes, complex physics
- MOVi-D: ~10k scenes, indoor environments
- MOVi-E: ~10k scenes, outdoor, 512x512 available

---

### Path 3: Add Real-World Datasets

After synthetic data (MOVi) training is stable, add real datasets for robustness:

#### PointOdyssey (Real videos with point tracks)
- Download: https://pointodyssey.com/
- Has ground truth point tracks
- Real-world videos
- Requires account creation (FREE)
- **Use Claude in Chrome** for account setup if needed

#### ScanNet/ScanNet++ (Indoor RGB-D)
- Download: http://www.scan-net.org/
- Requires academic license (FREE)
- Real indoor scenes with depth
- **Use Claude in Chrome** for license application

#### Co3Dv2 (Multi-view objects)
- Download: https://github.com/facebookresearch/co3d
- No account needed
- Good for object-centric tasks

---

## Current Commands Reference

```bash
# Check current data
ls -lh data/kubric/train/ | wc -l  # Should show 100
ls -lh data/kubric/val/ | wc -l    # Should show 20

# Quick 5-step test on MOVi-A
python scripts/train_simple.py \
    --model-config configs/model/vit_b_movi.yaml \
    --training-config configs/training/quick_test_movi.yaml \
    --data-dir data/kubric

# Inspect MOVi dataset
python scripts/download_movi.py \
    --dataset movi_a \
    --split train \
    --inspect \
    --inspect-idx 0

# Download more MOVi samples (incremental)
python scripts/download_movi.py \
    --dataset movi_a \
    --split train \
    --num-samples 500 \
    --output-dir data/movi_raw

# Convert new samples
python scripts/convert_movi_to_kubric.py \
    --dataset movi_a \
    --split train \
    --num-samples 500 \
    --output-dir data/kubric
```

---

## Recommended Next Action

**Start with Path 1** - Download full MOVi-A and train ViT-B for 10k steps:

This will:
1. Verify training stability at scale
2. Create a checkpoint for inference testing
3. Establish baseline metrics for tracking
4. Validate loss convergence on real data

Once this completes (~6-8 hours total):
1. Test inference (point tracking, depth reconstruction)
2. Visualize results to verify quality
3. Then scale to MOVi-B/C/D/E (Path 2)
4. Finally add real datasets (Path 3)

---

## Files Created

- `scripts/download_movi.py` - Download MOVi from Google Cloud Storage
- `scripts/convert_movi_to_kubric.py` - Convert MOVi → D4RT format
- `configs/model/vit_b_movi.yaml` - ViT-B config for MOVi (24 frames @ 256x256)
- `configs/training/quick_test_movi.yaml` - Quick test config
- `uv.lock` - Dependency lockfile (211 packages)

---

## Training Configuration

**Model: ViT-B for MOVi**
- Encoder: 12 layers, 768 hidden dim, ~86M params
- Decoder: 8 layers, 512 hidden dim, ~48M params
- Total: ~135M parameters

**Input**: 24 frames @ 256x256 (MOVi-A format)
**Patch size**: [2, 16, 16] (temporal × height × width)
**Query system**: 5-tuple (u, v, t_src, t_tgt, t_cam)

---

## What's Next?

**Choose one:**
1. 🚀 **Download full MOVi-A** (recommended - establish baseline)
2. 🔧 **Add MOVi-B/C/D/E** (more synthetic variety)
3. 🌐 **Add real datasets** (PointOdyssey, ScanNet, etc.)

Let me know which path you'd like to pursue!

---

## Sources

- [MOVi Dataset README - Kubric GitHub](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md)
- [Multi-Object Video (MOVi) - CVF Datasets](https://cove.thecvf.com/datasets/848)
- [Kubric: A scalable dataset generator - arXiv](https://arxiv.org/abs/2203.03570)
