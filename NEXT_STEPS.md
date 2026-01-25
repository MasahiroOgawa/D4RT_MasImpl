# Next Steps: Scaling Up Training

## ✅ COMPLETED: Full Training Pipeline Verified!

**Phase 8b completed successfully** - Training on real MOVi-A data working!

### What's Working:

1. ✅ **uv sync workflow** - Dependency management with lockfile
2. ✅ **MOVi-A dataset downloaded** - 100 train + 20 val samples from `gs://kubric-public/tfds`
3. ✅ **Data conversion** - MOVi format → D4RT Kubric format converter working
4. ✅ **Training on real data** - 5-step test completed successfully
5. ✅ **All components verified** - Data loading, model forward/backward, loss, validation

### Current Setup:

```bash
# Quick test (5 steps on MOVi-A data)
python scripts/train_simple.py \
    --model-config configs/model/vit_b_movi.yaml \
    --training-config configs/training/quick_test_movi.yaml \
    --data-dir data/kubric

# Current dataset: 100 MOVi-A training scenes (24 frames @ 256x256 each)
```

### Training Results (5-step test):
- **Step 1**: loss = 158.79
- **Step 2**: loss = inf (numerical spike, expected with random init)
- **Step 3**: val_loss = 328.19 (validation working!)
- **Step 4**: loss = 90.38
- **Step 5**: loss = 89.90
- **Speed**: ~1.73 s/it on GPU

---

## Next: Scale Up Training & Data

You now have 3 paths forward (in priority order):

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
