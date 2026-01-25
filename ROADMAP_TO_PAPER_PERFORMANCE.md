# Roadmap to Match D4RT Paper Performance

**Goal**: Achieve paper-level performance on public benchmarks (TAPVid-3D, TAP-Vid-DAVIS)

**Current Status** (as of Phase 11):
- ✅ Training pipeline working (FP32, all losses enabled)
- ✅ Trained 5000 steps on 1634 MOVi-A samples
- ✅ Validation loss: 2.75 (converged)
- ⚠️ Tracking quality: Poor (minimal movement, not following objects)
- ⚠️ Limited dataset: Only MOVi-A synthetic data
- ⚠️ No evaluation metrics implemented

---

## Paper Performance Targets

### Training Setup (from paper)
- **Datasets**: BlendedMVS, Co3Dv2, Dynamic Replica, Kubric, MVS-Synth, PointOdyssey, ScanNet++, ScanNet, Tartanair, VirtualKitti, Waymo Open
- **Training duration**: 2+ days on 64 TPU chips
- **Input**: 48-frame clips @ 256×256 resolution
- **Batch**: 2,048 random queries per batch
- **Optimizer**: AdamW (weight decay: 0.03)
- **Learning rate**: Warmup to 1e-4 (2,500 steps) → cosine decay to 1e-6
- **Gradient clipping**: L2-norm capped at 10

### Evaluation Benchmarks & Metrics
- **TAPVid-3D**: DriveTrack, ADT, PStudio (real-world videos)
  - Average Jaccard (AJ): **0.304** (camera coords with GT intrinsics)
  - APD3D: **0.410**
  - Occlusion Accuracy (OA): **0.875-0.897**
- **Static scenes**: ScanNet, Re10K (pose estimation)
- **Dynamic scenes**: MPI Sintel (depth + camera pose)
- **Additional**: KITTI, Bonn (depth evaluation)

### Efficiency
- **18-300× faster** than prior methods
- **40,180 max tracks** @ 1 FPS on single A100 GPU

---

## Phase 12: Scale Up Training on Synthetic Data

**Goal**: Train on full Kubric/MOVi dataset for 50k steps to verify model can learn proper tracking.

### 12a. Download Full MOVi-A Dataset
```bash
# Download full MOVi-A training set (9,703 samples, ~50GB)
python scripts/download_movi.py \
    --dataset movi_a \
    --split train \
    --output-dir data/movi_raw

# Convert to D4RT format
python scripts/convert_movi_to_kubric.py \
    --dataset movi_a \
    --split train \
    --output-dir data/kubric

# Download validation set (250 samples)
python scripts/download_movi.py \
    --dataset movi_a \
    --split validation \
    --output-dir data/movi_raw

python scripts/convert_movi_to_kubric.py \
    --dataset movi_a \
    --split validation \
    --output-dir data/kubric
```

**Expected time**: ~6-8 hours (download + conversion)

### 12b. Create 50k Training Config
```yaml
# configs/training/train_50k_movi.yaml
training:
  max_steps: 50000  # 10× longer
  batch_size: 1
  num_workers: 0
  gradient_accumulation_steps: 4
  mixed_precision: false  # Keep FP32 for stability
  gradient_checkpointing: true
  clip_grad_norm: 1.0  # Match paper (L2-norm = 10, but we use different scale)
  log_every_n_steps: 100
  val_every_n_steps: 1000
  save_every_n_steps: 5000
  num_queries_per_step: 128  # Restore to 128 (paper uses 2048)

optimizer:
  type: adamw
  lr: 1e-4  # Match paper (before warmup)
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0.03  # Match paper

scheduler:
  type: cosine_with_warmup
  warmup_steps: 2500  # Match paper
  T_max: 50000
  eta_min: 1e-6  # Match paper

loss_weights:
  l1_3d: 1.0
  l2_2d: 0.1
  normal: 0.0
  motion: 0.1
  visibility: 0.1
```

### 12c. Train for 50k Steps
```bash
# Resume from current checkpoint
python scripts/train_simple.py \
    --model-config configs/model/vit_b_movi.yaml \
    --training-config configs/training/train_50k_movi.yaml \
    --data-dir data/kubric \
    --resume-from checkpoints/checkpoint_step_0005000.pth
```

**Expected time**: ~10-12 hours on single GPU (45k additional steps @ 1.4 it/s)

**Success criteria**:
- Training loss < 1.0
- Validation loss < 1.5
- Visual inspection: Points should follow objects in validation videos

---

## Phase 13: Implement TAP-Vid Evaluation

**Goal**: Implement official TAP-Vid metrics for proper evaluation.

### 13a. Download TAP-Vid Benchmark
```bash
# TAPVid-3D dataset
pip install tapnet
python -c "from tapnet.utils import download_tapvid3d; download_tapvid3d('data/tapvid3d')"

# TAP-Vid-DAVIS (2D benchmark)
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip -d data/tapvid_davis
```

### 13b. Implement Evaluation Metrics
Create `scripts/evaluate_tapvid.py`:
- **Average Jaccard (AJ)**: Joint metric for accuracy + occlusion
- **Average Position Accuracy within δ (δ^avg)**: % points within threshold
- **Occlusion Accuracy (OA)**: Binary visibility prediction accuracy
- **APD3D**: Average % points within Δ error (3D version)

### 13c. Run Baseline Evaluation
```bash
# Evaluate current 5k checkpoint
python scripts/evaluate_tapvid.py \
    --checkpoint checkpoints/checkpoint_step_0005000.pth \
    --model-config configs/model/vit_b_movi.yaml \
    --dataset data/tapvid3d \
    --split validation

# After 50k training
python scripts/evaluate_tapvid.py \
    --checkpoint checkpoints/checkpoint_step_0050000.pth \
    --model-config configs/model/vit_b_movi.yaml \
    --dataset data/tapvid3d \
    --split validation
```

**Target metrics** (paper performance):
- AJ: 0.304
- APD3D: 0.410
- OA: 0.875

---

## Phase 14: Add Real-World Datasets

**Goal**: Train on diverse real-world data to match paper's multi-dataset training.

### Datasets to Add (priority order)

#### 14a. PointOdyssey (Real videos with 3D tracks)
```bash
# Download from https://pointodyssey.com/
# Requires free account registration
python scripts/download_pointodyssey.py \
    --output-dir data/pointodyssey \
    --split train
```
- 104 videos with dense 3D point tracks
- Real-world camera motion

#### 14b. Co3Dv2 (Multi-view objects)
```bash
# Download from Facebook Research
python scripts/download_co3d.py \
    --categories "all" \
    --output-dir data/co3d
```
- 19,000+ videos, 50 object categories
- Multi-view with camera poses

#### 14c. ScanNet++ (Indoor RGB-D)
```bash
# Download from http://www.scan-net.org/
# Requires academic license (FREE)
python scripts/download_scannet.py \
    --output-dir data/scannet \
    --type scannetpp
```
- 460 high-quality indoor scans
- Accurate depth and camera poses

#### 14d. Additional Datasets (Optional)
- **MVS-Synth**: Synthetic multi-view stereo
- **BlendedMVS**: Real MVS scenes
- **Waymo Open**: Autonomous driving (large-scale)
- **TartanAir**: Drone simulation
- **VirtualKitti**: Synthetic driving

### 14e. Create Multi-Dataset Training Config
```yaml
# configs/training/train_multidataset.yaml
dataset:
  type: multi_dataset
  datasets:
    - name: kubric
      weight: 0.3
      data_dir: data/kubric
    - name: pointodyssey
      weight: 0.2
      data_dir: data/pointodyssey
    - name: co3d
      weight: 0.2
      data_dir: data/co3d
    - name: scannet
      weight: 0.3
      data_dir: data/scannet
  # Sample from each dataset with probability proportional to weight
```

---

## Phase 15: Large-Scale Training (Paper Reproduction)

**Goal**: Train with paper-like setup to match performance.

### 15a. Upgrade Training Infrastructure
**Current**: Single GPU (11.6 GB), ~1.4 it/s
**Paper**: 64 TPUs, 2+ days

**Options**:
1. **Cloud GPUs**: AWS/GCP with 8× A100 GPUs (expensive but faster)
2. **Longer training**: Single GPU for 1-2 weeks
3. **Gradient accumulation**: Simulate larger batch size

### 15b. Final Training Run
```bash
# Train for 100k-200k steps on multi-dataset
python scripts/train_simple.py \
    --model-config configs/model/vit_b_movi.yaml \
    --training-config configs/training/train_multidataset.yaml \
    --max-steps 200000
```

**Expected time**: 2-4 weeks on single GPU

### 15c. Hyperparameter Tuning
- Loss weights: Adjust 3D/2D/motion/visibility ratios
- Learning rate schedule: Fine-tune warmup and decay
- Query sampling: Match paper's emphasis on "depth discontinuities and motion boundaries"
- Data augmentation: Add color jitter, spatial crops (paper uses these)

---

## Phase 16: Final Evaluation & Benchmarking

### 16a. Evaluate on All Benchmarks
```bash
# TAPVid-3D
python scripts/evaluate_tapvid.py \
    --checkpoint checkpoints/final.pth \
    --dataset data/tapvid3d

# TAP-Vid-DAVIS
python scripts/evaluate_tapvid.py \
    --checkpoint checkpoints/final.pth \
    --dataset data/tapvid_davis

# ScanNet (static scenes)
python scripts/evaluate_scannet.py \
    --checkpoint checkpoints/final.pth

# MPI Sintel (dynamic scenes)
python scripts/evaluate_sintel.py \
    --checkpoint checkpoints/final.pth
```

### 16b. Compare with Paper Results
Create comparison table:
| Metric | Paper | Ours | Gap |
|--------|-------|------|-----|
| AJ (TAPVid-3D) | 0.304 | ? | ? |
| APD3D | 0.410 | ? | ? |
| OA | 0.875 | ? | ? |
| Speed (tracks/s) | 40,180 | ? | ? |

### 16c. Publish Results
- Update README with benchmark results
- Create visualization videos
- Share trained checkpoints

---

## Estimated Timeline

| Phase | Task | Duration | Bottleneck |
|-------|------|----------|------------|
| 12 | Scale up MOVi training (50k steps) | 1-2 days | GPU time |
| 13 | Implement TAP-Vid evaluation | 1 day | Code complexity |
| 14 | Add real-world datasets | 3-5 days | Download time + storage |
| 15 | Multi-dataset training (200k steps) | 2-4 weeks | GPU time |
| 16 | Final evaluation | 1-2 days | - |
| **Total** | **End-to-end** | **4-6 weeks** | **GPU availability** |

---

## Resource Requirements

### Compute
- **Current**: Single GPU (11.6 GB VRAM)
- **Sufficient for**: Phases 12-14 (slower but workable)
- **Ideal**: 8× A100 GPUs for Phase 15

### Storage
- **Current**: ~10 GB (1634 MOVi-A samples + checkpoints)
- **After Phase 12**: ~60 GB (full MOVi-A)
- **After Phase 14**: ~200-500 GB (all real-world datasets)
- **Final**: ~1 TB (including checkpoints, logs, results)

### Cost Estimate (if using cloud)
- **AWS p4d.24xlarge** (8× A100): $32.77/hour
- **Phase 15 training** (2 weeks): ~$11,000
- **Alternative**: Single A100 for 4 weeks: ~$3,000

---

## Critical Success Factors

1. **Data Quality**: Ensure all datasets are properly converted to D4RT format
2. **Loss Convergence**: Monitor all loss components (3D, 2D, motion, visibility)
3. **Evaluation Metrics**: Implement exact TAP-Vid metrics (not approximations)
4. **Hyperparameter Tuning**: Match paper's training recipe as closely as possible
5. **Debugging**: Expect issues when adding new datasets - test each incrementally

---

## Next Immediate Actions

**Right now (Phase 12a)**:
1. Download full MOVi-A dataset (9,703 training samples)
2. Create `configs/training/train_50k_movi.yaml`
3. Start 50k training run overnight
4. Monitor training loss - should decrease below 1.0

**Tomorrow (Phase 13)**:
1. While training runs, implement TAP-Vid evaluation metrics
2. Download TAP-Vid benchmark datasets
3. Test evaluation on current 5k checkpoint

**This week (Phase 14)**:
1. Download PointOdyssey dataset
2. Create data converters for real-world datasets
3. Test multi-dataset training pipeline

---

## References

- [D4RT Paper](https://storage.googleapis.com/d4rt_assets/D4RT_paper.pdf)
- [D4RT Project Page](https://d4rt-paper.github.io/)
- [TAPVid-3D Benchmark](https://tapvid3d.github.io/)
- [TAP-Vid Benchmark](https://tapvid.github.io/)
- [PointOdyssey Dataset](https://pointodyssey.com/)
- [Co3Dv2 Dataset](https://github.com/facebookresearch/co3d)
- [ScanNet++ Dataset](http://www.scan-net.org/)
