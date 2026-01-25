# D4RT Implementation: Complete Plan to Match Paper Performance

## 🎯 ULTIMATE GOAL
**Reproduce D4RT paper results as closely as possible on public benchmarks**

---

## Target Performance (from paper)

### TAPVid-3D Benchmark (Real-world videos)
| Metric | Paper Target | Current | Status |
|--------|--------------|---------|--------|
| Average Jaccard (AJ) | **0.304** | Not measured | 📍 Need evaluation |
| APD3D | **0.410** | Not measured | 📍 Need evaluation |
| Occlusion Accuracy (OA) | **0.875-0.897** | Not measured | 📍 Need evaluation |

### Efficiency
- **Target**: 18-300× faster than prior methods
- **Target**: 40,180 tracks @ 1 FPS on single A100

---

## Complete Phase Breakdown

### ✅ PHASE 1-8: Initial Setup & Pipeline (COMPLETED)
**Duration**: Completed
**Goal**: Get basic training working

- [x] Phase 1: Environment setup (uv, dependencies)
- [x] Phase 2: Dataset acquisition (100 MOVi-A samples)
- [x] Phase 3: Data conversion pipeline
- [x] Phase 4: Model architecture implementation
- [x] Phase 5: Training loop implementation
- [x] Phase 6: Loss functions implementation
- [x] Phase 7: Memory optimizations (gradient checkpointing)
- [x] Phase 8: Initial 5k training run

**Result**: Training pipeline works, but model doesn't learn (FP16 bug)

---

### ✅ PHASE 9-10: Critical Bug Discovery & Fixes (COMPLETED)
**Duration**: Completed
**Goal**: Fix broken training

**Phase 9: Inference Testing**
- [x] Implement inference script
- [x] Test on validation videos
- [x] Discovery: Model not tracking objects at all

**Phase 10: Root Cause Analysis**
- [x] Discovered FP16 overflow causing inf/nan
- [x] Disabled FP16 → switched to FP32
- [x] Re-trained 5k steps successfully
- [x] Model now learns (loss: 146 → 2.4)

**Result**: Training works, but tracking quality still poor (need more training)

---

### ✅ PHASE 11: Fix 2D Projection Loss (COMPLETED)
**Duration**: Completed
**Goal**: Re-enable all loss components

- [x] Fixed 2D projection loss overflow (UV clamping)
- [x] Re-enabled motion loss
- [x] Re-enabled visibility loss
- [x] Fixed checkpointing symlink bug
- [x] Trained 5k steps with full loss function
- [x] Final validation loss: 2.75

**Result**: All losses working, model learning, but still minimal tracking movement

---

### 🚧 PHASE 12: Scale Up on Synthetic Data (CURRENT)
**Duration**: 1-2 days
**Goal**: Train long enough to learn proper tracking behavior

**Status**: 📍 **WE ARE HERE**

#### 12a. Download Full MOVi-A Dataset
- [ ] Download remaining MOVi-A samples (1,634 → 9,703)
- [ ] Download full validation set (20 → 250 samples)
- [ ] Convert all to D4RT format
- **Time**: ~6-8 hours
- **Storage**: ~50 GB

#### 12b. Train for 50k Steps
- [ ] Resume from checkpoint_step_0005000.pth
- [ ] Train with config: `train_50k_movi.yaml`
- [ ] Monitor losses (target: < 1.0 training, < 1.5 validation)
- [ ] Validate tracking visually at checkpoints
- **Time**: ~10-12 hours on single GPU
- **Expected**: Points should start following objects

**Success Criteria**:
- Training loss < 1.0
- Validation loss < 1.5
- Visual inspection: Points follow objects in videos
- Movement > 0.2 units/frame (currently 0.08)

---

### 📋 PHASE 13: Implement Proper Evaluation
**Duration**: 1-2 days
**Goal**: Measure performance using official metrics

#### 13a. Download TAP-Vid Benchmarks
- [ ] TAPVid-3D dataset (4,000+ real videos, 2.1M 3D trajectories)
- [ ] TAP-Vid-DAVIS (2D benchmark for comparison)
- **Time**: ~2-4 hours
- **Storage**: ~20 GB

#### 13b. Implement TAP-Vid Metrics
Create `scripts/evaluate_tapvid.py` with official metrics:
- [ ] **Average Jaccard (AJ)**: Joint accuracy + occlusion metric
- [ ] **APD3D**: % of points within Δ error (3D version)
- [ ] **Occlusion Accuracy (OA)**: Binary visibility prediction
- [ ] **δ^avg**: Average position accuracy within threshold

#### 13c. Baseline Evaluation
- [ ] Evaluate 5k checkpoint (current)
- [ ] Evaluate 50k checkpoint (after Phase 12)
- [ ] Compare against paper targets
- [ ] Identify performance gaps

**Expected Results**:
- 5k checkpoint: AJ ~0.05-0.10 (poor)
- 50k checkpoint: AJ ~0.15-0.20 (better but not paper-level)
- Gap to paper (AJ 0.304): ~50% improvement needed

---

### 📋 PHASE 14: Add Real-World Datasets
**Duration**: 3-5 days
**Goal**: Train on diverse data like the paper

#### Priority Dataset Downloads

**14a. PointOdyssey** (Real videos with 3D tracks)
- [ ] Register account at pointodyssey.com
- [ ] Download 104 training videos with dense 3D tracks
- [ ] Implement data converter to D4RT format
- **Time**: ~4-6 hours
- **Storage**: ~30 GB

**14b. Co3Dv2** (Multi-view objects)
- [ ] Download from Facebook Research
- [ ] 19,000+ videos, 50 object categories
- [ ] Implement multi-view data converter
- **Time**: ~8-12 hours
- **Storage**: ~100 GB

**14c. ScanNet++** (Indoor RGB-D)
- [ ] Register for academic license (free)
- [ ] Download 460 indoor scans
- [ ] Implement RGB-D data converter
- **Time**: ~12-24 hours
- **Storage**: ~150 GB

#### Optional (if needed for paper match):
- [ ] BlendedMVS (real MVS scenes)
- [ ] MVS-Synth (synthetic MVS)
- [ ] Waymo Open (autonomous driving, large-scale)
- [ ] TartanAir (drone simulation)
- [ ] VirtualKitti (synthetic driving)

#### 14d. Multi-Dataset Training Pipeline
- [ ] Create `MultiDatasetLoader` class
- [ ] Implement dataset weighting/sampling
- [ ] Test on small subset (verify no crashes)
- [ ] Create `train_multidataset.yaml` config

**Expected**: Training on 3-4 diverse datasets improves generalization

---

### 📋 PHASE 15: Large-Scale Training
**Duration**: 2-4 weeks (single GPU) OR 2-3 days (8× A100)
**Goal**: Match paper's training scale

#### 15a. Resource Decision
**Option A: Single GPU (slower but free)**
- Train for 100k-200k steps over 2-4 weeks
- Use what we have (11.6 GB GPU)
- Budget-friendly

**Option B: Cloud GPUs (faster but expensive)**
- AWS p4d.24xlarge: 8× A100 GPUs
- Cost: ~$32/hour = ~$11,000 for 2 weeks
- 10× faster training

#### 15b. Hyperparameter Tuning
Based on paper training recipe:
- [ ] Match learning rate schedule exactly (warmup 2500, cosine decay)
- [ ] Match weight decay (0.03)
- [ ] Match gradient clipping (L2-norm = 10)
- [ ] Tune loss weights if needed
- [ ] Add data augmentation (color jitter, spatial crops)
- [ ] Implement query sampling strategy (emphasize depth discontinuities)

#### 15c. Final Training Run
- [ ] Train on multi-dataset mix
- [ ] Target: 100k-200k steps
- [ ] Save checkpoints every 10k steps
- [ ] Monitor all metrics continuously
- [ ] Early stopping if validation plateaus

**Success Criteria**:
- Training loss < 0.5
- Validation loss < 1.0
- TAPVid-3D AJ > 0.25 (midway to paper)

---

### 📋 PHASE 16: Final Evaluation & Benchmarking
**Duration**: 1-2 days
**Goal**: Measure final performance vs paper

#### 16a. Comprehensive Evaluation
Run on all benchmarks:
- [ ] TAPVid-3D (3 subsets: DriveTrack, ADT, PStudio)
- [ ] TAP-Vid-DAVIS (2D tracking)
- [ ] ScanNet (static scene pose estimation)
- [ ] MPI Sintel (dynamic scene depth + pose)
- [ ] KITTI (depth estimation)

#### 16b. Performance Comparison Table
| Benchmark | Metric | Paper | Ours | Gap |
|-----------|--------|-------|------|-----|
| TAPVid-3D | AJ | 0.304 | ? | ? |
| TAPVid-3D | APD3D | 0.410 | ? | ? |
| TAPVid-3D | OA | 0.875 | ? | ? |
| Speed | tracks/s | 40,180 | ? | ? |

#### 16c. Analysis & Iteration
- [ ] Identify remaining performance gaps
- [ ] Hypothesize causes (data? training? architecture?)
- [ ] Decide: Good enough OR need another iteration?

#### 16d. Publication Prep
- [ ] Update README with final results
- [ ] Create visualization videos
- [ ] Release trained checkpoints
- [ ] Document training recipe
- [ ] (Optional) Write technical report

---

## Timeline Summary

| Phase | Duration | Bottleneck | Can Parallelize? |
|-------|----------|------------|------------------|
| **12: Scale up MOVi** | 1-2 days | GPU time | ❌ Sequential |
| **13: Evaluation metrics** | 1-2 days | Code complexity | ✅ Yes (during Phase 12) |
| **14: Add datasets** | 3-5 days | Download time | ✅ Yes (parallel downloads) |
| **15: Large-scale training** | 2-4 weeks | GPU time | ❌ Sequential |
| **16: Final evaluation** | 1-2 days | Compute | ❌ Sequential |
| **TOTAL** | **4-6 weeks** | **GPU availability** | Phases 13-14 can overlap |

---

## Resource Requirements Summary

### Compute
| Phase | GPU Needed | Can Use Current Hardware? |
|-------|------------|---------------------------|
| 12 | 1× GPU (11.6 GB) | ✅ Yes |
| 13 | CPU mostly | ✅ Yes |
| 14 | CPU (data prep) | ✅ Yes |
| 15 | 1-8× GPU | ⚠️ Slow on 1 GPU (4 weeks) |
| 16 | 1× GPU | ✅ Yes |

### Storage
| Phase | Storage Needed | Cumulative |
|-------|----------------|------------|
| Current | ~10 GB | 10 GB |
| After Phase 12 | +50 GB | 60 GB |
| After Phase 14 | +280 GB | 340 GB |
| After Phase 15 | +100 GB (checkpoints) | 440 GB |
| Final | ~500 GB total | 500 GB |

**Recommendation**: Ensure 500 GB+ free disk space before starting

---

## Critical Success Factors

### Must-Have for Paper Performance
1. ✅ **Training pipeline stability** (FP32, all losses working)
2. 📍 **Sufficient training steps** (100k-200k, not just 5k)
3. 📍 **Diverse datasets** (not just synthetic MOVi)
4. 📍 **Proper evaluation metrics** (official TAP-Vid code)
5. 📍 **Hyperparameter tuning** (match paper recipe)

### Nice-to-Have (but not essential)
- Multi-GPU training (faster but same result)
- Additional datasets beyond paper's list
- Custom data augmentation strategies

### Potential Roadblocks
1. **Dataset download time** (280 GB total, 12-24 hours)
2. **Storage limits** (need 500 GB free)
3. **Training time on single GPU** (4 weeks for 200k steps)
4. **Evaluation complexity** (TAP-Vid metrics not trivial)

---

## Decision Points

### After Phase 12 (50k training)
**Question**: Is tracking quality improving?
- ✅ **Yes** → Continue to Phase 13-14
- ❌ **No** → Debug loss functions, check data quality

### After Phase 14 (datasets added)
**Question**: Do we have enough diverse data?
- ✅ **Yes** (3-4 datasets) → Proceed to Phase 15
- ❌ **No** → Add more datasets from paper

### After Phase 15 (large-scale training)
**Question**: Are we within 10% of paper performance?
- ✅ **Yes** → Declare success, move to Phase 16
- ⚠️ **Marginal (10-30% gap)** → Fine-tune hyperparameters, train longer
- ❌ **No (>30% gap)** → Re-examine architecture, loss functions, data

---

## Immediate Next Actions (Phase 12-13)

### Right Now:
1. **Start MOVi-A download** (runs overnight, ~8 hours)
   ```bash
   nohup python scripts/download_movi.py --dataset movi_a --split train &
   ```

2. **Implement TAP-Vid evaluation** (while download runs)
   - Clone TAP-Vid repo
   - Implement evaluation script
   - Test on existing 5k checkpoint

### Tomorrow:
3. **Convert MOVi-A to D4RT format** (~6 hours)
4. **Start 50k training run** (overnight, ~12 hours)

### Day 3:
5. **Evaluate 50k checkpoint** on TAP-Vid metrics
6. **Analyze results** and decide on Phase 14 dataset priorities

---

## Expected Final Performance

### Conservative Estimate
- TAPVid-3D AJ: **0.20-0.25** (66-82% of paper)
- Reason: Single GPU, limited training time

### Optimistic Estimate
- TAPVid-3D AJ: **0.25-0.30** (82-98% of paper)
- Requires: All phases completed, good hyperparameter tuning

### Best Case
- TAPVid-3D AJ: **0.30+** (matches or exceeds paper)
- Requires: Cloud GPUs for Phase 15, extensive tuning

**Realistic Goal**: Aim for 80-90% of paper performance (AJ ~0.25-0.27)

---

## References

- [D4RT Paper (PDF)](https://storage.googleapis.com/d4rt_assets/D4RT_paper.pdf)
- [D4RT Project Page](https://d4rt-paper.github.io/)
- [TAPVid-3D Benchmark](https://tapvid3d.github.io/)
- [TAP-Vid Benchmark](https://tapvid.github.io/)
- [PointOdyssey Dataset](https://pointodyssey.com/)
- [Co3Dv2 Dataset](https://github.com/facebookresearch/co3d)
- [ScanNet++ Dataset](http://www.scan-net.org/)
