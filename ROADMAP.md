# D4RT Implementation Roadmap

**Goal**: Reproduce D4RT paper performance on TAPVid-3D benchmark.

## Target Metrics (Paper)

| Metric | Target |
|--------|--------|
| Average Jaccard (AJ) | 0.304 |
| APD3D | 0.410 |
| Occlusion Accuracy | 0.875 |

## Current Status

**Phase**: 12c completed, ready for Phase 14
**Best Checkpoint**: Step 50,000

### 50k Training Results (2026-02-06)

| Metric | Result | Target | % of Target |
|--------|--------|--------|-------------|
| **AJ** | 0.0320 | 0.304 | 10.5% |
| **APD3D** | 0.0321 | 0.410 | 7.8% |
| **OA** | 0.9609 | 0.875 | 110% |

- **AJ improved 80× from 0.0004** (GT fix worked!)
- Occlusion accuracy exceeds paper target
- Scale ratio: 1.12, Confidence: 0.95 (healthy)

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

AJ at 10% of target suggests we need more diverse training data.

**Progress**:
- [x] Created `PointOdysseyDataset` loader (`d4rt/data/datasets/pointodyssey.py`)
- [x] Created `MultiDataset` with weighted sampling (`d4rt/data/datasets/multi_dataset.py`)
- [x] Updated training script to support multi-dataset
- [x] Created `train_multidataset.yaml` config
- [x] Tested pipeline with Kubric + PointOdyssey sample
- [ ] Downloading PointOdyssey val split (~20GB)
- [ ] Train on Kubric + PointOdyssey val
- [ ] Download PointOdyssey train split (~134GB)

**Priority datasets**:
1. **PointOdyssey** - Synthetic videos with dense 3D tracks (val: 20 GB, train: 134 GB)
2. **Co3Dv2** - Multi-view objects (~100 GB) [future]
3. **ScanNet++** - Indoor RGB-D (~150 GB) [future]

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
