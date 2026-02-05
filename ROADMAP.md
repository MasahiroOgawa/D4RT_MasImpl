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

### Phase 14: Add Real-World Datasets

AJ at 10% of target suggests we need more diverse training data.

**Priority datasets**:
1. **PointOdyssey** - Real videos with 3D tracks (~30 GB)
2. **Co3Dv2** - Multi-view objects (~100 GB)
3. **ScanNet++** - Indoor RGB-D (~150 GB)

**Steps**:
1. Download PointOdyssey dataset
2. Create data converter to D4RT format
3. Implement multi-dataset training pipeline
4. Train on combined dataset

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
# Resume training
uv run python scripts/train.py --config-name=train_50k_movi_paper \
    +training.resume_from=checkpoints/checkpoint_latest.pth

# Quick evaluation (10 scenes)
uv run python scripts/quick_eval.py --checkpoint checkpoints/checkpoint_latest.pth

# Full evaluation (all val scenes)
uv run python scripts/quick_eval.py --checkpoint checkpoints/checkpoint_latest.pth --num_scenes 20
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
