# D4RT Implementation Roadmap

**Goal**: Reproduce D4RT paper performance on TAPVid-3D benchmark.

## Target Metrics (Paper)

| Metric | Target |
|--------|--------|
| Average Jaccard (AJ) | 0.304 |
| APD3D | 0.410 |
| Occlusion Accuracy | 0.875 |

## Current Status

**Phase**: Training with fixed GT format
**Step**: 11,000 / 50,000
**Best Val Loss**: 0.288 (step 8,000)

Training is running in background.

---

## Completed Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1-8 | Core implementation (model, data, losses, training) | Done |
| 9-11 | Bug fixes (FP16 overflow, UV normalization) | Done |
| 12a | Download full MOVi-A (9,703 scenes) | Done |
| 12b | First 50k training | Done (buggy GT) |
| 13 | TAP-Vid metrics implementation | Done |
| 13.5 | Paper loss functions + confidence warmup | Done |
| GT Fix | Fix dataset to use tracks_3d directly | Done |

**Key Bug Fixed**: GT was computed from depth at fixed pixels instead of using tracked 3D positions from `tracks.npz`. This caused the model to learn depth estimation instead of point tracking.

---

## Current Phase

### Phase 12c: Retrain with Fixed GT (IN PROGRESS)

Training restarted from scratch with correct ground truth format.

```bash
# Monitor training
tail -f training.log
./scripts/monitor_training.sh --quick

# Run intermediate evaluation
uv run python scripts/quick_eval.py --checkpoint checkpoints/checkpoint_latest.pth
```

**Success Criteria**:
- Val loss < 0.2
- AJ > 0.05 on validation set
- Scale ratio (pred_std/gt_std) > 0.3

---

## Remaining Phases

### Phase 14: Add Real-World Datasets

After 50k training shows improvement, add diverse data:

1. **PointOdyssey** - Real videos with 3D tracks (~30 GB)
2. **Co3Dv2** - Multi-view objects (~100 GB)
3. **ScanNet++** - Indoor RGB-D (~150 GB)

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
