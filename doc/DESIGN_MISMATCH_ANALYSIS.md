# D4RT Implementation Bug Analysis (RESOLVED)

## Summary

The extremely poor TAP-Vid-3D evaluation results (AJ: 0.0004 vs target 0.304) were due to an **implementation bug** in ground truth extraction, NOT a design mismatch.

**Status: FIXED in Phase 4 (commit 76d2f3c)**

## Original Misdiagnosis

Initially, this was incorrectly diagnosed as a "design mismatch" where D4RT was thought to be a "depth-at-fixed-pixels model". However, further investigation revealed:

1. **The paper clearly describes point tracking**: Table 1 states that point tracking produces "the 3D trajectory of the corresponding point throughout the video"
2. **Ground truth should come from trajectories**: Appendix A states "Queries are sampled from available ground truth point trajectories"

## Actual Bug (FIXED)

**The implementation was incorrectly extracting ground truth at fixed pixel locations instead of tracked positions.**

### The Bug (Before Fix)

In `query_sampling.py`, ground truth was extracted at FIXED pixel locations:
```python
# OLD (WRONG): Extract GT at same pixel in target frame
xyz = points_3d[t_tgt, v_px_src, u_px_src]  # Fixed pixels!
```

### The Fix (Phase 4)

Ground truth is now extracted at TRACKED pixel locations:
```python
# NEW (CORRECT): Extract GT at tracked pixel location
u_track, v_track = tracked_positions[i, t_tgt]  # Where point moved to
xyz = points_3d[t_tgt, v_track, u_track]  # Follows physical point!
```

### What D4RT Should Do (Paper Design)

Given query `(u, v, t_src, t_tgt, t_cam)`:
- Identify the physical point at pixel `(u, v)` in frame `t_src`
- Track that physical point to frame `t_tgt`
- Predict the 3D position of the tracked point in `t_cam` coordinate frame

## Evidence of Bug

### Before Fix (Phase 13 TAP-Vid Evaluation)

| Metric | Result | Target | Analysis |
|--------|--------|--------|----------|
| 3D Position (AJ, APD3D) | ~0.0005 | ~0.35 | **800× worse** - GT mismatch |
| Visibility (OA) | 93.6% | 87.5% | **Better than paper!** - unaffected by bug |

**Why visibility worked but tracking didn't:**
- Visibility is extracted from source frame (bug doesn't affect it)
- 3D positions were extracted at wrong pixels (bug directly causes failure)

## Bug Visualization

```
FRAME 0              FRAME 5              FRAME 10
+--------+          +--------+          +--------+
|   B    |    →    |  B     |    →    | B      |
| [0.5,  |          |  [0.6, |          |  [0.7, |
|  0.5]  |          |   0.4] |          |   0.3] |
|    •   |          |     •  |          |      • |
|        |          |        |          |        |
+--------+          +--------+          +--------+

Correct (Paper / Fixed Implementation):
- Query: "Where is the point from (0.5, 0.5, t=0) at t=10?"
- GT extracted at: tracked location (0.7, 0.3) in frame 10
- Model learns to predict: XYZ of the moving physical point

Buggy (Original Implementation):
- Query: "Where is the point from (0.5, 0.5, t=0) at t=10?"
- GT extracted at: FIXED location (0.5, 0.5) in frame 10 ← WRONG!
- Model learns to predict: XYZ of background, not the moving object
```

## Resolution

### Fix Applied (Phase 4 - commit 76d2f3c)

The bug was fixed by:
1. Generating point tracks during data preprocessing (`tracks.npz` files)
2. Loading tracked positions in `KubricDataset`
3. Using tracked pixel locations for GT extraction in `query_sampling.py`

### Next Steps (Post-Fix)

1. **Continue training** with the corrected ground truth extraction
2. **Re-evaluate** on TAP-Vid-3D after sufficient training
3. **Expected improvement**: AJ should approach paper target (~0.3) with proper training

## Lessons Learned

1. **Verify ground truth extraction carefully** - Training code should extract GT at the same positions the model will predict
2. **Check paper appendix for training details** - The paper's Appendix A explicitly mentions "point trajectories"
3. **Don't confuse implementation bugs with design choices** - Initial analysis incorrectly blamed the design

## References

- D4RT paper: https://arxiv.org/html/2512.08924v1
- Fix commit: 76d2f3c (Phase 4)
- Training GT extraction: `d4rt/data/query_sampling.py`
- TAP-Vid-3D paper: https://arxiv.org/abs/2407.05921
