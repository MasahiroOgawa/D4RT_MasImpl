# D4RT Scripts

This directory contains training, inference, evaluation, and test scripts for the D4RT implementation.

## Available Scripts

### 1. Training

#### `train.py` - Main Training Script

Train D4RT models with Hydra configuration system.

**Usage:**
```bash
# Train with default debug config
python scripts/train.py

# Train with specific model and config
python scripts/train.py model=vit_b training=debug

# Train with ViT-L
python scripts/train.py model=vit_l training=full_training

# Resume from checkpoint
python scripts/train.py training.resume_from=checkpoints/checkpoint_step_0010000.pth

# Override config parameters
python scripts/train.py training.batch_size=8 training.lr=5e-5
```

#### `test_training.py` - Training Infrastructure Test

Test training setup with dummy data to verify all components work correctly.

**Usage:**
```bash
python scripts/test_training.py
```

**Tests:**
- Training loop execution (10 steps)
- Validation loop
- Checkpoint save/load
- Loss computation
- Gradient flow

### 2. Inference

#### `infer_tracking.py` - Point Tracking

Track sparse points through a video.

**Usage:**
```bash
python scripts/infer_tracking.py \
    --checkpoint checkpoints/model.pth \
    --video path/to/video.mp4 \
    --points "[[0.5, 0.5], [0.3, 0.7]]" \
    --start-frame 0 \
    --output tracking_results.npz \
    --visualize tracking_viz.mp4
```

**Arguments:**
- `--checkpoint`: Path to trained model
- `--video`: Input video file
- `--points`: Points to track in normalized coordinates (0-1)
- `--start-frame`: Frame where points are defined
- `--output`: Output .npz file with trajectories
- `--visualize`: Optional visualization video

#### `infer_depth.py` - Depth Reconstruction

Reconstruct dense depth map for a video frame.

**Usage:**
```bash
python scripts/infer_depth.py \
    --checkpoint checkpoints/model.pth \
    --video path/to/video.mp4 \
    --frame 0 \
    --output depth_map.npz \
    --visualize depth_viz.png \
    --overlay depth_overlay.png
```

**Arguments:**
- `--checkpoint`: Path to trained model
- `--video`: Input video file
- `--frame`: Frame index to reconstruct
- `--output`: Output .npz file with depth map
- `--visualize`: Optional depth visualization (colormap)
- `--overlay`: Optional RGB+depth overlay
- `--batch-size`: Batch size for query processing (default: 4096)

#### `infer_pose.py` - Camera Pose Estimation

Estimate camera pose between frames.

**Usage:**
```bash
# Estimate pose for single frame
python scripts/infer_pose.py \
    --checkpoint checkpoints/model.pth \
    --video path/to/video.mp4 \
    --target-frame 10 \
    --reference-frame 0 \
    --output pose.npz

# Estimate full trajectory
python scripts/infer_pose.py \
    --checkpoint checkpoints/model.pth \
    --video path/to/video.mp4 \
    --reference-frame 0 \
    --output trajectory.npz \
    --visualize trajectory_viz.png
```

**Arguments:**
- `--checkpoint`: Path to trained model
- `--video`: Input video file
- `--target-frame`: Frame to estimate pose for (omit for full trajectory)
- `--reference-frame`: Reference frame (default: 0)
- `--output`: Output .npz file with pose/trajectory
- `--visualize`: Optional trajectory visualization (3D + top view)
- `--num-points`: Number of sparse points for estimation (default: 256)

### 3. Evaluation

#### `evaluate.py` - Comprehensive Evaluation

Evaluate trained models on multiple tasks and datasets.

**Usage:**
```bash
# Evaluate on all tasks
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pth \
    --dataset kubric \
    --data-dir data/kubric \
    --split val \
    --output evaluation_results.json

# Evaluate specific tasks
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pth \
    --dataset kubric \
    --tasks tracking depth \
    --max-samples 100

# Evaluate with custom settings
python scripts/evaluate.py \
    --checkpoint checkpoints/model.pth \
    --dataset kubric \
    --batch-size 2 \
    --num-workers 8 \
    --num-frames 48 \
    --resolution 256 256
```

**Arguments:**
- `--checkpoint`: Path to trained model
- `--dataset`: Dataset name (kubric, etc.)
- `--data-dir`: Data directory
- `--split`: Dataset split (train/val/test)
- `--tasks`: Tasks to evaluate (tracking, depth, pose)
- `--max-samples`: Maximum samples per task
- `--output`: Output JSON file with results

**Output Metrics:**
- **Tracking**: APE, MAE (x, y, z)
- **Depth**: MAE, RMSE, delta thresholds (1.25, 1.25², 1.25³)
- **Pose**: Rotation error (deg), translation error, ATE

### 4. Model Testing

#### `test_model.py` - Comprehensive Model Validation

Tests all core model components:
- Model initialization for all configs (ViT-B, ViT-L, ViT-g)
- Forward pass with dummy data
- Encoder and decoder separation
- Loss computation
- Gradient flow
- Parameter count verification

**Usage:**
```bash
python scripts/test_model.py
```

**Expected Output:**
```
D4RT MODEL VALIDATION TESTS
================================================================================
TEST 1: Model Initialization
...
✓ All tests passed!
```

### 2. Data Pipeline Testing

#### `test_data.py` - Data Pipeline Validation

Tests the complete data pipeline:
- Query sampling strategy (50/25/25 distribution)
- Data augmentation transforms
- Camera parameter utilities
- Batch collation
- Patch extraction

**Usage:**
```bash
python scripts/test_data.py
```

### 3. Model Information

#### `show_model_info.py` - Display Model Architecture

Shows detailed information about D4RT models including:
- Architecture breakdown
- Parameter counts
- Memory usage estimates
- Model comparisons

**Usage:**
```bash
# Show all models
python scripts/show_model_info.py --model all

# Show specific model
python scripts/show_model_info.py --model vit_b

# Compare models
python scripts/show_model_info.py --compare

# Show memory estimates
python scripts/show_model_info.py --model vit_g --memory

# Show everything
python scripts/show_model_info.py --model all --compare --memory
```

**Example Output:**
```
D4RT MODEL: VIT-B
================================================================================
Encoder:       86,470,656 parameters
Query Encoder: 1,841,408 parameters
Decoder:       144,443,393 parameters
------------------------------------------------------------
Total:         232,755,457 parameters
================================================================================
```

## Running Tests

### Quick Validation

Run both test suites to validate the implementation:

```bash
# Test model components
python scripts/test_model.py

# Test data pipeline
python scripts/test_data.py
```

### Before Training

Before starting training, verify:

1. **Model builds correctly:**
   ```bash
   python scripts/show_model_info.py --model vit_b
   ```

2. **All tests pass:**
   ```bash
   python scripts/test_model.py && python scripts/test_data.py
   ```

3. **Memory requirements:**
   ```bash
   python scripts/show_model_info.py --model vit_g --memory
   ```

## Test Coverage

### Model Tests (`test_model.py`)

- ✅ Model initialization
- ✅ Forward pass
- ✅ Encoder output shape
- ✅ Decoder output shape
- ✅ Loss computation
- ✅ Gradient flow
- ✅ Parameter counts

### Data Tests (`test_data.py`)

- ✅ Query sampling
- ✅ Query coordinate ranges
- ✅ Data transforms
- ✅ Camera utilities
- ✅ Batch collation
- ✅ Patch extraction

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project root:
```bash
cd /path/to/D4RT_MasImpl
python scripts/test_model.py
```

### CUDA Out of Memory

For ViT-g testing, you may need to:
- Reduce batch size
- Enable gradient checkpointing
- Use a GPU with more memory

### Test Failures

If tests fail:
1. Check the error message for specific issues
2. Verify all dependencies are installed
3. Make sure PyTorch is properly installed with CUDA support (if using GPU)

## Expected Test Times

On typical hardware:

| Test | CPU | GPU |
|------|-----|-----|
| Model Tests | ~2-5 min | ~30-60 sec |
| Data Tests | ~10-20 sec | ~5-10 sec |

## Next Steps

The implementation is now complete! To start using D4RT:

1. **Prepare datasets:**
   - Download Kubric data or generate locally
   - Place in `data/kubric/` directory

2. **Train a model:**
   ```bash
   # Start with tiny config for testing
   python scripts/train.py model=vit_b_tiny training=debug

   # Once verified, train full ViT-B
   python scripts/train.py model=vit_b training=debug
   ```

3. **Run inference:**
   ```bash
   # Point tracking
   python scripts/infer_tracking.py --checkpoint <path> --video <video>

   # Depth reconstruction
   python scripts/infer_depth.py --checkpoint <path> --video <video>

   # Camera pose estimation
   python scripts/infer_pose.py --checkpoint <path> --video <video>
   ```

4. **Evaluate:**
   ```bash
   python scripts/evaluate.py --checkpoint <path> --dataset kubric
   ```

## Implementation Status

✅ Phase 1: Enhanced documentation
✅ Phase 2: Project structure setup
✅ Phase 3: Core models (encoder, decoder, query encoder)
✅ Phase 4: Data pipeline (Kubric dataset, query sampling)
✅ Phase 5: Loss functions and test scripts
✅ Phase 6: Training infrastructure (trainer, optimizer, checkpointing)
✅ Phase 7: Inference modules (tracking, depth, pose)
✅ Phase 9: Evaluation system

⏸ Phase 8: Train and validate on real data (requires dataset preparation)
