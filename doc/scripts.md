# D4RT Scripts Reference

Command-line usage for all scripts in the D4RT implementation.

## Directory Structure

```
scripts/
├── train.py                    # Main training script
├── train_tmux.sh               # Tmux-based training with monitoring
├── eval_monitor.sh             # Background evaluation monitor
├── battery_monitor.sh          # Battery-aware training
├── evaluate.py                 # Generic evaluation
├── quick_eval.py               # Quick TAPVid-3D evaluation
├── evaluate_tapvid.py          # Official TAPVid-3D benchmark
├── evaluate_movi_tracks.py     # MOVi tracks evaluation
├── infer_tracking.py           # Point tracking inference
├── infer_depth.py              # Depth reconstruction
├── infer_pose.py               # Camera pose estimation
├── download_movi.py            # Download MOVi dataset
├── convert_movi_to_kubric.py   # Convert MOVi to Kubric format
├── add_tracks_to_movi.py       # Add point tracks to MOVi data
├── prepare_kubric_data.py      # Prepare Kubric data
├── show_model_info.py          # Model architecture info
├── visualize_predictions.py    # Visualize predictions vs GT
├── visualize_tracking_video.py # Create tracking visualization video
├── debug/                      # Debug scripts (19 files)
└── test/                       # Test scripts (5 files)
```

## Training

### `train.py` - Main Training Script

Train D4RT models with Hydra configuration.

```bash
# Train with default config
uv run python scripts/train.py --config-name train_50k_movi_paper --config-path ../configs/training

# Resume from checkpoint
uv run python scripts/train.py --config-name train_50k_movi_paper --config-path ../configs/training \
    +training.resume_from=checkpoints/checkpoint_step_0005000.pth

# Override parameters
uv run python scripts/train.py --config-name train_50k_movi_paper --config-path ../configs/training \
    training.batch_size=2 optimizer.lr=5e-5
```

### `train_tmux.sh` - Tmux Training Wrapper

Run training in tmux with automatic monitoring (recommended).

```bash
# Start training (fresh)
./scripts/train_tmux.sh

# Start with specific config
./scripts/train_tmux.sh train_50k_movi_paper

# Resume from checkpoint
./scripts/train_tmux.sh train_50k_movi_paper checkpoints/checkpoint_step_0005000.pth
```

**Features:**
- Survives terminal close / laptop lid close
- Battery monitoring (saves checkpoint when < 10%)
- Background evaluation every 1000 steps
- Status window for quick monitoring

**Tmux commands:**
- `tmux attach -t d4rt` - Attach to session
- `Ctrl+b, d` - Detach from session
- `Ctrl+b, n/p` - Switch windows
- `tmux kill-session -t d4rt` - Kill session

### `eval_monitor.sh` - Background Evaluation

Runs evaluation every N training iterations.

```bash
# Default: every 1000 steps
./scripts/eval_monitor.sh outputs/eval.log 1000

# Every 500 steps
./scripts/eval_monitor.sh outputs/eval.log 500
```

### `battery_monitor.sh` - Battery Monitor

Monitors battery and saves checkpoint when low.

```bash
./scripts/battery_monitor.sh
```

## Evaluation

### `evaluate.py` - Generic Evaluation

```bash
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_latest.pth \
    --dataset kubric \
    --data-dir data/kubric \
    --split val \
    --output results.json
```

### `quick_eval.py` - Quick TAPVid-3D Evaluation

Fast evaluation using TAPVid-3D metrics.

```bash
uv run python scripts/quick_eval.py \
    --checkpoint checkpoints/checkpoint_latest.pth \
    --num_scenes 5 \
    --alignment scale_shift
```

**Metrics:**
- **AJ (Average Jaccard)**: Target 0.304
- **APD3D (Average Position Distance 3D)**: Target 0.410
- **OA (Occlusion Accuracy)**: Target 0.875

### `evaluate_tapvid.py` - Official TAPVid-3D Benchmark

```bash
uv run python scripts/evaluate_tapvid.py \
    --checkpoint checkpoints/checkpoint_step_0050000.pth \
    --model-config configs/model/vit_b_movi.yaml \
    --dataset-dir data/tapvid3d \
    --split val
```

### `evaluate_movi_tracks.py` - MOVi Tracks Evaluation

```bash
uv run python scripts/evaluate_movi_tracks.py \
    --checkpoint checkpoints/checkpoint_step_0050000.pth \
    --model-config configs/model/vit_b_movi.yaml \
    --data-dir data/kubric \
    --split val \
    --max-samples 20 \
    --output results/movi_eval.json
```

## Inference

### `infer_tracking.py` - Point Tracking

Track sparse points through video.

```bash
uv run python scripts/infer_tracking.py \
    --checkpoint checkpoints/model.pth \
    --video path/to/video.mp4 \
    --points "[[0.5, 0.5], [0.3, 0.7]]" \
    --start-frame 0 \
    --output tracking.npz \
    --visualize tracking_viz.mp4
```

### `infer_depth.py` - Depth Reconstruction

Reconstruct dense depth map.

```bash
uv run python scripts/infer_depth.py \
    --checkpoint checkpoints/model.pth \
    --video path/to/video.mp4 \
    --frame 0 \
    --output depth.npz \
    --visualize depth_viz.png \
    --batch-size 4096
```

### `infer_pose.py` - Camera Pose Estimation

```bash
# Single frame pose
uv run python scripts/infer_pose.py \
    --checkpoint checkpoints/model.pth \
    --video path/to/video.mp4 \
    --target-frame 10 \
    --reference-frame 0 \
    --output pose.npz

# Full trajectory
uv run python scripts/infer_pose.py \
    --checkpoint checkpoints/model.pth \
    --video path/to/video.mp4 \
    --output trajectory.npz \
    --visualize trajectory_viz.png
```

## Data Preparation

### `download_movi.py` - Download MOVi Dataset

```bash
# Download MOVi-A (small scenes)
uv run python scripts/download_movi.py --variant a --output data/movi

# Download MOVi-E (complex scenes)
uv run python scripts/download_movi.py --variant e --output data/movi
```

### `convert_movi_to_kubric.py` - Convert MOVi to Kubric Format

```bash
uv run python scripts/convert_movi_to_kubric.py \
    --dataset movi_a \
    --split train \
    --num-samples 100 \
    --output-dir data/kubric
```

### `add_tracks_to_movi.py` - Add Point Tracks

Add TAP-Vid compatible point tracks to MOVi data.

```bash
uv run python scripts/add_tracks_to_movi.py \
    --data-dir data/kubric \
    --split val \
    --num-samples 5 \
    --num-points 256
```

### `prepare_kubric_data.py` - Prepare Kubric Data

```bash
uv run python scripts/prepare_kubric_data.py \
    --input data/movi \
    --output data/kubric \
    --split train
```

## Visualization

### `visualize_predictions.py` - Visualize Predictions vs GT

```bash
uv run python scripts/visualize_predictions.py \
    --checkpoint checkpoints/checkpoint_step_0050000.pth \
    --model-config configs/model/vit_b_movi.yaml \
    --data-dir data/kubric \
    --scene-idx 0
```

### `visualize_tracking_video.py` - Create Tracking Video

```bash
uv run python scripts/visualize_tracking_video.py \
    --checkpoint checkpoints/model.pth \
    --data-dir data/kubric \
    --scene-idx 0 \
    --output tracking_video.mp4
```

## Utilities

### `show_model_info.py` - Model Architecture Info

```bash
# Show all models
uv run python scripts/show_model_info.py --model all

# Show specific model
uv run python scripts/show_model_info.py --model vit_b

# Compare models with memory estimates
uv run python scripts/show_model_info.py --compare --memory
```

## Debug Scripts (`scripts/debug/`)

Development and debugging utilities (19 files):

| Script | Purpose |
|--------|---------|
| `debug_loss.py` | Analyze loss computation |
| `debug_gradient_flow.py` | Check gradient propagation |
| `debug_decoder.py` | Debug decoder behavior |
| `debug_tracking.py` | Debug tracking predictions |
| `debug_attention_stats.py` | Analyze attention patterns |
| `debug_training_diagnostics.py` | Training diagnostics |
| `debug_analyze_confidence_gradient.py` | Confidence gradient analysis |
| `debug_check_confidence_exploit.py` | Check confidence exploitation |
| `debug_check_learning.py` | Check if model is learning |
| `debug_investigate_loss.py` | Investigate loss components |
| `analyze_tracking_errors.py` | Visualize tracking errors |

## Test Scripts (`scripts/test/`)

Validation and testing (5 files):

| Script | Purpose |
|--------|---------|
| `test_model.py` | Model architecture tests |
| `test_data.py` | Data pipeline tests |
| `test_training.py` | Training loop tests |
| `test_gradient_fixes.py` | Gradient flow validation |
| `test_combined_fix.py` | Combined fix validation |

Run tests:
```bash
uv run python scripts/test/test_model.py
uv run python scripts/test/test_data.py
```

## Troubleshooting

### Import Errors

Run from project root:
```bash
cd /path/to/D4RT_MasImpl
uv run python scripts/train.py
```

### CUDA Out of Memory

- Reduce batch size
- Enable gradient checkpointing (`training.gradient_checkpointing: true`)
- Use smaller model (ViT-B instead of ViT-L)
