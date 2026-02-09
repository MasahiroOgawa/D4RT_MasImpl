# D4RT Scripts Reference

Command-line usage for all scripts in the D4RT implementation.

## Directory Structure

```
scripts/
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
├── quick_eval.py         # Quick TAPVid-3D evaluation
├── infer_tracking.py     # Point tracking inference
├── infer_depth.py        # Depth reconstruction
├── infer_pose.py         # Camera pose estimation
├── show_model_info.py    # Model architecture info
├── download_movi.py      # Download MOVi dataset
├── prepare_kubric_data.py # Prepare Kubric data
├── train_tmux.sh         # Tmux-based training with monitoring
├── eval_monitor.sh       # Background evaluation monitor
├── battery_monitor.sh    # Battery-aware training
├── debug/                # Debug scripts
└── test/                 # Test scripts
```

## Training

### `train.py` - Main Training Script

Train D4RT models with Hydra configuration.

```bash
# Train with default config
uv run python scripts/train.py --config-name train_50k_movi_paper --config-path ../configs/training

# Resume from checkpoint
uv run python scripts/train.py --config-name train_50k_movi_paper --config-path ../configs/training +training.resume_from=checkpoints/checkpoint_step_0005000.pth

# Override parameters
uv run python scripts/train.py --config-name train_50k_movi_paper --config-path ../configs/training training.batch_size=2 optimizer.lr=5e-5
```

### `train_tmux.sh` - Tmux Training Wrapper

Run training in tmux with automatic monitoring.

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

## Evaluation

### `evaluate.py` - Comprehensive Evaluation

```bash
# Evaluate on Kubric
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_latest.pth \
    --dataset kubric \
    --data-dir data/kubric \
    --split val \
    --output results.json

# Specific tasks
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/model.pth \
    --dataset kubric \
    --tasks tracking depth \
    --max-samples 100
```

### `quick_eval.py` - Quick TAPVid-3D Evaluation

Fast evaluation using TAPVid-3D metrics.

```bash
# Quick eval on 5 scenes
uv run python scripts/quick_eval.py \
    --checkpoint checkpoints/checkpoint_latest.pth \
    --num_scenes 5 \
    --alignment scale_shift
```

**Metrics:**
- **AJ (Average Jaccard)**: Target 0.304
- **APD3D (Average Position Distance 3D)**: Target 0.410
- **OA (Occlusion Accuracy)**: Target 0.875

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

### `prepare_kubric_data.py` - Prepare Kubric Data

```bash
uv run python scripts/prepare_kubric_data.py \
    --input data/movi \
    --output data/kubric \
    --split train
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

Development and debugging utilities:

| Script | Purpose |
|--------|---------|
| `debug_loss.py` | Analyze loss computation |
| `debug_gradient_flow.py` | Check gradient propagation |
| `debug_decoder.py` | Debug decoder behavior |
| `debug_tracking.py` | Debug tracking predictions |
| `debug_attention_stats.py` | Analyze attention patterns |
| `debug_training_diagnostics.py` | Training diagnostics |

## Test Scripts (`scripts/test/`)

Validation and testing:

| Script | Purpose |
|--------|---------|
| `test_model.py` | Model architecture tests |
| `test_data.py` | Data pipeline tests |
| `test_training.py` | Training loop tests |
| `test_gradient_fixes.py` | Gradient flow validation |

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
