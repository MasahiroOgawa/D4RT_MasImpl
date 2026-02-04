#!/bin/bash
# Monitor training progress and run intermediate evaluation
#
# Usage:
#   ./monitor_training.sh          # Full monitoring with evaluation
#   ./monitor_training.sh --quick  # Quick status only (no evaluation)
#   ./monitor_training.sh --eval   # Run evaluation only on latest checkpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
QUICK_MODE=false
EVAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --quick|-q) QUICK_MODE=true ;;
        --eval|-e) EVAL_ONLY=true ;;
    esac
done

if [ "$EVAL_ONLY" = true ]; then
    # Only run evaluation
    LATEST_CKPT=$(ls -t "$PROJECT_DIR/checkpoints/"checkpoint_step_*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Evaluating: $(basename "$LATEST_CKPT")"
        cd "$PROJECT_DIR"
        python scripts/quick_eval.py --checkpoint "$LATEST_CKPT" --num_scenes 20
    else
        echo "No checkpoint available for evaluation"
    fi
    exit 0
fi

echo "=== Training Status ==="
ps aux | grep "train.py" | grep -v grep | head -1 || echo "Training not running"

echo ""
echo "=== Current Step ==="
tail -5 "$PROJECT_DIR/logs/train_fixed_data.log" 2>/dev/null | grep -oP '\d+/50000' | tail -1 || echo "Unknown"

echo ""
echo "=== Recent Loss Values ==="
tail -10 "$PROJECT_DIR/logs/train_fixed_data.log" 2>/dev/null | grep -oP "loss=[\d.]+" | tail -5

echo ""
echo "=== Checkpoints ==="
ls -la "$PROJECT_DIR/checkpoints/"checkpoint_step_*.pth 2>/dev/null | tail -5 || echo "No checkpoints yet"

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

# Skip evaluation in quick mode
if [ "$QUICK_MODE" = true ]; then
    echo ""
    echo "(Quick mode: evaluation skipped. Run with --eval for evaluation)"
    exit 0
fi

# Run intermediate evaluation on latest checkpoint if available
echo ""
echo "=== Intermediate Evaluation ==="
LATEST_CKPT=$(ls -t "$PROJECT_DIR/checkpoints/"checkpoint_step_*.pth 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ]; then
    echo "Evaluating: $(basename "$LATEST_CKPT")"
    cd "$PROJECT_DIR"
    python scripts/quick_eval.py --checkpoint "$LATEST_CKPT" --num_scenes 10
else
    echo "No checkpoint available for evaluation yet"
    echo "First checkpoint will be saved at step 5000"
fi
