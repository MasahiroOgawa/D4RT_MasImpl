#!/bin/bash
# Tmux-based training script with battery monitoring and evaluation
# Usage: ./scripts/train_tmux.sh [config_name] [resume_checkpoint]
#
# Features:
# - Runs training in tmux (survives terminal close / laptop lid close)
# - Battery monitoring: saves checkpoint and suspends when < 10%
# - Background evaluation every 1000 iterations with Z correlation analysis
# - Easy to attach/detach: tmux attach -t d4rt
#
# Examples:
#   ./scripts/train_tmux.sh                                    # Default config (fresh start)
#   ./scripts/train_tmux.sh train_50k_movi_paper               # Specific config
#   ./scripts/train_tmux.sh train_50k_movi_paper checkpoint.pth # Resume from checkpoint

set -e

# Configuration
SESSION_NAME="d4rt"
CONFIG_NAME="${1:-train_50k_movi_paper}"
RESUME_CHECKPOINT="${2:-}"
PROJECT_DIR="/home/mas/proj/study/D4RT_MasImpl"
LOG_DIR="${PROJECT_DIR}/outputs"
EVAL_INTERVAL=1000  # Every 1000 steps

# Create log directory
mkdir -p "$LOG_DIR"

# Kill existing session if running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Killing existing tmux session: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

# Clear checkpoints for fresh start (unless resuming)
if [ -z "$RESUME_CHECKPOINT" ]; then
    echo "Clearing checkpoints for fresh start..."
    rm -f "${PROJECT_DIR}/checkpoints/"*.pth "${PROJECT_DIR}/checkpoints/"*.json 2>/dev/null || true
fi

# Build training command
TRAIN_CMD="cd $PROJECT_DIR && uv run python scripts/train.py --config-name $CONFIG_NAME --config-path ../configs/training"
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD +training.resume_from=$RESUME_CHECKPOINT"
fi

# Create timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="${LOG_DIR}/training_${CONFIG_NAME}_${TIMESTAMP}.log"
EVAL_LOG="${LOG_DIR}/eval_monitor_${TIMESTAMP}.log"
BATTERY_LOG="${LOG_DIR}/battery_monitor_${TIMESTAMP}.log"

echo "=============================================="
echo "D4RT Training in Tmux"
echo "=============================================="
echo "Session:    $SESSION_NAME"
echo "Config:     $CONFIG_NAME"
echo "Resume:     ${RESUME_CHECKPOINT:-None (fresh start)}"
echo "Train log:  $TRAIN_LOG"
echo "Eval log:   $EVAL_LOG"
echo "Eval interval: $EVAL_INTERVAL steps"
echo "=============================================="

# Create tmux session with training window
tmux new-session -d -s "$SESSION_NAME" -n training -c "$PROJECT_DIR"

# Start training in the first window
tmux send-keys -t "${SESSION_NAME}:training" "$TRAIN_CMD 2>&1 | tee $TRAIN_LOG" Enter

# Wait a moment for training to start
sleep 2

# Create evaluation monitor window (background eval every 1000 steps)
tmux new-window -t "$SESSION_NAME" -n eval -c "$PROJECT_DIR"
tmux send-keys -t "${SESSION_NAME}:eval" "sleep 120 && bash scripts/eval_monitor.sh $EVAL_LOG $EVAL_INTERVAL" Enter

# Create battery monitor window
tmux new-window -t "$SESSION_NAME" -n battery -c "$PROJECT_DIR"
tmux send-keys -t "${SESSION_NAME}:battery" "bash scripts/battery_monitor.sh 2>&1 | tee $BATTERY_LOG" Enter

# Create a status window for quick view
tmux new-window -t "$SESSION_NAME" -n status -c "$PROJECT_DIR"
tmux send-keys -t "${SESSION_NAME}:status" "watch -n 30 'echo \"=== Training ===\"; tail -3 $TRAIN_LOG 2>/dev/null; echo; echo \"=== Latest Eval ===\"; tail -15 $EVAL_LOG 2>/dev/null; echo; echo \"=== Checkpoints ===\"; ls -lt checkpoints/*.pth 2>/dev/null | head -3'" Enter

# Select training window
tmux select-window -t "${SESSION_NAME}:training"

echo ""
echo "Training started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach to session:  tmux attach -t $SESSION_NAME"
echo "  Detach from session: Ctrl+b, then d"
echo "  Switch windows:      Ctrl+b, then n (next) or p (previous)"
echo "  Kill session:        tmux kill-session -t $SESSION_NAME"
echo ""
echo "Windows:"
echo "  0: training - Training process"
echo "  1: eval     - Background evaluation (every 1000 steps)"
echo "  2: battery  - Battery monitor"
echo "  3: status   - Quick status view"
echo ""
echo "You can safely close this terminal. Training will continue."
echo "=============================================="
