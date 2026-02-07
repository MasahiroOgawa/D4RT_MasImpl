#!/bin/bash
# Monitor training by running quick evaluation periodically
# Usage: ./scripts/monitor_training_eval.sh [interval_minutes]
#
# Default: evaluates every 10 minutes
# Logs to: outputs/training_monitor.log
#
# Runs evaluation in BACKGROUND with low priority to not interfere with training

INTERVAL_MIN=${1:-10}
INTERVAL_SEC=$((INTERVAL_MIN * 60))
LOG_FILE="outputs/training_monitor.log"
NUM_SCENES=5

mkdir -p outputs

echo "========================================" | tee -a $LOG_FILE
echo "Training Monitor Started: $(date)" | tee -a $LOG_FILE
echo "Evaluation interval: ${INTERVAL_MIN} minutes" | tee -a $LOG_FILE
echo "Scenes per eval: ${NUM_SCENES}" | tee -a $LOG_FILE
echo "Mode: Background evaluation (non-blocking)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# Function to run evaluation in background
run_eval() {
    local CHECKPOINT="$1"
    local STEP="$2"

    {
        echo ""
        echo "----------------------------------------"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step $STEP"
        echo "----------------------------------------"

        # Run with nice (low priority) to not compete with training
        nice -n 10 uv run python scripts/quick_eval.py --checkpoint "$CHECKPOINT" --num_scenes $NUM_SCENES 2>&1 | \
            grep -E "(AJ|Pts Within|Occlusion|Scale ratio|confidence)"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Eval complete"
    } >> $LOG_FILE 2>&1
}

while true; do
    # Find latest checkpoint
    LATEST=$(ls -t checkpoints/checkpoint_step_*.pth 2>/dev/null | head -1)

    if [ -n "$LATEST" ]; then
        STEP=$(echo "$LATEST" | grep -oP 'step_\K[0-9]+')

        # Run evaluation in background (non-blocking)
        run_eval "$LATEST" "$STEP" &
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started background eval for step $STEP"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No checkpoint found" | tee -a $LOG_FILE
    fi

    echo "Next eval in ${INTERVAL_MIN} min..."
    sleep $INTERVAL_SEC
done
