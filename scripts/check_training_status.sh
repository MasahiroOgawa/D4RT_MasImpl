#!/bin/bash
# Quick status check for 50k training

echo "=== 50k Training Status ==="
echo ""

# Check if process is running
if ps aux | grep -q "[t]rain_simple.py"; then
    echo "✓ Training process running"
    ps aux | grep "[t]rain_simple.py" | awk '{print "  CPU:", $3"%, Memory:", $4"%, PID:", $2}'
else
    echo "✗ Training process NOT running"
fi

echo ""

# Check current step
CURRENT_STEP=$(grep "Training:" logs/train_50k_movi.log 2>/dev/null | grep -oP '\d+/50000' | tail -1)
if [ -n "$CURRENT_STEP" ]; then
    CURRENT=$(echo $CURRENT_STEP | cut -d'/' -f1)
    TOTAL=50000
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($CURRENT/$TOTAL)*100}")
    echo "Progress: $CURRENT_STEP ($PERCENT%)"

    REMAINING=$((TOTAL - CURRENT))
    echo "Remaining: $REMAINING steps"
else
    echo "No progress information available"
fi

echo ""

# Check GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU Util: %s%%, Memory Util: %s%%, Memory: %s/%s MB, Temp: %s°C\n", $1, $2, $3, $4, $5}'

echo ""

# Get latest loss
LATEST_LOSS=$(grep "Training:" logs/train_50k_movi.log | grep -oP 'loss=[\d.]+' | tail -1 | cut -d'=' -f2)
if [ -n "$LATEST_LOSS" ]; then
    echo "Latest loss: $LATEST_LOSS"
fi

echo ""

# Check for validation
LAST_VAL=$(grep "Validation loss:" logs/train_50k_movi.log 2>/dev/null | tail -1)
if [ -n "$LAST_VAL" ]; then
    echo "Last validation: $LAST_VAL"
else
    echo "No validation runs yet"
fi

echo ""
echo "Latest checkpoints:"
ls -lt checkpoints/checkpoint_step_*.pth 2>/dev/null | head -3
