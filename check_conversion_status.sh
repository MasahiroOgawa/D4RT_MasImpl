#!/bin/bash
# Quick status check for MOVi conversion

echo "=== MOVi-A Conversion Status ==="
echo ""

# Check if process is running
if ps aux | grep -q "[c]onvert_movi"; then
    echo "✓ Conversion process running"
    ps aux | grep "[c]onvert_movi" | awk '{print "  CPU:", $3"%, Memory:", $4"%, PID:", $2}'
else
    echo "✗ Conversion process NOT running"
fi

echo ""

# Check progress
PROGRESS=$(grep -oP '\d+/9703' logs/convert_movi_full.log 2>/dev/null | tail -1)
if [ -n "$PROGRESS" ]; then
    CURRENT=$(echo $PROGRESS | cut -d'/' -f1)
    TOTAL=9703
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($CURRENT/$TOTAL)*100}")
    echo "Progress: $PROGRESS ($PERCENT%)"

    # Calculate ETA
    REMAINING=$((TOTAL - CURRENT))
    echo "Remaining: $REMAINING samples"

    # Check rate from recent log entries
    RECENT_COUNT=$(tail -100 logs/convert_movi_full.log | grep -oP '\d+/9703' | wc -l)
    if [ $RECENT_COUNT -gt 5 ]; then
        echo "Recent activity: $RECENT_COUNT updates in last 100 log lines"
    fi
else
    echo "No progress information available"
fi

echo ""

# Check data size
if [ -d "data/kubric/train" ]; then
    SIZE=$(du -sh data/kubric/train | cut -f1)
    COUNT=$(ls data/kubric/train | wc -l)
    echo "Converted: $COUNT samples ($SIZE)"
else
    echo "No converted data yet"
fi

echo ""
echo "Latest log tail:"
tail -3 logs/convert_movi_full.log 2>/dev/null | grep -E "it/s|%"
