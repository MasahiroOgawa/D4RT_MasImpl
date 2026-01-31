#!/bin/bash
# Battery monitor - auto sleep when battery < 10%

THRESHOLD=10
CHECK_INTERVAL=60  # seconds

echo "Battery monitor started (threshold: ${THRESHOLD}%)"
echo "Training PID to monitor: $(pgrep -f 'train.py')"

while true; do
    CAPACITY=$(cat /sys/class/power_supply/BAT1/capacity 2>/dev/null)
    STATUS=$(cat /sys/class/power_supply/BAT1/status 2>/dev/null)

    if [ -z "$CAPACITY" ]; then
        echo "$(date): Cannot read battery level"
        sleep $CHECK_INTERVAL
        continue
    fi

    echo "$(date): Battery ${CAPACITY}% (${STATUS})"

    # Only sleep if discharging AND below threshold
    if [ "$STATUS" = "Discharging" ] && [ "$CAPACITY" -lt "$THRESHOLD" ]; then
        echo "$(date): Battery critical (${CAPACITY}%)! Suspending system..."

        # Save training checkpoint info
        echo "Last battery level: ${CAPACITY}%" > /tmp/battery_suspend.log
        echo "Time: $(date)" >> /tmp/battery_suspend.log

        # Suspend system
        systemctl suspend

        # After wake, wait a bit before checking again
        sleep 30
    fi

    sleep $CHECK_INTERVAL
done
