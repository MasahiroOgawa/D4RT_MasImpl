#!/bin/bash
# Battery monitor - auto sleep when battery < 10%

THRESHOLD=10
CHECK_INTERVAL=60  # seconds

echo "Battery monitor started (threshold: ${THRESHOLD}%)"
echo "Training PID to monitor: $(pgrep -f 'train.py')"

# Function to trigger emergency checkpoint save in training process
save_training_checkpoint() {
    # Find training process (train.py)
    TRAIN_PID=$(pgrep -f 'python.*train.py' | head -1)

    if [ -n "$TRAIN_PID" ]; then
        echo "$(date): Sending SIGUSR1 to training process (PID: $TRAIN_PID) to save checkpoint..."
        kill -USR1 "$TRAIN_PID"
        # Wait for checkpoint to be saved (up to 30 seconds)
        sleep 10
        echo "$(date): Checkpoint save signal sent, waited 10 seconds for save"
    else
        echo "$(date): No training process found, skipping checkpoint save"
    fi
}

# Function to suspend system with fallback methods
suspend_system() {
    echo "$(date): Attempting to suspend system..."

    # First, save training checkpoint
    save_training_checkpoint

    # Save battery info
    echo "Last battery level: ${CAPACITY}%" > /tmp/battery_suspend.log
    echo "Time: $(date)" >> /tmp/battery_suspend.log

    # Method 1: Try systemctl with sudo (requires passwordless sudo setup)
    if sudo -n systemctl suspend 2>/dev/null; then
        echo "$(date): Suspended via sudo systemctl suspend"
        return 0
    fi

    # Method 2: Try loginctl (works for logged-in users via polkit)
    if loginctl suspend 2>/dev/null; then
        echo "$(date): Suspended via loginctl suspend"
        return 0
    fi

    # Method 3: Try dbus (works if polkit allows it)
    if dbus-send --system --print-reply \
        --dest=org.freedesktop.login1 \
        /org/freedesktop/login1 \
        org.freedesktop.login1.Manager.Suspend boolean:true 2>/dev/null; then
        echo "$(date): Suspended via dbus"
        return 0
    fi

    # Method 4: Direct systemctl (might work with polkit)
    if systemctl suspend 2>/dev/null; then
        echo "$(date): Suspended via systemctl suspend"
        return 0
    fi

    # All methods failed
    echo "$(date): ERROR - All suspend methods failed!"
    echo "$(date): Please configure one of:"
    echo "  1. Add 'mas ALL=(ALL) NOPASSWD: /usr/bin/systemctl suspend' to /etc/sudoers"
    echo "  2. Ensure polkit allows suspend for your user"
    return 1
}

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

        if ! suspend_system; then
            # If suspend fails, keep trying every 10 seconds
            echo "$(date): CRITICAL - Suspend failed! Retrying in 10 seconds..."
            sleep 10
            continue
        fi

        # After wake, wait a bit before checking again
        sleep 30
    fi

    sleep $CHECK_INTERVAL
done
