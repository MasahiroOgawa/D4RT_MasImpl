#!/bin/bash
# Monitor training every 10 minutes: run quick eval and check Z metrics
# Usage: ./scripts/monitor_training.sh [log_file]

LOG_FILE="${1:-outputs/training_monitor_$(date +%Y%m%d_%H%M%S).log}"
EVAL_SCENES=5
INTERVAL=600  # 10 minutes

echo "========================================" | tee -a "$LOG_FILE"
echo "Training Monitor Started: $(date)" | tee -a "$LOG_FILE"
echo "Evaluation interval: $((INTERVAL/60)) minutes" | tee -a "$LOG_FILE"
echo "Scenes per eval: $EVAL_SCENES" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while true; do
    # Get current checkpoint step
    CKPT=$(readlink -f checkpoints/checkpoint_latest.pth 2>/dev/null)
    if [ -z "$CKPT" ]; then
        echo "[$(date +%H:%M:%S)] No checkpoint found, waiting..." | tee -a "$LOG_FILE"
        sleep 60
        continue
    fi

    STEP=$(basename "$CKPT" | grep -oP '\d+')

    echo "" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step $STEP" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"

    # Run quick eval with both alignments
    echo "Running evaluation..." | tee -a "$LOG_FILE"

    # Eval with scale_shift alignment
    EVAL_OUTPUT=$(uv run python scripts/quick_eval.py \
        --checkpoint checkpoints/checkpoint_latest.pth \
        --num_scenes $EVAL_SCENES \
        --alignment scale_shift 2>&1 | tail -20)

    # Extract metrics
    AJ=$(echo "$EVAL_OUTPUT" | grep "Average Jaccard" | grep -oP '[\d.]+' | head -1)
    APD=$(echo "$EVAL_OUTPUT" | grep "Avg Pts Within" | grep -oP '[\d.]+' | head -1)
    OA=$(echo "$EVAL_OUTPUT" | grep "Occlusion Accuracy" | grep -oP '[\d.]+' | head -1)
    SCALE_RATIO=$(echo "$EVAL_OUTPUT" | grep "Scale ratio" | grep -oP '[\d.]+' | head -1)

    echo "  AJ (scale_shift): $AJ  (target: 0.304)" | tee -a "$LOG_FILE"
    echo "  APD3D:            $APD  (target: 0.410)" | tee -a "$LOG_FILE"
    echo "  OA:               $OA  (target: 0.875)" | tee -a "$LOG_FILE"
    echo "  Scale ratio:      $SCALE_RATIO" | tee -a "$LOG_FILE"

    # Get Z metrics from training log
    TRAIN_LOG=$(ls -t outputs/training_*.log 2>/dev/null | head -1)
    if [ -n "$TRAIN_LOG" ]; then
        # Get recent loss values
        RECENT_LOSS=$(tail -100 "$TRAIN_LOG" | grep -oP 'loss=[\d.]+' | tail -1)
        echo "  Recent loss: $RECENT_LOSS" | tee -a "$LOG_FILE"
    fi

    # Run Z correlation analysis
    echo "" | tee -a "$LOG_FILE"
    echo "Z Correlation Analysis:" | tee -a "$LOG_FILE"
    Z_ANALYSIS=$(uv run python -c "
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from omegaconf import OmegaConf
from d4rt.models import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset

model_config = OmegaConf.load('configs/model/vit_b_movi.yaml')
model = build_d4rt_model(model_config).cuda()
ckpt = torch.load('checkpoints/checkpoint_latest.pth', map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

dataset = KubricDataset('data/kubric', 'val', num_frames=24, resolution=(256, 256), num_queries=64)
sample = dataset[0]

with torch.no_grad():
    video = sample['video'].unsqueeze(0).cuda()
    queries = {k: v.unsqueeze(0).cuda() for k, v in sample['queries'].items()}
    outputs = model(video, queries)

pred_xyz = outputs['xyz'][0].cpu().numpy()
gt_xyz = sample['targets']['xyz'].numpy()

for i, axis in enumerate(['X', 'Y', 'Z']):
    corr = np.corrcoef(pred_xyz[:, i], gt_xyz[:, i])[0, 1]
    print(f'  {axis}: corr={corr:.3f}, pred_mean={pred_xyz[:, i].mean():.2f}, gt_mean={gt_xyz[:, i].mean():.2f}')
" 2>&1 | grep -E "^\s+[XYZ]:")

    echo "$Z_ANALYSIS" | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "Next eval in $((INTERVAL/60)) min..." | tee -a "$LOG_FILE"

    sleep $INTERVAL
done
