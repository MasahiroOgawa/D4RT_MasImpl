#!/bin/bash
# Background evaluation monitor - runs quick eval every N seconds
# Usage: ./scripts/eval_monitor.sh [log_file] [interval_seconds]

LOG_FILE="${1:-outputs/eval_monitor.log}"
INTERVAL="${2:-600}"  # Default 10 minutes
EVAL_SCENES=5
PROJECT_DIR="/home/mas/proj/study/D4RT_MasImpl"

cd "$PROJECT_DIR"

echo "========================================" | tee "$LOG_FILE"
echo "Evaluation Monitor Started: $(date)" | tee -a "$LOG_FILE"
echo "Interval: $((INTERVAL/60)) minutes" | tee -a "$LOG_FILE"
echo "Scenes: $EVAL_SCENES" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while true; do
    # Check for checkpoint
    if [ ! -f "checkpoints/checkpoint_latest.pth" ]; then
        echo "[$(date '+%H:%M:%S')] Waiting for checkpoint..." | tee -a "$LOG_FILE"
        sleep 60
        continue
    fi

    # Get step from checkpoint filename
    CKPT_FILE=$(readlink -f checkpoints/checkpoint_latest.pth)
    STEP=$(basename "$CKPT_FILE" | grep -oP '\d+' || echo "unknown")

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step $STEP" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # Run quick eval (in background to not block)
    EVAL_OUTPUT=$(uv run python scripts/quick_eval.py \
        --checkpoint checkpoints/checkpoint_latest.pth \
        --num_scenes $EVAL_SCENES \
        --alignment scale_shift 2>&1)

    # Extract and display metrics
    AJ=$(echo "$EVAL_OUTPUT" | grep "Average Jaccard" | grep -oP '[\d.]+' | head -1)
    APD=$(echo "$EVAL_OUTPUT" | grep "Avg Pts Within" | grep -oP '[\d.]+' | head -1)
    OA=$(echo "$EVAL_OUTPUT" | grep "Occlusion Accuracy" | grep -oP '[\d.]+' | head -1)

    echo "Metrics:" | tee -a "$LOG_FILE"
    echo "  AJ:    ${AJ:-N/A}  (target: 0.304)" | tee -a "$LOG_FILE"
    echo "  APD3D: ${APD:-N/A}  (target: 0.410)" | tee -a "$LOG_FILE"
    echo "  OA:    ${OA:-N/A}  (target: 0.875)" | tee -a "$LOG_FILE"

    # Run Z correlation analysis
    echo "" | tee -a "$LOG_FILE"
    echo "Z Correlation Analysis:" | tee -a "$LOG_FILE"
    uv run python -c "
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

# Average over 3 scenes
all_corr = {'X': [], 'Y': [], 'Z': []}
all_pred_z = []
all_gt_z = []

for i in range(3):
    sample = dataset[i]
    with torch.no_grad():
        video = sample['video'].unsqueeze(0).cuda()
        queries = {k: v.unsqueeze(0).cuda() for k, v in sample['queries'].items()}
        outputs = model(video, queries)

    pred_xyz = outputs['xyz'][0].cpu().numpy()
    gt_xyz = sample['targets']['xyz'].numpy()

    for j, axis in enumerate(['X', 'Y', 'Z']):
        corr = np.corrcoef(pred_xyz[:, j], gt_xyz[:, j])[0, 1]
        all_corr[axis].append(corr)

    all_pred_z.extend(pred_xyz[:, 2].tolist())
    all_gt_z.extend(gt_xyz[:, 2].tolist())

# Print results
for axis in ['X', 'Y', 'Z']:
    mean_corr = np.mean(all_corr[axis])
    print(f'  {axis}: corr={mean_corr:.3f}')

pred_z_arr = np.array(all_pred_z)
gt_z_arr = np.array(all_gt_z)
print(f'  pred_z: mean={pred_z_arr.mean():.2f}, std={pred_z_arr.std():.2f}, negative={100*(pred_z_arr<0).mean():.1f}%')
print(f'  gt_z:   mean={gt_z_arr.mean():.2f}, std={gt_z_arr.std():.2f}')
" 2>&1 | grep -E "^\s+[XYZ]:|pred_z:|gt_z:" | tee -a "$LOG_FILE"

    # Check best_metrics.json for Z loss
    if [ -f "checkpoints/best_metrics.json" ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "Loss metrics:" | tee -a "$LOG_FILE"
        python3 -c "
import json
with open('checkpoints/best_metrics.json') as f:
    m = json.load(f)['all_metrics']
print(f\"  loss_depth_aux: {m.get('train/loss_depth_aux_raw', 'N/A')}\")
print(f\"  loss_3d_raw: {m.get('train/loss_3d_raw', 'N/A')}\")
print(f\"  pred_z_negative_ratio: {m.get('train/pred_z_negative_ratio', 'N/A')}\")
" 2>&1 | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "Next eval in $((INTERVAL/60)) min..." | tee -a "$LOG_FILE"

    sleep $INTERVAL
done
