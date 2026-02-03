#!/bin/bash
# Run evaluation after paper architecture training completes
# Usage: ./scripts/run_evaluation_after_training.sh

set -e

CHECKPOINT_DIR="checkpoints"
MODEL_CONFIG="configs/model/vit_b_d4rt.yaml"
DATA_DIR="data/kubric"
RESULTS_DIR="results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Check for latest paper architecture checkpoint
# Paper architecture checkpoints are ~2.6GB (vs ~1.6GB for old architecture)
LATEST_CHECKPOINT=$(ls -la "$CHECKPOINT_DIR"/checkpoint_step_*.pth 2>/dev/null | \
    awk '$5 > 2000000000 {print $NF}' | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No paper architecture checkpoint found (looking for >2GB files)"
    echo "Available checkpoints:"
    ls -lah "$CHECKPOINT_DIR"/checkpoint_step_*.pth 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

echo "=========================================="
echo "D4RT Paper Architecture Evaluation"
echo "=========================================="
echo "Checkpoint: $LATEST_CHECKPOINT"
echo "Model config: $MODEL_CONFIG"
echo "Data directory: $DATA_DIR"
echo ""

# Extract step number from checkpoint name
STEP=$(echo "$LATEST_CHECKPOINT" | grep -oE '[0-9]+' | tail -1)
OUTPUT_FILE="$RESULTS_DIR/eval_paper_arch_step_${STEP}.json"

echo "Output: $OUTPUT_FILE"
echo ""
echo "Starting evaluation..."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run evaluation
python scripts/evaluate_movi_tracks.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --model-config "$MODEL_CONFIG" \
    --data-dir "$DATA_DIR" \
    --split val \
    --output "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Paper targets for comparison:"
echo "  - AJ (Average Jaccard): 0.304"
echo "  - APD3D (Average % within Threshold): 0.410"
echo "  - OA (Occlusion Accuracy): should be high"
echo ""
cat "$OUTPUT_FILE"
