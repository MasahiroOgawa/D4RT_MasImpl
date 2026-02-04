#!/usr/bin/env python3
"""Quick evaluation script for intermediate checkpoint testing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from d4rt.models import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset
from d4rt.evaluation.metrics import compute_tapvid_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=10,
        help="Number of scenes to evaluate (default: 10 for quick check)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model_config = OmegaConf.load("configs/model/vit_b_d4rt.yaml")
    model = build_d4rt_model(model_config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    step = ckpt.get("step", "unknown")
    print(f"Loaded step {step}")

    model.eval()

    # Load validation data
    dataset = KubricDataset(
        data_dir="data/kubric",
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )

    # Evaluate on subset
    num_scenes = min(args.num_scenes, len(dataset))
    print(f"Evaluating on {num_scenes} scenes...")

    all_metrics = []
    for i in tqdm(range(num_scenes), desc="Evaluating"):
        sample = dataset[i]
        video = sample["video"].unsqueeze(0).to(device)
        queries = {k: v.unsqueeze(0).to(device) for k, v in sample["queries"].items()}
        targets = {k: v.unsqueeze(0).to(device) for k, v in sample["targets"].items()}

        with torch.no_grad():
            outputs = model(video, queries)

        # Compute metrics
        metrics = compute_tapvid_metrics(
            pred_xyz=outputs["xyz"].cpu().numpy(),
            pred_visibility=torch.sigmoid(outputs["visibility"]).cpu().numpy(),
            gt_xyz=targets["xyz"].cpu().numpy(),
            gt_visibility=targets["visibility"].cpu().numpy(),
        )
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    print(f"\n{'='*60}")
    print(f"INTERMEDIATE EVALUATION (Step {step}, {num_scenes} scenes)")
    print(f"{'='*60}")
    print(
        f"Average Jaccard (AJ):      {avg_metrics.get('average_jaccard', 0):.4f}  (target: 0.304)"
    )
    print(
        f"Avg Pts Within Thresh:     {avg_metrics.get('average_pts_within_thresh', 0):.4f}  (target: 0.410)"
    )
    print(
        f"Occlusion Accuracy (OA):   {avg_metrics.get('occlusion_accuracy', 0):.4f}  (target: 0.875)"
    )
    print(f"{'='*60}")

    # Also check prediction statistics
    with torch.no_grad():
        sample = dataset[0]
        video = sample["video"].unsqueeze(0).to(device)
        queries = {k: v.unsqueeze(0).to(device) for k, v in sample["queries"].items()}
        targets = {k: v.unsqueeze(0).to(device) for k, v in sample["targets"].items()}
        outputs = model(video, queries)

        pred_std = outputs["xyz"].std().item()
        gt_std = targets["xyz"].std().item()
        conf = torch.sigmoid(outputs["confidence"]).mean().item()

    print(f"\nDiagnostics:")
    print(f"  Pred std: {pred_std:.4f}")
    print(f"  GT std:   {gt_std:.4f}")
    print(f"  Scale ratio: {pred_std/gt_std:.4f} (should be > 0.2)")
    print(f"  Mean confidence: {conf:.4f} (should be > 0.5)")


if __name__ == "__main__":
    main()
