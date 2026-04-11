#!/usr/bin/env python3
"""Evaluate D4RT on TAP-Vid-3D train and test splits.

Runs inference on both splits to check:
1. Training fit (overfit check on train split)
2. Generalization (test split performance)

Usage:
    python scripts/eval_tapvid3d_splits.py \
        --checkpoint checkpoints_tapvid3d/checkpoint_latest.pth
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from d4rt.data.datasets.tapvid3d import TAPVid3DDataset
from d4rt.models import build_d4rt_model


def run_inference_on_dataset(model, dataset, device, max_samples=None):
    """Run D4RT inference on a TAPVid3DDataset, return per-sample metrics."""
    model.eval()
    results = []

    n = min(len(dataset), max_samples) if max_samples else len(dataset)

    for idx in tqdm(range(n), desc="Evaluating"):
        sample = dataset[idx]

        # Move to device
        video = sample["video"].unsqueeze(0).to(device)  # [1, T, 3, H, W]
        queries = {k: v.unsqueeze(0).to(device) for k, v in sample["queries"].items()}

        gt_xyz = sample["targets"]["xyz"].numpy()  # [N, 3]
        gt_vis = sample["targets"]["visibility"].numpy()[:, 0]  # [N]
        gt_uv = sample["targets"]["uv"].numpy()  # [N, 2]

        with torch.no_grad():
            outputs = model(video, queries)

        pred_xyz = outputs["xyz"][0].cpu().numpy()  # [N, 3]
        pred_vis = torch.sigmoid(outputs["visibility"][0]).cpu().numpy()[:, 0]  # [N]
        pred_uv = outputs["uv"][0].cpu().numpy()  # [N, 2]

        # --- Metrics ---
        visible = gt_vis > 0.5
        n_visible = visible.sum()

        # 3D error (with median scale alignment)
        if n_visible > 0:
            gt_norm = np.linalg.norm(gt_xyz[visible], axis=-1)
            pred_norm = np.linalg.norm(pred_xyz[visible], axis=-1)
            med_gt = np.median(gt_norm) if len(gt_norm) > 0 else 1.0
            med_pred = np.median(pred_norm) if len(pred_norm) > 0 else 1.0
            scale = med_gt / (med_pred + 1e-8)
            pred_aligned = pred_xyz * scale

            error_3d = np.linalg.norm(pred_aligned[visible] - gt_xyz[visible], axis=-1)
            mean_3d_error = error_3d.mean()

            # APD3D: % within thresholds
            thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]
            apd = {f"within_{t}m": float((error_3d < t).mean()) for t in thresholds}
        else:
            mean_3d_error = float("nan")
            apd = {f"within_{t}m": float("nan") for t in [0.05, 0.1, 0.2, 0.5, 1.0]}
            scale = float("nan")

        # 2D error
        if n_visible > 0:
            error_2d = np.linalg.norm(pred_uv[visible] - gt_uv[visible], axis=-1)
            mean_2d_error = error_2d.mean()
        else:
            mean_2d_error = float("nan")

        # Occlusion accuracy
        pred_occ = pred_vis < 0.5
        gt_occ = gt_vis < 0.5
        oa = float((pred_occ == gt_occ).mean())

        results.append(
            {
                "scene": sample["metadata"]["scene_id"],
                "subset": sample["metadata"]["subset"],
                "mean_3d_error": float(mean_3d_error),
                "mean_2d_error": float(mean_2d_error),
                "occlusion_accuracy": oa,
                "scale_factor": float(scale),
                "n_visible": int(n_visible),
                **apd,
            }
        )

    return results


def print_results(results, split_name):
    """Print aggregated results."""
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SPLIT ({len(results)} samples)")
    print(f"{'='*60}")

    # Filter out NaN results
    valid = [r for r in results if not np.isnan(r["mean_3d_error"])]
    if not valid:
        print("  No valid results (all samples had 0 visible points)")
        return

    keys = [
        "mean_3d_error",
        "mean_2d_error",
        "occlusion_accuracy",
        "within_0.05m",
        "within_0.1m",
        "within_0.2m",
        "within_0.5m",
        "within_1.0m",
    ]

    for k in keys:
        vals = [r[k] for r in valid if not np.isnan(r[k])]
        if vals:
            print(
                f"  {k:25s}: mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  std={np.std(vals):.4f}"
            )

    # Per-subset breakdown
    subsets = set(r["subset"] for r in valid)
    for subset in sorted(subsets):
        sub_results = [r for r in valid if r["subset"] == subset]
        print(f"\n  [{subset}] ({len(sub_results)} samples)")
        for k in ["mean_3d_error", "within_0.5m", "occlusion_accuracy"]:
            vals = [r[k] for r in sub_results if not np.isnan(r[k])]
            if vals:
                print(f"    {k:25s}: mean={np.mean(vals):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-config", type=str, default="configs/model/vit_b_movi.yaml")
    parser.add_argument("--data-dir", type=str, default="data/tapvid3d")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Build model
    print("Building model...")
    config = OmegaConf.load(args.model_config)
    model = build_d4rt_model(config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded step {step}")

    # Create train and test datasets (same split as training)
    common = dict(
        data_dir=args.data_dir,
        num_frames=24,
        resolution=(256, 256),
        num_queries=256,
        subsets=["drivetrack", "pstudio"],
        train_ratio=0.8,
        seed=42,
    )
    train_ds = TAPVid3DDataset(split="train", **common)
    test_ds = TAPVid3DDataset(split="val", **common)

    # Evaluate both splits
    print(f"\nEvaluating on TRAIN split ({len(train_ds)} samples)...")
    train_results = run_inference_on_dataset(model, train_ds, args.device, args.max_samples)
    print_results(train_results, "train")

    print(f"\nEvaluating on TEST split ({len(test_ds)} samples)...")
    test_results = run_inference_on_dataset(model, test_ds, args.device, args.max_samples)
    print_results(test_results, "test")

    # Save results
    if args.output:
        out = {"step": step, "train": train_results, "test": test_results}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
