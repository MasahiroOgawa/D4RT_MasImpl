#!/usr/bin/env python3
"""Visualize D4RT tracking results.

Loads a checkpoint and visualizes predicted vs ground truth tracks on sample videos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from d4rt.models import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset
from omegaconf import OmegaConf


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model config
    config_path = Path(__file__).parent.parent.parent / "configs" / "model" / "vit_b_movi.yaml"
    model_config = OmegaConf.load(config_path)

    # Build model
    model = build_d4rt_model(model_config)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def run_inference(model, sample, device):
    """Run inference on a sample."""
    video = sample["video"].unsqueeze(0).to(device)  # (1, T, 3, H, W)
    queries = {k: v.unsqueeze(0).to(device) for k, v in sample["queries"].items()}

    with torch.no_grad():
        output = model(video, queries)

    return output


def visualize_tracks(sample, output, save_path: str, num_frames: int = 8, num_points: int = 16):
    """Visualize predicted vs ground truth tracks."""
    video = sample["video"].numpy()  # (T, 3, H, W)
    T, _, H, W = video.shape

    # Get GT data
    gt_uv = sample["targets"]["uv"].numpy()  # (Q, 2)
    gt_vis = sample["targets"]["visibility"].numpy().squeeze()  # (Q,)
    # Query UV from separate u, v keys
    query_u = sample["queries"]["u"].numpy()  # (Q,)
    query_v = sample["queries"]["v"].numpy()  # (Q,)
    query_uv = np.stack([query_u, query_v], axis=-1)  # (Q, 2)
    t_src = sample["queries"]["t_src"].numpy().astype(int)  # (Q,)
    t_tgt = sample["queries"]["t_tgt"].numpy().astype(int)  # (Q,)

    # Get predictions
    pred_uv = output["uv"].squeeze(0).cpu().numpy()  # (Q, 2)
    pred_conf = torch.sigmoid(output["confidence"]).squeeze(0).squeeze(-1).cpu().numpy()  # (Q,)

    # Select subset of points
    num_points = min(num_points, len(gt_uv))
    indices = np.random.choice(len(gt_uv), num_points, replace=False)

    # Select frames to show
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    # Create figure
    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))

    # Colors for different points
    colors = plt.cm.tab20(np.linspace(0, 1, num_points))

    for col, fi in enumerate(frame_indices):
        # Get frame
        frame = video[fi].transpose(1, 2, 0)  # (H, W, 3)

        # Top row: Ground truth
        axes[0, col].imshow(frame)
        axes[0, col].set_title(f"Frame {fi} (GT)")
        axes[0, col].axis("off")

        # Bottom row: Predictions
        axes[1, col].imshow(frame)
        axes[1, col].set_title(f"Frame {fi} (Pred)")
        axes[1, col].axis("off")

        # Plot points
        for i, idx in enumerate(indices):
            # Check if this query's target frame matches current frame
            if t_tgt[idx] == fi:
                # Ground truth (top)
                gt_x = gt_uv[idx, 0] * (W - 1)
                gt_y = gt_uv[idx, 1] * (H - 1)
                marker = "o" if gt_vis[idx] > 0.5 else "x"
                axes[0, col].scatter(
                    gt_x,
                    gt_y,
                    c=[colors[i]],
                    marker=marker,
                    s=100,
                    edgecolors="white",
                    linewidths=1,
                )

                # Prediction (bottom)
                pred_x = pred_uv[idx, 0] * (W - 1)
                pred_y = pred_uv[idx, 1] * (H - 1)
                alpha = pred_conf[idx]
                axes[1, col].scatter(
                    pred_x,
                    pred_y,
                    c=[colors[i]],
                    marker="o",
                    s=100,
                    alpha=alpha,
                    edgecolors="white",
                    linewidths=1,
                )

            # Also show source point
            if t_src[idx] == fi:
                src_x = query_uv[idx, 0] * (W - 1)
                src_y = query_uv[idx, 1] * (H - 1)
                axes[0, col].scatter(
                    src_x, src_y, c=[colors[i]], marker="s", s=80, edgecolors="black", linewidths=2
                )
                axes[1, col].scatter(
                    src_x, src_y, c=[colors[i]], marker="s", s=80, edgecolors="black", linewidths=2
                )

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor="gray", edgecolor="black", label="Source (square)"),
        mpatches.Patch(facecolor="gray", edgecolor="white", label="Target visible (circle)"),
        plt.Line2D(
            [0],
            [0],
            marker="x",
            color="gray",
            linestyle="None",
            markersize=10,
            label="Target occluded",
        ),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {save_path}")


def visualize_error_distribution(sample, output, save_path: str):
    """Visualize error distribution."""
    H, W = sample["video"].shape[2], sample["video"].shape[3]

    gt_uv = sample["targets"]["uv"].numpy()
    gt_vis = sample["targets"]["visibility"].numpy().squeeze()
    pred_uv = output["uv"].squeeze(0).cpu().numpy()
    pred_conf = torch.sigmoid(output["confidence"]).squeeze(0).squeeze(-1).cpu().numpy()

    # Compute errors in pixels
    errors = np.sqrt(
        ((gt_uv[:, 0] - pred_uv[:, 0]) * (W - 1)) ** 2
        + ((gt_uv[:, 1] - pred_uv[:, 1]) * (H - 1)) ** 2
    )

    # Separate visible and occluded
    vis_errors = errors[gt_vis > 0.5]
    occ_errors = errors[gt_vis <= 0.5]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Error histogram
    axes[0].hist(errors, bins=50, alpha=0.7, label="All")
    axes[0].hist(vis_errors, bins=50, alpha=0.7, label="Visible")
    axes[0].hist(occ_errors, bins=50, alpha=0.7, label="Occluded")
    axes[0].set_xlabel("Error (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Error Distribution\nMean: {np.mean(errors):.2f} px")
    axes[0].legend()

    # Error vs confidence
    axes[1].scatter(pred_conf, errors, alpha=0.5, s=10)
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("Error (pixels)")
    axes[1].set_title("Error vs Confidence")

    # Confidence histogram
    axes[2].hist(pred_conf, bins=50, alpha=0.7)
    axes[2].set_xlabel("Confidence")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Confidence Distribution\nMean: {np.mean(pred_conf):.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved error analysis to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize D4RT tracking results")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_step_0050000.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/kubric",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print(f"Loading dataset from {args.data_dir}")
    dataset = KubricDataset(
        data_dir=args.data_dir,
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=64,
    )

    print(f"Visualizing {args.num_samples} samples...")

    for i in range(args.num_samples):
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]

        scene_name = sample["metadata"].get("scene", sample["metadata"].get("scene_id", "unknown"))
        print(f"\nSample {i+1}/{args.num_samples} (scene: {scene_name})")

        # Run inference
        output = run_inference(model, sample, device)

        # Visualize tracks
        track_path = output_dir / f"tracks_{i:03d}.png"
        visualize_tracks(sample, output, str(track_path))

        # Visualize errors
        error_path = output_dir / f"errors_{i:03d}.png"
        visualize_error_distribution(sample, output, str(error_path))

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
