#!/usr/bin/env python3
"""Create video visualization of D4RT tracking results.

Shows continuous point tracking - each query point tracked through ALL frames.
GT (green) vs Predicted (red) tracks overlaid on video.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import cv2

from d4rt.models import build_d4rt_model
from d4rt.data.datasets.kubric import KubricDataset
from omegaconf import OmegaConf


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_path = Path(__file__).parent.parent / "configs" / "model" / "vit_b_movi.yaml"
    model_config = OmegaConf.load(config_path)
    model = build_d4rt_model(model_config)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def track_points_all_frames(model, video, query_points, device):
    """
    Track query points through ALL frames.

    Args:
        model: D4RT model
        video: (T, 3, H, W) tensor
        query_points: list of (u, v, t_src) tuples - normalized coords and source frame
        device: torch device

    Returns:
        pred_uv: (N, T, 2) predicted UV for each point at each frame
        pred_vis: (N, T) predicted visibility
    """
    T = video.shape[0]
    N = len(query_points)

    video_batch = video.unsqueeze(0).to(device)  # (1, T, 3, H, W)

    all_pred_uv = []
    all_pred_vis = []

    with torch.no_grad():
        for u, v, t_src in query_points:
            # Track this point to ALL frames
            queries = {
                "u": torch.full((1, T), u, device=device, dtype=torch.float32),
                "v": torch.full((1, T), v, device=device, dtype=torch.float32),
                "t_src": torch.full((1, T), t_src, device=device, dtype=torch.float32),
                "t_tgt": torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0),
                "t_cam": torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0),
            }

            output = model(video_batch, queries)

            pred_uv = output["uv"].squeeze(0).cpu().numpy()  # (T, 2)
            pred_vis = (
                torch.sigmoid(output["visibility"]).squeeze(0).squeeze(-1).cpu().numpy()
            )  # (T,)

            all_pred_uv.append(pred_uv)
            all_pred_vis.append(pred_vis)

    return np.stack(all_pred_uv), np.stack(all_pred_vis)  # (N, T, 2), (N, T)


def create_tracking_video(
    video, gt_tracks_2d, gt_visibility, pred_uv, pred_vis, query_points, save_path: str
):
    """
    Create video with continuous GT and predicted tracks.

    Args:
        video: (T, 3, H, W) numpy array
        gt_tracks_2d: (N, T, 2) ground truth UV coordinates
        gt_visibility: (N, T) ground truth visibility
        pred_uv: (N, T, 2) predicted UV coordinates
        pred_vis: (N, T) predicted visibility
        query_points: list of (u, v, t_src) tuples
        save_path: output video path
    """
    T, _, H, W = video.shape
    N = len(query_points)

    # Colors for each point (cycle through distinct colors)
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue (BGR)
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Orange
        (255, 128, 0),  # Light blue
        (0, 128, 255),  # Orange-red
    ]

    # Create video writer - side by side (GT | Pred)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, 8.0, (W * 2, H))

    for t in range(T):
        # Get frame
        frame = video[t].transpose(1, 2, 0)  # (H, W, 3)
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_gt = frame_bgr.copy()
        frame_pred = frame_bgr.copy()

        for n in range(N):
            u_src, v_src, t_src = query_points[n]
            color = colors[n % len(colors)]

            # Only show tracking from source frame onwards
            if t < t_src:
                continue

            # Source point marker (square at t_src)
            if t == t_src:
                src_x = int(u_src * (W - 1))
                src_y = int(v_src * (H - 1))
                # Draw on both views
                cv2.rectangle(frame_gt, (src_x - 5, src_y - 5), (src_x + 5, src_y + 5), color, 2)
                cv2.rectangle(frame_pred, (src_x - 5, src_y - 5), (src_x + 5, src_y + 5), color, 2)
                cv2.putText(
                    frame_gt,
                    str(n),
                    (src_x + 7, src_y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
                cv2.putText(
                    frame_pred,
                    str(n),
                    (src_x + 7, src_y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

            # GT track (circle)
            if gt_visibility[n, t] > 0.5:
                gt_x = int(np.clip(gt_tracks_2d[n, t, 0], 0, 1) * (W - 1))
                gt_y = int(np.clip(gt_tracks_2d[n, t, 1], 0, 1) * (H - 1))
                cv2.circle(frame_gt, (gt_x, gt_y), 5, color, -1)
                cv2.circle(frame_gt, (gt_x, gt_y), 5, (255, 255, 255), 1)

                # Draw trail (connect to previous frame)
                if t > t_src and gt_visibility[n, t - 1] > 0.5:
                    prev_x = int(np.clip(gt_tracks_2d[n, t - 1, 0], 0, 1) * (W - 1))
                    prev_y = int(np.clip(gt_tracks_2d[n, t - 1, 1], 0, 1) * (H - 1))
                    cv2.line(frame_gt, (prev_x, prev_y), (gt_x, gt_y), color, 2)
            else:
                # Occluded - draw X
                gt_x = int(np.clip(gt_tracks_2d[n, t, 0], 0, 1) * (W - 1))
                gt_y = int(np.clip(gt_tracks_2d[n, t, 1], 0, 1) * (H - 1))
                cv2.drawMarker(frame_gt, (gt_x, gt_y), color, cv2.MARKER_CROSS, 8, 1)

            # Predicted track (circle)
            if pred_vis[n, t] > 0.5:
                pred_x = int(np.clip(pred_uv[n, t, 0], 0, 1) * (W - 1))
                pred_y = int(np.clip(pred_uv[n, t, 1], 0, 1) * (H - 1))
                cv2.circle(frame_pred, (pred_x, pred_y), 5, color, -1)
                cv2.circle(frame_pred, (pred_x, pred_y), 5, (255, 255, 255), 1)

                # Draw trail
                if t > t_src and pred_vis[n, t - 1] > 0.5:
                    prev_x = int(np.clip(pred_uv[n, t - 1, 0], 0, 1) * (W - 1))
                    prev_y = int(np.clip(pred_uv[n, t - 1, 1], 0, 1) * (H - 1))
                    cv2.line(frame_pred, (prev_x, prev_y), (pred_x, pred_y), color, 2)
            else:
                # Occluded prediction
                pred_x = int(np.clip(pred_uv[n, t, 0], 0, 1) * (W - 1))
                pred_y = int(np.clip(pred_uv[n, t, 1], 0, 1) * (H - 1))
                cv2.drawMarker(frame_pred, (pred_x, pred_y), color, cv2.MARKER_CROSS, 8, 1)

        # Add labels
        cv2.putText(
            frame_gt,
            f"Ground Truth (Frame {t})",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame_pred,
            f"Predicted (Frame {t})",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Combine side by side
        combined = np.hstack([frame_gt, frame_pred])
        out.write(combined)

    out.release()
    print(f"Saved tracking video to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Create tracking visualization video")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_step_0050000.pth")
    parser.add_argument("--data-dir", type=str, default="data/kubric")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print(f"Loading dataset from {args.data_dir}")
    dataset = KubricDataset(
        data_dir=args.data_dir,
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=256,  # Load more to select good ones
    )

    for i in range(args.num_samples):
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]
        scene_name = sample["metadata"].get("scene_id", f"scene_{idx}")

        print(f"\nProcessing sample {i+1}/{args.num_samples} ({scene_name})")

        video = sample["video"].numpy()  # (T, 3, H, W)
        T = video.shape[0]

        # Get all GT tracks from the sample
        # We need to reconstruct full tracks - the dataset gives us sampled queries
        # For proper visualization, let's select points that start at frame 0
        query_u = sample["queries"]["u"].numpy()
        query_v = sample["queries"]["v"].numpy()
        t_src = sample["queries"]["t_src"].numpy().astype(int)

        # Select points that start at frame 0 for cleaner visualization
        frame0_mask = t_src == 0
        frame0_indices = np.where(frame0_mask)[0]

        if len(frame0_indices) < args.num_points:
            # Use what we have plus some from other frames
            selected_indices = list(frame0_indices)
            other_indices = np.where(~frame0_mask)[0]
            np.random.shuffle(other_indices)
            selected_indices.extend(other_indices[: args.num_points - len(selected_indices)])
        else:
            selected_indices = np.random.choice(frame0_indices, args.num_points, replace=False)

        # Create query points list
        query_points = []
        for idx in selected_indices:
            query_points.append((query_u[idx], query_v[idx], int(t_src[idx])))

        print(f"  Tracking {len(query_points)} points through {T} frames...")

        # Run model to get predictions for all frames
        pred_uv, pred_vis = track_points_all_frames(
            model, torch.from_numpy(video).float(), query_points, device
        )

        # For GT, we need the full tracks from the dataset
        # The Kubric dataset stores tracks_2d in the scene data
        # Let's load it directly
        scene_dir = Path(args.data_dir) / "val" / scene_name
        tracks_file = scene_dir / "tracks.npz"

        if tracks_file.exists():
            tracks_data = np.load(tracks_file)
            all_tracks_2d = tracks_data.get("tracks_2d", None)  # (N, T, 2) in pixel coords
            all_visibility = tracks_data.get("visibility", None)  # (N, T)
            query_points_data = tracks_data.get("query_points", None)  # (N, 3) - t, y, x

            if all_tracks_2d is not None:
                H, W = video.shape[2], video.shape[3]

                # Find matching tracks for our query points
                gt_tracks_2d = []
                gt_visibility = []

                for u, v, t_s in query_points:
                    # Find the track that matches this query point
                    x_pixel = u * (W - 1)
                    y_pixel = v * (H - 1)

                    # Find closest query point in dataset
                    if query_points_data is not None:
                        # query_points format: (t, y, x)
                        dists = np.sqrt(
                            (query_points_data[:, 2] - x_pixel) ** 2
                            + (query_points_data[:, 1] - y_pixel) ** 2
                            + (query_points_data[:, 0] - t_s) ** 2 * 100  # Weight time more
                        )
                        best_idx = np.argmin(dists)

                        # Get this track, normalize to [0,1]
                        track_2d = all_tracks_2d[best_idx].copy()  # (T, 2) in pixels
                        track_2d[:, 0] /= W - 1  # x -> u
                        track_2d[:, 1] /= H - 1  # y -> v

                        gt_tracks_2d.append(track_2d)
                        gt_visibility.append(all_visibility[best_idx])
                    else:
                        # Fallback: use prediction as placeholder
                        gt_tracks_2d.append(pred_uv[len(gt_tracks_2d)])
                        gt_visibility.append(np.ones(T))

                gt_tracks_2d = np.stack(gt_tracks_2d)  # (N, T, 2)
                gt_visibility = np.stack(gt_visibility)  # (N, T)
            else:
                print("  Warning: No tracks_2d in dataset, using predictions for GT")
                gt_tracks_2d = pred_uv.copy()
                gt_visibility = pred_vis.copy()
        else:
            print(f"  Warning: tracks.npz not found, using predictions for GT")
            gt_tracks_2d = pred_uv.copy()
            gt_visibility = pred_vis.copy()

        # Create video
        output_path = f"{args.output_dir}/tracking_{scene_name}.mp4"
        create_tracking_video(
            video, gt_tracks_2d, gt_visibility, pred_uv, pred_vis, query_points, output_path
        )

    print(f"\nAll videos saved to {args.output_dir}/")
    print("\nView with: xdg-open outputs/tracking_*.mp4")


if __name__ == "__main__":
    main()
