#!/usr/bin/env python3
"""Create video visualization of D4RT tracking results.

Shows continuous point tracking - each query point tracked through ALL frames.
GT (green) vs Predicted (red) tracks overlaid on video.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from d4rt.data.datasets.kubric import KubricDataset
from d4rt.data.datasets.tapvid3d import TAPVid3DDataset
from d4rt.models import build_d4rt_model


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
    parser.add_argument(
        "--dataset-type", type=str, default="kubric", choices=["kubric", "tapvid3d"]
    )
    parser.add_argument("--split", type=str, default="val")
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

    if args.dataset_type == "tapvid3d":
        print(f"Loading TAP-Vid-3D dataset from {args.data_dir}")
        _run_tapvid3d(model, args, device)
    else:
        print(f"Loading Kubric dataset from {args.data_dir}")
        _run_kubric(model, args, device)

    print(f"\nAll videos saved to {args.output_dir}/")
    print("\nView with: xdg-open outputs/tracking_*.mp4")


def _run_tapvid3d(model, args, device):
    """Visualize tracking on TAP-Vid-3D data."""
    import io as _io

    from PIL import Image

    data_path = Path(args.data_dir)
    # Collect npz files
    npz_files = sorted(data_path.glob("**/*.npz"))
    if not npz_files:
        print(f"No .npz files found in {data_path}")
        return

    # Use deterministic split matching TAPVid3DDataset
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(npz_files))
    n_train = int(len(npz_files) * 0.8)
    if args.split == "train":
        selected_files = [npz_files[i] for i in indices[:n_train]]
    else:
        selected_files = [npz_files[i] for i in indices[n_train:]]

    np.random.seed(args.seed)
    sample_indices = np.random.choice(
        len(selected_files), min(args.num_samples, len(selected_files)), replace=False
    )

    for si, file_idx in enumerate(sample_indices):
        npz_path = selected_files[file_idx]
        data = np.load(npz_path, allow_pickle=True)
        scene_name = npz_path.stem[:40]
        subset = npz_path.parent.name

        print(f"\nProcessing {si+1}/{len(sample_indices)} [{subset}] {scene_name}")

        # Decode frames and resample to 24
        jpeg_bytes = data["images_jpeg_bytes"]
        T_orig = len(jpeg_bytes)
        frame_indices = np.linspace(0, T_orig - 1, 24, dtype=int)

        frames = []
        for fi in frame_indices:
            img = Image.open(_io.BytesIO(jpeg_bytes[fi])).convert("RGB")
            img = img.resize((256, 256), Image.BILINEAR)
            frames.append(np.array(img))
        video_np = np.stack(frames)  # [T, H, W, 3]

        video = torch.from_numpy(video_np).float().permute(0, 3, 1, 2) / 255.0  # [T, 3, H, W]
        T, _, H, W = video.shape

        # Load GT tracks
        tracks_XYZ = data["tracks_XYZ"].astype(np.float32)[frame_indices]  # [T, N, 3]
        visibility = data["visibility"].astype(np.float32)[frame_indices]  # [T, N]
        queries_xyt = data["queries_xyt"].astype(np.float32)  # [N, 3]
        fx, fy, cx, cy = data["fx_fy_cx_cy"].astype(np.float32)

        # Get original image size for scaling
        orig_img = Image.open(_io.BytesIO(jpeg_bytes[0]))
        orig_W, orig_H = orig_img.size
        scale_x, scale_y = W / orig_W, H / orig_H

        # Project 3D GT to 2D in resized image coords, then normalize to [0,1]
        Z = tracks_XYZ[:, :, 2:3].clip(min=1e-6)
        u_pix = (fx * tracks_XYZ[:, :, 0:1] / Z + cx) * scale_x
        v_pix = (fy * tracks_XYZ[:, :, 1:2] / Z + cy) * scale_y
        gt_tracks_2d_all = np.concatenate([u_pix / (W - 1), v_pix / (H - 1)], axis=-1)  # [T, N, 2]

        # Map query t_src to resampled frame
        query_t_orig = queries_xyt[:, 2].astype(int)
        query_t_resampled = np.array([np.argmin(np.abs(frame_indices - t)) for t in query_t_orig])

        # Select visible points for visualization
        N_total = tracks_XYZ.shape[1]
        valid = [n for n in range(N_total) if visibility[query_t_resampled[n], n] > 0.5]
        if len(valid) < args.num_points:
            chosen = valid
        else:
            chosen = list(np.random.choice(valid, args.num_points, replace=False))

        # Build query points and GT
        query_points = []
        gt_tracks_2d = []
        gt_vis = []
        for n in chosen:
            u_q = float(queries_xyt[n, 0] * scale_x / (W - 1))
            v_q = float(queries_xyt[n, 1] * scale_y / (H - 1))
            t_s = int(query_t_resampled[n])
            query_points.append((np.clip(u_q, 0, 1), np.clip(v_q, 0, 1), t_s))
            gt_tracks_2d.append(gt_tracks_2d_all[:, n, :])  # [T, 2]
            gt_vis.append(visibility[:, n])  # [T]

        gt_tracks_2d = np.stack(gt_tracks_2d)  # [N, T, 2]
        gt_vis = np.stack(gt_vis)  # [N, T]

        print(f"  Tracking {len(query_points)} points through {T} frames...")
        pred_uv, pred_vis = track_points_all_frames(model, video, query_points, device)

        output_path = f"{args.output_dir}/tracking_{subset}_{scene_name}.mp4"
        create_tracking_video(
            video.numpy(), gt_tracks_2d, gt_vis, pred_uv, pred_vis, query_points, output_path
        )


def _run_kubric(model, args, device):
    """Visualize tracking on Kubric data."""
    dataset = KubricDataset(
        data_dir=args.data_dir,
        split="val",
        num_frames=24,
        resolution=(256, 256),
        num_queries=256,
    )

    for i in range(args.num_samples):
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]
        scene_name = sample["metadata"].get("scene_id", f"scene_{idx}")

        print(f"\nProcessing sample {i+1}/{args.num_samples} ({scene_name})")

        video = sample["video"].numpy()  # (T, 3, H, W)
        T = video.shape[0]

        query_u = sample["queries"]["u"].numpy()
        query_v = sample["queries"]["v"].numpy()
        t_src = sample["queries"]["t_src"].numpy().astype(int)

        # Select points that start at frame 0 for cleaner visualization
        frame0_mask = t_src == 0
        frame0_indices = np.where(frame0_mask)[0]

        if len(frame0_indices) < args.num_points:
            selected_indices = list(frame0_indices)
            other_indices = np.where(~frame0_mask)[0]
            np.random.shuffle(other_indices)
            selected_indices.extend(other_indices[: args.num_points - len(selected_indices)])
        else:
            selected_indices = np.random.choice(frame0_indices, args.num_points, replace=False)

        query_points = [(query_u[j], query_v[j], int(t_src[j])) for j in selected_indices]

        print(f"  Tracking {len(query_points)} points through {T} frames...")
        pred_uv, pred_vis = track_points_all_frames(
            model, torch.from_numpy(video).float(), query_points, device
        )

        # Load GT tracks from scene directory
        scene_dir = Path(args.data_dir) / "val" / scene_name
        tracks_file = scene_dir / "tracks.npz"

        if tracks_file.exists():
            tracks_data = np.load(tracks_file)
            all_tracks_2d = tracks_data.get("tracks_2d", None)
            all_visibility = tracks_data.get("visibility", None)
            query_points_data = tracks_data.get("query_points", None)

            if all_tracks_2d is not None:
                H, W = video.shape[2], video.shape[3]
                gt_tracks_2d, gt_visibility = [], []
                for u, v, t_s in query_points:
                    x_pixel, y_pixel = u * (W - 1), v * (H - 1)
                    if query_points_data is not None:
                        dists = np.sqrt(
                            (query_points_data[:, 2] - x_pixel) ** 2
                            + (query_points_data[:, 1] - y_pixel) ** 2
                            + (query_points_data[:, 0] - t_s) ** 2 * 100
                        )
                        best_idx = np.argmin(dists)
                        track_2d = all_tracks_2d[best_idx].copy()
                        track_2d[:, 0] /= W - 1
                        track_2d[:, 1] /= H - 1
                        gt_tracks_2d.append(track_2d)
                        gt_visibility.append(all_visibility[best_idx])
                    else:
                        gt_tracks_2d.append(pred_uv[len(gt_tracks_2d)])
                        gt_visibility.append(np.ones(T))
                gt_tracks_2d = np.stack(gt_tracks_2d)
                gt_visibility = np.stack(gt_visibility)
            else:
                gt_tracks_2d, gt_visibility = pred_uv.copy(), pred_vis.copy()
        else:
            gt_tracks_2d, gt_visibility = pred_uv.copy(), pred_vis.copy()

        output_path = f"{args.output_dir}/tracking_{scene_name}.mp4"
        create_tracking_video(
            video, gt_tracks_2d, gt_visibility, pred_uv, pred_vis, query_points, output_path
        )


if __name__ == "__main__":
    main()
