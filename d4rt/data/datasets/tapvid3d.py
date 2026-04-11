"""TAP-Vid-3D dataset loader for D4RT.

Loads .npz files from TAP-Vid-3D benchmark (drivetrack, pstudio, etc.).
Handles JPEG-encoded images, frame resampling, coordinate transforms,
and train/test splitting.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..base_dataset import BaseVideoDataset


class TAPVid3DDataset(BaseVideoDataset):
    """
    TAP-Vid-3D benchmark dataset.

    Supports drivetrack (with extrinsics) and pstudio (camera-space).

    NPZ format:
        images_jpeg_bytes: [T] JPEG-encoded frames
        queries_xyt: [N, 3] query points (x_pixel, y_pixel, t_frame)
        tracks_XYZ: [T, N, 3] 3D trajectories
        visibility: [T, N] binary visibility
        fx_fy_cx_cy: [4] camera intrinsics
        extrinsics_w2c: [T, 4, 4] world-to-camera (optional, drivetrack only)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_frames: int = 24,
        resolution: Tuple[int, int] = (256, 256),
        num_queries: int = 128,
        transform: Optional[object] = None,
        subsets: Optional[List[str]] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        super().__init__(data_dir, split, num_frames, resolution, num_queries, transform)

        self.subsets = subsets or ["drivetrack", "pstudio"]
        data_path = Path(data_dir)

        # Collect all npz files from specified subsets
        all_files = []
        for subset in self.subsets:
            subset_dir = data_path / subset
            if subset_dir.exists():
                files = sorted(subset_dir.glob("*.npz"))
                all_files.extend(files)

        if not all_files:
            raise ValueError(f"No .npz files found in {data_path} for subsets {self.subsets}")

        # Deterministic train/test split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_files))
        n_train = int(len(all_files) * train_ratio)

        if split == "train":
            self.samples = [all_files[i] for i in indices[:n_train]]
        else:  # val or test
            self.samples = [all_files[i] for i in indices[n_train:]]

        print(f"TAPVid3D {split}: {len(self.samples)} samples from {self.subsets}")

    def _load_video_data(self, idx: int) -> Dict:
        """Load video frames and camera parameters from npz."""
        npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)

        # Decode JPEG frames
        jpeg_bytes = data["images_jpeg_bytes"]
        T_orig = len(jpeg_bytes)

        # Uniformly sample num_frames
        frame_indices = np.linspace(0, T_orig - 1, self.num_frames, dtype=int)

        H_tgt, W_tgt = self.resolution

        # Get original image size from first frame for intrinsics scaling
        first_img = Image.open(io.BytesIO(jpeg_bytes[0]))
        orig_W, orig_H = first_img.size

        frames = []
        for fi in frame_indices:
            img = Image.open(io.BytesIO(jpeg_bytes[fi])).convert("RGB")
            img = img.resize((W_tgt, H_tgt), Image.BILINEAR)
            frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)

        frames = torch.stack(frames)  # [T, 3, H, W]

        # Scale intrinsics for resize
        fx, fy, cx, cy = data["fx_fy_cx_cy"].astype(np.float32)
        scale_x = W_tgt / orig_W
        scale_y = H_tgt / orig_H
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y

        K = np.array(
            [
                [fx_scaled, 0, cx_scaled],
                [0, fy_scaled, cy_scaled],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        intrinsics = torch.from_numpy(K).unsqueeze(0).expand(self.num_frames, -1, -1).clone()

        # Extrinsics (provided for reference; tracks_XYZ is already in camera space)
        if "extrinsics_w2c" in data:
            ext_all = data["extrinsics_w2c"].astype(np.float32)  # [T_orig, 4, 4]
            ext_sampled = ext_all[frame_indices]
            extrinsics = torch.from_numpy(ext_sampled)
        else:
            extrinsics = (
                torch.eye(4, dtype=torch.float32)
                .unsqueeze(0)
                .expand(self.num_frames, -1, -1)
                .clone()
            )

        # Store metadata needed by _load_ground_truth
        self._current_data = data
        self._current_frame_indices = frame_indices
        self._current_orig_size = (orig_H, orig_W)

        return {
            "frames": frames,
            "cameras": {"intrinsics": intrinsics, "extrinsics": extrinsics},
            "metadata": {
                "scene_id": npz_path.stem,
                "split": self.split,
                "subset": npz_path.parent.name,
            },
        }

    def _load_ground_truth(self, idx: int, video_data: Dict) -> Dict:
        """Load ground truth queries and targets."""
        data = self._current_data
        frame_indices = self._current_frame_indices
        orig_H, orig_W = self._current_orig_size

        T = self.num_frames
        H_tgt, W_tgt = self.resolution

        # Load tracks data
        tracks_XYZ = data["tracks_XYZ"].astype(np.float32)  # [T_orig, N_total, 3]
        visibility = data["visibility"].astype(np.float32)  # [T_orig, N_total]
        queries_xyt = data["queries_xyt"].astype(np.float32)  # [N_total, 3]

        # Resample to our frame indices
        tracks_XYZ = tracks_XYZ[frame_indices]  # [T, N_total, 3]
        visibility = visibility[frame_indices]  # [T, N_total]

        N_total = tracks_XYZ.shape[1]

        # tracks_XYZ is already in per-frame camera coordinates for both subsets
        tracks_3d = tracks_XYZ

        # Project 3D→2D using scaled intrinsics
        fx, fy, cx, cy = data["fx_fy_cx_cy"].astype(np.float32)
        scale_x = W_tgt / orig_W
        scale_y = H_tgt / orig_H

        # Compute 2D projections in resized image space
        # tracks_3d: [T, N, 3] in camera coordinates
        Z = tracks_3d[:, :, 2:3].clip(min=1e-6)  # [T, N, 1]
        u_pixel = fx * tracks_3d[:, :, 0:1] / Z + cx  # original pixel
        v_pixel = fy * tracks_3d[:, :, 1:2] / Z + cy
        u_pixel = u_pixel * scale_x  # scaled pixel
        v_pixel = v_pixel * scale_y
        tracks_2d = np.concatenate([u_pixel, v_pixel], axis=-1)  # [T, N, 2]

        # Map query source frame to resampled index
        query_t_orig = queries_xyt[:, 2].astype(int)  # original frame indices
        # Find closest resampled frame for each query
        query_t_resampled = np.array([np.argmin(np.abs(frame_indices - t)) for t in query_t_orig])

        # Filter out queries with invalid data
        valid_mask = np.ones(N_total, dtype=bool)
        for n in range(N_total):
            t_src = query_t_resampled[n]
            if not visibility[t_src, n]:
                valid_mask[n] = False
                continue
            xyz = tracks_3d[t_src, n]
            if not np.all(np.isfinite(xyz)) or xyz[2] <= 0:
                valid_mask[n] = False

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            # Fallback: use all points
            valid_indices = np.arange(N_total)

        # Sample queries
        num_queries = min(self.num_queries, len(valid_indices))
        chosen = np.random.choice(
            valid_indices, size=num_queries, replace=len(valid_indices) < num_queries
        )

        # Build query data
        t_src = torch.from_numpy(query_t_resampled[chosen]).float()
        query_x = queries_xyt[chosen, 0]  # original pixel x
        query_y = queries_xyt[chosen, 1]  # original pixel y
        u_norm = torch.from_numpy(query_x * scale_x / (W_tgt - 1)).float().clamp(0, 1)
        v_norm = torch.from_numpy(query_y * scale_y / (H_tgt - 1)).float().clamp(0, 1)

        # Random target frames
        t_tgt = torch.randint(0, T, (num_queries,)).float()

        queries = {
            "u": u_norm,
            "v": v_norm,
            "t_src": t_src,
            "t_tgt": t_tgt,
            "t_cam": t_tgt,
        }

        # Extract targets at t_tgt
        t_tgt_int = t_tgt.long().numpy()
        xyz = np.array([tracks_3d[t_tgt_int[i], chosen[i]] for i in range(num_queries)])
        uv = np.array([tracks_2d[t_tgt_int[i], chosen[i]] for i in range(num_queries)])
        vis = np.array([visibility[t_tgt_int[i], chosen[i]] for i in range(num_queries)])

        # Normalize uv to [0, 1]
        uv_norm = np.zeros_like(uv)
        uv_norm[:, 0] = uv[:, 0] / (W_tgt - 1)
        uv_norm[:, 1] = uv[:, 1] / (H_tgt - 1)

        targets = {
            "xyz": torch.from_numpy(xyz).float(),
            "uv": torch.from_numpy(uv_norm).float(),
            "visibility": torch.from_numpy(vis).float().unsqueeze(-1),
        }

        # Cleanup temp state
        del self._current_data, self._current_frame_indices, self._current_orig_size

        return {"queries": queries, "targets": targets}


def create_tapvid3d_dataloaders(
    config: Dict,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create TAP-Vid-3D train and validation dataloaders."""
    from ..transforms import get_train_transforms, get_val_transforms

    subsets = config.get("subsets", ["drivetrack", "pstudio"])
    train_ratio = config.get("train_ratio", 0.8)

    train_dataset = TAPVid3DDataset(
        data_dir=config["data_dir"],
        split="train",
        num_frames=config.get("num_frames", 24),
        resolution=tuple(config.get("resolution", [256, 256])),
        num_queries=config.get("num_queries", 128),
        transform=get_train_transforms(config),
        subsets=subsets,
        train_ratio=train_ratio,
    )

    val_dataset = TAPVid3DDataset(
        data_dir=config["data_dir"],
        split="val",
        num_frames=config.get("num_frames", 24),
        resolution=tuple(config.get("resolution", [256, 256])),
        num_queries=config.get("num_queries", 128),
        transform=get_val_transforms(),
        subsets=subsets,
        train_ratio=train_ratio,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=BaseVideoDataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=BaseVideoDataset.collate_fn,
    )

    return train_loader, val_loader
