"""Utility functions for extracting patches from video frames."""

import torch
import torch.nn.functional as F
from typing import Tuple


def extract_patches(
    video: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    t_src: torch.Tensor,
    patch_size: int = 9,
    padding_mode: str = 'border',
) -> torch.Tensor:
    """
    Extract patches from video frames at specified (u, v, t) locations.

    Args:
        video: [B, T, C, H, W] video tensor
        u: [B, N] normalized u coordinates in [0, 1]
        v: [B, N] normalized v coordinates in [0, 1]
        t_src: [B, N] source frame indices (integers)
        patch_size: Size of patch to extract (default: 9)
        padding_mode: Padding mode for grid_sample ('zeros', 'border', 'reflection')

    Returns:
        patches: [B, N, C, patch_size, patch_size] extracted patches
    """
    B, T, C, H, W = video.shape
    N = u.shape[1]

    # Ensure coordinates are in valid range
    u = torch.clamp(u, 0.0, 1.0)
    v = torch.clamp(v, 0.0, 1.0)

    # Convert normalized coordinates to pixel coordinates
    u_px = u * (W - 1)  # [B, N]
    v_px = v * (H - 1)  # [B, N]

    # Clamp frame indices
    t_src = torch.clamp(t_src, 0, T - 1)

    patches_list = []

    for b in range(B):
        # Select frames for this batch element
        frames_b = video[b]  # [T, C, H, W]

        # Get unique frame indices for this batch element
        t_src_b = t_src[b]  # [N]
        u_px_b = u_px[b]  # [N]
        v_px_b = v_px[b]  # [N]

        patches_b = []

        for n in range(N):
            # Get frame index and coordinates for this query
            t_idx = t_src_b[n].long().item()
            u_center = u_px_b[n]
            v_center = v_px_b[n]

            # Get frame
            frame = frames_b[t_idx]  # [C, H, W]

            # Extract patch using grid_sample for sub-pixel accuracy
            patch = extract_single_patch(
                frame.unsqueeze(0),  # [1, C, H, W]
                u_center.unsqueeze(0),  # [1]
                v_center.unsqueeze(0),  # [1]
                patch_size,
                padding_mode,
            )  # [1, C, patch_size, patch_size]

            patches_b.append(patch.squeeze(0))  # [C, patch_size, patch_size]

        patches_b = torch.stack(patches_b, dim=0)  # [N, C, patch_size, patch_size]
        patches_list.append(patches_b)

    patches = torch.stack(patches_list, dim=0)  # [B, N, C, patch_size, patch_size]

    return patches


def extract_single_patch(
    frame: torch.Tensor,
    u_center: torch.Tensor,
    v_center: torch.Tensor,
    patch_size: int,
    padding_mode: str = 'border',
) -> torch.Tensor:
    """
    Extract a single patch from a frame using grid_sample.

    Args:
        frame: [1, C, H, W] single frame
        u_center: [1] u coordinate (pixel space)
        v_center: [1] v coordinate (pixel space)
        patch_size: Size of patch
        padding_mode: Padding mode

    Returns:
        patch: [1, C, patch_size, patch_size]
    """
    _, C, H, W = frame.shape

    # Create grid for patch
    half_size = patch_size // 2
    grid_u = torch.arange(-half_size, half_size + 1, device=frame.device, dtype=frame.dtype)
    grid_v = torch.arange(-half_size, half_size + 1, device=frame.device, dtype=frame.dtype)
    grid_v, grid_u = torch.meshgrid(grid_v, grid_u, indexing='ij')

    # Offset grid by center coordinates
    grid_u = grid_u + u_center.unsqueeze(-1).unsqueeze(-1)  # [1, patch_size, patch_size]
    grid_v = grid_v + v_center.unsqueeze(-1).unsqueeze(-1)  # [1, patch_size, patch_size]

    # Normalize to [-1, 1] for grid_sample
    grid_u_norm = (grid_u / (W - 1)) * 2 - 1
    grid_v_norm = (grid_v / (H - 1)) * 2 - 1

    # Stack to create grid [1, patch_size, patch_size, 2]
    grid = torch.stack([grid_u_norm, grid_v_norm], dim=-1)

    # Sample patch
    patch = F.grid_sample(
        frame,
        grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True,
    )  # [1, C, patch_size, patch_size]

    return patch


def extract_patches_batched(
    video: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    t_src: torch.Tensor,
    patch_size: int = 9,
) -> torch.Tensor:
    """
    More efficient batched version for when all queries are from the same frame.

    Args:
        video: [B, T, C, H, W] video tensor
        u: [B, N] normalized u coordinates
        v: [B, N] normalized v coordinates
        t_src: [B, N] source frame indices (all same within batch)
        patch_size: Size of patch

    Returns:
        patches: [B, N, C, patch_size, patch_size]
    """
    B, T, C, H, W = video.shape
    N = u.shape[1]

    # For simplicity, fall back to sequential version
    # TODO: Optimize for same-frame case
    return extract_patches(video, u, v, t_src, patch_size)
