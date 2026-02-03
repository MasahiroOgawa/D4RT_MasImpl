"""Query sampling strategies for D4RT training."""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class QuerySampler:
    """
    Strategic query sampler for D4RT training.

    Implements the sampling strategy from the paper:
    - 50% visible points (has GT 3D position)
    - 25% occluded points (tests visibility prediction)
    - 25% random background points (ensures coverage)
    """

    def __init__(
        self,
        num_queries: int = 2048,
        visible_ratio: float = 0.5,
        occluded_ratio: float = 0.25,
        random_ratio: float = 0.25,
        temporal_sampling: str = 'uniform',
    ):
        """
        Initialize query sampler.

        Args:
            num_queries: Total number of queries to sample
            visible_ratio: Ratio of visible points
            occluded_ratio: Ratio of occluded points
            random_ratio: Ratio of random points
            temporal_sampling: Temporal sampling strategy ('uniform')
        """
        assert abs(visible_ratio + occluded_ratio + random_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.num_queries = num_queries
        self.visible_ratio = visible_ratio
        self.occluded_ratio = occluded_ratio
        self.random_ratio = random_ratio
        self.temporal_sampling = temporal_sampling

        # Calculate number of queries for each category
        self.num_visible = int(num_queries * visible_ratio)
        self.num_occluded = int(num_queries * occluded_ratio)
        self.num_random = num_queries - self.num_visible - self.num_occluded

    def sample_queries(
        self,
        video_shape: Tuple[int, int, int, int],  # (T, C, H, W)
        visibility_mask: Optional[torch.Tensor] = None,  # (T, H, W)
        depth_map: Optional[torch.Tensor] = None,  # (T, H, W)
        points_3d: Optional[torch.Tensor] = None,  # (T, H, W, 3)
    ) -> Dict[str, torch.Tensor]:
        """
        Sample queries for training.

        Args:
            video_shape: Shape of video (T, C, H, W)
            visibility_mask: [T, H, W] boolean mask of visible pixels
            depth_map: [T, H, W] depth values (optional)
            points_3d: [T, H, W, 3] 3D positions (optional)

        Returns:
            queries: Dictionary with query components
        """
        T, C, H, W = video_shape

        # Initialize lists for query components
        u_list = []
        v_list = []
        t_src_list = []
        t_tgt_list = []
        t_cam_list = []

        # 1. Sample visible points
        if visibility_mask is not None and self.num_visible > 0:
            visible_queries = self._sample_visible_points(
                visibility_mask, self.num_visible, T, H, W
            )
            u_list.append(visible_queries['u'])
            v_list.append(visible_queries['v'])
            t_src_list.append(visible_queries['t_src'])
            t_tgt_list.append(visible_queries['t_tgt'])
            t_cam_list.append(visible_queries['t_cam'])

        # 2. Sample occluded points
        if visibility_mask is not None and self.num_occluded > 0:
            occluded_queries = self._sample_occluded_points(
                visibility_mask, depth_map, self.num_occluded, T, H, W
            )
            u_list.append(occluded_queries['u'])
            v_list.append(occluded_queries['v'])
            t_src_list.append(occluded_queries['t_src'])
            t_tgt_list.append(occluded_queries['t_tgt'])
            t_cam_list.append(occluded_queries['t_cam'])

        # 3. Sample random points
        if self.num_random > 0:
            random_queries = self._sample_random_points(
                self.num_random, T, H, W
            )
            u_list.append(random_queries['u'])
            v_list.append(random_queries['v'])
            t_src_list.append(random_queries['t_src'])
            t_tgt_list.append(random_queries['t_tgt'])
            t_cam_list.append(random_queries['t_cam'])

        # Concatenate all queries
        queries = {
            'u': torch.cat(u_list, dim=0),
            'v': torch.cat(v_list, dim=0),
            't_src': torch.cat(t_src_list, dim=0),
            't_tgt': torch.cat(t_tgt_list, dim=0),
            't_cam': torch.cat(t_cam_list, dim=0),
        }

        # Shuffle queries
        perm = torch.randperm(self.num_queries)
        for key in queries:
            queries[key] = queries[key][perm]

        return queries

    def _sample_visible_points(
        self,
        visibility_mask: torch.Tensor,
        num_samples: int,
        T: int,
        H: int,
        W: int,
    ) -> Dict[str, torch.Tensor]:
        """Sample visible points."""
        # Find visible pixels
        visible_coords = torch.nonzero(visibility_mask, as_tuple=False)  # [N, 3] (t, h, w)

        if len(visible_coords) == 0:
            # Fall back to random sampling if no visible points
            return self._sample_random_points(num_samples, T, H, W)

        # Sample from visible coordinates
        num_available = len(visible_coords)
        if num_available >= num_samples:
            indices = torch.randperm(num_available)[:num_samples]
            sampled_coords = visible_coords[indices]
        else:
            # Sample with replacement if not enough visible points
            indices = torch.randint(0, num_available, (num_samples,))
            sampled_coords = visible_coords[indices]

        # Extract coordinates
        t_src = sampled_coords[:, 0].long()
        v = sampled_coords[:, 1].float() / (H - 1)  # Normalize to [0, 1]
        u = sampled_coords[:, 2].float() / (W - 1)

        # Sample temporal coordinates
        t_tgt, t_cam = self._sample_temporal_coords(num_samples, T)

        return {
            'u': u,
            'v': v,
            't_src': t_src,
            't_tgt': t_tgt,
            't_cam': t_cam,
        }

    def _sample_occluded_points(
        self,
        visibility_mask: torch.Tensor,
        depth_map: Optional[torch.Tensor],
        num_samples: int,
        T: int,
        H: int,
        W: int,
    ) -> Dict[str, torch.Tensor]:
        """Sample occluded points."""
        # Find occluded pixels (not visible but has depth)
        if depth_map is not None:
            has_depth = depth_map > 0
            occluded_mask = (~visibility_mask) & has_depth
        else:
            occluded_mask = ~visibility_mask

        occluded_coords = torch.nonzero(occluded_mask, as_tuple=False)

        if len(occluded_coords) == 0:
            # Fall back to random sampling
            return self._sample_random_points(num_samples, T, H, W)

        # Sample from occluded coordinates
        num_available = len(occluded_coords)
        if num_available >= num_samples:
            indices = torch.randperm(num_available)[:num_samples]
            sampled_coords = occluded_coords[indices]
        else:
            indices = torch.randint(0, num_available, (num_samples,))
            sampled_coords = occluded_coords[indices]

        # Extract coordinates
        t_src = sampled_coords[:, 0].long()
        v = sampled_coords[:, 1].float() / (H - 1)
        u = sampled_coords[:, 2].float() / (W - 1)

        # Sample temporal coordinates
        t_tgt, t_cam = self._sample_temporal_coords(num_samples, T)

        return {
            'u': u,
            'v': v,
            't_src': t_src,
            't_tgt': t_tgt,
            't_cam': t_cam,
        }

    def _sample_random_points(
        self,
        num_samples: int,
        T: int,
        H: int,
        W: int,
    ) -> Dict[str, torch.Tensor]:
        """Sample random points uniformly."""
        # Uniform spatial sampling
        u = torch.rand(num_samples)  # [0, 1]
        v = torch.rand(num_samples)  # [0, 1]

        # Uniform temporal sampling
        t_src = torch.randint(0, T, (num_samples,))
        t_tgt, t_cam = self._sample_temporal_coords(num_samples, T)

        return {
            'u': u,
            'v': v,
            't_src': t_src,
            't_tgt': t_tgt,
            't_cam': t_cam,
        }

    def _sample_temporal_coords(
        self,
        num_samples: int,
        T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample temporal coordinates."""
        if self.temporal_sampling == 'uniform':
            t_tgt = torch.randint(0, T, (num_samples,))
            t_cam = torch.randint(0, T, (num_samples,))
        else:
            raise ValueError(f"Unknown temporal sampling: {self.temporal_sampling}")

        return t_tgt, t_cam


def extract_ground_truth_at_queries(
    queries: Dict[str, torch.Tensor],
    points_3d: torch.Tensor,  # [T, H, W, 3]
    visibility: torch.Tensor,  # [T, H, W]
    cameras_intrinsics: torch.Tensor,  # [T, 3, 3]
    cameras_extrinsics: torch.Tensor,  # [T, 4, 4]
    normals: Optional[torch.Tensor] = None,  # [T, H, W, 3]
    tracked_positions: Optional[torch.Tensor] = None,  # [N, T, 2] tracked (u, v) coords
) -> Dict[str, torch.Tensor]:
    """
    Extract ground truth values at query locations.

    CRITICAL: This function now supports tracked point positions. If tracked_positions
    is provided, ground truth is extracted at TRACKED pixel locations (following moving
    objects), not at FIXED pixel locations. This is essential for proper point tracking
    as described in the D4RT paper.

    Args:
        queries: Dictionary with query coordinates
        points_3d: [T, H, W, 3] 3D positions
        visibility: [T, H, W] visibility mask
        cameras_intrinsics: [T, 3, 3] camera intrinsics
        cameras_extrinsics: [T, 4, 4] camera extrinsics
        normals: [T, H, W, 3] surface normals (optional)
        tracked_positions: [N, T, 2] tracked (u, v) pixel coordinates (CRITICAL FIX)

    Returns:
        targets: Dictionary with ground truth values
    """
    num_queries = len(queries['u'])
    T, H, W, _ = points_3d.shape

    # Convert normalized coordinates to pixel indices (source frame)
    u_px_src = (queries['u'] * (W - 1)).long()
    v_px_src = (queries['v'] * (H - 1)).long()
    t_src = queries['t_src'].long()
    t_tgt = queries['t_tgt'].long()
    t_cam = queries['t_cam'].long()

    # Clamp to valid range
    u_px_src = torch.clamp(u_px_src, 0, W - 1)
    v_px_src = torch.clamp(v_px_src, 0, H - 1)
    t_src = torch.clamp(t_src, 0, T - 1)
    t_tgt = torch.clamp(t_tgt, 0, T - 1)
    t_cam = torch.clamp(t_cam, 0, T - 1)

    # Extract 3D positions
    # CRITICAL FIX: Use tracked positions if available, otherwise fall back to fixed pixels
    if tracked_positions is not None:
        # Use tracked positions: extract GT at TRACKED pixel locations
        xyz = []
        for i in range(num_queries):
            t_i = t_tgt[i]
            # Get tracked pixel coordinates at target time [u, v] in pixels
            u_track = torch.clamp(tracked_positions[i, t_i, 0].long(), 0, W - 1)
            v_track = torch.clamp(tracked_positions[i, t_i, 1].long(), 0, H - 1)
            xyz.append(points_3d[t_i, v_track, u_track])
        xyz = torch.stack(xyz, dim=0)  # [N, 3]
    else:
        # OLD BEHAVIOR: Fixed pixels (BUG - doesn't track moving objects!)
        xyz = points_3d[t_tgt, v_px_src, u_px_src]  # [N, 3]

    # Extract visibility (always from source frame)
    vis = visibility[t_src, v_px_src, u_px_src].float()  # [N]

    # Project to 2D for 2D loss
    uv_2d = []
    for i in range(num_queries):
        K = cameras_intrinsics[t_cam[i]]
        T_mat = cameras_extrinsics[t_cam[i]]

        # Project 3D point to 2D
        xyz_i = xyz[i].unsqueeze(0)  # [1, 3]
        xyz_hom = torch.cat([xyz_i, torch.ones(1, 1)], dim=1)  # [1, 4]
        xyz_cam = (T_mat @ xyz_hom.T).T[:, :3]  # [1, 3]

        if xyz_cam[0, 2] > 0:  # Valid depth
            uv_proj = (K @ xyz_cam.T).T  # [1, 3]
            uv_proj = uv_proj[:, :2] / uv_proj[:, 2:3]  # [1, 2]
            uv_2d.append(uv_proj[0])
        else:
            uv_2d.append(torch.tensor([u_px_src[i].float(), v_px_src[i].float()]))

    uv_2d = torch.stack(uv_2d, dim=0)  # [N, 2]

    # CRITICAL: Normalize UV to [0, 1] range to match model's sigmoid output
    # Model outputs UV with sigmoid in [0, 1], so GT must be normalized too
    uv_2d_normalized = uv_2d.clone()
    uv_2d_normalized[:, 0] = uv_2d[:, 0] / (W - 1)  # u: pixel -> [0, 1]
    uv_2d_normalized[:, 1] = uv_2d[:, 1] / (H - 1)  # v: pixel -> [0, 1]
    uv_2d_normalized = torch.clamp(uv_2d_normalized, 0.0, 1.0)

    targets = {
        'xyz': xyz,
        'uv': uv_2d_normalized,  # Now in [0, 1] range
        'visibility': vis,
    }

    # Extract normals if available
    if normals is not None:
        if tracked_positions is not None:
            # Use tracked positions for normals too
            normals_gt = []
            for i in range(num_queries):
                t_i = t_tgt[i]
                u_track = torch.clamp(tracked_positions[i, t_i, 0].long(), 0, W - 1)
                v_track = torch.clamp(tracked_positions[i, t_i, 1].long(), 0, H - 1)
                normals_gt.append(normals[t_i, v_track, u_track])
            normals_gt = torch.stack(normals_gt, dim=0)  # [N, 3]
        else:
            # OLD BEHAVIOR: Fixed pixels
            normals_gt = normals[t_tgt, v_px_src, u_px_src]  # [N, 3]
        targets['normals'] = normals_gt

    return targets
