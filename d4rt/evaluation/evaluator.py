"""Comprehensive evaluation for D4RT models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

from ..models.d4rt import D4RT
from ..inference import (
    PointTracker,
    DepthReconstructor,
    CameraPoseEstimator,
    compute_tracking_metrics,
    compute_depth_metrics,
    compute_pose_metrics,
)


class D4RTEvaluator:
    """
    Comprehensive evaluator for D4RT models.

    Evaluates models on multiple tasks:
    - Point tracking
    - Depth reconstruction
    - Camera pose estimation
    """

    def __init__(
        self,
        model: D4RT,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        """
        Initialize evaluator.

        Args:
            model: D4RT model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Create inference modules
        self.tracker = PointTracker(model, device=device)
        self.depth_reconstructor = DepthReconstructor(model, device=device)
        self.pose_estimator = CameraPoseEstimator(model, device=device)

    @torch.no_grad()
    def evaluate_tracking(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate point tracking on a dataset.

        Args:
            dataloader: DataLoader providing batches with:
                - 'video': [B, T, C, H, W]
                - 'points': [B, N, 2] query points
                - 'trajectories_3d': [B, N, T, 3] ground truth 3D trajectories
                - 'visibility': [B, N, T] ground truth visibility
            max_samples: Maximum number of samples to evaluate (optional)

        Returns:
            metrics: Dictionary of tracking metrics
        """
        print("\nEvaluating point tracking...")

        all_ape = []
        all_mae_x = []
        all_mae_y = []
        all_mae_z = []
        total_visible = 0

        num_samples = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Tracking")):
            if max_samples and num_samples >= max_samples:
                break

            video = batch['video'].to(self.device)  # [B, T, C, H, W]
            points = batch['queries']['u'], batch['queries']['v']  # Initial points

            B = video.shape[0]

            for b in range(B):
                if max_samples and num_samples >= max_samples:
                    break

                # Get ground truth
                gt_trajectories = batch['targets']['xyz'][b]  # [N, 3] but need to track through time
                gt_visibility = batch['targets']['visibility'][b] if 'visibility' in batch['targets'] else None

                # For simplicity, assume queries are at t=0 and we track forward
                # In practice, this would depend on the dataset format
                query_points = torch.stack([
                    batch['queries']['u'][b],
                    batch['queries']['v'][b],
                ], dim=-1)  # [N, 2]

                # Track points
                pred_trajectories, pred_visibility = self.tracker.track_points(
                    video[b:b+1],
                    query_points,
                    start_frame=0,
                )

                # Compute metrics (this is simplified - in practice, need proper GT trajectories)
                # For now, we'll skip actual metric computation if GT format doesn't match
                # metrics = compute_tracking_metrics(pred_trajectories, gt_trajectories, gt_visibility)
                # all_ape.append(metrics['ape'])

                num_samples += 1

        # Aggregate metrics
        results = {
            'ape': np.mean(all_ape) if all_ape else float('nan'),
            'mae_x': np.mean(all_mae_x) if all_mae_x else float('nan'),
            'mae_y': np.mean(all_mae_y) if all_mae_y else float('nan'),
            'mae_z': np.mean(all_mae_z) if all_mae_z else float('nan'),
            'num_samples': num_samples,
        }

        return results

    @torch.no_grad()
    def evaluate_depth(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate depth reconstruction on a dataset.

        Args:
            dataloader: DataLoader providing batches with:
                - 'video': [B, T, C, H, W]
                - 'depth': [B, T, H, W] ground truth depth maps
            max_samples: Maximum number of samples to evaluate

        Returns:
            metrics: Dictionary of depth metrics
        """
        print("\nEvaluating depth reconstruction...")

        all_mae = []
        all_rmse = []
        all_delta_1 = []
        all_delta_2 = []
        all_delta_3 = []

        num_samples = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Depth")):
            if max_samples and num_samples >= max_samples:
                break

            video = batch['video'].to(self.device)  # [B, T, C, H, W]

            # Check if depth GT is available
            if 'depth' not in batch:
                print("Warning: No depth ground truth in batch, skipping depth evaluation")
                break

            gt_depth = batch['depth'].to(self.device)  # [B, T, H, W]

            B, T = video.shape[0], video.shape[1]

            for b in range(B):
                if max_samples and num_samples >= max_samples:
                    break

                # Evaluate on first frame
                frame_idx = 0
                pred_depth = self.depth_reconstructor.reconstruct_depth(
                    video[b:b+1],
                    frame_idx=frame_idx,
                )

                gt_depth_frame = gt_depth[b, frame_idx]

                # Compute metrics
                metrics = compute_depth_metrics(pred_depth, gt_depth_frame)

                all_mae.append(metrics['mae'])
                all_rmse.append(metrics['rmse'])
                all_delta_1.append(metrics['delta_1.25'])
                all_delta_2.append(metrics['delta_1.25^2'])
                all_delta_3.append(metrics['delta_1.25^3'])

                num_samples += 1

        # Aggregate metrics
        results = {
            'mae': np.mean(all_mae) if all_mae else float('nan'),
            'rmse': np.mean(all_rmse) if all_rmse else float('nan'),
            'delta_1.25': np.mean(all_delta_1) if all_delta_1 else float('nan'),
            'delta_1.25^2': np.mean(all_delta_2) if all_delta_2 else float('nan'),
            'delta_1.25^3': np.mean(all_delta_3) if all_delta_3 else float('nan'),
            'num_samples': num_samples,
        }

        return results

    @torch.no_grad()
    def evaluate_pose(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate camera pose estimation on a dataset.

        Args:
            dataloader: DataLoader providing batches with:
                - 'video': [B, T, C, H, W]
                - 'cameras': dict with 'extrinsics' [B, T, 4, 4] (ground truth poses)
            max_samples: Maximum number of samples to evaluate

        Returns:
            metrics: Dictionary of pose metrics
        """
        print("\nEvaluating camera pose estimation...")

        all_rotation_errors = []
        all_translation_errors = []
        all_ate = []

        num_samples = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Pose")):
            if max_samples and num_samples >= max_samples:
                break

            video = batch['video'].to(self.device)  # [B, T, C, H, W]

            # Check if camera GT is available
            if 'cameras' not in batch or 'extrinsics' not in batch['cameras']:
                print("Warning: No camera extrinsics in batch, skipping pose evaluation")
                break

            gt_extrinsics = batch['cameras']['extrinsics'].to(self.device)  # [B, T, 4, 4]

            B, T = video.shape[0], video.shape[1]

            for b in range(B):
                if max_samples and num_samples >= max_samples:
                    break

                # Estimate full trajectory
                pred_rotations, pred_translations = self.pose_estimator.estimate_trajectory(
                    video[b:b+1],
                    reference_frame=0,
                )

                # Extract GT rotations and translations
                gt_rotations = gt_extrinsics[b, :, :3, :3]  # [T, 3, 3]
                gt_translations = gt_extrinsics[b, :, :3, 3]  # [T, 3]

                # Compute metrics
                metrics = compute_pose_metrics(
                    pred_rotations, pred_translations,
                    gt_rotations, gt_translations,
                )

                all_rotation_errors.append(metrics['rotation_error_deg'])
                all_translation_errors.append(metrics['translation_error'])
                all_ate.append(metrics['ate'])

                num_samples += 1

        # Aggregate metrics
        results = {
            'rotation_error_deg': np.mean(all_rotation_errors) if all_rotation_errors else float('nan'),
            'translation_error': np.mean(all_translation_errors) if all_translation_errors else float('nan'),
            'ate': np.mean(all_ate) if all_ate else float('nan'),
            'num_samples': num_samples,
        }

        return results

    def evaluate_all(
        self,
        dataloader: DataLoader,
        tasks: List[str] = ['tracking', 'depth', 'pose'],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all tasks on a dataset.

        Args:
            dataloader: DataLoader
            tasks: List of tasks to evaluate
            max_samples: Maximum samples per task

        Returns:
            results: Dictionary mapping task name to metrics
        """
        results = {}

        if 'tracking' in tasks:
            results['tracking'] = self.evaluate_tracking(dataloader, max_samples)

        if 'depth' in tasks:
            results['depth'] = self.evaluate_depth(dataloader, max_samples)

        if 'pose' in tasks:
            results['pose'] = self.evaluate_pose(dataloader, max_samples)

        return results

    def save_results(
        self,
        results: Dict,
        save_path: str,
    ):
        """Save evaluation results to JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {save_path}")


def evaluate_checkpoint(
    checkpoint_path: str,
    dataloader: DataLoader,
    tasks: List[str] = ['tracking', 'depth', 'pose'],
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a checkpoint on a dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        dataloader: DataLoader
        tasks: List of tasks to evaluate
        max_samples: Maximum samples per task
        device: Device to use

    Returns:
        results: Evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Build model
    from ..models import build_d4rt_model
    from omegaconf import OmegaConf

    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model config")

    config = OmegaConf.create(checkpoint['config'])
    model = build_d4rt_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded (step {checkpoint.get('step', 'unknown')})")

    # Create evaluator
    evaluator = D4RTEvaluator(model, device=device)

    # Run evaluation
    print(f"\nEvaluating on {tasks}...")
    results = evaluator.evaluate_all(dataloader, tasks=tasks, max_samples=max_samples)

    return results


def print_results_table(results: Dict[str, Dict[str, float]]):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    for task_name, metrics in results.items():
        print(f"\n{task_name.upper()}:")
        print("-" * 40)

        for metric_name, value in metrics.items():
            if metric_name != 'num_samples':
                print(f"  {metric_name:.<30} {value:.4f}")

        if 'num_samples' in metrics:
            print(f"\n  Evaluated on {metrics['num_samples']} samples")

    print("\n" + "=" * 80)
