"""Checkpointing utilities for D4RT."""

import torch
import os
from pathlib import Path
from typing import Dict, Optional, Any
import json


class CheckpointManager:
    """Manages saving and loading checkpoints."""

    def __init__(
        self,
        save_dir: str,
        keep_last_n: int = 3,
        save_best: bool = True,
        monitor: str = 'val/loss_total',
        mode: str = 'min',
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save best checkpoint
            monitor: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for best checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode

        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_history = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
    ) -> str:
        """
        Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state (optional)
            step: Training step
            metrics: Current metrics
            config: Training configuration (optional)

        Returns:
            checkpoint_path: Path to saved checkpoint
        """
        # Prepare checkpoint dict
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if config is not None:
            checkpoint['config'] = config

        # Save regular checkpoint
        checkpoint_name = f'checkpoint_step_{step:07d}.pth'
        checkpoint_path = self.save_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint history
        self.checkpoint_history.append((step, checkpoint_path))

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        # Save best checkpoint if applicable
        if self.save_best:
            self._save_best_checkpoint(checkpoint, metrics, checkpoint_path)

        # Save latest checkpoint link
        latest_path = self.save_dir / 'checkpoint_latest.pth'
        if latest_path.exists():
            latest_path.unlink()
        os.symlink(checkpoint_path.name, latest_path)

        return str(checkpoint_path)

    def _save_best_checkpoint(
        self,
        checkpoint: Dict,
        metrics: Dict[str, float],
        checkpoint_path: Path,
    ):
        """Save best checkpoint based on monitored metric."""
        if self.monitor not in metrics:
            return

        current_metric = metrics[self.monitor]

        is_best = False
        if self.mode == 'min':
            is_best = current_metric < self.best_metric
        else:
            is_best = current_metric > self.best_metric

        if is_best:
            self.best_metric = current_metric
            best_path = self.save_dir / 'checkpoint_best.pth'

            # Remove old best checkpoint if exists
            if best_path.exists():
                best_path.unlink()

            # Create symlink to best checkpoint
            os.symlink(checkpoint_path.name, best_path)

            # Save best metrics
            best_metrics_path = self.save_dir / 'best_metrics.json'
            with open(best_metrics_path, 'w') as f:
                json.dump({
                    'step': checkpoint['step'],
                    'metric': self.monitor,
                    'value': current_metric,
                    'all_metrics': metrics,
                }, f, indent=2)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return

        # Sort by step
        self.checkpoint_history.sort(key=lambda x: x[0])

        # Remove oldest checkpoints
        to_remove = self.checkpoint_history[:-self.keep_last_n]
        for step, path in to_remove:
            if path.exists() and not path.is_symlink():
                # Don't remove if it's the best checkpoint
                best_path = self.save_dir / 'checkpoint_best.pth'
                if best_path.exists() and best_path.resolve() == path.resolve():
                    continue
                path.unlink()

        # Update history
        self.checkpoint_history = self.checkpoint_history[-self.keep_last_n:]

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)

        Returns:
            checkpoint: Checkpoint dictionary with metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint

    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find latest checkpoint in save directory.

        Returns:
            checkpoint_path: Path to latest checkpoint, or None if not found
        """
        latest_path = self.save_dir / 'checkpoint_latest.pth'
        if latest_path.exists():
            return latest_path.resolve()

        # Fallback: find by filename
        checkpoints = sorted(self.save_dir.glob('checkpoint_step_*.pth'))
        if checkpoints:
            return checkpoints[-1]

        return None

    def find_best_checkpoint(self) -> Optional[Path]:
        """
        Find best checkpoint.

        Returns:
            checkpoint_path: Path to best checkpoint, or None if not found
        """
        best_path = self.save_dir / 'checkpoint_best.pth'
        if best_path.exists():
            return best_path.resolve()

        return None
