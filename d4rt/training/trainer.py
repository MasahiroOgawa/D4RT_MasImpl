"""Main training loop for D4RT."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Optional, Any
from pathlib import Path
import time
from tqdm import tqdm

from .optimizer import build_optimizer, build_scheduler, GradientClipper
from .checkpointing import CheckpointManager
from ..losses import CompositeLoss


class D4RTTrainer:
    """
    Main trainer for D4RT.

    Supports:
    - Single GPU and multi-GPU training (DDP)
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing and logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: CompositeLoss,
        config: Dict[str, Any],
        device: torch.device,
        logger: Optional[Any] = None,
        distributed: bool = False,
        local_rank: int = 0,
    ):
        """
        Initialize trainer.

        Args:
            model: D4RT model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            loss_fn: Loss function
            config: Training configuration
            device: Device to train on
            logger: Logger (WandB, TensorBoard, etc.)
            distributed: Whether using distributed training
            local_rank: Local rank for distributed training
        """
        self.config = config
        self.device = device
        self.logger = logger
        self.distributed = distributed
        self.local_rank = local_rank
        self.is_main_process = (not distributed) or (local_rank == 0)

        # Model
        self.model = model.to(device)
        if distributed:
            self.model = DDP(
                model,
                device_ids=[local_rank],
                find_unused_parameters=config.get('find_unused_parameters', False),
            )

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        self.loss_fn = loss_fn.to(device)

        # Optimizer and scheduler
        self.optimizer = build_optimizer(self.model, config.get('optimizer', {}))
        self.scheduler = build_scheduler(self.optimizer, config.get('scheduler', {}))

        # Training settings
        self.max_steps = config.get('max_steps', 100000)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.log_every_n_steps = config.get('log_every_n_steps', 100)
        self.val_every_n_steps = config.get('val_every_n_steps', 1000)
        self.save_every_n_steps = config.get('save_every_n_steps', 5000)

        # Mixed precision
        self.use_amp = config.get('mixed_precision', None) in ['fp16', 'bf16']
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Gradient clipping
        self.grad_clipper = GradientClipper(
            max_norm=config.get('clip_grad_norm', 1.0)
        )

        # Checkpointing
        if self.is_main_process:
            self.checkpoint_manager = CheckpointManager(
                save_dir=config.get('save_dir', 'checkpoints'),
                keep_last_n=config.get('save_top_k', 3),
                save_best=config.get('save_best', True),
                monitor=config.get('monitor', 'val/loss_total'),
                mode=config.get('mode', 'min'),
            )
        else:
            self.checkpoint_manager = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.

        Args:
            resume_from: Path to checkpoint to resume from (optional)
        """
        # Resume from checkpoint if specified
        if resume_from and self.is_main_process:
            self._load_checkpoint(resume_from)

        if self.is_main_process:
            print(f"Starting training for {self.max_steps} steps")
            print(f"Training on device: {self.device}")
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"Mixed precision: {self.use_amp}")

        # Training loop
        self.model.train()
        train_iter = iter(self.train_loader)

        pbar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            disable=not self.is_main_process,
            desc="Training",
        )

        while self.global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart iterator
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                self.current_epoch += 1

            # Training step
            loss, metrics = self._train_step(batch)

            # Logging
            if self.global_step % self.log_every_n_steps == 0:
                self._log_metrics(metrics, prefix='train')

            # Validation
            if self.global_step % self.val_every_n_steps == 0:
                val_metrics = self._validate()
                self._log_metrics(val_metrics, prefix='val')

                # Save checkpoint
                if self.is_main_process:
                    all_metrics = {**metrics, **val_metrics}
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model.module if self.distributed else self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.global_step,
                        metrics=all_metrics,
                        config=self.config,
                    )

                self.model.train()

            # Regular checkpoint saving
            if (self.global_step % self.save_every_n_steps == 0 and
                self.global_step % self.val_every_n_steps != 0):
                if self.is_main_process:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model.module if self.distributed else self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        step=self.global_step,
                        metrics=metrics,
                        config=self.config,
                    )

            pbar.update(1)
            pbar.set_postfix({'loss': f"{metrics['train/loss_total']:.4f}"})

        pbar.close()

        if self.is_main_process:
            print(f"Training completed at step {self.global_step}")

    def _train_step(self, batch: Dict) -> tuple[float, Dict[str, float]]:
        """
        Single training step.

        Args:
            batch: Training batch

        Returns:
            loss: Loss value
            metrics: Dictionary of metrics
        """
        # Move batch to device
        batch = self._move_batch_to_device(batch)

        # Forward pass with mixed precision
        with autocast('cuda', enabled=self.use_amp):
            # Forward
            outputs = self.model(batch['video'], batch['queries'])

            # Compute loss
            loss, loss_dict = self.loss_fn(
                predictions=outputs,
                targets=batch['targets'],
                cameras=batch['cameras'],
                queries=batch['queries'],
            )

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            grad_norm = self.grad_clipper.clip(self.model.parameters())

            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            loss_dict['train/grad_norm'] = grad_norm
            loss_dict['train/lr'] = self.optimizer.param_groups[0]['lr']

        # Increment step counter
        self.global_step += 1

        # Prepare metrics
        metrics = {f'train/{k}': v for k, v in loss_dict.items()}
        metrics['train/step'] = self.global_step
        metrics['train/epoch'] = self.current_epoch

        return loss.item() * self.gradient_accumulation_steps, metrics

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Validation loop.

        Returns:
            metrics: Validation metrics
        """
        if self.is_main_process:
            print("\nRunning validation...")

        self.model.eval()

        total_loss = 0.0
        total_loss_dict = {}
        num_batches = 0

        for batch in tqdm(
            self.val_loader,
            disable=not self.is_main_process,
            desc="Validation",
        ):
            batch = self._move_batch_to_device(batch)

            # Forward pass
            outputs = self.model(batch['video'], batch['queries'])

            # Compute loss
            loss, loss_dict = self.loss_fn(
                predictions=outputs,
                targets=batch['targets'],
                cameras=batch['cameras'],
                queries=batch['queries'],
            )

            # Accumulate
            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] = total_loss_dict.get(k, 0.0) + v

            num_batches += 1

        # Average metrics
        avg_metrics = {
            f'val/{k}': v / num_batches
            for k, v in total_loss_dict.items()
        }

        if self.is_main_process:
            print(f"Validation loss: {avg_metrics['val/loss_total']:.4f}")

        return avg_metrics

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}

        for key, value in batch.items():
            if key == 'metadata':
                device_batch[key] = value
            elif isinstance(value, dict):
                device_batch[key] = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value.to(self.device, non_blocking=True)

        return device_batch

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics to logger."""
        if not self.is_main_process:
            return

        if self.logger is not None:
            self.logger.log(metrics, step=self.global_step)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        if self.checkpoint_manager is None:
            return

        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model.module if self.distributed else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.global_step = checkpoint.get('step', 0)
        print(f"Resumed from step {self.global_step}")
