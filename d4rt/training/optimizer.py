"""Optimizer and scheduler utilities for D4RT."""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from typing import Dict, Any


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer from config.

    Supports parameter group-specific learning rates for cross-attention
    Q/K projections (qk_lr_multiplier) to help with gradient flow.

    Args:
        model: Model to optimize
        config: Optimizer configuration

    Returns:
        optimizer: Configured optimizer
    """
    optimizer_type = config.get('type', 'adamw').lower()
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 0.05)
    qk_lr_multiplier = config.get('qk_lr_multiplier', 1.0)

    # Build parameter groups
    if qk_lr_multiplier != 1.0:
        # Separate Q/K parameters from others for higher LR
        qk_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Match cross-attention Q and K projections
            if 'cross_attn' in name and ('q_proj' in name or 'k_proj' in name):
                qk_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': qk_params, 'lr': lr * qk_lr_multiplier, 'name': 'qk_params'},
            {'params': other_params, 'lr': lr, 'name': 'other_params'},
        ]

        print(f"Optimizer: Q/K params ({len(qk_params)}) LR={lr * qk_lr_multiplier:.2e}, "
              f"Other params ({len(other_params)}) LR={lr:.2e}")
    else:
        # Standard single parameter group
        param_groups = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == 'adamw':
        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay,
        )
    elif optimizer_type == 'adam':
        optimizer = Adam(
            param_groups,
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay,
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build learning rate scheduler from config.

    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration

    Returns:
        scheduler: Configured scheduler
    """
    scheduler_type = config.get('type', 'cosine').lower()
    warmup_steps = config.get('warmup_steps', 0)

    if scheduler_type == 'cosine':
        T_max = config.get('T_max', 100000)
        eta_min = config.get('eta_min', 1e-6)

        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max - warmup_steps,
            eta_min=eta_min,
        )
    elif scheduler_type == 'constant':
        # No scheduling, just constant LR
        main_scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Add warmup if specified
    if warmup_steps > 0 and main_scheduler is not None:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = main_scheduler

    return scheduler


class GradientClipper:
    """Gradient clipping utility."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipper.

        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm to use
        """
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self, parameters) -> float:
        """
        Clip gradients.

        Args:
            parameters: Model parameters

        Returns:
            total_norm: Total gradient norm before clipping
        """
        if self.max_norm <= 0:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.max_norm,
            norm_type=self.norm_type,
        )

        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
