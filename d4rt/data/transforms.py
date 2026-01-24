"""Data augmentation and transforms for D4RT."""

import torch
import torchvision.transforms.functional as TF
import numpy as np
from typing import Dict, Tuple, Optional
import random


class VideoTransform:
    """Base class for video transforms."""

    def __call__(self, video_data: Dict, ground_truth: Dict) -> Tuple[Dict, Dict]:
        """
        Apply transform.

        Args:
            video_data: Dictionary with video frames and metadata
            ground_truth: Dictionary with ground truth data

        Returns:
            transformed_video_data, transformed_ground_truth
        """
        raise NotImplementedError


class Normalize(VideoTransform):
    """Normalize RGB values."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, video_data: Dict, ground_truth: Dict) -> Tuple[Dict, Dict]:
        frames = video_data['frames']
        frames = (frames - self.mean) / self.std
        video_data['frames'] = frames
        return video_data, ground_truth


class RandomHorizontalFlip(VideoTransform):
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, video_data: Dict, ground_truth: Dict) -> Tuple[Dict, Dict]:
        if random.random() < self.p:
            # Flip frames
            frames = video_data['frames']
            frames = torch.flip(frames, dims=[-1])  # Flip width dimension
            video_data['frames'] = frames

            # Update camera intrinsics (flip principal point)
            if 'cameras' in video_data:
                intrinsics = video_data['cameras']['intrinsics']
                W = frames.shape[-1]
                intrinsics = intrinsics.clone()
                intrinsics[:, 0, 2] = W - 1 - intrinsics[:, 0, 2]  # Flip cx
                video_data['cameras']['intrinsics'] = intrinsics

        return video_data, ground_truth


class RandomCrop(VideoTransform):
    """Random crop."""

    def __init__(self, crop_size: Tuple[int, int] = (224, 224)):
        self.crop_size = crop_size

    def __call__(self, video_data: Dict, ground_truth: Dict) -> Tuple[Dict, Dict]:
        frames = video_data['frames']
        T, C, H, W = frames.shape
        crop_h, crop_w = self.crop_size

        if H <= crop_h and W <= crop_w:
            return video_data, ground_truth

        # Random crop position
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # Crop frames
        frames = frames[:, :, top:top+crop_h, left:left+crop_w]
        video_data['frames'] = frames

        # Update camera intrinsics
        if 'cameras' in video_data:
            intrinsics = video_data['cameras']['intrinsics']
            intrinsics = intrinsics.clone()
            intrinsics[:, 0, 2] -= left  # Adjust cx
            intrinsics[:, 1, 2] -= top   # Adjust cy
            video_data['cameras']['intrinsics'] = intrinsics

        return video_data, ground_truth


class ColorJitter(VideoTransform):
    """Color jittering."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, video_data: Dict, ground_truth: Dict) -> Tuple[Dict, Dict]:
        frames = video_data['frames']
        T = frames.shape[0]

        # Apply same jitter to all frames
        brightness_factor = random.uniform(
            max(0, 1 - self.brightness), 1 + self.brightness
        )
        contrast_factor = random.uniform(
            max(0, 1 - self.contrast), 1 + self.contrast
        )
        saturation_factor = random.uniform(
            max(0, 1 - self.saturation), 1 + self.saturation
        )
        hue_factor = random.uniform(-self.hue, self.hue)

        # Apply to each frame
        frames_list = []
        for t in range(T):
            frame = frames[t]
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            frame = TF.adjust_saturation(frame, saturation_factor)
            frame = TF.adjust_hue(frame, hue_factor)
            frames_list.append(frame)

        frames = torch.stack(frames_list, dim=0)
        video_data['frames'] = frames

        return video_data, ground_truth


class Compose(VideoTransform):
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video_data: Dict, ground_truth: Dict) -> Tuple[Dict, Dict]:
        for transform in self.transforms:
            video_data, ground_truth = transform(video_data, ground_truth)
        return video_data, ground_truth


def get_train_transforms(config: Dict) -> Compose:
    """
    Get training transforms from config.

    Args:
        config: Configuration dictionary

    Returns:
        transforms: Composed transforms
    """
    transforms_list = []

    aug_config = config.get('augmentation', {})

    # Random crop
    if aug_config.get('random_crop', False):
        crop_size = tuple(aug_config.get('crop_size', [224, 224]))
        transforms_list.append(RandomCrop(crop_size))

    # Horizontal flip
    if aug_config.get('horizontal_flip', False):
        flip_prob = aug_config.get('flip_prob', 0.5)
        transforms_list.append(RandomHorizontalFlip(flip_prob))

    # Color jitter
    if aug_config.get('color_jitter', False):
        transforms_list.append(ColorJitter(
            brightness=aug_config.get('brightness', 0.2),
            contrast=aug_config.get('contrast', 0.2),
            saturation=aug_config.get('saturation', 0.2),
            hue=aug_config.get('hue', 0.1),
        ))

    # Normalization (always apply)
    transforms_list.append(Normalize())

    return Compose(transforms_list)


def get_val_transforms() -> Compose:
    """Get validation transforms (normalization only)."""
    return Compose([Normalize()])
