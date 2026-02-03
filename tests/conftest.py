"""Shared test fixtures for D4RT tests."""
import pytest
import torch


@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_video(device):
    """Sample video [B=1, T=24, C=3, H=256, W=256]."""
    return torch.randn(1, 24, 3, 256, 256, device=device)


@pytest.fixture
def sample_video_small(device):
    """Smaller sample video for faster tests [B=1, T=8, C=3, H=128, W=128]."""
    return torch.randn(1, 8, 3, 128, 128, device=device)


@pytest.fixture
def sample_queries(device):
    """Sample queries dict with (u, v, t_src, t_tgt, t_cam)."""
    B, N = 1, 32
    queries = {
        'u': torch.rand(B, N, device=device),  # [0, 1]
        'v': torch.rand(B, N, device=device),  # [0, 1]
        't_src': torch.randint(0, 24, (B, N), device=device),
        't_tgt': torch.randint(0, 24, (B, N), device=device),
        't_cam': torch.randint(0, 24, (B, N), device=device),
    }
    return queries


@pytest.fixture
def sample_aspect_ratio(device):
    """Sample aspect ratio [B=1]."""
    return torch.tensor([1.0], device=device)  # Square video


@pytest.fixture
def d4rt_model_config():
    """Config matching paper architecture (Figure 7)."""
    return {
        'encoder': {
            'input_resolution': [24, 256, 256],
            'patch_size': [2, 16, 16],
            'hidden_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'use_paper_blocks': True,  # Figure 7: each block has local + global
            'use_patch_norm': False,   # No patch normalization per paper
        },
        'decoder': {
            'query_dim': 512,
            'context_dim': 768,
            'hidden_dim': 512,
            'num_layers': 8,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'output_uv': True,
            'output_normals': True,
            'output_motion': True,
        },
        'query_encoder': {
            'fourier': {
                'num_frequencies': 10,
                'max_frequency': 9,
            },
            'temporal': {
                'max_frames': 128,
                'embedding_dim': 256,
            },
            'patch_cnn': {
                'patch_size': 9,
                'channels': [64, 128],
                'output_dim': 256,
            },
            'output_dim': 512,
        },
    }


@pytest.fixture
def d4rt_model_config_small():
    """Smaller config for faster tests."""
    return {
        'encoder': {
            'input_resolution': [8, 128, 128],
            'patch_size': [2, 16, 16],
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 4,
            'mlp_ratio': 4.0,
            'use_paper_blocks': True,
            'use_patch_norm': False,
        },
        'decoder': {
            'query_dim': 256,
            'context_dim': 256,
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 4,
            'mlp_ratio': 4.0,
            'output_uv': True,
            'output_normals': True,
            'output_motion': True,
        },
        'query_encoder': {
            'fourier': {
                'num_frequencies': 10,
                'max_frequency': 9,
            },
            'temporal': {
                'max_frames': 128,
                'embedding_dim': 64,
            },
            'patch_cnn': {
                'patch_size': 9,
                'channels': [32, 64],
                'output_dim': 64,
            },
            'output_dim': 256,
        },
    }


@pytest.fixture
def sample_encoder_features(device):
    """Sample encoder features [B=1, num_patches=3072, embed_dim=768]."""
    return torch.randn(1, 3072, 768, device=device)


@pytest.fixture
def sample_query_features(device):
    """Sample query features [B=1, N=32, hidden_dim=512]."""
    return torch.randn(1, 32, 512, device=device)


@pytest.fixture
def sample_predictions(device):
    """Sample model predictions with all output heads."""
    N = 32
    return {
        'xyz': torch.randn(1, N, 3, device=device),
        'uv': torch.rand(1, N, 2, device=device),  # [0, 1]
        'normals': torch.nn.functional.normalize(
            torch.randn(1, N, 3, device=device), dim=-1
        ),
        'motion': torch.randn(1, N, 3, device=device),
        'visibility': torch.randn(1, N, 1, device=device),
        'confidence': torch.randn(1, N, 1, device=device),
    }


@pytest.fixture
def sample_targets(device):
    """Sample ground truth targets."""
    N = 32
    return {
        'xyz': torch.randn(1, N, 3, device=device),
        'uv': torch.rand(1, N, 2, device=device),
        'normals': torch.nn.functional.normalize(
            torch.randn(1, N, 3, device=device), dim=-1
        ),
        'motion': torch.randn(1, N, 3, device=device),
        'visibility': torch.randint(0, 2, (1, N, 1), device=device).float(),
    }
