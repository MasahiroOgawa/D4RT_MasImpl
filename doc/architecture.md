# D4RT Architecture

This document describes the detailed architecture of the D4RT model based on Figure 7 from the original paper.

## Overview

D4RT uses a unified transformer architecture that encodes video into a global scene representation, then decodes arbitrary queries into 3D positions and auxiliary outputs.

## Architecture Diagram (Figure 7)

```
Input Video [B, T, C, H, W]              Original Aspect Ratio
(Resize to square 256×256)                      W/H
         │                                       │
         ▼                                       ▼
┌─────────────────────┐                  ┌──────────────┐
│     Tokenizer       │                  │      FC      │
│  (2×16×16 patches)  │                  │ (scalar→768) │
│  (NO normalization) │                  │              │
└─────────────────────┘                  └──────────────┘
         │                                       │
         ▼                                       │
┌─────────────────────┐                          │
│        PE           │                          │
│ (pos encoding)      │               (NO PE for AR token)
└─────────────────────┘                          │
         │                                       │
         ▼                                       ▼
   [B, 3072, 768]                          [B, 1, 768]
   (video + PE)                            (AR token)
         │                                       │
         └───────────── Concat ──────────────────┘
                          │
                          ▼
                   [B, 3073, 768]
                          │
         ┌────────────────┴────────────────┐
         │       N Encoder Blocks          │
         │  ┌───────────────────────────┐  │
         │  │ Per-Frame Self-Attention  │  │  ← Local (256 tokens/frame)
         │  │           ↓               │  │    (AR token participates in global only)
         │  │          MLP              │  │
         │  │           ↓               │  │
         │  │ Global Self-Attention     │  │  ← All 3073 tokens
         │  │           ↓               │  │
         │  │          MLP              │  │
         │  └───────────────────────────┘  │
         │         (× N layers)            │
         └─────────────────────────────────┘
                          │
                          ▼
                  Remove AR token
                          │
                          ▼
              Global Scene Representation F
                   [B, 3072, 768]
                          │
                          ▼
┌─────────────────────────────────────┐
│    Query Encoder                    │
│  - Fourier(u,v) → 40D               │
│  - Temporal embeddings → 768D       │
│  - 9×9 RGB patch CNN → 256D         │
│  → Concat + Project → 512D          │
└─────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────┐
│   Cross-Attention Decoder (8L)      │
│  Self-Attn → Cross-Attn → FFN       │
└─────────────────────────────────────┘
                          │
                          ▼
   Output Heads:
   - xyz: 3D position [B, N, 3]
   - uv: 2D coordinates [B, N, 2]
   - normals: surface normals [B, N, 3]
   - motion: displacement [B, N, 3]
   - visibility: occlusion [B, N, 1]
   - confidence: prediction quality [B, N, 1]
```

## Components

### 1. Video Tokenizer

Converts input video into patch tokens:
- **Input**: Video tensor `[B, T, C, H, W]` (e.g., `[1, 24, 3, 256, 256]`)
- **Patch size**: `(2, 16, 16)` - temporal × height × width
- **Output**: `[B, num_patches, embed_dim]` = `[B, 3072, 768]`
  - 24 frames / 2 = 12 temporal patches
  - 256 / 16 = 16 spatial patches per dimension
  - Total: 12 × 16 × 16 = 3072 patches

**Note**: No LayerNorm after patch embedding to preserve brightness information.

### 2. Aspect Ratio Token

Embeds the original aspect ratio (W/H) as a separate token (Paper p.3):
- **Input**: Scalar W/H ratio
- **Process**: FC layer `(1 → embed_dim)`
- **Output**: Single token `[B, 1, 768]`
- **Important**: NO positional encoding for AR token

### 3. Positional Encoding

- Applied only to video tokens, NOT to AR token
- Learnable positional embeddings: `[1, 3072, 768]`

### 4. Encoder Blocks

Each encoder block contains **BOTH** attention types in sequence (Figure 7):

```
Input tokens [B, 3073, 768]
       │
       ▼
┌─────────────────────────┐
│ Per-Frame Self-Attention │  ← Only video tokens (256 per frame)
│     (Local attention)    │    AR token skips this
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│         MLP             │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Global Self-Attention  │  ← All 3073 tokens
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│         MLP             │
└─────────────────────────┘
       │
       ▼
Output tokens [B, 3073, 768]
```

**Why both attention types per block?**
- **Local attention**: Preserves spatial diversity within each frame, prevents token homogenization
- **Global attention**: Enables cross-frame temporal reasoning

### 5. Query Encoder

Encodes query points into feature vectors:
- **Spatial encoding**: Fourier features of (u, v) coordinates → 40D
- **Temporal encoding**: Learnable embeddings for t_src, t_tgt → 768D each
- **Appearance encoding**: 9×9 RGB patch at query location → CNN → 256D
- **Final**: Concatenate + project → 512D

### 6. Cross-Attention Decoder

8 transformer decoder layers:
- Self-attention among query tokens
- Cross-attention to encoder features F
- Feed-forward network

### 7. Output Heads

Linear projections from decoder output (512D) to:

| Output | Dimension | Description |
|--------|-----------|-------------|
| xyz | 3 | 3D world position |
| uv | 2 | 2D image coordinates |
| normals | 3 | Surface normal (unit normalized) |
| motion | 3 | Motion displacement |
| visibility | 1 | Occlusion logit |
| confidence | 1 | Prediction confidence logit |

## Model Configurations

| Model | Encoder Layers | Hidden Dim | Heads | Parameters |
|-------|---------------|------------|-------|------------|
| ViT-B | 12 | 768 | 12 | ~230M |
| ViT-L | 24 | 1024 | 16 | ~451M |
| ViT-g | 40 | 1408 | 16 | ~1.144B |

## Related Files

- `d4rt/models/encoder.py` - Spatio-temporal ViT encoder
- `d4rt/models/decoder.py` - Cross-attention decoder
- `d4rt/models/components/aspect_ratio_token.py` - Aspect ratio embedding
- `d4rt/models/components/encoder_block.py` - Local + global attention block
- `configs/model/vit_b_d4rt.yaml` - ViT-B configuration
