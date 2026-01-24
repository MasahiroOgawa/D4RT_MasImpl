# D4RT TRAINING LOOP - DETAILED SPECIFICATIONS

## Architecture Specifications

### Encoder: Spatio-Temporal ViT
```python
class SpatioTemporalViT:
    """3D Vision Transformer for video encoding"""

    # Model Configurations:
    # ViT-B (small):  12 layers, 768 dim, 12 heads, ~86M params
    # ViT-L (medium): 24 layers, 1024 dim, 16 heads, ~307M params
    # ViT-g (large):  40 layers, 1408 dim, 16 heads, ~1B params

    # Input/Output Specifications
    patch_size = (2, 16, 16)  # (temporal, height, width)
    input_resolution = (48, 256, 256)  # (T, H, W)
    num_patches = 6144  # (48/2) * (256/16) * (256/16) = 24 * 16 * 16

    # Output: Global scene representation F [B, 6144, hidden_dim]
```

### Decoder: Cross-Attention Transformer
```python
class CrossAttentionDecoder:
    """8-layer cross-attention decoder"""
    num_layers = 8
    hidden_dim = 512
    num_heads = 8
    params = 144M  # Independent of encoder size

    # Input: Query embeddings [B, N, 512]
    # Context: Encoder features F [B, 6144, encoder_hidden_dim]
    # Output: 3D positions (x, y, z) [B, N, 3]
```

## Query Encoding Specification

```python
def encode_query(u, v, t_src, t_tgt, t_cam, video):
    """
    Encodes a 5-tuple query into decoder input

    Query components:
    - (u, v): Normalized pixel coordinates [0, 1]
    - t_src: Source frame index where point is observed
    - t_tgt: Target timestamp to predict 3D position
    - t_cam: Camera reference frame for coordinate system

    Encoding process:
    """

    # 1. Fourier Positional Encoding for (u, v)
    # Uses 10 frequency bands → 40 dimensions total
    freqs = 2^k for k in range(10)  # [1, 2, 4, ..., 512]
    uv_features = []
    for freq in freqs:
        uv_features.append([sin(u * freq * π), cos(u * freq * π)])
        uv_features.append([sin(v * freq * π), cos(v * freq * π)])
    # Result: [B, N, 40]

    # 2. Learned Temporal Embeddings
    # Separate embedding tables for each temporal component
    t_src_emb = temporal_embedding_table[t_src]   # [B, N, 256]
    t_tgt_emb = temporal_embedding_table[t_tgt]   # [B, N, 256]
    t_cam_emb = temporal_embedding_table[t_cam]   # [B, N, 256]

    # 3. Local RGB Patch Features
    # Extract 9×9 patch centered at (u, v) from frame t_src
    patch = extract_patch(video[t_src], u, v, size=9)  # [B, N, 3, 9, 9]
    # Process through small CNN
    patch_features = patch_cnn(patch)  # [B, N, 256]

    # 4. Concatenate and Project
    query = concat([uv_features, t_src_emb, t_tgt_emb, t_cam_emb, patch_features])
    # Total: 40 + 256 + 256 + 256 + 256 = 1064 dims
    query = linear_projection(query)  # [B, N, 512] (decoder hidden_dim)

    return query
```

## Query Sampling Strategy

```python
def sample_training_queries(batch, num_queries=2048):
    """
    Strategic sampling for effective training

    Sampling distribution:
    - 50% (1024): Visible points in source frame (has GT 3D)
    - 25% (512):  Occluded points (tests visibility prediction)
    - 25% (512):  Random background points (coverage)
    """

    # Visible points: Points with known 3D ground truth
    visible_mask = batch['visibility_mask']  # [B, T, H, W]
    visible_queries = sample_from_mask(visible_mask, n=1024)

    # Occluded points: Points behind objects or out of view
    occluded_mask = ~visible_mask & batch['has_depth']
    occluded_queries = sample_from_mask(occluded_mask, n=512)

    # Random points: Uniform sampling across entire frame
    random_queries = sample_uniform_grid(n=512)

    # Temporal sampling: Uniformly sample t_src, t_tgt, t_cam
    t_src = random.randint(0, T-1, size=num_queries)
    t_tgt = random.randint(0, T-1, size=num_queries)
    t_cam = random.randint(0, T-1, size=num_queries)

    queries = {
        'u': concat([visible_queries.u, occluded_queries.u, random_queries.u]),
        'v': concat([visible_queries.v, occluded_queries.v, random_queries.v]),
        't_src': t_src,
        't_tgt': t_tgt,
        't_cam': t_cam
    }

    return queries
```

## Multi-Task Loss Functions

```python
# Loss weights (as per paper)
LOSS_WEIGHTS = {
    'l1_3d': 1.0,        # Primary 3D position loss
    'l2_2d': 0.1,        # 2D reprojection error
    'normal': 0.05,      # Surface normal alignment
    'motion': 0.1,       # Temporal consistency
    'visibility': 0.1,   # Occlusion prediction
}

def compute_losses(predictions, batch, cameras):
    """
    Computes all training losses

    Args:
        predictions: {
            'xyz': [B, N, 3],        # Predicted 3D positions
            'visibility': [B, N, 1], # Occlusion logits
        }
        batch: {
            'gt_xyz': [B, N, 3],
            'gt_uv': [B, N, 2],
            'gt_visibility': [B, N],
            'gt_normals': [B, N, 3],
        }
        cameras: {
            'intrinsics': [B, T, 3, 3],
            'extrinsics': [B, T, 4, 4],
        }
    """

    # 1. L1 3D Position Loss (primary supervision)
    # Normalize by scene scale to make loss scale-invariant
    scene_scale = batch['scene_bounds'].max()
    loss_3d = l1_loss(predictions['xyz'] / scene_scale,
                      batch['gt_xyz'] / scene_scale)

    # 2. 2D Reprojection Loss
    # Project predicted 3D to 2D using camera parameters
    pred_uv = project_3d_to_2d(predictions['xyz'],
                               cameras['intrinsics'],
                               cameras['extrinsics'])
    loss_2d = l2_loss(pred_uv, batch['gt_uv'])

    # 3. Surface Normal Loss
    # Compute normals from predicted point cloud
    pred_normals = estimate_normals_from_points(predictions['xyz'])
    loss_normal = 1 - cosine_similarity(pred_normals, batch['gt_normals'])

    # 4. Motion Consistency Loss
    # Consecutive frames should have smooth motion
    xyz_t = predictions['xyz'][batch['t_tgt'] == t]
    xyz_t1 = predictions['xyz'][batch['t_tgt'] == t+1]
    predicted_motion = xyz_t1 - xyz_t
    loss_motion = l1_loss(predicted_motion, batch['gt_motion'])

    # 5. Visibility Loss
    # Binary classification: is point visible?
    loss_visibility = binary_cross_entropy(
        predictions['visibility'],
        batch['gt_visibility']
    )

    # Weighted sum
    total_loss = (
        LOSS_WEIGHTS['l1_3d'] * loss_3d +
        LOSS_WEIGHTS['l2_2d'] * loss_2d +
        LOSS_WEIGHTS['normal'] * loss_normal +
        LOSS_WEIGHTS['motion'] * loss_motion +
        LOSS_WEIGHTS['visibility'] * loss_visibility
    )

    return total_loss, {
        'loss_3d': loss_3d.item(),
        'loss_2d': loss_2d.item(),
        'loss_normal': loss_normal.item(),
        'loss_motion': loss_motion.item(),
        'loss_visibility': loss_visibility.item(),
    }
```

## Complete Training Loop

```python
for batch in dataloader:
    # Input: Video frames [B, T, C, H, W]
    # Typical: B=4, T=48, C=3, H=W=256
    video_frames = batch['video']

    # STAGE 1: ENCODE
    # Generate global scene representation F
    # Output shape: [B, 6144, hidden_dim]
    F = encoder(video_frames)

    # STAGE 2: SAMPLE QUERIES
    # Sample 2048 queries per batch using strategic distribution
    queries = sample_training_queries(batch, num_queries=2048)

    # STAGE 3: ENCODE QUERIES
    # Convert 5-tuple queries to decoder input embeddings
    query_embeddings = encode_query(
        queries['u'], queries['v'],
        queries['t_src'], queries['t_tgt'], queries['t_cam'],
        video_frames
    )  # [B, 2048, 512]

    # STAGE 4: DECODE
    # Cross-attention decoder predicts 3D positions
    predictions = decoder(query_embeddings, F)
    # Output: {
    #     'xyz': [B, 2048, 3],
    #     'visibility': [B, 2048, 1]
    # }

    # STAGE 5: COMPUTE LOSSES
    total_loss, loss_dict = compute_losses(
        predictions,
        batch,
        cameras=batch['cameras']
    )

    # STAGE 6: OPTIMIZE
    total_loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()

    # Logging
    if step % 100 == 0:
        log_metrics(loss_dict)

    if step % 1000 == 0:
        save_checkpoint(model, optimizer, step)
```

## Training Hyperparameters

```python
# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05
)

# Learning rate scheduler
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=500000,  # Total training steps
    eta_min=1e-6
)

# Warmup for first 10k steps
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=10000
)

# Training configuration
config = {
    'batch_size': 4,  # per GPU
    'num_gpus': 8,
    'gradient_accumulation_steps': 1,
    'max_steps': 500000,
    'mixed_precision': 'bf16',  # Use bfloat16 on A100/H100
    'gradient_checkpointing': True,  # Save memory for ViT-g
    'num_queries_per_step': 2048,
}
```