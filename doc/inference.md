# D4RT INFERENCE PIPELINE - DETAILED SPECIFICATIONS

## Overview

D4RT inference is a two-stage process:
1. **Encode once**: Process the entire video to get global representation F
2. **Query many times**: Perform various tasks by querying F with different query patterns

This design enables efficient inference - the expensive encoder runs once, then lightweight queries can be processed in batches.

## Query Construction

```python
def create_query(u, v, t_src, t_tgt, t_cam, video):
    """
    Creates a query embedding for the decoder

    Args:
        u, v: Normalized pixel coordinates [0, 1]
        t_src: Frame where point is observed (appearance reference)
        t_tgt: Target timestamp to predict 3D position
        t_cam: Camera reference frame for coordinate system
        video: Input video [T, 3, H, W]

    Returns:
        query: Query embedding [512] ready for decoder
    """

    # 1. Fourier encoding for (u, v) - 10 frequencies
    uv_features = fourier_encode([u, v], num_freqs=10)  # 40 dims

    # 2. Temporal embeddings (learned)
    t_src_emb = temporal_embedding[t_src]  # 256 dims
    t_tgt_emb = temporal_embedding[t_tgt]  # 256 dims
    t_cam_emb = temporal_embedding[t_cam]  # 256 dims

    # 3. Extract 9×9 RGB patch from frame t_src
    patch = extract_patch(video[t_src], u, v, size=9)  # [3, 9, 9]

    # 4. Process patch through CNN
    patch_features = patch_cnn(patch)  # 256 dims

    # 5. Concatenate and project to decoder dimension
    query = concat([uv_features, t_src_emb, t_tgt_emb, t_cam_emb, patch_features])
    query = linear_project(query)  # → 512 dims

    return query
```

## Task 1: Point Tracking

Track sparse points through the video by fixing spatial location and varying time.

```python
def track_points(model, video, initial_points, start_frame=0):
    """
    Track sparse points through video

    Args:
        model: D4RT model
        video: Input video [T, 3, H, W], T=48, H=W=256
        initial_points: [N, 2] array of (u, v) coordinates in [0, 1]
        start_frame: Frame index where points are defined

    Returns:
        trajectories: [N, T, 3] array of (x, y, z) in camera coords
        visibility: [N, T] boolean array indicating if point is visible
    """

    T, _, H, W = video.shape
    N = len(initial_points)

    model.eval()
    with torch.no_grad():
        # STAGE 1: Encode video (once)
        F = model.encoder(video.unsqueeze(0))  # [1, 6144, hidden_dim]

        trajectories = []
        visibilities = []

        # STAGE 2: Query for each frame
        for t in range(T):
            # Create queries: fixed (u,v,t_src), varying t_tgt
            queries = []
            for point in initial_points:
                query = model.query_encoder(
                    u=point[0],
                    v=point[1],
                    t_src=start_frame,   # Where point appears
                    t_tgt=t,              # Target time
                    t_cam=t,              # Camera frame = target frame
                    video=video
                )
                queries.append(query)

            queries = torch.stack(queries)  # [N, 512]

            # Decode to get 3D positions
            output = model.decoder(queries.unsqueeze(0), F)
            xyz = output['xyz'].squeeze(0)  # [N, 3]
            vis = output['visibility'].squeeze(0)  # [N, 1]

            trajectories.append(xyz)
            visibilities.append(vis > 0)

        trajectories = torch.stack(trajectories, dim=1)  # [N, T, 3]
        visibilities = torch.stack(visibilities, dim=1)  # [N, T]

    return trajectories, visibilities


# Usage example
initial_points = torch.tensor([
    [0.5, 0.5],  # Center of frame
    [0.3, 0.7],  # Top-left region
])
trajectories, visibility = track_points(model, video, initial_points, start_frame=0)

# trajectories[0] contains 3D trajectory of first point across all frames
# visibility[0] indicates when first point is visible (not occluded)
```

## Task 2: Dense Depth Reconstruction

Reconstruct depth for all pixels in a specific frame.

```python
def reconstruct_depth(model, video, frame_idx, batch_size=4096):
    """
    Reconstruct dense depth map for a specific frame

    Args:
        model: D4RT model
        video: Input video [T, 3, H, W]
        frame_idx: Frame index to reconstruct depth for
        batch_size: Number of queries to process at once (for memory)

    Returns:
        depth_map: [H, W] depth values (z-coordinates in camera space)
    """

    T, C, H, W = video.shape

    model.eval()
    with torch.no_grad():
        # STAGE 1: Encode video (once)
        F = model.encoder(video.unsqueeze(0))  # [1, 6144, hidden_dim]

        # STAGE 2: Create dense grid of pixel coordinates
        u_coords = torch.linspace(0, 1, W)
        v_coords = torch.linspace(0, 1, H)
        u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')

        # Flatten to list of all pixels
        u_flat = u_grid.reshape(-1)  # [H*W]
        v_flat = v_grid.reshape(-1)  # [H*W]
        num_pixels = H * W

        # STAGE 3: Process in batches for memory efficiency
        depth_values = []

        for i in range(0, num_pixels, batch_size):
            batch_u = u_flat[i:i+batch_size]
            batch_v = v_flat[i:i+batch_size]
            batch_n = len(batch_u)

            # Create queries for this batch
            # Query pattern: same frame for source, target, and camera
            queries = model.query_encoder(
                u=batch_u,
                v=batch_v,
                t_src=torch.full((batch_n,), frame_idx),
                t_tgt=torch.full((batch_n,), frame_idx),
                t_cam=torch.full((batch_n,), frame_idx),
                video=video
            )  # [batch_n, 512]

            # Decode to get 3D positions
            output = model.decoder(queries.unsqueeze(0), F)
            xyz = output['xyz'].squeeze(0)  # [batch_n, 3]

            # Extract depth (z-coordinate)
            depth = xyz[:, 2]
            depth_values.append(depth)

        # Concatenate and reshape to image
        depth_map = torch.cat(depth_values)  # [H*W]
        depth_map = depth_map.reshape(H, W)  # [H, W]

    return depth_map


# Usage example
frame_idx = 24  # Middle frame
depth_map = reconstruct_depth(model, video, frame_idx)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(depth_map.cpu().numpy(), cmap='turbo')
plt.colorbar(label='Depth (m)')
plt.title(f'Depth Map - Frame {frame_idx}')
plt.savefig('depth_map.png')
```

## Task 3: Camera Pose Estimation

Estimate relative camera pose between two frames using Umeyama alignment.

```python
def estimate_camera_pose(model, video, target_frame, ref_frame=0, num_points=256):
    """
    Estimate camera pose (R, t) from ref_frame to target_frame

    Algorithm:
    1. Sample sparse grid of points
    2. Get 3D positions in reference camera: decoder(t_cam=ref)
    3. Get same points in target camera: decoder(t_cam=target)
    4. Solve rigid transformation using Umeyama algorithm

    Args:
        model: D4RT model
        video: Input video [T, 3, H, W]
        target_frame: Frame to estimate pose for
        ref_frame: Reference frame (usually 0)
        num_points: Number of sparse points to use (more = more robust)

    Returns:
        R: [3, 3] rotation matrix from ref to target
        t: [3] translation vector from ref to target
        scale: Scalar scale factor (if estimating similarity transform)
    """

    model.eval()
    with torch.no_grad():
        # STAGE 1: Encode video (once)
        F = model.encoder(video.unsqueeze(0))  # [1, 6144, hidden_dim]

        # STAGE 2: Sample sparse grid of points
        # Use stratified sampling for better coverage
        grid_size = int(np.sqrt(num_points))
        u_coords = torch.linspace(0.1, 0.9, grid_size)
        v_coords = torch.linspace(0.1, 0.9, grid_size)
        u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')
        sparse_u = u_grid.reshape(-1)[:num_points]
        sparse_v = v_grid.reshape(-1)[:num_points]

        # STAGE 3: Get 3D positions in reference camera
        queries_ref = model.query_encoder(
            u=sparse_u,
            v=sparse_v,
            t_src=torch.full((num_points,), ref_frame),
            t_tgt=torch.full((num_points,), ref_frame),
            t_cam=torch.full((num_points,), ref_frame),  # Key: ref camera
            video=video
        )
        output_ref = model.decoder(queries_ref.unsqueeze(0), F)
        xyz_ref = output_ref['xyz'].squeeze(0)  # [num_points, 3]

        # STAGE 4: Get same points in target camera
        queries_tgt = model.query_encoder(
            u=sparse_u,  # Same spatial locations
            v=sparse_v,
            t_src=torch.full((num_points,), ref_frame),  # Still observe from ref
            t_tgt=torch.full((num_points,), target_frame),
            t_cam=torch.full((num_points,), target_frame),  # Key: target camera
            video=video
        )
        output_tgt = model.decoder(queries_tgt.unsqueeze(0), F)
        xyz_tgt = output_tgt['xyz'].squeeze(0)  # [num_points, 3]

        # STAGE 5: Solve for rigid transformation using Umeyama
        # Aligns xyz_ref to xyz_tgt
        R, t, scale = umeyama_alignment(
            xyz_ref.cpu().numpy(),
            xyz_tgt.cpu().numpy(),
            estimate_scale=False  # Rigid transformation only
        )

    return R, t, scale


def umeyama_alignment(X, Y, estimate_scale=False):
    """
    Umeyama algorithm for rigid/similarity transformation

    Finds R, t, s such that: Y ≈ s * R @ X + t

    Args:
        X: [N, 3] source points
        Y: [N, 3] target points
        estimate_scale: If True, estimate scale; if False, s=1

    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
        s: Scalar scale factor
    """

    # Center the point clouds
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Compute cross-covariance matrix
    H = X_centered.T @ Y_centered  # [3, 3]

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Scale factor
    if estimate_scale:
        var_X = (X_centered ** 2).sum() / len(X)
        s = S.sum() / var_X
    else:
        s = 1.0

    # Translation vector
    t = Y_mean - s * R @ X_mean

    return R, t, s


# Usage example
R, t, scale = estimate_camera_pose(model, video, target_frame=24, ref_frame=0)

# Convert to 4×4 transformation matrix
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t

print(f"Camera moved from frame 0 to frame 24:")
print(f"Rotation:\n{R}")
print(f"Translation: {t}")
```

## Task 4: Novel View Synthesis

Reconstruct scene from a new camera viewpoint.

```python
def synthesize_novel_view(model, video, novel_camera_pose, ref_frame=0):
    """
    Render scene from a novel camera viewpoint

    Args:
        model: D4RT model
        video: Input video [T, 3, H, W]
        novel_camera_pose: [4, 4] transformation matrix for new camera
        ref_frame: Reference frame to use

    Returns:
        novel_view: [H, W, 3] RGB image from novel viewpoint
        depth_map: [H, W] depth map from novel viewpoint
    """

    # This requires:
    # 1. Define dense grid in novel camera view
    # 2. For each pixel, query the model with appropriate t_cam transform
    # 3. Reconstruct RGB and depth
    # (Implementation details depend on rendering approach)

    pass  # Left as exercise - requires camera projection utilities
```

## Batched Inference for Efficiency

```python
def efficient_inference(model, video, queries_batch):
    """
    Process multiple queries in parallel

    Args:
        queries_batch: List of query dicts with keys:
            ['u', 'v', 't_src', 't_tgt', 't_cam']

    Returns:
        results: Batch of 3D positions and visibility
    """

    # Encode once
    F = model.encoder(video.unsqueeze(0))

    # Batch encode all queries
    all_u = torch.cat([q['u'] for q in queries_batch])
    all_v = torch.cat([q['v'] for q in queries_batch])
    # ... (concatenate all query components)

    query_embeddings = model.query_encoder(all_u, all_v, ...)

    # Decode in one forward pass
    output = model.decoder(query_embeddings.unsqueeze(0), F)

    return output
```

## Memory Considerations

For dense tasks (depth reconstruction), memory usage scales with:
- **Encoder**: ~8GB for ViT-g with 48 frames @ 256×256
- **Decoder queries**: ~1MB per 1000 queries

Recommended batch sizes:
- Point tracking (sparse): 1000-5000 points simultaneously
- Depth reconstruction (dense): 4096-8192 pixels per batch
- Camera pose estimation: 256-512 points

## Performance Tips

1. **Cache encoder output**: Encode video once, reuse F for multiple tasks
2. **Batch queries**: Process spatial grids in batches rather than one-by-one
3. **Use mixed precision**: torch.cuda.amp for 2× speedup
4. **Precompute query embeddings**: For fixed query patterns (e.g., depth grid)