# D4RT Inference

## Quick Start

```bash
python scripts/evaluate.py --config configs/model/vit_b_d4rt.yaml --checkpoint checkpoints/model.pth
```

## How D4RT Inference Works

D4RT inference is a two-stage process:

1. **Encode once**: Process the entire video to create global scene representation F
2. **Query many times**: Ask questions about specific points by decoding queries against F

This design is efficient because the expensive encoding happens only once, then lightweight queries can be processed in batches.

## The Query System

A query is a 5-tuple that asks: "Where is the 3D position of a specific point?"

| Component | Meaning |
|-----------|---------|
| **(u, v)** | Pixel coordinates (normalized 0-1) identifying the point |
| **t_src** | Frame where the point is observed (appearance reference) |
| **t_tgt** | Frame to predict the 3D position at |
| **t_cam** | Camera frame for the coordinate system |

By varying these components, D4RT performs different tasks with the same model.

## Tasks

### 1. Point Tracking

**Goal**: Track a point through time in 3D.

**Query Pattern**:
- Fix (u, v, t_src) = initial point location
- Vary t_tgt = 0, 1, 2, ... T-1
- Set t_cam = t_tgt (get position in each frame's camera)

**Output**: 3D trajectory of the point across all frames.

### 2. Depth Reconstruction

**Goal**: Get depth map for a single frame.

**Query Pattern**:
- Vary (u, v) = grid of all pixels
- Fix t_src = t_tgt = t_cam = target frame

**Output**: Dense depth map where each pixel has a Z value.

**Sub-pixel depth**: Since queries use continuous (u, v) coordinates, you can query at higher resolution than the input video to get sub-pixel accurate depth.

### 3. Dense Tracking

**Goal**: Track all pixels through the video.

**Query Pattern**:
- Vary (u, v) = grid of all pixels
- Fix t_src = source frame
- Vary t_tgt = all frames
- Set t_cam = t_tgt

**Output**: Full scene flow - 3D trajectory for every pixel.

### 4. Point Cloud Generation

**Goal**: Reconstruct 3D point cloud from a frame.

**Query Pattern**:
- Vary (u, v) = grid of all pixels
- Fix t_src = t_tgt = t_cam = target frame

**Output**: 3D point cloud with XYZ positions, plus optional colors and normals.

### 5. Camera Pose Estimation

**Goal**: Estimate relative camera pose between two frames.

**Method**:
1. Query sparse grid of points in reference frame (t_cam = 0)
2. Query same points with t_cam = target frame
3. Use Umeyama algorithm to find rotation R and translation t that aligns the two point sets

**Output**: Relative camera transformation (R, t) from reference to target.

### 6. Long-term Prediction

**Goal**: Predict where points will be beyond the video.

**Query Pattern**:
- Fix (u, v, t_src) = point to track
- Set t_tgt > T (beyond video length)

The model extrapolates motion patterns learned from the video.

## Query Encoding

Each query is converted to a feature vector before decoding:

1. **Spatial encoding**: Fourier features of (u, v) capture fine spatial details
2. **Temporal encoding**: Learned embeddings for t_src, t_tgt, t_cam
3. **Appearance encoding**: CNN processes 9×9 RGB patch at (u, v, t_src)

These are concatenated and projected to create the query embedding.

## Outputs

For each query, the model outputs:

| Output | Description |
|--------|-------------|
| **xyz** | 3D position in t_cam coordinate frame |
| **uv** | 2D image coordinates |
| **visibility** | Whether the point is visible (not occluded) |
| **confidence** | Model's confidence in the prediction |
| **normals** | Surface normal at the point |
| **motion** | Motion displacement |

## Batching for Efficiency

For tasks requiring many queries (e.g., dense depth with 65536 pixels):

1. Encode video once → F
2. Split queries into batches (e.g., 4096 per batch)
3. Process each batch through decoder
4. Concatenate results

This keeps memory usage bounded while leveraging GPU parallelism.

## Tips

1. **Encode once**: Always cache the encoder output F when processing multiple query patterns
2. **Batch queries**: Process 1000-4096 queries per batch for efficiency
3. **Use confidence**: Filter low-confidence predictions for cleaner results
4. **Visibility filtering**: For tracking, use visibility output to detect occlusions
