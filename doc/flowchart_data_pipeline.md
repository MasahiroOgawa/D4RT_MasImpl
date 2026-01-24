# D4RT Data Pipeline Flowchart

## Overall Data Pipeline

```mermaid
flowchart TD
    Start([Raw Dataset<br/>Videos + Annotations]) --> Load[Load Video Sequence<br/>Original resolution<br/>Variable FPS]

    Load --> Preprocess

    subgraph Preprocess[Preprocessing]
        Resize[Resize Frames<br/>Target: 256 × 256<br/>Maintain aspect ratio]
        Sample[Sample Frames<br/>Target: 48 frames<br/>Uniform temporal sampling]
        Normalize[Normalize RGB<br/>Mean: 0.485, 0.456, 0.406<br/>Std: 0.229, 0.224, 0.225]
    end

    Preprocess --> LoadAnnotations[Load Annotations<br/>Depth maps<br/>Camera params<br/>Point tracks<br/>Visibility masks]

    LoadAnnotations --> Augment

    subgraph Augment[Data Augmentation Training only]
        Crop[Random Crop<br/>Spatial jittering]
        Flip[Horizontal Flip<br/>50% probability]
        Color[Color Jitter<br/>Brightness, contrast<br/>Saturation, hue]
    end

    Augment --> CreateBatch[Create Batch Dict<br/>video, depth, cameras<br/>tracks_3d, visibility]

    CreateBatch --> SampleQueries[Sample Training Queries<br/>See query sampling flowchart]

    SampleQueries --> Return[Return Batch<br/>Ready for training]

    style Preprocess fill:#e1f5ff
    style Augment fill:#ffe1e1
```

## Dataset-Specific Pipelines

```mermaid
flowchart TD
    Start{Dataset Type} --> Kubric[Kubric<br/>Synthetic]
    Start --> PointOdyssey[PointOdyssey<br/>Real videos]
    Start --> ScanNet[ScanNet<br/>RGB-D]
    Start --> Co3D[Co3Dv2<br/>Multi-view objects]
    Start --> Waymo[Waymo<br/>Autonomous driving]

    Kubric --> KubricPipe[Generate on-the-fly<br/>Full GT: depth, motion<br/>Perfect camera params<br/>Configurable scenes]

    PointOdyssey --> OdysseyPipe[Load video + tracks<br/>Dense point annotations<br/>Occlusion labels<br/>Long-term tracking]

    ScanNet --> ScanNetPipe[Load RGB + depth<br/>Camera poses from SLAM<br/>Indoor scenes<br/>Reconstruct 3D]

    Co3D --> Co3DPipe[Load multi-view images<br/>Camera poses<br/>Object masks<br/>Sparse point clouds]

    Waymo --> WaymoPipe[Load LiDAR + RGB<br/>Project 3D to 2D<br/>Outdoor driving scenes<br/>Moving objects]

    KubricPipe --> Common[Common Pipeline<br/>Resize, sample, normalize]
    OdysseyPipe --> Common
    ScanNetPipe --> Common
    Co3DPipe --> Common
    WaymoPipe --> Common

    style Kubric fill:#90EE90
    style PointOdyssey fill:#87CEEB
    style ScanNet fill:#FFB6C1
    style Co3D fill:#DDA0DD
    style Waymo fill:#F0E68C
```

## Query Sampling Strategy

```mermaid
flowchart TD
    Start[Batch Data<br/>B videos<br/>Each with GT] --> CheckSplit{Training or<br/>Validation?}

    CheckSplit -->|Training| TrainSample[Sample 2048 queries<br/>Strategic distribution]
    CheckSplit -->|Validation| ValSample[Sample 1024 queries<br/>Fixed pattern]

    TrainSample --> Visible

    subgraph StrategicSampling[Strategic Sampling Training]
        Visible[Visible Points: 50%<br/>1024 queries<br/>Has GT 3D position<br/>Sample from visible_mask]

        Occluded[Occluded Points: 25%<br/>512 queries<br/>Behind objects<br/>Sample from occluded_mask]

        Random[Random Points: 25%<br/>512 queries<br/>Uniform grid<br/>Background coverage]
    end

    Visible --> Temporal
    Occluded --> Temporal
    Random --> Temporal

    Temporal[Sample Temporal Coords<br/>t_src ~ Uniform 0, T-1<br/>t_tgt ~ Uniform 0, T-1<br/>t_cam ~ Uniform 0, T-1]

    ValSample --> ValGrid[Regular Grid<br/>Spatial: 32×32<br/>Temporal: All frames<br/>Consistent evaluation]

    ValGrid --> Temporal

    Temporal --> ExtractPatch[Extract RGB Patches<br/>9×9 around u,v<br/>From frame t_src]

    ExtractPatch --> GetGT[Get Ground Truth<br/>Project 3D to camera<br/>Compute visibility<br/>Get normals, motion]

    GetGT --> Bundle[Bundle Query Data<br/>Coordinates: u,v,t<br/>Patches: RGB<br/>Ground truth: xyz,vis]

    Bundle --> Return[Return Query Batch<br/>Ready for model]

    style StrategicSampling fill:#fff5e1
    style Temporal fill:#e1f5ff
```

## Ground Truth Extraction

```mermaid
flowchart TD
    Start[Query u,v,t_src,t_tgt,t_cam] --> Check3D{Has 3D<br/>Ground Truth?}

    Check3D -->|Yes: Kubric, ScanNet| Direct[Direct 3D Coordinates<br/>xyz_world from depth map<br/>or point cloud]

    Check3D -->|Yes: PointOdyssey| Track[From Point Tracks<br/>Track ID → 3D position<br/>Linear interpolation]

    Check3D -->|Partial: Co3D, Waymo| Reconstruct[Reconstruct 3D<br/>Triangulation<br/>or LiDAR projection]

    Direct --> Transform
    Track --> Transform
    Reconstruct --> Transform

    Transform[Transform to Camera Frame<br/>xyz_cam = R @ xyz_world + t<br/>Using camera extrinsics t_cam]

    Transform --> Project[Project to 2D<br/>uv_proj = K @ xyz_cam<br/>For 2D loss]

    Project --> Visibility[Compute Visibility<br/>Is point occluded?<br/>Depth test or annotation]

    Visibility --> Normal[Compute Normal<br/>From depth gradients<br/>or point cloud]

    Normal --> Motion[Compute Motion<br/>xyz t - xyz t-1<br/>Temporal derivative]

    Motion --> Bundle[Bundle Ground Truth<br/>xyz: 3D position<br/>uv: 2D projection<br/>vis: visibility flag<br/>normal: surface normal<br/>motion: 3D velocity]

    Bundle --> Return[Return GT Dict]

    style Transform fill:#e1f5ff
    style Visibility fill:#ffe1e1
```

## Data Augmentation

```mermaid
flowchart TD
    Start[Original Batch] --> SpatialAug

    subgraph SpatialAug[Spatial Augmentation]
        Crop[Random Crop<br/>Crop size: 224×224<br/>From 256×256<br/>Update camera intrinsics]

        Flip[Random Horizontal Flip<br/>Probability: 0.5<br/>Flip coordinates<br/>Update camera]

        Resize[Resize Back<br/>To 256×256<br/>For consistent input]
    end

    SpatialAug --> ColorAug

    subgraph ColorAug[Color Augmentation]
        Brightness[Brightness<br/>Factor: ±0.2]
        Contrast[Contrast<br/>Factor: ±0.2]
        Saturation[Saturation<br/>Factor: ±0.2]
        Hue[Hue<br/>Factor: ±0.1]
    end

    ColorAug --> TemporalAug

    subgraph TemporalAug[Temporal Augmentation]
        FrameDrop[Frame Dropout<br/>Randomly drop 10% frames<br/>Simulate missing data]

        TimeReverse[Time Reversal<br/>Probability: 0.1<br/>Reverse frame order<br/>Test temporal invariance]
    end

    TemporalAug --> UpdateCoords[Update Coordinates<br/>Adjust u,v for crop/flip<br/>Adjust t for frame drop<br/>Maintain consistency]

    UpdateCoords --> Validate[Validate Batch<br/>Check shapes<br/>Check ranges<br/>Ensure GT aligned]

    Validate --> Return[Augmented Batch]

    style SpatialAug fill:#e1f5ff
    style ColorAug fill:#ffe1e1
    style TemporalAug fill:#fff5e1
```

## Dataloader Configuration

```mermaid
flowchart LR
    subgraph Datasets[Dataset Mix]
        D1[Kubric<br/>15%]
        D2[PointOdyssey<br/>10%]
        D3[ScanNet<br/>15%]
        D4[Co3Dv2<br/>10%]
        D5[Waymo<br/>15%]
        D6[Others<br/>35%]
    end

    D1 --> Sampler[Weighted Sampler<br/>Sample per dataset weight<br/>Ensure balanced training]
    D2 --> Sampler
    D3 --> Sampler
    D4 --> Sampler
    D5 --> Sampler
    D6 --> Sampler

    Sampler --> Loader[DataLoader<br/>Batch size: 4 per GPU<br/>Num workers: 4<br/>Pin memory: True<br/>Prefetch: 2]

    Loader --> Batch[Training Batch<br/>Ready for GPU]

    style Datasets fill:#f0f0f0
    style Sampler fill:#e1f5ff
    style Loader fill:#ffe1e1
```

## Batch Structure

```mermaid
flowchart TD
    Batch[Training Batch Dict] --> Video[video<br/>B, T, 3, H, W<br/>torch.float32<br/>Normalized RGB]

    Batch --> Queries[queries<br/>u, v: B, N<br/>t_src, t_tgt, t_cam: B, N<br/>patches: B, N, 3, 9, 9]

    Batch --> Targets[targets<br/>xyz: B, N, 3<br/>uv: B, N, 2<br/>visibility: B, N<br/>normals: B, N, 3<br/>motion: B, N, 3]

    Batch --> Cameras[cameras<br/>intrinsics: B, T, 3, 3<br/>extrinsics: B, T, 4, 4<br/>K and T matrices]

    Batch --> Meta[metadata<br/>scene_bounds: B, 6<br/>dataset_name: List str<br/>frame_ids: B, T]

    style Batch fill:#f0f0f0
    style Video fill:#e1f5ff
    style Queries fill:#ffe1e1
    style Targets fill:#fff5e1
    style Cameras fill:#f5e1ff
```
