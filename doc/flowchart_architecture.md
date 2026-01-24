# D4RT Architecture Flowchart

## Overall Model Architecture

```mermaid
flowchart TD
    Video[Input Video<br/>B × T × 3 × H × W<br/>4 × 48 × 3 × 256 × 256] --> Encoder
    Queries[Query 5-tuples<br/>u, v, t_src, t_tgt, t_cam<br/>B × N queries] --> QueryEncoder

    subgraph Encoder[ENCODER: Spatio-Temporal ViT]
        Patch[3D Patch Embedding<br/>Conv3d 2×16×16<br/>6144 patches]
        Patch --> PosEmb[3D Positional Encoding<br/>Learned 6144×hidden_dim]
        PosEmb --> Blocks[Transformer Blocks<br/>ViT-B: 12 layers<br/>ViT-L: 24 layers<br/>ViT-g: 40 layers]
        Blocks --> EncOut[Output: F<br/>B × 6144 × hidden_dim]
    end

    subgraph QueryEncoder[QUERY ENCODER]
        Fourier[Fourier Encoding<br/>u, v → 40 dims<br/>10 frequencies]
        TempEmb[Temporal Embeddings<br/>t_src, t_tgt, t_cam<br/>→ 3 × 256 dims]
        PatchCNN[RGB Patch CNN<br/>9×9 patch<br/>→ 256 dims]

        Fourier --> Concat[Concatenate<br/>1064 dims total]
        TempEmb --> Concat
        PatchCNN --> Concat

        Concat --> Project[Linear Projection<br/>1064 → 512]
        Project --> QOut[Query Embeddings<br/>B × N × 512]
    end

    subgraph Decoder[DECODER: Cross-Attention Transformer]
        QOut --> Layer1[Layer 1<br/>Self-Attn + Cross-Attn + FFN]
        Layer1 --> Layer2[Layer 2]
        Layer2 --> Layer3[...]
        Layer3 --> Layer8[Layer 8]
        Layer8 --> Head[Output Head<br/>Linear 512 → 3]
        Head --> DecOut[Predictions<br/>xyz: B × N × 3<br/>vis: B × N × 1]
    end

    EncOut -.Cross-Attention.-> Layer1
    EncOut -.Cross-Attention.-> Layer2
    EncOut -.Cross-Attention.-> Layer3
    EncOut -.Cross-Attention.-> Layer8

    style Encoder fill:#e1f5ff
    style QueryEncoder fill:#fff5e1
    style Decoder fill:#ffe1e1
```

## Encoder Detail: Spatio-Temporal ViT

```mermaid
flowchart TD
    Input[Video Tensor<br/>B × T × 3 × H × W<br/>48 × 3 × 256 × 256] --> Permute[Permute<br/>B × 3 × T × H × W<br/>For Conv3D]

    Permute --> Conv3D[3D Convolution<br/>Kernel: 2×16×16<br/>Stride: 2×16×16<br/>Out channels: hidden_dim]

    Conv3D --> Flatten[Flatten Patches<br/>B × hidden_dim × T' × H' × W'<br/>→ B × 6144 × hidden_dim<br/>T'=24, H'=16, W'=16]

    Flatten --> AddPos[Add Positional Embedding<br/>Learned 3D position codes<br/>B × 6144 × hidden_dim]

    AddPos --> Block1[Transformer Block 1<br/>Self-Attention<br/>LayerNorm<br/>FFN]

    Block1 --> Block2[Transformer Block 2]
    Block2 --> BlockN[...<br/>N blocks total]
    BlockN --> BlockLast[Transformer Block N<br/>ViT-B: N=12<br/>ViT-L: N=24<br/>ViT-g: N=40]

    BlockLast --> Output[Global Features F<br/>B × 6144 × hidden_dim<br/>ViT-B: 768<br/>ViT-L: 1024<br/>ViT-g: 1408]

    style Conv3D fill:#e1f5ff
    style Block1 fill:#ffe1f5
    style BlockLast fill:#ffe1f5
```

## Transformer Block Detail

```mermaid
flowchart TD
    Input[Input<br/>B × N × hidden_dim] --> Norm1[LayerNorm]

    Norm1 --> SelfAttn[Multi-Head Self-Attention<br/>Q = K = V = input<br/>Attention h₁,...,hₙ heads<br/>Concat and project]

    SelfAttn --> Residual1[Residual Connection<br/>output = input + attn_out]

    Residual1 --> Norm2[LayerNorm]

    Norm2 --> FFN[Feed-Forward Network<br/>Linear hidden_dim → 4×hidden_dim<br/>GELU activation<br/>Linear 4×hidden_dim → hidden_dim]

    FFN --> Residual2[Residual Connection<br/>output = input + ffn_out]

    Residual2 --> Output[Output<br/>B × N × hidden_dim]

    style SelfAttn fill:#e1f5ff
    style FFN fill:#ffe1e1
```

## Decoder Detail: Cross-Attention Layer

```mermaid
flowchart TD
    QueryEmb[Query Embeddings<br/>B × N × 512] --> Norm1[LayerNorm]
    EncFeatures[Encoder Features F<br/>B × 6144 × enc_dim] --> Project[Project to 512 dims<br/>If needed]

    Norm1 --> SelfAttn[Self-Attention<br/>Q = K = V = queries<br/>Queries attend to each other]

    SelfAttn --> Residual1[Residual + queries]

    Residual1 --> Norm2[LayerNorm]
    Project --> CrossAttn

    Norm2 --> CrossAttn[Cross-Attention<br/>Q = queries<br/>K = V = encoder features<br/>Queries attend to F]

    CrossAttn --> Residual2[Residual + queries]

    Residual2 --> Norm3[LayerNorm]

    Norm3 --> FFN[Feed-Forward Network<br/>512 → 2048 → 512<br/>GELU activation]

    FFN --> Residual3[Residual + queries]

    Residual3 --> Output[Output<br/>B × N × 512]

    style SelfAttn fill:#e1f5ff
    style CrossAttn fill:#ffe1e1
    style FFN fill:#fff5e1
```

## Query Encoding Detail

```mermaid
flowchart TD
    U[u coordinate<br/>0,1] --> FourierU
    V[v coordinate<br/>0,1] --> FourierV

    subgraph FourierEncoding[Fourier Positional Encoding]
        FourierU[For u:<br/>sin u·2^k·π, cos u·2^k·π<br/>k = 0,1,...,9<br/>20 dims]
        FourierV[For v:<br/>sin v·2^k·π, cos v·2^k·π<br/>k = 0,1,...,9<br/>20 dims]
    end

    FourierU --> ConcatFourier[Concatenate<br/>40 dims total]
    FourierV --> ConcatFourier

    TSrc[t_src<br/>frame index] --> EmbTableSrc[Embedding Table<br/>max_frames × 256<br/>Learned]
    TTgt[t_tgt<br/>frame index] --> EmbTableTgt[Embedding Table<br/>max_frames × 256<br/>Learned]
    TCam[t_cam<br/>frame index] --> EmbTableCam[Embedding Table<br/>max_frames × 256<br/>Learned]

    EmbTableSrc --> TSrcEmb[t_src_emb<br/>256 dims]
    EmbTableTgt --> TTgtEmb[t_tgt_emb<br/>256 dims]
    EmbTableCam --> TCamEmb[t_cam_emb<br/>256 dims]

    Video[Video<br/>T × 3 × H × W] --> Extract[Extract 9×9 Patch<br/>at u,v,t_src<br/>3 × 9 × 9]

    Extract --> CNN[Patch CNN<br/>Conv 3→64 + ReLU<br/>Conv 64→128 + ReLU<br/>AdaptiveAvgPool<br/>Linear 128→256]

    CNN --> PatchFeats[Patch Features<br/>256 dims]

    ConcatFourier --> FinalConcat[Concatenate All<br/>40 + 256×3 + 256<br/>= 1064 dims]
    TSrcEmb --> FinalConcat
    TTgtEmb --> FinalConcat
    TCamEmb --> FinalConcat
    PatchFeats --> FinalConcat

    FinalConcat --> LinearProj[Linear Projection<br/>1064 → 512]

    LinearProj --> Output[Query Embedding<br/>512 dims<br/>Ready for decoder]

    style FourierEncoding fill:#e1f5ff
    style CNN fill:#ffe1e1
    style LinearProj fill:#fff5e1
```

## Model Size Comparison

```mermaid
flowchart LR
    subgraph ViT-B[ViT-B Small]
        EB[Encoder<br/>12 layers<br/>768 dim<br/>86M params]
        DB[Decoder<br/>8 layers<br/>512 dim<br/>144M params]
    end

    subgraph ViT-L[ViT-L Medium]
        EL[Encoder<br/>24 layers<br/>1024 dim<br/>307M params]
        DL[Decoder<br/>8 layers<br/>512 dim<br/>144M params]
    end

    subgraph ViT-g[ViT-g Large]
        EG[Encoder<br/>40 layers<br/>1408 dim<br/>1B params]
        DG[Decoder<br/>8 layers<br/>512 dim<br/>144M params]
    end

    Total1[Total: ~230M]
    Total2[Total: ~451M]
    Total3[Total: ~1.144B]

    ViT-B --> Total1
    ViT-L --> Total2
    ViT-g --> Total3

    style ViT-B fill:#e1f5ff
    style ViT-L fill:#ffe1f5
    style ViT-g fill:#fff5e1
```
