# TRAINING LOOP
for batch in dataloader:
    # 1. Input: Video frames [B, T, C, H, W]
    video_frames = batch['video']
    
    # 2. ENCODE: Generate Global Scene Representation (F)
    # Uses a spatio-temporal ViT (e.g., ViT-g) with self-attention
    F = encoder(video_frames) 
    
    # 3. SAMPLE QUERIES: Pick N points for supervision
    # A query q = (u, v, t_src, t_tgt, t_cam)
    # (u, v) = 2D pixel coord, t_src = source frame
    # t_tgt = timestamp of 3D point, t_cam = camera reference frame
    queries = sample_random_queries(batch, num_queries=N)
    
    # 4. DECODE: Query the Latent Representation
    # Each query independently cross-attends to F
    # Local RGB patches (e.g., 9x9) are often concatenated here
    predicted_3D_points = decoder(queries, F)
    
    # 5. LOSS COMPUTATION: Multi-task supervision
    # L1 Loss on normalized 3D positions
    loss_3d = compute_l1_loss(predicted_3D_points, batch['gt_3d'])
    
    # Auxiliary losses (2D projection, normals, motion, visibility)
    loss_aux = compute_auxiliary_losses(predicted_3D_points, batch)
    
    total_loss = w1 * loss_3d + w2 * loss_aux
    
    # 6. OPTIMIZE
    total_loss.backward()
    optimizer.step()