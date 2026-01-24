# INFERENCE PIPELINE
def infer_4d_scene(video_clip):
    # STAGE 1: Feature Extraction (Once per video)
    # Produce the latent representation F
    latent_representation = encoder(video_clip)
    
    # STAGE 2: Task-Specific Querying (On-demand)
    
    # Example A: POINT TRACKING
    # Keep (u, v, t_src) fixed, vary t_tgt from 1 to T
    tracking_queries = create_queries(u=fixed, v=fixed, t_src=0, t_tgt=range(T), t_cam=range(T))
    point_trajectory = decoder(tracking_queries, latent_representation)
    
    # Example B: DEPTH MAP (Frame t)
    # Iterate through all pixels (u, v) for a specific target time
    depth_queries = create_queries(u=grid(W), v=grid(H), t_src=t, t_tgt=t, t_cam=t)
    depth_map = decoder(depth_queries, latent_representation)
    
    # Example C: CAMERA POSE (Extrinsics)
    # Solve for R, t using predicted 3D points relative to t_cam
    pose_queries = create_queries(u=sparse_grid, v=sparse_grid, t_src=0, t_tgt=t, t_cam=0)
    relative_3d_points = decoder(pose_queries, latent_representation)
    camera_pose = umeyama_algorithm(relative_3d_points) 
    
    return point_trajectory, depth_map, camera_pose