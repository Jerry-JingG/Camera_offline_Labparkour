
@parkour_field_to_mesh
def parkour_beam_terrain(
    difficulty: float,
    cfg: extreme_parkour_terrains_cfg.ExtremeParkourBeamTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[trimesh.Trimesh]]:
    """
    Terrain with suspended beams (overhead obstacles) that the robot must pass under.
    Returns:
        height_field_raw: The ground height field (mostly flat).
        goals: Goal positions.
        goal_heights: Height of goals.
        extra_meshes: List of suspended beam meshes.
    """
    beam_length_expr = eval(cfg.beam_length, {"difficulty": difficulty})
    beam_length_pixels = round(beam_length_expr / cfg.horizontal_scale)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale) # Y direction
    height_field_raw = np.zeros((width_pixels, length_pixels))

    mid_y = length_pixels // 2
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)

    half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
    
    # Beam parameters
    beam_height_range = eval(cfg.beam_height_range, {"difficulty": difficulty})
    beam_depth = cfg.beam_depth # Vertical thickness (m)

    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    
    # Initialize flat ground at platform_height
    height_field_raw[:] = platform_height
    
    dis_x = platform_len
    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height
    goals[0] = [platform_len - 1, mid_y]
    
    extra_meshes = []
    
    # Helper to create beam mesh
    def create_beam(center_x_idx, center_y_idx, length_idx, width_idx, clearance_m):
        # Convert indices to meters (relative to terrain origin 0,0)
        # However, trimesh coordinates in 'extra_meshes' need to be relative to the sub-terrain's origin.
        # But wait, parkour_field_to_mesh wrapper does offset:
        # "mesh.apply_transform(transform)" -> Moves the combined mesh to global position.
        # "z_gen" is returned as local height field.
        # So extra_meshes should be defined in the local frame of the sub-terrain.
        # Local frame: x [0, size_x], y [0, size_y], z relative to 0.
        
        # Dimensions in meters
        size_x = length_idx * cfg.horizontal_scale
        size_y = width_idx * cfg.horizontal_scale
        size_z = beam_depth
        
        # Center position in meters
        pos_x = center_x_idx * cfg.horizontal_scale
        pos_y = center_y_idx * cfg.horizontal_scale
        # Z position: platform_height (m) + clearance + half_thickness
        platform_h_m = platform_height * cfg.vertical_scale
        pos_z = platform_h_m + clearance_m + size_z / 2.0
        
        transform = np.eye(4)
        transform[:3, 3] = [pos_x, pos_y, pos_z]
        
        box = trimesh.creation.box(extents=[size_x, size_y, size_z], transform=transform)
        return box

    for i in range(num_goals - 2):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        
        # Determine beam parameters for this step
        # Randomize clearance within range
        clearance = np.random.uniform(beam_height_range[0], beam_height_range[1])
        
        # Beam width (y-axis): Spans slightly more than valid path or full width?
        # Let's make it wide enough to block the path but maybe not the whole terrain to save polys?
        # Or just make it span the whole "half_valid_width" area times 2 plus some margin.
        beam_width_pixels = (half_valid_width * 2) + round(0.4 / cfg.horizontal_scale) # +40cm margin
        
        # Create beam mesh
        # Center of beam in X is dis_x
        # Center of beam in Y is mid_y
        beam_mesh = create_beam(dis_x, mid_y, beam_length_pixels, beam_width_pixels, clearance)
        extra_meshes.append(beam_mesh)
        
        goals[i+1] = [dis_x + rand_x//2, mid_y] # Goal is past the beam?
        # Typically goals are placed *on* the obstacles for stepping stones.
        # For passing *under*, the goal should be after the obstacle.
        # Here rand_x is the gap to the next obstacle.
        # In hurdle terrain: goals[i+1] = [dis_x-rand_x//2, mid_y + rand_y] (Middle of the gap before the hurdle?)
        # Let's look at hurdle logic:
        # dis_x += rand_x (increments position)
        # height_field_raw[dis_x-stone_len//2:dis_x+stone_len//2] = hurdle (places hurdle at new dis_x)
        # goals[i+1] = [dis_x-rand_x//2, mid_y] -> This places goal halfway between previous obstacle and current obstacle.
        
        # So for beams:
        # 1. Increment dis_x to new beam position.
        # 2. Place beam at dis_x.
        # 3. Goal should probably be AT the beam (under it) or slightly after?
        # If we use the same logic as hurdle: goal is placed before the obstacle.
        # This drives the robot TOWARDS the obstacle.
        # So: goals[i+1] = [dis_x - rand_x//2, mid_y] matches hurdle logic.
        
        goals[i+1] = [dis_x - rand_x//2, mid_y]

    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > width_pixels:
        final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    
    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    
    if cfg.apply_roughness:
        # Make the ground rough, but keep the area under beams relatively flat?
        # The original random_uniform_terrain adds noise everywhere.
        # Maybe acceptable for advanced curriculum.
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale, extra_meshes
