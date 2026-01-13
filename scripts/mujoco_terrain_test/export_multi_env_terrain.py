#!/usr/bin/env python3
"""
Export multi-env parkour terrain for MuJoCo based on IsaacLab rules.

Generates a grid of terrain blocks following IsaacLab's proportion-based distribution:
- parkour_gap: 20%
- parkour_hurdle: 20%  
- parkour_flat: 20%
- parkour_step: 20%
- parkour_beam: 20% (suspended overhead beams - robot must duck under)

All obstacles are CENTER-ALIGNED (no Y-axis random offset).

Usage:
    python export_multi_env_terrain.py [--num_rows 3] [--num_cols 5] [--difficulty 0.5]
"""

import numpy as np
from PIL import Image
import os
import argparse

# ============================================================================
# Terrain parameters matching IsaacLab config
# ============================================================================
HORIZONTAL_SCALE = 0.08  # meters per pixel
VERTICAL_SCALE = 0.005   # height scale
TERRAIN_SIZE = (16.0, 4.0)  # (length_x, width_y) per env block
NUM_GOALS = 10  # Number of goals/obstacles per terrain (matching IsaacLab)

# Terrain type proportions (matching EXTREME_PARKOUR_TERRAINS_CFG with beam instead of slope)
# NOTE: Order determines column layout. Beam is placed in middle (column 2) for better visibility
TERRAIN_PROPORTIONS = {
    "parkour_gap": 0.2,
    "parkour_hurdle": 0.2,
    "parkour_beam": 0.2,  # Beam in middle column (was flat)
    "parkour_step": 0.2,
    "parkour_flat": 0.2,  # Flat moved to last column (was beam)
}


class TerrainConfig:
    """Configuration matching IsaacLab's extreme_parkour_terrains_cfg."""
    def __init__(self):
        self.horizontal_scale = HORIZONTAL_SCALE
        self.vertical_scale = VERTICAL_SCALE
        self.size = TERRAIN_SIZE
        
        # Common parameters
        self.platform_len = 1.0
        self.platform_height = 0.0
        self.apply_roughness = True
        self.noise_range = (0.02, 0.06)
        
        # Gap terrain params
        self.gap_x_range = (0.8, 1.5)
        self.gap_half_valid_width = (0.6, 1.2)
        self.gap_depth = (0.2, 1.0)
        
        # Hurdle terrain params
        self.hurdle_x_range = (1.2, 2.2)
        self.hurdle_half_valid_width = (0.4, 0.8)
        
        # Step terrain params
        self.step_x_range = (0.3, 1.5)
        self.step_half_valid_width = (0.5, 1.0)
        
        # Beam terrain params (replaces parkour slope)
        self.beam_x_range = (1.2, 2.2)
        self.beam_half_valid_width = (0.5, 0.8)
        self.beam_depth = 0.2  # Vertical thickness of beam


def add_roughness(height_field: np.ndarray, cfg: TerrainConfig, difficulty: float) -> np.ndarray:
    """Add random roughness to walkable surfaces (height >= 0)."""
    if not cfg.apply_roughness:
        return height_field
    
    noise_amplitude = cfg.noise_range[0] + difficulty * (cfg.noise_range[1] - cfg.noise_range[0])
    noise = np.random.uniform(-noise_amplitude, noise_amplitude, height_field.shape)
    noise_scaled = noise / cfg.vertical_scale
    
    mask = height_field >= 0
    height_field[mask] += noise_scaled[mask]
    return height_field


# ============================================================================
# Terrain Generators (matching IsaacLab's extreme_parkour_terrians.py)
# ============================================================================

def generate_gap_terrain(cfg: TerrainConfig, difficulty: float) -> np.ndarray:
    """
    Generate gap terrain - CENTER ALIGNED.
    Based on parkour_gap_terrain from extreme_parkour_terrians.py
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field = np.zeros((width_pixels, length_pixels))
    
    mid_y = length_pixels // 2
    
    # Gap parameters from config
    gap_size = 0.1 + 0.7 * difficulty
    gap_size_pixels = round(gap_size / cfg.horizontal_scale)
    gap_depth = -round(np.random.uniform(*cfg.gap_depth) / cfg.vertical_scale)
    
    half_valid_width = round(np.random.uniform(*cfg.gap_half_valid_width) / cfg.horizontal_scale)
    
    dis_x_min = round(cfg.gap_x_range[0] / cfg.horizontal_scale) + gap_size_pixels
    dis_x_max = round(cfg.gap_x_range[1] / cfg.horizontal_scale) + gap_size_pixels
    
    # Platform
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field[0:platform_len, :] = platform_height
    
    dis_x = platform_len
    last_dis_x = dis_x
    
    # Generate gaps (NUM_GOALS - 2 gaps between start and end)
    num_gaps = NUM_GOALS - 2
    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        
        if dis_x >= width_pixels - platform_len:
            break
        
        # Gap at center (no Y offset)
        height_field[dis_x - gap_size_pixels // 2 : dis_x + gap_size_pixels // 2, :] = gap_depth
        
        # Side walls (areas outside valid path, centered)
        height_field[last_dis_x:dis_x, :mid_y - half_valid_width] = gap_depth
        height_field[last_dis_x:dis_x, mid_y + half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
    
    height_field = add_roughness(height_field, cfg, difficulty)
    return height_field


def generate_hurdle_terrain(cfg: TerrainConfig, difficulty: float, apply_flat: bool = False) -> np.ndarray:
    """
    Generate hurdle terrain - CENTER ALIGNED.
    Based on parkour_hurdle_terrain from extreme_parkour_terrians.py
    If apply_flat=True, generates flat terrain (no hurdles).
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field = np.zeros((width_pixels, length_pixels))
    
    mid_y = length_pixels // 2
    
    stone_len = 0.1 + 0.3 * difficulty
    stone_len_pixels = round(stone_len / cfg.horizontal_scale)
    
    hurdle_height_min = 0.1 + 0.1 * difficulty
    hurdle_height_max = 0.15 + 0.25 * difficulty
    
    half_valid_width = round(np.random.uniform(*cfg.hurdle_half_valid_width) / cfg.horizontal_scale)
    
    dis_x_min = round(cfg.hurdle_x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.hurdle_x_range[1] / cfg.horizontal_scale)
    
    # Platform
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field[0:platform_len, :] = platform_height
    
    dis_x = platform_len
    
    # Generate hurdles
    num_hurdles = NUM_GOALS - 2
    for i in range(num_hurdles):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        
        if dis_x >= width_pixels - platform_len:
            break
        
        if not apply_flat:
            hurdle_height = np.random.uniform(hurdle_height_min, hurdle_height_max)
            hurdle_height_pixels = round(hurdle_height / cfg.vertical_scale)
            
            # Hurdle centered at mid_y
            height_field[dis_x - stone_len_pixels // 2 : dis_x + stone_len_pixels // 2,
                        mid_y - half_valid_width : mid_y + half_valid_width] = hurdle_height_pixels
    
    height_field = add_roughness(height_field, cfg, difficulty)
    return height_field


def generate_step_terrain(cfg: TerrainConfig, difficulty: float) -> np.ndarray:
    """
    Generate step terrain - CENTER ALIGNED stairs going up then down.
    Based on parkour_step_terrain from extreme_parkour_terrians.py
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field = np.zeros((width_pixels, length_pixels))
    
    mid_y = length_pixels // 2
    
    step_height = 0.1 + 0.35 * difficulty
    step_height_pixels = round(step_height / cfg.vertical_scale)
    
    half_valid_width = round(np.random.uniform(*cfg.step_half_valid_width) / cfg.horizontal_scale)
    
    dis_x_min = round(cfg.step_x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.step_x_range[1] / cfg.horizontal_scale)
    
    # Platform
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field[0:platform_len, :] = platform_height
    
    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    
    num_steps = NUM_GOALS - 2
    for i in range(num_steps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        
        if i < num_steps // 2:
            stair_height += step_height_pixels
        elif i > num_steps // 2:
            stair_height -= step_height_pixels
        
        if dis_x + rand_x >= width_pixels - platform_len:
            break
            
        height_field[dis_x:dis_x + rand_x, mid_y - half_valid_width:mid_y + half_valid_width] = stair_height
        dis_x += rand_x
        
        # Side walls at 0 height
        height_field[last_dis_x:dis_x, :mid_y - half_valid_width] = 0
        height_field[last_dis_x:dis_x, mid_y + half_valid_width:] = 0
        
        last_dis_x = dis_x
    
    height_field = add_roughness(height_field, cfg, difficulty)
    return height_field


def generate_beam_terrain(cfg: TerrainConfig, difficulty: float) -> tuple:
    """
    Generate beam terrain - CENTER ALIGNED suspended overhead beams.
    
    Beams are represented as elevated obstacles that the robot must pass under.
    Returns both the heightfield and beam box positions.
    
    Returns:
        height_field: Ground heightfield
        beam_boxes: List of (x, y, z_relative, size_x, size_y, size_z) for suspended beams
                   z_relative is height above the flat ground (in heightfield meters, i.e. 0 = flat ground)
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field = np.zeros((width_pixels, length_pixels))
    
    mid_y = length_pixels // 2
    
    # Beam parameters
    beam_length = 0.3 + 0.3 * difficulty
    beam_length_pixels = round(beam_length / cfg.horizontal_scale)
    
    # Beam clearance (height above ground) - robot must duck under
    # Increased height (0.5-0.7m) to ensure beams are clearly visible above terrain
    beam_height_min = 0.5 - 0.1 * difficulty
    beam_height_max = 0.7 - 0.1 * difficulty
    
    half_valid_width = round(np.random.uniform(*cfg.beam_half_valid_width) / cfg.horizontal_scale)
    
    dis_x_min = round(cfg.beam_x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.beam_x_range[1] / cfg.horizontal_scale)
    
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    
    # Flat ground for beam terrain (platform_height = 0 in heightfield units)
    height_field[:] = platform_height
    
    # Store the ground height in meters (this is the reference for beam placement)
    ground_height_m = platform_height * cfg.vertical_scale  # = 0.0m for flat ground
    
    dis_x = platform_len
    beam_boxes = []
    
    num_beams = NUM_GOALS - 2
    for i in range(num_beams):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        
        if dis_x >= width_pixels - platform_len:
            break
        
        # Calculate beam parameters
        clearance = np.random.uniform(beam_height_min, beam_height_max)
        
        # Beam dimensions in meters
        beam_x_m = dis_x * cfg.horizontal_scale
        beam_y_m = mid_y * cfg.horizontal_scale
        # beam_z_m is relative to flat ground (ground_height_m = 0)
        # clearance is the distance from ground to bottom of beam
        # beam center is at ground_height + clearance + half beam thickness
        beam_z_relative = ground_height_m + clearance + cfg.beam_depth / 2
        beam_size_x = beam_length_pixels * cfg.horizontal_scale
        beam_size_y = (half_valid_width * 2 + round(0.4 / cfg.horizontal_scale)) * cfg.horizontal_scale
        beam_size_z = cfg.beam_depth
        
        beam_boxes.append((beam_x_m, beam_y_m, beam_z_relative, beam_size_x, beam_size_y, beam_size_z))
    
    height_field = add_roughness(height_field, cfg, difficulty)
    return height_field, beam_boxes


# ============================================================================
# Terrain type generator mapping
# ============================================================================
TERRAIN_GENERATORS = {
    "parkour_gap": lambda cfg, d: (generate_gap_terrain(cfg, d), []),
    "parkour_hurdle": lambda cfg, d: (generate_hurdle_terrain(cfg, d, apply_flat=False), []),
    "parkour_flat": lambda cfg, d: (generate_hurdle_terrain(cfg, d, apply_flat=True), []),
    "parkour_step": lambda cfg, d: (generate_step_terrain(cfg, d), []),
    "parkour_beam": lambda cfg, d: generate_beam_terrain(cfg, d),
}


def generate_multi_env_terrain(
    cfg: TerrainConfig,
    num_rows: int,
    num_cols: int,
    difficulty_range: tuple = (0.0, 1.0),
) -> tuple:
    """
    Generate a grid of terrain blocks following IsaacLab's curriculum layout.
    
    Returns:
        combined_terrain: 2D height field array
        terrain_info: List of (row, col, terrain_type, difficulty) for each block
        all_beams: List of beam box tuples for MuJoCo scene
    """
    width_pixels_per_env = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels_per_env = int(cfg.size[1] / cfg.horizontal_scale)
    
    total_width = width_pixels_per_env * num_rows
    total_length = length_pixels_per_env * num_cols
    
    combined = np.zeros((total_width, total_length), dtype=np.float32)
    terrain_info = []
    all_beams = []  # Store all beam boxes
    
    # Calculate column assignments based on proportions
    terrain_types = list(TERRAIN_PROPORTIONS.keys())
    proportions = np.array(list(TERRAIN_PROPORTIONS.values()))
    proportions /= proportions.sum()
    cumsum = np.cumsum(proportions)
    
    col_terrain_types = []
    for col in range(num_cols):
        col_frac = (col + 0.001) / num_cols
        terrain_idx = np.searchsorted(cumsum, col_frac)
        terrain_idx = min(terrain_idx, len(terrain_types) - 1)
        col_terrain_types.append(terrain_types[terrain_idx])
    
    print(f"\nTerrain distribution:")
    for ttype in terrain_types:
        count = col_terrain_types.count(ttype)
        print(f"  {ttype}: {count}/{num_cols} columns ({count/num_cols*100:.1f}%)")
    
    # Generate each terrain block
    for row in range(num_rows):
        for col in range(num_cols):
            # Curriculum: difficulty increases with row
            difficulty = difficulty_range[0] + (difficulty_range[1] - difficulty_range[0]) * row / max(1, num_rows - 1)
            
            terrain_type = col_terrain_types[col]
            generator = TERRAIN_GENERATORS[terrain_type]
            
            # Generate terrain block (returns tuple: heightfield, beam_boxes)
            block, beams = generator(cfg, difficulty)
            
            # Place in combined terrain
            row_start = row * width_pixels_per_env
            row_end = (row + 1) * width_pixels_per_env
            col_start = col * length_pixels_per_env
            col_end = (col + 1) * length_pixels_per_env
            
            combined[row_start:row_end, col_start:col_end] = block
            terrain_info.append((row, col, terrain_type, difficulty))
            
            # Handle beam boxes - offset to global position
            for (bx, by, bz, sx, sy, sz) in beams:
                global_x = bx + row * cfg.size[0]
                global_y = by + col * cfg.size[1]
                all_beams.append((global_x, global_y, bz, sx, sy, sz))
    
    return combined, terrain_info, all_beams


def height_field_to_png(height_field: np.ndarray, output_path: str, vertical_scale: float) -> tuple:
    """Convert height field to PNG for MuJoCo hfield."""
    heights_meters = height_field * vertical_scale
    
    min_height = heights_meters.min()
    max_height = heights_meters.max()
    height_range = max_height - min_height
    
    if height_range == 0:
        height_range = 1.0
    
    normalized = (heights_meters - min_height) / height_range * 255
    normalized = normalized.astype(np.uint8)
    
    img = Image.fromarray(normalized.T)
    img.save(output_path)
    
    return min_height, max_height


def create_mujoco_scene(
    output_path: str,
    hfield_path: str,
    total_width: float,
    total_length: float,
    min_height: float,
    max_height: float,
    num_rows: int,
    num_cols: int,
    beams: list = None,
):
    """
    Create MuJoCo XML scene with the combined terrain and beam obstacles.
    
    MuJoCo hfield coordinate system:
    - hfield size = (half_x, half_y, height_range, base)
    - PNG normalized to [0,1], where 0=black (min height), 1=white (max height)
    - World z = geom_pos_z + base + normalized * height_range
    
    For flat ground (heightfield value = 0m):
    - normalized_flat = (0 - min_height) / height_range
    - flat_world_z = geom_pos_z + base + normalized_flat * height_range
                   = (-min_height) + base + (0 - min_height) / height_range * height_range
                   = (-min_height) + base + (0 - min_height)
                   = -min_height + base - min_height
                   = base - 2*min_height  (if min_height < 0)
    
    Beams z_relative is relative to flat ground (0m in heightfield), so:
    - beam_world_z = flat_world_z + z_relative
    """
    half_x = total_width / 2
    half_y = total_length / 2
    height_range = max_height - min_height
    base_thickness = 0.1
    
    # Calculate flat ground world z coordinate
    # Flat ground has heightfield value = 0m
    # In PNG: normalized_flat = (0 - min_height) / height_range
    # In MuJoCo: flat_world_z = geom_pos_z + base + normalized_flat * height_range
    geom_pos_z = -min_height
    normalized_flat = (0 - min_height) / height_range if height_range > 0 else 0
    flat_world_z = geom_pos_z + base_thickness + normalized_flat * height_range
    
    print(f"\nMuJoCo coordinate calculation:")
    print(f"  min_height: {min_height:.4f}m, max_height: {max_height:.4f}m")
    print(f"  height_range: {height_range:.4f}m")
    print(f"  geom_pos_z: {geom_pos_z:.4f}m")
    print(f"  normalized_flat: {normalized_flat:.4f}")
    print(f"  flat_world_z: {flat_world_z:.4f}m (this is where z=0 heightfield maps to)")
    
    hfield_filename = os.path.basename(hfield_path)
    
    xml_content = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="multi_env_parkour_terrain_with_beams">
  <compiler angle="radian" autolimits="true"/>
  
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="150" elevation="-20"/>
    <quality shadowsize="4096"/>
  </visual>
  
  <asset>
    <!-- Multi-env terrain height field -->
    <hfield name="parkour_terrain" 
            file="{hfield_filename}"
            size="{half_x} {half_y} {height_range:.4f} {base_thickness}"/>
    
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="3072"/>
    <material name="terrain_mat" rgba="0.55 0.5 0.45 1" specular="0.1"/>
    <material name="beam_mat" rgba="0.6 0.3 0.2 1" specular="0.3"/>
    <material name="ball_mat" rgba="0.9 0.2 0.2 1"/>
  </asset>
  
  <worldbody>
    <!-- Terrain -->
    <geom name="terrain" type="hfield" hfield="parkour_terrain" 
          pos="0 0 {geom_pos_z:.4f}" material="terrain_mat"/>'''
    
    # Add beam obstacles
    if beams:
        for i, (bx, by, bz_relative, sx, sy, sz) in enumerate(beams):
            # Convert to MuJoCo world coordinates (centered at origin)
            mx = bx - half_x
            my = by - half_y
            # bz_relative is relative to flat ground (0m in heightfield)
            # Convert to world z by adding flat_world_z
            mz = flat_world_z + bz_relative
            xml_content += f'''
    <!-- Beam {i} -->
    <geom name="beam_{i}" type="box" pos="{mx:.3f} {my:.3f} {mz:.3f}" 
          size="{sx/2:.3f} {sy/2:.3f} {sz/2:.3f}" material="beam_mat"/>'''
    
    # Add test balls for each row
    env_width = TERRAIN_SIZE[0]
    for row in range(num_rows):
        ball_x = -half_x + 1.0 + row * env_width
        ball_y = 0
        xml_content += f'''
    <body name="test_ball_{row}" pos="{ball_x:.2f} {ball_y:.2f} 0.5">
      <freejoint/>
      <geom type="sphere" size="0.08" material="ball_mat" mass="0.5"/>
    </body>'''
    
    xml_content += f'''
    
    <!-- Lighting -->
    <light pos="0 0 15" dir="0 0 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
  </worldbody>
</mujoco>
'''
    
    with open(output_path, 'w') as f:
        f.write(xml_content)


def main():
    parser = argparse.ArgumentParser(description="Export multi-env parkour terrain for MuJoCo (with beam)")
    parser.add_argument("--num_rows", type=int, default=3, help="Number of difficulty levels (rows)")
    parser.add_argument("--num_cols", type=int, default=5, help="Number of terrain columns")
    parser.add_argument("--difficulty_min", type=float, default=0.3, help="Minimum difficulty")
    parser.add_argument("--difficulty_max", type=float, default=0.8, help="Maximum difficulty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    cfg = TerrainConfig()
    
    print("=" * 70)
    print("Multi-Env Parkour Terrain Export (with BEAM terrain)")
    print("=" * 70)
    print(f"\nGrid: {args.num_rows} rows × {args.num_cols} cols = {args.num_rows * args.num_cols} envs")
    print(f"Each env: {cfg.size[0]}m × {cfg.size[1]}m")
    print(f"Total terrain: {cfg.size[0] * args.num_rows}m × {cfg.size[1] * args.num_cols}m")
    print(f"Difficulty range: {args.difficulty_min} - {args.difficulty_max}")
    print(f"Obstacles: CENTER-ALIGNED (no Y-axis random offset)")
    
    # Generate terrain
    combined, terrain_info, beams = generate_multi_env_terrain(
        cfg,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        difficulty_range=(args.difficulty_min, args.difficulty_max),
    )
    
    total_width = cfg.size[0] * args.num_rows
    total_length = cfg.size[1] * args.num_cols
    
    print(f"\nGenerated terrain shape: {combined.shape}")
    print(f"Height range: {combined.min() * cfg.vertical_scale:.3f}m to {combined.max() * cfg.vertical_scale:.3f}m")
    print(f"Number of beam obstacles: {len(beams)}")
    
    # Export
    hfield_path = os.path.join(output_dir, "multi_env_terrain_heightmap.png")
    min_height, max_height = height_field_to_png(combined, hfield_path, cfg.vertical_scale)
    
    scene_path = os.path.join(output_dir, "multi_env_terrain_scene.xml")
    create_mujoco_scene(
        scene_path, hfield_path,
        total_width, total_length,
        min_height, max_height,
        args.num_rows, args.num_cols,
        beams
    )
    
    print(f"\nExported files:")
    print(f"  Heightmap: {hfield_path}")
    print(f"  Scene: {scene_path}")
    
    print("\n" + "=" * 70)
    print("To test in MuJoCo:")
    print(f"  python -m mujoco.viewer {scene_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
