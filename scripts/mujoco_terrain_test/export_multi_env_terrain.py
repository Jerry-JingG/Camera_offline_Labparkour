#!/usr/bin/env python3
"""
Export multi-env parkour terrain for MuJoCo based on IsaacLab rules.

Generates a grid of terrain blocks following IsaacLab's proportion-based distribution:
- parkour_gap: 20%
- parkour_hurdle: 20%  
- parkour_flat: 20%
- parkour_step: 20%
- parkour: 20% (inclined stones)

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

# Terrain type proportions (matching EXTREME_PARKOUR_TERRAINS_CFG)
TERRAIN_PROPORTIONS = {
    "parkour_gap": 0.2,
    "parkour_hurdle": 0.2,
    "parkour_flat": 0.2,
    "parkour_step": 0.2,
    "parkour": 0.2,
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
        
        # Parkour (inclined stones) params
        self.parkour_pit_depth = (0.2, 1.0)
        self.parkour_stone_width = 1.0
        self.parkour_last_stone_len = 1.6


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


def generate_parkour_terrain(cfg: TerrainConfig, difficulty: float) -> np.ndarray:
    """
    Generate parkour terrain with inclined stones - CENTER ALIGNED.
    Based on parkour_terrain from extreme_parkour_terrians.py
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    pit_depth = -round(np.random.uniform(*cfg.parkour_pit_depth) / cfg.vertical_scale)
    height_field = np.full((width_pixels, length_pixels), pit_depth, dtype=np.float32)
    
    mid_y = length_pixels // 2
    
    # Stone parameters
    stone_len_min = 0.9 - 0.3 * difficulty
    stone_len_max = 1.0 - 0.2 * difficulty
    stone_len = np.random.uniform(stone_len_min, stone_len_max)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len_pixels = round(stone_len / cfg.horizontal_scale)
    
    stone_width = round(cfg.parkour_stone_width / cfg.horizontal_scale)
    last_stone_len = round(cfg.parkour_last_stone_len / cfg.horizontal_scale)
    
    incline_height = 0.25 * difficulty
    incline_height_pixels = round(incline_height / cfg.vertical_scale)
    
    x_range_min = -0.1
    x_range_max = 0.1 + 0.3 * difficulty
    
    # Platform
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field[0:platform_len, :] = platform_height
    
    dis_x_min = stone_len_pixels + round(x_range_min / cfg.horizontal_scale)
    dis_x_max = stone_len_pixels + round(x_range_max / cfg.horizontal_scale)
    
    dis_x = platform_len - np.random.randint(max(1, dis_x_min), max(2, dis_x_max)) + stone_len_pixels // 2
    dis_z = 0
    
    num_stones = NUM_GOALS - 2
    for i in range(num_stones):
        dis_x += np.random.randint(max(1, dis_x_min), max(2, dis_x_max))
        
        if dis_x >= width_pixels - last_stone_len:
            break
        
        # Inclined stone centered at mid_y (no Y offset)
        if i == num_stones - 1:
            # Last stone (larger)
            heights = np.tile(np.linspace(-incline_height_pixels, incline_height_pixels, stone_width), 
                            (last_stone_len, 1))
            x_start = max(0, dis_x - last_stone_len // 2)
            x_end = min(width_pixels, dis_x + last_stone_len // 2)
            y_start = max(0, mid_y - stone_width // 2)
            y_end = min(length_pixels, mid_y + stone_width // 2)
            height_field[x_start:x_end, y_start:y_end] = heights[:x_end-x_start, :y_end-y_start].astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height_pixels, incline_height_pixels, stone_width), 
                            (stone_len_pixels, 1))
            x_start = max(0, dis_x - stone_len_pixels // 2)
            x_end = min(width_pixels, dis_x + stone_len_pixels // 2)
            y_start = max(0, mid_y - stone_width // 2)
            y_end = min(length_pixels, mid_y + stone_width // 2)
            height_field[x_start:x_end, y_start:y_end] = heights[:x_end-x_start, :y_end-y_start].astype(int) + dis_z
    
    # End platform
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 / cfg.horizontal_scale)
    if final_platform_start < width_pixels:
        height_field[final_platform_start:, :] = platform_height
    
    height_field = add_roughness(height_field, cfg, difficulty)
    return height_field


# ============================================================================
# Terrain type generator mapping
# ============================================================================
TERRAIN_GENERATORS = {
    "parkour_gap": lambda cfg, d: generate_gap_terrain(cfg, d),
    "parkour_hurdle": lambda cfg, d: generate_hurdle_terrain(cfg, d, apply_flat=False),
    "parkour_flat": lambda cfg, d: generate_hurdle_terrain(cfg, d, apply_flat=True),
    "parkour_step": lambda cfg, d: generate_step_terrain(cfg, d),
    "parkour": lambda cfg, d: generate_parkour_terrain(cfg, d),
}


def generate_multi_env_terrain(
    cfg: TerrainConfig,
    num_rows: int,
    num_cols: int,
    difficulty_range: tuple = (0.0, 1.0),
) -> tuple[np.ndarray, list]:
    """
    Generate a grid of terrain blocks following IsaacLab's curriculum layout.
    
    Returns:
        combined_terrain: 2D height field array
        terrain_info: List of (row, col, terrain_type, difficulty) for each block
    """
    width_pixels_per_env = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels_per_env = int(cfg.size[1] / cfg.horizontal_scale)
    
    total_width = width_pixels_per_env * num_rows
    total_length = length_pixels_per_env * num_cols
    
    combined = np.zeros((total_width, total_length), dtype=np.float32)
    terrain_info = []
    
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
            
            # Generate terrain block
            block = generator(cfg, difficulty)
            
            # Place in combined terrain
            row_start = row * width_pixels_per_env
            row_end = (row + 1) * width_pixels_per_env
            col_start = col * length_pixels_per_env
            col_end = (col + 1) * length_pixels_per_env
            
            combined[row_start:row_end, col_start:col_end] = block
            terrain_info.append((row, col, terrain_type, difficulty))
    
    return combined, terrain_info


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
):
    """Create MuJoCo XML scene with the combined terrain."""
    half_x = total_width / 2
    half_y = total_length / 2
    height_range = max_height - min_height
    base_thickness = 0.1
    
    hfield_filename = os.path.basename(hfield_path)
    
    xml_content = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="multi_env_parkour_terrain">
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
    <material name="ball_mat" rgba="0.9 0.2 0.2 1"/>
  </asset>
  
  <worldbody>
    <!-- Terrain -->
    <geom name="terrain" type="hfield" hfield="parkour_terrain" 
          pos="0 0 {-min_height:.4f}" material="terrain_mat"/>
    
    <!-- Test balls - one per row at the start -->'''
    
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
    parser = argparse.ArgumentParser(description="Export multi-env parkour terrain for MuJoCo")
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
    print("Multi-Env Parkour Terrain Export (IsaacLab-style)")
    print("=" * 70)
    print(f"\nGrid: {args.num_rows} rows × {args.num_cols} cols = {args.num_rows * args.num_cols} envs")
    print(f"Each env: {cfg.size[0]}m × {cfg.size[1]}m")
    print(f"Total terrain: {cfg.size[0] * args.num_rows}m × {cfg.size[1] * args.num_cols}m")
    print(f"Difficulty range: {args.difficulty_min} - {args.difficulty_max}")
    print(f"Obstacles: CENTER-ALIGNED (no Y-axis random offset)")
    
    # Generate terrain
    combined, terrain_info = generate_multi_env_terrain(
        cfg,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        difficulty_range=(args.difficulty_min, args.difficulty_max),
    )
    
    total_width = cfg.size[0] * args.num_rows
    total_length = cfg.size[1] * args.num_cols
    
    print(f"\nGenerated terrain shape: {combined.shape}")
    print(f"Height range: {combined.min() * cfg.vertical_scale:.3f}m to {combined.max() * cfg.vertical_scale:.3f}m")
    
    # Export
    hfield_path = os.path.join(output_dir, "multi_env_terrain_heightmap.png")
    min_height, max_height = height_field_to_png(combined, hfield_path, cfg.vertical_scale)
    
    scene_path = os.path.join(output_dir, "multi_env_terrain_scene.xml")
    create_mujoco_scene(
        scene_path, hfield_path,
        total_width, total_length,
        min_height, max_height,
        args.num_rows, args.num_cols
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
