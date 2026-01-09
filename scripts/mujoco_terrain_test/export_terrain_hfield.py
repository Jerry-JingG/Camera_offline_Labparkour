#!/usr/bin/env python3
"""
Export various parkour terrains as MuJoCo hfield.

This script generates parkour terrain heightfields and exports them
in a format compatible with MuJoCo's hfield geom type.

Usage:
    python export_terrain_hfield.py [terrain_type]
    
    terrain_type: gap, hurdle, flat, step, parkour (default: gap)

Output:
    - {terrain_type}_terrain_heightmap.png: Height field image for MuJoCo
    - {terrain_type}_terrain_scene.xml: MuJoCo scene file with the terrain
"""

import numpy as np
from PIL import Image
import os
import argparse

# Terrain generation parameters (matching IsaacLab config)
HORIZONTAL_SCALE = 0.08  # meters per pixel
VERTICAL_SCALE = 0.005   # height scale
TERRAIN_SIZE = (8.0, 4.0)  # (length_x, width_y) in meters
DIFFICULTY = 0.5  # 0.0 to 1.0

# Common parameters
PLATFORM_LEN = 1.0  # meters
PLATFORM_HEIGHT = 0.0  # meters
Y_RANGE = (-0.4, 0.4)  # lateral offset range
NUM_GOALS = 6


class TerrainConfig:
    """Base configuration for terrains."""
    def __init__(self):
        self.horizontal_scale = HORIZONTAL_SCALE
        self.vertical_scale = VERTICAL_SCALE
        self.size = TERRAIN_SIZE
        self.platform_len = PLATFORM_LEN
        self.platform_height = PLATFORM_HEIGHT
        self.y_range = Y_RANGE
        self.apply_roughness = True
        self.apply_flat = False
        
        # Roughness parameters
        self.noise_range = (0.02, 0.06)
        self.noise_step = 0.005


class GapTerrainConfig(TerrainConfig):
    """Gap terrain configuration."""
    def __init__(self):
        super().__init__()
        self.gap_size_formula = '0.1 + 0.7*difficulty'
        self.gap_depth = (0.2, 1.0)
        self.x_range = (0.8, 1.5)
        self.half_valid_width = (0.6, 1.2)


class HurdleTerrainConfig(TerrainConfig):
    """Hurdle terrain configuration."""
    def __init__(self):
        super().__init__()
        self.x_range = (1.2, 2.2)
        self.half_valid_width = (0.4, 0.8)
        self.stone_len_formula = '0.1 + 0.3 * difficulty'
        self.hurdle_height_range_formula = '0.1+0.1*difficulty, 0.15+0.25*difficulty'


class FlatTerrainConfig(HurdleTerrainConfig):
    """Flat terrain (hurdle with apply_flat=True)."""
    def __init__(self):
        super().__init__()
        self.apply_flat = True


class StepTerrainConfig(TerrainConfig):
    """Step terrain configuration."""
    def __init__(self):
        super().__init__()
        self.x_range = (0.3, 1.5)
        self.half_valid_width = (0.5, 1.0)
        self.step_height_formula = '0.1 + 0.35*difficulty'


def add_roughness(height_field: np.ndarray, cfg: TerrainConfig, difficulty: float) -> np.ndarray:
    """Add random roughness to the terrain."""
    if not cfg.apply_roughness:
        return height_field
    
    noise_amplitude = cfg.noise_range[0] + difficulty * (cfg.noise_range[1] - cfg.noise_range[0])
    noise = np.random.uniform(-noise_amplitude, noise_amplitude, height_field.shape)
    noise_scaled = noise / cfg.vertical_scale
    
    # Only add noise where terrain is not a gap (height >= 0)
    mask = height_field >= 0
    height_field[mask] += noise_scaled[mask]
    
    return height_field


def generate_gap_terrain(difficulty: float, cfg: GapTerrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate a gap terrain height field."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    
    mid_y = length_pixels // 2
    
    gap_size = eval(cfg.gap_size_formula, {"difficulty": difficulty})
    gap_size_pixels = round(gap_size / cfg.horizontal_scale)
    
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale) + gap_size_pixels
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale) + gap_size_pixels
    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)
    
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height
    
    gap_depth = -round(np.random.uniform(cfg.gap_depth[0], cfg.gap_depth[1]) / cfg.vertical_scale)
    half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
    
    goals = np.zeros((NUM_GOALS, 2))
    goals[0] = [platform_len - 1, mid_y]
    
    dis_x = platform_len
    last_dis_x = dis_x
    
    for i in range(NUM_GOALS - 2):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        
        height_field_raw[dis_x - gap_size_pixels // 2 : dis_x + gap_size_pixels // 2, :] = gap_depth
        height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width] = gap_depth
        height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    
    final_dis_x = min(dis_x + np.random.randint(dis_x_min, dis_x_max), width_pixels - 1)
    goals[-1] = [final_dis_x, mid_y]
    
    height_field_raw = add_roughness(height_field_raw, cfg, difficulty)
    return height_field_raw, goals * cfg.horizontal_scale


def generate_hurdle_terrain(difficulty: float, cfg: HurdleTerrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate a hurdle terrain height field."""
    stone_len = eval(cfg.stone_len_formula, {"difficulty": difficulty})
    stone_len = round(stone_len / cfg.horizontal_scale)
    
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    
    mid_y = length_pixels // 2
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)
    
    half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
    hurdle_height_range = eval(cfg.hurdle_height_range_formula, {"difficulty": difficulty})
    hurdle_height_max = round(hurdle_height_range[1] / cfg.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / cfg.vertical_scale)
    
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height
    
    goals = np.zeros((NUM_GOALS, 2))
    goals[0] = [platform_len - 1, mid_y]
    
    dis_x = platform_len
    
    for i in range(NUM_GOALS - 2):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        
        if not cfg.apply_flat:
            # Add hurdle obstacle
            height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2, :] = np.random.randint(hurdle_height_min, hurdle_height_max)
            height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2, :mid_y + rand_y - half_valid_width] = 0
            height_field_raw[dis_x - stone_len // 2:dis_x + stone_len // 2, mid_y + rand_y + half_valid_width:] = 0
        
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    
    final_dis_x = min(dis_x + np.random.randint(dis_x_min, dis_x_max), width_pixels - 1)
    goals[-1] = [final_dis_x, mid_y]
    
    height_field_raw = add_roughness(height_field_raw, cfg, difficulty)
    return height_field_raw, goals * cfg.horizontal_scale


def generate_step_terrain(difficulty: float, cfg: StepTerrainConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate a step terrain height field."""
    step_height = eval(cfg.step_height_formula, {"difficulty": difficulty})
    
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    
    mid_y = length_pixels // 2
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)
    
    step_height_pixels = round(step_height / cfg.vertical_scale)
    half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
    
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height
    
    goals = np.zeros((NUM_GOALS, 2))
    goals[0] = [platform_len - 1, mid_y]
    
    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    
    num_stones = NUM_GOALS - 2
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        
        # Go up first half, then down
        if i < num_stones // 2:
            stair_height += step_height_pixels
        elif i > num_stones // 2:
            stair_height -= step_height_pixels
        
        height_field_raw[dis_x:dis_x + rand_x, :] = stair_height
        dis_x += rand_x
        
        # Create side walls
        height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width] = 0
        height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = 0
        
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    
    final_dis_x = min(dis_x + np.random.randint(dis_x_min, dis_x_max), width_pixels - 1)
    goals[-1] = [final_dis_x, mid_y]
    
    height_field_raw = add_roughness(height_field_raw, cfg, difficulty)
    return height_field_raw, goals * cfg.horizontal_scale


def height_field_to_png(
    height_field: np.ndarray,
    output_path: str,
    vertical_scale: float,
) -> tuple[float, float]:
    """Convert height field to a PNG image for MuJoCo hfield."""
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
    
    print(f"Height field saved to: {output_path}")
    print(f"  Dimensions: {height_field.shape}")
    print(f"  Height range: {min_height:.3f}m to {max_height:.3f}m")
    
    return min_height, max_height


def create_mujoco_scene(
    output_path: str,
    hfield_path: str,
    terrain_size: tuple,
    min_height: float,
    max_height: float,
    terrain_type: str,
):
    """Create a MuJoCo XML scene file with the terrain hfield."""
    half_x = terrain_size[0] / 2
    half_y = terrain_size[1] / 2
    height_range = max_height - min_height
    base_thickness = 0.1
    
    hfield_filename = os.path.basename(hfield_path)
    
    xml_content = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="{terrain_type}_terrain_test">
  <compiler angle="radian" autolimits="true"/>
  
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  
  <asset>
    <!-- Height field terrain -->
    <hfield name="parkour_terrain" 
            file="{hfield_filename}"
            size="{half_x} {half_y} {height_range:.4f} {base_thickness}"/>
    
    <!-- Materials -->
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" 
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" 
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="terrain_mat" rgba="0.6 0.5 0.4 1" specular="0.1"/>
  </asset>
  
  <worldbody>
    <!-- Terrain using hfield -->
    <geom name="terrain" type="hfield" hfield="parkour_terrain" 
          pos="0 0 {-min_height:.4f}" material="terrain_mat"/>
    
    <!-- Test sphere to visualize physics -->
    <body name="test_ball" pos="0.5 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="1 0.3 0.3 1" mass="1"/>
    </body>
    
    <!-- Light -->
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
  </worldbody>
</mujoco>
'''
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"MuJoCo scene saved to: {output_path}")


# Terrain generators mapping
TERRAIN_GENERATORS = {
    "gap": (generate_gap_terrain, GapTerrainConfig),
    "hurdle": (generate_hurdle_terrain, HurdleTerrainConfig),
    "flat": (generate_hurdle_terrain, FlatTerrainConfig),
    "step": (generate_step_terrain, StepTerrainConfig),
}


def main():
    parser = argparse.ArgumentParser(description="Export parkour terrain as MuJoCo hfield")
    parser.add_argument("terrain_type", nargs="?", default="gap",
                        choices=list(TERRAIN_GENERATORS.keys()),
                        help="Type of terrain to generate")
    parser.add_argument("--difficulty", type=float, default=0.5,
                        help="Difficulty level (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Get output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    terrain_type = args.terrain_type
    generator_func, config_class = TERRAIN_GENERATORS[terrain_type]
    cfg = config_class()
    
    print("=" * 50)
    print(f"{terrain_type.upper()} Terrain to MuJoCo HField Export")
    print("=" * 50)
    print(f"\nTerrain parameters:")
    print(f"  Type: {terrain_type}")
    print(f"  Size: {cfg.size[0]}m x {cfg.size[1]}m")
    print(f"  Horizontal scale: {cfg.horizontal_scale}m/pixel")
    print(f"  Vertical scale: {cfg.vertical_scale}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Apply flat: {cfg.apply_flat}")
    print()
    
    # Generate terrain
    print(f"Generating {terrain_type} terrain...")
    height_field, goals = generator_func(args.difficulty, cfg)
    
    print(f"  Height field shape: {height_field.shape}")
    print(f"  Goals: {len(goals)} positions")
    
    # Export to PNG
    hfield_path = os.path.join(output_dir, f"{terrain_type}_terrain_heightmap.png")
    min_height, max_height = height_field_to_png(
        height_field,
        hfield_path,
        cfg.vertical_scale,
    )
    
    # Create MuJoCo scene
    scene_path = os.path.join(output_dir, f"{terrain_type}_terrain_scene.xml")
    create_mujoco_scene(
        scene_path,
        hfield_path,
        cfg.size,
        min_height,
        max_height,
        terrain_type,
    )
    
    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print(f"\nTo test in MuJoCo, run:")
    print(f"  python -m mujoco.viewer {scene_path}")


if __name__ == "__main__":
    main()
