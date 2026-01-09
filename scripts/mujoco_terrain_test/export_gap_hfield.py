#!/usr/bin/env python3
"""
Minimal test script to export gap terrain as MuJoCo hfield.

This script generates a parkour gap terrain heightfield and exports it
in a format compatible with MuJoCo's hfield geom type.

Usage:
    python export_gap_hfield.py

Output:
    - gap_terrain_heightmap.png: Height field image for MuJoCo
    - gap_terrain_scene.xml: MuJoCo scene file with the terrain
"""

import numpy as np
from PIL import Image
import os

# Terrain generation parameters (matching IsaacLab config)
HORIZONTAL_SCALE = 0.08  # meters per pixel
VERTICAL_SCALE = 0.005   # height scale
TERRAIN_SIZE = (8.0, 4.0)  # (length_x, width_y) in meters
DIFFICULTY = 0.5  # 0.0 to 1.0

# Gap terrain specific parameters
GAP_SIZE_FORMULA = '0.1 + 0.7*difficulty'
GAP_DEPTH = (0.2, 1.0)  # min, max depth
PLATFORM_LEN = 1.0  # meters
PLATFORM_HEIGHT = 0.0  # meters
X_RANGE = (0.8, 1.5)  # distance between gaps
Y_RANGE = (-0.4, 0.4)  # lateral offset range
HALF_VALID_WIDTH = (0.6, 1.2)  # valid path width range

NUM_GOALS = 6


def generate_gap_terrain(
    difficulty: float,
    terrain_size: tuple,
    horizontal_scale: float,
    vertical_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a gap terrain height field.
    
    Adapted from parkour_isaaclab/terrains/extreme_parkour/extreme_parkour_terrians.py
    
    Returns:
        height_field: 2D numpy array of heights (in vertical_scale units)
        goals: Goal positions for visualization
    """
    width_pixels = int(terrain_size[0] / horizontal_scale)
    length_pixels = int(terrain_size[1] / horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    
    mid_y = length_pixels // 2
    
    # Calculate gap size based on difficulty
    gap_size = eval(GAP_SIZE_FORMULA, {"difficulty": difficulty})
    gap_size_pixels = round(gap_size / horizontal_scale)
    
    dis_x_min = round(X_RANGE[0] / horizontal_scale) + gap_size_pixels
    dis_x_max = round(X_RANGE[1] / horizontal_scale) + gap_size_pixels
    
    dis_y_min = round(Y_RANGE[0] / horizontal_scale)
    dis_y_max = round(Y_RANGE[1] / horizontal_scale)
    
    platform_len = round(PLATFORM_LEN / horizontal_scale)
    platform_height = round(PLATFORM_HEIGHT / vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height
    
    # Random gap depth
    gap_depth = -round(np.random.uniform(GAP_DEPTH[0], GAP_DEPTH[1]) / vertical_scale)
    half_valid_width = round(np.random.uniform(HALF_VALID_WIDTH[0], HALF_VALID_WIDTH[1]) / horizontal_scale)
    
    goals = np.zeros((NUM_GOALS, 2))
    goals[0] = [platform_len - 1, mid_y]
    
    dis_x = platform_len
    last_dis_x = dis_x
    
    for i in range(NUM_GOALS - 2):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        
        # Create gap
        height_field_raw[dis_x - gap_size_pixels // 2 : dis_x + gap_size_pixels // 2, :] = gap_depth
        
        # Create side walls (areas where robot would fall)
        height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width] = gap_depth
        height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = gap_depth
        
        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    
    # Final goal
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    if final_dis_x > width_pixels:
        final_dis_x = width_pixels - 1
    goals[-1] = [final_dis_x, mid_y]
    
    return height_field_raw, goals * horizontal_scale


def height_field_to_png(
    height_field: np.ndarray,
    output_path: str,
    vertical_scale: float,
) -> tuple[float, float]:
    """
    Convert height field to a PNG image for MuJoCo hfield.
    
    MuJoCo hfield uses grayscale images where:
    - Black (0) = lowest point
    - White (255) = highest point
    
    Returns:
        (min_height, max_height) in meters
    """
    # Convert to actual heights in meters
    heights_meters = height_field * vertical_scale
    
    min_height = heights_meters.min()
    max_height = heights_meters.max()
    height_range = max_height - min_height
    
    if height_range == 0:
        height_range = 1.0  # Avoid division by zero
    
    # Normalize to 0-255 range
    normalized = (heights_meters - min_height) / height_range * 255
    normalized = normalized.astype(np.uint8)
    
    # MuJoCo expects (nrow, ncol) where x is along rows
    # Transpose to match MuJoCo's convention
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
):
    """
    Create a MuJoCo XML scene file with the terrain hfield.
    """
    half_x = terrain_size[0] / 2
    half_y = terrain_size[1] / 2
    height_range = max_height - min_height
    base_thickness = 0.1  # Thickness below the terrain
    
    # Get relative path to hfield image
    hfield_filename = os.path.basename(hfield_path)
    
    xml_content = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="gap_terrain_test">
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
    <material name="terrain_mat" rgba="0.5 0.5 0.5 1" specular="0.1"/>
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


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Gap Terrain to MuJoCo HField Export")
    print("=" * 50)
    print(f"\nTerrain parameters:")
    print(f"  Size: {TERRAIN_SIZE[0]}m x {TERRAIN_SIZE[1]}m")
    print(f"  Horizontal scale: {HORIZONTAL_SCALE}m/pixel")
    print(f"  Vertical scale: {VERTICAL_SCALE}")
    print(f"  Difficulty: {DIFFICULTY}")
    print()
    
    # Generate terrain
    print("Generating gap terrain...")
    height_field, goals = generate_gap_terrain(
        difficulty=DIFFICULTY,
        terrain_size=TERRAIN_SIZE,
        horizontal_scale=HORIZONTAL_SCALE,
        vertical_scale=VERTICAL_SCALE,
    )
    
    print(f"  Height field shape: {height_field.shape}")
    print(f"  Goals: {len(goals)} positions")
    
    # Export to PNG
    hfield_path = os.path.join(output_dir, "gap_terrain_heightmap.png")
    min_height, max_height = height_field_to_png(
        height_field,
        hfield_path,
        VERTICAL_SCALE,
    )
    
    # Create MuJoCo scene
    scene_path = os.path.join(output_dir, "gap_terrain_scene.xml")
    create_mujoco_scene(
        scene_path,
        hfield_path,
        TERRAIN_SIZE,
        min_height,
        max_height,
    )
    
    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print(f"\nTo test in MuJoCo, run:")
    print(f"  python -m mujoco.viewer {scene_path}")
    print()
    print("Or in Python:")
    print("  import mujoco")
    print("  import mujoco.viewer")
    print(f'  model = mujoco.MjModel.from_xml_path("{scene_path}")')
    print("  mujoco.viewer.launch(model)")


if __name__ == "__main__":
    main()
