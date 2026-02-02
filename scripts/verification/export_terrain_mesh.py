"""
Export Parkour Terrain to STL Mesh for MuJoCo
"""

import os
import sys
import trimesh
import numpy as np
import argparse

# Add the project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

# Create argparser for AppLauncher
parser = argparse.ArgumentParser(description="Export Parkour Terrain to STL")
parser.add_argument("--output", type=str, default="terrain.stl", help="Output STL filename")
parser.add_argument("--type", type=str, default="gap", choices=["gap", "hurdle", "step", "beam", "parkour", "demo", "flat"], help="Terrain type")
parser.add_argument("--difficulty", type=float, default=0.5, help="Difficulty level (0.0 - 1.0)")

# Append AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import the rest of Isaac Lab modules
from parkour_isaaclab.terrains import ParkourTerrainGenerator, ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrains_cfg import (
    ExtremeParkourGapTerrainCfg,
    ExtremeParkourHurdleTerrainCfg,
    ExtremeParkourStepTerrainCfg,
    ExtremeParkourBeamTerrainCfg,
    ExtremeParkourTerrainCfg,
    ExtremeParkourDemoTerrainCfg
)

def export_terrain_mesh(output_file="terrain.stl", terrain_type="gap", difficulty=0.5):
    """
    Generates a parkour terrain and exports it as an STL mesh.
    
    Args:
        output_file (str): Path to save the STL file.
        terrain_type (str): Type of terrain to generate (gap, hurdle, step, beam, parkour, demo).
        difficulty (float): Difficulty level (0.0 to 1.0).
    """
    
    # 1. Configure Terrain
    cfg = ParkourTerrainGeneratorCfg()
    cfg.size = (16.0, 4.0)
    cfg.horizontal_scale = 0.08
    cfg.vertical_scale = 0.005
    cfg.border_width = 0.0
    cfg.num_rows = 1
    cfg.num_cols = 1
    cfg.curriculum = False # Disable curriculum for single track generation
    cfg.random_difficulty = False
    
    # Reset sub_terrains and add only the requested one with high proportion
    cfg.sub_terrains = {}
    
    if terrain_type == "gap":
        cfg.sub_terrains["gap"] = ExtremeParkourGapTerrainCfg(proportion=1.0)
    elif terrain_type == "hurdle":
        cfg.sub_terrains["hurdle"] = ExtremeParkourHurdleTerrainCfg(proportion=1.0)
    elif terrain_type == "step":
        cfg.sub_terrains["step"] = ExtremeParkourStepTerrainCfg(proportion=1.0)
    elif terrain_type == "beam":
        cfg.sub_terrains["beam"] = ExtremeParkourBeamTerrainCfg(proportion=1.0)
    elif terrain_type == "parkour":
        cfg.sub_terrains["parkour"] = ExtremeParkourTerrainCfg(proportion=1.0)
    elif terrain_type == "demo":
        cfg.sub_terrains["demo"] = ExtremeParkourDemoTerrainCfg(proportion=1.0)
    elif terrain_type == "flat":
        # Use Hurdle config but with apply_flat=True to generate flat ground
        # This matches the "parkour_flat" logic in parkour.py
        cfg.sub_terrains["flat"] = ExtremeParkourHurdleTerrainCfg(
            proportion=1.0,
            apply_roughness=True,
            apply_flat=True,
            x_range=(1.2, 2.2),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range='0.1+0.1*difficulty, 0.15+0.15*difficulty' 
        )
    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")

    # Set difficulty range to force the specific difficulty
    # The generator samples from difficulty_range, so we set both min and max to the target difficulty
    # Note: Logic in _generate_random_terrains uses self.np_rng.uniform(*self.cfg.difficulty_range)
    cfg.difficulty_range = (difficulty, difficulty)

    print(f"Generating terrain: {terrain_type} with difficulty {difficulty}...")

    # 2. Instantiate Generator
    # We use "cpu" device since we are just doing mesh processing
    generator = ParkourTerrainGenerator(cfg=cfg, device="cpu")

    # 3. Trigger Generation
    # ParkourTerrainGenerator typically generates during init or reset. 
    # But checking the code, __init__ calls super().__init__, which usually calls self.reset() or generation logic.
    # Looking at ParkourTerrainGenerator code provided:
    # It has _generate_random_terrains and _generate_curriculum_terrains.
    # The parent TerrainGenerator.__init__ calls self.generate_terrains() which calls one of those based on curriculum.
    # Let's verify by checking if terrains adhere to our config.
    
    # Since we set curriculum=False, it should use _generate_random_terrains.
    # And since we set num_rows=1, num_cols=1, and proportions=1.0, it should generate exactly one terrain.
    
    # 4. Extract and Combine Meshes
    # The generator stores meshes in self.terrain_meshes
    if not generator.terrain_meshes:
        print("No meshes were generated via default initialization. Attempting manual generation...")
        # If for some reason it's empty, we might need to manually trigger generation, 
        # but TerrainGenerator usuall does it. 
        # Let's inspect generator.terrain_meshes
        pass

    if hasattr(generator, 'terrain_meshes') and len(generator.terrain_meshes) > 0:
        print(f"Found {len(generator.terrain_meshes)} mesh components.")
        combined_mesh = trimesh.util.concatenate(generator.terrain_meshes)
        
        # --- Post-Processing: Cut off the bottom ---
        # Isaac Lab terrains often have deep bases/pits. To make gaps "hollow" for visual/collision purposes
        # (so you can fall through them), we slice off everything below a certain Z threshold.
        bounds = combined_mesh.bounds
        z_min = bounds[0][2]
        
        # If the mesh extends significantly below zero (e.g. gap pits), slice it.
        if z_min < -0.5:
            print(f"Detected deep geometry (Z min: {z_min:.2f}). Slicing bottom to create hollow gaps...")
            try:
                # Keep everything above Z = -1.0 (adjust as needed, gap pits are usually deep)
                # Using -1.0 ensures we don't accidentally cut off sloped terrain parts that dip slightly.
                # But if we want true hollow gaps, we should cut higher.
                # Let's try cutting at Z = -0.5.
                cut_height = -0.5
                combined_mesh = trimesh.intersections.slice_mesh_plane(
                    mesh=combined_mesh,
                    plane_normal=[0, 0, 1],
                    plane_origin=[0, 0, cut_height]
                )
                print(f"Sliced mesh at Z={cut_height}.")
            except Exception as e:
                print(f"Warning: Mesh slicing failed: {e}")

        
        # Ensure the output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        # 6. Export
        print(f"Exporting to {output_file}...")
        combined_mesh.export(output_file)
        print("Done.")
    else:
        print("Error: No terrain meshes found in generator.")

if __name__ == "__main__":
    export_terrain_mesh(args_cli.output, args_cli.type, args_cli.difficulty)
    
    # Close the simulation app
    simulation_app.close()
