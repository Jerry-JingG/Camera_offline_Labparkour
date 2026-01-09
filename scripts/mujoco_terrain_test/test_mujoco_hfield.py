#!/usr/bin/env python3
"""
Test script to visualize the gap terrain in MuJoCo and verify collision.

This script launches the MuJoCo viewer with the generated terrain.
A test ball is included to verify that collisions work correctly with the hfield.

Usage:
    python test_mujoco_hfield.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import os


def main():
    # Load the scene
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(script_dir, "output", "hurdle_terrain_scene.xml")
    
    if not os.path.exists(scene_path):
        print(f"Error: Scene file not found at {scene_path}")
        print("Please run export_gap_hfield.py first to generate the scene.")
        return
    
    print("=" * 50)
    print("MuJoCo HField Collision Test")
    print("=" * 50)
    print(f"\nLoading scene: {scene_path}")
    
    # Load model and create data
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    print(f"\nModel info:")
    print(f"  Number of geoms: {model.ngeom}")
    print(f"  Number of hfields: {model.nhfield}")
    print(f"  Number of bodies: {model.nbody}")
    
    # Print hfield info
    if model.nhfield > 0:
        hfield_id = 0
        nrow = model.hfield_nrow[hfield_id]
        ncol = model.hfield_ncol[hfield_id]
        size = model.hfield_size[hfield_id]
        print(f"\nHField info:")
        print(f"  Dimensions: {nrow} x {ncol}")
        print(f"  Size (half_x, half_y, height, base): {size}")
    
    print("\n" + "=" * 50)
    print("Instructions:")
    print("=" * 50)
    print("1. Watch the test ball fall and interact with the terrain")
    print("2. If the ball falls into gaps, collision is working correctly!")
    print("3. Use mouse to rotate view, scroll to zoom")
    print("4. Press 'R' to reset simulation")
    print("5. Press 'Space' to pause/resume")
    print("6. Close window to exit")
    print()
    
    # Launch viewer
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
