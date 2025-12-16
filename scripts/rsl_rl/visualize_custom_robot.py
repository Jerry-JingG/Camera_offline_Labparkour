# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize a custom robot USD model in the parkour terrain environment.

This script loads the Go2 teacher parkour terrain environment and imports a custom
USD robot model for visualization purposes. No training is performed.

Usage:
    python visualize_custom_robot.py --num_envs 1 --usd_path /path/to/your/robot.usd
    
Example with wlg_usd:
    python visualize_custom_robot.py --num_envs 1 --usd_path /home/droplet/IsaacLab/Camera_offline_Labparkour/wlg_usd/wlg.usd
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Make sure project-root packages are importable when running via absolute script path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

# Add argparse arguments
parser = argparse.ArgumentParser(description="Visualize custom robot USD in parkour terrain.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--usd_path",
    type=str,
    default="/home/droplet/IsaacLab/Camera_offline_Labparkour/wlg_usd/wlg.usd",
    help="Path to the USD file of the custom robot.",
)
parser.add_argument("--robot_scale", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Scale of the robot (x, y, z).")
parser.add_argument("--robot_pos", type=float, nargs=3, default=[0.0, 0.0, 0.1], help="Position (x, y, z). Z is absolute height, default 0.1m.")
parser.add_argument("--num_robots", type=int, default=1, help="Number of custom robots to spawn around origin.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import torch
import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom

import parkour_tasks  # noqa: F401


def add_custom_robot_to_stage(usd_path: str, position: tuple = (0.0, 0.0, 0.5), scale: tuple = (1.0, 1.0, 1.0), rotation_z: float = 0.0, prim_path: str = "/World/CustomRobot"):
    """Add a custom USD robot model to the stage for visualization.
    
    Args:
        usd_path: Path to the USD file.
        position: Initial position (x, y, z) for the robot.
        scale: Scale factors (x, y, z) for the robot.
        rotation_z: Rotation around Z axis in degrees.
        prim_path: USD prim path for the robot.
    """
    import omni.usd
    from pxr import UsdGeom, Gf, UsdPhysics
    
    # Get the current stage
    stage = omni.usd.get_context().get_stage()
    
    # Create an Xform prim for the robot
    xform = UsdGeom.Xform.Define(stage, prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    
    # Add reference to the USD file
    prim.GetReferences().AddReference(usd_path)
    
    # Set transform using the standard USD transform matrix approach
    try:
        # Clear existing xform ops and set new transform
        xform.ClearXformOpOrder()
        
        # Add transform operations with double precision
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]))
        
        # Add rotation around Z axis
        if rotation_z != 0.0:
            rotate_op = xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
            rotate_op.Set(rotation_z)
        
        scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(scale[0], scale[1], scale[2]))
    except Exception as e:
        print(f"[WARN] Could not set xform ops, using default transform: {e}")
    
    # Disable physics on the custom robot to prevent conflicts with simulation
    _disable_physics_recursively(stage, prim_path)
    
    return prim_path


def _disable_physics_recursively(stage, prim_path: str):
    """Recursively disable all physics on prims under the given path.
    
    This removes articulation roots, rigid bodies, colliders, and joints
    to ensure the USD is purely for visualization.
    """
    from pxr import UsdPhysics, PhysxSchema
    
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    
    # List of physics APIs and schemas to remove
    physics_api_types = [
        UsdPhysics.ArticulationRootAPI,
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.CollisionAPI,
        UsdPhysics.MassAPI,
    ]
    
    # Try to remove PhysX specific schemas if available
    physx_api_types = []
    try:
        physx_api_types = [
            PhysxSchema.PhysxArticulationAPI,
            PhysxSchema.PhysxRigidBodyAPI,
            PhysxSchema.PhysxCollisionAPI,
        ]
    except Exception:
        pass  # PhysxSchema may not be available in all configurations
    
    for descendant in stage.Traverse():
        if not str(descendant.GetPath()).startswith(prim_path):
            continue
            
        # Remove USD Physics APIs
        for api_type in physics_api_types:
            try:
                if descendant.HasAPI(api_type):
                    descendant.RemoveAPI(api_type)
            except Exception:
                pass
        
        # Remove PhysX APIs
        for api_type in physx_api_types:
            try:
                if descendant.HasAPI(api_type):
                    descendant.RemoveAPI(api_type)
            except Exception:
                pass
        
        # Deactivate joint prims
        if descendant.IsA(UsdPhysics.Joint):
            try:
                descendant.SetActive(False)
            except Exception:
                pass


def update_custom_robot_position(prim_path: str, position: tuple, offset: tuple = (1.0, 0.0, 0.0)):
    """Update the position of the custom robot to follow the Go2 robot.
    
    Args:
        prim_path: Path to the custom robot prim.
        position: Position of the Go2 robot (x, y, z).
        offset: Offset from Go2 position (x, y, z) to place the custom robot.
    """
    import omni.usd
    from pxr import UsdGeom, Gf
    
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    
    xform = UsdGeom.Xformable(prim)
    
    # Calculate new position with offset
    new_pos = Gf.Vec3d(
        position[0] + offset[0],
        position[1] + offset[1],
        position[2] + offset[2]
    )
    
    # Find and update the translate op
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(new_pos)
            return
    
    # If no translate op exists, add one
    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(new_pos)


def get_ground_height_at_position(x: float, y: float, robot_height_offset: float = 0.0):
    """Get the ground height at a specific position using raycast.
    
    Args:
        x: X coordinate.
        y: Y coordinate.
        robot_height_offset: Height offset above ground for the robot base.
        
    Returns:
        Tuple of (x, y, z) position with correct ground height.
    """
    # Use PhysX raycast to find ground height
    ground_height = 0.0
    try:
        from omni.physx import get_physx_scene_query_interface
        
        # Ray origin high above the ground
        origin = (x, y, 50.0)
        # Ray direction pointing down
        direction = (0.0, 0.0, -1.0)
        
        # Perform raycast
        hit = get_physx_scene_query_interface().raycast_closest(
            origin, direction, 100.0  # max distance
        )
        
        if hit["hit"]:
            ground_height = hit["position"][2]
            print(f"[INFO] Raycast hit at height: {ground_height:.3f}")
        else:
            print("[WARN] Raycast did not hit ground, using default height 0.4")
            ground_height = 0.4  # Default height similar to Go2
    except Exception as e:
        print(f"[WARN] Raycast failed: {e}, using default height 0.4")
        ground_height = 0.4
    
    # Final position: fixed x/y, ground height + offset
    final_z = ground_height + robot_height_offset
    
    return (x, y, final_z)


def main():
    """Visualize custom robot in parkour terrain."""
    # Check if the USD file exists
    if not os.path.exists(args_cli.usd_path):
        print(f"[ERROR] USD file not found: {args_cli.usd_path}")
        return
    
    # Parse the Go2 teacher parkour environment configuration
    # Using EVAL config for better visualization with debug info
    task_name = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0"
    env_cfg = parse_env_cfg(
        task_name, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    
    # Reduce number of environments for better visualization
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Enable debug visualization for terrain
    if hasattr(env_cfg, 'parkours') and hasattr(env_cfg.parkours, 'base_parkour'):
        env_cfg.parkours.base_parkour.debug_vis = True
    
    print(f"[INFO] Creating environment: {task_name}")
    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    
    # Create the environment
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)
    
    # Convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # Reset the environment to initialize terrain
    obs, extras = env.reset()
    
    # Get robot info for reference
    robot = env.unwrapped.scene["robot"]
    go2_position = robot.data.root_pos_w[0].cpu().numpy()
    print(f"[INFO] Go2 robot position: {go2_position}")
    
    # Spawn multiple robots around origin
    import random
    import math
    
    num_robots = args_cli.num_robots
    height_offset = args_cli.robot_pos[2]
    
    print(f"[INFO] Spawning {num_robots} custom robots around origin...")
    
    for i in range(num_robots):
        if i == 0:
            # First robot at origin
            spawn_x, spawn_y = 0.0, 0.0
            rotation_z = 0.0
        else:
            # Additional robots in a circle around origin with random offset
            angle = (2 * math.pi / (num_robots - 1)) * (i - 1) + random.uniform(-0.3, 0.3)
            radius = 1.5 + random.uniform(-0.3, 0.3)  # ~1.5m from center
            spawn_x = radius * math.cos(angle)
            spawn_y = radius * math.sin(angle)
            rotation_z = random.uniform(0, 360)  # Random orientation
        
        # Use fixed height (no raycast)
        spawn_position = (spawn_x, spawn_y, height_offset)
        prim_path = f"/World/CustomRobot_{i}"
        
        add_custom_robot_to_stage(
            usd_path=args_cli.usd_path,
            position=spawn_position,
            scale=tuple(args_cli.robot_scale),
            rotation_z=rotation_z,
            prim_path=prim_path,
        )
        print(f"[INFO] Robot {i}: ({spawn_x:.1f}, {spawn_y:.1f}, {height_offset:.2f}), rot {rotation_z:.0f}°")
    
    print("\n" + "=" * 60)
    print(f"Environment and {num_robots} custom robot(s) loaded successfully!")
    print("=" * 60)
    print(f"\nCustom robot USD: {args_cli.usd_path}")
    print(f"Height offset above ground: {height_offset}")
    print("\nTo adjust: --num_robots N  --robot_pos 0 0 <height_offset>")
    print("Press Ctrl+C to exit.")
    print("=" * 60 + "\n")
    
    # Keep the simulation running (custom robot stays at fixed position)
    while simulation_app.is_running():
        try:
            # Step the simulation with zero actions (just for visualization)
            with torch.inference_mode():
                num_actions = env.action_space.shape[-1]
                actions = torch.zeros(args_cli.num_envs, num_actions, device=env.unwrapped.device)
                obs, rewards, dones, truncated, extras = env.step(actions)
            
            # Small delay for smoother visualization
            time.sleep(0.01)
            
        except KeyboardInterrupt:
            print("\n[INFO] Exiting visualization...")
            break
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
