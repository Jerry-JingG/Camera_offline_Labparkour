#!/usr/bin/env python3
"""
Compare 12-dimensional actions between Isaac Lab and MuJoCo simulation.

Reads:
  - Isaac Lab recorded actions (from v3 format verification file)
  - MuJoCo recorded actions (from simple binary format)

Generates:
  - 4x3 subplot comparison curves
  - Error metrics (MSE, MaxDiff per joint)
"""

import argparse
import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_isaac_actions(filepath: str) -> np.ndarray:
    """Read actions from Isaac Lab v3 format verification file.
    
    v3 format:
        Header (28 bytes): version, num_frames, prop_dim, depth_dim, action_dim, mems_dim, tau_dim
        Body: proprio + depth + mems + action + tau per frame
    """
    with open(filepath, "rb") as f:
        # Read header
        version = struct.unpack("<i", f.read(4))[0]
        num_frames = struct.unpack("<i", f.read(4))[0]
        prop_dim = struct.unpack("<i", f.read(4))[0]
        depth_dim = struct.unpack("<i", f.read(4))[0]
        action_dim = struct.unpack("<i", f.read(4))[0]
        mems_dim = struct.unpack("<i", f.read(4))[0]
        
        if version == 3:
            tau_dim = struct.unpack("<i", f.read(4))[0]
        else:
            tau_dim = 0
        
        print(f"[Isaac Lab] v{version} format: {num_frames} frames, action_dim={action_dim}")
        
        # Calculate frame size
        frame_size = prop_dim + depth_dim + mems_dim + action_dim + tau_dim
        
        # Read all frames
        actions = []
        for i in range(num_frames):
            frame_data = np.frombuffer(f.read(frame_size * 4), dtype=np.float32)
            
            # Extract action: offset = prop_dim + depth_dim + mems_dim
            action_offset = prop_dim + depth_dim + mems_dim
            action = frame_data[action_offset:action_offset + action_dim]
            actions.append(action)
        
        return np.array(actions)


def read_mujoco_actions(filepath: str) -> np.ndarray:
    """Read actions from MuJoCo simple binary format.
    
    Format:
        Header (12 bytes): num_frames, action_dim, padding
        Body: action_dim floats per frame
    """
    with open(filepath, "rb") as f:
        # Read header
        num_frames = struct.unpack("<i", f.read(4))[0]
        action_dim = struct.unpack("<i", f.read(4))[0]
        _ = struct.unpack("<i", f.read(4))[0]  # padding
        
        print(f"[MuJoCo] {num_frames} frames, action_dim={action_dim}")
        
        # Read all frames
        actions = []
        for i in range(num_frames):
            action = np.frombuffer(f.read(action_dim * 4), dtype=np.float32)
            actions.append(action)
        
        return np.array(actions)


def plot_comparison(isaac_actions: np.ndarray, mujoco_actions: np.ndarray, 
                    output_path: str, joint_names: list = None):
    """Generate 4x3 subplot comparison curves."""
    
    num_frames = min(len(isaac_actions), len(mujoco_actions))
    isaac_actions = isaac_actions[:num_frames]
    mujoco_actions = mujoco_actions[:num_frames]
    
    if joint_names is None:
        # Default joint names in training order
        joint_names = [
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
        ]
    
    # Create figure with 4x3 subplots
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle(f"Actions Comparison: Isaac Lab vs MuJoCo ({num_frames} frames)", fontsize=14)
    
    time_steps = np.arange(num_frames)
    
    # Calculate overall metrics
    mse_per_joint = np.mean((isaac_actions - mujoco_actions) ** 2, axis=0)
    max_diff_per_joint = np.max(np.abs(isaac_actions - mujoco_actions), axis=0)
    
    for i in range(12):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.plot(time_steps, isaac_actions[:, i], 'b-', label='Isaac Lab', linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, mujoco_actions[:, i], 'r--', label='MuJoCo', linewidth=1.5, alpha=0.8)
        
        ax.set_title(f"{joint_names[i]}\nMSE={mse_per_joint[i]:.2e}, MaxDiff={max_diff_per_joint[i]:.4f}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Action")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Output] Saved comparison plot to: {output_path}")
    
    # Print summary metrics
    print("\n" + "=" * 60)
    print("                    Summary Metrics")
    print("=" * 60)
    print(f"Total frames compared: {num_frames}")
    print(f"Overall MSE:          {np.mean(mse_per_joint):.6e}")
    print(f"Overall MaxDiff:      {np.max(max_diff_per_joint):.6f}")
    print("-" * 60)
    print("Per-joint MSE:")
    for i, name in enumerate(joint_names):
        print(f"  {name:12s}: MSE={mse_per_joint[i]:.6e}, MaxDiff={max_diff_per_joint[i]:.6f}")
    print("=" * 60)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare actions between Isaac Lab and MuJoCo")
    parser.add_argument("--isaac_path", type=str, required=True,
                        help="Path to Isaac Lab verification file (v3 format)")
    parser.add_argument("--mujoco_path", type=str, required=True,
                        help="Path to MuJoCo action recording file")
    parser.add_argument("--output_path", type=str, 
                        default="actions_comparison.png",
                        help="Output path for comparison plot")
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.isaac_path).exists():
        print(f"[Error] Isaac Lab file not found: {args.isaac_path}")
        sys.exit(1)
    if not Path(args.mujoco_path).exists():
        print(f"[Error] MuJoCo file not found: {args.mujoco_path}")
        sys.exit(1)
    
    # Read data
    print("\n[Reading] Isaac Lab actions...")
    isaac_actions = read_isaac_actions(args.isaac_path)
    
    print("\n[Reading] MuJoCo actions...")
    mujoco_actions = read_mujoco_actions(args.mujoco_path)
    
    # Generate comparison plot
    print(f"\n[Plotting] Generating comparison ({len(isaac_actions)} vs {len(mujoco_actions)} frames)...")
    plot_comparison(isaac_actions, mujoco_actions, args.output_path)
    
    print("\n[Done] Comparison complete!")


if __name__ == "__main__":
    main()
