#!/usr/bin/env python3
"""
Compare proprio observations recorded from MuJoCo and IsaacLab.

Usage:
    python compare_proprio.py \
        --mujoco /path/to/mujoco_proprio_verification.bin \
        --isaac /path/to/isaac_proprio_verification.bin \
        --plot --yaml_output proprio_diff.yaml

Binary file format (both have same format):
    Header (12 bytes):
        int32: num_frames
        int32: proprio_dim (53)
        int32: proprio_dim (padding)
    Body (repeated num_frames times):
        float32 * proprio_dim: proprio data
"""

import argparse
import struct
import numpy as np
from pathlib import Path
import yaml
import os


# Proprio dimension labels (matching IsaacLab ExtremeParkourObservations)
PROPRIO_LABELS = [
    # [0-2] Angular velocity in body frame * 0.25
    "ang_vel_x*0.25", "ang_vel_y*0.25", "ang_vel_z*0.25",
    # [3-4] Roll and Pitch
    "roll", "pitch",
    # [5] Zeros placeholder
    "zeros_0",
    # [6-7] delta_yaw and delta_next_yaw
    "delta_yaw", "delta_next_yaw",
    # [8-9] Zeros (commands[:, 0:2] * 0)
    "zeros_1", "zeros_2",
    # [10] Command x velocity
    "cmd_x",
    # [11-12] Terrain type indicators
    "terrain_type", "inv_terrain_type",
    # [13-24] Joint positions - default (12 joints)
    "joint_pos_FL_hip", "joint_pos_FL_thigh", "joint_pos_FL_calf",
    "joint_pos_FR_hip", "joint_pos_FR_thigh", "joint_pos_FR_calf",
    "joint_pos_RL_hip", "joint_pos_RL_thigh", "joint_pos_RL_calf",
    "joint_pos_RR_hip", "joint_pos_RR_thigh", "joint_pos_RR_calf",
    # [25-36] Joint velocities * 0.05 (12 joints)
    "joint_vel_FL_hip", "joint_vel_FL_thigh", "joint_vel_FL_calf",
    "joint_vel_FR_hip", "joint_vel_FR_thigh", "joint_vel_FR_calf",
    "joint_vel_RL_hip", "joint_vel_RL_thigh", "joint_vel_RL_calf",
    "joint_vel_RR_hip", "joint_vel_RR_thigh", "joint_vel_RR_calf",
    # [37-48] Previous actions (12 joints)
    "action_FL_hip", "action_FL_thigh", "action_FL_calf",
    "action_FR_hip", "action_FR_thigh", "action_FR_calf",
    "action_RL_hip", "action_RL_thigh", "action_RL_calf",
    "action_RR_hip", "action_RR_thigh", "action_RR_calf",
    # [49-52] Contact indicators (4 feet)
    "contact_FL", "contact_FR", "contact_RL", "contact_RR",
]


def load_proprio_file(filepath: str) -> np.ndarray:
    """Load proprio data from binary file."""
    with open(filepath, "rb") as f:
        # Read header
        num_frames = struct.unpack("<i", f.read(4))[0]
        proprio_dim = struct.unpack("<i", f.read(4))[0]
        _ = struct.unpack("<i", f.read(4))[0]  # padding
        
        print(f"  Loaded: num_frames={num_frames}, proprio_dim={proprio_dim}")
        
        # Read frames
        frames = []
        for _ in range(num_frames):
            frame = np.frombuffer(f.read(proprio_dim * 4), dtype=np.float32).copy()
            frames.append(frame)
        
        return np.stack(frames)  # [num_frames, proprio_dim]


def compare_proprio(mujoco_data: np.ndarray, isaac_data: np.ndarray) -> list:
    """Compare proprio data and print differences. Returns per-dim mean diffs."""
    num_frames = min(len(mujoco_data), len(isaac_data))
    print(f"\n{'='*80}")
    print(f"Comparing {num_frames} frames")
    print(f"{'='*80}")
    
    # Per-dimension statistics
    print(f"\n{'Dimension':<30} {'Idx':>4} {'MuJoCo Mean':>12} {'Isaac Mean':>12} {'Diff Mean':>12} {'Max Diff':>10}")
    print("-" * 80)
    
    all_diffs = []
    for dim_idx in range(53):
        mj_vals = mujoco_data[:num_frames, dim_idx]
        is_vals = isaac_data[:num_frames, dim_idx]
        diff = mj_vals - is_vals
        
        mean_mj = np.mean(mj_vals)
        mean_is = np.mean(is_vals)
        mean_diff = np.mean(np.abs(diff))
        max_diff = np.max(np.abs(diff))
        
        all_diffs.append(mean_diff)
        
        label = PROPRIO_LABELS[dim_idx] if dim_idx < len(PROPRIO_LABELS) else f"dim_{dim_idx}"
        
        # Highlight large differences
        highlight = " ⚠️" if mean_diff > 0.01 else ""
        print(f"{label:<30} [{dim_idx:>2}] {mean_mj:>12.6f} {mean_is:>12.6f} {mean_diff:>12.6f} {max_diff:>10.6f}{highlight}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_mean_diff = np.mean(all_diffs)
    max_dim_diff_idx = np.argmax(all_diffs)
    max_dim_label = PROPRIO_LABELS[max_dim_diff_idx] if max_dim_diff_idx < len(PROPRIO_LABELS) else f"dim_{max_dim_diff_idx}"
    
    print(f"Total mean absolute difference: {total_mean_diff:.6f}")
    print(f"Largest difference dimension: [{max_dim_diff_idx}] {max_dim_label} (mean_diff={all_diffs[max_dim_diff_idx]:.6f})")
    
    # Group-wise analysis
    groups = [
        ("Angular velocity [0-2]", 0, 3),
        ("Roll/Pitch [3-4]", 3, 5),
        ("Zeros [5,8,9]", 5, 6),
        ("Delta yaw [6-7]", 6, 8),
        ("Command [10]", 10, 11),
        ("Terrain [11-12]", 11, 13),
        ("Joint pos [13-24]", 13, 25),
        ("Joint vel [25-36]", 25, 37),
        ("Actions [37-48]", 37, 49),
        ("Contact [49-52]", 49, 53),
    ]
    
    print(f"\nGroup-wise mean absolute difference:")
    for group_name, start, end in groups:
        group_diff = np.mean(all_diffs[start:end])
        status = "✅" if group_diff < 0.01 else "⚠️" if group_diff < 0.1 else "❌"
        print(f"  {status} {group_name}: {group_diff:.6f}")
    
    return all_diffs


def plot_comparisons(mujoco_data: np.ndarray, isaac_data: np.ndarray, output_dir: str) -> None:
    """Plot comparison graphs for each dimension."""
    import matplotlib.pyplot as plt
    
    num_frames = min(len(mujoco_data), len(isaac_data))
    frames_range = np.arange(num_frames)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for all dimensions (organized by groups)
    groups = [
        ("angular_velocity", [0, 1, 2]),
        ("roll_pitch", [3, 4]),
        ("zeros_delta_yaw", [5, 6, 7, 8, 9]),
        ("command_terrain", [10, 11, 12]),
        ("joint_pos", list(range(13, 25))),
        ("joint_vel", list(range(25, 37))),
        ("actions", list(range(37, 49))),
        ("contact", [49, 50, 51, 52]),
    ]
    
    for group_name, dim_indices in groups:
        n_dims = len(dim_indices)
        n_cols = min(3, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
        if n_dims == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_dims > 1 else axes
        
        for i, dim_idx in enumerate(dim_indices):
            ax = axes[i] if n_dims > 1 else axes[0]
            label = PROPRIO_LABELS[dim_idx] if dim_idx < len(PROPRIO_LABELS) else f"dim_{dim_idx}"
            
            mj_vals = mujoco_data[:num_frames, dim_idx]
            is_vals = isaac_data[:num_frames, dim_idx]
            diff = mj_vals - is_vals
            
            ax.plot(frames_range, mj_vals, 'b-', label='MuJoCo', alpha=0.7, linewidth=1)
            ax.plot(frames_range, is_vals, 'r--', label='Isaac', alpha=0.7, linewidth=1)
            ax.fill_between(frames_range, mj_vals, is_vals, alpha=0.3, color='gray')
            
            ax.set_title(f"[{dim_idx}] {label}", fontsize=9)
            ax.set_xlabel("Frame", fontsize=8)
            ax.set_ylabel("Value", fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(dim_indices), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"proprio_{group_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    # Create overview heatmap of all dimensions
    fig, ax = plt.subplots(figsize=(14, 8))
    
    diff_matrix = mujoco_data[:num_frames] - isaac_data[:num_frames]  # [frames, dims]
    im = ax.imshow(diff_matrix.T, aspect='auto', cmap='RdBu_r', 
                   vmin=-0.5, vmax=0.5, interpolation='nearest')
    
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Dimension", fontsize=10)
    ax.set_title("MuJoCo - Isaac Difference Heatmap", fontsize=12)
    
    # Add dimension labels on y-axis (every 5th)
    yticks = list(range(0, 53, 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"[{i}] {PROPRIO_LABELS[i][:15]}" for i in yticks], fontsize=7)
    
    plt.colorbar(im, ax=ax, label="Difference (MuJoCo - Isaac)")
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, "proprio_diff_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {heatmap_path}")


def save_yaml_diff(mujoco_data: np.ndarray, isaac_data: np.ndarray, output_path: str) -> None:
    """Save per-frame, per-dimension diff to YAML file."""
    num_frames = min(len(mujoco_data), len(isaac_data))
    
    # Calculate diffs
    diff_data = {
        "metadata": {
            "num_frames": int(num_frames),
            "proprio_dim": 53,
            "description": "Per-frame, per-dimension difference (MuJoCo - Isaac)"
        },
        "dimension_labels": PROPRIO_LABELS,
        "summary": {
            "per_dim_mean_abs_diff": {},
            "per_dim_max_abs_diff": {},
        },
        "frames": []
    }
    
    # Per-dimension summary
    for dim_idx in range(53):
        label = PROPRIO_LABELS[dim_idx]
        mj_vals = mujoco_data[:num_frames, dim_idx]
        is_vals = isaac_data[:num_frames, dim_idx]
        diff = mj_vals - is_vals
        
        diff_data["summary"]["per_dim_mean_abs_diff"][label] = float(np.mean(np.abs(diff)))
        diff_data["summary"]["per_dim_max_abs_diff"][label] = float(np.max(np.abs(diff)))
    
    # Per-frame data
    for frame_idx in range(num_frames):
        frame_entry = {
            "frame": frame_idx,
            "dimensions": {}
        }
        
        for dim_idx in range(53):
            label = PROPRIO_LABELS[dim_idx]
            mj_val = float(mujoco_data[frame_idx, dim_idx])
            is_val = float(isaac_data[frame_idx, dim_idx])
            diff = mj_val - is_val
            
            frame_entry["dimensions"][label] = {
                "mujoco": round(mj_val, 6),
                "isaac": round(is_val, 6),
                "diff": round(diff, 6),
            }
        
        diff_data["frames"].append(frame_entry)
    
    # Write YAML
    with open(output_path, 'w') as f:
        yaml.dump(diff_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"  Saved YAML diff: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare proprio observations from MuJoCo and IsaacLab")
    parser.add_argument("--mujoco", type=str, 
                        default="/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/mujoco_proprio_verification.bin",
                        help="Path to MuJoCo proprio recording")
    parser.add_argument("--isaac", type=str,
                        default="/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/isaac_proprio_verification.bin",
                        help="Path to IsaacLab proprio recording")
    parser.add_argument("--frame", type=int, default=None,
                        help="Print detailed comparison for a specific frame")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plots for each dimension")
    parser.add_argument("--plot_output", type=str,
                        default="/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/proprio_plots",
                        help="Output directory for plots")
    parser.add_argument("--yaml_output", type=str,
                        default="/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/proprio_diff.yaml",
                        help="Output path for YAML diff file")
    parser.add_argument("--no_yaml", action="store_true",
                        help="Disable YAML output")
    args = parser.parse_args()
    
    print("Loading MuJoCo proprio data...")
    mujoco_data = load_proprio_file(args.mujoco)
    
    print("Loading IsaacLab proprio data...")
    isaac_data = load_proprio_file(args.isaac)
    
    all_diffs = compare_proprio(mujoco_data, isaac_data)
    
    # Generate plots if requested
    if args.plot:
        print(f"\nGenerating comparison plots...")
        plot_comparisons(mujoco_data, isaac_data, args.plot_output)
    
    # Save YAML diff
    if not args.no_yaml:
        print(f"\nSaving YAML diff...")
        save_yaml_diff(mujoco_data, isaac_data, args.yaml_output)
    
    # Detailed frame comparison if requested
    if args.frame is not None:
        frame_idx = args.frame
        print(f"\n{'='*80}")
        print(f"Detailed comparison for Frame {frame_idx}")
        print(f"{'='*80}")
        print(f"{'Dimension':<30} {'Idx':>4} {'MuJoCo':>12} {'Isaac':>12} {'Diff':>12}")
        print("-" * 80)
        
        for dim_idx in range(53):
            mj_val = mujoco_data[frame_idx, dim_idx]
            is_val = isaac_data[frame_idx, dim_idx]
            diff = mj_val - is_val
            
            label = PROPRIO_LABELS[dim_idx] if dim_idx < len(PROPRIO_LABELS) else f"dim_{dim_idx}"
            highlight = " ⚠️" if abs(diff) > 0.01 else ""
            print(f"{label:<30} [{dim_idx:>2}] {mj_val:>12.6f} {is_val:>12.6f} {diff:>12.6f}{highlight}")


if __name__ == "__main__":
    main()
