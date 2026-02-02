#!/usr/bin/env python3
"""
对比 Isaac Lab 和 MuJoCo 的关节力矩（tau）输出

读取两边录制的 tau 数据，生成 12 个关节的对比图
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 关节名称（训练顺序：FL, FR, RL, RR）
JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# 关节力矩限位（从 base.yaml）
TORQUE_LIMITS = 33.5  # N·m


def load_isaac_tau(filepath):
    """加载 Isaac Lab 录制的 tau 数据（v3 格式）"""
    with open(filepath, 'rb') as f:
        # 读取 header (28 bytes)
        version, num_frames, prop_dim, depth_dim, action_dim, mems_dim, tau_dim = \
            struct.unpack('<7i', f.read(28))

        print(f"[INFO] Loading Isaac Lab data:")
        print(f"  Version: {version}")
        print(f"  Frames: {num_frames}")
        print(f"  Tau dim: {tau_dim}")

        if version != 3:
            raise ValueError(f"Expected version 3, got {version}")

        # 读取每帧数据
        tau_data = []
        frame_size = prop_dim + depth_dim + mems_dim + action_dim + tau_dim

        for i in range(num_frames):
            frame_data = struct.unpack(f'<{frame_size}f', f.read(frame_size * 4))
            # 提取 tau（最后 tau_dim 个元素）
            tau = np.array(frame_data[-tau_dim:])
            tau_data.append(tau)

        return np.array(tau_data)


def load_mujoco_tau(filepath):
    """加载 MuJoCo 录制的 tau 数据"""
    with open(filepath, 'rb') as f:
        # 读取 header (12 bytes)
        num_frames, tau_dim, _ = struct.unpack('<3i', f.read(12))

        print(f"[INFO] Loading MuJoCo data:")
        print(f"  Frames: {num_frames}")
        print(f"  Tau dim: {tau_dim}")

        # 读取每帧数据
        tau_data = []
        for i in range(num_frames):
            tau = np.array(struct.unpack(f'<{tau_dim}f', f.read(tau_dim * 4)))
            tau_data.append(tau)

        return np.array(tau_data)


def compute_statistics(isaac_tau, mujoco_tau):
    """计算统计信息"""
    # 确保帧数一致
    min_frames = min(len(isaac_tau), len(mujoco_tau))
    isaac_tau = isaac_tau[:min_frames]
    mujoco_tau = mujoco_tau[:min_frames]

    # 计算差异
    diff = isaac_tau - mujoco_tau
    mse = np.mean(diff ** 2, axis=0)  # 每个关节的 MSE
    mae = np.mean(np.abs(diff), axis=0)  # 每个关节的 MAE
    max_diff = np.max(np.abs(diff), axis=0)  # 每个关节的最大差异

    return {
        'mse': mse,
        'mae': mae,
        'max_diff': max_diff,
        'diff': diff
    }


def plot_tau_comparison(isaac_tau, mujoco_tau, max_frames=200):
    """绘制 tau 对比图"""
    min_frames = min(len(isaac_tau), len(mujoco_tau), max_frames)
    isaac_tau = isaac_tau[:min_frames]
    mujoco_tau = mujoco_tau[:min_frames]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle('Joint Torque Comparison: Isaac Lab vs MuJoCo', fontsize=16)

    time_steps = np.arange(min_frames)

    for joint_idx in range(12):
        row = joint_idx // 3
        col = joint_idx % 3
        ax = axes[row, col]

        # 绘制 Isaac Lab tau
        ax.plot(time_steps, isaac_tau[:, joint_idx],
                label='Isaac Lab', color='blue', alpha=0.7, linewidth=1.5)

        # 绘制 MuJoCo tau
        ax.plot(time_steps, mujoco_tau[:, joint_idx],
                label='MuJoCo', color='red', alpha=0.7, linewidth=1.5)

        # 绘制力矩限位
        ax.axhline(y=TORQUE_LIMITS, color='orange', linestyle='--',
                   alpha=0.5, label='Torque Limit')
        ax.axhline(y=-TORQUE_LIMITS, color='orange', linestyle='--', alpha=0.5)

        ax.set_title(JOINT_NAMES[joint_idx], fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylabel('Torque (N·m)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        # 标记超限区域
        isaac_violations = (np.abs(isaac_tau[:, joint_idx]) > TORQUE_LIMITS)
        mujoco_violations = (np.abs(mujoco_tau[:, joint_idx]) > TORQUE_LIMITS)

        if np.any(isaac_violations):
            ax.fill_between(time_steps, -TORQUE_LIMITS, TORQUE_LIMITS,
                           where=isaac_violations, color='blue', alpha=0.2)
        if np.any(mujoco_violations):
            ax.fill_between(time_steps, -TORQUE_LIMITS, TORQUE_LIMITS,
                           where=mujoco_violations, color='red', alpha=0.2)

    plt.tight_layout()
    return fig


def plot_difference(isaac_tau, mujoco_tau, max_frames=200):
    """绘制差异图"""
    min_frames = min(len(isaac_tau), len(mujoco_tau), max_frames)
    diff = isaac_tau[:min_frames] - mujoco_tau[:min_frames]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle('Torque Difference: Isaac Lab - MuJoCo', fontsize=16)

    time_steps = np.arange(min_frames)

    for joint_idx in range(12):
        row = joint_idx // 3
        col = joint_idx % 3
        ax = axes[row, col]

        ax.plot(time_steps, diff[:, joint_idx],
                color='purple', alpha=0.7, linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        ax.set_title(JOINT_NAMES[joint_idx], fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylabel('Difference (N·m)', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    # 数据路径
    isaac_path = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/isaac_tau_verification.bin")
    mujoco_path = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/mujoco_action_verification.bin")

    # 检查文件存在
    if not isaac_path.exists():
        print(f"[ERROR] Isaac Lab data not found: {isaac_path}")
        return
    if not mujoco_path.exists():
        print(f"[ERROR] MuJoCo data not found: {mujoco_path}")
        return

    # 加载数据
    print("\n" + "="*70)
    isaac_tau = load_isaac_tau(isaac_path)
    mujoco_tau = load_mujoco_tau(mujoco_path)

    # 计算统计信息
    print("\n" + "="*70)
    print("[INFO] Computing statistics...")
    stats = compute_statistics(isaac_tau, mujoco_tau)

    print("\n[INFO] Per-joint statistics:")
    print(f"{'Joint':<15} {'MSE':<12} {'MAE':<12} {'Max Diff':<12}")
    print("-" * 55)
    for i, name in enumerate(JOINT_NAMES):
        print(f"{name:<15} {stats['mse'][i]:<12.6f} {stats['mae'][i]:<12.6f} {stats['max_diff'][i]:<12.6f}")

    print(f"\n[INFO] Overall statistics:")
    print(f"  Mean MSE:      {np.mean(stats['mse']):.6f}")
    print(f"  Mean MAE:      {np.mean(stats['mae']):.6f}")
    print(f"  Max Diff:      {np.max(stats['max_diff']):.6f}")

    # 绘图
    print("\n" + "="*70)
    print("[INFO] Generating plots...")

    fig1 = plot_tau_comparison(isaac_tau, mujoco_tau, max_frames=200)
    output_path1 = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/tau_comparison.png")
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"[INFO] Comparison plot saved to: {output_path1}")

    fig2 = plot_difference(isaac_tau, mujoco_tau, max_frames=200)
    output_path2 = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/tau_difference.png")
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"[INFO] Difference plot saved to: {output_path2}")

    print("\n" + "="*70)
    print("[INFO] Done! Check the plots to diagnose the issue.")
    print("\n[HINT] Look for:")
    print("  - Large differences in specific joints")
    print("  - Torque saturation (hitting limits)")
    print("  - Oscillations or instability")
    print("  - Phase shifts between Isaac Lab and MuJoCo")


if __name__ == "__main__":
    main()

