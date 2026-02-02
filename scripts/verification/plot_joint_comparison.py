#!/usr/bin/env python3
"""
对比 Isaac Lab 的 action 和 MuJoCo 计算的关节位置目标

从 Isaac Lab 录制的验证数据中读取 action，
然后计算 MuJoCo 的关节位置目标：target_pos = action * 0.25 + default_pos
绘制 12 个关节的对比图
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Isaac Lab 的默认关节位置（训练顺序：FL, FR, RL, RR）
DEFAULT_DOF_POS = np.array([
    0.10, 0.80, -1.50,   # FL (front left): hip=+0.1
    -0.10, 0.80, -1.50,  # FR (front right): hip=-0.1
    0.10, 1.00, -1.50,   # RL (rear left): hip=+0.1, thigh=1.0
    -0.10, 1.00, -1.50   # RR (rear right): hip=-0.1, thigh=1.0
])

# 关节名称（训练顺序）
JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]

# 关节限位（从 MuJoCo XML）
JOINT_LIMITS = {
    "hip": (-1.0472, 1.0472),      # ±60°
    "front_thigh": (-1.5708, 3.4907),  # front hip
    "back_thigh": (-0.5236, 4.5379),   # back hip
    "calf": (-2.7227, -0.83776)
}

def get_joint_limits(joint_idx):
    """获取关节限位"""
    joint_type = joint_idx % 3  # 0=hip, 1=thigh, 2=calf
    if joint_type == 0:  # hip
        return JOINT_LIMITS["hip"]
    elif joint_type == 1:  # thigh
        if joint_idx < 6:  # front legs
            return JOINT_LIMITS["front_thigh"]
        else:  # back legs
            return JOINT_LIMITS["back_thigh"]
    else:  # calf
        return JOINT_LIMITS["calf"]


def load_verification_data(filepath):
    """加载 Isaac Lab 录制的验证数据（v2 格式）"""
    with open(filepath, 'rb') as f:
        # 读取 header (24 bytes)
        version, num_frames, prop_dim, depth_dim, action_dim, mems_dim = struct.unpack('<6i', f.read(24))

        print(f"[INFO] Loading verification data:")
        print(f"  Version: {version}")
        print(f"  Frames: {num_frames}")
        print(f"  Action dim: {action_dim}")

        if version != 2:
            raise ValueError(f"Unsupported version: {version}")

        # 读取每帧数据
        actions = []
        frame_size = prop_dim + depth_dim + mems_dim + action_dim

        for i in range(num_frames):
            frame_data = struct.unpack(f'<{frame_size}f', f.read(frame_size * 4))
            # 提取 action（最后 action_dim 个元素）
            action = np.array(frame_data[-action_dim:])
            actions.append(action)

        return np.array(actions)


def compute_mujoco_target_pos(actions, action_scale=0.25):
    """
    计算 MuJoCo 的关节位置目标

    target_pos = action * action_scale + default_pos
    """
    num_frames, action_dim = actions.shape
    target_positions = np.zeros((num_frames, action_dim))

    for i in range(num_frames):
        target_positions[i] = actions[i] * action_scale + DEFAULT_DOF_POS

    return target_positions


def check_joint_limits(target_positions):
    """检查关节位置是否超出限位"""
    num_frames, num_joints = target_positions.shape
    violations = []

    for joint_idx in range(num_joints):
        lower, upper = get_joint_limits(joint_idx)
        joint_data = target_positions[:, joint_idx]

        below_lower = np.sum(joint_data < lower)
        above_upper = np.sum(joint_data > upper)

        if below_lower > 0 or above_upper > 0:
            violations.append({
                'joint': JOINT_NAMES[joint_idx],
                'below_lower': below_lower,
                'above_upper': above_upper,
                'min_val': joint_data.min(),
                'max_val': joint_data.max(),
                'limits': (lower, upper)
            })

    return violations


def plot_joint_comparison(actions, target_positions, max_frames=200):
    """绘制关节对比图"""
    num_frames = min(len(actions), max_frames)
    actions = actions[:num_frames]
    target_positions = target_positions[:num_frames]

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    fig.suptitle('Joint Position Targets (MuJoCo) vs Actions (Isaac Lab)', fontsize=16)

    time_steps = np.arange(num_frames)

    for joint_idx in range(12):
        row = joint_idx // 3
        col = joint_idx % 3
        ax = axes[row, col]

        # 绘制 action（原始输出）
        ax.plot(time_steps, actions[:, joint_idx],
                label='Action (raw)', color='blue', alpha=0.7, linewidth=1.5)

        # 绘制 target position（action * 0.25 + default）
        ax.plot(time_steps, target_positions[:, joint_idx],
                label='Target Pos (scaled)', color='red', alpha=0.7, linewidth=1.5)

        # 绘制 default position
        ax.axhline(y=DEFAULT_DOF_POS[joint_idx],
                   color='green', linestyle='--', alpha=0.5, label='Default Pos')

        # 绘制关节限位
        lower, upper = get_joint_limits(joint_idx)
        ax.axhline(y=lower, color='orange', linestyle=':', alpha=0.5, label='Lower Limit')
        ax.axhline(y=upper, color='orange', linestyle=':', alpha=0.5, label='Upper Limit')

        ax.set_title(JOINT_NAMES[joint_idx], fontsize=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Position (rad)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='best')

        # 标记超限区域
        violations = (target_positions[:, joint_idx] < lower) | (target_positions[:, joint_idx] > upper)
        if np.any(violations):
            ax.fill_between(time_steps, lower, upper,
                           where=violations, color='red', alpha=0.2)

    plt.tight_layout()
    return fig


def main():
    # 数据路径
    verify_data_path = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/play_student_verify_data.bin")

    if not verify_data_path.exists():
        print(f"[ERROR] Verification data not found: {verify_data_path}")
        print("Please run Isaac Lab with RECORD_DATA=true first.")
        return

    # 加载数据
    print("\n" + "="*60)
    actions = load_verification_data(verify_data_path)
    print(f"[INFO] Loaded {len(actions)} frames of action data")

    # 计算 MuJoCo 目标位置
    target_positions = compute_mujoco_target_pos(actions)

    # 检查关节限位
    print("\n" + "="*60)
    print("[INFO] Checking joint limits...")
    violations = check_joint_limits(target_positions)

    if violations:
        print(f"\n[WARNING] Found {len(violations)} joints with limit violations:")
        for v in violations:
            print(f"\n  Joint: {v['joint']}")
            print(f"    Limits: [{v['limits'][0]:.4f}, {v['limits'][1]:.4f}]")
            print(f"    Actual: [{v['min_val']:.4f}, {v['max_val']:.4f}]")
            print(f"    Violations: {v['below_lower']} below, {v['above_upper']} above")
    else:
        print("[INFO] ✓ All joint positions within limits")

    # 统计信息
    print("\n" + "="*60)
    print("[INFO] Action statistics:")
    print(f"  Mean: {actions.mean():.4f}")
    print(f"  Std:  {actions.std():.4f}")
    print(f"  Min:  {actions.min():.4f}")
    print(f"  Max:  {actions.max():.4f}")

    print("\n[INFO] Target position statistics:")
    print(f"  Mean: {target_positions.mean():.4f}")
    print(f"  Std:  {target_positions.std():.4f}")
    print(f"  Min:  {target_positions.min():.4f}")
    print(f"  Max:  {target_positions.max():.4f}")

    # 绘图
    print("\n" + "="*60)
    print("[INFO] Generating plots...")
    fig = plot_joint_comparison(actions, target_positions, max_frames=200)

    # 保存图片
    output_path = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/joint_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Plot saved to: {output_path}")

    # 显示图片
    plt.show()

    print("\n" + "="*60)
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
