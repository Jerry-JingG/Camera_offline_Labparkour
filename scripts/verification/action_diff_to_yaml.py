#!/usr/bin/env python3
"""
计算前 20 帧的 action diff 并输出为 YAML 格式
"""

import struct
import numpy as np
import yaml
from pathlib import Path

# 关节名称（训练顺序：FL, FR, RL, RR）
JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
]


def load_isaac_actions(filepath):
    """加载 Isaac Lab 录制的 action 数据（v3 格式）"""
    with open(filepath, 'rb') as f:
        # 读取 header (28 bytes)
        version, num_frames, prop_dim, depth_dim, action_dim, mems_dim, tau_dim = \
            struct.unpack('<7i', f.read(28))

        print(f"[Isaac Lab] v{version}: {num_frames} frames, action_dim={action_dim}")

        actions = []
        frame_size = prop_dim + depth_dim + mems_dim + action_dim + tau_dim
        action_offset = prop_dim + depth_dim + mems_dim

        for i in range(num_frames):
            frame_data = struct.unpack(f'<{frame_size}f', f.read(frame_size * 4))
            action = np.array(frame_data[action_offset:action_offset + action_dim])
            actions.append(action)

        return np.array(actions)


def load_mujoco_actions(filepath):
    """加载 MuJoCo 录制的 action 数据"""
    with open(filepath, 'rb') as f:
        # 读取 header (12 bytes)
        num_frames, action_dim, _ = struct.unpack('<3i', f.read(12))

        print(f"[MuJoCo] {num_frames} frames, action_dim={action_dim}")

        actions = []
        for i in range(num_frames):
            action = np.array(struct.unpack(f'<{action_dim}f', f.read(action_dim * 4)))
            actions.append(action)

        return np.array(actions)


def main():
    # 数据路径
    isaac_path = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/isaac_tau_verification.bin")
    mujoco_path = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/mujoco_action_verification.bin")
    output_path = Path("/home/droplet/IsaacLab/Camera_offline_Labparkour/obs_output/action_diff.yaml")

    # 检查文件
    if not isaac_path.exists():
        print(f"[ERROR] Isaac Lab data not found: {isaac_path}")
        return
    if not mujoco_path.exists():
        print(f"[ERROR] MuJoCo data not found: {mujoco_path}")
        return

    # 加载数据
    isaac_actions = load_isaac_actions(isaac_path)
    mujoco_actions = load_mujoco_actions(mujoco_path)

    # 取前 200 帧
    num_frames = min(200, len(isaac_actions), len(mujoco_actions))
    isaac_actions = isaac_actions[:num_frames]
    mujoco_actions = mujoco_actions[:num_frames]

    # 计算 diff
    diff = isaac_actions - mujoco_actions

    # 构建 YAML 结构
    result = {
        "summary": {
            "num_frames": int(num_frames),
            "total_mse": float(np.mean(diff ** 2)),
            "total_mae": float(np.mean(np.abs(diff))),
            "max_diff": float(np.max(np.abs(diff))),
        },
        "per_joint_stats": {},
        "frame_by_frame": []
    }

    # 每个关节的统计
    for j, name in enumerate(JOINT_NAMES):
        joint_diff = diff[:, j]
        result["per_joint_stats"][name] = {
            "mse": float(np.mean(joint_diff ** 2)),
            "mae": float(np.mean(np.abs(joint_diff))),
            "max_diff": float(np.max(np.abs(joint_diff))),
            "mean_diff": float(np.mean(joint_diff)),
        }

    # 每帧每关节的 diff
    for i in range(num_frames):
        frame_data = {
            "frame": i,
            "isaac_actions": {},
            "mujoco_actions": {},
            "diff": {}
        }
        for j, name in enumerate(JOINT_NAMES):
            frame_data["isaac_actions"][name] = float(isaac_actions[i, j])
            frame_data["mujoco_actions"][name] = float(mujoco_actions[i, j])
            frame_data["diff"][name] = float(diff[i, j])
        result["frame_by_frame"].append(frame_data)

    # 输出 YAML
    with open(output_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n[OUTPUT] Saved to: {output_path}")
    print(f"\n[SUMMARY]")
    print(f"  Frames:    {num_frames}")
    print(f"  Total MSE: {result['summary']['total_mse']:.6f}")
    print(f"  Total MAE: {result['summary']['total_mae']:.6f}")
    print(f"  Max Diff:  {result['summary']['max_diff']:.6f}")


if __name__ == "__main__":
    main()
