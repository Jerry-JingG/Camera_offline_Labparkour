#!/usr/bin/env python3
"""可视化验证数据中的 PyTorch 输出和 C++ 推理结果对比。

用法:
    python scripts/visualize_verification.py \\
        --verify_data obs_output/play_student_verify_data.bin \\
        --cpp_output /tmp/cpp_actions.npy \\
        --output_dir /tmp/verification_plots
"""

import argparse
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_verification_data(filepath: str, max_frames: int = -1) -> Tuple[Dict, List[Dict]]:
    """加载 v2 格式验证数据"""
    with open(filepath, "rb") as f:
        # 读取头部
        version = struct.unpack("<i", f.read(4))[0]
        if version != 2:
            raise ValueError(f"Only v2 format supported, got version {version}")

        num_frames = struct.unpack("<i", f.read(4))[0]
        prop_dim = struct.unpack("<i", f.read(4))[0]
        depth_dim = struct.unpack("<i", f.read(4))[0]
        action_dim = struct.unpack("<i", f.read(4))[0]
        mems_dim = struct.unpack("<i", f.read(4))[0]

        header = {
            "version": version,
            "num_frames": num_frames,
            "prop_dim": prop_dim,
            "depth_dim": depth_dim,
            "action_dim": action_dim,
            "mems_dim": mems_dim,
        }

        # 读取帧数据
        frames_to_read = num_frames if max_frames < 0 else min(max_frames, num_frames)
        frames = []

        for i in range(frames_to_read):
            proprio = np.frombuffer(f.read(prop_dim * 4), dtype=np.float32)
            depth = np.frombuffer(f.read(depth_dim * 4), dtype=np.float32)
            mems = np.frombuffer(f.read(mems_dim * 4), dtype=np.float32)
            action = np.frombuffer(f.read(action_dim * 4), dtype=np.float32)

            frames.append({
                "proprio": proprio,
                "depth": depth,
                "mems": mems,
                "action": action,  # PyTorch ground truth
            })

        print(f"[INFO] Loaded {len(frames)} frames from {filepath}")
        return header, frames


def plot_action_comparison(
    pytorch_actions: np.ndarray,
    cpp_actions: np.ndarray,
    output_dir: Path,
    action_names: List[str] = None,
):
    """绘制 PyTorch vs C++ 动作对比图"""
    num_frames, action_dim = pytorch_actions.shape

    if action_names is None:
        action_names = [f"Joint {i}" for i in range(action_dim)]

    # 1. 时间序列对比图（所有关节）
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(action_dim):
        ax = axes[i]
        ax.plot(pytorch_actions[:, i], label="PyTorch", linewidth=2, alpha=0.7)
        ax.plot(cpp_actions[:, i], label="C++ LibTorch", linewidth=2, alpha=0.7, linestyle="--")
        ax.set_title(action_names[i])
        ax.set_xlabel("Frame")
        ax.set_ylabel("Action Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "action_timeseries.png", dpi=150)
    print(f"[INFO] Saved: {output_dir / 'action_timeseries.png'}")
    plt.close()

    # 2. 误差分析图
    errors = np.abs(pytorch_actions - cpp_actions)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 2.1 每帧的最大误差
    ax = axes[0, 0]
    max_errors_per_frame = errors.max(axis=1)
    ax.plot(max_errors_per_frame, linewidth=2)
    ax.set_title("Maximum Error Per Frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Max Absolute Error")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-3, color='r', linestyle='--', label='Threshold (1e-3)')
    ax.legend()

    # 2.2 每个关节的平均误差
    ax = axes[0, 1]
    mean_errors_per_joint = errors.mean(axis=0)
    ax.bar(range(action_dim), mean_errors_per_joint)
    ax.set_title("Mean Error Per Joint")
    ax.set_xlabel("Joint Index")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xticks(range(action_dim))
    ax.set_xticklabels([f"J{i}" for i in range(action_dim)], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # 2.3 误差热力图
    ax = axes[1, 0]
    im = ax.imshow(errors.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_title("Error Heatmap (Joint × Frame)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Joint Index")
    plt.colorbar(im, ax=ax, label="Absolute Error")

    # 2.4 误差分布直方图
    ax = axes[1, 1]
    ax.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_title("Error Distribution")
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Frequency")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1e-3, color='r', linestyle='--', label='Threshold (1e-3)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "error_analysis.png", dpi=150)
    print(f"[INFO] Saved: {output_dir / 'error_analysis.png'}")
    plt.close()

    # 3. 统计摘要
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    print(f"总帧数: {num_frames}")
    print(f"动作维度: {action_dim}")
    print(f"\n误差统计:")
    print(f"  Mean Error:    {errors.mean():.6e}")
    print(f"  Std Error:     {errors.std():.6e}")
    print(f"  Max Error:     {errors.max():.6e}")
    print(f"  Min Error:     {errors.min():.6e}")
    print(f"  Median Error:  {np.median(errors):.6e}")
    print(f"\n每关节最大误差:")
    for i in range(action_dim):
        print(f"  {action_names[i]:15s}: {errors[:, i].max():.6e}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="可视化验证数据对比")
    parser.add_argument("--verify_data", type=str, required=True, help="验证数据文件路径 (.bin)")
    parser.add_argument("--cpp_output", type=str, default=None, help="C++ 输出的动作数据 (.npy)")
    parser.add_argument("--output_dir", type=str, default="/tmp/verification_plots", help="输出图表目录")
    parser.add_argument("--max_frames", type=int, default=-1, help="最大帧数（-1 表示全部）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载验证数据
    header, frames = load_verification_data(args.verify_data, args.max_frames)

    # 提取 PyTorch actions
    pytorch_actions = np.array([frame["action"] for frame in frames])

    # 如果提供了 C++ 输出，加载并对比
    if args.cpp_output:
        cpp_actions = np.load(args.cpp_output)

        if cpp_actions.shape != pytorch_actions.shape:
            raise ValueError(f"Shape mismatch: PyTorch {pytorch_actions.shape} vs C++ {cpp_actions.shape}")

        # Go2 机器人关节名称
        action_names = [
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
        ]

        plot_action_comparison(pytorch_actions, cpp_actions, output_dir, action_names)
    else:
        # 只可视化 PyTorch 数据
        print("[INFO] No C++ output provided, visualizing PyTorch data only")

        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i in range(header["action_dim"]):
            ax = axes[i]
            ax.plot(pytorch_actions[:, i], linewidth=2)
            ax.set_title(f"Joint {i}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Action Value")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "pytorch_actions.png", dpi=150)
        print(f"[INFO] Saved: {output_dir / 'pytorch_actions.png'}")
        plt.close()


if __name__ == "__main__":
    main()
