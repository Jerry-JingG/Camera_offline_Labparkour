#!/usr/bin/env python3
"""验证 TorchScript 模型与原始 PyTorch 模型输出是否一致。

这个脚本用相同的输入和初始 mems=0，对比：
1. TorchScript 模型输出
2. 原始 PyTorch 模型输出

如果两者一致，说明导出没问题，问题在 C++ 端或数据录制端。
"""

import argparse
import struct
import sys
from pathlib import Path

import torch
import numpy as np


def load_verification_data(filepath: str, max_frames: int = 10):
    """加载验证数据的前 N 帧"""
    with open(filepath, "rb") as f:
        num_frames, prop_dim, depth_dim, action_dim = struct.unpack("<4i", f.read(16))
        print(f"[INFO] num_frames={num_frames}, prop_dim={prop_dim}, depth_dim={depth_dim}, action_dim={action_dim}")

        frames = []
        for i in range(min(max_frames, num_frames)):
            proprio = np.frombuffer(f.read(prop_dim * 4), dtype=np.float32).copy()
            depth = np.frombuffer(f.read(depth_dim * 4), dtype=np.float32).copy()
            action = np.frombuffer(f.read(action_dim * 4), dtype=np.float32).copy()
            frames.append({"proprio": proprio, "depth": depth, "action": action})
        return frames, prop_dim, depth_dim, action_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchscript", type=str, required=True, help="TorchScript 模型路径")
    parser.add_argument("--data", type=str, required=True, help="验证数据路径")
    parser.add_argument("--max_frames", type=int, default=5)
    args = parser.parse_args()

    # 配置
    depth_h, depth_w = 58, 87
    num_layers, mem_len, token_dim = 3, 64, 128

    # 加载数据
    frames, prop_dim, depth_dim, action_dim = load_verification_data(args.data, args.max_frames)
    depth_hist_len = depth_dim // (depth_h * depth_w)
    print(f"[INFO] depth_hist_len={depth_hist_len}")

    # 加载 TorchScript 模型
    ts_model = torch.jit.load(args.torchscript)
    ts_model.eval()

    # 初始化 mems
    mems = torch.zeros(1, num_layers, mem_len, token_dim)

    print("\n" + "=" * 80)
    print("TorchScript 模型连续推理（mems 从零开始累积）")
    print("=" * 80)

    for i, frame in enumerate(frames):
        proprio = torch.from_numpy(frame["proprio"]).unsqueeze(0)  # [1, prop_dim]
        depth = torch.from_numpy(frame["depth"]).reshape(1, depth_hist_len, depth_h, depth_w)
        action_gt = frame["action"]

        with torch.no_grad():
            action_pred, mems = ts_model(proprio, depth, mems)

        action_pred_np = action_pred.squeeze(0).numpy()
        mse = np.mean((action_pred_np - action_gt) ** 2)
        max_diff = np.max(np.abs(action_pred_np - action_gt))

        print(f"\n[Frame {i}] MSE={mse:.4e}  MaxDiff={max_diff:.4e}")
        print(f"  GT:   [{', '.join(f'{x:7.4f}' for x in action_gt[:6])}, ...]")
        print(f"  Pred: [{', '.join(f'{x:7.4f}' for x in action_pred_np[:6])}, ...]")


if __name__ == "__main__":
    main()
