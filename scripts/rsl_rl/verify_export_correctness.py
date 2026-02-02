#!/usr/bin/env python3
"""验证 TorchScript 导出正确性：对比原始 PyTorch 模型和 TorchScript 模型输出。

使用相同的随机输入，验证两者输出完全一致（MSE < 1e-10）。
这可以确认导出过程没有引入误差。
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "parkour_tasks"))

from train_student_from_dataset import MultiModalStudentPolicy


def load_pytorch_model(ckpt_path: str):
    """加载原始 PyTorch 模型"""
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {})

    num_prop = int(meta["num_prop"])
    action_dim = int(meta["action_dim"])
    camera_resolution = tuple(meta.get("camera_resolution", [58, 87]))
    prop_hist_len = int(meta.get("prop_hist_len", 1))
    depth_hist_len = int(meta.get("depth_hist_len", 4))
    mem_len = int(meta.get("sequence_length", 64))

    fusion_cfg = {"num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.1, "attn_dropout": 0.1, "grid_size": 4}
    temporal_cfg = {"num_layers": 3, "num_heads": 4, "d_inner": 256, "mem_len": mem_len, "dropout": 0.1, "attn_dropout": 0.1}
    action_head_cfg = {"hidden_dims": (256, 256), "tanh_output": False, "action_scale": 1.0}

    model = MultiModalStudentPolicy(
        proprio_dim=num_prop,
        action_dim=action_dim,
        camera_resolution=camera_resolution,
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
        token_dim=128,
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="原始 PyTorch checkpoint")
    parser.add_argument("--torchscript", type=str, required=True, help="导出的 TorchScript 模型")
    parser.add_argument("--num_steps", type=int, default=10, help="测试步数")
    args = parser.parse_args()

    # 加载模型
    print("[INFO] 加载 PyTorch 模型...")
    pytorch_model, meta = load_pytorch_model(args.checkpoint)

    print("[INFO] 加载 TorchScript 模型...")
    ts_model = torch.jit.load(args.torchscript)
    ts_model.eval()

    # 配置
    num_prop = int(meta["num_prop"])
    prop_hist_len = int(meta.get("prop_hist_len", 1))
    depth_hist_len = int(meta.get("depth_hist_len", 4))
    camera_resolution = tuple(meta.get("camera_resolution", [58, 87]))
    mem_len = int(meta.get("sequence_length", 64))
    num_layers = 3
    token_dim = 128

    print(f"[INFO] 配置: prop_hist_len={prop_hist_len}, depth_hist_len={depth_hist_len}")

    # 初始化 mems - 使用空 mems (mem_len=0) 作为初始状态
    pytorch_mems = pytorch_model.temporal_model.reset_mems(1)
    # TorchScript 模型也使用空 mems 初始化，格式为 [B, L, 0, C]
    ts_mems = torch.zeros(1, num_layers, 0, token_dim)

    # 固定随机种子
    torch.manual_seed(42)

    print("\n" + "=" * 80)
    print("PyTorch vs TorchScript 对比验证")
    print("=" * 80)

    all_passed = True
    for i in range(args.num_steps):
        # 生成随机输入
        proprio = torch.randn(1, prop_hist_len * num_prop)
        depth = torch.randn(1, depth_hist_len, camera_resolution[0], camera_resolution[1])

        with torch.no_grad():
            # PyTorch 推理
            pytorch_action, _, pytorch_mems = pytorch_model.forward_step(proprio, depth, mems=pytorch_mems)

            # TorchScript 推理
            ts_action, ts_mems = ts_model(proprio, depth, ts_mems)

        # 对比
        mse = torch.mean((pytorch_action - ts_action) ** 2).item()
        max_diff = torch.max(torch.abs(pytorch_action - ts_action)).item()

        status = "PASS" if mse < 1e-10 else "FAIL"
        if mse >= 1e-10:
            all_passed = False

        print(f"[Step {i:2d}] MSE={mse:.2e}  MaxDiff={max_diff:.2e}  [{status}]")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 验证通过：TorchScript 导出正确，与 PyTorch 输出完全一致")
    else:
        print("✗ 验证失败：TorchScript 输出与 PyTorch 不一致，检查导出逻辑")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
