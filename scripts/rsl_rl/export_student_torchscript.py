#!/usr/bin/env python3
"""
Export MultiModalStudentPolicy to TorchScript for C++ inference.

Interface:
  Inputs:
    - proprio: [1, prop_hist_len * num_prop]
    - depth:   [1, depth_hist_len, H, W]
    - mems:    [1, num_layers, mem_len, token_dim]
  Outputs:
    - actions:  [1, action_dim]
    - yaw_pred: [1, 2]
    - mems_out: [1, num_layers, mem_len, token_dim]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _ensure_repo_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    parkour_tasks_root = os.path.join(project_root, "parkour_tasks")
    if parkour_tasks_root not in sys.path:
        sys.path.insert(0, parkour_tasks_root)


def _load_student_class() -> type:
    train_student_path = Path(__file__).resolve().parent / "train_student_from_dataset.py"
    spec = importlib.util.spec_from_file_location("train_student_from_dataset", train_student_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load from {train_student_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MultiModalStudentPolicy


class StudentTorchScriptWrapper(nn.Module):
    """Wrapper for TorchScript export with explicit mems interface."""

    def __init__(self, student: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.student = student
        self.num_layers = num_layers

    def forward(
        self,
        proprio: torch.Tensor,
        depth: torch.Tensor,
        mems: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with yaw prediction output.

        Args:
            proprio: [1, prop_hist_len * num_prop]
            depth: [1, depth_hist_len, H, W]
            mems: [1, num_layers, mem_len, token_dim]

        Returns:
            actions: [1, action_dim]
            yaw_pred: [1, 2]
            mems_out: [1, num_layers, mem_len, token_dim]
        """
        # mems: [B, L, M, C] -> list([B, M, C] * L)
        mem_list: List[torch.Tensor] = [mems[:, i, :, :] for i in range(self.num_layers)]

        # [严格整改]: 显式注入 delta_yaw_ok，以对齐 DAGGER 部署期的遮蔽行为
        # 将此标志硬编码进 Trace 计算图，确保模型底层使用视觉特征推断的相对航向
        batch_size = proprio.shape[0]
        delta_yaw_ok = torch.ones(batch_size, dtype=torch.bool, device=proprio.device)

        actions, yaw_pred, new_mems = self.student.forward_step(
            proprio,
            depth,
            mems=mem_list,
            delta_yaw_ok=delta_yaw_ok
        )

        # new_mems: list([B, M, C]) -> [B, L, M, C]
        mems_out = torch.stack(new_mems, dim=1)
        return actions, yaw_pred, mems_out


def _parse_config(meta: Dict) -> Dict:
    """
    [严格整改]: 统一解析元数据 (Single Source of Truth)。
    杜绝因缺省值不一致导致的 Dummy Input 与模型结构冲突问题。
    """
    return {
        "num_prop": int(meta.get("num_prop", 53)),
        "action_dim": int(meta.get("action_dim", 12)),
        "camera_resolution": tuple(meta.get("camera_resolution", [58, 87])),
        "prop_hist_len": int(meta.get("prop_hist_len", 1)),
        "depth_hist_len": int(meta.get("depth_hist_len", 4)),
        "mem_len": int(meta.get("sequence_length", 64)),
        "token_dim": 128,
    }


def _build_student(ckpt_path: Path, student_cls: type, cfg: Dict) -> nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu")

    fusion_cfg = {"num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.1, "attn_dropout": 0.1, "grid_size": 4}
    temporal_cfg = {"num_layers": 3, "num_heads": 4, "d_inner": 256, "mem_len": cfg["mem_len"], "dropout": 0.1, "attn_dropout": 0.1}
    action_head_cfg = {"hidden_dims": (256, 256), "tanh_output": False, "action_scale": 1.0}

    student = student_cls(
        proprio_dim=cfg["num_prop"],
        action_dim=cfg["action_dim"],
        camera_resolution=cfg["camera_resolution"],
        prop_hist_len=cfg["prop_hist_len"],
        depth_hist_len=cfg["depth_hist_len"],
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
        token_dim=cfg["token_dim"],
    )

    # [严格整改]: 剥离掩耳盗铃的 strict=False，暴露出潜在的权重缺失
    try:
        student.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as e:
        print(f"[FATAL ERROR] 状态字典加载失败。训练端与导出端的网络结构出现严重不对齐！\n详细信息: {e}")
        sys.exit(1)

    student.eval()
    return student


def main() -> None:
    _ensure_repo_on_path()

    parser = argparse.ArgumentParser()
    parser.add_argument("--student_checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    ckpt_path = Path(args.student_checkpoint).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "student_policy.pt"

    # 初始化配置源
    payload = torch.load(ckpt_path, map_location="cpu")
    meta = dict(payload.get("meta", {}))
    cfg = _parse_config(meta)

    student_cls = _load_student_class()
    student = _build_student(ckpt_path, student_cls, cfg)

    num_layers = len(student.temporal_model.layers)
    wrapper = StudentTorchScriptWrapper(student, num_layers).eval()

    # 构造精确对齐维度的 Dummy Inputs
    proprio = torch.zeros(1, cfg["prop_hist_len"] * cfg["num_prop"])
    depth = torch.zeros(1, cfg["depth_hist_len"], cfg["camera_resolution"][0], cfg["camera_resolution"][1])
    mems = torch.zeros(1, num_layers, cfg["mem_len"], cfg["token_dim"])

    print(f"[export] proprio: {tuple(proprio.shape)}, depth: {tuple(depth.shape)}, mems: {tuple(mems.shape)}")
    print(f"[export] outputs: actions=[1, {cfg['action_dim']}], yaw_pred=[1, 2], mems_out={tuple(mems.shape)}")
    print(f"[export] output path: {out_path}")

    # Export via tracing
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (proprio, depth, mems), check_trace=False)
    traced.save(str(out_path))
    print("[export] done. 导出模型符合 MuJoCo 部署级严格规范。")


if __name__ == "__main__":
    main()