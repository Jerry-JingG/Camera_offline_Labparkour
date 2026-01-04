#!/usr/bin/env python3
"""
Export DAGGER student (MultiModalStudentPolicy + Transformer-XL mems) to an ONNX model
that is easy to run from rl_sar (C++ / onnxruntime).

The exported ONNX interface is stateless except for explicit mems:
  Inputs:
    - proprio:  [1, prop_hist_len * num_prop]
    - depth:    [1, depth_hist_len, H, W]
    - mems:     [1, num_layers, mem_len, token_dim]
  Outputs:
    - actions:  [1, action_dim]
    - mems_out: [1, num_layers, mem_len, token_dim]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


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
        raise ImportError(f"Unable to load MultiModalStudentPolicy from {train_student_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "MultiModalStudentPolicy"):
        raise AttributeError(f"{train_student_path} does not define MultiModalStudentPolicy")
    return module.MultiModalStudentPolicy  # type: ignore[return-value]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DAGGER student to rl_sar-friendly ONNX with mems.")
    p.add_argument("--student_checkpoint", type=str, required=True, help="Path to student_epoch_*.pt checkpoint.")
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory (e.g., rl_sar/policy/go2/parkour_student_dagger).",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    p.add_argument(
        "--depth_resize",
        type=int,
        default=64,
        help=(
            "在导出图中先把 depth 双线性 resize 到 NxN（默认 64），"
            "用于规避 DepthEncoder 内部 adaptive_avg_pool2d 的 ONNX 不支持问题；"
            "设为 0 表示不做 resize（可能导出失败）。"
        ),
    )
    p.add_argument("--verbose", action="store_true", default=False, help="Verbose ONNX export.")
    return p.parse_args()


class _StudentMemsWrapper(torch.nn.Module):
    def __init__(self, student: torch.nn.Module, num_layers: int, depth_resize: int) -> None:
        super().__init__()
        self.student = student
        self.num_layers = int(num_layers)
        self.depth_resize = int(depth_resize)

    def forward(
        self,
        proprio: torch.Tensor,
        depth: torch.Tensor,
        mems: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE:
        # DepthEncoder 内部在 H/W 与 grid_size(=4) 不整除时会走 adaptive_avg_pool2d，
        # 但 PyTorch 的 ONNX 导出仅支持“输出尺寸是输入尺寸因子”的 adaptive pooling。
        # 因此这里先把 depth resize 到一个能让卷积输出变成 4x4 的固定尺寸（默认 64x64），
        # 从而避免触发 adaptive_avg_pool2d 分支。
        if self.depth_resize > 0:
            depth = F.interpolate(
                depth,
                size=(self.depth_resize, self.depth_resize),
                mode="bilinear",
                align_corners=False,
            )
        # mems: [B, L, M, C] -> list([B, M, C] * L)
        mem_list = [mems[:, i, :, :] for i in range(self.num_layers)]
        actions, _yaw_pred, new_mems = self.student.forward_step(proprio, depth, mems=mem_list)
        # new_mems: list([B, M, C]) -> [B, L, M, C]
        mems_out = torch.stack(new_mems, dim=1)
        return actions, mems_out


def _build_student_from_checkpoint(
    ckpt_path: Path,
    student_cls: type,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Expected a dict checkpoint with key 'model_state_dict'.")
    meta: Dict[str, object] = dict(payload.get("meta", {}))

    num_prop = int(meta["num_prop"])
    action_dim = int(meta["action_dim"])
    camera_resolution = tuple(meta.get("camera_resolution", [58, 87]))
    prop_hist_len = int(meta.get("prop_hist_len", 3))
    depth_hist_len = int(meta.get("depth_hist_len", 4))
    mem_len = int(meta.get("sequence_length", 64))

    fusion_cfg = {
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.1,
        "attn_dropout": 0.1,
        "grid_size": 4,
    }
    temporal_cfg = {
        "num_layers": 3,
        "num_heads": 4,
        "d_inner": 256,
        "mem_len": mem_len,
        "dropout": 0.1,
        "attn_dropout": 0.1,
    }
    action_head_cfg = {"hidden_dims": (256, 256), "tanh_output": False, "action_scale": 1.0}

    student = student_cls(
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
    student.load_state_dict(payload["model_state_dict"])
    student.eval()
    return student, meta


def main() -> None:
    _ensure_repo_on_path()
    args = _parse_args()

    ckpt_path = Path(args.student_checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "student_policy.onnx"

    student_cls = _load_student_class()
    student, meta = _build_student_from_checkpoint(ckpt_path, student_cls)

    num_prop = int(meta["num_prop"])
    action_dim = int(meta["action_dim"])
    camera_resolution = tuple(meta.get("camera_resolution", [58, 87]))
    prop_hist_len = int(meta.get("prop_hist_len", 3))
    depth_hist_len = int(meta.get("depth_hist_len", 4))
    mem_len = int(meta.get("sequence_length", 64))
    token_dim = 128
    num_layers = len(student.temporal_model.layers)

    height, width = int(camera_resolution[0]), int(camera_resolution[1])

    wrapper = _StudentMemsWrapper(student, num_layers=num_layers, depth_resize=int(args.depth_resize)).eval()

    # fixed-shape dummy inputs for a single env (B=1)
    proprio = torch.zeros(1, prop_hist_len * num_prop, dtype=torch.float32)
    depth = torch.zeros(1, depth_hist_len, height, width, dtype=torch.float32)
    mems = torch.zeros(1, num_layers, mem_len, token_dim, dtype=torch.float32)

    print(
        "[export] input shapes:"
        f" proprio={tuple(proprio.shape)}, depth={tuple(depth.shape)}, mems={tuple(mems.shape)}"
    )
    print(f"[export] output file: {out_path}")
    print(f"[export] action_dim={action_dim}, num_layers={num_layers}, mem_len={mem_len}, token_dim={token_dim}")
    if int(args.depth_resize) > 0:
        print(f"[export] depth_resize={int(args.depth_resize)} (export graph will resize depth to {int(args.depth_resize)}x{int(args.depth_resize)})")
    else:
        print("[export] depth_resize=0 (no resize; export may fail if DepthEncoder uses adaptive_avg_pool2d)")

    # Export ONNX
    torch.onnx.export(
        wrapper,
        (proprio, depth, mems),
        str(out_path),
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        verbose=bool(args.verbose),
        input_names=["proprio", "depth", "mems"],
        output_names=["actions", "mems_out"],
        dynamic_axes={},
    )
    print("[export] done.")


if __name__ == "__main__":
    main()
