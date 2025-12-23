"""
Stateful (Batch-Aligned) Training Script

TransformerXL网络在监督训练时, 它的输入流是这样的:
输入batch0, batch1, batch2...  batch_i是不同环境同一段时间内教师模型与环境交互的切片
batch_i[j]与batch_i+1[j]必须是同一环境下连续的两片时间内教师模型与环境交互的切片
这样才可以训练transformerxl网络利用历史状态

因此：
batch_size必须等于num_envs, 并且batch0和batch1之间不能有时间片重叠
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warning] wandb not installed. Run `pip install wandb` to enable logging.")

# Ensure repo roots are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULES_ROOT = Path(PROJECT_ROOT) / "parkour_tasks" / "parkour_tasks" / "extreme_parkour_task" / "modules"


def load_symbol(module_path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"student_policy.{symbol}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{module_path} does not define {symbol}")
    return getattr(module, symbol)


JointPoseActionHead = load_symbol(MODULES_ROOT / "actionheads" / "joint_action_head.py", "JointPoseActionHead")
MultiModalFusionTransformer = load_symbol(
    MODULES_ROOT / "encoders" / "fusion_transformer.py", "MultiModalFusionTransformer"
)
TransformerXLTemporal = load_symbol(MODULES_ROOT / "temperal" / "txl.py", "TransformerXLTemporal")
DepthEncoder = load_symbol(MODULES_ROOT / "tokenizers" / "depth_encoder.py", "DepthEncoder")
ProprioEncoder = load_symbol(MODULES_ROOT / "tokenizers" / "proprio_encoder.py", "ProprioEncoder")


class SequenceAggregator:
    """
    [经过优化] 向量化版本：移除所有 Python for 循环，使用 Numpy 矩阵操作。
    解决 GPU 等待 CPU 数据的问题。
    """
    def __init__(
        self,
        num_envs: int,
        prop_hist_len: int,
        depth_hist_len: int,
        sequence_len: int,
        num_prop: int = 53,
        depth_shape: Tuple[int, int] = (58, 87)
    ) -> None:
        self.num_envs = num_envs
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.sequence_len = sequence_len
        self.num_prop = num_prop
        
        # 1. 历史 Buffer: 预分配内存，不再使用 deque
        self.prop_history = np.zeros((num_envs, prop_hist_len, num_prop), dtype=np.float32)
        self.depth_history = np.zeros((num_envs, depth_hist_len, *depth_shape), dtype=np.float32)

        # 2. 序列 Buffer: 预分配内存
        self.seq_prop = np.zeros((num_envs, sequence_len, prop_hist_len * num_prop), dtype=np.float32)
        self.seq_depth = np.zeros((num_envs, sequence_len, depth_hist_len, *depth_shape), dtype=np.float32)
        
        self.seq_action = None 
        self.seq_done = np.zeros((num_envs, sequence_len), dtype=bool)
        
        self.current_seq_step = 0

    def reset(self) -> None:
        self.prop_history.fill(0)
        self.depth_history.fill(0)
        self.current_seq_step = 0

    def push_step(self, obs_prop, depth_frame, teacher_actions, done):
        # --- 1. 更新历史 (整体左移) ---
        self.prop_history = np.roll(self.prop_history, -1, axis=1)
        self.depth_history = np.roll(self.depth_history, -1, axis=1)
        
        # 填入最新数据
        self.prop_history[:, -1, :] = obs_prop
        self.depth_history[:, -1, :, :] = depth_frame

        # --- 2. 存入序列 Buffer ---
        # Flatten Proprio: [Num_Envs, Hist_Len, Dim] -> [Num_Envs, Hist_Len * Dim]
        current_prop_flat = self.prop_history.reshape(self.num_envs, -1)
        
        idx = self.current_seq_step
        if self.seq_action is None:
             self.seq_action = np.zeros((self.num_envs, self.sequence_len, teacher_actions.shape[-1]), dtype=np.float32)

        self.seq_prop[:, idx] = current_prop_flat
        self.seq_depth[:, idx] = self.depth_history 
        self.seq_action[:, idx] = teacher_actions
        self.seq_done[:, idx] = done

        # --- 3. 处理 Done (批量清零) ---
        if np.any(done):
            self.prop_history[done] = 0
            self.depth_history[done] = 0

        # --- 4. 检查 Batch 是否完成 ---
        self.current_seq_step += 1
        if self.current_seq_step == self.sequence_len:
            batch = self._pack_batch()
            self.current_seq_step = 0 
            return batch
        
        return None

    def _pack_batch(self):
        return {
            "proprio": self.seq_prop.copy(),
            "depth": self.seq_depth.copy(),
            "actions": self.seq_action.copy(),
            "dones": self.seq_done.copy()
        }


class TeacherDatasetStreamer:
    """
    collect采集到的数据是[total_steps, num_envs, s&a], 而训练时需要[num_envs, sequence_len, s&a]的batch数据
    所以需要使用两个for循环, 重新排列数据
    """

    def __init__(
        self,
        dataset_dir: Path,
        sequence_len: int,
        prop_hist_len: int,
        depth_hist_len: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.meta = self._load_meta(dataset_dir)
        self.num_envs = int(self.meta["num_envs"])
        self.sequence_len = sequence_len
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.depth_dtype = self.meta.get("depth_dtype", "uint16")
        self.depth_scale = float(self.meta.get("depth_scale", 1000.0))
        self.camera_resolution = tuple(self.meta.get("camera_resolution", [64, 64]))
        shards_root = dataset_dir / "shards"
        self.shards: List[Path] = sorted(shards_root.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No dataset shards found in {shards_root}")
        self.aggregator = SequenceAggregator(
            num_envs=self.num_envs,
            prop_hist_len=prop_hist_len,
            depth_hist_len=depth_hist_len,
            sequence_len=sequence_len,
        )

    @staticmethod
    def _load_meta(dataset_dir: Path) -> Dict[str, object]:
        meta_path = dataset_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Dataset missing meta.json: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as meta_file:
            return json.load(meta_file)

    def iter_batches(self, max_sequences: Optional[int] = None) -> Iterator[Dict[str, np.ndarray]]:
        """对step_idx做循环, 每一次循环调用push_step(), 每sequence_len次循环会返回一个batch_data"""
        self.aggregator.reset()
        batches_yielded = 0

        for shard_path in self.shards:
            with np.load(shard_path, allow_pickle=False) as shard:
                obs_prop = shard["obs_prop"].astype(np.float32)
                actions = shard["action_teacher"].astype(np.float32)
                dones = shard["done"].astype(bool)
                depth = shard["depth"]
                num_steps = obs_prop.shape[0]

                for step_idx in range(num_steps):
                    depth_frame = self._convert_depth(depth[step_idx])
                    # Push step and check if a batch is ready
                    batch_data = self.aggregator.push_step(
                        obs_prop=obs_prop[step_idx],
                        depth_frame=depth_frame,
                        teacher_actions=actions[step_idx],
                        done=dones[step_idx].reshape(-1),
                    )

                    if batch_data is not None:
                        yield batch_data
                        # 注意：这里的 max_sequences 语义略有变化，变成 max_batches
                        batches_yielded += 1
                        if max_sequences is not None and batches_yielded >= max_sequences:
                            self.aggregator.reset()
                            return

        self.aggregator.reset()

    def _convert_depth(self, depth_np: np.ndarray) -> np.ndarray:
        if self.depth_dtype == "uint16":
            depth_np = depth_np.astype(np.float32) / self.depth_scale
        else:
            depth_np = depth_np.astype(np.float32)
        return depth_np


class MultiModalStudentPolicy(nn.Module):
    """Full student policy combining tokenizers, fusion transformer, temporal TXL, and action head."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        camera_resolution: Tuple[int, int],
        prop_hist_len: int,
        depth_hist_len: int,
        fusion_cfg: Dict[str, object],
        temporal_cfg: Dict[str, object],
        action_head_cfg: Dict[str, object],
        token_dim: int = 128,
    ) -> None:
        super().__init__()
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        height, width = camera_resolution

        self.proprio_encoder = ProprioEncoder(
            state_dim=proprio_dim,
            hist_len=prop_hist_len,
            token_dim=token_dim,
            hidden_dims=fusion_cfg.get("prop_hidden_dims", (256, 256)),
            dropout=fusion_cfg.get("prop_dropout", 0.1),
        )
        self.depth_encoder = DepthEncoder(
            in_frames=depth_hist_len,
            in_size=max(height, width),
            token_dim=token_dim,
            grid_size=fusion_cfg.get("grid_size", 4),
            dropout=fusion_cfg.get("depth_dropout", 0.1),
        )
        self.fusion_transformer = MultiModalFusionTransformer(
            token_dim=token_dim,
            num_layers=fusion_cfg.get("num_layers", 2),
            num_heads=fusion_cfg.get("num_heads", 4),
            mlp_ratio=fusion_cfg.get("mlp_ratio", 2.0),
            dropout=fusion_cfg.get("dropout", 0.1),
            attn_dropout=fusion_cfg.get("attn_dropout", 0.1),
            add_modality_embed=True,
            norm_first=True,
        )
        self.temporal_model = TransformerXLTemporal(
            d_model=token_dim,
            n_layer=temporal_cfg.get("num_layers", 3),
            n_head=temporal_cfg.get("num_heads", 4),
            d_inner=temporal_cfg.get("d_inner", 256),
            mem_len=temporal_cfg.get("mem_len", 64),
            dropout=temporal_cfg.get("dropout", 0.1),
            attn_dropout=temporal_cfg.get("attn_dropout", 0.1),
            norm_first=temporal_cfg.get("norm_first", True),
            clamp_len=temporal_cfg.get("clamp_len", None),
            use_rel_pos=temporal_cfg.get("use_rel_pos", True),
        )
        self.action_head = JointPoseActionHead(
            d_model=token_dim,
            action_dim=action_dim,
            hidden_dims=action_head_cfg.get("hidden_dims", (256, 256)),
            tanh_output=action_head_cfg.get("tanh_output", False),  # 应该使用激活函数吗？教师模型tanh_encoder_output = False，会输出>1的action
            action_scale=action_head_cfg.get("action_scale", 1.0),
        )

    def forward(self, proprio_seq: Tensor, depth_seq: Tensor) -> Tensor:
        """
        Args:
            proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
            depth_seq: Tensor[B, S, depth_hist_len, H, W]

        Returns:
            Predicted action means of shape [B, S, action_dim]
        """

        batch_size, seq_len, feat_dim = proprio_seq.shape
        prop_encoded = self.proprio_encoder(
            proprio_seq.reshape(batch_size * seq_len, feat_dim)
        )  # [B*S, 1, C]
        depth_encoded = self.depth_encoder(
            depth_seq.reshape(batch_size * seq_len, depth_seq.size(2), depth_seq.size(3), depth_seq.size(4))
        )  # [B*S, T, C]
        fused = self.fusion_transformer(prop_encoded, depth_encoded)
        fused_seq = fused["all_pooled"].reshape(batch_size, seq_len, -1)
        temporal_out, _ = self.temporal_model(
            fused_seq,
            mems=None,
            causal_mask=True,
            return_mems=False,
        )
        actions = self.action_head.forward_sequence(temporal_out)["mean"]
        return actions

    def forward_with_mems(
        self,
        proprio_seq: Tensor,
        depth_seq: Tensor,
        mems: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass with segment recurrence memory support.
        
        Args:
            proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
            depth_seq: Tensor[B, S, depth_hist_len, H, W]
            mems: Optional list of memory tensors from previous segment (should be detached)
        
        Returns:
            actions: Predicted action means of shape [B, S, action_dim]
            new_mems: List of new memory tensors for next segment
        """
        batch_size, seq_len, feat_dim = proprio_seq.shape
        
        # 1. Encode proprio and depth
        prop_encoded = self.proprio_encoder(
            proprio_seq.reshape(batch_size * seq_len, feat_dim)
        )  # [B*S, 1, C]
        depth_encoded = self.depth_encoder(
            depth_seq.reshape(batch_size * seq_len, depth_seq.size(2), depth_seq.size(3), depth_seq.size(4))
        )  # [B*S, T, C]
        
        # 2. Multi-modal fusion
        fused = self.fusion_transformer(prop_encoded, depth_encoded)
        fused_seq = fused["all_pooled"].reshape(batch_size, seq_len, -1)
        
        # 3. Temporal modeling with memory
        temporal_out, new_mems = self.temporal_model(
            fused_seq,
            mems=mems,           # Pass previous segment's mems (should be detached by caller)
            causal_mask=True,
            return_mems=True,    # Return new mems for next segment
        )
        
        # 4. Action head
        actions = self.action_head.forward_sequence(temporal_out)["mean"]
        return actions, new_mems

    def forward_step(
        self,
        proprio: Tensor,
        depth: Tensor,
        mems: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Single-step forward for online inference with memory.
        
        Args:
            proprio: Tensor[B, prop_hist_len * proprio_dim] - flattened proprio history
            depth: Tensor[B, depth_hist_len, H, W] - stacked depth frames
            mems: Optional list of memory tensors from previous step
        
        Returns:
            actions: Predicted actions of shape [B, action_dim]
            new_mems: List of new memory tensors for next step
        """
        batch_size = proprio.shape[0]
        
        # Add sequence dimension S=1
        proprio_seq = proprio.unsqueeze(1)  # [B, 1, feat_dim]
        depth_seq = depth.unsqueeze(1)      # [B, 1, depth_hist_len, H, W]
        
        # Use forward_with_mems
        actions_seq, new_mems = self.forward_with_mems(proprio_seq, depth_seq, mems=mems)
        
        # Remove sequence dimension
        actions = actions_seq.squeeze(1)
        print("debug: using mem in dagger.")  # [B, action_dim]
        return actions, new_mems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train new transformer student policy from collected teacher datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to collect.py output directory.")
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device (e.g., cuda:0 or cpu).")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of passes over the dataset.")
    # parser.add_argument("--batch_size", type=int, default=8)  batch_size需要等于num_envs!!!
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence_length = mem_len 是一般transformerxl网络的默认实现")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length (in steps) for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="Number of stacked depth frames per sample.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold (L2 norm).")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging training metrics.")
    parser.add_argument("--max_sequences_per_epoch", type=int, default=None, help="Optional cap on sequences per epoch.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to store checkpoints (defaults to dataset dir).")
    # parser.add_argument("--resume", type=str, default=None)  已被student_checkpoint代替
    # wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="offline-BC", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (defaults to auto-generated).")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (team/username).")
    return parser.parse_args()


def build_student_from_dataset(
    streamer: TeacherDatasetStreamer,
    prop_hist_len: int,
    depth_hist_len: int,
) -> Tuple[MultiModalStudentPolicy, int]:
    meta = streamer.meta
    proprio_dim = int(meta.get("num_prop", 0))
    action_dim = int(meta.get("action_dim", 0))
    if proprio_dim <= 0 or action_dim <= 0:
        infer_prop, infer_action = infer_dataset_dims(streamer.shards[0])
        proprio_dim = infer_prop
        action_dim = infer_action
        meta.setdefault("num_prop", proprio_dim)
        meta.setdefault("action_dim", action_dim)
    camera_resolution = tuple(meta.get("camera_resolution", [64, 64]))
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
        "mem_len": 64,
        "dropout": 0.1,
        "attn_dropout": 0.1,
    }
    action_head_cfg = {
        "hidden_dims": (256, 256),
        "tanh_output": False,   # 教师模型tanh_encoder_output = False，会输出>1的action
        "action_scale": 1,  # 教师模型没有使用action_sacle
    }
    model = MultiModalStudentPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
        token_dim=128,
    )
    return model, action_dim


def infer_dataset_dims(shard_path: Path) -> Tuple[int, int]:
    with np.load(shard_path, allow_pickle=False) as shard:
        obs_prop = shard["obs_prop"]
        action_teacher = shard["action_teacher"]
        return int(obs_prop.shape[-1]), int(action_teacher.shape[-1])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    meta: Dict[str, object],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[checkpoint] Saved to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, int]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    epoch = int(payload.get("epoch", 0))
    global_step = int(payload.get("global_step", 0))
    print(f"[checkpoint] Resumed from {path} (epoch={epoch}, global_step={global_step})")
    return epoch, global_step


def run_training() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else dataset_dir / "student_policy"
    streamer = TeacherDatasetStreamer(
        dataset_dir=dataset_dir,
        sequence_len=args.sequence_length,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
    )
    model, _ = build_student_from_dataset(streamer, args.prop_hist_len, args.depth_hist_len)
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    if args.student_checkpoint:
        resume_path = Path(args.student_checkpoint).expanduser().resolve()
        start_epoch, global_step = load_checkpoint(resume_path, model, optimizer)

    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("[warning] --wandb flag set but wandb is not installed. Skipping wandb logging.")
    
    if use_wandb:
        wandb_config = {
            "dataset": str(dataset_dir),
            "num_epochs": args.num_epochs,
            "sequence_length": args.sequence_length,
            "prop_hist_len": args.prop_hist_len,
            "depth_hist_len": args.depth_hist_len,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "device": str(device),
            "num_envs": streamer.num_envs,
            "proprio_dim": streamer.meta.get("num_prop", 0),
            "action_dim": streamer.meta.get("action_dim", 0),
            "camera_resolution": streamer.meta.get("camera_resolution", [64, 64]),
        }
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=wandb_config,
            resume="allow" if args.student_checkpoint else None,
        )
        # Log model architecture
        wandb.watch(model, log="gradients", log_freq=args.log_interval)
        print(f"[wandb] Initialized: project={args.wandb_project}, run={wandb.run.name}")

    print(
        f"[info] Starting training for {args.num_epochs} epochs "
        f"on dataset {dataset_dir} using device {device}."
    )

    training_start_time = time.time()
    total_sequences = 0

    try:
        for epoch in range(start_epoch, args.num_epochs):
            epoch_start = time.time()
            model.train()
            
            # Epoch-level metrics collectors
            epoch_losses = []
            epoch_diff_rmses = []
            epoch_rel_rmses = []
            epoch_grad_norms = []
            epoch_grad_norm_maxs = []
            epoch_teacher_action_rms = []
            epoch_teacher_action_abs_max = []
            
            num_updates = 0
            epoch_sequences = 0

            # 直接迭代 Batch (无需再组装 samples)
            for batch_data in streamer.iter_batches(max_sequences=args.max_sequences_per_epoch):
                batch_start = time.time()
                metrics = train_batch(model, optimizer, batch_data, device, args.grad_clip)
                batch_time = time.time() - batch_start

                # Accumulate metrics
                epoch_losses.append(metrics["loss"])
                epoch_diff_rmses.append(metrics["diff_rmse"])
                epoch_rel_rmses.append(metrics["rel_rmse"])
                epoch_grad_norms.append(metrics["grad_norm"])
                epoch_grad_norm_maxs.append(metrics["grad_norm_max"])
                epoch_teacher_action_rms.append(metrics["teacher_action_rms"])
                epoch_teacher_action_abs_max.append(metrics["teacher_action_abs_max"])
                
                num_updates += 1
                global_step += 1
                epoch_sequences += streamer.num_envs  # Each batch contains num_envs sequences
                total_sequences += streamer.num_envs

                # Log step metrics to wandb
                if use_wandb:
                    elapsed_time = time.time() - training_start_time
                    sequences_per_sec = streamer.num_envs / max(batch_time, 1e-6)
                    # Compute running loss_std from epoch_losses collected so far
                    loss_std = np.std(epoch_losses) if len(epoch_losses) > 1 else 0.0
                    
                    wandb.log({
                        # train/ metrics
                        "train/loss": metrics["loss"],
                        "train/loss_std": loss_std,
                        "train/diff_rmse": metrics["diff_rmse"],
                        "train/rel_rmse": metrics["rel_rmse"],
                        "train/grad_norm": metrics["grad_norm"],
                        "train/grad_norm_max": metrics["grad_norm_max"],
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        # teacher/ metrics
                        "teacher/action_rms": metrics["teacher_action_rms"],
                        "teacher/action_abs_max": metrics["teacher_action_abs_max"],
                        # perf/ metrics
                        "perf/sequences_per_sec": sequences_per_sec,
                        "perf/total_sequences": total_sequences,
                        # time/ metrics
                        "time/elapsed_s": elapsed_time,
                        # progress/ metrics
                        "progress/epoch": epoch,
                        "progress/global_step": global_step,
                        "progress/sequences_this_epoch": epoch_sequences,
                    }, step=global_step)

                if args.log_interval > 0 and num_updates % args.log_interval == 0:
                    avg_loss = np.mean(epoch_losses)
                    print(f"[epoch {epoch}] step {global_step} | updates={num_updates} | avg_loss={avg_loss:.6f}")

            epoch_time = time.time() - epoch_start
            
            # Compute epoch-level statistics
            epoch_loss_mean = np.mean(epoch_losses) if epoch_losses else 0.0
            epoch_loss_std = np.std(epoch_losses) if epoch_losses else 0.0
            epoch_loss_min = np.min(epoch_losses) if epoch_losses else 0.0
            epoch_loss_max = np.max(epoch_losses) if epoch_losses else 0.0
            
            print(f"[epoch {epoch}] completed in {epoch_time:.1f}s | avg_loss={epoch_loss_mean:.6f}")

            # Log epoch-level metrics to wandb
            if use_wandb:
                wandb.log({
                    # epoch/ summary metrics
                    "epoch/loss_mean": epoch_loss_mean,
                    "epoch/loss_std": epoch_loss_std,
                    "epoch/loss_min": epoch_loss_min,
                    "epoch/loss_max": epoch_loss_max,
                    "epoch/diff_rmse_mean": np.mean(epoch_diff_rmses) if epoch_diff_rmses else 0.0,
                    "epoch/rel_rmse_mean": np.mean(epoch_rel_rmses) if epoch_rel_rmses else 0.0,
                    "epoch/grad_norm_mean": np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0,
                    "epoch/duration_s": epoch_time,
                    "epoch/sequences_total": epoch_sequences,
                    "epoch/sequences_per_sec": epoch_sequences / max(epoch_time, 1e-6),
                }, step=global_step)

            if epoch % 100 == 99:
                ckpt_path = save_dir / f"student_epoch_{epoch:04d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, epoch + 1, global_step, streamer.meta)
                
                # Log checkpoint to wandb
                if use_wandb:
                    wandb.save(str(ckpt_path))

    finally:
        # Ensure wandb is properly closed
        if use_wandb:
            wandb.finish()
            print("[wandb] Run finished.")


def train_batch(
    model: MultiModalStudentPolicy,
    optimizer: torch.optim.Optimizer,
    batch_data: Dict[str, np.ndarray],
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    """
    Train a single batch and return detailed metrics.
    
    Returns:
        Dict containing:
        - loss: MSE loss value
        - diff_rmse: Root mean square error between predictions and teacher actions
        - rel_rmse: Relative RMSE (normalized by teacher action RMS)
        - grad_norm: L2 norm of gradients
        - grad_norm_max: Maximum gradient norm across parameters
        - teacher_action_rms: RMS of teacher actions
        - teacher_action_abs_max: Max absolute value of teacher actions
    """
    proprio = torch.from_numpy(batch_data["proprio"]).to(device)
    depth = torch.from_numpy(batch_data["depth"]).to(device)
    teacher_actions = torch.from_numpy(batch_data["actions"]).to(device)

    predictions = model(proprio, depth)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(predictions, teacher_actions)
    
    # Compute additional metrics (before backward to avoid extra computation)
    with torch.no_grad():
        diff = predictions - teacher_actions
        diff_rmse = torch.sqrt(torch.mean(diff ** 2)).item()
        teacher_action_rms = torch.sqrt(torch.mean(teacher_actions ** 2)).item()
        teacher_action_abs_max = torch.max(torch.abs(teacher_actions)).item()
        rel_rmse = diff_rmse / max(teacher_action_rms, 1e-8)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Compute gradient norms before clipping
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.data.norm(2).item())
    
    grad_norm = np.sqrt(sum(g ** 2 for g in grad_norms)) if grad_norms else 0.0
    grad_norm_max = max(grad_norms) if grad_norms else 0.0
    
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    return {
        "loss": float(loss.item()),
        "diff_rmse": diff_rmse,
        "rel_rmse": rel_rmse,
        "grad_norm": grad_norm,
        "grad_norm_max": grad_norm_max,
        "teacher_action_rms": teacher_action_rms,
        "teacher_action_abs_max": teacher_action_abs_max,
    }


if __name__ == "__main__":
    run_training()
