"""
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

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warning] wandb not installed. Run `pip install wandb` to enable logging.")

from utils.dropout_manager import CameraDropoutManager

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
    prop_histories和depth_histories用于聚合fusion transformer需要的聚合历史输入
    我对prop_histories和depth_histories初始化时进行了填零操作, 这样第一个环境步transformerxl就可以输出action
    seq_prop[i]存储了第i个环境的感知观测序列
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

        # 1. History Buffer: 行为类似deque, 不过现在使用np.roll实现
        self.prop_histories = torch.zeros((num_envs, prop_hist_len, num_prop), dtype=torch.float32)
        self.depth_histories = torch.zeros((num_envs, depth_hist_len, *depth_shape), dtype=torch.float32)

        # 2. Sequence Buffer: 预分配内存，用于存储一个完整的 Sequence Batch
        self.seq_prop = torch.zeros((num_envs, sequence_len, prop_hist_len * num_prop), dtype=torch.float32)
        self.seq_depth = torch.zeros((num_envs, sequence_len, depth_hist_len, *depth_shape), dtype=torch.float32)

        self.seq_action = None
        self.seq_done = torch.zeros((num_envs, sequence_len), dtype=torch.bool)

        self.current_seq_step = 0

    def reset(self) -> None:
        self.prop_histories.zero_()
        self.depth_histories.zero_()

        self.seq_prop.zero_()
        self.seq_depth.zero_()
        self.seq_action = None
        self.seq_done.zero_()

        self.current_seq_step = 0

    def push_step(self, obs_prop: np.ndarray, depth_frame: np.ndarray, teacher_actions: np.ndarray, done: np.ndarray):
        """
        接收 Numpy 数据，转为 Tensor 并更新历史 buffer 和 sequence buffer。
        """
        # --- 0. 转换为 Tensor ---
        obs_prop_t = torch.from_numpy(obs_prop)
        depth_frame_t = torch.from_numpy(depth_frame)
        teacher_actions_t = torch.from_numpy(teacher_actions)
        done_t = torch.from_numpy(done)

        # --- 1. 更新历史 (整体左移) ---
        self.prop_histories = torch.roll(self.prop_histories, -1, dims=1)
        self.depth_histories = torch.roll(self.depth_histories, -1, dims=1)

        # 填入最新数据
        self.prop_histories[:, -1, :] = obs_prop_t
        self.depth_histories[:, -1, :, :] = depth_frame_t

        # --- 2. 填入 Sequence Buffer ---
        idx = self.current_seq_step

        # Lazy Init for actions
        if self.seq_action is None:
            action_dim = teacher_actions.shape[-1]
            self.seq_action = torch.zeros((self.num_envs, self.sequence_len, action_dim), dtype=torch.float32)

        # Flatten Proprio: [Num_Envs, Hist_Len, num_prop] -> [Num_Envs, Hist_Len * num_prop]
        current_prop_flat = self.prop_histories.reshape(self.num_envs, -1)

        self.seq_prop[:, idx] = current_prop_flat
        self.seq_depth[:, idx] = self.depth_histories
        self.seq_action[:, idx] = teacher_actions_t
        self.seq_done[:, idx] = done_t

        # --- 3. 处理 Done (批量清零) ---
        if done_t.any():
            self.prop_histories[done_t] = 0.0
            self.depth_histories[done_t] = 0.0

        # --- 4. 检查 Batch 是否完成 ---
        self.current_seq_step += 1
        if self.current_seq_step == self.sequence_len:
            batch = self._pack_batch()
            self.current_seq_step = 0
            return batch

        return None

    def _pack_batch(self):
        # 返回 Tensor 副本，防止下一轮循环修改 buffer 影响 dataloader 队列
        return {
            "proprio": self.seq_prop.clone(),
            "depth": self.seq_depth.clone(),
            "actions": self.seq_action.clone(),
            "dones": self.seq_done.clone()
        }


class TeacherDatasetStreamer:
    """
    collect采集到的数据是[total_steps, num_envs, s&a], 而训练时需要[num_envs, sequence_len, s&a]的batch数据
    """

    def __init__(
        self,
        dataset_dir: Path,
        sequence_len: int,
        prop_hist_len: int,
        depth_hist_len: int,
        use_dropout: bool = False,
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

        # Camera Dropout Augmentation
        self.dropout_manager = None
        if use_dropout:
            print("[Streamer] Training-time Camera Dropout Augmentation: ENABLED")
            dt = float(self.meta.get("step_dt", 0.02))
            # 我们使用 CPU 版本的 tensor 进行增强，因为 dataloader 运行在 CPU 上
            self.dropout_manager = CameraDropoutManager(
                num_envs=self.num_envs,
                device=torch.device("cpu"),
                dt=dt,
                prob_start_offline=0.0,
                online_duration_range=(2.0, 20.0),
                offline_duration_range=(1.0, 10.0)
            )

    @staticmethod
    def _load_meta(dataset_dir: Path) -> Dict[str, object]:
        meta_path = dataset_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Dataset missing meta.json: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as meta_file:
            return json.load(meta_file)

    def iter_batches(self, max_sequences: Optional[int] = None) -> Iterator[Dict[str, Tensor]]:
        """
        迭代器：读取 Shard -> Step by Step 推入 Aggregator -> Yield Batch
        """
        self.aggregator.reset()
        batches_yielded = 0

        prev_dones = torch.zeros(self.num_envs, dtype=torch.bool)
        for shard_path in self.shards:
            with np.load(shard_path, allow_pickle=False) as shard:
                obs_prop = shard["obs_prop"].astype(np.float32)
                actions = shard["action_teacher"].astype(np.float32)
                dones = shard["done"].astype(bool)
                depth = shard["depth"]
                num_steps = obs_prop.shape[0]

                for step_idx in range(num_steps):
                    current_obs_prop = obs_prop[step_idx].copy()
                    depth_frame = self._convert_depth(depth[step_idx])

                    if self.dropout_manager is not None:
                        self.dropout_manager.reset_env(prev_dones)
                        depth_tensor = torch.from_numpy(depth_frame)
                        prop_tensor = torch.from_numpy(current_obs_prop)

                        self.dropout_manager.update(depth_image=depth_tensor, obs_prop=prop_tensor)
                        depth_frame = depth_tensor.numpy()
                        current_obs_prop = prop_tensor.numpy()

                    prev_dones = torch.from_numpy(dones[step_idx].reshape(-1))

                    # Push step and check if a batch is ready
                    batch_data = self.aggregator.push_step(
                        obs_prop=current_obs_prop,
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

        # self.aggregator.reset()

    def _convert_depth(self, depth_np: np.ndarray) -> np.ndarray:
        if self.depth_dtype == "uint16":
            depth_np = (depth_np.astype(np.float32) / self.depth_scale) - 0.5
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
        Original simple forward (stateless).
        Args:
            proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
            depth_seq: Tensor[B, S, depth_hist_len, H, W]
        """
        actions, _ = self.forward_with_mems(proprio_seq, depth_seq, mems=None)
        return actions

    def forward_with_mems(
        self,
        proprio_seq: Tensor,
        depth_seq: Tensor,
        mems: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass with segment recurrence memory support (TBPTT).

        Args:
            proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
            depth_seq: Tensor[B, S, depth_hist_len, H, W]
            mems: Optional list of memory tensors from previous segment (should be detached)

        Returns:
            actions: Predicted action means of shape [B, S, action_dim]
            new_mems: List of new memory tensors for next segment
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

        # Temporal modeling with memory
        temporal_out, new_mems = self.temporal_model(
            fused_seq,
            mems=mems,           # Pass previous segment's mems
            causal_mask=True,
            return_mems=True,    # Return new mems for next segment
        )
        actions = self.action_head.forward_sequence(temporal_out)["mean"]
        return actions, new_mems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train new transformer student policy from collected teacher datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to collect.py output directory.")
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device (e.g., cuda:0 or cpu).")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of passes over the dataset.")
    # parser.add_argument("--batch_size", type=int, default=8)  batch_size implicitly equals num_envs in dataset we collected
    parser.add_argument("--sequence_length", type=int, default=64, help="TransformerXL segment length during training (should match mem_len).")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length (in steps) for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="Number of stacked depth frames per sample.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold (L2 norm).")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging training metrics.")
    parser.add_argument("--max_sequences_per_epoch", type=int, default=None, help="Optional cap on sequences per epoch.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to store checkpoints (defaults to dataset dir).")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="offline-BC", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (defaults to auto-generated).")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (team/username).")

    # Camera Dropout augmentation
    parser.add_argument("--use_dropout", action="store_true", default=False, help="Enable training-time random camera dropout.")

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
        use_dropout=args.use_dropout
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

        if streamer.dropout_manager is not None:
            streamer.dropout_manager.reset()

        # Initialize memory for TransformerXL segment recurrence
        mems = None  # Will be populated after first batch

        # 直接迭代 Batch (无需再组装 samples)
        for batch_data in streamer.iter_batches(max_sequences=args.max_sequences_per_epoch):
            batch_start = time.time()
            metrics, mems = train_batch(model, optimizer, batch_data, device, args.grad_clip, mems=mems)
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
                    # dropout/ metrics
                    "dropout/rate": batch_data.get("dropout_rate", 0.0),
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
                dropout_rate = batch_data.get("dropout_rate", 0.0)
                print(f"[epoch {epoch}] step {global_step} | updates={num_updates} | avg_loss={avg_loss:.6f} | dropout={dropout_rate:.1%}")

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

    # Ensure wandb is properly closed
    if use_wandb:
        wandb.finish()
        print("[wandb] Run finished.")


def train_batch(
    model: MultiModalStudentPolicy,
    optimizer: torch.optim.Optimizer,
    batch_data: Dict[str, Tensor],
    device: torch.device,
    grad_clip: float,
    mems: Optional[List[Tensor]] = None,
) -> Tuple[Dict[str, float], Optional[List[Tensor]]]:
    """
    Train a single batch with stateful memory management.

    Args:
        mems: Memory tensors from the previous batch (detached).
    Returns:
        metrics: Dict of loss and other stats.
        new_mems: Memory tensors for the next batch (detached).
    """
    proprio = batch_data["proprio"].to(device)
    depth = batch_data["depth"].to(device)
    teacher_actions = batch_data["actions"].to(device)
    dones = batch_data["dones"].to(device)  # [B, S]

    # Use forward_with_mems for segment recurrence training
    predictions, new_mems = model.forward_with_mems(proprio, depth, mems=mems)

    loss = torch.nn.functional.mse_loss(predictions, teacher_actions)

    # Compute additional metrics
    with torch.no_grad():
        diff = predictions - teacher_actions
        diff_rmse = torch.sqrt(torch.mean(diff ** 2)).item()
        teacher_action_rms = torch.sqrt(torch.mean(teacher_actions ** 2)).item()
        teacher_action_abs_max = torch.max(torch.abs(teacher_actions)).item()
        rel_rmse = diff_rmse / max(teacher_action_rms, 1e-8)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Compute gradient norms (before clipping)
    grad_norms = [p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None]
    grad_norm = np.sqrt(sum(g ** 2 for g in grad_norms)) if grad_norms else 0.0
    grad_norm_max = max(grad_norms) if grad_norms else 0.0

    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    """
    原代码在 train batch中的mem处理存在缺陷: 如果只检查最后一个done, 如果最后一步环境done了, 清空该环境的mems
    如果一个 Episode 在序列中间结束，在这个结束点之前的所有 Memory 对于下一个 Batch 来说都是污染数据，必须全部清除，而不仅仅是检查最后一步。
    """
    if new_mems is not None:
        # dones shape: [Batch, Seq_Len]
        # new_mems shape: List of [Batch, Mem_Len, D_Model]
        batch_size = dones.shape[0]

        for b in range(batch_size):
            # 找到该环境在当前序列中所有 done 的位置
            done_indices = torch.nonzero(dones[b])

            if done_indices.numel() > 0:
                # 找到最后一个 done 的索引
                last_done_pos = done_indices.max().item()

                # 清空该位置及之前的记忆
                # 下一个 Batch 将从 last_done_pos + 1 的上下文开始继续
                for layer_mem in new_mems:
                    layer_mem[b, :last_done_pos + 1, :] = 0.0

    metrics = {
        "loss": float(loss.item()),
        "diff_rmse": diff_rmse,
        "rel_rmse": rel_rmse,
        "grad_norm": grad_norm,
        "grad_norm_max": grad_norm_max,
        "teacher_action_rms": teacher_action_rms,
        "teacher_action_abs_max": teacher_action_abs_max,
    }
    return metrics, new_mems


if __name__ == "__main__":
    run_training()
