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
from typing import Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import wandb

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


@dataclass
class BatchSequenceSample:
    """A batch-aligned training sample spanning all environments."""

    proprio: np.ndarray  # [B, S, prop_hist_len * proprio_dim]
    depth: np.ndarray  # [B, S, depth_hist_len, H, W]
    actions: np.ndarray  # [B, S, action_dim]
    dones: np.ndarray  # [B, S]


class TeacherDatasetStreamer:
    """Streams teacher trajectories and exposes fused temporal samples."""

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

    @staticmethod
    def _load_meta(dataset_dir: Path) -> Dict[str, object]:
        meta_path = dataset_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Dataset missing meta.json: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as meta_file:
            return json.load(meta_file)

    def iter_batches(self, max_batches: Optional[int] = None) -> Iterator[BatchSequenceSample]:
        """Yield batch-aligned temporal segments without overlap."""

        prop_histories: List[Deque[np.ndarray]] = [
            deque(maxlen=self.prop_hist_len) for _ in range(self.num_envs)
        ]
        depth_histories: List[Deque[np.ndarray]] = [
            deque(maxlen=self.depth_hist_len) for _ in range(self.num_envs)
        ]
        segment_prop: List[np.ndarray] = []
        segment_depth: List[np.ndarray] = []
        segment_actions: List[np.ndarray] = []
        segment_dones: List[np.ndarray] = []
        yielded = 0
        proprio_dim: Optional[int] = None
        action_dim: Optional[int] = None

        for shard_path in self.shards:
            with np.load(shard_path, allow_pickle=False) as shard:
                obs_prop = shard["obs_prop"].astype(np.float32)
                actions = shard["action_teacher"].astype(np.float32)
                dones = shard["done"].astype(bool)
                depth = shard["depth"]
                if obs_prop.shape[1] != self.num_envs:
                    raise ValueError(
                        f"Shard {shard_path} expected num_envs={self.num_envs}, got {obs_prop.shape[1]}"
                    )
                if actions.shape[1] != self.num_envs or dones.shape[1] != self.num_envs:
                    raise ValueError(f"Shard {shard_path} env dimension mismatch in actions/dones")

                proprio_dim = proprio_dim or obs_prop.shape[-1]
                action_dim = action_dim or actions.shape[-1]
                if proprio_dim != obs_prop.shape[-1]:
                    raise ValueError(f"Shard {shard_path} proprio_dim mismatch: {obs_prop.shape[-1]} vs {proprio_dim}")
                if action_dim != actions.shape[-1]:
                    raise ValueError(f"Shard {shard_path} action_dim mismatch: {actions.shape[-1]} vs {action_dim}")
                num_steps = obs_prop.shape[0]

                for step_idx in range(num_steps):
                    obs_step = obs_prop[step_idx]
                    actions_step = actions[step_idx]
                    dones_step = dones[step_idx].astype(bool).reshape(self.num_envs)
                    depth_step = self._convert_depth(depth[step_idx])
                    if depth_step.shape[0] != self.num_envs:
                        raise ValueError(
                            f"Shard {shard_path} depth env dimension {depth_step.shape[0]} != {self.num_envs}"
                        )

                    for env_id in range(self.num_envs):
                        prop_histories[env_id].append(obs_step[env_id].astype(np.float32, copy=False))
                        depth_histories[env_id].append(depth_step[env_id].astype(np.float32, copy=False))

                    all_ready = all(
                        len(prop_histories[env_id]) == self.prop_hist_len
                        and len(depth_histories[env_id]) == self.depth_hist_len
                        for env_id in range(self.num_envs)
                    )

                    if all_ready:
                        prop_stack = np.stack(
                            [
                                np.concatenate(list(prop_histories[env_id]), axis=0).astype(np.float32, copy=False)
                                for env_id in range(self.num_envs)
                            ],
                            axis=0,
                        )
                        depth_stack = np.stack(
                            [
                                np.stack(list(depth_histories[env_id]), axis=0).astype(np.float32, copy=False)
                                for env_id in range(self.num_envs)
                            ],
                            axis=0,
                        )
                        segment_prop.append(prop_stack)
                        segment_depth.append(depth_stack)
                        segment_actions.append(actions_step.astype(np.float32, copy=False))
                        segment_dones.append(dones_step.astype(bool, copy=False))

                        if len(segment_prop) == self.sequence_len:
                            proprio_seq = np.swapaxes(np.stack(segment_prop, axis=0), 0, 1)
                            depth_seq = np.swapaxes(np.stack(segment_depth, axis=0), 0, 1)
                            actions_seq = np.swapaxes(np.stack(segment_actions, axis=0), 0, 1)
                            dones_seq = np.swapaxes(np.stack(segment_dones, axis=0), 0, 1)
                            assert proprio_seq.shape[0] == self.num_envs, "Batch size must equal num_envs"
                            assert depth_seq.shape[0] == self.num_envs, "Batch size must equal num_envs"
                            assert actions_seq.shape[0] == self.num_envs, "Batch size must equal num_envs"
                            assert dones_seq.shape[0] == self.num_envs, "Batch size must equal num_envs"
                            assert proprio_seq.shape[1] == self.sequence_len, "Sequence length mismatch"
                            assert depth_seq.shape[1] == self.sequence_len, "Sequence length mismatch"
                            assert actions_seq.shape[1] == self.sequence_len, "Sequence length mismatch"
                            assert dones_seq.shape[1] == self.sequence_len, "Sequence length mismatch"
                            if proprio_dim is not None:
                                assert (
                                    proprio_seq.shape[2] == self.prop_hist_len * proprio_dim
                                ), "Proprio feature dimension mismatch"
                            if action_dim is not None:
                                assert actions_seq.shape[2] == action_dim, "Action dimension mismatch"
                            assert depth_seq.shape[2] == self.depth_hist_len, "Depth history length mismatch"
                            yield BatchSequenceSample(
                                proprio=proprio_seq,
                                depth=depth_seq,
                                actions=actions_seq,
                                dones=dones_seq,
                            )
                            yielded += 1
                            segment_prop.clear()
                            segment_depth.clear()
                            segment_actions.clear()
                            segment_dones.clear()
                            if max_batches is not None and yielded >= max_batches:
                                return

                    # Reset histories immediately when an env is done.
                    for env_id in range(self.num_envs):
                        if dones_step[env_id]:
                            prop_histories[env_id].clear()
                            depth_histories[env_id].clear()

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
            tanh_output=action_head_cfg.get("tanh_output", False),
            action_scale=action_head_cfg.get("action_scale", 1.0),
        )

    def forward(
        self,
        proprio_seq: Tensor,
        depth_seq: Tensor,
        mems: Optional[List[Optional[Tensor]]] = None,
        return_mems: bool = False,
    ):
        """
        Args:
            proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
            depth_seq: Tensor[B, S, depth_hist_len, H, W]
            mems: Optional TXL memories (list of [B, M, d]).
            return_mems: Whether to return updated memories.

        Returns:
            Predicted action means of shape [B, S, action_dim] and optionally new mems.
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
        temporal_out, new_mems = self.temporal_model(
            fused_seq,
            mems=mems,
            causal_mask=True,
            return_mems=True,
        )
        actions = self.action_head.forward_sequence(temporal_out)["mean"]
        if return_mems:
            return actions, new_mems
        return actions

    def forward_step(
        self,
        proprio_step: Tensor,
        depth_step: Tensor,
        mems: Optional[List[Optional[Tensor]]] = None,
        return_mems: bool = True,
    ):
        """
        Run a single-timestep inference pass with recurrent memories.

        Args:
            proprio_step: Tensor[B, 1, prop_hist_len * proprio_dim] or Tensor[B, prop_hist_len * proprio_dim]
            depth_step: Tensor[B, 1, depth_hist_len, H, W] or Tensor[B, depth_hist_len, H, W]
            mems: Optional TXL memories (list of [B, M, d]).
            return_mems: Whether to return updated memories.
        """

        if proprio_step.dim() == 2:
            proprio_step = proprio_step.unsqueeze(1)
        if proprio_step.dim() != 3 or proprio_step.size(1) != 1:
            raise ValueError("proprio_step must have shape [B, 1, prop_hist_len * proprio_dim]")

        if depth_step.dim() == 4:
            depth_step = depth_step.unsqueeze(1)
        if depth_step.dim() != 5 or depth_step.size(1) != 1:
            raise ValueError("depth_step must have shape [B, 1, depth_hist_len, H, W]")

        return self.forward(proprio_step, depth_step, mems=mems, return_mems=return_mems)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train new transformer student policy from collected teacher datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to collect.py output directory.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device (e.g., cuda:0 or cpu).")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of passes over the dataset.")
    parser.add_argument("--sequence_length", type=int, default=16, help="Temporal window size for training samples.")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length (in steps) for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="Number of stacked depth frames per sample.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold (L2 norm).")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging training metrics.")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None, help="Optional cap on segments per epoch.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to store checkpoints (defaults to dataset dir).")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume training from.")
    parser.add_argument("--wandb_project", type=str, default="robot_camera_offline_student", help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Optional W&B entity.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="Optional list of W&B tags.")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="W&B logging mode (online or offline).",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
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
        "tanh_output": False,
        "action_scale": 1.0,
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

    def init_batch_mems(batch_size: int) -> List[torch.Tensor]:
        """Allocate unified TXL memories [layer][B, mem_len, C] on the model device."""
        mem_len = model.temporal_model.mem_len
        token_dim = model.temporal_model.d_model
        device_t = next(model.parameters()).device
        dtype_t = next(model.parameters()).dtype
        return [
            torch.zeros(batch_size, mem_len, token_dim, device=device_t, dtype=dtype_t)
            for _ in range(len(model.temporal_model.layers))
        ]

    def get_current_lr(opt: torch.optim.Optimizer) -> float:
        for group in opt.param_groups:
            if "lr" in group:
                return float(group["lr"])
        return 0.0

    if args.use_wandb:
        wandb_config = {
            **vars(args),
            "dataset_dir": str(dataset_dir),
            "num_envs": streamer.num_envs,
            "num_shards": len(streamer.shards),
            "camera_resolution": streamer.camera_resolution,
            "max_batches_per_epoch": args.max_batches_per_epoch,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            config=wandb_config,
        )
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.watch(model, log="all", log_freq=max(1, args.log_interval))

    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        start_epoch, global_step = load_checkpoint(resume_path, model, optimizer)

    print(
        f"[info] Starting training for {args.num_epochs} epochs "
        f"on dataset {dataset_dir} using device {device}."
    )

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        num_updates = 0
        batches_seen = 0
        mems = init_batch_mems(streamer.num_envs)

        for batch in streamer.iter_batches(max_batches=args.max_batches_per_epoch):
            loss, grad_norm, mems = train_batch(model, optimizer, batch, device, args.grad_clip, mems)
            running_loss += loss
            num_updates += 1
            batches_seen += 1
            global_step += 1

            if args.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss,
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                        "train/grad_norm": grad_norm,
                        "train/lr": get_current_lr(optimizer),
                    }
                )

            if args.log_interval > 0 and num_updates % args.log_interval == 0:
                avg_loss = running_loss / max(1, num_updates)
                print(
                    f"[epoch {epoch}] step {global_step} | "
                    f"updates={num_updates} | avg_loss={avg_loss:.6f} | batches={batches_seen}"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(1, num_updates)
        batches_per_sec = batches_seen / max(1e-8, epoch_time)
        print(
            f"[epoch {epoch}] completed in {epoch_time:.1f}s | "
            f"updates={num_updates} | batches={batches_seen} | avg_loss={avg_loss:.6f}"
        )

        if args.use_wandb:
            wandb.log(
                {
                    "train/epoch_avg_loss": avg_loss,
                    "train/epoch_time": epoch_time,
                    "train/epoch": epoch,
                    "train/num_updates": num_updates,
                    "train/batches_seen": batches_seen,
                    "train/batches_per_sec": batches_per_sec,
                }
            )

        ckpt_path = save_dir / f"student_epoch_{epoch:04d}.pt"
        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, global_step, streamer.meta)

    if args.use_wandb:
        wandb.finish()


def train_batch(
    model: MultiModalStudentPolicy,
    optimizer: torch.optim.Optimizer,
    batch: BatchSequenceSample,
    device: torch.device,
    grad_clip: float,
    mems: List[torch.Tensor],
) -> Tuple[float, Optional[float], List[torch.Tensor]]:
    """Single optimization step on one batch-aligned segment."""

    proprio = torch.from_numpy(batch.proprio).to(device)
    depth = torch.from_numpy(batch.depth).to(device)
    teacher_actions = torch.from_numpy(batch.actions).to(device)
    dones = torch.from_numpy(batch.dones).to(device)

    batch_size, seq_len, prop_feat = proprio.shape
    assert batch_size == depth.shape[0] == teacher_actions.shape[0] == dones.shape[0], "B dimension mismatch"
    assert batch_size == mems[0].size(0), "Memory batch size must match num_envs"
    assert depth.shape[1] == seq_len and teacher_actions.shape[1] == seq_len, "S dimension mismatch"
    assert len(mems) == len(model.temporal_model.layers), "Memory layers length mismatch"
    assert prop_feat % model.prop_hist_len == 0, "Proprio feature dimension must align with prop history length"
    mem_len_expected = model.temporal_model.mem_len
    token_dim = model.temporal_model.d_model
    for layer_mem in mems:
        assert layer_mem.dim() == 3, "Each memory tensor must be [B, M, C]"
        assert layer_mem.size(0) == batch_size, "Memory batch size mismatch"
        assert layer_mem.size(2) == token_dim, "Memory token_dim mismatch"
        if mem_len_expected > 0:
            assert layer_mem.size(1) == mem_len_expected, "Memory length mismatch"

    current_mems: List[torch.Tensor] = [m.detach().to(device) for m in mems]
    preds_per_step: List[Tensor] = []

    for t in range(seq_len):
        proprio_step = proprio[:, t : t + 1, :]
        depth_step = depth[:, t : t + 1, ...]
        pred_step, new_mems = model.forward_step(proprio_step, depth_step, mems=current_mems, return_mems=True)
        preds_per_step.append(pred_step)

        dones_t = dones[:, t]
        if dones_t.dtype != torch.bool:
            dones_t = dones_t.bool()
        reset_mask = (~dones_t).view(batch_size, 1, 1).to(pred_step.device)
        current_mems = [layer_mem * reset_mask for layer_mem in new_mems]

    predictions = torch.cat(preds_per_step, dim=1)
    assert predictions.shape == teacher_actions.shape, "Prediction/action shape mismatch"
    loss = torch.nn.functional.mse_loss(predictions, teacher_actions)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm: Optional[float] = None
    if grad_clip > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    optimizer.step()

    mems_out = [m.detach() for m in current_mems]
    return float(loss.item()), grad_norm, mems_out


if __name__ == "__main__":
    run_training()
