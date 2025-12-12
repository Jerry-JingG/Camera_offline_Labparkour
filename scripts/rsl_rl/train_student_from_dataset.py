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
class SequenceSample:
    """One training sample consisting of a contiguous sequence."""

    env_id: int
    proprio: np.ndarray  # [S, hist_len * state_dim]
    depth: np.ndarray  # [S, depth_hist_len, H, W]
    actions: np.ndarray  # [S, action_dim]
    dones: np.ndarray  # [S]


class SequenceAggregator:
    """Aggregates per-environment steps into temporal windows for training."""

    def __init__(
        self,
        num_envs: int,
        prop_hist_len: int,
        depth_hist_len: int,
        sequence_len: int,
    ) -> None:
        self.num_envs = num_envs
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.sequence_len = sequence_len
        self.prop_histories: List[Deque[np.ndarray]] = [
            deque(maxlen=prop_hist_len) for _ in range(num_envs)
        ]
        self.depth_histories: List[Deque[np.ndarray]] = [
            deque(maxlen=depth_hist_len) for _ in range(num_envs)
        ]
        self.sequence_buffers: List[Deque[Dict[str, np.ndarray]]] = [
            deque(maxlen=sequence_len) for _ in range(num_envs)
        ]

    def reset(self) -> None:
        for prop_hist in self.prop_histories:
            prop_hist.clear()
        for depth_hist in self.depth_histories:
            depth_hist.clear()
        for seq_buf in self.sequence_buffers:
            seq_buf.clear()

    def push_step(
        self,
        obs_prop: np.ndarray,
        depth_frame: np.ndarray,
        teacher_actions: np.ndarray,
        done: np.ndarray,
    ) -> List[SequenceSample]:
        """Consume a synchronized environment step and emit ready sequences."""

        ready: List[SequenceSample] = []
        for env_id in range(self.num_envs):
            prop_hist = self.prop_histories[env_id]
            depth_hist = self.depth_histories[env_id]
            seq_buf = self.sequence_buffers[env_id]

            prop_hist.append(obs_prop[env_id].astype(np.float32, copy=False))
            depth_hist.append(depth_frame[env_id].astype(np.float32, copy=False))

            if len(prop_hist) < self.prop_hist_len or len(depth_hist) < self.depth_hist_len:
                if done[env_id]:
                    self._reset_env(env_id)
                continue

            prop_stack = np.concatenate(list(prop_hist), axis=0).astype(np.float32, copy=False)
            depth_stack = np.stack(list(depth_hist), axis=0).astype(np.float32, copy=False)

            seq_buf.append(
                {
                    "prop": prop_stack,
                    "depth": depth_stack,
                    "action": teacher_actions[env_id].astype(np.float32, copy=False),
                    "done": bool(done[env_id]),
                }
            )

            if len(seq_buf) == self.sequence_len:
                ready.append(self._pack_sequence(env_id))

            if done[env_id]:
                self._reset_env(env_id)

        return ready

    def _pack_sequence(self, env_id: int) -> SequenceSample:
        seq = list(self.sequence_buffers[env_id])
        proprio = np.stack([step["prop"] for step in seq], axis=0)
        depth = np.stack([step["depth"] for step in seq], axis=0)
        actions = np.stack([step["action"] for step in seq], axis=0)
        dones = np.array([step["done"] for step in seq], dtype=bool)
        return SequenceSample(env_id=env_id, proprio=proprio, depth=depth, actions=actions, dones=dones)

    def _reset_env(self, env_id: int) -> None:
        self.prop_histories[env_id].clear()
        self.depth_histories[env_id].clear()
        self.sequence_buffers[env_id].clear()


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

    def iter_sequences(self, max_sequences: Optional[int] = None) -> Iterator[SequenceSample]:
        """Yield SequenceSample entries for one epoch."""

        self.aggregator.reset()
        yielded = 0

        for shard_path in self.shards:
            with np.load(shard_path, allow_pickle=False) as shard:
                obs_prop = shard["obs_prop"].astype(np.float32)
                actions = shard["action_teacher"].astype(np.float32)
                dones = shard["done"].astype(bool)
                depth = shard["depth"]
                num_steps = obs_prop.shape[0]

                for step_idx in range(num_steps):
                    depth_frame = self._convert_depth(depth[step_idx])
                    ready = self.aggregator.push_step(
                        obs_prop=obs_prop[step_idx],
                        depth_frame=depth_frame,
                        teacher_actions=actions[step_idx],
                        done=dones[step_idx].reshape(-1),
                    )
                    for sample in ready:
                        yield sample
                        yielded += 1
                        if max_sequences is not None and yielded >= max_sequences:
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
            tanh_output=action_head_cfg.get("tanh_output", False),
            action_scale=action_head_cfg.get("action_scale", 1.0),
        )
        # Filled in by training loop: one memory list per env_id.
        self._env_mems: List[List[Tensor]] = []

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train new transformer student policy from collected teacher datasets."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to collect.py output directory.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device (e.g., cuda:0 or cpu).")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of passes over the dataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of sequences per optimization step.")
    parser.add_argument("--sequence_length", type=int, default=16, help="Temporal window size for training samples.")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length (in steps) for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="Number of stacked depth frames per sample.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold (L2 norm).")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between logging training metrics.")
    parser.add_argument("--max_sequences_per_epoch", type=int, default=None, help="Optional cap on sequences per epoch.")
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

    def reset_all_env_mems() -> None:
        """Initialize TXL memories for each env_id (batch dimension = 1)."""
        model._env_mems = [model.temporal_model.reset_mems(1) for _ in range(streamer.num_envs)]  # type: ignore[attr-defined]

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
        reset_all_env_mems()
        batch: List[SequenceSample] = []
        running_loss = 0.0
        num_updates = 0
        sequences_seen = 0

        for sample in streamer.iter_sequences(max_sequences=args.max_sequences_per_epoch):
            batch.append(sample)
            sequences_seen += 1
            if len(batch) < args.batch_size:
                continue

            loss, grad_norm = train_batch(model, optimizer, batch, device, args.grad_clip)
            running_loss += loss
            num_updates += 1
            global_step += 1
            batch.clear()

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
                    f"updates={num_updates} | avg_loss={avg_loss:.6f} | sequences={sequences_seen}"
                )

        if batch:
            loss, grad_norm = train_batch(model, optimizer, batch, device, args.grad_clip)
            running_loss += loss
            num_updates += 1
            global_step += 1
            batch.clear()
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

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(1, num_updates)
        seqs_per_sec = sequences_seen / max(1e-8, epoch_time)
        print(
            f"[epoch {epoch}] completed in {epoch_time:.1f}s | "
            f"updates={num_updates} | sequences={sequences_seen} | avg_loss={avg_loss:.6f}"
        )

        if args.use_wandb:
            wandb.log(
                {
                    "train/epoch_avg_loss": avg_loss,
                    "train/epoch_time": epoch_time,
                    "train/epoch": epoch,
                    "train/num_updates": num_updates,
                    "train/sequences_seen": sequences_seen,
                    "train/sequences_per_sec": seqs_per_sec,
                }
            )

        ckpt_path = save_dir / f"student_epoch_{epoch:04d}.pt"
        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, global_step, streamer.meta)

    if args.use_wandb:
        wandb.finish()


def train_batch(
    model: MultiModalStudentPolicy,
    optimizer: torch.optim.Optimizer,
    batch_samples: Sequence[SequenceSample],
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, Optional[float]]:
    proprio = torch.from_numpy(np.stack([sample.proprio for sample in batch_samples], axis=0)).to(device)
    depth = torch.from_numpy(np.stack([sample.depth for sample in batch_samples], axis=0)).to(device)
    teacher_actions = torch.from_numpy(
        np.stack([sample.actions for sample in batch_samples], axis=0)
    ).to(device)

    # Assemble per-layer memories for this mini-batch (B may mix envs).
    num_layers = len(model.temporal_model.layers)
    mems_in: List[torch.Tensor] = []
    for layer_idx in range(num_layers):
        mems_layer = [model._env_mems[s.env_id][layer_idx] for s in batch_samples]  # type: ignore[attr-defined]
        # 对齐长度：不同 env 的 mem 可能是空（0 长度）或已达 mem_len，需要左侧零填充到同一长度再 concat
        target_len = max(m.size(1) for m in mems_layer)
        if target_len == 0:
            mems_in.append(torch.cat(mems_layer, dim=0))
        else:
            padded: List[torch.Tensor] = []
            for m in mems_layer:
                if m.size(1) == target_len:
                    padded.append(m)
                else:
                    pad_len = target_len - m.size(1)
                    pad = torch.zeros(m.size(0), pad_len, m.size(2), device=m.device, dtype=m.dtype)
                    padded.append(torch.cat([pad, m], dim=1))
            mems_in.append(torch.cat(padded, dim=0))

    predictions, new_mems = model(proprio, depth, mems=mems_in, return_mems=True)
    loss = torch.nn.functional.mse_loss(predictions, teacher_actions)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm: Optional[float] = None
    if grad_clip > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
    optimizer.step()

    # Update per-env memories (detach to break graph) and reset on done.
    with torch.no_grad():
        for batch_idx, sample in enumerate(batch_samples):
            env_id = sample.env_id
            if sample.dones.any():
                model._env_mems[env_id] = model.temporal_model.reset_mems(1)
            else:
                updated = [mem[batch_idx : batch_idx + 1].detach() for mem in new_mems]  # type: ignore[arg-type]
                model._env_mems[env_id] = updated

    return float(loss.item()), grad_norm


if __name__ == "__main__":
    run_training()
