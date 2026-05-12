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
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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
from transformerxl.student_policy import MultiModalStudentPolicy
from transformerxl.temporal.txl import TransformerXLTemporal
from utils.student_utils import SequenceAggregator, build_student_model


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
        device: torch.device,
        use_dropout: bool = False
    ) -> None:
        self.dataset_dir = dataset_dir
        self.meta = self._load_meta(dataset_dir)
        self.num_envs = int(self.meta["num_envs"])
        self.sequence_len = sequence_len
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.depth_dtype = self.meta.get("depth_dtype", "uint16")
        self.depth_scale = float(self.meta.get("depth_scale", 1000.0))
        shards_root = dataset_dir / "shards"
        self.shards: List[Path] = sorted(shards_root.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No dataset shards found in {shards_root}")
        self.device = device

        dataset_layout = infer_dataset_layout(self.shards[0])
        self.proprio_dim = int(self.meta.get("num_prop", dataset_layout["num_prop"]))
        self.action_dim = int(self.meta.get("action_dim", dataset_layout["action_dim"]))
        self.extra_info_dim = int(self.meta.get("extra_info_dim", dataset_layout["extra_info_dim"]))
        self.camera_resolution = tuple(
            self.meta.get("camera_resolution", list(dataset_layout["camera_resolution"]))
        )
        self.meta.setdefault("num_prop", self.proprio_dim)
        self.meta.setdefault("action_dim", self.action_dim)
        self.meta.setdefault("extra_info_dim", self.extra_info_dim)
        self.meta.setdefault("camera_resolution", list(self.camera_resolution))

        self.aggregator = SequenceAggregator(
            num_envs=self.num_envs,
            device=self.device,
            prop_hist_len=prop_hist_len,
            depth_hist_len=depth_hist_len,
            sequence_len=sequence_len,
            num_prop=self.proprio_dim,
            depth_shape=self.camera_resolution,  # type: ignore[arg-type]
            extra_info_dim=self.extra_info_dim,
        )

        # Camera Dropout Augmentation
        self.dropout_manager = None
        if use_dropout:
            print("[Streamer] Training-time Camera Dropout Augmentation: ENABLED")
            dt = float(self.meta.get("step_dt", 0.02))
            self.dropout_manager = CameraDropoutManager(
                num_envs=self.num_envs,
                device=self.device,
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

        prev_dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for shard_path in self.shards:
            with np.load(shard_path, allow_pickle=False) as shard:
                obs_prop = shard["obs_prop"].astype(np.float32)
                actions = shard["action_teacher"].astype(np.float32)
                dones = shard["done"].astype(bool)
                depth = shard["depth"]
                extra_infos = shard["extra_info"].astype(np.float32)
                num_steps = obs_prop.shape[0]

                for step_idx in range(num_steps):
                    current_obs_prop = obs_prop[step_idx]
                    depth_frame = self._convert_depth(depth[step_idx])
                    if depth_frame.ndim == 4 and depth_frame.shape[1] == 1:
                        depth_frame = depth_frame[:, 0]

                    extra_info = extra_infos[step_idx]

                    # --- 提前转换为 Tensor ---
                    current_obs_prop_t = torch.from_numpy(current_obs_prop).to(self.device)
                    depth_frame_t = torch.from_numpy(depth_frame).to(self.device)
                    actions_t = torch.from_numpy(actions[step_idx]).to(self.device)
                    dones_t = torch.from_numpy(dones[step_idx].reshape(-1)).to(self.device)
                    extra_info_t = torch.from_numpy(extra_info).to(self.device)

                    # --- dropout 操作直接操作 Tensor ---
                    if self.dropout_manager is not None:
                        self.dropout_manager.reset_env(prev_dones)
                        self.dropout_manager.update(depth_image=depth_frame_t, obs_prop=current_obs_prop_t)

                    prev_dones = dones_t.clone()

                    # Push step and check if a batch is ready (传入 Tensor)
                    batch_data = self.aggregator.push_step(
                        obs_prop=current_obs_prop_t,
                        depth_frame=depth_frame_t,
                        teacher_actions=actions_t,
                        done=dones_t,
                        extra_info=extra_info_t
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
    mem_len: int,
) -> Tuple[MultiModalStudentPolicy, int]:
    proprio_dim = streamer.proprio_dim
    action_dim = streamer.action_dim
    camera_resolution = streamer.camera_resolution
    model = build_student_model(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,  # type: ignore[arg-type]
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        mem_len=mem_len,
        token_dim=128
    )
    return model, action_dim


def infer_dataset_layout(shard_path: Path) -> Dict[str, object]:
    with np.load(shard_path, allow_pickle=False) as shard:
        obs_prop = shard["obs_prop"]
        action_teacher = shard["action_teacher"]
        extra_info = shard["extra_info"]
        depth = shard["depth"]
        return {
            "num_prop": int(obs_prop.shape[-1]),
            "action_dim": int(action_teacher.shape[-1]),
            "extra_info_dim": int(extra_info.shape[-1]),
            "camera_resolution": tuple(int(v) for v in depth.shape[-2:]),
        }


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
    device = torch.device(args.device)
    streamer = TeacherDatasetStreamer(
        dataset_dir=dataset_dir,
        sequence_len=args.sequence_length,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        device=device,
        use_dropout=args.use_dropout
    )
    model, _ = build_student_from_dataset(streamer, args.prop_hist_len, args.depth_hist_len, args.sequence_length)
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
        mems = None
        mem_dones = None

        # 直接迭代 Batch (无需再组装 samples)
        for batch_data in streamer.iter_batches(max_sequences=args.max_sequences_per_epoch):
            batch_start = time.time()
            metrics, mems, mem_dones = train_batch(
                model,
                optimizer,
                batch_data,
                device,
                args.grad_clip,
                mems=mems,
                mem_dones=mem_dones,
            )
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
                    "train/action_loss": metrics["action_loss"],
                    "train/yaw_loss": metrics["yaw_loss"],
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

    final_ckpt_path = save_dir / "student_final.pt"
    save_checkpoint(final_ckpt_path, model, optimizer, args.num_epochs, global_step, streamer.meta)
    if use_wandb:
        wandb.save(str(final_ckpt_path))

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
    mem_dones: Optional[Tensor] = None,
) -> Tuple[Dict[str, float], Optional[List[Tensor]], Optional[Tensor]]:
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
    true_yaws = batch_data["extra_infos"].to(device)

    current_mem_len = mems[0].size(1) if mems else 0
    if current_mem_len > 0:
        if mem_dones is None:
            raise ValueError("mem_dones must be provided when recurrent mems are carried across batches.")
        mem_dones = mem_dones[:, -current_mem_len:]
        full_dones = torch.cat([mem_dones, dones], dim=1)
    else:
        full_dones = dones

    predictions, pred_yaws, new_mems = model.forward_with_mems(
        proprio,
        depth,
        mems=mems,
        full_dones=full_dones,
    )

    action_loss = torch.nn.functional.mse_loss(predictions, teacher_actions)
    yaw_loss = torch.nn.functional.mse_loss(pred_yaws, true_yaws)
    loss = action_loss + yaw_loss

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

    next_mems: Optional[List[Tensor]] = None
    next_mem_dones: Optional[Tensor] = None
    if new_mems is not None:
        next_mems = TransformerXLTemporal.detach_mems(new_mems)
        next_mem_len = next_mems[0].size(1) if next_mems else 0
        if next_mem_len > 0:
            next_mem_dones = full_dones[:, -next_mem_len:].detach().clone()
        else:
            next_mem_dones = dones.new_empty(dones.size(0), 0)

    metrics = {
        "loss": float(loss.item()),
        "action_loss": float(action_loss.item()),
        "yaw_loss": float(yaw_loss.item()),
        "diff_rmse": diff_rmse,
        "rel_rmse": rel_rmse,
        "grad_norm": grad_norm,
        "grad_norm_max": grad_norm_max,
        "teacher_action_rms": teacher_action_rms,
        "teacher_action_abs_max": teacher_action_abs_max,
    }
    return metrics, next_mems, next_mem_dones


if __name__ == "__main__":
    run_training()
