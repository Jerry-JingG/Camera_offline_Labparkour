"""
原代码在 train batch中的mem处理存在缺陷: 如果sequence中有一个环境done了, 直接对一整个环境清空
后果：丢失了有效上下文：第 6-63 步（新 Episode 的 58 个 Step) 的记忆被删除了。
已修改为: 该sequence中只要有done的步骤, 那么在最后一个done的步骤之前的所有mem被清空
"""

# 训练流程本质是两部分代码交替进行
# 1. 学生模型与环境交互，同时采集学生观测，学生输出与教师输出 -》采集的轨迹长度达到sequence_length时，环境停止
# 2. 采集到的数据作为batch，计算损失并更新学生模型

from __future__ import annotations
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# Ensure project-local packages (parkour_isaaclab, parkour_tasks, etc.) are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

RSL_RL_DIR = os.path.join(PROJECT_ROOT, "scripts", "rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

# === Import aggregator and student policy (same directory) ===
from transformerxl.student_policy import MultiModalStudentPolicy
from transformerxl.temporal.txl import TransformerXLTemporal
from utils.student_utils import (
    SequenceAggregator,
    StudentOnlineRunner,
    ResetSettleManager,
    build_student_model,
    load_env_and_teacher
)

DEPTH_CROP_LEFT_COLS = 10


def _left_crop_resize_depth(depth_image: Tensor, left_cols: int = DEPTH_CROP_LEFT_COLS) -> Tensor:
    """Crop the left camera columns and resize back to the policy resolution."""
    if left_cols <= 0:
        return depth_image

    if depth_image.dim() == 3:
        depth_4d = depth_image.unsqueeze(1)
        squeeze_channel = True
    elif depth_image.dim() == 4 and depth_image.shape[1] == 1:
        depth_4d = depth_image
        squeeze_channel = False
    else:
        raise ValueError(
            "depth_image must have shape [N,H,W] or [N,1,H,W], "
            f"got {tuple(depth_image.shape)}"
        )

    height, width = depth_4d.shape[-2:]
    if left_cols >= width:
        raise ValueError(f"left_cols={left_cols} must be smaller than image width={width}")

    cropped = depth_4d[..., :, left_cols:]
    resized = F.interpolate(
        cropped,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    resized = torch.clamp(resized, min=-0.5, max=0.5)
    return resized.squeeze(1) if squeeze_channel else resized


def _enable_encoder_only_training(model: MultiModalStudentPolicy) -> List[Tensor]:
    model.requires_grad_(False)
    model.proprio_encoder.requires_grad_(True)
    model.depth_encoder.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _set_encoder_only_train_mode(model: MultiModalStudentPolicy) -> None:
    model.train()
    model.fusion_transformer.eval()
    model.temporal_model.eval()
    model.action_head.eval()
    model.yaw_head.eval()


def parse_args():
    parser = argparse.ArgumentParser("Train Student via DAgger")
    parser.add_argument("--task", type=str, required=True, help="Isaac task name")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, default=None, help="Pretrained student or resume path")
    parser.add_argument("--num_envs", type=int, default=64)
    # DAgger specific
    parser.add_argument("--num_iters", type=int, default=5000, help="Total DAgger iterations (batches)")
    parser.add_argument("--num_pretrain_iters", type=int, default=0, help="Iterations where only teacher acts (warmup)")
    parser.add_argument("--teacher_mixture", action="store_true", help="Mix teacher and student actions")
    parser.add_argument("--beta_start", type=float, default=0.5, help="Initial probability of using teacher action")
    parser.add_argument("--beta_end", type=float, default=0.0, help="Final probability of using teacher action")
    parser.add_argument("--beta_decay_iters", type=int, default=2000, help="Iterations to decay beta")

    # Model / Training
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--prop_hist_len", type=int, default=1)
    parser.add_argument("--depth_hist_len", type=int, default=1)
    parser.add_argument(
        "--encoder_only_training",
        action="store_true",
        help="Freeze fusion/temporal/action heads and train only proprio/depth encoders.",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--reset_settle_steps",
        type=int,
        default=25,
        help="Zero-action settling steps after reset; these steps are treated as invalid TXL pseudo-episodes.",
    )

    # Augmentation
    parser.add_argument("--use_dropout", action="store_true", help="Enable camera dropout during collection")

    # Logging
    parser.add_argument("--save_dir", type=str, default="outputs/students/trainxl_from_dataset")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dagger-parkour")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    # RSL-RL / AppLauncher args
    import cli_args
    from isaaclab.app import AppLauncher
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup WandB
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        print("[warning] wandb not installed. Run `pip install wandb` to enable logging.")
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    elif args.wandb:
        print("[Warning] WandB requested but not installed.")

    # 2. Load Env & Teacher
    vec_env, teacher_policy, agent_cfg, simulation_app = load_env_and_teacher(args)

    # 3. Infer dimensions
    obs, extras = vec_env.get_observations()
    if "depth_camera" not in extras["observations"]:
        raise RuntimeError("Env must provide 'depth_camera' observation.")

    depth_sample = extras["observations"]["depth_camera"]
    camera_resolution = (int(depth_sample.shape[-2]), int(depth_sample.shape[-1]))
    proprio_dim = int(agent_cfg.estimator.num_prop)
    action_dim = vec_env.unwrapped.action_space.shape[1] if hasattr(vec_env.unwrapped.action_space, "shape") else obs.shape[1]

    print(f"[Info] Prop Dim: {proprio_dim}, Action Dim: {action_dim}, Cam Res: {camera_resolution}")

    # 4. Initialize Student Model
    student_model = build_student_model(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        mem_len=args.sequence_length,
        token_dim=128
    ).to(device)
    if args.encoder_only_training:
        trainable_parameters = _enable_encoder_only_training(student_model)
        train_mode_name = "encoder_only"
        trainable_modules = ["proprio_encoder", "depth_encoder"]
        print("[Info] Encoder-only training enabled: proprio_encoder + depth_encoder")
    else:
        trainable_parameters = [parameter for parameter in student_model.parameters() if parameter.requires_grad]
        train_mode_name = "full"
        trainable_modules = [
            "proprio_encoder",
            "depth_encoder",
            "fusion_transformer",
            "temporal_model",
            "action_head",
            "yaw_head",
        ]
        print("[Info] Encoder-only training disabled: full student training")
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iters, eta_min=1e-4)
    print(f"[Info] Fixed student depth crop enabled: left={DEPTH_CROP_LEFT_COLS} cols, resize back to {camera_resolution}")

    # Load checkpoint if provided
    start_iter = 0
    if args.student_checkpoint:
        print(f"[Info] Loading student checkpoint: {args.student_checkpoint}")
        ckpt = torch.load(args.student_checkpoint, map_location=device)
        student_model.load_state_dict(ckpt["model_state_dict"])
        # Optional: load optimizer if resuming DAgger run
        if "optimizer_state_dict" in ckpt and ckpt.get("meta", {}).get("is_dagger", False):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_iter = ckpt.get("iter", 0)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # 5. Initialize Helpers
    # Aggregator: Collects (obs, teacher_action, done) for TRAINING
    aggregator = SequenceAggregator(
        num_envs=args.num_envs,
        device=device,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        sequence_len=args.sequence_length,
        num_prop=proprio_dim,
        depth_shape=camera_resolution
    )

    # Inference Runner: Manages state for STUDENT ACTING
    student_runner = StudentOnlineRunner(
        model=student_model,
        num_envs=args.num_envs,
        proprio_dim=proprio_dim,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        camera_resolution=camera_resolution,
        device=device
    )
    student_runner.reset()

    # Dropout Manager
    from utils.dropout_manager import CameraDropoutManager
    dropout_manager = None
    if args.use_dropout:
        print("[Info] Camera Dropout Enabled.")
        dropout_manager = CameraDropoutManager(
            num_envs=args.num_envs,
            device=device,
            dt=vec_env.unwrapped.step_dt,
            prob_start_offline=0.0,
            prob_cam_offline=0.5,
            online_duration_range=(2.0, 10.0),
            offline_duration_range=(1.0, 5.0)
        )

    # 6. DAgger Loop
    # Important: 'train_mems' are persistent across batches to allow TBPTT
    train_mems: Optional[List[Tensor]] = None
    mem_dones = None

    # Stats buffers
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    track_progress_hist = deque(maxlen=100)
    current_returns = torch.zeros(args.num_envs, device=device)
    current_lengths = torch.zeros(args.num_envs, device=device)
    dones_bool = torch.zeros(args.num_envs, dtype=torch.bool, device=device)  # Track prev dones
    settle_manager = ResetSettleManager(args.num_envs, args.reset_settle_steps, device)
    if args.reset_settle_steps > 0:
        settle_s = args.reset_settle_steps * float(vec_env.unwrapped.step_dt)
        print(f"[Info] Reset settle enabled: {args.reset_settle_steps} steps (~{settle_s:.2f}s)")

    base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")
    num_goals = int(getattr(base_parkour, "num_goals", 0))

    print(f"[Info] Starting DAgger loop for {args.num_iters} iterations...")
    from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
    for it in range(start_iter, args.num_iters):
        iter_start = time.time()
        student_model.eval()  # Eval mode for rollout

        """ Collection Phase (Run until we have a full sequence batch) """
        batch_data = None
        while batch_data is None and simulation_app.is_running():
            settle_mask = settle_manager.active_mask()
            valid_mask = ~settle_mask

            # A. Get Raw Observation
            depth_image = extras["observations"]["depth_camera"]  # [N, 1, H, W] or [N, H, W]
            if depth_image.dim() == 4:
                depth_image = depth_image.squeeze(1)

            # B. Teacher Action (Expert) - Uses CLEAN observations
            with torch.no_grad():
                teacher_actions = teacher_policy(obs, hist_encoding=True)

            # C. Dropout
            # Create copies for student (augmented) vs teacher (clean)
            student_prop = obs[:, :proprio_dim].clone()
            _ , _, yaw = euler_xyz_from_quat(base_parkour.robot.data.root_quat_w)
            current_yaw = wrap_to_pi(yaw)
            student_prop[:, 6] = -current_yaw
            student_prop[:, 7] = 0
            extra_info = torch.cat([base_parkour.target_yaw.clone().unsqueeze(-1), base_parkour.next_target_yaw.clone().unsqueeze(-1)], dim=-1)

            # student_prop[:, 12] = dones_bool.float()
            student_depth = _left_crop_resize_depth(depth_image.clone())

            if dropout_manager:
                dropout_manager.reset_env(dones_bool)
                dropout_manager.update(depth_image=student_depth, obs_prop=student_prop)

            # D. Student Action (Learner) - Uses AUGMENTED observations
            student_actions = student_runner.act(
                student_prop,
                student_depth,
                prev_done=settle_manager.prev_txl_done,
            )

            goal_idx_before_step = None
            if base_parkour is not None and num_goals > 0:
                goal_idx_before_step = base_parkour.cur_goal_idx.detach().cpu().numpy().copy()

            # E. Action Selection (Mixture)
            if it < args.num_pretrain_iters:
                selected_actions = teacher_actions
            else:
                selected_actions = student_actions
                if args.teacher_mixture:
                    # Decay beta
                    progress = max(it - args.num_pretrain_iters, 0)
                    mix_frac = min(progress / max(args.beta_decay_iters, 1), 1.0)
                    beta = args.beta_start + (args.beta_end - args.beta_start) * mix_frac

                    # Sample mask
                    mask = torch.rand(args.num_envs, device=device) < beta
                    selected_actions = torch.where(mask.unsqueeze(-1), teacher_actions, student_actions)

            actions_to_env = selected_actions.clone()
            if settle_mask.any():
                actions_to_env[settle_mask] = 0.0

            # F. Step Environment
            obs, rewards, dones, extras = vec_env.step(actions_to_env)
            dones_bool = dones.squeeze(-1).bool()  # [N]
            txl_dones = dones_bool | settle_mask

            # G. Store in Aggregator
            batch_data = aggregator.push_step(
                obs_prop=student_prop,
                depth_frame=student_depth,
                teacher_actions=teacher_actions,
                done=txl_dones,  # TXL pseudo-episode boundary for reset settle.
                extra_info=extra_info,
                valid_mask=valid_mask,
            )

            # H. Handle resets and stats. TXL done also clears short token histories;
            # the temporal mem itself stays masked by txl_dones/full_dones.
            if txl_dones.any():
                student_runner.reset_done(txl_dones)

            rewards_f = rewards.squeeze(-1)
            if valid_mask.any():
                current_returns[valid_mask] += rewards_f[valid_mask]
                current_lengths[valid_mask] += 1

            active_dones = dones_bool & valid_mask
            if active_dones.any():
                done_indices = torch.nonzero(active_dones).squeeze(-1)
                for idx in done_indices:
                    ep_returns.append(current_returns[idx].item())
                    ep_lengths.append(current_lengths[idx].item())

                    if goal_idx_before_step is not None:
                        # 拿到该环境 step 前的 goal index
                        final_goal_idx = goal_idx_before_step[idx.item()]
                        # 计算归一化进度 (0.0 ~ 1.0)
                        progress = np.clip(final_goal_idx / num_goals, 0.0, 1.0)
                        track_progress_hist.append(progress)

                    current_returns[idx] = 0.0
                    current_lengths[idx] = 0.0

            settle_manager.step(dones_bool, txl_dones)

        """batch sequence data assembled, train on batch data"""
        if args.encoder_only_training:
            _set_encoder_only_train_mode(student_model)
        else:
            student_model.train()

        # Prepare Batch, Shape: [B, S, ...]
        b_prop = batch_data["proprio"].to(device)
        b_depth = batch_data["depth"].to(device)
        b_actions = batch_data["actions"].to(device)
        b_dones = batch_data["dones"].to(device)
        true_yaws = batch_data["extra_infos"].to(device)
        b_valid = batch_data["valid_mask"].to(device).float()

        full_dones = torch.cat([mem_dones, b_dones], dim=1) if mem_dones is not None else b_dones.clone()

        # Forward with segment recurrence
        pred_actions, pred_yaws, new_train_mems = student_model.forward_with_mems(
            b_prop, b_depth, mems=train_mems, full_dones=full_dones
        )
        valid_count = b_valid.sum().clamp_min(1.0)
        action_loss_per_step = torch.mean((pred_actions - b_actions) ** 2, dim=-1)
        yaw_loss_per_step = torch.mean((pred_yaws - true_yaws) ** 2, dim=-1)
        action_loss = (action_loss_per_step * b_valid).sum() / valid_count
        yaw_loss = (yaw_loss_per_step * b_valid).sum() / valid_count
        loss = action_loss + yaw_loss

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
        optimizer.step()
        #scheduler.step()

        # Update train_mems for next iteration
        if new_train_mems is not None:
            # shape of mems is [num_layers, num_envs, Mem_Len, D_Model], do Truncated BPTT
            train_mems = TransformerXLTemporal.detach_mems(new_train_mems)
        else:
            train_mems = None
        mem_dones = b_dones.clone()

        # --- Logging ---
        dt = time.time() - iter_start
        if (it + 1) % 10 == 0:
            with torch.no_grad():
                diff_rmse = torch.sqrt((action_loss_per_step * b_valid).sum() / valid_count).item()
                teacher_rms_per_step = torch.mean(b_actions ** 2, dim=-1)
                teacher_rms = torch.sqrt((teacher_rms_per_step * b_valid).sum() / valid_count).item()
                valid_ratio = b_valid.mean().item()

            print(f"[Iter {it+1}] Action Loss: {action_loss.item():.5f} | Yaw Loss: {yaw_loss.item():.5f} | Time: {dt:.2f}s | RMSE: {diff_rmse:.4f} | Valid: {valid_ratio:.3f}")

            if args.wandb and WANDB_AVAILABLE:
                log_data = {
                    "dagger/action_loss": action_loss.item(),
                    "dagger/yaw_loss": yaw_loss.item(),
                    "dagger/diff_rmse": diff_rmse,
                    "dagger/teacher_rms": teacher_rms,
                    "dagger/valid_ratio": valid_ratio,
                    "dagger/iter_time": dt,
                    "rollout/ep_return_mean": np.mean(ep_returns) if ep_returns else 0.0,
                    "rollout/ep_len_mean": np.mean(ep_lengths) if ep_lengths else 0.0,
                    "params/beta": beta if args.teacher_mixture and it >= args.num_pretrain_iters else (1.0 if it < args.num_pretrain_iters else 0.0)
                }

                if len(track_progress_hist) > 0:
                    track_arr = np.array(track_progress_hist)
                    log_data.update({
                        "rollout/track_progress_mean": float(np.mean(track_arr)),
                        # 记录跑完全程 70% 以上的比例
                        "rollout/track_progress_late_ratio": float(np.mean(track_arr >= 0.7)),
                    })

                # --- 新增: 地形难度等级 (Terrain Level) ---
                if base_parkour is not None:
                    # 获取当前所有环境地形等级的平均值
                    # terrain_levels 是一个 Tensor
                    avg_level = float(base_parkour.terrain.terrain_levels.float().mean().item())
                    log_data["rollout/terrain_level_mean"] = avg_level
                # ----------------------------------------

                wandb.log(log_data, step=it)

        # --- Checkpointing ---
        if (it + 1) % 2000 == 0:
            ckpt_path = save_dir / f"student_dagger_{it:04d}.pt"
            torch.save({
                "model_state_dict": student_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "iter": it + 1,
                "meta": {
                    "is_dagger": True,
                    "task": args.task,
                    "num_prop": proprio_dim,
                    "action_dim": action_dim,
                    "camera_resolution": camera_resolution,
                    "reset_settle_steps": args.reset_settle_steps,
                    "depth_crop_left_cols": DEPTH_CROP_LEFT_COLS,
                    "depth_crop_mode": "fixed_left_crop_resize",
                    "encoder_only_training": args.encoder_only_training,
                    "train_mode": train_mode_name,
                    "trainable_modules": trainable_modules,
                }
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    vec_env.close()
    simulation_app.close()
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
