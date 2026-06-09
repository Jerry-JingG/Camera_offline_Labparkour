from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

RSL_RL_DIR = os.path.join(PROJECT_ROOT, "scripts", "rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

TXL_STUDENT_ROOT = os.path.join(PROJECT_ROOT, "scripts", "txl_student")
if TXL_STUDENT_ROOT not in sys.path:
    sys.path.insert(0, TXL_STUDENT_ROOT)

from renet_policy import RENetBCPolicy, RENetOnlineRunner
from utils.dropout_manager import CameraDropoutManager
from utils.student_utils import load_env_and_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train RENet-style OP/VP GRU student via DAgger")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=64)

    parser.add_argument("--num_iters", type=int, default=500000)
    parser.add_argument("--num_pretrain_iters", type=int, default=10000)
    parser.add_argument("--teacher_mixture", action="store_true")
    parser.add_argument("--beta_start", type=float, default=0.8)
    parser.add_argument("--beta_end", type=float, default=0.0)
    parser.add_argument("--beta_decay_iters", type=int, default=60000)

    parser.add_argument("--prop_hist_len", type=int, default=10)
    parser.add_argument("--depth_hist_len", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--model_dropout", type=float, default=0.0)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--save_dir", type=str, default="outputs/renet/dagger")
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="renet-bc")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    import cli_args
    from isaaclab.app import AppLauncher

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    start_iter = int(payload.get("iter", 0))
    print(f"[checkpoint] Resumed from {path} at iter={start_iter}")
    return start_iter


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": iteration,
            "meta": meta,
        },
        path,
    )
    print(f"[checkpoint] Saved to {path}")


def teacher_mix_beta(args: argparse.Namespace, iteration: int) -> float:
    if iteration < args.num_pretrain_iters:
        return 1.0
    progress = max(iteration - args.num_pretrain_iters, 0)
    mix_frac = min(progress / max(args.beta_decay_iters, 1), 1.0)
    return args.beta_start + (args.beta_end - args.beta_start) * mix_frac


def checkpoint_meta(
    args: argparse.Namespace,
    proprio_dim: int,
    action_dim: int,
    camera_resolution: tuple[int, int],
) -> dict:
    return {
        "is_renet_bc": True,
        "task": args.task,
        "num_prop": proprio_dim,
        "action_dim": action_dim,
        "camera_resolution": camera_resolution,
        "prop_hist_len": args.prop_hist_len,
        "depth_hist_len": args.depth_hist_len,
        "hidden_dim": args.hidden_dim,
        "embed_dim": args.embed_dim,
        "uses_delta_yaw_input": True,
        "prob_cam_offline": 1.0,
    }


def main() -> None:  # noqa: C901
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        import wandb

        wandb_available = True
    except ImportError:
        wandb_available = False
        wandb = None  # type: ignore[assignment]
        if args.wandb:
            print("[warning] wandb requested but not installed.")

    if args.wandb and wandb_available:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    vec_env, teacher_policy, agent_cfg, simulation_app = load_env_and_teacher(args)
    obs, extras = vec_env.get_observations()

    depth_sample = extras["observations"].get("depth_camera")
    if depth_sample is None:
        raise RuntimeError("Env must provide depth_camera observations.")
    camera_resolution = (int(depth_sample.shape[-2]), int(depth_sample.shape[-1]))
    proprio_dim = int(agent_cfg.estimator.num_prop)
    action_dim = (
        vec_env.unwrapped.action_space.shape[1]
        if hasattr(vec_env.unwrapped.action_space, "shape")
        else int(obs.shape[1])
    )
    print(
        f"[Info] proprio={proprio_dim} action={action_dim} camera={camera_resolution} "
        f"prop_hist={args.prop_hist_len} depth_hist={args.depth_hist_len}"
    )

    model = RENetBCPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.model_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_iter = 0
    if args.student_checkpoint:
        start_iter = load_checkpoint(Path(args.student_checkpoint).expanduser().resolve(), model, optimizer, device)

    runner = RENetOnlineRunner(
        model=model,
        num_envs=args.num_envs,
        proprio_dim=proprio_dim,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        camera_resolution=camera_resolution,
        device=device,
    )
    runner.reset()

    dropout_manager = CameraDropoutManager(
            num_envs=args.num_envs,
            device=device,
            dt=float(vec_env.unwrapped.step_dt),
            prob_start_offline=0.0,
            prob_cam_offline=1.0,
            online_duration_range=(3.0, 15.0),
            offline_duration_range=(0.0, 5.0),
        )
    print("[Info] Camera dropout enabled with oracle OP/VP switching.")

    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    track_progress_hist = deque(maxlen=100)
    current_returns = torch.zeros(args.num_envs, device=device)
    current_lengths = torch.zeros(args.num_envs, device=device)
    prev_dones = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")
    num_goals = int(getattr(base_parkour, "num_goals", 0))

    print(f"[Info] Starting RENet-style DAgger for {args.num_iters} steps.")
    for it in range(start_iter, args.num_iters):
        if not simulation_app.is_running():
            break
        iter_start = time.time()
        model.train()

        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("Env must provide depth_camera observations.")
        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)

        with torch.no_grad():
            teacher_actions = teacher_policy(obs, hist_encoding=True)

        student_prop = obs[:, :proprio_dim].clone()
        student_depth = depth_image.clone()

        if dropout_manager is not None:
            dropout_manager.reset_env(prev_dones)
            dropout_manager.update(depth_image=student_depth, obs_prop=student_prop)
            use_op_mask = dropout_manager.offline_state.clone()
        else:
            use_op_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

        output = runner.forward_step(
            obs_prop=student_prop,
            depth_image=student_depth,
            use_op_mask=use_op_mask,
            prev_done=prev_dones,
        )
        student_actions = output["actions"]  # type: ignore[assignment]

        loss = F.mse_loss(student_actions, teacher_actions) 

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        beta = teacher_mix_beta(args, it)
        if it < args.num_pretrain_iters:
            actions_to_env = teacher_actions
        else:
            actions_to_env = student_actions.detach()
            if args.teacher_mixture and beta > 0.0:
                mix_mask = torch.rand(args.num_envs, device=device) < beta
                actions_to_env = torch.where(mix_mask.unsqueeze(-1), teacher_actions, actions_to_env)

        goal_idx_before_step = None
        if base_parkour is not None and num_goals > 0:
            goal_idx_before_step = base_parkour.cur_goal_idx.detach().cpu().numpy().copy()

        obs_next, rewards, dones, extras = vec_env.step(actions_to_env)
        done_mask = dones.squeeze(-1).bool()
        rewards_flat = rewards.squeeze(-1)

        current_returns += rewards_flat
        current_lengths += 1
        if done_mask.any():
            done_indices = torch.nonzero(done_mask).squeeze(-1)
            for idx in done_indices:
                ep_returns.append(float(current_returns[idx].item()))
                ep_lengths.append(float(current_lengths[idx].item()))
                if goal_idx_before_step is not None:
                    progress = np.clip(goal_idx_before_step[idx.item()] / max(num_goals, 1), 0.0, 1.0)
                    track_progress_hist.append(float(progress))
                current_returns[idx] = 0.0
                current_lengths[idx] = 0.0
            runner.reset_done(done_mask)

        obs = obs_next
        prev_dones = done_mask

        if (it + 1) % args.log_interval == 0:
            with torch.no_grad():
                diff_rmse = torch.sqrt(torch.mean((student_actions - teacher_actions) ** 2)).item()
                teacher_rms = torch.sqrt(torch.mean(teacher_actions**2)).item()
                dropout_rate = use_op_mask.float().mean().item()
                iter_time = time.time() - iter_start
            
            print(f"[Iter {it + 1}] loss={loss.item():.5f} | beta={beta:.3f} | rmse={diff_rmse:.4f} | time={iter_time:.2f}")

            if args.wandb and wandb_available:
                log_data = {
                    "train/loss": loss.item(),
                    "train/diff_rmse": diff_rmse,
                    "teacher/action_rms": teacher_rms,
                    "dropout/rate": dropout_rate,
                    "params/beta": beta,
                    "perf/iter_time_s": iter_time,
                    "rollout/ep_return_mean": float(np.mean(ep_returns)) if ep_returns else 0.0,
                    "rollout/ep_len_mean": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
                }
                if track_progress_hist:
                    progress_arr = np.array(track_progress_hist)
                    log_data["rollout/track_progress_mean"] = float(np.mean(progress_arr))
                    log_data["rollout/track_progress_late_ratio"] = float(np.mean(progress_arr >= 0.7))
                if base_parkour is not None:
                    log_data["rollout/terrain_level_mean"] = float(
                        base_parkour.terrain.terrain_levels.float().mean().item()
                    )
                wandb.log(log_data, step=it)

        if (it + 1) % args.save_interval == 0:
            save_checkpoint(
                save_dir / f"renet_dagger_{it + 1:05d}.pt",
                model=model,
                optimizer=optimizer,
                iteration=it + 1,
                meta=checkpoint_meta(args, proprio_dim, action_dim, camera_resolution),
            )

    final_iter = it + 1 if "it" in locals() else start_iter
    save_checkpoint(
        save_dir / "renet_final.pt",
        model=model,
        optimizer=optimizer,
        iteration=final_iter,
        meta=checkpoint_meta(args, proprio_dim, action_dim, camera_resolution),
    )

    vec_env.close()
    simulation_app.close()
    if args.wandb and wandb_available:
        wandb.finish()


if __name__ == "__main__":
    main()
