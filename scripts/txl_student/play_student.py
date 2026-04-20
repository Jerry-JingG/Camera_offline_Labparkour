"""
运行示例:
python scripts/rsl_rl/play_student.py --task Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0//
--student_checkpoint path_to_your_txl_student.pt --device cuda:0 --num_envs 8//
--mem_len 128 --prop_hist_len 1 --depth_hist_len 1 --headless (--use_dropout)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util
import numpy as np

import torch

from isaaclab.app import AppLauncher

# Ensure project roots are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

RSL_RL_DIR = os.path.join(PROJECT_ROOT, "scripts", "rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

import cli_args  # isort: skip

# Load MultiModalStudentPolicy directly
from utils.student_utils import (
    find_latest_student_checkpoint,
    load_student_policy_for_play,
    StudentOnlineRunner
)


def parse_args_play() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an offline-trained student policy in Isaac parkour envs.")
    parser.add_argument("--task", type=str, required=True, help="Isaac task name.")
    parser.add_argument("--student_checkpoint", type=str, default=None, help="Path to a specific student checkpoint file.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing student_epoch_*.pt; used when --student_checkpoint is not provided.",
    )
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="History length for depth tokens.")
    parser.add_argument("--mem_len", type=int, default=64, help="TransformerXL memory length (S).")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps to run.")
    parser.add_argument("--use_dropout", action="store_true", default=False, help="Simulate camera dropout.")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args_play()
    headless = getattr(args, "headless", False)
    disable_fabric = getattr(args, "disable_fabric", False)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import parkour_tasks  # noqa: F401  # ensure tasks register after Isaac app is initialized

    import gymnasium as gym
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from vecenv_wrapper import ParkourRslRlVecEnvWrapper

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not disable_fabric,
    )
    if not headless:
        env_cfg.viewer.origin_type = "world"
        spacing = float(env_cfg.scene.env_spacing)
        grid = int(np.ceil(np.sqrt(args.num_envs)))
        env_cfg.viewer.eye = [spacing * grid * 0.5, spacing * grid * 0.5, 3.0]
        env_cfg.viewer.lookat = [0.0, 0.0, 0.5]
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not headless else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args.student_checkpoint is not None:
        ckpt_candidate = Path(args.student_checkpoint).expanduser().resolve()
        if ckpt_candidate.is_dir():
            ckpt_path = find_latest_student_checkpoint(ckpt_candidate)
        else:
            ckpt_path = ckpt_candidate
    else:
        if args.checkpoint_dir is None:
            raise ValueError("Either --student_checkpoint or --checkpoint_dir must be provided.")
        ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
        ckpt_path = find_latest_student_checkpoint(ckpt_dir)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    device = torch.device(args.device)
    student_model, meta = load_student_policy_for_play(
        ckpt_path,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        device=device,
        mem_len=args.mem_len,
        token_dim=128
    )
    proprio_dim = int(meta["num_prop"])
    camera_resolution = tuple(meta.get("camera_resolution", (58, 87)))

    runner = StudentOnlineRunner(
        model=student_model,
        num_envs=vec_env.num_envs,
        proprio_dim=proprio_dim,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        camera_resolution=camera_resolution,  # type: ignore[arg-type]
        device=device,
    )
    runner.reset()

    from utils.dropout_manager import CameraDropoutManager
    dropout_manager = None
    if args.use_dropout:
        step_dt = float(vec_env.unwrapped.step_dt)
        dropout_manager = CameraDropoutManager(
            num_envs=args.num_envs,
            device=device,
            dt=step_dt,
            prob_start_offline=0.0,
            prob_cam_offline=1.0,
            online_duration_range=(5.0, 5.0),
            offline_duration_range=(2.0, 2.0)
        )
        print("[Play] Camera Dropout Simulation: ENABLED")

    obs, extras = vec_env.get_observations()
    dones_bool = torch.zeros(vec_env.num_envs, device=device, dtype=torch.bool)
    step = 0

    from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
    base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")

    while simulation_app.is_running() and step < args.max_steps:
        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用 TeacherCam 任务。")
        obs_prop = obs[:, :proprio_dim]
        _ , _, yaw = euler_xyz_from_quat(base_parkour.robot.data.root_quat_w)
        current_yaw = wrap_to_pi(yaw)
        obs_prop[:, 6] = -current_yaw
        obs_prop[:, 7] = 0
        obs_prop[:, 12] = dones_bool.float()

        if dropout_manager:
            dropout_manager.reset_env(dones_bool)
            dropout_manager.update(depth_image=depth_image, obs_prop=obs_prop)

        student_action = runner.act(obs_prop, depth_image)

        if step % 50 == 0:
            try:
                mean_norm = student_action.norm(dim=-1).mean().item()
            except Exception:
                mean_norm = float("nan")
            print(f"[student_play] step={step} mean_action_norm={mean_norm:.6f}")

        obs_next, rews, dones, extras = vec_env.step(student_action)
        dones_bool = dones.squeeze(-1).bool()

        # Vectorized reset for done envs
        if dones_bool.any():
            runner.reset_done(dones_bool)

        obs = obs_next
        step += 1

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
