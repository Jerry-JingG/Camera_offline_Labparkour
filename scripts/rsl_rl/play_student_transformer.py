#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play / validate the Transformer student policy in simulation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


def _ensure_repo_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    parkour_tasks_root = os.path.join(project_root, "parkour_tasks")
    if parkour_tasks_root not in sys.path:
        sys.path.insert(0, parkour_tasks_root)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Transformer student policy.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
    parser.add_argument("--student_checkpoint", type=str, required=True, help="Path to student checkpoint.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric/USD I/O.")
    parser.add_argument("--disable_depth_debug_vis", action="store_true", help="Disable cv2 depth debug window.")

    # Student history config (must match training)
    parser.add_argument("--prop_hist_len", type=int, default=3)
    parser.add_argument("--depth_hist_len", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=64, help="TXL mem_len used for inference.")

    # runtime controls
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0 = run until closed).")
    parser.add_argument("--print_interval", type=int, default=200, help="Print rollout stats every N steps.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match sim dt.")

    # optional video recording (rgb_array)
    parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
    parser.add_argument("--video_length", type=int, default=500, help="Video length in steps.")

    # append standard arguments
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_arg_parser()
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _extract_depth(observations: Dict) -> torch.Tensor:
    depth = observations.get("depth_camera")
    if depth is None:
        raise RuntimeError("depth_camera not found in observations; use a task that provides depth_camera.")
    if isinstance(depth, dict):
        depth = next(iter(depth.values()))
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"depth_camera must be a torch.Tensor, got {type(depth)}")
    # [B, H, W, 1] -> [B, H, W]
    if depth.dim() == 4 and depth.size(-1) == 1:
        depth = depth.squeeze(-1)
    if depth.dim() != 3:
        raise ValueError(f"Expected depth to have shape [B, H, W], got {tuple(depth.shape)}")
    return depth


class HistoryBuffer:
    def __init__(self, num_envs: int, prop_hist_len: int, depth_hist_len: int, prop_dim: int, depth_shape: Tuple[int, int]):
        from collections import deque

        self.num_envs = int(num_envs)
        self.prop_hist_len = int(prop_hist_len)
        self.depth_hist_len = int(depth_hist_len)
        self.prop_dim = int(prop_dim)
        self.depth_shape = (int(depth_shape[0]), int(depth_shape[1]))
        self.prop_histories = [deque(maxlen=self.prop_hist_len) for _ in range(self.num_envs)]
        self.depth_histories = [deque(maxlen=self.depth_hist_len) for _ in range(self.num_envs)]
        self._zeros_prop = np.zeros((self.prop_dim,), dtype=np.float32)
        self._zeros_depth = np.zeros(self.depth_shape, dtype=np.float32)
        for env_id in range(self.num_envs):
            self.reset_env(env_id)

    def reset_env(self, env_id: int) -> None:
        self.prop_histories[env_id].clear()
        self.depth_histories[env_id].clear()
        for _ in range(self.prop_hist_len):
            self.prop_histories[env_id].append(self._zeros_prop.copy())
        for _ in range(self.depth_hist_len):
            self.depth_histories[env_id].append(self._zeros_depth.copy())

    def append_and_reset(self, obs_prop_np, depth_np, dones_np) -> None:
        for env_id in range(self.num_envs):
            self.prop_histories[env_id].append(obs_prop_np[env_id])
            self.depth_histories[env_id].append(depth_np[env_id])
            if bool(dones_np[env_id]):
                self.reset_env(env_id)

    def build_inputs_with_current(self, obs_prop_np, depth_np):
        prop_batch = []
        depth_batch = []
        for env_id in range(self.num_envs):
            prop_hist = list(self.prop_histories[env_id]) + [obs_prop_np[env_id]]
            depth_hist = list(self.depth_histories[env_id]) + [depth_np[env_id]]
            prop_batch.append(np.concatenate(prop_hist[-self.prop_hist_len :], axis=0))
            depth_batch.append(np.stack(depth_hist[-self.depth_hist_len :], axis=0))
        return np.stack(prop_batch), np.stack(depth_batch)


def main() -> None:
    _ensure_repo_on_path()
    import parkour_tasks  # noqa: F401  # trigger gym env registration

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import parse_env_cfg
    from train_student_from_dataset import MultiModalStudentPolicy
    from vecenv_wrapper import ParkourRslRlVecEnvWrapper

    print("[DEBUG] parsing env_cfg...")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=getattr(args_cli, "device", None),
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # 屏蔽 delta_yaw_ok 观测组，避免空 shape 触发 ObservationManager 维度拼接错误
    if hasattr(env_cfg.observations, "delta_yaw_ok"):
        env_cfg.observations.delta_yaw_ok = None

    # Optionally disable depth debug window (cv2.imshow).
    if getattr(args_cli, "disable_depth_debug_vis", False):
        try:
            depth_cam_cfg = env_cfg.observations.depth_camera.depth_cam
            if "debug_vis" in depth_cam_cfg.params:
                depth_cam_cfg.params["debug_vis"] = False
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to disable depth debug_vis: {exc}")

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    print("[DEBUG] creating env via gym.make...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print(f"[DEBUG] gym.make done, env type={type(env)}")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        print(f"[DEBUG] converted to single-agent env: {type(env)}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(args_cli.student_checkpoint), "videos", "play_student_txl"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print("[DEBUG] wrapping env with ParkourRslRlVecEnvWrapper...")
    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("[DEBUG] VecEnv wrapper constructed.")

    # 在 GUI 模式下，时间线默认是暂停的。此时环境已初始化完成，再触发播放以避免在构造过程中触发回调。
    if not getattr(args_cli, "headless", False):
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().play()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to auto-play timeline: {exc}")

    student_device = torch.device(args_cli.device)

    # infer dims
    print("[DEBUG] fetching initial observations...")
    obs, extras = vec_env.get_observations()
    print(f"[DEBUG] got observations: obs shape={tuple(obs.shape)}, keys={list(extras['observations'].keys())}")
    obs = obs.to(student_device)
    depth0 = _extract_depth(extras["observations"]).to(student_device)
    cam_res = (int(depth0.shape[-2]), int(depth0.shape[-1]))

    estimator_cfg = agent_cfg.to_dict()["estimator"]
    num_prop = int(estimator_cfg["num_prop"])
    action_dim = int(vec_env.num_actions)

    fusion_cfg = {"num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.1, "attn_dropout": 0.1, "grid_size": 4}
    temporal_cfg = {"num_layers": 3, "num_heads": 4, "d_inner": 256, "mem_len": args_cli.sequence_length, "dropout": 0.1, "attn_dropout": 0.1}
    action_head_cfg = {"hidden_dims": (256, 256), "tanh_output": False, "action_scale": 1.0}

    student = MultiModalStudentPolicy(
        proprio_dim=num_prop,
        action_dim=action_dim,
        camera_resolution=cam_res,
        prop_hist_len=args_cli.prop_hist_len,
        depth_hist_len=args_cli.depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
    ).to(student_device)
    student.eval()

    # load checkpoint
    ckpt = torch.load(args_cli.student_checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    student.load_state_dict(state_dict)
    print(f"[INFO] Loaded student checkpoint: {args_cli.student_checkpoint}")

    history = HistoryBuffer(
        num_envs=args_cli.num_envs,
        prop_hist_len=args_cli.prop_hist_len,
        depth_hist_len=args_cli.depth_hist_len,
        prop_dim=num_prop,
        depth_shape=cam_res,
    )
    txl_mems = None

    ep_returns = np.zeros(args_cli.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args_cli.num_envs, dtype=np.int64)
    ep_return_hist = deque(maxlen=200)
    ep_len_hist = deque(maxlen=200)

    dt = float(getattr(vec_env.unwrapped, "step_dt", 0.0))
    step = 0
    while simulation_app.is_running():
        print(f"[DEBUG] loop start, step={step}, app_running={simulation_app.is_running()}")
        step_start_t = time.time()
        if args_cli.max_steps and step >= args_cli.max_steps:
            break

        depth = _extract_depth(extras["observations"]).to(student_device)
        obs_prop = obs[:, :num_prop]

        obs_prop_np = obs_prop.detach().cpu().numpy().astype(np.float32)
        depth_np = depth.detach().cpu().numpy().astype(np.float32)

        prop_in_np, depth_in_np = history.build_inputs_with_current(obs_prop_np, depth_np)
        prop_in = torch.from_numpy(prop_in_np).to(student_device)
        depth_in = torch.from_numpy(depth_in_np).to(student_device)

        with torch.inference_mode():
            actions, new_mems = student.forward_step(prop_in, depth_in, mems=txl_mems)

        print(f"[DEBUG] step {step}: before env.step")
        obs_next, rewards, dones, infos = vec_env.step(actions.to(vec_env.device))
        print(f"[DEBUG] step {step}: after env.step, reward_mean={rewards.mean().item():.4f}")

        rewards_np = rewards.squeeze(-1).detach().cpu().numpy()
        ep_returns += rewards_np
        ep_lengths += 1

        dones_np = dones.detach().cpu().numpy().astype(bool).reshape(-1)
        if dones_np.any():
            done_ids = np.nonzero(dones_np)[0]
            ep_return_hist.extend(ep_returns[done_ids].tolist())
            ep_len_hist.extend(ep_lengths[done_ids].tolist())
            ep_returns[done_ids] = 0.0
            ep_lengths[done_ids] = 0

        # 更新 TXL memory，并对 done 的环境清零
        if new_mems is not None:
            done_mask = torch.from_numpy(dones_np).to(student_device)
            for mem in new_mems:
                if mem is None or mem.numel() == 0:
                    continue
                mem[done_mask] = 0.0
            txl_mems = new_mems

        # 更新历史缓存
        history.append_and_reset(obs_prop_np, depth_np, dones_np)

        obs = obs_next.to(student_device)
        if isinstance(infos, dict) and "observations" in infos:
            extras = infos
        step += 1

        if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
            if len(ep_return_hist) > 0:
                print(
                    f"[step {step}] "
                    f"ep_return_mean={float(np.mean(ep_return_hist)):.2f} "
                    f"ep_len_mean={float(np.mean(ep_len_hist)):.1f}"
                )
            else:
                print(f"[step {step}] (no completed episodes yet)")

        if args_cli.real_time and dt > 0:
            sleep_time = dt - (time.time() - step_start_t)
            if sleep_time > 0:
                time.sleep(sleep_time)

    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
