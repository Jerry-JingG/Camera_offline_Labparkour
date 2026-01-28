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
from typing import Dict, Tuple, List

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
    parser.add_argument("--prop_hist_len", type=int, default=1)
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


def load_student_from_checkpoint(
    ckpt_path: str | os.PathLike,
    device: torch.device,
    prop_hist_len: int,
    depth_hist_len: int,
    sequence_length: int,
    fallback_dims: Tuple[int, int, Tuple[int, int]],
):
    """Load student model using metadata stored in checkpoint (dataset-style)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    if meta:
        proprio_dim = int(meta["num_prop"])
        action_dim = int(meta["action_dim"])
        camera_resolution = tuple(meta.get("camera_resolution", fallback_dims[2]))  # type: ignore[arg-type]
        mem_len = int(meta.get("sequence_length", sequence_length))
    else:
        proprio_dim, action_dim, camera_resolution = fallback_dims
        mem_len = sequence_length

    fusion_cfg = {"num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.1, "attn_dropout": 0.1, "grid_size": 4}
    temporal_cfg = {
        "num_layers": 3,
        "num_heads": 4,
        "d_inner": 256,
        "mem_len": mem_len,
        "dropout": 0.1,
        "attn_dropout": 0.1,
    }
    action_head_cfg = {"hidden_dims": (256, 256), "tanh_output": False, "action_scale": 1.0}

    student = MultiModalStudentPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,  # type: ignore[arg-type]
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
    ).to(device)
    student.load_state_dict(state_dict)
    student.eval()
    return student, proprio_dim, action_dim, camera_resolution


class StudentOnlineRunner:
    """Maintain short histories + TXL memories and run single-step inference."""

    def __init__(
        self,
        model: MultiModalStudentPolicy,
        num_envs: int,
        proprio_dim: int,
        prop_hist_len: int,
        depth_hist_len: int,
        camera_resolution: Tuple[int, int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.num_envs = num_envs
        self.proprio_dim = proprio_dim
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.camera_resolution = camera_resolution
        self.device = device
        self.num_layers = len(self.model.temporal_model.layers)

        self.prop_histories: List[deque] = [deque(maxlen=prop_hist_len) for _ in range(num_envs)]
        self.depth_histories: List[deque] = [deque(maxlen=depth_hist_len) for _ in range(num_envs)]
        self.mems: List[torch.Tensor] | None = None
        self.reset()

    def reset(self) -> None:
        """Pre-fill history with zeros so first step is ready, and clear TXL mems."""
        for env_id in range(self.num_envs):
            self.prop_histories[env_id].clear()
            self.depth_histories[env_id].clear()
            for _ in range(self.prop_hist_len):
                self.prop_histories[env_id].append(torch.zeros(self.proprio_dim, device=self.device))
            for _ in range(self.depth_hist_len):
                self.depth_histories[env_id].append(torch.zeros(*self.camera_resolution, device=self.device))
        self.mems = None

    def reset_done(self, done_mask: torch.Tensor) -> None:
        """Clear histories and TXL mems for envs that are done."""
        done_mask = done_mask.to(self.device)
        for env_id, done in enumerate(done_mask):
            if bool(done):
                self.prop_histories[env_id].clear()
                self.depth_histories[env_id].clear()
                for _ in range(self.prop_hist_len):
                    self.prop_histories[env_id].append(torch.zeros(self.proprio_dim, device=self.device))
                for _ in range(self.depth_hist_len):
                    self.depth_histories[env_id].append(torch.zeros(*self.camera_resolution, device=self.device))
        if self.mems is not None:
            for mem in self.mems:
                if mem is None or mem.numel() == 0:
                    continue
                mem[done_mask] = 0.0

    def act(self, obs_prop: torch.Tensor, depth_image: torch.Tensor) -> torch.Tensor:
        obs_prop = obs_prop.to(self.device)
        depth_image = depth_image.to(self.device)

        if depth_image.dim() == 4 and depth_image.shape[-1] == 1:
            depth_image = depth_image.squeeze(-1)
        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)

        for env_id in range(self.num_envs):
            cur_obs = obs_prop[env_id].clone() 
            
            self.prop_histories[env_id].append(cur_obs)
            self.depth_histories[env_id].append(depth_image[env_id])

        prop_batch = []
        depth_batch = []
        for env_id in range(self.num_envs):
            prop_stack = torch.cat(list(self.prop_histories[env_id])[-self.prop_hist_len :], dim=0)
            depth_stack = torch.stack(list(self.depth_histories[env_id])[-self.depth_hist_len :], dim=0)
            prop_batch.append(prop_stack)
            depth_batch.append(depth_stack)

        prop_batch_t = torch.stack(prop_batch)  # [B, prop_hist_len * proprio_dim]
        depth_batch_t = torch.stack(depth_batch)  # [B, depth_hist_len, H, W]

        with torch.no_grad():
             # Fix unpacking: actions, _, new_mems
            actions_step, _, new_mems = self.model.forward_step(prop_batch_t, depth_batch_t, mems=self.mems)
        
        self.mems = new_mems
        return actions_step


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

    # infer dims from env as fallback
    print("[DEBUG] fetching initial observations...")
    obs, extras = vec_env.get_observations()
    print(f"[DEBUG] got observations: obs shape={tuple(obs.shape)}, keys={list(extras['observations'].keys())}")
    depth0 = _extract_depth(extras["observations"]).to(student_device)
    fallback_cam_res = (int(depth0.shape[-2]), int(depth0.shape[-1]))
    fallback_num_prop = int(agent_cfg.estimator.num_prop)
    fallback_action_dim = int(vec_env.num_actions)

    student, num_prop, action_dim, cam_res = load_student_from_checkpoint(
        args_cli.student_checkpoint,
        device=student_device,
        prop_hist_len=args_cli.prop_hist_len,
        depth_hist_len=args_cli.depth_hist_len,
        sequence_length=args_cli.sequence_length,
        fallback_dims=(fallback_num_prop, fallback_action_dim, fallback_cam_res),
    )
    print(f"[INFO] Loaded student checkpoint: {args_cli.student_checkpoint}")

    runner = StudentOnlineRunner(
        model=student,
        num_envs=args_cli.num_envs,
        proprio_dim=num_prop,
        prop_hist_len=args_cli.prop_hist_len,
        depth_hist_len=args_cli.depth_hist_len,
        camera_resolution=cam_res,
        device=student_device,
    )

    ep_returns = np.zeros(args_cli.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args_cli.num_envs, dtype=np.int64)
    ep_return_hist = deque(maxlen=200)
    ep_len_hist = deque(maxlen=200)

    dt = float(getattr(vec_env.unwrapped, "step_dt", 0.0))
    step = 0
    while simulation_app.is_running():
        step_start_t = time.time()
        if args_cli.max_steps and step >= args_cli.max_steps:
            break

        depth = _extract_depth(extras["observations"]).to(student_device)
        obs_prop = obs[:, :num_prop]

        actions = runner.act(obs_prop, depth)
        obs_next, rewards, dones, infos = vec_env.step(actions.to(vec_env.device))

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
        runner.reset_done(dones.squeeze(-1).bool())

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
