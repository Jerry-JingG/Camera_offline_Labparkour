"""
运行示例:
python scripts/rsl_rl/evaluate_student.py --task Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Eval-v0//
--student_checkpoint path_to_your_txl_student.pt --device cuda:0//
--mem_len 128 --prop_hist_len 1 --depth_hist_len 1 --headless (--use_dropout)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import collections
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util
import numpy as np
import statistics
from tqdm import tqdm

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


def parse_args_eval() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an offline-trained student policy in Isaac parkour envs.")
    parser.add_argument("--task", type=str, required=True, help="Isaac task name (must contain 'Eval').")
    parser.add_argument("--student_checkpoint", type=str, default=None, help="Path to a specific student checkpoint file.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing student_epoch_*.pt; used when --student_checkpoint is not provided.",
    )
    parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments for evaluation.")
    parser.add_argument("--total_steps", type=int, default=2000, help="Number of steps to evaluate.")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="History length for depth tokens.")
    parser.add_argument("--mem_len", type=int, default=64, help="TransformerXL memory length (S).")
    parser.add_argument("--use_dropout", action="store_true", default=False, help="Simulate camera dropout.")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args_eval()

    # ---------------- 强制评估标准限定 ----------------
    if args.task.find('Eval') == -1:
        print("[Error] The --task argument must contain 'Eval' to use the correct evaluation configuration.")
        return

    headless = getattr(args, "headless", False)
    disable_fabric = getattr(args, "disable_fabric", False)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import parkour_tasks  # noqa: F401
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

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not headless else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---------------- 加载学生策略模型 ----------------
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
        print("[Eval] Camera Dropout Simulation: ENABLED")

    obs, extras = vec_env.get_observations()
    dones_bool = torch.zeros(vec_env.num_envs, device=device, dtype=torch.bool)

    # ------------------ Evaluation 指标初始化 ------------------
    total_steps = args.total_steps
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    edge_violation_buffer = deque(maxlen=total_steps)
    num_waypoints_per_terrain = collections.defaultdict(list)

    # 获取每个 env 的地形 ID (Shape: [num_envs])
    try:
        terrain_types_tensor = vec_env.unwrapped.scene.terrain.terrain_types
    except AttributeError:
        print("[Warning] Could not find terrain_types. Make sure the terrain generator exposes it.")
        terrain_types_tensor = torch.zeros(vec_env.num_envs, dtype=torch.long, device=device)

    # 尝试映射地形 ID 到字符串名字 (提取 active 的 sub_terrains)
    terrain_names = {}
    try:
        sub_terrains = vec_env.unwrapped.cfg.scene.terrain.terrain_generator.sub_terrains
        for i, name in enumerate(sub_terrains.keys()):
            terrain_names[i] = name
    except Exception as e:
        print(f"[Warning] Failed to map terrain names: {e}")

    # 使用 1D 向量来追踪以便避免潜在的张量广播 Shape 错误
    cur_reward_sum = torch.zeros(vec_env.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(vec_env.num_envs, dtype=torch.float, device=device)

    try:
        reward_feet_edge = vec_env.unwrapped.reward_manager.get_term_cfg("reward_feet_edge").func
        base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")
    except Exception as e:
        print(f"[Warning] Could not extract specific reward/parkour components: {e}")
        reward_feet_edge = None
        base_parkour = None

    print(f"\n[Info] Starting evaluation for {total_steps} steps on {vec_env.num_envs} environments...")

    from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
    for _ in tqdm(range(total_steps)):
        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用包含相机观测的环境配置。")

        obs_prop = obs[:, :proprio_dim]
        _ , _, yaw = euler_xyz_from_quat(base_parkour.robot.data.root_quat_w)
        current_yaw = wrap_to_pi(yaw)
        obs_prop[:, 6] = -current_yaw
        obs_prop[:, 7] = 0
        # obs_prop[:, 12] = dones_bool.float()

        if dropout_manager:
            dropout_manager.reset_env(dones_bool)
            dropout_manager.update(depth_image=depth_image, obs_prop=obs_prop)

        # 预测 Action
        student_action = runner.act(obs_prop, depth_image, prev_done=dones_bool)

        # Snapshot current goal index before step
        if base_parkour is not None:
            cur_goal_idx = base_parkour.cur_goal_idx.clone().view(-1)

        # 环境推演
        obs_next, rews, dones, extras = vec_env.step(student_action)
        dones_bool = dones.squeeze(-1).bool()

        # ----------------- 更新统计指标 -----------------
        if reward_feet_edge is not None:
            # sum(dim=1) 获取每一个环境的违规次数统计
            edge_violation_buffer.extend(reward_feet_edge.feet_at_edge.sum(dim=1).float().cpu().numpy().tolist())

        cur_reward_sum += rews.view(-1)
        cur_episode_length += 1

        # 提取结束环境的索引以将当前统计追加到全局 Buffer 中
        done_indices = dones_bool.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            rewbuffer.extend(cur_reward_sum[done_indices].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[done_indices].cpu().numpy().tolist())
            if base_parkour is not None:
                waypoints = cur_goal_idx[done_indices].cpu().numpy().tolist()
                num_waypoints_buffer.extend(waypoints)

                # 记录每种地形上跑了多少个waypoints
                done_terrain_ids = terrain_types_tensor[done_indices].cpu().numpy().tolist()
                for t_id, wp in zip(done_terrain_ids, waypoints):
                    num_waypoints_per_terrain[t_id].append(wp)

            # 及时重置 Done 掉的环境（原 evaluation.py 里似乎忘了清零episode_length）
            cur_reward_sum[done_indices] = 0.0
            cur_episode_length[done_indices] = 0.0

        if dones_bool.any():
            runner.reset_done(dones_bool)

        obs = obs_next

    # ------------------ 打印统计结果 ------------------
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    if len(rewbuffer) > 0:
        rew_mean = statistics.mean(rewbuffer)
        rew_std = statistics.stdev(rewbuffer) if len(rewbuffer) > 1 else 0.0
        print("Mean reward:             {:.2f} \u00B1 {:.2f}".format(rew_mean, rew_std))

        len_mean = statistics.mean(lenbuffer)
        len_std = statistics.stdev(lenbuffer) if len(lenbuffer) > 1 else 0.0
        print("Mean episode length:     {:.2f} \u00B1 {:.2f}".format(len_mean, len_std))
    else:
        print("No episodes finished during evaluation.")

    if len(num_waypoints_buffer) > 0:
        num_waypoints_mean = np.mean(np.array(num_waypoints_buffer).astype(float)/7.0)  # 尽管total_waypoints是8个，isaaclab parkour源码是除以的7.0
        num_waypoints_std = np.std(np.array(num_waypoints_buffer).astype(float)/7.0)
        print("Mean number of waypoints: {:.2f} \u00B1 {:.2f}".format(num_waypoints_mean, num_waypoints_std))

    if len(edge_violation_buffer) > 0:
        edge_violation_mean = np.mean(edge_violation_buffer)
        edge_violation_std = np.std(edge_violation_buffer)
        # 该值统计了机器人的脚（足端）踩在地形边缘（即将滑落、踩空）的次数，应当越低越好
        print("Mean edge violation:      {:.2f} \u00B1 {:.2f}".format(edge_violation_mean, edge_violation_std))

    if len(num_waypoints_per_terrain) > 0:
        print("\n--- Completion Rate by Terrain Type ---")
        # 按照 Terrain ID 排序打印
        for t_id, wps in sorted(num_waypoints_per_terrain.items()):
            t_name = terrain_names.get(t_id, f"TerrainType_{t_id}")
            mean_val = np.mean(np.array(wps) / 7.0)
            std_val = np.std(np.array(wps) / 7.0)
            print(f"{t_name:<18}: {mean_val:.2f} \u00B1 {std_val:.2f} (from {len(wps)} episodes)")
    print("="*50)

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
