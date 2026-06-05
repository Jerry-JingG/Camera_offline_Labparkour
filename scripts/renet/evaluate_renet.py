"""
Example:
python scripts/renet/evaluate_renet.py \
  --task Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Eval-v0 \
  --student_checkpoint outputs/renet/dagger/renet_final.pt \
  --device cuda:0 \
  --num_envs 256 \
  --total_steps 2000 \
  --headless \
  --use_dropout
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher


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

import cli_args  # isort: skip
from renet_policy import RENetBCPolicy, RENetOnlineRunner  # isort: skip
from utils.dropout_manager import CameraDropoutManager  # isort: skip


def parse_args_eval() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RENet-style BC policy in Isaac parkour envs.")
    parser.add_argument("--task", type=str, required=True, help="Isaac task name (must contain 'Eval').")
    parser.add_argument(
        "--student_checkpoint",
        "--renet_checkpoint",
        dest="student_checkpoint",
        type=str,
        default=None,
        help="Path to a specific RENet checkpoint file, or a directory containing RENet checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing renet_final.pt or renet_dagger_*.pt checkpoints.",
    )
    parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments.")
    parser.add_argument("--total_steps", type=int, default=2000, help="Number of steps to evaluate.")
    parser.add_argument("--prop_hist_len", type=int, default=None, help="Override proprioception history length.")
    parser.add_argument("--depth_hist_len", type=int, default=None, help="Override depth history length.")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Override GRU hidden dimension.")
    parser.add_argument("--embed_dim", type=int, default=None, help="Override branch embedding dimension.")
    parser.add_argument("--use_dropout", action="store_true", default=False, help="Simulate camera dropout.")
    parser.add_argument("--prob_start_offline", type=float, default=0.0, help="Initial dropout probability.")
    parser.add_argument("--online_duration_min", type=float, default=5.0, help="Minimum online duration in seconds.")
    parser.add_argument("--online_duration_max", type=float, default=5.0, help="Maximum online duration in seconds.")
    parser.add_argument("--offline_duration_min", type=float, default=2.0, help="Minimum offline duration in seconds.")
    parser.add_argument("--offline_duration_max", type=float, default=2.0, help="Maximum offline duration in seconds.")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def find_latest_renet_checkpoint(ckpt_dir: Path) -> Path:
    """Return renet_final.pt if present, otherwise the highest numbered renet_dagger checkpoint."""
    final_path = ckpt_dir / "renet_final.pt"
    if final_path.is_file():
        return final_path

    candidates = list(ckpt_dir.glob("renet_dagger_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No RENet checkpoints found in {ckpt_dir}")

    latest_iter = -1
    latest_path: Optional[Path] = None
    pattern = re.compile(r"renet_dagger_(\d+)\.pt")
    for path in candidates:
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration > latest_iter:
            latest_iter = iteration
            latest_path = path

    if latest_path is None:
        raise FileNotFoundError(f"No checkpoints matched pattern renet_dagger_*.pt in {ckpt_dir}")
    return latest_path


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.student_checkpoint is not None:
        ckpt_candidate = Path(args.student_checkpoint).expanduser().resolve()
        if ckpt_candidate.is_dir():
            return find_latest_renet_checkpoint(ckpt_candidate)
        return ckpt_candidate

    if args.checkpoint_dir is None:
        raise ValueError("Either --student_checkpoint/--renet_checkpoint or --checkpoint_dir must be provided.")
    return find_latest_renet_checkpoint(Path(args.checkpoint_dir).expanduser().resolve())


def get_action_dim(env: Any, obs: torch.Tensor) -> int:
    action_space = getattr(env.unwrapped, "action_space", None)
    action_shape = getattr(action_space, "shape", None)
    if action_shape is not None:
        if len(action_shape) >= 2:
            return int(action_shape[1])
        if len(action_shape) == 1:
            return int(action_shape[0])
    return int(obs.shape[-1])


def load_renet_policy_for_eval(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    fallback_num_prop: int,
    fallback_action_dim: int,
) -> tuple[RENetBCPolicy, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta: dict[str, Any] = dict(payload.get("meta", {}))

    proprio_dim = int(meta.get("num_prop", fallback_num_prop))
    action_dim = int(meta.get("action_dim", fallback_action_dim))
    prop_hist_len = int(args.prop_hist_len if args.prop_hist_len is not None else meta.get("prop_hist_len", 10))
    depth_hist_len = int(args.depth_hist_len if args.depth_hist_len is not None else meta.get("depth_hist_len", 2))
    hidden_dim = int(args.hidden_dim if args.hidden_dim is not None else meta.get("hidden_dim", 64))
    embed_dim = int(args.embed_dim if args.embed_dim is not None else meta.get("embed_dim", 128))

    model = RENetBCPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    meta.update(
        {
            "num_prop": proprio_dim,
            "action_dim": action_dim,
            "prop_hist_len": prop_hist_len,
            "depth_hist_len": depth_hist_len,
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
        }
    )
    return model, meta


def main() -> None:  # noqa: C901
    args = parse_args_eval()

    if "Eval" not in args.task:
        print("[Error] The --task argument must contain 'Eval' to use the evaluation configuration.")
        return

    headless = getattr(args, "headless", False)
    disable_fabric = getattr(args, "disable_fabric", False)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import parkour_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
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

    obs, extras = vec_env.get_observations()
    depth_sample = extras["observations"].get("depth_camera")
    if depth_sample is None:
        raise RuntimeError("Current task does not provide depth_camera observations.")
    if depth_sample.dim() == 4 and depth_sample.shape[1] == 1:
        depth_sample_hw = depth_sample.squeeze(1)
    elif depth_sample.dim() == 3:
        depth_sample_hw = depth_sample
    else:
        raise RuntimeError(f"Expected depth_camera [N, H, W] or [N, 1, H, W], got {tuple(depth_sample.shape)}.")

    fallback_num_prop = int(getattr(getattr(agent_cfg, "estimator", None), "num_prop", obs.shape[-1]))
    fallback_action_dim = get_action_dim(vec_env, obs)

    ckpt_path = resolve_checkpoint_path(args)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    device = torch.device(args.device)
    model, meta = load_renet_policy_for_eval(
        checkpoint_path=ckpt_path,
        args=args,
        device=device,
        fallback_num_prop=fallback_num_prop,
        fallback_action_dim=fallback_action_dim,
    )
    proprio_dim = int(meta["num_prop"])
    if proprio_dim > int(obs.shape[-1]):
        raise RuntimeError(f"Checkpoint expects num_prop={proprio_dim}, but env observation dim is {obs.shape[-1]}.")
    prop_hist_len = int(meta["prop_hist_len"])
    depth_hist_len = int(meta["depth_hist_len"])
    camera_resolution = (int(depth_sample_hw.shape[-2]), int(depth_sample_hw.shape[-1]))
    meta_camera_resolution = tuple(meta.get("camera_resolution", camera_resolution))
    if tuple(meta_camera_resolution) != tuple(camera_resolution):
        print(
            f"[Warning] Checkpoint camera_resolution={meta_camera_resolution}, "
            f"env depth_camera={camera_resolution}. Using env resolution for history buffers."
        )

    runner = RENetOnlineRunner(
        model=model,
        num_envs=vec_env.num_envs,
        proprio_dim=proprio_dim,
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        camera_resolution=camera_resolution,
        device=device,
    )
    runner.reset()

    dropout_manager: Optional[CameraDropoutManager] = None
    if args.use_dropout:
        dropout_manager = CameraDropoutManager(
            num_envs=vec_env.num_envs,
            device=device,
            dt=float(vec_env.unwrapped.step_dt),
            prob_start_offline=args.prob_start_offline,
            prob_cam_offline=1.0,
            online_duration_range=(args.online_duration_min, args.online_duration_max),
            offline_duration_range=(args.offline_duration_min, args.offline_duration_max),
        )
        print("[Eval] Camera dropout simulation: ENABLED")
    else:
        print("[Eval] Camera dropout simulation: DISABLED. VP branch is always selected.")

    total_steps = int(args.total_steps)
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    edge_violation_buffer = deque(maxlen=total_steps)
    num_waypoints_per_terrain = collections.defaultdict(list)

    terrain_names = {}
    try:
        sub_terrains = vec_env.unwrapped.cfg.scene.terrain.terrain_generator.sub_terrains
        for i, name in enumerate(sub_terrains.keys()):
            terrain_names[i] = name
    except Exception as exc:
        print(f"[Warning] Failed to map terrain names: {exc}")

    cur_reward_sum = torch.zeros(vec_env.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(vec_env.num_envs, dtype=torch.float, device=device)

    try:
        reward_feet_edge = vec_env.unwrapped.reward_manager.get_term_cfg("reward_feet_edge").func
        base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")
    except Exception as exc:
        print(f"[Warning] Could not extract reward/parkour components: {exc}")
        reward_feet_edge = None
        base_parkour = None

    dones_bool = torch.zeros(vec_env.num_envs, device=device, dtype=torch.bool)
    offline_frame_count = 0
    total_frame_count = 0

    print(f"[Info] Loading RENet checkpoint from: {ckpt_path}")
    print(
        f"[Info] proprio={proprio_dim} action={int(meta['action_dim'])} camera={camera_resolution} "
        f"prop_hist={prop_hist_len} depth_hist={depth_hist_len} "
        f"hidden={int(meta['hidden_dim'])} embed={int(meta['embed_dim'])}"
    )
    print(f"[Info] Starting evaluation for {total_steps} steps on {vec_env.num_envs} environments...")

    for _ in tqdm(range(total_steps)):
        if not simulation_app.is_running():
            break

        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("Current task does not provide depth_camera observations.")

        obs_prop = obs[:, :proprio_dim].clone()
        student_depth = depth_image.clone()

        if dropout_manager is not None:
            dropout_manager.reset_env(dones_bool)
            dropout_manager.update(depth_image=student_depth, obs_prop=obs_prop)
            use_op_mask = dropout_manager.offline_state.clone()
        else:
            use_op_mask = torch.zeros(vec_env.num_envs, dtype=torch.bool, device=device)

        offline_frame_count += int(use_op_mask.sum().item())
        total_frame_count += int(use_op_mask.numel())

        actions = runner.act(
            obs_prop=obs_prop,
            depth_image=student_depth,
            use_op_mask=use_op_mask,
            prev_done=dones_bool,
        )

        if base_parkour is not None:
            cur_goal_idx = base_parkour.cur_goal_idx.clone().view(-1)
            cur_terrain_ids = base_parkour.env_class.clone().long().view(-1)

        obs_next, rews, dones, extras = vec_env.step(actions)
        dones_bool = dones.squeeze(-1).bool()

        if reward_feet_edge is not None:
            edge_violation_buffer.extend(
                reward_feet_edge.feet_at_edge.sum(dim=1).float().cpu().numpy().tolist()
            )

        cur_reward_sum += rews.view(-1)
        cur_episode_length += 1

        done_indices = dones_bool.nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            rewbuffer.extend(cur_reward_sum[done_indices].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[done_indices].cpu().numpy().tolist())

            if base_parkour is not None:
                waypoints = cur_goal_idx[done_indices].cpu().numpy().tolist()
                num_waypoints_buffer.extend(waypoints)

                done_terrain_ids = cur_terrain_ids[done_indices].cpu().numpy().tolist()
                for t_id, wp in zip(done_terrain_ids, waypoints):
                    num_waypoints_per_terrain[t_id].append(wp)

            cur_reward_sum[done_indices] = 0.0
            cur_episode_length[done_indices] = 0.0

        if dones_bool.any():
            runner.reset_done(dones_bool)

        obs = obs_next
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    if len(rewbuffer) > 0:
        rew_mean = statistics.mean(rewbuffer)
        rew_std = statistics.stdev(rewbuffer) if len(rewbuffer) > 1 else 0.0
        print("Mean reward:              {:.2f} \u00B1 {:.2f}".format(rew_mean, rew_std))

        len_mean = statistics.mean(lenbuffer)
        len_std = statistics.stdev(lenbuffer) if len(lenbuffer) > 1 else 0.0
        print("Mean episode length:      {:.2f} \u00B1 {:.2f}".format(len_mean, len_std))
    else:
        print("No episodes finished during evaluation.")

    if len(num_waypoints_buffer) > 0:
        waypoints_arr = np.array(num_waypoints_buffer).astype(float) / 7.0
        print("Mean number of waypoints: {:.2f} \u00B1 {:.2f}".format(np.mean(waypoints_arr), np.std(waypoints_arr)))

    if len(edge_violation_buffer) > 0:
        print(
            "Mean edge violation:      {:.2f} \u00B1 {:.2f}".format(
                np.mean(edge_violation_buffer), np.std(edge_violation_buffer)
            )
        )

    if total_frame_count > 0:
        print("OP selected ratio:        {:.3f}".format(offline_frame_count / total_frame_count))

    if len(num_waypoints_per_terrain) > 0:
        print("\n--- Completion Rate by Terrain Type ---")
        for t_id, wps in sorted(num_waypoints_per_terrain.items()):
            t_name = terrain_names.get(t_id, f"TerrainType_{t_id}")
            values = np.array(wps, dtype=float) / 7.0
            print(f"{t_name:<18}: {np.mean(values):.2f} \u00B1 {np.std(values):.2f} (from {len(wps)} episodes)")

    print("=" * 50)

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
