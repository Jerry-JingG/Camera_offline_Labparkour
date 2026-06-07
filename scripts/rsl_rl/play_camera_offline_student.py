from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ISAACLAB_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
for source_dir in (
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_assets",
    "isaaclab_rl",
    "isaaclab_mimic",
):
    source_path = os.path.join(ISAACLAB_ROOT, "source", source_dir)
    if os.path.isdir(source_path) and source_path not in sys.path:
        sys.path.insert(0, source_path)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)
RSL_RL_DIR = os.path.join(PROJECT_ROOT, "scripts", "rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from camera_offline_student import (  # isort: skip
    DEFAULT_CAMERA_OFFLINE_ROOT,
    DEFAULT_STUDENT_CHECKPOINT,
    build_camera_dropout_manager,
    build_camera_offline_student_runner,
)
from trace_recorder import IsaacPlayTraceRecorder  # isort: skip


class CameraOfflineActionFilter:
    def __init__(
        self,
        *,
        hold_steps: int,
        ramp_steps: int,
        warmup_delta_limit: float,
        warmup_tail_steps: int,
        action_lpf_alpha: float,
        action_delta_limit: float,
        action_clip: float,
    ) -> None:
        self.hold_steps = max(0, int(hold_steps))
        self.ramp_steps = max(0, int(ramp_steps))
        self.warmup_delta_limit = max(0.0, float(warmup_delta_limit))
        self.warmup_tail_steps = max(0, int(warmup_tail_steps))
        self.action_lpf_alpha = float(action_lpf_alpha)
        self.action_delta_limit = max(0.0, float(action_delta_limit))
        self.action_clip = max(0.0, float(action_clip))
        self._prev_policy_action: torch.Tensor | None = None
        self._warmup_delta_active = False

    def reset(self) -> None:
        self._prev_policy_action = None
        self._warmup_delta_active = False

    def apply(self, action: torch.Tensor, step: int) -> torch.Tensor:
        target = action
        if step < self.hold_steps:
            target = torch.zeros_like(action)
        elif self.ramp_steps > 0:
            ramp_index = step - self.hold_steps
            if ramp_index < self.ramp_steps:
                ratio = float(ramp_index + 1) / float(self.ramp_steps)
                target = action * ratio

        delta_limit_override = None
        if self.warmup_delta_limit > 0.0:
            warmup_delta_steps = self.hold_steps + self.ramp_steps + self.warmup_tail_steps
            if step < warmup_delta_steps:
                self._warmup_delta_active = True
            if self._warmup_delta_active:
                delta_limit_override = self.warmup_delta_limit

        filtered = self._filter_policy_action(target, delta_limit_override=delta_limit_override)
        if (
            self._warmup_delta_active
            and self.warmup_delta_limit > 0.0
            and step >= self.hold_steps + self.ramp_steps + self.warmup_tail_steps
        ):
            remaining = torch.max(torch.abs(filtered.detach() - target.detach())).item()
            if remaining <= self.warmup_delta_limit:
                self._warmup_delta_active = False
        return filtered

    def _filter_policy_action(
        self,
        action: torch.Tensor,
        delta_limit_override: float | None = None,
    ) -> torch.Tensor:
        original_shape = action.shape
        filtered = action.reshape(-1)
        if self._prev_policy_action is None or self._prev_policy_action.shape != filtered.shape:
            self._prev_policy_action = torch.zeros_like(filtered)

        alpha = self.action_lpf_alpha
        if alpha < 1.0:
            filtered = alpha * filtered + (1.0 - alpha) * self._prev_policy_action

        delta_limit = self.action_delta_limit
        if delta_limit_override is not None and delta_limit_override > 0.0:
            delta_limit = float(delta_limit_override)
            if self.action_delta_limit > 0.0:
                delta_limit = min(delta_limit, self.action_delta_limit)

        if delta_limit > 0.0:
            delta = torch.clamp(
                filtered - self._prev_policy_action,
                -delta_limit,
                delta_limit,
            )
            filtered = self._prev_policy_action + delta

        if self.action_clip > 0.0:
            filtered = torch.clamp(filtered, -self.action_clip, self.action_clip)

        self._prev_policy_action = filtered.detach()
        return filtered.reshape(original_shape)


def parse_args_play() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play the latest Camera_offline TXL student in Isaac and record MuJoCo-compatible traces."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0",
        help="Isaac task name.",
    )
    parser.add_argument("--camera_offline_root", type=str, default=DEFAULT_CAMERA_OFFLINE_ROOT)
    parser.add_argument("--student_checkpoint", type=str, default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="Maximum steps to run. 0 means until closed.")
    parser.add_argument("--mem_len", type=int, default=0, help="Override Transformer-XL memory length. 0 uses checkpoint/default.")
    parser.add_argument("--command_x", type=float, default=0.0)
    parser.add_argument(
        "--terrain_type",
        type=str,
        default=None,
        help="Force one terrain type, e.g. parkour_flat, flat, hurdle, stairs.",
    )
    parser.add_argument(
        "--terrain_flag",
        choices=("auto", "flat", "parkour", "hurdles", "stairs"),
        default="auto",
        help="Override obs[11:13] terrain flag. auto keeps the Isaac observation.",
    )
    parser.add_argument(
        "--single_subterrain",
        action="store_true",
        default=False,
        help="Shrink the generated terrain to one sub-terrain tile for easier GUI play.",
    )
    parser.add_argument(
        "--single_subterrain_border_width",
        type=float,
        default=0.0,
        help="Border width in meters when --single_subterrain is enabled.",
    )
    parser.add_argument(
        "--single_subterrain_difficulty",
        type=float,
        default=0.8,
        help="Fixed terrain difficulty when --single_subterrain is enabled.",
    )
    parser.add_argument(
        "--free_cam",
        action="store_true",
        default=False,
        help="Start the GUI with the old world camera instead of tracking the robot.",
    )
    parser.add_argument("--disable_depth_debug_vis", action="store_true", default=False)
    parser.add_argument("--use_dropout", action="store_true", default=False)
    parser.add_argument("--use_agent_clip_actions", action="store_true", default=False)

    parser.add_argument("--camera_offline_hold_steps", type=int, default=0)
    parser.add_argument("--camera_offline_ramp_steps", type=int, default=0)
    parser.add_argument("--camera_offline_warmup_delta_limit", type=float, default=0.0)
    parser.add_argument("--camera_offline_warmup_tail_steps", type=int, default=0)
    parser.add_argument("--action_lpf_alpha", type=float, default=1.0)
    parser.add_argument("--action_delta_limit", type=float, default=0.0)
    parser.add_argument("--action_clip", type=float, default=0.0)

    parser.add_argument("--record_trace", action="store_true", default=False)
    parser.add_argument("--record_csv", type=str, default=None)
    parser.add_argument("--record_depth_dir", type=str, default=None)
    parser.add_argument("--record_depth_every", type=int, default=5)
    parser.add_argument("--record_env_id", type=int, default=0)
    parser.add_argument("--record_steps", type=int, default=0, help="Number of steps to record. 0 means until closed.")
    parser.add_argument("--record_foot_force_threshold", type=float, default=2.0)
    parser.add_argument(
        "--record_obs_mode",
        choices=("mujoco_raw", "student_input"),
        default="mujoco_raw",
        help="Whether obs_* records MuJoCo-style raw yaw slots or the exact student input.",
    )

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _canonical_terrain_name(terrain_type: str) -> str:
    terrain_type = terrain_type.strip()
    if not terrain_type:
        raise ValueError("--terrain_type cannot be empty.")
    return terrain_type if terrain_type.startswith("parkour_") else f"parkour_{terrain_type}"


def _apply_single_terrain(env_cfg, terrain_type: str) -> None:
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    sub_terrains = getattr(terrain_generator, "sub_terrains", None)
    terrain_name = _canonical_terrain_name(terrain_type)
    if not sub_terrains or terrain_name not in sub_terrains:
        available = ", ".join(sub_terrains.keys()) if sub_terrains else "<none>"
        print(f"[WARN] --terrain_type {terrain_type} requested, but {terrain_name} was not found. Available: {available}")
        return

    for name, sub_terrain in sub_terrains.items():
        sub_terrain.proportion = 1.0 if name == terrain_name else 0.0

    selected_cfg = sub_terrains[terrain_name]
    if terrain_name == "parkour_flat":
        selected_cfg.apply_flat = True
        selected_cfg.apply_roughness = False
        selected_cfg.noise_range = (0.0, 0.0)
        selected_cfg.pad_height = 0.0
        terrain_generator.random_difficulty = False
        terrain_generator.difficulty_range = (0.0, 0.0)
        print("[INFO] Smooth flat terrain enabled.")
    else:
        if hasattr(selected_cfg, "apply_flat"):
            selected_cfg.apply_flat = False
        print(f"[INFO] Single terrain enabled: using {terrain_name}.")


def _apply_single_subterrain(env_cfg, *, border_width: float, difficulty: float) -> None:
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_generator is None:
        print("[WARN] --single_subterrain requested, but the task has no terrain generator.")
        return

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1
    terrain_generator.border_width = max(0.0, float(border_width))
    terrain_generator.curriculum = False
    terrain_generator.random_difficulty = False
    terrain_generator.difficulty_range = (difficulty, difficulty)
    if hasattr(terrain_cfg, "max_init_terrain_level"):
        terrain_cfg.max_init_terrain_level = 0

    tile_size = getattr(terrain_generator, "size", None)
    if tile_size is not None:
        print(
            "[INFO] Single-subterrain play enabled: "
            f"num_rows=1, num_cols=1, tile_size={tile_size}, "
            f"border_width={terrain_generator.border_width}, difficulty={difficulty:.2f}."
        )
    else:
        print("[INFO] Single-subterrain play enabled: num_rows=1, num_cols=1.")


def _apply_command_x(env_cfg, command_x: float) -> None:
    try:
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.ranges.lin_vel_x = (float(command_x), float(command_x))
        command_cfg.ranges.heading = (0.0, 0.0)
        command_cfg.resampling_time_range = (60.0, 60.0)
        print(f"[INFO] Fixed command_x={float(command_x):.3f}, heading=0.")
    except AttributeError:
        print("[WARN] Could not force base_velocity command.")


def _apply_terrain_flag_override(obs_prop: torch.Tensor, terrain_flag: str) -> None:
    if terrain_flag == "auto":
        return
    if obs_prop.shape[-1] < 13:
        raise ValueError("terrain flag override requires proprio dim >= 13")
    is_flat = terrain_flag == "flat"
    values = (0.0, 1.0) if is_flat else (1.0, 0.0)
    obs_prop[:, 11] = values[0]
    obs_prop[:, 12] = values[1]


def _disable_depth_debug_vis(env_cfg) -> None:
    try:
        env_cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False
        print("[INFO] Depth debug visualization disabled.")
    except AttributeError:
        pass


def _get_depth_max_distance(env) -> float:
    try:
        return float(env.unwrapped.scene["depth_camera"].cfg.max_distance)
    except Exception:
        return 2.0


def _make_depth_trace(env, depth_camera: torch.Tensor, env_id: int) -> dict[str, torch.Tensor]:
    policy_depth_input = depth_camera
    try:
        raw_depth = env.unwrapped.scene["depth_camera"].data.output["distance_to_camera"].squeeze(-1)
        depth_max_distance = _get_depth_max_distance(env)
    except Exception:
        raw_depth = policy_depth_input
        depth_max_distance = 2.0

    raw_env = raw_depth[env_id] if raw_depth.dim() >= 3 else raw_depth
    target_hw = tuple(policy_depth_input.shape[-2:])
    processed = torch.nan_to_num(raw_env, nan=depth_max_distance, posinf=depth_max_distance, neginf=0.0)
    processed = torch.clamp(processed, 0.0, depth_max_distance)
    if tuple(processed.shape[-2:]) != target_hw:
        processed = F.interpolate(
            processed[None, None],
            size=target_hw,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    processed = processed / depth_max_distance - 0.5

    return {
        "raw_depth_m": raw_env.detach().cpu(),
        "processed_capture_depth": processed.detach().cpu(),
        "policy_depth_input": policy_depth_input[env_id].detach().cpu(),
    }


def _get_foot_force_isaac(env) -> torch.Tensor | None:
    try:
        contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        foot_ids, _ = contact_sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        foot_forces = contact_sensor.data.net_forces_w_history[:, 0, foot_ids]
        return torch.norm(foot_forces, dim=-1)
    except Exception:
        return None


def _get_root_state_w(env) -> torch.Tensor | None:
    try:
        return env.unwrapped.scene["robot"].data.root_state_w
    except Exception:
        return None


def _get_root_lin_vel_b(env) -> torch.Tensor | None:
    try:
        return env.unwrapped.scene["robot"].data.root_lin_vel_b
    except Exception:
        return None


def _get_current_yaw(vec_env) -> torch.Tensor:
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

    try:
        base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")
        root_quat_w = base_parkour.robot.data.root_quat_w
    except Exception:
        root_quat_w = vec_env.unwrapped.scene["robot"].data.root_quat_w
    _, _, yaw = euler_xyz_from_quat(root_quat_w)
    return wrap_to_pi(yaw)


def _configure_play_viewer(env_cfg, *, num_envs: int, free_cam: bool) -> None:
    if free_cam:
        env_cfg.viewer.origin_type = "world"
        spacing = float(env_cfg.scene.env_spacing)
        grid = int(np.ceil(np.sqrt(num_envs)))
        env_cfg.viewer.eye = [spacing * grid * 0.5, spacing * grid * 0.5, 3.0]
        env_cfg.viewer.lookat = [0.0, 0.0, 0.5]
        print("[INFO] Viewer starts in world camera mode.")
        return

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = [0.0, 2.6, 1.6]
    env_cfg.viewer.lookat = [0.0, 0.0, 0.5]
    print("[INFO] Viewer starts in robot-follow mode. Press NUMPAD_0 for free cam, NUMPAD_1 to return.")


def main() -> None:
    args = parse_args_play()
    args.enable_cameras = True

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
    if args.terrain_type:
        _apply_single_terrain(env_cfg, args.terrain_type)
    if args.single_subterrain:
        _apply_single_subterrain(
            env_cfg,
            border_width=args.single_subterrain_border_width,
            difficulty=args.single_subterrain_difficulty,
        )
        if args.num_envs > 1:
            print(
                "[WARN] --single_subterrain with num_envs > 1 will place all envs on the same terrain tile. "
                "Use --num_envs 1 for clean GUI play."
            )
    _apply_command_x(env_cfg, args.command_x)
    if args.disable_depth_debug_vis:
        _disable_depth_debug_vis(env_cfg)
    if not headless:
        _configure_play_viewer(env_cfg, num_envs=args.num_envs, free_cam=args.free_cam)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not headless else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    clip_actions = agent_cfg.clip_actions if args.use_agent_clip_actions else None
    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=clip_actions)

    if not 0 <= args.record_env_id < vec_env.num_envs:
        raise ValueError(f"--record_env_id must be in [0, {vec_env.num_envs - 1}], got {args.record_env_id}")

    student_runner = build_camera_offline_student_runner(
        checkpoint_path=args.student_checkpoint,
        camera_offline_root=args.camera_offline_root,
        device=args.device,
        num_envs=vec_env.num_envs,
        mem_len=args.mem_len if args.mem_len > 0 else None,
    )
    print(f"[INFO] Student checkpoint: {student_runner.checkpoint_path}")
    print(
        "[INFO] Student inputs: "
        f"proprio={student_runner.num_prop}, "
        f"depth={student_runner.camera_resolution}, "
        f"prop_hist={student_runner.prop_hist_len}, "
        f"depth_hist={student_runner.depth_hist_len}, "
        f"mem_len={student_runner.sequence_length}"
    )

    dropout_manager = None
    if args.use_dropout:
        dropout_manager = build_camera_dropout_manager(
            camera_offline_root=args.camera_offline_root,
            num_envs=vec_env.num_envs,
            device=student_runner.device,
            dt=float(vec_env.unwrapped.step_dt),
        )
        print("[INFO] Camera-offline input dropout: enabled")

    action_filter = CameraOfflineActionFilter(
        hold_steps=args.camera_offline_hold_steps,
        ramp_steps=args.camera_offline_ramp_steps,
        warmup_delta_limit=args.camera_offline_warmup_delta_limit,
        warmup_tail_steps=args.camera_offline_warmup_tail_steps,
        action_lpf_alpha=args.action_lpf_alpha,
        action_delta_limit=args.action_delta_limit,
        action_clip=args.action_clip,
    )

    recorder = None
    if args.record_trace:
        record_csv = args.record_csv
        if record_csv is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            record_csv = os.path.join(
                PROJECT_ROOT,
                "logs",
                "isaacsim_camera_offline_play",
                f"isaacsim_obs_action_{timestamp}.csv",
            )
        recorder = IsaacPlayTraceRecorder(
            record_csv,
            depth_dir=args.record_depth_dir,
            depth_every=args.record_depth_every,
            max_steps=args.record_steps,
            foot_force_threshold=args.record_foot_force_threshold,
            depth_max_distance=_get_depth_max_distance(env),
            record_env_id=args.record_env_id,
        )

    obs, extras = vec_env.get_observations()
    prev_done = torch.zeros(vec_env.num_envs, device=student_runner.device, dtype=torch.bool)
    step = 0

    try:
        while simulation_app.is_running() and (args.max_steps <= 0 or step < args.max_steps):
            depth_image = extras["observations"].get("depth_camera")
            if depth_image is None:
                raise RuntimeError("Current task does not provide depth_camera observations; use a TeacherCam task.")
            if depth_image.dim() == 4 and depth_image.shape[1] == 1:
                depth_image = depth_image.squeeze(1)

            proprio_raw = obs[:, : student_runner.num_prop].clone()
            current_yaw = _get_current_yaw(vec_env).to(device=obs.device, dtype=obs.dtype)

            record_proprio = proprio_raw.clone()
            record_proprio[:, 6] = -current_yaw
            record_proprio[:, 7] = -current_yaw
            _apply_terrain_flag_override(record_proprio, args.terrain_flag)

            student_proprio = record_proprio.clone()
            student_proprio[:, 7] = 0.0
            student_proprio_runner = student_proprio.to(student_runner.device)
            depth_runner = depth_image.to(student_runner.device)

            if dropout_manager is not None:
                dropout_manager.reset_env(prev_done)
                dropout_manager.update(
                    depth_image=depth_runner,
                    obs_prop=student_proprio_runner,
                )
            if args.record_obs_mode == "student_input":
                record_proprio = student_proprio_runner.to(obs.device).clone()

            actions_raw, yaw_pred = student_runner.act(
                student_proprio_runner,
                depth_runner,
                prev_done=prev_done,
            )
            actions = actions_raw.to(obs.device)
            yaw_pred_env = yaw_pred.to(obs.device)
            if not torch.isfinite(actions).all():
                raise RuntimeError("camera_offline student produced non-finite actions")
            if not torch.isfinite(yaw_pred_env).all():
                raise RuntimeError("camera_offline student produced non-finite yaw_pred")

            actions = action_filter.apply(actions, step)
            policy_obs_6_8 = 1.5 * yaw_pred_env

            if recorder is not None:
                depth_trace = _make_depth_trace(env, depth_runner.to(obs.device), args.record_env_id)
                recorder.write(
                    policy_step_index=step,
                    proprio_obs=record_proprio,
                    policy_action=actions,
                    depth_yaw=yaw_pred_env,
                    policy_obs_6_8=policy_obs_6_8,
                    foot_force_isaac=_get_foot_force_isaac(env),
                    depth_trace=depth_trace,
                    root_state_w=_get_root_state_w(env),
                    root_lin_vel_b=_get_root_lin_vel_b(env),
                )

            obs, _, dones, extras = vec_env.step(actions)
            done_mask_env = dones.squeeze(-1).bool()
            prev_done = done_mask_env.to(student_runner.device)
            if done_mask_env.any():
                student_runner.reset_done(prev_done)
                action_filter.reset()
            step += 1

            if recorder is not None and args.record_steps > 0 and step >= args.record_steps:
                print(f"[INFO] Recorded {step} policy steps. Exiting play loop.")
                break
    finally:
        if recorder is not None:
            recorder.close()
        vec_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
