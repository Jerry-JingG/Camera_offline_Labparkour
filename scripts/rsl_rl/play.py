# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--free_cam", action="store_true", default=False, help="Disable follow camera for free-look.")
parser.add_argument("--smooth_flat_terrain", action="store_true", default=False, help="Force a smooth parkour_flat terrain.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help="Force a single parkour terrain type, e.g. parkour_hurdle or hurdle.",
)
parser.add_argument("--record_trace", action="store_true", default=False, help="Record env0 obs/action/depth artifacts.")
parser.add_argument("--record_csv", type=str, default=None, help="CSV path for --record_trace.")
parser.add_argument("--record_depth_dir", type=str, default=None, help="Depth artifact directory for --record_trace.")
parser.add_argument("--record_depth_every", type=int, default=5, help="Depth artifact write interval in policy steps.")
parser.add_argument("--record_env_id", type=int, default=0, help="Environment index to record.")
parser.add_argument("--record_steps", type=int, default=300, help="Number of policy steps to record. 0 means until closed.")
parser.add_argument("--record_foot_force_threshold", type=float, default=2.0, help="Contact threshold metadata for trace CSV.")
parser.add_argument(
    "--disable_depth_debug_vis", action="store_true", default=False, help="Disable depth-camera debug cv2 windows."
)
parser.add_argument(
    "--stationary_record",
    action="store_true",
    default=False,
    help="After a zero-action warmup, record a zero-action standing trace.",
)
parser.add_argument(
    "--stand_warmup_steps",
    type=int,
    default=100,
    help="Zero-action steps to run before starting --stationary_record recording.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.stationary_record:
    args_cli.record_trace = True
# always enable cameras to record video
if args_cli.video or args_cli.record_trace:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import torch.nn.functional as F

from modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg
from trace_recorder import IsaacPlayTraceRecorder

from exporter import (
export_teacher_policy_as_jit, 
export_teacher_policy_as_onnx,
export_deploy_policy_as_jit, 
export_deploy_policy_as_onnx,
)
from vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def _apply_smooth_flat_terrain(env_cfg) -> None:
    _apply_single_terrain(env_cfg, "parkour_flat")


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
        print("[INFO] Smooth flat terrain enabled: using parkour_flat only, with roughness disabled.")
    else:
        if hasattr(selected_cfg, "apply_flat"):
            selected_cfg.apply_flat = False
        print(f"[INFO] Single terrain enabled: using {terrain_name} only.")


def _disable_depth_debug_vis(env_cfg) -> None:
    try:
        env_cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False
        print("[INFO] Depth debug visualization disabled.")
    except AttributeError:
        pass


def _apply_zero_command(env_cfg) -> None:
    try:
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.heading = (0.0, 0.0)
        command_cfg.resampling_time_range = (60.0, 60.0)
        print("[INFO] Zero command enabled: command_x and heading targets fixed at 0.")
    except AttributeError:
        print("[WARN] Could not force base_velocity command to zero for stationary recording.")


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


def _zero_actions(env, obs: torch.Tensor) -> torch.Tensor:
    return torch.zeros((env.num_envs, env.num_actions), dtype=obs.dtype, device=env.device)


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


def _compute_depth_logging(
    *,
    env,
    extras: dict,
    obs: torch.Tensor,
    num_prop: int,
    depth_encoder,
    record_env_id: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    depth_camera = extras["observations"]["depth_camera"].to(env.device)
    depth_trace = _make_depth_trace(env, depth_camera, record_env_id)
    with torch.inference_mode():
        obs_student = obs[:, :num_prop].clone()
        obs_student[:, 6:8] = 0
        depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
        yaw = depth_latent_and_yaw[:, -2:]
    return depth_trace, yaw, 1.5 * yaw


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # 临时绕过 delta_yaw_ok 观测组，避免空 shape 触发 ObservationManager 维度拼接报错
    if hasattr(env_cfg.observations, "delta_yaw_ok"):
        env_cfg.observations.delta_yaw_ok = None

    if args_cli.terrain_type is not None:
        _apply_single_terrain(env_cfg, args_cli.terrain_type)
    elif args_cli.smooth_flat_terrain:
        _apply_smooth_flat_terrain(env_cfg)

    if args_cli.disable_depth_debug_vis:
        _disable_depth_debug_vis(env_cfg)

    if args_cli.stationary_record:
        _apply_zero_command(env_cfg)
    
    if args_cli.free_cam:
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.origin_type = "world"
        print("[INFO] Free camera enabled. Follow camera disabled.")

    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    estimator = ppo_runner.get_estimator_inference_policy(device=env.device) 
    if agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        policy_nn = ppo_runner.alg.depth_actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_deploy_policy_as_jit(policy_nn, 
                                    estimator,
                                    depth_encoder,
                                    ppo_runner.obs_normalizer, 
                                    path=export_model_dir, 
                                    filename="policy.pt")
        export_deploy_policy_as_onnx(
                            policy_nn, 
                            estimator,
                            depth_encoder,
                            agent_cfg,
                            normalizer=ppo_runner.obs_normalizer, 
                            path=export_model_dir, 
                            filename="policy.onnx"
                        )

    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = ppo_runner.alg.policy
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
        export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_teacher_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    dt = env.unwrapped.step_dt
    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_paras["num_prop"]
    num_scan = estimator_paras["num_scan"]
    num_priv_explicit = estimator_paras["num_priv_explicit"]
    # reset environment
    obs, extras = env.get_observations()
    timestep = 0

    if args_cli.stationary_record and args_cli.stand_warmup_steps > 0:
        print(f"[INFO] Stationary warmup: {args_cli.stand_warmup_steps} zero-action steps before recording.")
        for _ in range(args_cli.stand_warmup_steps):
            if not simulation_app.is_running():
                break
            actions = _zero_actions(env, obs)
            obs, _, _, extras = env.step(actions)
        print("[INFO] Stationary warmup complete. Starting trace recording.")

    recorder = None
    if args_cli.record_trace:
        if not 0 <= args_cli.record_env_id < env.num_envs:
            raise ValueError(f"--record_env_id must be in [0, {env.num_envs - 1}], got {args_cli.record_env_id}")
        record_csv = args_cli.record_csv
        if record_csv is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            record_csv = os.path.join(log_dir, "traces", f"isaacsim_obs_action_{timestamp}.csv")
        recorder = IsaacPlayTraceRecorder(
            record_csv,
            depth_dir=args_cli.record_depth_dir,
            depth_every=args_cli.record_depth_every,
            max_steps=args_cli.record_steps,
            foot_force_threshold=args_cli.record_foot_force_threshold,
            depth_max_distance=_get_depth_max_distance(env),
            record_env_id=args_cli.record_env_id,
        )

    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            proprio_obs = obs[:, :num_prop].clone()
            depth_trace = None
            depth_yaw = None
            policy_obs_6_8 = torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)
            # run everything in inference mode
            if args_cli.stationary_record:
                actions = _zero_actions(env, obs)
                if agent_cfg.algorithm.class_name == "DistillationWithExtractor":
                    depth_trace, depth_yaw, policy_obs_6_8 = _compute_depth_logging(
                        env=env,
                        extras=extras,
                        obs=obs,
                        num_prop=num_prop,
                        depth_encoder=depth_encoder,
                        record_env_id=args_cli.record_env_id,
                    )
            elif agent_cfg.algorithm.class_name != "DistillationWithExtractor":
                with torch.inference_mode():
                    # agent stepping
                    obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                    actions = policy(obs, hist_encoding = True)
                # env stepping
            else:
                depth_camera = extras["observations"]['depth_camera'].to(env.device)
                depth_trace = _make_depth_trace(env, depth_camera, args_cli.record_env_id) if recorder is not None else None
                with torch.inference_mode():
                    if env.unwrapped.common_step_counter %5 == 0:
                        obs_student = obs[:, :num_prop].clone()
                        obs_student[:, 6:8] = 0
                        depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                        depth_latent = depth_latent_and_yaw[:, :-2]
                        yaw = depth_latent_and_yaw[:, -2:]
                    obs[:, 6:8] = 1.5*yaw
                    depth_yaw = yaw.clone()
                    policy_obs_6_8 = obs[:, 6:8].clone()
                    # obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                    actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)

            if recorder is not None:
                recorder.write(
                    policy_step_index=timestep,
                    proprio_obs=proprio_obs,
                    policy_action=actions,
                    depth_yaw=depth_yaw,
                    policy_obs_6_8=policy_obs_6_8,
                    foot_force_isaac=_get_foot_force_isaac(env),
                    depth_trace=depth_trace,
                    root_state_w=_get_root_state_w(env),
                    root_lin_vel_b=_get_root_lin_vel_b(env),
                )

            obs, _, _, extras = env.step(actions)
            timestep += 1

            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            if recorder is not None and args_cli.record_steps > 0 and timestep >= args_cli.record_steps:
                print(f"[INFO] Recorded {timestep} policy steps. Exiting play loop.")
                break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        if recorder is not None:
            recorder.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
