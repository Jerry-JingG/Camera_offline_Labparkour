"""
使用说明：
1. 先在 Isaac Lab 环境中启动本工程依赖的 Omniverse/Isaac Sim。
2. 执行示例：
   python scripts/rsl_rl/collect.py \
       --task Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0 \
       --num_envs 32 --total_steps 20000 \
       --checkpoint logs/rsl_rl/unitree_go2_parkour/<run>/model_4000.pt \
       --out outputs/datasets/teacher_cam \
       --depth-encoder-checkpoint <student_depth_encoder.pt>
3. 采集脚本会：
   - 使用教师策略与环境交互（可自定义并行环境数与步数）。
   - 同步抓取深度相机原始数据，并可按学生流程实时提取 latent / yaw。
   - 将每 shard（默认为 1000 个环境步）写入 .npz 文件，并输出 meta 与统计信息。
"""
from __future__ import annotations
    
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from isaaclab.app import AppLauncher

# 本地导入
import cli_args  # isort: skip

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行解析器，兼顾 RSL-RL 与 AppLauncher 的公共参数。"""

    parser = argparse.ArgumentParser(description="采集教师策略与深度相机数据。")
    parser.add_argument("--task", type=str, required=True, help="需要采集的 Isaac 任务名。")
    parser.add_argument("--num_envs", type=int, default=16, help="并行环境数量。")
    parser.add_argument("--total_steps", type=int, required=True, help="需要采集的环境步数（单次 step 全部 env 同步计数）。")
    parser.add_argument("--shard_size", type=int, default=1024, help="每个数据分片包含的 step 数量。")
    parser.add_argument("--out", type=str, required=True, help="数据集输出目录。")
    parser.add_argument("--depth-encoder-checkpoint", type=str, default=None, help="学生深度编码器权重路径（可选）。")
    parser.add_argument("--latent-interval", type=int, default=5, help="深度编码器更新 latent 的步间隔。")
    parser.add_argument("--dataset-format", choices=["npz"], default="npz", help="数据写入格式，目前支持 npz。")
    parser.add_argument("--depth-dtype", choices=["float32", "uint16"], default="uint16", help="深度图保存精度。")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="当 depth-dtype=uint16 时的缩放倍数。")
    parser.add_argument("--video", action="store_true", default=False, help="是否在采集时录制视频（仅用于调试）。")
    parser.add_argument("--video_length", type=int, default=500, help="视频长度。")
    parser.add_argument("--real-time", action="store_true", default=False, help="尽量按真实时间节奏采集。")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="使用官方预训练教师权重。")
    parser.add_argument("--resume_dataset", action="store_true", help="若输出目录存在则追加采集，否则提示是否覆盖。")

    parser.add_argument("--noised_observation", action="store_true", default=False, help="教师模型的观测输入是否受到扰动")
    parser.add_argument("--noised_action", action="store_true", default=False, help="教师模型的动作输出是否受到扰动")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


class RunningStats:
    """矢量化的在线统计器，用于记录均值与标准差，可从历史状态恢复。"""

    def __init__(self, dim: int, state: Optional[Dict[str, object]] = None):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        if state:
            self.load_state(state)

    def update(self, batch: np.ndarray):
        """使用批量样本（N, dim）更新统计量。"""

        if batch.size == 0:
            return
        batch = batch.reshape(-1, self.dim)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        if total == 0:
            return
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def load_state(self, state: Dict[str, object]):
        """根据已有统计（mean/std/count）恢复内部状态。"""

        count = int(state.get("count", 0))
        if count <= 0:
            return
        mean = np.array(state.get("mean", np.zeros(self.dim)), dtype=np.float64)
        std = np.array(state.get("std", np.zeros(self.dim)), dtype=np.float64)
        if mean.shape[0] != self.dim:
            return
        self.count = count
        self.mean = mean
        self.m2 = (std**2) * max(self.count - 1, 0)

    def finalize(self) -> Dict[str, List[float]]:
        """返回用于落盘的字典。"""

        if self.count < 2:
            var = np.zeros_like(self.mean)
        else:
            var = self.m2 / (self.count - 1)
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": np.sqrt(var).astype(np.float32).tolist(),
            "count": int(self.count),
        }


def extract_state_from_cache(
    cache: Optional[Dict[str, np.ndarray]],
    prefix: str,
    fallback_count: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    """从已存在的 stats 缓存中提取指定前缀的统计信息。"""

    if not cache:
        return None
    mean_key = f"{prefix}_mean"
    std_key = f"{prefix}_std"
    if mean_key not in cache or std_key not in cache:
        return None
    count_key = f"{prefix}_count"
    count_arr = cache.get(count_key)
    if count_arr is not None and count_arr.size > 0:
        count = int(count_arr[0])
    elif fallback_count is not None:
        count = int(fallback_count)
    else:
        count = 0
    return {
        "mean": cache[mean_key],
        "std": cache[std_key],
        "count": count,
    }


class ShardedNPZWriter:
    """以分片 npz 格式写入数据，避免单文件过大，可指定起始分片编号。"""

    def __init__(self, out_dir: Path, start_index: int = 0):
        self.out_dir = out_dir
        self.shard_dir = self.out_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.shard_idx = start_index

    def write(self, buffer: Dict[str, List[np.ndarray]]):
        """将当前缓冲堆叠后写入 npz。"""

        if not buffer:
            return
        arrays = {key: np.stack(value, axis=0) for key, value in buffer.items()}
        shard_path = self.shard_dir / f"shard_{self.shard_idx:05d}.npz"
        np.savez_compressed(shard_path, **arrays)
        self.shard_idx += 1


def convert_depth(depth_image: torch.Tensor, dtype: str, scale: float) -> np.ndarray:
    """根据用户配置将深度图转换到目标精度。"""

    depth_np = depth_image.cpu().numpy()
    if dtype == "float32":
        return depth_np.astype(np.float32)
    scaled = np.clip(depth_np * scale, 0, np.iinfo(np.uint16).max)
    return scaled.astype(np.uint16)


def prepare_depth_encoder(
    checkpoint_path: str | None,
    estimator_cfg: dict,
    policy_cfg,
    device: torch.device,
):
    """根据可选的 checkpoint 构建学生深度编码器。"""

    if checkpoint_path is None:
        return None

    from scripts.rsl_rl.modules.feature_extractors.depth_backbone import (  # noqa: WPS433
        DepthOnlyFCBackbone58x87,
        RecurrentDepthBackbone,
    )

    scan_output_dim = policy_cfg.scan_encoder_dims[-1]
    backbone = DepthOnlyFCBackbone58x87(scan_output_dim)
    depth_cfg = {"num_prop": estimator_cfg["num_prop"]}
    depth_encoder = RecurrentDepthBackbone(backbone, depth_cfg).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "depth_encoder_state_dict" in state:
            depth_encoder.load_state_dict(state["depth_encoder_state_dict"])
        elif "state_dict" in state:
            depth_encoder.load_state_dict(state["state_dict"])
        else:
            depth_encoder.load_state_dict(state)
    else:
        depth_encoder.load_state_dict(state)
    depth_encoder.eval()
    return depth_encoder


def reset_depth_hidden_states(depth_encoder, done_mask: torch.Tensor):
    """对 Done 的环境通道清零隐藏状态，避免历史污染。"""

    if depth_encoder is None:
        return
    if depth_encoder.hidden_states.shape[1] == 0:
        return
    if done_mask.any():
        depth_encoder.hidden_states[:, done_mask, :] = 0


def generate_proprio_noise(
    obs_prop: torch.Tensor,
    noise_scale: float,
    device: torch.device
) -> torch.Tensor:
    """
    对本体观测 (53维) 施加结构化噪声。
    通道定义参考 observations.py:
    0-3: Base Ang Vel (3)
    3-5: IMU/Projected Gravity (2)
    5-8: Delta Yaws (3) -> 逻辑值, 不加噪声
    8-11: Commands (3) -> 指令, 绝对不加
    11-13: Terrain Bools (2) -> 标志位, 绝对不加
    13-25: Joint Pos (12)
    25-37: Joint Vel (12)
    37-49: Action History (12) -> 内部状态, 通常不加
    49-53: Contacts (4) -> 足部接触布尔值, 不加
    """
    batch_size, dim = obs_prop.shape

    # 定义噪声标准差向量
    std_vec = torch.zeros(dim, device=device)

    # 设置各物理量的基准噪声强度
    std_vec[0:3] = 0.5     # Ang Vel (rad/s)
    std_vec[3:5] = 0.1    # IMU (rad)
    std_vec[13:25] = 0.05  # Joint Pos (rad)
    std_vec[25:37] = 1.0   # Joint Vel (rad/s)

    # 生成并叠加噪声
    noise = torch.randn_like(obs_prop) * std_vec * noise_scale

    return noise


def main():  # noqa: C901
    """主入口：加载教师策略、循环采集并落盘。"""

    parser = build_arg_parser()
    args_cli = parser.parse_args()
    if args_cli.video:
        args_cli.enable_cameras = True

    out_dir = Path(args_cli.out)
    resume_dataset = args_cli.resume_dataset
    existing_total_steps = 0
    existing_meta: Dict[str, object] = {}
    shard_start_index = 0
    stats_cache: Optional[Dict[str, np.ndarray]] = None

    if out_dir.exists():
        shards_dir = out_dir / "shards"
        if resume_dataset:
            print(f"[collect] 检测到已存在的数据集目录：{out_dir}，将继续追加新的采集数据。")
            if shards_dir.exists():
                shard_start_index = len(sorted(shards_dir.glob("shard_*.npz")))
            meta_path = out_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as meta_file:
                    existing_meta = json.load(meta_file)
                    existing_total_steps = existing_meta.get("total_steps", 0)
            stats_path = out_dir / "stats.npz"
            if stats_path.exists():
                with np.load(stats_path) as stats_npz:
                    stats_cache = {key: stats_npz[key].copy() for key in stats_npz.files}
        else:
            response = input(
                f"[collect] 目标目录 {out_dir} 已存在，是否删除并重新采集? (y/n): "
            ).strip().lower()
            if response == "y":
                shutil.rmtree(out_dir)
                print(f"[collect] 已清空 {out_dir}。")
                shard_start_index = 0
                existing_meta = {}
                existing_total_steps = 0
                stats_cache = None
            else:
                print("[collect] 取消采集，未对数据进行修改。")
                return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 启动 Omniverse/Isaac Sim 应用，后续采集依赖该模拟实例。
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import parkour_tasks  # noqa: F401  # 确保任务在 App 启动后完成注册
    import gymnasium as gym
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
    from vecenv_wrapper import ParkourRslRlVecEnvWrapper

    # 解析任务配置，并允许通过 CLI 覆盖设备、并行环境数量等参数。
    device_cli = getattr(args_cli, "device", None)
    disable_fabric = getattr(args_cli, "disable_fabric", False)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device_cli,
        num_envs=args_cli.num_envs,
        use_fabric=not disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 与训练/回放脚本相同，自动解析教师策略 checkpoint。
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            raise RuntimeError("未找到可用的预训练教师 checkpoint。")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 构建 Gym 环境，并在需要时添加视频录制包装。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(args_cli.out, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # VecEnvWrapper 负责桥接 Isaac 环境与 RSL-RL runner 的接口。
    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建 OnPolicyRunner，用于载入教师策略与估计器（privileged states 估计器）。
    runner = OnPolicyRunnerWithExtractor(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=vec_env.device)
    estimator = runner.get_estimator_inference_policy(device=vec_env.device)

    estimator_cfg = agent_cfg.to_dict()["estimator"]
    num_prop = estimator_cfg["num_prop"]
    num_scan = estimator_cfg["num_scan"]
    num_priv_explicit = estimator_cfg["num_priv_explicit"]
    priv_start = num_prop + num_scan
    priv_end = priv_start + num_priv_explicit

    # 可选加载学生深度编码器，便于在线生成 latent 与 yaw。
    depth_encoder = prepare_depth_encoder(
        args_cli.depth_encoder_checkpoint,
        estimator_cfg,
        agent_cfg.policy,
        vec_env.device,
    )
    latent_interval = max(1, args_cli.latent_interval)

    # 获取首次观测，同时初始化 episode 相关计数器。
    obs, extras = vec_env.get_observations()
    episode_ids = torch.arange(vec_env.num_envs, device=vec_env.device, dtype=torch.long)
    next_episode_id = episode_ids[-1].item() + 1
    step_in_episode = torch.zeros(vec_env.num_envs, device=vec_env.device, dtype=torch.long)

    # 准备输出目录与缓冲区。
    out_dir = Path(args_cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardedNPZWriter(out_dir, start_index=shard_start_index)
    buffer: Dict[str, List[np.ndarray]] = defaultdict(list)
    steps_in_buffer = 0

    # 在线累计观测与动作的统计信息，用于训练时归一化。
    fallback_sample_count = None
    if resume_dataset and existing_total_steps > 0:
        num_envs_meta = existing_meta.get("num_envs", args_cli.num_envs)
        fallback_sample_count = int(num_envs_meta * existing_total_steps)

    obs_stats = RunningStats(
        num_prop,
        state=extract_state_from_cache(stats_cache, "obs_prop", fallback_sample_count),
    )
    action_stats = RunningStats(
        vec_env.num_actions,
        state=extract_state_from_cache(stats_cache, "action", fallback_sample_count),
    )
    depth_latent_initial = extract_state_from_cache(
        stats_cache,
        "depth_latent",
        fallback_sample_count,
    )
    depth_latent_stats = (
        RunningStats(32, state=depth_latent_initial)
        if depth_encoder is not None
        else None
    )

    total_steps = args_cli.total_steps
    total_iterations = 0
    progress_interval = max(1, total_steps // 50)
    latent_cache = torch.zeros((vec_env.num_envs, 32), device=vec_env.device)
    yaw_cache = torch.zeros((vec_env.num_envs, 2), device=vec_env.device)

    perturb_prob = 0.3  # 30% 的概率使用噪声动作
    noise_scale = 0.2

    while total_iterations < total_steps:
        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用 TeacherCam 任务。")

        """ observation扰动逻辑 """
        if args_cli.noised_observation:
            prop_noise = generate_proprio_noise(
                obs[:, :num_prop],
                noise_scale=noise_scale,
                device=vec_env.device
            )
            depth_noise = torch.randn_like(depth_image) * noise_scale
            use_noise_mask = torch.rand(vec_env.num_envs, device=vec_env.device) < perturb_prob
            if use_noise_mask.any():
                obs[:, :num_prop] += (prop_noise * use_noise_mask.unsqueeze(-1))
                mask_expanded = use_noise_mask.view(vec_env.num_envs, 1, 1)
                depth_image += (depth_noise * mask_expanded)
                depth_image = torch.clamp(depth_image, min=0.0)  # 深度图通常不能为负

        obs_prop = obs[:, :num_prop]
        obs_prop_cpu = obs_prop.detach().cpu().numpy().astype(np.float32)

        obs_est = obs.clone()
        priv_est = estimator(obs_est[:, :num_prop])
        obs_est[:, priv_start:priv_end] = priv_est

        actions = policy(obs_est, hist_encoding=True)
        actions_cpu = actions.detach().cpu().numpy().astype(np.float32)

        """ 对env施加扰动后的动作从而到达特殊状态，采集未扰动的action作为label """
        if args_cli.noised_action:
            # 考虑了向量化环境，生成一个随机掩码，决定哪些环境在这个 step 使用噪声
            use_noise_mask = torch.rand(vec_env.num_envs, device=vec_env.device) < perturb_prob
            if use_noise_mask.any():
                noise = torch.randn_like(actions) * noise_scale
                # 只对被选中的环境添加噪声
                actions = actions + (noise * use_noise_mask.unsqueeze(-1))

        if depth_encoder is not None:
            if total_iterations % latent_interval == 0:
                obs_student = obs[:, :num_prop].clone()
                obs_student[:, 6:8] = 0
                with torch.inference_mode():
                    depth_out = depth_encoder(depth_image, obs_student)
                depth_encoder.detach_hidden_states()
                latent_cache = depth_out[:, :-2]
                yaw_cache = depth_out[:, -2:]

        # 环境前进一步，返回新的观测、奖励与终止标记。
        obs_next, rews, dones, extras = vec_env.step(actions)
        rews_cpu = rews.detach().cpu().numpy().astype(np.float32)
        dones_bool = dones.squeeze(-1).bool()

        # 将当前步的数据压入缓冲，待达到 shard 后统一写盘。
        buffer["obs_prop"].append(obs_prop_cpu)
        buffer["action_teacher"].append(actions_cpu)
        buffer["reward"].append(rews_cpu)
        buffer["done"].append(dones_bool.cpu().numpy())
        buffer["episode_id"].append(episode_ids.detach().cpu().numpy())
        buffer["step_in_episode"].append(step_in_episode.detach().cpu().numpy())
        buffer["depth"].append(convert_depth(depth_image, args_cli.depth_dtype, args_cli.depth_scale))
        buffer["priv_estimate"].append(priv_est.detach().cpu().numpy().astype(np.float32))

        if depth_encoder is not None:
            buffer["depth_latent"].append(latent_cache.detach().cpu().numpy().astype(np.float32))
            buffer["yaw"].append(yaw_cache.detach().cpu().numpy().astype(np.float32))

        obs_stats.update(obs_prop_cpu)
        action_stats.update(actions_cpu)
        if depth_latent_stats is not None:
            depth_latent_stats.update(latent_cache.detach().cpu().numpy())

        steps_in_buffer += 1
        total_iterations += 1

        step_in_episode += 1
        reset_mask = dones_bool
        if reset_mask.any():
            step_in_episode[reset_mask] = 0
            num_reset = int(reset_mask.sum().item())
            new_ids = torch.arange(next_episode_id, next_episode_id + num_reset, device=vec_env.device)
            episode_ids[reset_mask] = new_ids
            next_episode_id += num_reset
            reset_depth_hidden_states(depth_encoder, reset_mask)
        episode_ids[~reset_mask] = episode_ids[~reset_mask]

        obs = obs_next

        if (
            total_iterations % progress_interval == 0
            or total_iterations == total_steps
        ):
            percent = total_iterations / total_steps * 100.0
            print(
                f"[collect] 进度：{total_iterations}/{total_steps} ({percent:5.1f}%)",
                flush=True,
            )

        if steps_in_buffer >= args_cli.shard_size:
            writer.write(buffer)
            buffer = defaultdict(list)
            steps_in_buffer = 0

    if steps_in_buffer > 0:
        writer.write(buffer)
        buffer = defaultdict(list)

    # 落盘统计信息（均值/标准差），方便训练脚本直接加载。
    stats_payload = {
        "obs_prop": obs_stats.finalize(),
        "action": action_stats.finalize(),
        "depth_latent": depth_latent_stats.finalize() if depth_latent_stats else None,
    }

    stats_arrays = {
        "obs_prop_mean": np.array(stats_payload["obs_prop"]["mean"], dtype=np.float32),
        "obs_prop_std": np.array(stats_payload["obs_prop"]["std"], dtype=np.float32),
        "obs_prop_count": np.array([stats_payload["obs_prop"]["count"]], dtype=np.int64),
        "action_mean": np.array(stats_payload["action"]["mean"], dtype=np.float32),
        "action_std": np.array(stats_payload["action"]["std"], dtype=np.float32),
        "action_count": np.array([stats_payload["action"]["count"]], dtype=np.int64),
    }
    if stats_payload["depth_latent"]:
        stats_arrays["depth_latent_mean"] = np.array(
            stats_payload["depth_latent"]["mean"], dtype=np.float32
        )
        stats_arrays["depth_latent_std"] = np.array(
            stats_payload["depth_latent"]["std"], dtype=np.float32
        )
        stats_arrays["depth_latent_count"] = np.array(
            [stats_payload["depth_latent"]["count"]], dtype=np.int64
        )

    np.savez(out_dir / "stats.npz", **stats_arrays)

    camera_shape = depth_image.shape

    # 记录元信息，包含任务名、分片列表、相机分辨率等。
    total_recorded_steps = existing_total_steps + total_iterations
    shard_files = sorted(p.name for p in writer.shard_dir.glob("*.npz"))
    meta = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "total_steps": total_recorded_steps,
        "dataset_format": args_cli.dataset_format,
        "depth_dtype": args_cli.depth_dtype,
        "depth_scale": args_cli.depth_scale,
        "latent_interval": latent_interval,
        "depth_encoder_checkpoint": args_cli.depth_encoder_checkpoint,
        "camera_resolution": list(camera_shape[-2:]),
        "step_dt": float(vec_env.unwrapped.step_dt),
        "fields": shard_files,
        "resume": resume_dataset,
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as meta_file:
        json.dump(meta, meta_file, ensure_ascii=False, indent=2)

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
