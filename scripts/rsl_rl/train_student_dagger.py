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
from typing import Dict, List, Tuple, Optional

# Ensure project-local packages (parkour_isaaclab, parkour_tasks, etc.) are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

# === Import aggregator and student policy (same directory) ===
from train_student_from_dataset import SequenceAggregator, MultiModalStudentPolicy, TransformerXLTemporal


# ====== Isaac Lab / task loading (same as collect.py) ======
def load_env_and_teacher(args):
    # 必须先实例化 AppLauncher / SimulationApp，再导入依赖 Omniverse 的模块
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    # 触发 parkour_tasks 中 Gym 环境注册（包括 TeacherCam 任务）
    import parkour_tasks  # noqa: F401
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import parse_env_cfg
    import cli_args
    from modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
    from vecenv_wrapper import ParkourRslRlVecEnvWrapper

    # 与 collect.py 保持一致：从 CLI 中读取 device / disable_fabric 参数
    device_cli = getattr(args, "device", None)
    disable_fabric = getattr(args, "disable_fabric", False)
    env_cfg = parse_env_cfg(
        args.task,
        device=device_cli,
        num_envs=args.num_envs,
        use_fabric=not disable_fabric,
    )

    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)

    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load teacher (same as collect.py)
    runner = OnPolicyRunnerWithExtractor(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.teacher_checkpoint, load_optimizer=False)
    teacher_policy = runner.get_inference_policy(device=vec_env.device)

    return vec_env, teacher_policy, agent_cfg, simulation_app


class StudentOnlineRunner:
    """
    Vectorized Student Runner with TransformerXL Memory.

    This version:
    1. Uses tensors for history buffers (no deques/loops).
    2. Uses Batched Mems (List[Tensor]) instead of List[List[Tensor]].
    3. Directly calls model.forward_with_mems.
    """

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
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.device = device

        # History Buffer: 行为类似deque, 不过现在使用np.roll实现
        self.prop_histories = torch.zeros(
            num_envs, prop_hist_len, proprio_dim,
            dtype=torch.float32, device=device
        )
        self.depth_histories = torch.zeros(
            num_envs, depth_hist_len, *camera_resolution,
            dtype=torch.float32, device=device
        )

        # TransformerXL Memory, Shape: List[Tensor], where each Tensor is [Num_Envs, Mem_Len, D_Model]
        self.mems: Optional[List[torch.Tensor]] = None

    def reset(self) -> None:
        """Reset all environments."""
        self.prop_histories.zero_()
        self.depth_histories.zero_()
        # Directly set to None. TransformerXLTemporal handles it automatically.
        self.mems = None

    def reset_done(self, done_mask: torch.Tensor) -> None:
        """
        Reset histories and memories for done environments.
        Args:
            done_mask: Boolean tensor of shape [num_envs]
        """
        if not done_mask.any():
            return

        self.prop_histories[done_mask] = 0
        self.depth_histories[done_mask] = 0

        if self.mems is not None:
            for i in range(len(self.mems)):
                self.mems[i][done_mask] = 0

    def act(
        self,
        obs_prop: torch.Tensor,     # [num_envs, proprio_dim]
        depth_image: torch.Tensor,  # [num_envs, H, W] or [num_envs, 1, H, W]
    ) -> torch.Tensor:
        """
        Perform one inference step using cached memory.
        """
        obs_prop = obs_prop.to(self.device)
        depth_image = depth_image.to(self.device)

        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)

        self.prop_histories = torch.roll(self.prop_histories, -1, dims=1)
        self.depth_histories = torch.roll(self.depth_histories, -1, dims=1)

        self.prop_histories[:, -1, :] = obs_prop
        self.depth_histories[:, -1, :, :] = depth_image

        # Prepare inputs for the model
        # 1. Flatten proprio history: [B, Hist, Dim] -> [B, Hist*Dim]
        # 2. Add Sequence dimension S=1: [B, S=1, Features]
        prop_input = self.prop_histories.view(self.num_envs, -1).unsqueeze(1)

        # Depth Input: [B, S=1, Hist, H, W]
        depth_input = self.depth_histories.unsqueeze(1)

        # [Answer 4] Direct Model Call
        with torch.no_grad():
            actions, self.mems = self.model.forward_with_mems(
                prop_input,
                depth_input,
                mems=self.mems
            )

        return actions.squeeze(1)  # Remove Sequence dim -> [B, Action_Dim]


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
    parser.add_argument("--prop_hist_len", type=int, default=3)
    parser.add_argument("--depth_hist_len", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

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
    fusion_cfg = {"num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.1, "grid_size": 4}
    temporal_cfg = {"num_layers": 3, "num_heads": 4, "d_inner": 256, "mem_len": 64, "dropout": 0.1}
    action_head_cfg = {"hidden_dims": (256, 256), "tanh_output": False, "action_scale": 1.0}

    student_model = MultiModalStudentPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

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

    # 5. Initialize Helpers
    # Aggregator: Collects (obs, teacher_action, done) for TRAINING
    aggregator = SequenceAggregator(
        num_envs=args.num_envs,
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

    # Stats buffers
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    track_progress_hist = deque(maxlen=100)
    current_returns = torch.zeros(args.num_envs, device=device)
    current_lengths = torch.zeros(args.num_envs, device=device)
    dones_bool = torch.zeros(args.num_envs, dtype=torch.bool, device=device)  # Track prev dones

    try:
        # 尝试从 unwrapped 环境中获取 parkour_manager
        base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")
        num_goals = int(getattr(base_parkour, "num_goals", 0))
    except Exception:
        base_parkour = None
        num_goals = 0
        print("[Warning] Could not get parkour_manager. Track progress logging disabled.")

    print(f"[Info] Starting DAgger loop for {args.num_iters} iterations...")
    for it in range(start_iter, args.num_iters):
        iter_start = time.time()
        student_model.eval()  # Eval mode for rollout

        """ Collection Phase (Run until we have a full sequence batch) """
        batch_data = None
        while batch_data is None and simulation_app.is_running():
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
            student_prop[:, 12] = dones_bool.float()
            student_depth = depth_image.clone()

            if dropout_manager:
                dropout_manager.reset_env(dones_bool)
                dropout_manager.update(depth_image=student_depth, obs_prop=student_prop)

            # D. Student Action (Learner) - Uses AUGMENTED observations
            student_actions = student_runner.act(student_prop, student_depth)

            goal_idx_before_step = None
            if base_parkour is not None and num_goals > 0:
                goal_idx_before_step = base_parkour.cur_goal_idx.detach().cpu().numpy().copy()

            # E. Action Selection (Mixture)
            if it < args.num_pretrain_iters:
                actions_to_env = teacher_actions
            else:
                actions_to_env = student_actions
                if args.teacher_mixture:
                    # Decay beta
                    progress = max(it - args.num_pretrain_iters, 0)
                    mix_frac = min(progress / max(args.beta_decay_iters, 1), 1.0)
                    beta = args.beta_start + (args.beta_end - args.beta_start) * mix_frac

                    # Sample mask
                    mask = torch.rand(args.num_envs, device=device) < beta
                    actions_to_env = torch.where(mask.unsqueeze(-1), teacher_actions, student_actions)

            # F. Step Environment
            obs, rewards, dones, extras = vec_env.step(actions_to_env)
            dones_bool = dones.squeeze(-1).bool()  # [N]

            # G. Store in Aggregator
            batch_data = aggregator.push_step(
                obs_prop=student_prop.cpu().numpy(),
                depth_frame=student_depth.cpu().numpy(),
                teacher_actions=teacher_actions.cpu().numpy(),
                done=dones_bool.cpu().numpy()
            )

            # H. Handle Resets for Inference Runner
            if dones_bool.any():
                student_runner.reset_done(dones_bool)

                # Update stats
                current_returns += rewards.squeeze(-1)
                current_lengths += 1

                done_indices = torch.nonzero(dones_bool).squeeze(-1)
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
            else:
                current_returns += rewards.squeeze(-1)
                current_lengths += 1

        """batch sequence data assembled, train on batch data"""
        student_model.train()

        # Prepare Batch, Shape: [B, S, ...]
        b_prop = batch_data["proprio"].to(device)
        b_depth = batch_data["depth"].to(device)
        b_actions = batch_data["actions"].to(device)
        b_dones = batch_data["dones"].to(device)

        # Forward with segment recurrence
        pred_actions, new_train_mems = student_model.forward_with_mems(
            b_prop, b_depth, mems=train_mems
        )

        loss = nn.functional.mse_loss(pred_actions, b_actions)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        optimizer.step()

        # Update train_mems for next iteration
        if new_train_mems is not None:
            # shape of mems is [num_layers, num_envs, Mem_Len, D_Model], do Truncated BPTT
            train_mems = TransformerXLTemporal.detach_mems(new_train_mems)
            batch_size = b_dones.shape[0]

            # 遍历每一个环境 (Batch Dimension)
            for b in range(batch_size):
                # 查找当前环境在该序列中所有 done 为 True 的位置
                # torch.nonzero 返回 shape [N, 1]
                done_indices = torch.nonzero(b_dones[b])

                if done_indices.numel() > 0:
                    # 找到该序列中 *最后一个* done 的位置
                    last_done_idx = done_indices.max().item()
                    # 核心修正逻辑:
                    # 如果在第t步结束了Episode，那么t及t之前的所有Memory都属于旧Episode。
                    # 下一个 Batch (从 t+1 或 S+1 开始) 不应该看到这些信息。
                    # 因此将 [0, last_done_idx] 闭区间的mem清零。
                    for layer_mem in train_mems:
                        # layer_mem shape: [Batch, Mem_Len, D_Model]
                        layer_mem[b, :last_done_idx + 1, :] = 0.0

        else:
            train_mems = None

        # --- Logging ---
        dt = time.time() - iter_start
        if (it + 1) % 10 == 0:
            with torch.no_grad():
                diff_rmse = torch.sqrt(torch.mean((pred_actions - b_actions) ** 2)).item()
                teacher_rms = torch.sqrt(torch.mean(b_actions ** 2)).item()

            print(f"[Iter {it+1}] Loss: {loss.item():.5f} | Time: {dt:.2f}s | RMSE: {diff_rmse:.4f}")

            if args.wandb and WANDB_AVAILABLE:
                log_data = {
                    "dagger/loss": loss.item(),
                    "dagger/diff_rmse": diff_rmse,
                    "dagger/teacher_rms": teacher_rms,
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
                "iter": it + 1,
                "meta": {
                    "is_dagger": True,
                    "task": args.task,
                    "num_prop": proprio_dim,
                    "action_dim": action_dim,
                    "camera_resolution": camera_resolution
                }
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    vec_env.close()
    simulation_app.close()
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
