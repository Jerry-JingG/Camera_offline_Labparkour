"""
Online DAGGER Trainer
Student always acts; teacher provides labels.
Aggregator & Student model are imported from train_student_from_dataset
so token construction, TXL input format, and training logic are fully consistent.

Usage Example:
    python train_student_dagger.py \
        --task Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0 \
        --teacher_checkpoint path/to/teacher.pt \
        --student_checkpoint path/to/student.pt \
        --num_iters 2000 \
        --sequence_length 64 \
        --num_envs 16 \
        --device cuda:0
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

# Ensure project-local packages (parkour_isaaclab, parkour_tasks, etc.) are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

# === Import aggregator and student policy (same directory) ===
from train_student_from_dataset import SequenceAggregator, MultiModalStudentPolicy, TransformerXLTemporal
from utils.dropout_manager import CameraDropoutManager


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

    # Optionally turn off depth-camera debug visualization window (cv2.imshow).
    if getattr(args, "disable_depth_debug_vis", False):
        try:
            depth_cam_cfg = env_cfg.observations.depth_camera.depth_cam
            if "debug_vis" in depth_cam_cfg.params:
                depth_cam_cfg.params["debug_vis"] = False
                print("[INFO] Disabled depth camera debug_vis for DAGGER run.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to disable depth debug_vis: {exc}")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)

    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load teacher (same as collect.py)
    runner = OnPolicyRunnerWithExtractor(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.teacher_checkpoint, load_optimizer=False)
    teacher_policy = runner.get_inference_policy(device=vec_env.device)

    return vec_env, teacher_policy, agent_cfg


# ============================================================
def parse_args():
    p = argparse.ArgumentParser("train_student_dagger")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--teacher_checkpoint", type=str, required=True)
    p.add_argument("--student_checkpoint", type=str, default=None)
    p.add_argument(
        "--teacher_hist_encoding",
        action="store_true",
        help="If set, teacher uses historical encoding (hist_encoding=True) like train.py distillation; default False.",
    )

    p.add_argument(
        "--disable_depth_debug_vis",
        action="store_true",
        help="Disable cv2 depth camera debug window.",
    )

    p.add_argument("--num_iters", type=int, default=2000)

    p.add_argument("--sequence_length", type=int, default=64)
    p.add_argument("--prop_hist_len", type=int, default=1)
    p.add_argument("--depth_hist_len", type=int, default=4)
    p.add_argument(
        "--num_pretrain_iters",
        type=int,
        default=0,
        help="预热迭代次数：在这段迭代内由 Teacher 执行环境，学生只学习。",
    )
    p.add_argument(
        "--teacher_mixture",
        action="store_true",
        help="开启后：在学生执行阶段，以概率 beta 让教师接管动作（beta 会按迭代衰减）。",
    )
    p.add_argument(
        "--teacher_mixture_beta_start",
        type=float,
        default=0.6,
        help="mixture 初始 teacher 概率 beta_start。",
    )
    p.add_argument(
        "--teacher_mixture_beta_end",
        type=float,
        default=0.1,
        help="mixture 最低 teacher 概率 beta_end。",
    )
    p.add_argument(
        "--teacher_mixture_decay_iters",
        type=int,
        default=800,
        help="从 beta_start 线性衰减到 beta_end 所需的迭代数。",
    )

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--save_dir", type=str, default="student_dagger_outputs")

    # Camera dropout for student (simulates camera blackout)
    p.add_argument(
        "--camera_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of camera being offline for student (0.0=disabled). Teacher always sees clean depth.",
    )

    # 追加 RSL-RL 相关参数（resume / logger 等），保持与 train.py 一致的 CLI 行为
    try:
        import cli_args  # type: ignore

        cli_args.add_rsl_rl_args(p)
    except ImportError:
        pass

    # 追加 IsaacLab AppLauncher 相关参数，以便支持 --headless / --enable_cameras 等 CLI 选项
    try:
        from isaaclab.app import AppLauncher  # type: ignore

        AppLauncher.add_app_launcher_args(p)
    except ImportError:
        # 若未安装 IsaacLab，则忽略这些额外参数（主要用于本地测试脚本语法）
        pass

    return p.parse_args()


# ============================================================
def main():
    args = parse_args()
    device = torch.device(args.device)

    # -------------------------- wandb setup (optional) --------------------------
    use_wandb = False
    wandb_run = None
    if getattr(args, "logger", None) == "wandb":
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[WARN] wandb selected but not installed; skipping wandb logging.")
        else:
            wandb_run = wandb.init(
                project=getattr(args, "log_project_name", None) or "student-dagger",
                name=getattr(args, "run_name", None),
                config={k: v for k, v in vars(args).items() if k != "logger"},
                reinit=True,
            )
            use_wandb = True

    # load env + teacher
    vec_env, teacher_policy, agent_cfg = load_env_and_teacher(args)

    # initial observation to infer dims
    obs, extras = vec_env.get_observations()
    obs = obs.to(device)

    # infer proprio + depth dims
    if "depth_camera" in extras["observations"]:
        depth0 = extras["observations"]["depth_camera"]
        cam_res = (int(depth0.shape[-2]), int(depth0.shape[-1]))
    else:
        raise RuntimeError("Depth camera not found in observation")

    # Isaac's agent config gives num_prop
    num_prop = int(agent_cfg.estimator.num_prop)
    action_dim = int(getattr(vec_env, "num_actions", obs.shape[1]))
    # meta 信息用于与 dataset 训练保持一致的 checkpoint 结构
    ckpt_meta = {
        "num_prop": num_prop,
        "action_dim": action_dim,
        "camera_resolution": cam_res,
        "prop_hist_len": args.prop_hist_len,
        "depth_hist_len": args.depth_hist_len,
        "sequence_length": args.sequence_length,
        "task": args.task,
    }

    # ===== Build Student Policy (same config as offline training) =====
    fusion_cfg = {"num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.1, "attn_dropout": 0.1, "grid_size": 4}
    temporal_cfg = {"num_layers": 3, "num_heads": 4, "d_inner": 256, "mem_len": args.sequence_length, "dropout": 0.1, "attn_dropout": 0.1}
    action_head_cfg = {"hidden_dims": (256, 256), "tanh_output": False, "action_scale": 1.0}

    student = MultiModalStudentPolicy(
        proprio_dim=num_prop,
        action_dim=action_dim,
        camera_resolution=cam_res,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
    ).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.student_checkpoint:
        ckpt = torch.load(args.student_checkpoint, map_location="cpu")
        student.load_state_dict(ckpt["model_state_dict"])
        print(f"[load] resumed student from {args.student_checkpoint}")

    # ===== Aggregator (same as offline trainer) =====
    aggregator = SequenceAggregator(
        num_envs=args.num_envs,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        sequence_len=args.sequence_length,
        num_prop=num_prop,
        depth_shape=cam_res,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Camera dropout manager: applies dropout to student's depth only
    dropout_manager = None
    if args.camera_dropout_prob > 0:
        dropout_manager = CameraDropoutManager(
            num_envs=args.num_envs,
            device=device,
            dt=vec_env.unwrapped.step_dt,
            prob_start_offline=args.camera_dropout_prob,
        )
        print(f"[INFO] Camera dropout enabled with prob={args.camera_dropout_prob}")

    global_step = 0
    # Episode stats buffers for online logging
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    ep_return_hist = deque(maxlen=256)
    ep_length_hist = deque(maxlen=256)
    track_progress_hist = deque(maxlen=256)
    timeout_hist = deque(maxlen=256)
    # TXL 记忆状态（按层存放），用于在线推理加速
    txl_mems = None
    # 训练时的记忆状态 (Segment Recurrence)
    train_mems = None
    max_ep_len = float(getattr(vec_env.unwrapped, "max_episode_length", 0.0))
    # 子地形目标信息（来自 base_parkour 事件）
    try:
        base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")
        num_goals = int(getattr(base_parkour, "num_goals", 0))
    except Exception:
        base_parkour = None
        num_goals = None

    # ===== main training loop =====
    train_start_t = time.time()
    for it in range(args.num_iters):
        start_t = time.time()
        batch = None
        episodes_this_iter = 0

        obs, extras = vec_env.get_observations()
        obs = obs.to(device)

        # Yaw buffer for loss calculation
        yaws_buffer = []
        # True yaw buffer for training loss (extracted before masking)
        true_yaws_buffer = []

        while batch is None:
            # --- prepare numpy obs for aggregator ---
            obs_prop_np = obs[:, :num_prop].cpu().numpy()
            depth_np = extras["observations"]["depth_camera"].cpu().numpy()

            # --- teacher labels ---
            # teacher_policy 是 OnPolicyRunnerWithExtractor.get_inference_policy 返回的可调用对象，
            # 接口与 ActorCriticRMA.act_inference 对齐。
            with torch.no_grad():
                teacher_actions = teacher_policy(obs, hist_encoding=args.teacher_hist_encoding)
            teacher_actions_cpu = teacher_actions.cpu()
            teacher_actions_np = teacher_actions_cpu.numpy()

            # --- Store true yaw BEFORE masking for loss calculation ---
            # This ensures we have ground truth yaw even though batch contains masked data
            true_yaw_current = obs_prop_np[:, 6:8].copy()  # [B, 2]
            true_yaws_buffer.append(true_yaw_current)

            # --- Mask privileged information for student ---
            # Teacher policy can see delta_yaw (indices 6-7) in obs_buf
            # Student policy should not have access to this privileged direction info
            # Indices:
            #   6: delta_yaw (current target direction - robot yaw)
            #   7: delta_next_yaw (next target direction - robot yaw)
            obs_prop_np_masked = obs_prop_np.copy()
            obs_prop_np_masked[:, 6:8] = 0  # Zero out delta_yaw and delta_next_yaw

            # --- student acting ---
            # build student input from its own histories (aggregator stores them)
            # aggregator.prop_history: [N, H, D], contains steps [t-H, ..., t-1]
            # We want [N, H, D] containing [t-H+1, ..., t]
            
            # Efficient Vectorized Construction
            # Proprio: shift left and append current
            curr_prop_hist = aggregator.prop_history.copy()
            curr_prop_hist = np.roll(curr_prop_hist, -1, axis=1)
            # Apply masking to the student's current observation input
            curr_prop_hist[:, -1, :] = obs_prop_np_masked
            
            # Depth: apply dropout for student only
            curr_depth_hist = aggregator.depth_history.copy()
            curr_depth_hist = np.roll(curr_depth_hist, -1, axis=1)
            # Apply camera dropout to student's current depth frame
            depth_for_student = depth_np.copy()
            if dropout_manager is not None:
                depth_t_tmp = torch.from_numpy(depth_for_student).float().to(device)
                depth_t_tmp = dropout_manager.update(depth_t_tmp)
                depth_for_student = depth_t_tmp.cpu().numpy()
            curr_depth_hist[:, -1, :, :] = depth_for_student

            # Flatten Proprio: [N, H, D] -> [N, H*D]
            prop_flat = curr_prop_hist.reshape(args.num_envs, -1)
            
            prop_step = torch.from_numpy(prop_flat).float().to(device)
            depth_step = torch.from_numpy(curr_depth_hist).float().to(device)

            # TXL 单步推理：使用记忆状态 txl_mems
            student.eval()
            with torch.no_grad():
                # Define delta_yaw_ok masking (start with all True)
                delta_yaw_ok = torch.ones(args.num_envs, dtype=torch.bool, device=device)

                # Forward pass with delta_yaw_ok to get yaw predictions
                actions_step, yaw_pred_step, new_mems = student.forward_step(
                    prop_step, depth_step, mems=txl_mems, delta_yaw_ok=delta_yaw_ok
                )
                student_act = actions_step.cpu()

                # Calculate yaw error for loss (true - predicted)
                # Note: yaw_pred_step is the unscaled prediction, model scales by 1.5 internally
                true_yaw = torch.from_numpy(obs_prop_np[:, 6:8]).float().to(device)  # [B, 2]
                yaw_error = true_yaw - yaw_pred_step * 1.5  # Compare with scaled prediction
                yaws_buffer.append(yaw_error.detach())

            # 在环境 step 前缓存当前的 goal 索引（否则 step 内部 reset 后 cur_goal_idx 会被清零）
            if base_parkour is not None and num_goals and num_goals > 0:
                goal_idx_before_step = base_parkour.cur_goal_idx.detach().cpu().numpy().copy()
            else:
                goal_idx_before_step = None

            # --- env step (student acts / teacher acts) ---
            # 预热阶段：由 Teacher 推进环境；之后由 Student 推进（可选 mixture）
            if it < args.num_pretrain_iters:
                act_to_env = teacher_actions_cpu
            else:
                act_to_env = student_act
                if args.teacher_mixture:
                    progress = max(it - args.num_pretrain_iters, 0)
                    decay = max(args.teacher_mixture_decay_iters, 1)
                    mix_frac = min(progress / decay, 1.0)
                    beta = args.teacher_mixture_beta_start + (args.teacher_mixture_beta_end - args.teacher_mixture_beta_start) * mix_frac
                    beta = float(np.clip(beta, 0.0, 1.0))
                    mask = (torch.rand(args.num_envs) < beta).unsqueeze(-1)
                    act_to_env = torch.where(mask, teacher_actions_cpu, student_act)

            obs, rewards, dones, infos = vec_env.step(act_to_env.to(vec_env.device))
            obs = obs.to(device)

            if "observations" in infos:
                extras = infos
            # 记录 episode 级指标
            rewards_np = rewards.squeeze(-1).cpu().numpy()
            ep_returns += rewards_np
            ep_lengths += 1

            dones_np = dones.cpu().numpy().astype(bool).reshape(-1)
            if isinstance(extras, dict) and "time_outs" in extras:
                timeouts_np = extras["time_outs"].cpu().numpy().astype(bool).reshape(-1)
            else:
                timeouts_np = np.zeros_like(dones_np, dtype=bool)

            done_indices = np.nonzero(dones_np)[0]
            if len(done_indices) > 0:
                ep_return_hist.extend(ep_returns[done_indices].tolist())
                ep_length_hist.extend(ep_lengths[done_indices].tolist())

                # ====== 赛道进度：仅基于本步前的 cur_goal_idx / num_goals ======
                if goal_idx_before_step is not None and num_goals and num_goals > 0:
                    goal_vals = np.array(goal_idx_before_step).reshape(-1)[done_indices]
                    norm_progress = np.clip(goal_vals / float(num_goals), 0.0, 1.0)
                    track_progress_hist.extend(norm_progress.tolist())

                timeout_hist.extend(timeouts_np[done_indices].astype(int).tolist())
                ep_returns[done_indices] = 0.0
                ep_lengths[done_indices] = 0
                episodes_this_iter += len(done_indices)

                # Reset dropout state for done environments
                if dropout_manager is not None:
                    dropout_manager.reset_env(torch.from_numpy(dones_np).to(device))

            # 更新 TXL 记忆：对已经 done 的环境清零对应的 memory
            if new_mems is not None:
                done_mask = torch.from_numpy(dones_np).to(device)
                for i, mem in enumerate(new_mems):
                    if mem is None or mem.numel() == 0:
                        continue
                    # mem: [B, M, C]，将 done 的 env 的历史置零
                    mem[done_mask] = 0.0
                txl_mems = new_mems

            # --- push to aggregator ---
            # CRITICAL: Use masked proprio to match inference distribution
            # During inference, student never sees true delta_yaw (indices 6:8)
            # Training must use the same masked distribution
            batch = aggregator.push_step(
                obs_prop=obs_prop_np_masked,
                depth_frame=depth_np,
                teacher_actions=teacher_actions_np,
                done=dones_np,
            )

        # ====== train on batch ======
        student.train()
        proprio_t = torch.from_numpy(batch["proprio"]).float().to(device)
        depth_t = torch.from_numpy(batch["depth"]).float().to(device)
        teacher_t = torch.from_numpy(batch["actions"]).float().to(device)

        # 训练阶段：按完整序列前向，与离线监督训练保持一致
        # 使用 Segment Recurrence: 传入上一段的 train_mems
        pred, yaw_pred, new_train_mems = student.forward_with_mems(proprio_t, depth_t, mems=train_mems)
        
        # Detach memory for next segment to stop gradient backprop through segments
        train_mems = TransformerXLTemporal.detach_mems(new_train_mems)

        # Handle Done: 如果某环境在此段序列中经历了 done，则将其 memory 重置
        # batch["dones"]: [B, S]
        # 只要序列中有任意 step 为 done，下一段就无法简单接续上一段的 memory (保守策略：全清)
        # 或者更精细地：如果最后一个 step 是 done，肯定清；如果在中间 done，TransformerXL 已经混淆了前后 episode，
        # 还是清了比较安全，或者依赖 mask (但 TXL 实现里 mask 只管 attention)。
        # 这里采用：只要有 done，就清空该 env 的 memory。
        any_done = torch.from_numpy(batch["dones"]).any(dim=1).to(device)  # [B]
        if any_done.any():
            for l_idx in range(len(train_mems)):
                if train_mems[l_idx] is not None:
                    # 使用非原地操作，避免修改 graph 中引用的 underlying storage
                    mask = any_done.view(-1, 1, 1)
                    train_mems[l_idx] = torch.where(mask, torch.zeros_like(train_mems[l_idx]), train_mems[l_idx])

        loss_actions = nn.functional.mse_loss(pred, teacher_t)

        # Calculate yaw loss using true yaw values stored before masking
        # true_yaws_buffer contains [B, 2] arrays for each step in the sequence
        # Stack them to get [B, S, 2] shape matching yaw_pred
        true_yaw_train = torch.from_numpy(np.stack(true_yaws_buffer, axis=1)).float().to(device)  # [B, S, 2]

        # Note: yaw_pred is the unscaled prediction from the model
        # The model internally scales by 1.5 for proprio replacement, but returns unscaled
        # So we compare scaled prediction with true yaw
        loss_yaw = nn.functional.mse_loss(yaw_pred * 1.5, true_yaw_train)

        # Total loss
        loss = loss_actions + loss_yaw

        # 监控标签与残差的幅值，便于判断 loss 量级
        with torch.no_grad():
            teacher_rms = torch.sqrt(torch.mean(teacher_t ** 2)).item()
            diff_rmse = torch.sqrt(torch.mean((pred - teacher_t) ** 2)).item()
            teacher_abs_max = torch.max(torch.abs(teacher_t)).item()
            rel_rmse = diff_rmse / (teacher_rms + 1e-8)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()

        global_step += 1

        iter_time = time.time() - start_t
        elapsed = time.time() - train_start_t
        avg_iter_time = elapsed / (it + 1)
        remaining_iters = args.num_iters - it - 1
        eta_seconds = remaining_iters * avg_iter_time
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        # throughput: steps per second
        steps_per_iter = args.num_envs * args.sequence_length
        steps_per_sec = steps_per_iter / iter_time if iter_time > 0 else 0.0

        print(
            f"[iter {it}] loss={loss.item():.5f} (action={loss_actions.item():.5f}, yaw={loss_yaw.item():.5f})   "
            f"time={iter_time:.2f}s   eta={eta_str}   steps/s={steps_per_sec:.2f}   global_step={global_step}"
        )

        # wandb logging
        if use_wandb:
            wandb_metrics = {
                "train/loss": loss.item(),
                "train/loss_actions": loss_actions.item(),
                "train/loss_yaw": loss_yaw.item(),
                "time/iter_s": iter_time,
                "time/eta_s": eta_seconds,
                "time/elapsed_s": elapsed,
                "perf/steps_per_s": steps_per_sec,
                "rollout/episodes_this_iter": episodes_this_iter,
                "teacher/action_rms": teacher_rms,
                "teacher/action_abs_max": teacher_abs_max,
                "train/diff_rmse": diff_rmse,
                "train/rel_rmse": rel_rmse,
            }
            if len(ep_return_hist) > 0:
                wandb_metrics.update(
                    {
                        "rollout/ep_return_mean": float(np.mean(ep_return_hist)),
                        "rollout/ep_return_std": float(np.std(ep_return_hist)),
                        "rollout/ep_len_mean": float(np.mean(ep_length_hist)),
                        "rollout/ep_len_std": float(np.std(ep_length_hist)),
                    }
                )
            if len(track_progress_hist) > 0:
                track_arr = np.array(track_progress_hist)
                wandb_metrics.update(
                    {
                        "rollout/track_progress_mean": float(np.mean(track_arr)),
                        "rollout/track_progress_late_ratio": float(np.mean(track_arr >= 0.7)),
                    }
                )
            if len(timeout_hist) > 0:
                timeout_rate = float(np.mean(timeout_hist))
                wandb_metrics["rollout/timeout_rate"] = timeout_rate
            
            # --- New Metric: Terrain Level ---
            if base_parkour is not None:
                # base_parkour.terrain is the ParkourTerrainImporter which holds terrain_levels
                avg_level = float(base_parkour.terrain.terrain_levels.float().mean().item())
                wandb_metrics["rollout/terrain_level_mean"] = avg_level

            # Camera dropout stats
            if dropout_manager is not None:
                wandb_metrics["camera/offline_ratio"] = dropout_manager.offline_state.float().mean().item()

            wandb.log(wandb_metrics, step=global_step)

        # save
        if (it + 1) % 100 == 0:
            ckpt = {
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iter": it,
                "global_step": global_step,
                "meta": ckpt_meta,
            }
            ckpt_path = save_dir / f"student_epoch_{it+1:06d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"[save] {ckpt_path}")

    # final save
    final_ckpt = {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": args.num_iters,
        "global_step": global_step,
        "meta": ckpt_meta,
    }
    torch.save(final_ckpt, save_dir / "student_epoch_final.pt")
    print("[done] training finished.")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
