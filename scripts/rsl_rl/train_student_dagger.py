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

import numpy as np
import torch
from torch import nn, Tensor

# === Import aggregator and student policy (same directory) ===
from train_student_from_dataset import SequenceAggregator, MultiModalStudentPolicy


# ====== Isaac Lab / task loading (same as collect.py) ======
def load_env_and_teacher(args):
    import gymnasium as gym
    from isaaclab.app import AppLauncher
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_tasks.utils import parse_env_cfg
    import cli_args
    from modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
    from vecenv_wrapper import ParkourRslRlVecEnvWrapper

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env_cfg = parse_env_cfg(args.task, device=None, num_envs=args.num_envs, use_fabric=True)
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

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_iters", type=int, default=2000)

    p.add_argument("--sequence_length", type=int, default=64)
    p.add_argument("--prop_hist_len", type=int, default=3)
    p.add_argument("--depth_hist_len", type=int, default=4)

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--save_dir", type=str, default="student_dagger_outputs")

    return p.parse_args()


# ============================================================
def main():
    args = parse_args()
    device = torch.device(args.device)

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
        prop_dim=num_prop,
        depth_shape=cam_res,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    # ===== main training loop =====
    for it in range(args.num_iters):
        start_t = time.time()
        batch = None

        obs, extras = vec_env.get_observations()
        obs = obs.to(device)

        while batch is None:
            # --- prepare numpy obs for aggregator ---
            obs_prop_np = obs[:, :num_prop].cpu().numpy()
            depth_np = extras["observations"]["depth_camera"].cpu().numpy()

            # --- teacher labels ---
            with torch.no_grad():
                teacher_actions = teacher_policy.act_inference(obs, hist_encoding=False, scandots_latent=None)
            teacher_actions_np = teacher_actions.cpu().numpy()

            # --- student acting ---
            # build student input from its own histories (aggregator stores them)
            prop_batch = []
            depth_batch = []
            for i in range(args.num_envs):
                prop_hist = list(aggregator.prop_histories[i])
                depth_hist = list(aggregator.depth_histories[i])
                prop_batch.append(np.concatenate(prop_hist, axis=0))
                depth_batch.append(np.stack(depth_hist, axis=0))

            prop_t = torch.from_numpy(np.stack(prop_batch)).float().to(device).unsqueeze(1)
            depth_t = torch.from_numpy(np.stack(depth_batch)).float().to(device).unsqueeze(1)

            student.eval()
            with torch.no_grad():
                pred_actions, _ = student(prop_t, depth_t, mems=None)
                student_act = pred_actions[:, -1, :].cpu()

            # --- env step (student acts) ---
            obs, rewards, dones, infos = vec_env.step(student_act.to(vec_env.device))
            obs = obs.to(device)

            if "observations" in infos:
                extras = infos

            dones_np = dones.cpu().numpy().astype(bool)

            # --- push to aggregator ---
            batch = aggregator.push_step(
                obs_prop=obs_prop_np,
                depth_frame=depth_np,
                teacher_actions=teacher_actions_np,
                done=dones_np,
            )

        # ====== train on batch ======
        student.train()
        proprio_t = torch.from_numpy(batch["proprio"]).float().to(device)
        depth_t = torch.from_numpy(batch["depth"]).float().to(device)
        teacher_t = torch.from_numpy(batch["actions"]).float().to(device)

        pred, _ = student(proprio_t, depth_t, mems=None)
        loss = nn.functional.mse_loss(pred, teacher_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()

        global_step += 1

        print(f"[iter {it}] loss={loss.item():.5f}   time={time.time()-start_t:.2f}s   global_step={global_step}")

        # save
        if (it + 1) % 100 == 0:
            ckpt = {
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iter": it,
                "global_step": global_step,
            }
            torch.save(ckpt, save_dir / f"student_dagger_{it+1:06d}.pt")
            print(f"[save] {save_dir / f'student_dagger_{it+1:06d}.pt'}")

    # final save
    torch.save({"model_state_dict": student.state_dict()}, save_dir / "student_dagger_final.pt")
    print("[done] training finished.")


if __name__ == "__main__":
    main()
