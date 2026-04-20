"""
Online PPO finetuning of the TransformerXL student policy.

Training regime (per iteration)
--------------------------------
  1. Capture the initial TXL memory state (start_mems).
  2. Collect `sequence_length` env steps, directly fetching history from the Runner.
  3. Bootstrap value estimate at the end of the segment.
  4. Compute GAE advantages + returns.
  5. PPO multi-epoch updates: re-feed the sequence using the SAME `start_mems` clone.
  6. TBPTT: detach the TXL mems from the LAST epoch, clear done-env mems.
  7. Sync the cleaned mems back to the Runner -> next iteration.

Architecture
------------
  Actor : MultiModalStudentPolicyRL  (subclass adding forward_with_mems_rl)
  Critic: StudentCritic              (MLP on privileged sim obs, never deployed)
          → optionally warm-started from teacher checkpoint
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

RSL_RL_DIR = os.path.join(PROJECT_ROOT, "scripts", "rsl_rl")
if RSL_RL_DIR not in sys.path:
    sys.path.insert(0, RSL_RL_DIR)

# ── local imports ─────────────────
from transformerxl.temporal.txl import TransformerXLTemporal
from utils.student_utils import (
    StudentOnlineRunner,
    build_student_model,
    load_env_and_teacher
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warning] wandb not installed, run `pip install wandb` to enable logging.")

# ══════════════════════════════════════════════════════════════════════════════
#  Critic: privileged MLP (training-only, never deployed)
# ══════════════════════════════════════════════════════════════════════════════

class StudentCritic(nn.Module):
    """
    Simple MLP value network conditioned on privileged simulator observations.
    Matches the hidden-dim layout of the teacher critic (512 → 256 → 128).
    """

    def __init__(
        self,
        privileged_obs_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = privileged_obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: Tensor) -> Tensor:
        """obs: [..., D]  →  value: [..., 1]"""
        return self.net(obs)

# ══════════════════════════════════════════════════════════════════════════════
#  Unified PPO Buffer
# ══════════════════════════════════════════════════════════════════════════════

class PPOBuffer:
    """Unified buffer for Rollout data."""
    def __init__(self, num_envs, seq_len, prop_flat_dim, depth_shape, action_dim, priv_obs_dim, device):
        self.seq_len = seq_len
        self.num_envs = num_envs
        self.step = 0

        # RL Scalars
        self.rewards = torch.zeros(num_envs, seq_len, device=device)
        self.dones = torch.zeros(num_envs, seq_len, dtype=torch.bool, device=device)
        self.log_probs = torch.zeros(num_envs, seq_len, device=device)
        self.values = torch.zeros(num_envs, seq_len, device=device)

        # Observations and Actions
        self.proprio = torch.zeros(num_envs, seq_len, prop_flat_dim, device=device)
        self.depth = torch.zeros(num_envs, seq_len, *depth_shape, device=device)
        self.actions = torch.zeros(num_envs, seq_len, action_dim, device=device)
        self.extra_info = torch.zeros(num_envs, seq_len, 2, device=device)
        self.priv_obs = torch.zeros(num_envs, seq_len, priv_obs_dim, device=device)

    def reset(self):
        self.step = 0
        self.rewards.zero_()
        self.dones.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.proprio.zero_()
        self.depth.zero_()
        self.actions.zero_()
        self.extra_info.zero_()
        self.priv_obs.zero_()

    def push(self, prop_hist, depth_hist, action, reward, done, log_prob, value, priv_ob, extra_info):
        i = self.step
        self.proprio[:, i] = prop_hist.detach()
        self.depth[:, i] = depth_hist.detach()
        self.actions[:, i] = action.detach()
        self.rewards[:, i] = reward.detach()
        self.dones[:, i] = done.detach()
        self.log_probs[:, i] = log_prob.detach()
        self.values[:, i] = value.squeeze(-1).detach()
        self.priv_obs[:, i] = priv_ob.detach()
        self.extra_info[:, i] = extra_info.detach()

        self.step += 1
        return self.step == self.seq_len


# ══════════════════════════════════════════════════════════════════════════════
#  GAE
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae(
    rewards: Tensor,   # [N, S]
    values: Tensor,    # [N, S]
    dones: Tensor,     # [N, S]  bool or float
    last_value: Tensor,# [N]     bootstrap value V(s_{S+1})
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """Returns advantages [N, S] and returns [N, S] (= advantages + values)."""
    N, S = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)
    not_done = 1.0 - dones.float()

    for t in reversed(range(S)):
        next_val = last_value if t == S - 1 else values[:, t + 1]
        delta = rewards[:, t] + gamma * next_val * not_done[:, t] - values[:, t]
        last_gae = delta + gamma * lam * not_done[:, t] * last_gae
        advantages[:, t] = last_gae

    return advantages, advantages + values


# ══════════════════════════════════════════════════════════════════════════════
#  Teacher-critic warm-start helper
# ══════════════════════════════════════════════════════════════════════════════

def try_load_teacher_critic(
    critic: StudentCritic,
    teacher_ckpt_path: str,
) -> bool:
    try:
        ckpt = torch.load(teacher_ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", {})
        for prefix in ("v.", "critic.", "value_net."):
            sub = {k[len(prefix):]: v
                   for k, v in state.items() if k.startswith(prefix)}
            if not sub:
                continue
            try:
                critic.net.load_state_dict(sub, strict=True)
                print(f"[critic] Warm-started from teacher checkpoint (prefix='{prefix}').")
                return True
            except RuntimeError:
                continue
        print("[critic] No matching critic weights found – training from scratch.")
        return False
    except Exception as exc:
        print(f"[critic] Load failed ({exc}) – training from scratch.")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path: Path,
    actor: nn.Module,
    critic: nn.Module,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    iteration: int,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer": actor_opt.state_dict(),
        "critic_optimizer": critic_opt.state_dict(),
        "iteration": iteration,
        "meta": meta,
    }, path)
    print(f"[ckpt] Saved → {path}")


def load_checkpoint(
    path: Path,
    actor: nn.Module,
    critic: nn.Module,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    critic.load_state_dict(ckpt["critic_state_dict"])
    actor_opt.load_state_dict(ckpt["actor_optimizer"])
    critic_opt.load_state_dict(ckpt["critic_optimizer"])
    it = int(ckpt.get("iteration", 0))
    print(f"[ckpt] Resumed from {path} (iteration={it})")
    return it


def load_actor_only(
    path: Path,
    actor: nn.Module,
    device: torch.device,
) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    key = "actor_state_dict" if "actor_state_dict" in ckpt else "model_state_dict"
    actor.load_state_dict(ckpt[key])
    print(f"[ckpt] Actor weights loaded from {path} (key='{key}').")


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Online PPO RL finetuning of TransformerXL student")

    # ── environment ────────────────────────────────────────────────────────────
    p.add_argument("--task",               type=str, required=True)
    p.add_argument("--teacher_checkpoint", type=str, required=True)
    p.add_argument("--student_checkpoint", type=str, default=None)
    p.add_argument("--student_is_dagger",  action="store_true")
    p.add_argument("--num_envs",           type=int, default=64)

    # ── training schedule ──────────────────────────────────────────────────────
    p.add_argument("--num_iters",          type=int, default=10_000)

    # ── PPO ────────────────────────────────────────────────────────────────────
    p.add_argument("--sequence_length",   type=int,   default=64)
    p.add_argument("--ppo_epochs",        type=int,   default=4,
                   help="Number of epochs to train on the rollout data.")
    p.add_argument("--clip_param",        type=float, default=0.2)
    p.add_argument("--entropy_coef",      type=float, default=0.005)
    p.add_argument("--value_loss_coef",   type=float, default=0.5)
    p.add_argument("--yaw_loss_coef",     type=float, default=1.0)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--lam",               type=float, default=0.95)
    p.add_argument("--normalize_adv",     action="store_true", default=True)

    # ── optimiser ──────────────────────────────────────────────────────────────
    p.add_argument("--actor_lr",    type=float, default=1e-4)
    p.add_argument("--critic_lr",   type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--grad_clip",   type=float, default=1.0)

    # ── critic warm-start / freezing ───────────────────────────────────────────
    p.add_argument("--load_teacher_critic", action="store_true")
    p.add_argument("--freeze_critic_iters", type=int, default=0)

    # ── student architecture ───────────────────────────────────────────────────
    p.add_argument("--prop_hist_len",  type=int, default=3)
    p.add_argument("--depth_hist_len", type=int, default=4)

    # ── augmentation ───────────────────────────────────────────────────────────
    p.add_argument("--use_dropout", action="store_true")

    # ── logging / saving ───────────────────────────────────────────────────────
    p.add_argument("--save_dir",      type=str, default="outputs/students/rl")
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--log_interval",  type=int, default=10)
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_project", type=str, default="student-rl")
    p.add_argument("--wandb_run_name",type=str, default=None)
    p.add_argument("--wandb_entity",  type=str, default=None)

    import cli_args
    from isaaclab.app import AppLauncher
    cli_args.add_rsl_rl_args(p)
    AppLauncher.add_app_launcher_args(p)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():  # noqa: C901
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb ──────────────────────────────────────────────────────────────────
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   entity=args.wandb_entity, config=vars(args))

    # ── environment + teacher ──────────────────────────────────────────────────
    vec_env, teacher_policy, agent_cfg, simulation_app = load_env_and_teacher(args)
    obs, extras = vec_env.get_observations()

    depth_sample = extras["observations"]["depth_camera"]
    camera_resolution = (int(depth_sample.shape[-2]), int(depth_sample.shape[-1]))
    proprio_dim = int(agent_cfg.estimator.num_prop)
    priv_obs_dim = int(obs.shape[1])
    action_dim = (vec_env.unwrapped.action_space.shape[1]
                         if hasattr(vec_env.unwrapped.action_space, "shape")
                         else int(obs.shape[1]))

    print(f"[Info] proprio={proprio_dim}  priv_obs={priv_obs_dim}  "
          f"action={action_dim}  cam={camera_resolution}")

    # ── build actor & critic ───────────────────────────────────────────────────
    actor = build_student_model(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        mem_len=args.sequence_length,
        token_dim=128
    ).to(device)

    critic = StudentCritic(
        privileged_obs_dim=priv_obs_dim,
        hidden_dims=(512, 256, 128),
    ).to(device)

    if args.load_teacher_critic:
        try_load_teacher_critic(critic, args.teacher_checkpoint)

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay)

    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        actor_opt, T_max=args.num_iters, eta_min=args.actor_lr * 0.1
    )

    # ── checkpoint loading ─────────────────────────────────────────────────────
    start_iter = 0
    if args.student_checkpoint:
        ckpt_path = Path(args.student_checkpoint)
        if args.student_is_dagger:
            load_actor_only(ckpt_path, actor, device)
        else:
            start_iter = load_checkpoint(ckpt_path, actor, critic, actor_opt, critic_opt, device)

    # ── unified buffer and runner ──────────────────────────────────────────────
    ppo_buffer = PPOBuffer(
        num_envs=args.num_envs,
        seq_len=args.sequence_length,
        prop_flat_dim=proprio_dim * args.prop_hist_len,
        depth_shape=(args.depth_hist_len, camera_resolution[0], camera_resolution[1]),
        action_dim=action_dim,
        priv_obs_dim=priv_obs_dim,
        device=device
    )

    student_runner = StudentOnlineRunner(
        model=actor,
        num_envs=args.num_envs,
        proprio_dim=proprio_dim,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        camera_resolution=camera_resolution,
        device=device,
    )
    student_runner.reset()

    # ── camera dropout ─────────────────────────────────────────────────────────
    from utils.dropout_manager import CameraDropoutManager
    dropout_manager = None
    if args.use_dropout:
        print("[Info] CameraDropoutManager enabled.")
        dropout_manager = CameraDropoutManager(
            num_envs=args.num_envs,
            device=device,
            dt=vec_env.unwrapped.step_dt,
            prob_start_offline=0.0,
            prob_cam_offline=0.5,
            online_duration_range=(2.0, 10.0),
            offline_duration_range=(1.0, 5.0),
        )

    # ── episode-level tracking ─────────────────────────────────────────────────
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    track_progress_buf = deque(maxlen=100)
    current_returns = torch.zeros(args.num_envs, device=device)
    current_lengths = torch.zeros(args.num_envs, device=device)
    dones_bool = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    base_parkour = vec_env.unwrapped.parkour_manager.get_term("base_parkour")
    num_goals = int(getattr(base_parkour, "num_goals", 0))

    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

    print(f"[RL] Starting training  iters={args.num_iters}  "
          f"seq_len={args.sequence_length}  num_envs={args.num_envs}  "
          f"ppo_epochs={args.ppo_epochs}")

    # ══════════════════════════════════════════════════════════════════════════
    train_mems: Optional[List[Tensor]] = None
    for it in range(start_iter, args.num_iters):
        iter_start = time.time()

        actor.eval()
        critic.eval()
        ppo_buffer.reset()

        # ────────────────────────────────────────────────────────────────────
        #  (A)  Rollout collection – sequence_length steps
        # ────────────────────────────────────────────────────────────────────
        while simulation_app.is_running():
            depth_image = extras["observations"]["depth_camera"]
            if depth_image.dim() == 4:
                depth_image = depth_image.squeeze(1)

            student_prop = obs[:, :proprio_dim].clone()
            _, _, yaw = euler_xyz_from_quat(base_parkour.robot.data.root_quat_w)
            student_prop[:, 6] = -wrap_to_pi(yaw)
            student_prop[:, 7] = 0
            extra_info = torch.cat([
                base_parkour.target_yaw.clone().unsqueeze(-1),
                base_parkour.next_target_yaw.clone().unsqueeze(-1),
            ], dim=-1)
            student_prop[:, 12] = dones_bool.float()
            student_depth = depth_image.clone()

            if dropout_manager is not None:
                dropout_manager.reset_env(dones_bool)
                dropout_manager.update(depth_image=student_depth, obs_prop=student_prop)

            priv_ob = obs.clone()

            # stochastic action from student policy
            action, log_prob, _entropy = student_runner.act_rl(student_prop, student_depth)

            with torch.no_grad():
                value = critic(priv_ob).squeeze(-1)

            goal_before = base_parkour.cur_goal_idx.detach().cpu().numpy().copy() if num_goals > 0 else None

            # step environment
            obs, rewards, dones, extras = vec_env.step(action)
            dones_bool = dones.squeeze(-1).bool()
            rewards_f = rewards.squeeze(-1)

            # Direct history fetch from runner to unified buffer
            is_full = ppo_buffer.push(
                prop_hist=student_runner.prop_histories.view(args.num_envs, -1),
                depth_hist=student_runner.depth_histories,
                action=action,
                reward=rewards_f,
                done=dones_bool,
                log_prob=log_prob,
                value=value,
                priv_ob=priv_ob,
                extra_info=extra_info
            )

            # Episode statistics
            current_returns += rewards_f
            current_lengths += 1
            if dones_bool.any():
                student_runner.reset_done(dones_bool)
                for idx in torch.nonzero(dones_bool).squeeze(-1):
                    ep_returns.append(current_returns[idx].item())
                    ep_lengths.append(current_lengths[idx].item())
                    if goal_before is not None:
                        prog = float(np.clip(goal_before[idx.item()] / num_goals, 0.0, 1.0))
                        track_progress_buf.append(prog)
                    current_returns[idx] = 0.0
                    current_lengths[idx] = 0.0

            if is_full:
                break

        if not simulation_app.is_running():
            break

        # ────────────────────────────────────────────────────────────────────
        #  (B)  Bootstrap value and compute GAE
        # ────────────────────────────────────────────────────────────────────
        with torch.no_grad():
            last_value = critic(obs.to(device)).squeeze(-1)

        advantages, returns = compute_gae(
            ppo_buffer.rewards, ppo_buffer.values, ppo_buffer.dones, last_value,
            gamma=args.gamma, lam=args.lam,
        )
        if args.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ────────────────────────────────────────────────────────────────────
        #  (C)  PPO Multi-Epoch Update
        # ────────────────────────────────────────────────────────────────────
        actor.train()
        critic.train()

        critic_frozen = (it < args.freeze_critic_iters)
        for p in critic.parameters():
            p.requires_grad_(not critic_frozen)

        N, S = advantages.shape
        final_epoch_mems = None

        for epoch in range(args.ppo_epochs):
            # Re-evaluate log-probs under current policy
            _, new_log_probs, new_entropy, pred_yaws, epoch_out_mems = actor.forward_with_mems_rl(
                ppo_buffer.proprio,
                ppo_buffer.depth,
                old_actions=ppo_buffer.actions,
                mems=train_mems
            )

            new_values = critic(ppo_buffer.priv_obs.reshape(N * S, -1)).reshape(N, S)

            # PPO surrogate
            log_ratio = new_log_probs - ppo_buffer.log_probs
            ratio = torch.exp(log_ratio)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1.0 - args.clip_param, 1.0 + args.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            val_clipped = ppo_buffer.values + (new_values - ppo_buffer.values).clamp(
                -args.clip_param, args.clip_param
            )
            value_loss = torch.max(
                F.mse_loss(new_values, returns),
                F.mse_loss(val_clipped, returns),
            )

            # Auxiliary yaw loss
            yaw_loss = F.mse_loss(pred_yaws, ppo_buffer.extra_info)

            # Total loss
            total_loss = (policy_loss
                          + args.value_loss_coef * value_loss
                          - args.entropy_coef * new_entropy.mean()
                          + args.yaw_loss_coef * yaw_loss)

            actor_opt.zero_grad()
            critic_opt.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
            actor_opt.step()

            if not critic_frozen:
                nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                critic_opt.step()

            # Store the resulting mems from the very last epoch to pass back to the runner
            final_epoch_mems = epoch_out_mems

        # actor_scheduler.step()

        # ────────────────────────────────────────────────────────────────────
        #  (D)  Sync clean mems back to the Runner
        # ────────────────────────────────────────────────────────────────────
        # Process the newly computed mems from the last epoch to mask out done states,
        # ensuring no state drift and perfectly continuous TBPTT.
        if final_epoch_mems is not None:
            train_mems = TransformerXLTemporal.detach_mems(final_epoch_mems)
            for b in range(N):
                done_indices = torch.nonzero(ppo_buffer.dones[b])
                if done_indices.numel() > 0:
                    last_done = done_indices.max().item()
                    for layer_mem in train_mems:
                        layer_mem[b, :last_done + 1, :] = 0.0

            # 强制同步 Runner 的隐状态
            student_runner.mems = train_mems
        else:
            student_runner.mems = None

        # ────────────────────────────────────────────────────────────────────
        #  (E)  Logging
        # ────────────────────────────────────────────────────────────────────
        iter_time = time.time() - iter_start

        if (it + 1) % args.log_interval == 0:
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > args.clip_param).float().mean().item()

            print(
                f"[Iter {it+1:5d}]  "
                f"policy={policy_loss.item():.4f}  "
                f"value={value_loss.item():.4f}  "
                f"entropy={new_entropy.mean().item():.4f}  "
                f"kl={approx_kl:.4f}  "
                f"clip={clip_frac:.3f}  "
                f"lr={actor_scheduler.get_last_lr()[0]:.2e}  "
                f"time={iter_time:.2f}s"
            )

            if use_wandb:
                log_dict: Dict = {
                    "ppo/policy_loss": policy_loss.item(),
                    "ppo/value_loss": value_loss.item(),
                    "ppo/entropy": new_entropy.mean().item(),
                    "ppo/yaw_loss": yaw_loss.item(),
                    "ppo/total_loss": total_loss.item(),
                    "ppo/approx_kl": approx_kl,
                    "ppo/clip_frac": clip_frac,
                    "params/actor_lr": actor_scheduler.get_last_lr()[0],
                    "params/critic_frozen": int(critic_frozen),
                    "rollout/ep_return_mean": np.mean(ep_returns) if ep_returns else 0.0,
                    "rollout/ep_len_mean": np.mean(ep_lengths) if ep_lengths else 0.0,
                    "perf/iter_time_s": iter_time,
                }
                if track_progress_buf:
                    arr = np.array(track_progress_buf)
                    log_dict["rollout/track_progress_mean"] = float(np.mean(arr))
                    log_dict["rollout/track_progress_late_ratio"] = float(np.mean(arr >= 0.7))
                if base_parkour is not None:
                    log_dict["rollout/terrain_level_mean"] = float(
                        base_parkour.terrain.terrain_levels.float().mean().item()
                    )
                wandb.log(log_dict, step=it)

        # ────────────────────────────────────────────────────────────────────
        #  (F)  Checkpointing
        # ────────────────────────────────────────────────────────────────────
        if (it + 1) % args.save_interval == 0:
            save_checkpoint(
                save_dir / f"student_rl_{it + 1:05d}.pt",
                actor, critic, actor_opt, critic_opt,
                it + 1,
                meta={
                    "task": args.task,
                    "num_prop": proprio_dim,
                    "action_dim": action_dim,
                    "camera_resolution": camera_resolution,
                    "is_rl": True,
                },
            )

    # ── cleanup ────────────────────────────────────────────────────────────────
    vec_env.close()
    simulation_app.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
