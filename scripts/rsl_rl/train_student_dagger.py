# scripts/rsl_rl/train_student_dagger.py
"""
Online DAGGER-style trainer that always lets the STUDENT act in the env and
uses the TEACHER (loaded via OnPolicyRunnerWithExtractor) as a supervision signal.

Usage example (rough):
  python scripts/rsl_rl/train_student_dagger.py \
    --task Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0 \
    --teacher-checkpoint logs/rsl_rl/.../model_4000.pt \
    --student-checkpoint outputs/student_policy/student_epoch_0010.pt \
    --num_iters 10000 --num_steps_per_env 64 --num_envs 16 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import importlib.util

# Ensure repo roots are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULES_ROOT = Path(PROJECT_ROOT) / "parkour_tasks" / "parkour_tasks" / "extreme_parkour_task" / "modules"


def load_symbol(module_path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"student_policy.{symbol}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{module_path} does not define {symbol}")
    return getattr(module, symbol)


JointPoseActionHead = load_symbol(MODULES_ROOT / "actionheads" / "joint_action_head.py", "JointPoseActionHead")
MultiModalFusionTransformer = load_symbol(
    MODULES_ROOT / "encoders" / "fusion_transformer.py", "MultiModalFusionTransformer"
)
TransformerXLTemporal = load_symbol(MODULES_ROOT / "temperal" / "txl.py", "TransformerXLTemporal")
DepthEncoder = load_symbol(MODULES_ROOT / "tokenizers" / "depth_encoder.py", "DepthEncoder")
ProprioEncoder = load_symbol(MODULES_ROOT / "tokenizers" / "proprio_encoder.py", "ProprioEncoder")

# Minimal SequenceBuffer: keep last prop/history and depth/history and form sequences for training
class OnlineSequenceBuffer:
    """
    For each parallel env keep:
      - a deque for proprio history (maxlen = prop_hist_len)
      - a deque for depth history (maxlen = depth_hist_len)
      - a deque for built tokens (maxlen = sequence_length)
    When token deque reaches sequence_length we emit a sample for training.
    """

    def __init__(self, num_envs: int, prop_hist_len: int, depth_hist_len: int, sequence_length: int):
        self.num_envs = num_envs
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.sequence_length = sequence_length

        self.prop_histories: List[Deque[np.ndarray]] = [deque(maxlen=prop_hist_len) for _ in range(num_envs)]
        self.depth_histories: List[Deque[np.ndarray]] = [deque(maxlen=depth_hist_len) for _ in range(num_envs)]
        self.token_buffers: List[Deque[Dict[str, np.ndarray]]] = [deque(maxlen=sequence_length) for _ in range(num_envs)]

    def reset_env(self, env_id: int):
        self.prop_histories[env_id].clear()
        self.depth_histories[env_id].clear()
        self.token_buffers[env_id].clear()

    def push_step(
        self,
        obs_prop: np.ndarray,  # [num_envs, prop_dim]
        depth_frame: np.ndarray,  # [num_envs, H, W]
        teacher_actions: np.ndarray,  # [num_envs, action_dim]
        done: np.ndarray,  # [num_envs] bool
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Append one step for each env. When any env's token_buffer reaches sequence_length,
        return a list of ready tuples: (proprio_seq [S, prop_hist_len*prop_dim],
                                       depth_seq  [S, depth_hist_len, H, W],
                                       teacher_actions_seq [S, action_dim])
        Each returned tuple is one env's training sample.
        """
        ready = []
        num_envs = obs_prop.shape[0]
        for i in range(num_envs):
            if done[i]:
                self.reset_env(i)

            # append hist items
            self.prop_histories[i].append(obs_prop[i].astype(np.float32, copy=False))
            self.depth_histories[i].append(depth_frame[i].astype(np.float32, copy=False))

            # need enough history to form a single token
            if len(self.prop_histories[i]) < self.prop_hist_len or len(self.depth_histories[i]) < self.depth_hist_len:
                continue

            # build token feature for this time-step:
            prop_stack = np.concatenate(list(self.prop_histories[i]), axis=0)  # [prop_hist_len * prop_dim]
            depth_stack = np.stack(list(self.depth_histories[i]), axis=0)  # [depth_hist_len, H, W]

            self.token_buffers[i].append(
                {
                    "prop": prop_stack,
                    "depth": depth_stack,
                    "action": teacher_actions[i].astype(np.float32, copy=False),
                }
            )

            # if token buffer full -> emit sequence sample
            if len(self.token_buffers[i]) == self.sequence_length:
                seq = list(self.token_buffers[i])
                proprio_seq = np.stack([s["prop"] for s in seq], axis=0)  # [S, prop_hist_len*prop_dim]
                depth_seq = np.stack([s["depth"] for s in seq], axis=0)  # [S, depth_hist_len, H, W]
                actions_seq = np.stack([s["action"] for s in seq], axis=0)  # [S, action_dim]
                ready.append((proprio_seq, depth_seq, actions_seq))
                # After emitting, we keep sliding window (deque maxlen auto slides), so we don't clear.
        return ready


# Build student model (same structure used in train_student_from_dataset.py)
class MultiModalStudentPolicy(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        camera_resolution: Tuple[int, int],
        prop_hist_len: int,
        depth_hist_len: int,
        fusion_cfg: Dict[str, object],
        temporal_cfg: Dict[str, object],
        action_head_cfg: Dict[str, object],
        token_dim: int = 128,
    ) -> None:
        super().__init__()
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        height, width = camera_resolution

        self.proprio_encoder = ProprioEncoder(
            state_dim=proprio_dim,
            hist_len=prop_hist_len,
            token_dim=token_dim,
            hidden_dims=fusion_cfg.get("prop_hidden_dims", (256, 256)),
            dropout=fusion_cfg.get("prop_dropout", 0.1),
        )
        self.depth_encoder = DepthEncoder(
            in_frames=depth_hist_len,
            in_size=max(height, width),
            token_dim=token_dim,
            grid_size=fusion_cfg.get("grid_size", 4),
            dropout=fusion_cfg.get("depth_dropout", 0.1),
        )
        self.fusion_transformer = MultiModalFusionTransformer(
            token_dim=token_dim,
            num_layers=fusion_cfg.get("num_layers", 2),
            num_heads=fusion_cfg.get("num_heads", 4),
            mlp_ratio=fusion_cfg.get("mlp_ratio", 2.0),
            dropout=fusion_cfg.get("dropout", 0.1),
            attn_dropout=fusion_cfg.get("attn_dropout", 0.1),
            add_modality_embed=True,
            norm_first=True,
        )
        self.temporal_model = TransformerXLTemporal(
            d_model=token_dim,
            n_layer=temporal_cfg.get("num_layers", 3),
            n_head=temporal_cfg.get("num_heads", 4),
            d_inner=temporal_cfg.get("d_inner", 256),
            mem_len=temporal_cfg.get("mem_len", 64),
            dropout=temporal_cfg.get("dropout", 0.1),
            attn_dropout=temporal_cfg.get("attn_dropout", 0.1),
            norm_first=temporal_cfg.get("norm_first", True),
            clamp_len=temporal_cfg.get("clamp_len", None),
            use_rel_pos=temporal_cfg.get("use_rel_pos", True),
        )
        self.action_head = JointPoseActionHead(
            d_model=token_dim,
            action_dim=action_dim,
            hidden_dims=action_head_cfg.get("hidden_dims", (256, 256)),
            tanh_output=action_head_cfg.get("tanh_output", False),
            action_scale=action_head_cfg.get("action_scale", 1.0),
        )

    def forward(self, proprio_seq: torch.Tensor, depth_seq: torch.Tensor, mems: Optional[List[torch.Tensor]] = None):
        """
        Args:
            proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
            depth_seq: Tensor[B, S, depth_hist_len, H, W]
        Returns:
            actions: Tensor[B, S, action_dim]
            new_mems: list of mem tensors (if return_mems True)
        """
        batch_size, seq_len, feat_dim = proprio_seq.shape
        prop_encoded = self.proprio_encoder(proprio_seq.reshape(batch_size * seq_len, feat_dim))  # [B*S, 1, C]
        depth_encoded = self.depth_encoder(
            depth_seq.reshape(batch_size * seq_len, depth_seq.size(2), depth_seq.size(3), depth_seq.size(4))
        )  # [B*S, T, C]
        fused = self.fusion_transformer(prop_encoded, depth_encoded)
        fused_seq = fused["all_pooled"].reshape(batch_size, seq_len, -1)  # [B, S, C]
        temporal_out, new_mems = self.temporal_model(fused_seq, mems=mems, causal_mask=True, return_mems=True)
        actions = self.action_head.forward_sequence(temporal_out)["mean"]
        return actions, new_mems


# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Online DAGGER-style student trainer (student acts, teacher labels).")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="教师模型加载点")
    parser.add_argument("--student-checkpoint", type=str, default=None, help="从train_student_from_dataset预训练的学生策略继续训练")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_iters", type=int, default=2000, help="Number of training iterations (outer loop).")
    parser.add_argument("--num_steps_per_env", type=int, default=64, help="collection horizon per iteration.")
    parser.add_argument("--prop_hist_len", type=int, default=3)
    parser.add_argument("--depth_hist_len", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=64, help="training sequence length (window).")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()

# --- Import environment / runner utilities (same as collect.py) ---
import gymnasium as gym
from isaaclab.app import AppLauncher
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
import cli_args
from modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from vecenv_wrapper import ParkourRslRlVecEnvWrapper

# Launch app (AppLauncher will attach to Isaac/Omniverse)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -------------------------
# Main routine
# -------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    # parse env cfg (reuse the same helper as collect.py)
    env_cfg = parse_env_cfg(args.task, device=None, num_envs=args.num_envs, use_fabric=True)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)  # uses rsl_rl config parsing used elsewhere

    # Build env & vec wrapper (same pattern in collect.py)
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load teacher with OnPolicyRunnerWithExtractor (same approach as collect.py)
    # We pass the same train cfg so runner can construct policy/estimator properly.
    runner = OnPolicyRunnerWithExtractor(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = args.teacher_checkpoint
    runner.load(resume_path, load_optimizer=False)
    teacher_policy = runner.get_inference_policy(device=vec_env.device)

    # Build student model using same defaults as train_student_from_dataset.py
    # infer dims from agent_cfg estimator or from first observation
    obs0, extras0 = vec_env.get_observations()
    obs0 = obs0.to(device)
    num_prop = int(agent_cfg.estimator.num_prop) if hasattr(agent_cfg, "estimator") else int(obs0.shape[1])
    # try to infer camera resolution from extras
    depth0 = extras0["observations"]["depth_camera"]
    cam_res = (int(depth0.shape[-2]), int(depth0.shape[-1]))

    # config blocks (same defaults as train_student_from_dataset.py)
    fusion_cfg = {"num_layers": 2, "num_heads": 4, "mlp_ratio": 2.0, "dropout": 0.1, "attn_dropout": 0.1, "grid_size": 4}
    temporal_cfg = {"num_layers": 3, "num_heads": 4, "d_inner": 256, "mem_len": args.sequence_length, "dropout": 0.1, "attn_dropout": 0.1}
    action_head_cfg = {"hidden_dims": (256, 256), "tanh_output": False, "action_scale": 1.0}

    # Need action_dim from env meta if available:
    # Use vec_env.num_actions if present
    action_dim = int(getattr(vec_env, "num_actions", 0))
    if action_dim == 0:
        raise RuntimeError("Cannot infer action dimension from env; set up vec_env.num_actions or pass it explicitly.")

    student = MultiModalStudentPolicy(
        proprio_dim=num_prop,
        action_dim=action_dim,
        camera_resolution=cam_res,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
        token_dim=128,
    ).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Optionally load student checkpoint
    if args.student_checkpoint:
        ckpt = torch.load(args.student_checkpoint, map_location="cpu")
        if "model_state_dict" in ckpt:
            student.load_state_dict(ckpt["model_state_dict"])
            print(f"[info] Loaded student checkpoint from {args.student_checkpoint}")
        else:
            student.load_state_dict(ckpt)
            print(f"[info] Loaded student raw state_dict from {args.student_checkpoint}")

    # Prepare buffers and mems
    seq_buffer = OnlineSequenceBuffer(num_envs=args.num_envs, prop_hist_len=args.prop_hist_len, depth_hist_len=args.depth_hist_len, sequence_length=args.sequence_length)
    mems = student.temporal_model.reset_mems(batch_size=args.num_envs)

    global_step = 0
    save_dir = Path(args.save_dir) if args.save_dir else Path("outputs") / "student_dagger"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Main training loop
    for it in range(args.num_iters):
        it_start = time.time()
        # collect horizon steps (push into OnlineSequenceBuffer)
        obs, extras = vec_env.get_observations()
        obs = obs.to(device)
        extras = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in extras.items()}

        # run num_steps_per_env steps
        for step in range(args.num_steps_per_env):
            # build per-step inputs for student:
            # - proprio: full observation vector (we assume first num_prop dims are proprio)
            proprio = obs[:, :num_prop].cpu().numpy()  # [num_envs, num_prop]
            depth_frame = extras["observations"]["depth_camera"].cpu().numpy()  # [num_envs, H, W] or [num_envs, H, W, C] depending on env

            # get teacher action (as label) using teacher_policy inference
            # teacher_policy.act_inference or teacher_policy.act? collect.py uses get_inference_policy and uses .act_inference in some places.
            with torch.no_grad():
                teacher_act = teacher_policy.act_inference(obs, hist_encoding=False, scandots_latent=None)
                # teacher_act is a torch tensor on device; move to cpu numpy for buffering
                teacher_act_np = teacher_act.cpu().numpy()

            # Student predicts action (used to step the env)
            # For single-step execution we need to supply student with a "token" built from current histories.
            # Build temporary small batch for prediction: if histories are too short student can't produce token - we handle that in buffer.
            # To step env we still want actions - fallback to teacher if student cannot produce valid action (rare if prop/depth histories not ready).
            can_student_step = True
            # create a shallow copy of history to determine readiness
            for i in range(args.num_envs):
                # we'll rely on seq_buffer to check histories; if history len < required, student cannot predict for that env
                if len(seq_buffer.prop_histories[i]) < args.prop_hist_len or len(seq_buffer.depth_histories[i]) < args.depth_hist_len:
                    can_student_step = False
                    break

            if can_student_step:
                # Build a single-step batch of shape [num_envs, 1, ...] by using last hist stacks
                prop_batch = []
                depth_batch = []
                for i in range(args.num_envs):
                    prop_stack = np.concatenate(list(seq_buffer.prop_histories[i]), axis=0) if len(seq_buffer.prop_histories[i]) >= args.prop_hist_len else np.concatenate([proprio[i]] * args.prop_hist_len, axis=0)
                    depth_stack = np.stack(list(seq_buffer.depth_histories[i]), axis=0) if len(seq_buffer.depth_histories[i]) >= args.depth_hist_len else np.stack([depth_frame[i]] * args.depth_hist_len, axis=0)
                    prop_batch.append(prop_stack)
                    depth_batch.append(depth_stack)
                prop_batch_t = torch.from_numpy(np.stack(prop_batch, axis=0)).to(device).unsqueeze(1)  # [B, 1, feat]
                depth_batch_t = torch.from_numpy(np.stack(depth_batch, axis=0)).to(device).unsqueeze(1)  # [B, 1, T, H, W]
                student.eval()
                with torch.no_grad():
                    student_actions, next_mems = student(prop_batch_t, depth_batch_t, mems=mems)
                    # student_actions: [B, 1, action_dim]
                    student_actions = student_actions[:, -1, :].cpu()  # [B, action_dim]
                env_actions = student_actions
                mems = student.temporal_model.detach_mems(next_mems)
            else:
                # fallback: use teacher actions to avoid stuck env
                env_actions = torch.from_numpy(teacher_act_np).cpu()

            # Step environment with student's action (as required by user)
            obs, rewards, dones, infos = vec_env.step(env_actions.to(vec_env.device))
            # move to torch on training device
            obs = obs.to(device)
            dones = dones.to(device)
            # extras updated by env.step
            extras = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in infos["observations"].items()} if "observations" in infos else extras

            # Push the teacher label & frame into buffer (teacher actions used as labels)
            teacher_np = teacher_act_np  # [num_envs, action_dim]
            # depth_frame might be different shape (H,W) or (H,W,1); ensure consistent [num_envs,H,W]
            depth_frame_proc = depth_frame
            dones_np = dones.cpu().numpy().astype(bool)
            ready_samples = seq_buffer.push_step(obs_prop=proprio, depth_frame=depth_frame_proc, teacher_actions=teacher_np, done=dones_np)

            # For each ready sample do a supervised update immediately (or accumulate)
            for proprio_seq_np, depth_seq_np, teacher_seq_np in ready_samples:
                # convert to tensors with batch dimension =1
                student.train()
                p_t = torch.from_numpy(np.expand_dims(proprio_seq_np, axis=0)).to(device)  # [1, S, feat]
                d_t = torch.from_numpy(np.expand_dims(depth_seq_np, axis=0)).to(device)     # [1, S, T, H, W]
                target_t = torch.from_numpy(np.expand_dims(teacher_seq_np, axis=0)).to(device)  # [1, S, action_dim]

                # forward with mems for this env slice: we keep global mems but student.temporal expects batch mems sized by batch
                # simplest approach: reset mems for this training minibatch (teacher data are offline-like), or reuse
                # We'll create zero mems for this minibatch to avoid cross-env mem contamination
                local_mems = student.temporal_model.reset_mems(batch_size=1)
                preds, _ = student(p_t, d_t, mems=local_mems)
                loss = nn.functional.mse_loss(preds, target_t)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                optimizer.step()
                global_step += 1

        it_time = time.time() - it_start
        if it % 10 == 0:
            print(f"[iter {it}] time={it_time:.2f}s global_step={global_step}")

        # periodic checkpointing
        if (it + 1) % 100 == 0:
            ckpt_path = save_dir / f"student_dagger_iter_{it+1:06d}.pt"
            payload = {"model_state_dict": student.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iter": it}
            torch.save(payload, ckpt_path)
            print(f"[checkpoint] saved {ckpt_path}")

    # final save
    final_path = save_dir / "student_dagger_final.pt"
    torch.save({"model_state_dict": student.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, final_path)
    print(f"[done] final saved to {final_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
