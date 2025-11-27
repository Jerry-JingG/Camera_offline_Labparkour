from __future__ import annotations

from collections import deque
import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util
import numpy as np

import torch

from isaaclab.app import AppLauncher

# Ensure project roots are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

import cli_args  # isort: skip

# Load MultiModalStudentPolicy directly from the training script path to avoid name collisions with ROS packages.
TRAIN_STUDENT_PATH = Path(__file__).resolve().parent / "train_student_from_dataset.py"
_spec = importlib.util.spec_from_file_location("train_student_from_dataset", TRAIN_STUDENT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load MultiModalStudentPolicy from {TRAIN_STUDENT_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
MultiModalStudentPolicy = _module.MultiModalStudentPolicy


def find_latest_student_checkpoint(ckpt_dir: Path) -> Path:
    """Return the checkpoint with the highest epoch index in ckpt_dir."""
    candidates = list(ckpt_dir.glob("student_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No student checkpoints found in {ckpt_dir}")

    latest_epoch = -1
    latest_path: Path | None = None
    pattern = re.compile(r"student_epoch_(\d+)\.pt")
    for path in candidates:
        match = pattern.match(path.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = path
    if latest_path is None:
        raise FileNotFoundError(f"No student checkpoints matched pattern student_epoch_*.pt in {ckpt_dir}")
    return latest_path


def load_student_policy_for_play(
    checkpoint_path: Path,
    prop_hist_len: int,
    depth_hist_len: int,
    device: torch.device,
) -> Tuple[MultiModalStudentPolicy, Dict[str, object]]:
    """Load a trained student policy and associated metadata for playback."""
    payload = torch.load(checkpoint_path, map_location=device)
    meta: Dict[str, object] = dict(payload.get("meta", {}))

    proprio_dim = int(meta["num_prop"])
    action_dim = int(meta["action_dim"])
    camera_resolution = tuple(meta.get("camera_resolution", [64, 64]))

    fusion_cfg = {
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.1,
        "attn_dropout": 0.1,
        "grid_size": 4,
    }
    temporal_cfg = {
        "num_layers": 3,
        "num_heads": 4,
        "d_inner": 256,
        "mem_len": 64,
        "dropout": 0.1,
        "attn_dropout": 0.1,
    }
    action_head_cfg = {
        "hidden_dims": (256, 256),
        "tanh_output": False,
        "action_scale": 1.0,
    }

    model = MultiModalStudentPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,  # type: ignore[arg-type]
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
        token_dim=128,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta


class StudentOnlineRunner:
    """
    Maintains two-level histories and runs the student policy with a temporal window.

    For each env, short histories gather prop/dep frames to form per-step tokens, and
    longer sequence deques of length `sequence_length` buffer those tokens so the model
    sees [S, ...] sequences (matching offline training). Only the last step from the
    forward pass is used as the action.
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
        sequence_length: int = 16,
    ) -> None:
        self.model = model
        self.num_envs = num_envs
        self.proprio_dim = proprio_dim
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.camera_resolution = camera_resolution
        self.device = device
        self.sequence_length = sequence_length

        self.prop_histories: List[deque] = [deque(maxlen=prop_hist_len) for _ in range(num_envs)]
        self.depth_histories: List[deque] = [deque(maxlen=depth_hist_len) for _ in range(num_envs)]
        self.seq_prop_histories: List[deque] = [deque(maxlen=sequence_length) for _ in range(num_envs)]
        self.seq_depth_histories: List[deque] = [deque(maxlen=sequence_length) for _ in range(num_envs)]

    def reset(self) -> None:
        """Clear histories for all environments."""
        for hist in self.prop_histories:
            hist.clear()
        for hist in self.depth_histories:
            hist.clear()
        for hist in self.seq_prop_histories:
            hist.clear()
        for hist in self.seq_depth_histories:
            hist.clear()

    def reset_done(self, done_mask: torch.Tensor) -> None:
        """Clear histories for environments where done_mask is True."""
        for env_id, done in enumerate(done_mask):
            if bool(done):
                self.prop_histories[env_id].clear()
                self.depth_histories[env_id].clear()
                self.seq_prop_histories[env_id].clear()
                self.seq_depth_histories[env_id].clear()

    def num_ready_envs(self) -> int:
        """Return number of envs with full history buffers."""
        ready = 0
        for prop_seq_hist, depth_seq_hist in zip(self.seq_prop_histories, self.seq_depth_histories):
            if len(prop_seq_hist) == self.sequence_length and len(depth_seq_hist) == self.sequence_length:
                ready += 1
        return ready

    def act(
        self,
        obs_prop: torch.Tensor,  # [num_envs, proprio_dim]
        depth_image: torch.Tensor,  # [num_envs, H, W] or [num_envs, 1, H, W]
    ) -> torch.Tensor:
        """
        Update histories and compute student actions for all envs.
        Returns a tensor of shape [num_envs, action_dim].
        """
        obs_prop = obs_prop.to(self.device)
        depth_image = depth_image.to(self.device)

        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)

        # Append current step to histories
        for env_id in range(self.num_envs):
            self.prop_histories[env_id].append(obs_prop[env_id])
            self.depth_histories[env_id].append(depth_image[env_id])

        ready_indices: List[int] = []
        for env_id in range(self.num_envs):
            if len(self.prop_histories[env_id]) == self.prop_hist_len and len(self.depth_histories[env_id]) == self.depth_hist_len:
                prop_stack = torch.cat(list(self.prop_histories[env_id]), dim=0)  # [prop_hist_len * proprio_dim]
                depth_stack = torch.stack(list(self.depth_histories[env_id]), dim=0)  # [depth_hist_len, H, W]
                self.seq_prop_histories[env_id].append(prop_stack)
                self.seq_depth_histories[env_id].append(depth_stack)
                if (
                    len(self.seq_prop_histories[env_id]) == self.sequence_length
                    and len(self.seq_depth_histories[env_id]) == self.sequence_length
                ):
                    ready_indices.append(env_id)

        actions = torch.zeros(self.num_envs, self.model.action_head.action_dim, device=self.device)

        if ready_indices:
            prop_tensors: List[torch.Tensor] = []
            depth_tensors: List[torch.Tensor] = []
            for env_id in ready_indices:
                prop_seq = torch.stack(list(self.seq_prop_histories[env_id]), dim=0)  # [S, prop_hist_len * proprio_dim]
                depth_seq = torch.stack(list(self.seq_depth_histories[env_id]), dim=0)  # [S, depth_hist_len, H, W]
                prop_tensors.append(prop_seq.unsqueeze(0))  # [1, S, prop_hist_len * proprio_dim]
                depth_tensors.append(depth_seq.unsqueeze(0))  # [1, S, depth_hist_len, H, W]
            proprios_ready = torch.cat(prop_tensors, dim=0)  # [N_ready, S, prop_hist_len * proprio_dim]
            depths_ready = torch.cat(depth_tensors, dim=0)  # [N_ready, S, depth_hist_len, H, W]
            with torch.no_grad():
                pred_ready = self.model(proprios_ready, depths_ready)  # [N_ready, S, action_dim]
            actions_ready = pred_ready[:, -1, :]  # take action from last timestep of the sequence
            for idx, env_id in enumerate(ready_indices):
                actions[env_id] = actions_ready[idx]

        return actions


def parse_args_play() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an offline-trained student policy in Isaac parkour envs.")
    parser.add_argument("--task", type=str, required=True, help="Isaac task name.")
    parser.add_argument("--student_checkpoint", type=str, default=None, help="Path to a specific student checkpoint file.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing student_epoch_*.pt; used when --student_checkpoint is not provided.",
    )
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="History length for depth tokens.")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=16,
        help="Temporal sequence length S for the student TXL during play.",
    )
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps to run.")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args_play()
    headless = getattr(args, "headless", False)
    disable_fabric = getattr(args, "disable_fabric", False)
    if not headless:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import parkour_tasks  # noqa: F401  # ensure tasks register after Isaac app is initialized

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
    if not headless:
        env_cfg.viewer.origin_type = "world"
        spacing = float(env_cfg.scene.env_spacing)
        grid = int(np.ceil(np.sqrt(args.num_envs)))
        env_cfg.viewer.eye = [spacing * grid * 0.5, spacing * grid * 0.5, 3.0]
        env_cfg.viewer.lookat = [0.0, 0.0, 0.5]
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if not headless else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

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
    )
    proprio_dim = int(meta["num_prop"])
    camera_resolution = tuple(meta.get("camera_resolution", [64, 64]))

    runner = StudentOnlineRunner(
        model=student_model,
        num_envs=vec_env.num_envs,
        proprio_dim=proprio_dim,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        camera_resolution=camera_resolution,  # type: ignore[arg-type]
        device=device,
        sequence_length=args.sequence_length,
    )
    runner.reset()

    obs, extras = vec_env.get_observations()
    step = 0
    while simulation_app.is_running() and step < args.max_steps:
        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用 TeacherCam 任务。")

        obs_prop = obs[:, :proprio_dim]
        obs_prop_t = obs_prop.to(device)
        depth_t = depth_image.to(device)

        actions = runner.act(obs_prop_t, depth_t)
        obs_next, rews, dones, extras = vec_env.step(actions)
        done_mask = dones.squeeze(-1).bool()
        if done_mask.any():
            runner.reset_done(done_mask)

        obs = obs_next
        step += 1

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
