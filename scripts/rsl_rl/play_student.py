"""
Stateful (TransformerXL Memory-Based) Inference Script - Optimized Vectorized Version
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

# Load MultiModalStudentPolicy directly
TRAIN_STUDENT_PATH = Path(__file__).resolve().parent / "train_student_from_dataset.py"
_spec = importlib.util.spec_from_file_location("train_student_from_dataset", TRAIN_STUDENT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load MultiModalStudentPolicy from {TRAIN_STUDENT_PATH}")
_module = importlib.util.module_from_spec(_spec)
# Register into sys.modules before execution so dataclasses can resolve module references.
sys.modules[_spec.name] = _module
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
    mem_len: int = 64,
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
        "mem_len": mem_len,
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
    parser.add_argument("--mem_len", type=int, default=64, help="TransformerXL memory length (S).")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps to run.")
    parser.add_argument("--use_dropout", action="store_true", default=False, help="Simulate camera dropout.")

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
        mem_len=args.mem_len,
    )
    proprio_dim = int(meta["num_prop"])
    camera_resolution = tuple(meta.get("camera_resolution", (58, 87)))

    runner = StudentOnlineRunner(
        model=student_model,
        num_envs=vec_env.num_envs,
        proprio_dim=proprio_dim,
        prop_hist_len=args.prop_hist_len,
        depth_hist_len=args.depth_hist_len,
        camera_resolution=camera_resolution,  # type: ignore[arg-type]
        device=device,
    )
    runner.reset()

    from utils.dropout_manager import CameraDropoutManager
    dropout_manager = None
    if args.use_dropout:
        step_dt = float(vec_env.unwrapped.step_dt)
        dropout_manager = CameraDropoutManager(
            num_envs=args.num_envs,
            device=device,
            dt=step_dt,
            prob_start_offline=0.0,
            prob_cam_offline=1.0,
            online_duration_range=(5.0, 5.0),
            offline_duration_range=(2.0, 2.0)
        )
        print("[Play] Camera Dropout Simulation: ENABLED")

    obs, extras = vec_env.get_observations()
    dones_bool = torch.zeros(vec_env.num_envs, device=device, dtype=torch.bool)
    step = 0
    while simulation_app.is_running() and step < args.max_steps:
        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用 TeacherCam 任务。")
        obs_prop = obs[:, :proprio_dim]
        obs_prop[:, 7] = 0
        obs_prop[:, 12] = dones_bool.float()

        if dropout_manager:
            dropout_manager.reset_env(dones_bool)
            dropout_manager.update(depth_image=depth_image, obs_prop=obs_prop)

        student_action = runner.act(obs_prop, depth_image)

        if step % 50 == 0:
            try:
                mean_norm = student_action.norm(dim=-1).mean().item()
            except Exception:
                mean_norm = float("nan")
            print(f"[student_play] step={step} mean_action_norm={mean_norm:.6f}")

        obs_next, rews, dones, extras = vec_env.step(student_action)
        dones_bool = dones.squeeze(-1).bool()

        # Vectorized reset for done envs
        if dones_bool.any():
            runner.reset_done(dones_bool)

        obs = obs_next
        step += 1

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
