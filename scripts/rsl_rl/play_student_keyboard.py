"""
DAGGER v4.0 学生策略推理脚本 - 键盘控制版本

本脚本用于在IsaacLab仿真中运行训练好的学生策略，使用键盘控制替代waypoint导航。
这确保了训练和推理架构的一致性：
- 训练时：delta_yaw 来自 waypoint 计算
- 推理时：delta_yaw 来自键盘输入（替代 waypoint）

键盘控制：
- W/↑: 前进
- S/↓: 后退（慢速）
- A/←: 左转
- D/→: 右转
- SPACE: 紧急停止
- M: 切换控制模式 (keyboard/hybrid/autonomous)
- 1-9: 选择控制的环境
- 0: 控制所有环境

观测向量注入点（与训练一致）：
- Index 6: delta_yaw (方向命令)
- Index 7: delta_next_yaw (下一步方向命令)
- Index 10: lin_vel_x (前进速度命令)
"""

from __future__ import annotations

from collections import deque
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib.util
import numpy as np

import torch

from utils.camera_blackout_manager import CameraBlackoutManager
from utils.dropout_manager import CameraDropoutManager
from utils.keyboard_command_manager import KeyboardCommandManager, ControlMode

from isaaclab.app import AppLauncher

# Ensure project roots are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PARKOUR_TASKS_ROOT = os.path.join(PROJECT_ROOT, "parkour_tasks")
if PARKOUR_TASKS_ROOT not in sys.path:
    sys.path.insert(0, PARKOUR_TASKS_ROOT)

import cli_args  # isort: skip

# Load MultiModalStudentPolicy directly from the training script path
TRAIN_STUDENT_PATH = Path(__file__).resolve().parent / "train_student_from_dataset.py"
_spec = importlib.util.spec_from_file_location("train_student_from_dataset", TRAIN_STUDENT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load MultiModalStudentPolicy from {TRAIN_STUDENT_PATH}")
_module = importlib.util.module_from_spec(_spec)
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
    """Load a trained student policy and associated metadata for playback.

    Note: prop_hist_len and depth_hist_len are read from checkpoint metadata.
    The function parameters are ignored but kept for API compatibility.
    """
    payload = torch.load(checkpoint_path, map_location=device)
    meta: Dict[str, object] = dict(payload.get("meta", {}))

    proprio_dim = int(meta["num_prop"])
    action_dim = int(meta["action_dim"])
    camera_resolution = tuple(meta.get("camera_resolution", [64, 64]))

    # Read prop_hist_len and depth_hist_len from checkpoint metadata
    ckpt_prop_hist_len = int(meta.get("prop_hist_len", 1))
    ckpt_depth_hist_len = int(meta.get("depth_hist_len", 4))

    # Warn if command-line values differ from checkpoint values
    if prop_hist_len != ckpt_prop_hist_len:
        print(f"[WARNING] prop_hist_len mismatch: CLI={prop_hist_len}, checkpoint={ckpt_prop_hist_len}. Using checkpoint value.")
    if depth_hist_len != ckpt_depth_hist_len:
        print(f"[WARNING] depth_hist_len mismatch: CLI={depth_hist_len}, checkpoint={ckpt_depth_hist_len}. Using checkpoint value.")

    # Use checkpoint values
    prop_hist_len = ckpt_prop_hist_len
    depth_hist_len = ckpt_depth_hist_len

    # Store in meta for downstream use
    meta["prop_hist_len"] = prop_hist_len
    meta["depth_hist_len"] = depth_hist_len

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
    使用 TransformerXL 记忆机制的学生策略在线推理器。

    每个时间步只输入 1 个 token (S=1)，复用 mems 中的历史信息，
    推理效率从 O(S²) 降低到 O(S)。
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
        self.proprio_dim = proprio_dim
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.camera_resolution = camera_resolution
        self.device = device

        # 短期记忆：存储最近 N 帧的本体感知和深度图像
        self.prop_histories: List[deque] = [deque(maxlen=prop_hist_len) for _ in range(num_envs)]
        self.depth_histories: List[deque] = [deque(maxlen=depth_hist_len) for _ in range(num_envs)]

        # TransformerXL 长期记忆
        self.num_layers = len(self.model.temporal_model.layers)
        self.mem_len = self.model.temporal_model.mem_len
        self.d_model = self.model.temporal_model.d_model
        self.mems: List[List[torch.Tensor]] = []

        self.reset()

    def reset(self) -> None:
        """重置所有环境的短期记忆和长期记忆"""
        for i in range(self.num_envs):
            self.prop_histories[i].clear()
            self.depth_histories[i].clear()

            for _ in range(self.prop_hist_len):
                self.prop_histories[i].append(
                    torch.zeros(self.proprio_dim, device=self.device)
                )
            for _ in range(self.depth_hist_len):
                self.depth_histories[i].append(
                    torch.zeros(*self.camera_resolution, device=self.device)
                )

        self.mems = []
        for _ in range(self.num_envs):
            env_mems = [
                torch.zeros(1, 0, self.d_model, device=self.device)
                for _ in range(self.num_layers)
            ]
            self.mems.append(env_mems)

    def reset_done(self, done_mask: torch.Tensor) -> None:
        """重置已终止环境的记忆"""
        for env_id, done in enumerate(done_mask):
            if bool(done):
                self.prop_histories[env_id].clear()
                self.depth_histories[env_id].clear()

                for _ in range(self.prop_hist_len):
                    self.prop_histories[env_id].append(
                        torch.zeros(self.proprio_dim, device=self.device)
                    )
                for _ in range(self.depth_hist_len):
                    self.depth_histories[env_id].append(
                        torch.zeros(*self.camera_resolution, device=self.device)
                    )

                self.mems[env_id] = [
                    torch.zeros(1, 0, self.d_model, device=self.device)
                    for _ in range(self.num_layers)
                ]

    def act(
        self,
        obs_prop: torch.Tensor,
        depth_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        使用 TransformerXL 记忆机制计算动作。

        Returns:
            actions: 形状为 [num_envs, action_dim] 的动作张量
            yaw_pred: 形状为 [num_envs, 2] 的 yaw 预测张量（可选）
        """
        obs_prop = obs_prop.to(self.device)
        depth_image = depth_image.to(self.device)

        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)

        prop_tokens = []
        depth_tokens = []

        for env_id in range(self.num_envs):
            self.prop_histories[env_id].append(obs_prop[env_id])
            self.depth_histories[env_id].append(depth_image[env_id])

            prop_stack = torch.cat(list(self.prop_histories[env_id]), dim=0)
            depth_stack = torch.stack(list(self.depth_histories[env_id]), dim=0)

            prop_tokens.append(prop_stack)
            depth_tokens.append(depth_stack)

        prop_batch = torch.stack(prop_tokens).unsqueeze(1)
        depth_batch = torch.stack(depth_tokens).unsqueeze(1)

        with torch.no_grad():
            actions, yaw_pred, new_mems_batch = self._forward_with_mems(prop_batch, depth_batch)

        for env_id in range(self.num_envs):
            for layer_id in range(self.num_layers):
                self.mems[env_id][layer_id] = new_mems_batch[layer_id][env_id:env_id+1].detach()

        yaw_out = yaw_pred.squeeze(1) if yaw_pred is not None else None
        return actions.squeeze(1), yaw_out

    def _forward_with_mems(
        self,
        proprio_seq: torch.Tensor,
        depth_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """带记忆的前向传播"""
        batched_mems = []
        for layer_id in range(self.num_layers):
            env_mems = [self.mems[env_id][layer_id] for env_id in range(self.num_envs)]
            mem_lens = [m.size(1) for m in env_mems]
            max_mem_len = max(mem_lens) if mem_lens else 0

            if max_mem_len == 0:
                layer_mems = torch.zeros(
                    self.num_envs, 0, self.d_model, device=self.device
                )
            else:
                padded_mems = []
                for m in env_mems:
                    current_len = m.size(1)
                    if current_len < max_mem_len:
                        pad_len = max_mem_len - current_len
                        padding = torch.zeros(1, pad_len, self.d_model, device=self.device)
                        m = torch.cat([padding, m], dim=1)
                    padded_mems.append(m)
                layer_mems = torch.cat(padded_mems, dim=0)

            batched_mems.append(layer_mems)

        actions, yaw_pred, new_mems = self.model.forward_step(
            proprio_seq.squeeze(1),
            depth_seq.squeeze(1),
            mems=batched_mems,
        )

        actions = actions.unsqueeze(1)
        yaw_pred = yaw_pred.unsqueeze(1) if yaw_pred is not None else None

        return actions, yaw_pred, new_mems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play student policy with keyboard control (replaces waypoint navigation)."
    )
    parser.add_argument("--task", type=str, required=True, help="Isaac task name.")
    parser.add_argument("--student_checkpoint", type=str, default=None,
                        help="Path to a specific student checkpoint file.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing student_epoch_*.pt.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="History length for depth tokens.")
    parser.add_argument("--mem_len", type=int, default=64, help="TransformerXL memory length.")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum steps to run.")

    # Keyboard control arguments
    parser.add_argument(
        "--control_mode",
        type=str,
        choices=["keyboard", "hybrid", "autonomous"],
        default="keyboard",
        help="Control mode: 'keyboard' (full control), 'hybrid' (override when pressed), 'autonomous' (no injection).",
    )
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.3,
        help="Command smoothing factor (0=no smoothing, 1=instant). Default: 0.3",
    )
    parser.add_argument(
        "--default_forward_speed",
        type=float,
        default=0.5,
        help="Default forward speed when W is pressed (m/s). Default: 0.5",
    )
    parser.add_argument(
        "--yaw_rate_scale",
        type=float,
        default=0.7,
        help="Scale factor for yaw rate (1.0 = full range [-1.5, 1.5]). Default: 0.7",
    )

    # Camera mode selection
    parser.add_argument(
        "--camera_mode",
        type=str,
        choices=["dropout", "blackout", "normal"],
        default="normal",
        help="Camera mode: 'normal' (always on), 'dropout' (matches training), 'blackout' (always off).",
    )
    parser.add_argument(
        "--camera_dropout_prob",
        type=float,
        default=0.3,
        help="Camera dropout probability when using 'dropout' mode.",
    )

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def print_keyboard_help() -> None:
    """Print keyboard control help."""
    print("=" * 60)
    print("KEYBOARD CONTROLS:")
    print("  W / UP    : Move forward")
    print("  S / DOWN  : Move backward (slow)")
    print("  A / LEFT  : Turn left")
    print("  D / RIGHT : Turn right")
    print("  SPACE     : Emergency stop")
    print("  M         : Cycle control mode (keyboard/hybrid/autonomous)")
    print("  1-9       : Select environment to control")
    print("  0         : Control all environments")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    # Validate arguments
    if not 0.0 <= args.camera_dropout_prob <= 1.0:
        raise ValueError(f"camera_dropout_prob must be between 0.0 and 1.0, got {args.camera_dropout_prob}")

    headless = getattr(args, "headless", False)
    disable_fabric = getattr(args, "disable_fabric", False)
    if not headless:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import parkour_tasks  # noqa: F401

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

    # Load checkpoint
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

    print(f"[INFO] Loading checkpoint: {ckpt_path}")

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

    # Use prop_hist_len and depth_hist_len from checkpoint metadata
    prop_hist_len = int(meta["prop_hist_len"])
    depth_hist_len = int(meta["depth_hist_len"])
    print(f"[INFO] Using checkpoint values: prop_hist_len={prop_hist_len}, depth_hist_len={depth_hist_len}")

    runner = StudentOnlineRunner(
        model=student_model,
        num_envs=vec_env.num_envs,
        proprio_dim=proprio_dim,
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        camera_resolution=camera_resolution,  # type: ignore[arg-type]
        device=device,
    )
    runner.reset()

    # Initialize camera manager
    camera_manager = None
    if args.camera_mode == "blackout":
        camera_manager = CameraBlackoutManager(
            num_envs=vec_env.num_envs,
            device=device,
        )
        print(f"[INFO] Camera mode: BLACKOUT - all envs will have blackout camera.")
    elif args.camera_mode == "dropout":
        camera_manager = CameraDropoutManager(
            num_envs=vec_env.num_envs,
            device=device,
            dt=vec_env.unwrapped.step_dt,
            prob_start_offline=args.camera_dropout_prob,
        )
        print(f"[INFO] Camera mode: DROPOUT - prob={args.camera_dropout_prob}")
    else:
        print(f"[INFO] Camera mode: NORMAL - camera always on.")

    # Initialize keyboard command manager
    control_mode = ControlMode(args.control_mode)
    keyboard_manager = KeyboardCommandManager(
        num_envs=vec_env.num_envs,
        device=device,
        control_mode=control_mode,
        smoothing_factor=args.smoothing_factor,
        default_forward_speed=args.default_forward_speed,
        yaw_rate_scale=args.yaw_rate_scale,
    )
    keyboard_manager.setup_keyboard()
    print_keyboard_help()

    obs, extras = vec_env.get_observations()
    step = 0

    print(f"\n[INFO] Starting inference loop (max_steps={args.max_steps})...")
    print(f"[INFO] Press WASD to control the robot, M to change mode, SPACE to stop.\n")

    while simulation_app.is_running() and step < args.max_steps:
        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用 TeacherCam 任务。")

        # Apply camera effects (dropout/blackout) if enabled
        if camera_manager is not None:
            depth_image = camera_manager.update(depth_image)

        obs_prop = obs[:, :proprio_dim]

        # Update keyboard state and inject commands
        keyboard_manager.step()
        obs_prop = keyboard_manager.inject_commands(obs_prop)

        # Run inference
        student_action, yaw_pred = runner.act(obs_prop, depth_image)

        # Periodic logging
        if step % 100 == 0:
            try:
                mean_norm = student_action.norm(dim=-1).mean().item()
            except Exception:
                mean_norm = float("nan")
            print(f"[step={step:5d}] action_norm={mean_norm:.4f} | {keyboard_manager.get_status_string()}")

        # Step environment
        obs_next, rews, dones, extras = vec_env.step(student_action)
        done_mask = dones.squeeze(-1).bool()
        if done_mask.any():
            runner.reset_done(done_mask)
            keyboard_manager.reset_env(done_mask)
            if camera_manager is not None:
                camera_manager.reset_env(done_mask)

        obs = obs_next
        step += 1

    print(f"\n[INFO] Inference completed. Total steps: {step}")
    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
