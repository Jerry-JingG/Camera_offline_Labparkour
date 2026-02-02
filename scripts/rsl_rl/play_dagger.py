from __future__ import annotations

from collections import deque
import argparse
import os
import re
import struct
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
    Maintains short histories for tokenization and runs the student policy step-by-step.

    Each call to ``act`` feeds only the current step (S=1) to the Transformer-XL; long
    temporal context is carried through the per-layer memories ``self.mems``.
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
        self.num_layers = len(self.model.temporal_model.layers)

        self.prop_histories: List[deque] = [deque(maxlen=prop_hist_len) for _ in range(num_envs)]
        self.depth_histories: List[deque] = [deque(maxlen=depth_hist_len) for _ in range(num_envs)]
        # TXL memory per env (list of tensors per layer)
        self.mems: List[List[torch.Tensor]] = []

    def reset(self) -> None:
        """Clear histories for all environments."""
        for hist in self.prop_histories:
            hist.clear()
        for hist in self.depth_histories:
            hist.clear()
        # reset mems for all envs
        self.mems = [self.model.temporal_model.reset_mems(1) for _ in range(self.num_envs)]

    def reset_done(self, done_mask: torch.Tensor) -> None:
        """Clear histories for environments where done_mask is True."""
        # Normalize to 1D boolean tensor to avoid 0-d iteration errors when num_envs=1.
        mask = torch.as_tensor(done_mask, device=self.device).flatten().to(dtype=torch.bool)
        for env_id, done in enumerate(mask.tolist()):
            if done:
                self.prop_histories[env_id].clear()
                self.depth_histories[env_id].clear()
                self.mems[env_id] = self.model.temporal_model.reset_mems(1)

    def num_ready_envs(self) -> int:
        """Return number of envs whose short histories are ready for inference."""
        ready = 0
        for prop_hist, depth_hist in zip(self.prop_histories, self.depth_histories):
            if len(prop_hist) == self.prop_hist_len and len(depth_hist) == self.depth_hist_len:
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
        prop_tensors: List[torch.Tensor] = []
        depth_tensors: List[torch.Tensor] = []
        for env_id in range(self.num_envs):
            if len(self.prop_histories[env_id]) == self.prop_hist_len and len(self.depth_histories[env_id]) == self.depth_hist_len:
                prop_stack = torch.cat(list(self.prop_histories[env_id]), dim=0)  # [prop_hist_len * proprio_dim]
                depth_stack = torch.stack(list(self.depth_histories[env_id]), dim=0)  # [depth_hist_len, H, W]
                prop_tensors.append(prop_stack)  # [prop_hist_len * proprio_dim]
                depth_tensors.append(depth_stack)  # [depth_hist_len, H, W]
                ready_indices.append(env_id)

        actions = torch.zeros(self.num_envs, self.model.action_head.action_dim, device=self.device)

        if ready_indices:
            mems_ready: List[torch.Tensor] = []
            proprios_ready = torch.stack(prop_tensors, dim=0)  # [N_ready, prop_hist_len * proprio_dim]
            depths_ready = torch.stack(depth_tensors, dim=0)  # [N_ready, depth_hist_len, H, W]
            # gather and pad mems across envs for each layer
            for layer_idx in range(self.num_layers):
                layer_mems: List[torch.Tensor] = [self.mems[env_id][layer_idx] for env_id in ready_indices]
                target_len = max(m.size(1) for m in layer_mems)
                if target_len == 0:
                    mems_ready.append(torch.cat(layer_mems, dim=0))
                else:
                    padded: List[torch.Tensor] = []
                    for m in layer_mems:
                        if m.size(1) == target_len:
                            padded.append(m)
                        else:
                            pad_len = target_len - m.size(1)
                            pad = torch.zeros(m.size(0), pad_len, m.size(2), device=m.device, dtype=m.dtype)
                            padded.append(torch.cat([pad, m], dim=1))
                    mems_ready.append(torch.cat(padded, dim=0))
            with torch.no_grad():
                pred_ready, yaw_pred_ready, new_mems = self.model.forward_step(  # type: ignore[attr-defined]
                    proprios_ready, depths_ready, mems=mems_ready
                )  # [N_ready, action_dim]
            actions_ready = pred_ready  # single-step output
            for idx, env_id in enumerate(ready_indices):
                actions[env_id] = actions_ready[idx]
                # update mems for this env (detach to avoid graph accumulation)
                self.mems[env_id] = [mem[idx : idx + 1].detach() for mem in new_mems]  # type: ignore[arg-type]

        return actions


def _save_verification_data(
    output_path: str,
    frames: List[Dict[str, np.ndarray]],
    prop_dim: int,
    depth_dim: int,
    action_dim: int,
    mems_dim: int,
) -> None:
    """Save recorded frames to binary file for C++ verification (v2 format with mems).

    File format v2:
        Header (24 bytes):
            int32: version (= 2)
            int32: num_frames
            int32: prop_dim
            int32: depth_dim
            int32: action_dim
            int32: mems_dim
        Body (repeated num_frames times):
            float32 * prop_dim: proprio data
            float32 * depth_dim: depth data (flattened)
            float32 * mems_dim: mems data (flattened, 推理前状态)
            float32 * action_dim: action data
    """
    num_frames = len(frames)
    print(f"[INFO] Saving {num_frames} frames (v2 format with mems) to {output_path}")

    with open(output_path, "wb") as f:
        # Write header
        f.write(struct.pack("<i", 2))  # version
        f.write(struct.pack("<i", num_frames))
        f.write(struct.pack("<i", prop_dim))
        f.write(struct.pack("<i", depth_dim))
        f.write(struct.pack("<i", action_dim))
        f.write(struct.pack("<i", mems_dim))

        # Write frames
        for frame in frames:
            f.write(frame["proprio"].astype(np.float32).tobytes())
            f.write(frame["depth"].astype(np.float32).tobytes())
            f.write(frame["mems"].astype(np.float32).tobytes())
            f.write(frame["action"].astype(np.float32).tobytes())

    print(f"[INFO] Saved successfully. Header: version=2, frames={num_frames}, prop={prop_dim}, depth={depth_dim}, mems={mems_dim}, action={action_dim}")


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
    parser.add_argument("--prop_hist_len", type=int, default=1, help="History length for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="History length for depth tokens.")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=16,
        help="Temporal sequence length S for the student TXL during play.",
    )
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps to run.")

    # 录制验证数据相关参数
    parser.add_argument(
        "--record_data",
        action="store_true",
        help="Enable recording of (proprio, depth, action) for env_id=0 for verification.",
    )
    parser.add_argument(
        "--record_output",
        type=str,
        default="/tmp/dagger_verify_data.bin",
        help="Output path for recorded verification data (binary format).",
    )

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

    # ========== 录制相关初始化 ==========
    record_data = getattr(args, "record_data", False)
    record_output = getattr(args, "record_output", "/tmp/dagger_verify_data.bin")
    recorded_frames: List[Dict[str, np.ndarray]] = []
    # mems_dim = num_layers * mem_len * token_dim = 3 * 64 * 128 = 24576
    mems_dim = runner.num_layers * 64 * 128
    if record_data:
        print(f"[INFO] Recording enabled. Will save to: {record_output}")
        print(f"[INFO] mems_dim = {mems_dim} (num_layers={runner.num_layers}, mem_len=64, token_dim=128)")
    # =====================================

    obs, extras = vec_env.get_observations()
    step = 0
    while simulation_app.is_running() and step < args.max_steps:
        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用 TeacherCam 任务。")

        obs_prop = obs[:, :proprio_dim]
        obs_prop_t = obs_prop.to(device)
        depth_t = depth_image.to(device)

        # 捕获推理前的 mems 状态 (用于录制)
        pre_mems_0 = None
        if record_data and len(runner.mems) > 0:
            # mems[0] 是 env_id=0 的 mems，格式是 List[Tensor]，每个 [1, mem_len, token_dim]
            pre_mems_0 = torch.stack([m.squeeze(0) for m in runner.mems[0]], dim=0).cpu().numpy()  # [num_layers, mem_len, token_dim]

        actions = runner.act(obs_prop_t, depth_t)

        # ========== 录制: 在推理后保存 env_id=0 的输入和输出 ==========
        if record_data:
            # 在 act() 之后读取更新后的历史 (包含当前帧)
            prop_hist_0 = runner.prop_histories[0]
            depth_hist_0 = runner.depth_histories[0]

            # 只在历史填满时录制 (与 act() 内部逻辑一致)
            if len(prop_hist_0) == runner.prop_hist_len and len(depth_hist_0) == runner.depth_hist_len:
                prop_stack_0 = torch.cat(list(prop_hist_0), dim=0).cpu().numpy()
                depth_stack_0 = torch.stack(list(depth_hist_0), dim=0).cpu().numpy()
                action_0 = actions[0].cpu().numpy()

                # 保存推理前的 mems 状态
                mems_0 = pre_mems_0.flatten().copy() if pre_mems_0 is not None else np.zeros(mems_dim, dtype=np.float32)

                recorded_frames.append({
                    "proprio": prop_stack_0.copy(),
                    "depth": depth_stack_0.flatten().copy(),
                    "mems": mems_0,
                    "action": action_0.copy(),
                })
        # =====================================================

        if step % 50 == 0:
            try:
                mean_norm = actions.norm(dim=-1).mean().item()
            except Exception:
                mean_norm = float("nan")
            rec_info = f" (recorded: {len(recorded_frames)} frames)" if record_data else ""
            print(f"[student_play] step={step} mean_action_norm={mean_norm:.6f}{rec_info}")
        obs_next, rews, dones, extras = vec_env.step(actions)
        # Keep done mask as 1D to avoid 0-d tensors when num_envs=1
        done_mask = dones.view(-1).bool()
        if done_mask.any():
            runner.reset_done(done_mask)

        obs = obs_next
        step += 1

    # ========== 保存录制数据 ==========
    if record_data and recorded_frames:
        _save_verification_data(
            record_output,
            recorded_frames,
            prop_dim=args.prop_hist_len * proprio_dim,
            depth_dim=args.depth_hist_len * camera_resolution[0] * camera_resolution[1],
            action_dim=12,
            mems_dim=mems_dim,
        )
    elif record_data:
        print("[WARNING] Recording enabled but no frames were captured.")
    # =================================

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
