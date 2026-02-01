"""
一般的 transformer 网络训练时接收一个 sequence 的输入然后对这一整个 sequence 做输出，并且对一整个 sequence 的输出做监督
这相当于执行了 sequence_length 次 inference
由于 transformerxl 是因果注意力的，序列的前几个输出也相当于是 sequence 没有满就输出了的
所以即使transformer是sequence by sequence训练的, 它可以学习到输入长度不足sequence_length时的输出
"""

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

from utils.camera_blackout_manager import CameraBlackoutManager

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
    
    Args:
        mem_len: TransformerXL memory length. Can be larger than training value
                 to attend to longer history at inference time.
    """
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
        "mem_len": mem_len,  # 可以在推理时设置比训练更大的值
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

        # 短期记忆：存储最近 N 帧的本体感知和深度图像，用于构建单个 token
        self.prop_histories: List[deque] = [deque(maxlen=prop_hist_len) for _ in range(num_envs)]
        self.depth_histories: List[deque] = [deque(maxlen=depth_hist_len) for _ in range(num_envs)]

        # TransformerXL 长期记忆：每个环境独立维护
        # mems 是一个 List[Tensor]，每层一个，形状为 [B, mem_len, d_model]
        # 由于每个环境需要独立 reset，我们为每个环境单独维护 mems
        self.num_layers = len(self.model.temporal_model.layers)
        self.mem_len = self.model.temporal_model.mem_len
        self.d_model = self.model.temporal_model.d_model
        self.mems: List[List[torch.Tensor]] = []  # [num_envs][num_layers] -> [1, mem_len, d_model]
        self.step_count = 0  # 用于控制 debug 打印频率

        self.reset()

    def reset(self) -> None:
        """重置所有环境的短期记忆和长期记忆"""
        # 重置短期记忆（观测历史）
        for i in range(self.num_envs):
            self.prop_histories[i].clear()
            self.depth_histories[i].clear()

            for _ in range(self.prop_hist_len):
                self.prop_histories[i].append(
                    torch.zeros(self.proprio_dim, device=self.device)
                )
            for _ in range(self.depth_hist_len):
                self.depth_histories[i].append(
                    torch.zeros(
                        *self.camera_resolution,
                        device=self.device,
                    )
                )

        # 重置 TransformerXL 长期记忆
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
                # 重置短期记忆
                self.prop_histories[env_id].clear()
                self.depth_histories[env_id].clear()

                for _ in range(self.prop_hist_len):
                    self.prop_histories[env_id].append(
                        torch.zeros(self.proprio_dim, device=self.device)
                    )
                for _ in range(self.depth_hist_len):
                    self.depth_histories[env_id].append(
                        torch.zeros(
                            *self.camera_resolution,
                            device=self.device,
                        )
                    )

                # 重置 TransformerXL 长期记忆
                self.mems[env_id] = [
                    torch.zeros(1, 0, self.d_model, device=self.device)
                    for _ in range(self.num_layers)
                ]

    # def debug_print_mems(self, step: int, env_id: int = 0) -> None:
    #     """
    #     打印指定环境的 TransformerXL 记忆状态，用于调试。
        
    #     Args:
    #         step: 当前时间步
    #         env_id: 要查看的环境 ID（默认为 0）
    #     """
    #     if env_id >= self.num_envs:
    #         print(f"[debug_mems] env_id={env_id} 超出范围 (num_envs={self.num_envs})")
    #         return
        
    #     print(f"\n{'='*60}")
    #     print(f"[debug_mems] Step={step}, Env={env_id}")
    #     print(f"{'='*60}")
        
    #     for layer_id in range(self.num_layers):
    #         mem = self.mems[env_id][layer_id]
    #         mem_len = mem.size(1)
            
    #         if mem_len == 0:
    #             print(f"  Layer {layer_id}: mem_len=0 (empty)")
    #         else:
    #             mem_mean = mem.mean().item()
    #             mem_std = mem.std().item()
    #             mem_norm = mem.norm().item()
    #             mem_abs_max = mem.abs().max().item()
                
    #             print(f"  Layer {layer_id}: mem_len={mem_len:3d} | "
    #                   f"mean={mem_mean:+.4f} | std={mem_std:.4f} | "
    #                   f"norm={mem_norm:.4f} | abs_max={mem_abs_max:.4f}")
        
    #     # 打印所有环境的 mem 长度概览
    #     all_mem_lens = [self.mems[i][0].size(1) for i in range(self.num_envs)]
    #     print(f"  All envs mem_len: {all_mem_lens}")
    #     print(f"{'='*60}\n")

    def act(
        self,
        obs_prop: torch.Tensor,  # [num_envs, proprio_dim]
        depth_image: torch.Tensor,  # [num_envs, H, W] or [num_envs, 1, H, W]
    ) -> torch.Tensor:
        """
        使用 TransformerXL 记忆机制计算动作。
        每步只输入 1 个 token (S=1)，复用 mems 中的历史信息。
        
        Returns:
            actions: 形状为 [num_envs, action_dim] 的动作张量
        """
        obs_prop = obs_prop.to(self.device)
        depth_image = depth_image.to(self.device)

        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)

        # 更新短期记忆并构建当前时间步的输入 token
        prop_tokens = []
        depth_tokens = []

        for env_id in range(self.num_envs):
            self.prop_histories[env_id].append(obs_prop[env_id])
            self.depth_histories[env_id].append(depth_image[env_id])
            
            # 拼接历史帧构建单个 token
            prop_stack = torch.cat(list(self.prop_histories[env_id]), dim=0)  # [prop_hist_len * proprio_dim]
            depth_stack = torch.stack(list(self.depth_histories[env_id]), dim=0)  # [depth_hist_len, H, W]
            
            prop_tokens.append(prop_stack)
            depth_tokens.append(depth_stack)

        # 构建批次输入，S=1
        prop_batch = torch.stack(prop_tokens).unsqueeze(1)  # [B, 1, prop_hist_len * proprio_dim]
        depth_batch = torch.stack(depth_tokens).unsqueeze(1)  # [B, 1, depth_hist_len, H, W]

        with torch.no_grad():
            # 使用 forward_with_mems 进行带记忆的推理
            actions, new_mems_batch = self._forward_with_mems(prop_batch, depth_batch)

        # 更新每个环境的 mems
        for env_id in range(self.num_envs):
            for layer_id in range(self.num_layers):
                self.mems[env_id][layer_id] = new_mems_batch[layer_id][env_id:env_id+1].detach()

        # # Debug: 每 50 步打印一次 mems 状态
        # self.step_count += 1
        # if self.step_count % 50 == 0:
        #     self.debug_print_mems(self.step_count, env_id=0)

        return actions.squeeze(1)  # [B, action_dim]

    def _forward_with_mems(
        self,
        proprio_seq: torch.Tensor,  # [B, 1, prop_feat_dim]
        depth_seq: torch.Tensor,    # [B, 1, depth_hist_len, H, W]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        带记忆的前向传播，复用模型的各个组件。
        
        Returns:
            actions: [B, 1, action_dim]
            new_mems: List[Tensor]，每层一个，形状为 [B, new_mem_len, d_model]
        """
        batch_size = proprio_seq.shape[0]
        
        # 1. 编码当前时间步的输入
        prop_encoded = self.model.proprio_encoder(
            proprio_seq.reshape(batch_size, -1)
        )  # [B, 1, token_dim]
        
        depth_encoded = self.model.depth_encoder(
            depth_seq.reshape(batch_size, depth_seq.size(2), depth_seq.size(3), depth_seq.size(4))
        )  # [B, T, token_dim]
        
        # 2. 多模态融合
        fused = self.model.fusion_transformer(prop_encoded, depth_encoded)
        fused_token = fused["all_pooled"].unsqueeze(1)  # [B, 1, token_dim]
        
        # 3. 合并所有环境的 mems 为批次格式（处理不同长度的情况）
        # 当某些环境 reset 后，其 mems 长度为 0，需要填充到最大长度
        batched_mems = []
        for layer_id in range(self.num_layers):
            env_mems = [self.mems[env_id][layer_id] for env_id in range(self.num_envs)]
            mem_lens = [m.size(1) for m in env_mems]
            max_mem_len = max(mem_lens) if mem_lens else 0
            
            if max_mem_len == 0:
                # 所有环境的 mems 都是空的
                layer_mems = torch.zeros(
                    self.num_envs, 0, self.d_model, device=self.device
                )
            else:
                # 将所有 mems 左侧填充到相同长度
                padded_mems = []
                for m in env_mems:
                    current_len = m.size(1)
                    if current_len < max_mem_len:
                        # 左侧填充零（保持时间对齐：最新的在右侧）
                        pad_len = max_mem_len - current_len
                        padding = torch.zeros(1, pad_len, self.d_model, device=self.device)
                        m = torch.cat([padding, m], dim=1)
                    padded_mems.append(m)
                layer_mems = torch.cat(padded_mems, dim=0)  # [B, max_mem_len, d_model]
            
            batched_mems.append(layer_mems)
        
        # 4. 使用 TransformerXL 进行时序建模
        temporal_out, new_mems = self.model.temporal_model(
            fused_token,
            mems=batched_mems,
            causal_mask=True,
            return_mems=True,
        )  # temporal_out: [B, 1, token_dim], new_mems: List[Tensor]
        
        # 5. 动作头输出
        actions = self.model.action_head.forward_sequence(temporal_out)["mean"]  # [B, 1, action_dim]
        
        return actions, new_mems


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
        "--mem_len",
        type=int,
        default=64,
        help="TransformerXL memory length. Can be larger than training to attend to longer history.",
    )
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum steps to run.")

    # 录制验证数据相关参数
    parser.add_argument(
        "--record_data",
        action="store_true",
        help="Enable recording of (proprio, depth, mems, action) for env_id=0 for verification.",
    )
    parser.add_argument(
        "--record_output",
        type=str,
        default="/tmp/play_student_verify_data.bin",
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

    # 初始化相机全黑管理器
    blackout_manager = CameraBlackoutManager(
        num_envs=vec_env.num_envs,
        device=device,
    )
    print(f"[INFO] CameraBlackoutManager initialized: all {vec_env.num_envs} envs will have blackout camera.")

    # ========== 录制相关初始化 ==========
    record_data = getattr(args, "record_data", False)
    record_output = getattr(args, "record_output", "/tmp/play_student_verify_data.bin")
    recorded_frames: List[Dict[str, np.ndarray]] = []
    # mems_dim = num_layers * mem_len * token_dim
    mems_dim = runner.num_layers * runner.mem_len * runner.d_model
    if record_data:
        print(f"[INFO] Recording enabled. Will save to: {record_output}")
        print(f"[INFO] mems_dim = {mems_dim} (num_layers={runner.num_layers}, mem_len={runner.mem_len}, token_dim={runner.d_model})")
    # =====================================

    obs, extras = vec_env.get_observations()
    step = 0
    
    # Debug: 记录前 5 步的 actions
    DEBUG_STEPS = 5
    debug_actions_buffer = []
    
    while simulation_app.is_running() and step < args.max_steps:
        depth_image = extras["observations"].get("depth_camera")
        if depth_image is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用 TeacherCam 任务。")

        # 应用相机全黑效果
        depth_image = blackout_manager.update(depth_image)
        if step < 3:
            print(f"[DEBUG] depth_image stats: min={depth_image.min():.4f}, max={depth_image.max():.4f}, mean={depth_image.mean():.4f}")
        obs_prop = obs[:, :proprio_dim]

        # Mask privileged information (delta_yaw at indices 6-7)
        # Student policy should not have access to privileged direction info during inference
        obs_prop = obs_prop.clone()  # Clone to avoid modifying original obs
        obs_prop[:, 6:8] = 0  # Zero out delta_yaw and delta_next_yaw

        # 捕获推理前的 mems 状态 (用于录制)
        # 注意：TorchScript 模型期望固定大小的 mems [num_layers, mem_len, token_dim]
        # 但 PyTorch 运行时 mems 是动态增长的，开始时可能为空或长度小于 mem_len
        # 因此需要左侧零填充到完整的 mem_len
        pre_mems_0 = None

        # Debug: 检查 mems 是否为空
        if record_data and step < 5:
            if len(runner.mems) == 0:
                print(f"[DEBUG] Frame {step}: runner.mems is EMPTY (no environments initialized)")
            elif len(runner.mems[0]) == 0:
                print(f"[DEBUG] Frame {step}: runner.mems[0] is EMPTY (no layers)")
            else:
                print(f"[DEBUG] Frame {step}: runner.mems[0] has {len(runner.mems[0])} layers")

        if record_data and len(runner.mems) > 0:
            # Debug: 输出前 5 帧的 mems 状态
            if step < 5:
                mems_lengths = [m.size(1) for m in runner.mems[0]]
                print(f"[DEBUG] Frame {step}: mems[0] lengths = {mems_lengths} (expected: {runner.mem_len})")
                print(f"        num_layers = {len(runner.mems[0])}, d_model = {runner.d_model}")

            padded_mems = []
            for layer_idx, m in enumerate(runner.mems[0]):
                # m 的形状是 [1, current_len, d_model]，current_len 可能 < mem_len
                current_len = m.size(1)
                if current_len < runner.mem_len:
                    pad_len = runner.mem_len - current_len
                    padding = torch.zeros(1, pad_len, runner.d_model, device=runner.device)
                    m_padded = torch.cat([padding, m], dim=1)  # 左侧填充零

                    # Debug: 输出填充信息
                    if step < 5:
                        print(f"        Layer {layer_idx}: padded {pad_len} zeros (left), current_len={current_len}")
                else:
                    m_padded = m
                    if step < 5:
                        print(f"        Layer {layer_idx}: no padding needed, current_len={current_len}")

                padded_mems.append(m_padded.squeeze(0))  # [mem_len, d_model]
            pre_mems_0 = torch.stack(padded_mems, dim=0).cpu().numpy()  # [num_layers, mem_len, token_dim]

            # Debug: 输出填充后的统计信息
            if step < 5:
                print(f"        Final mems shape: {pre_mems_0.shape}")
                print(f"        Mems stats: min={pre_mems_0.min():.4f}, max={pre_mems_0.max():.4f}, mean={pre_mems_0.mean():.4f}")
                print(f"        Zero ratio: {(pre_mems_0 == 0).sum() / pre_mems_0.size * 100:.1f}%")

        student_action = runner.act(obs_prop, depth_image)

        # ========== 录制: 在推理后保存 env_id=0 的输入和输出 ==========
        if record_data:
            # 在 act() 之后读取更新后的历史 (包含当前帧)
            prop_hist_0 = runner.prop_histories[0]
            depth_hist_0 = runner.depth_histories[0]

            # 拼接历史帧构建输入
            prop_input = torch.cat(list(prop_hist_0), dim=0).cpu().numpy()  # [prop_hist_len * proprio_dim]
            depth_input = torch.stack(list(depth_hist_0), dim=0).cpu().numpy()  # [depth_hist_len, H, W]
            depth_flat = depth_input.flatten()  # [depth_hist_len * H * W]

            action_output = student_action[0].cpu().numpy()  # [action_dim]

            # 保存帧数据
            if pre_mems_0 is not None:
                mems_flat = pre_mems_0.flatten()  # [num_layers * mem_len * token_dim]
                recorded_frames.append({
                    "proprio": prop_input,
                    "depth": depth_flat,
                    "mems": mems_flat,
                    "action": action_output,
                })
        # =============================================================
        
        # Debug: 输出 env 0 前 5 步的 actions
        if step < DEBUG_STEPS:
            env0_action = student_action[0].detach().cpu().numpy()
            debug_actions_buffer.append(env0_action.copy())
            print(f"\n{'='*60}")
            print(f"[DEBUG] Step {step} - Env 0 Action (dim={len(env0_action)}):")
            print(f"  Action values: {env0_action}")
            print(f"  Action norm: {np.linalg.norm(env0_action):.6f}")
            print(f"  Action min: {env0_action.min():.6f}, max: {env0_action.max():.6f}")
            print(f"{'='*60}")
        
        # Debug: 在第 5 步后打印汇总
        if step == DEBUG_STEPS:
            print(f"\n{'#'*60}")
            print(f"[DEBUG SUMMARY] First {DEBUG_STEPS} steps actions for Env 0:")
            for i, act in enumerate(debug_actions_buffer):
                print(f"  Step {i}: norm={np.linalg.norm(act):.4f}, "
                      f"mean={act.mean():.4f}, std={act.std():.4f}")
            print(f"{'#'*60}\n")
        
        if step % 50 == 0:
            try:
                mean_norm = student_action.norm(dim=-1).mean().item()
            except Exception:
                mean_norm = float("nan")
            # print(f"[student_play] step={step} mean_action_norm={mean_norm:.6f}")
        obs_next, rews, dones, extras = vec_env.step(student_action)
        done_mask = dones.squeeze(-1).bool()
        if done_mask.any():
            runner.reset_done(done_mask)
            blackout_manager.reset_env(done_mask)

        obs = obs_next
        step += 1

    # ========== 保存录制数据 ==========
    if record_data and recorded_frames:
        _save_verification_data(
            record_output,
            recorded_frames,
            prop_dim=proprio_dim * args.prop_hist_len,
            depth_dim=camera_resolution[0] * camera_resolution[1] * args.depth_hist_len,
            action_dim=int(meta["action_dim"]),
            mems_dim=mems_dim,
        )
    elif record_data:
        print("[WARNING] Recording enabled but no frames were captured.")
    # =================================

    vec_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
