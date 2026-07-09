import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from transformerxl.student_policy import MultiModalStudentPolicy


# ==========================================
# 1. 模型构建与检查点管理
# ==========================================

def build_student_model(
    proprio_dim: int,
    action_dim: int,
    camera_resolution: Tuple[int, int],
    prop_hist_len: int,
    depth_hist_len: int,
    mem_len: int = 64,
    token_dim: int = 128,
) -> MultiModalStudentPolicy:
    """统一配置并构建学生模型，消除各文件中的硬编码配置。"""
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

    return MultiModalStudentPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
        token_dim=token_dim,
    )


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
    token_dim: int = 128,
) -> Tuple[MultiModalStudentPolicy, Dict[str, object]]:
    """Load a trained student policy and associated metadata for playback."""
    payload = torch.load(checkpoint_path, map_location=device)
    meta: Dict[str, object] = dict(payload.get("meta", {}))

    proprio_dim = int(meta["num_prop"])
    action_dim = int(meta["action_dim"])
    camera_resolution = tuple(meta.get("camera_resolution", [64, 64]))

    model = build_student_model(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution, # type: ignore
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        mem_len=mem_len,
        token_dim=token_dim
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta


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

# ==========================================
# 2. 在线推理/动作执行器 (DAgger / Play / Eval)
# ==========================================

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
        self.prop_hist = torch.zeros(
            num_envs, prop_hist_len, proprio_dim,
            dtype=torch.float32, device=device
        )
        self.depth_hist = torch.zeros(
            num_envs, depth_hist_len, *camera_resolution,
            dtype=torch.float32, device=device
        )
        self.dones_hist = torch.zeros(num_envs, self.model.temporal_model.mem_len, dtype=torch.bool, device=device)
        self.current_mem_len=0

        # TransformerXL Memory, Shape: List[Tensor], where each Tensor is [Num_Envs, Mem_Len, D_Model]
        self.mems: Optional[List[torch.Tensor]] = None

    def reset(self) -> None:
        """Reset all environments."""
        self.prop_hist.zero_()
        self.depth_hist.zero_()
        self.dones_hist.zero_()
        self.current_mem_len=0
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

        self.prop_hist[done_mask] = 0
        self.depth_hist[done_mask] = 0

        # if self.mems is not None:
        #     for i in range(len(self.mems)):
        #         self.mems[i][done_mask] = 0

    def act(
        self,
        obs_prop: torch.Tensor,     # [num_envs, proprio_dim]
        depth_image: torch.Tensor,  # [num_envs, H, W] or [num_envs, 1, H, W]
        prev_done
    ) -> torch.Tensor:
        """
        Perform one inference step using cached memory.
        """
        obs_prop = obs_prop.to(self.device)
        depth_image = depth_image.to(self.device)

        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)

        self.prop_hist = torch.roll(self.prop_hist, -1, dims=1)
        self.depth_hist = torch.roll(self.depth_hist, -1, dims=1)
        self.dones_hist = torch.roll(self.dones_hist, -1, dims=1)

        self.prop_hist[:, -1, :] = obs_prop
        self.depth_hist[:, -1, :, :] = depth_image
        self.dones_hist[:, -1] = prev_done

        # Prepare inputs for the model
        # 1. Flatten proprio history: [B, Hist, Dim] -> [B, Hist*Dim]
        # 2. Add Sequence dimension S=1: [B, S=1, Features]
        prop_input = self.prop_hist.view(self.num_envs, -1).unsqueeze(1)

        # Depth Input: [B, S=1, Hist, H, W]
        depth_input = self.depth_hist.unsqueeze(1)

        current_done = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        if self.current_mem_len > 0:
            actual_dones_hist = self.dones_hist[:, -self.current_mem_len:] 
        else:
            actual_dones_hist = torch.empty(self.num_envs, 0, dtype=torch.bool, device=self.device)
        full_dones = torch.cat([actual_dones_hist, current_done], dim=1)

        # [Answer 4] Direct Model Call
        with torch.no_grad():
            actions, _, self.mems = self.model.forward_with_mems(
                prop_input,
                depth_input,
                mems=self.mems,
                full_dones=full_dones
            )
        self.current_mem_len = min(self.current_mem_len + 1, self.model.temporal_model.mem_len)
        return actions.squeeze(1)  # Remove Sequence dim -> [B, Action_Dim]

    def act_rl(
        self,
        obs_prop: torch.Tensor,  # [N, proprio_dim]
        depth_image: torch.Tensor,  # [N, H, W]
        prev_done
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: action [N, A],  log_prob [N],  entropy [N]
        self.mems is updated in-place (no grad).
        """
        obs_prop = obs_prop.to(self.device)
        depth_image = depth_image.to(self.device)

        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)

        self.prop_hist = torch.roll(self.prop_hist, -1, dims=1)
        self.depth_hist = torch.roll(self.depth_hist, -1, dims=1)
        self.dones_hist = torch.roll(self.dones_hist, -1, dims=1)

        self.prop_hist[:, -1, :] = obs_prop
        self.depth_hist[:, -1, :, :] = depth_image
        self.dones_hist[:, -1] = prev_done

        prop_input = self.prop_hist.view(self.num_envs, -1).unsqueeze(1)
        depth_input = self.depth_hist.unsqueeze(1)

        current_done = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        if self.current_mem_len > 0:
            actual_dones_hist = self.dones_hist[:, -self.current_mem_len:] 
        else:
            actual_dones_hist = torch.empty(self.num_envs, 0, dtype=torch.bool, device=self.device)
        full_dones = torch.cat([actual_dones_hist, current_done], dim=1)

        with torch.no_grad():
            actions, log_probs, entropy, _, self.mems = self.model.forward_with_mems_rl(
                prop_input, depth_input, old_actions=None, mems=self.mems, full_dones=full_dones
            )
        self.current_mem_len = min(self.current_mem_len + 1, self.model.temporal_model.mem_len)
        return actions.squeeze(1), log_probs.squeeze(1), entropy.squeeze(1)


# ==========================================
# 3. 序列数据收集器 (Offline BC / DAgger)
# ==========================================

class SequenceAggregator:
    """
    prop_hist和depth_hist用于聚合fusion transformer需要的聚合历史输入
    我对prop_hist和depth_hist初始化时进行了填零操作, 这样第一个环境步transformerxl就可以输出action
    seq_prop[i]存储了第i个环境的感知观测序列
    """
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        prop_hist_len: int,
        depth_hist_len: int,
        sequence_len: int,
        num_prop: int = 53,
        depth_shape: Tuple[int, int] = (58, 87),
        extra_info_dim: int = 2,
    ) -> None:
        self.num_envs = num_envs
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.sequence_len = sequence_len
        self.num_prop = num_prop
        self.device = device

        # 1. History Buffer: 行为类似deque, 不过现在使用np.roll实现
        self.prop_hist = torch.zeros((num_envs, prop_hist_len, num_prop), dtype=torch.float32, device=self.device)
        self.depth_hist = torch.zeros((num_envs, depth_hist_len, *depth_shape), dtype=torch.float32, device=self.device)

        # 2. Sequence Buffer: 预分配内存，用于存储一个完整的 Sequence Batch
        self.seq_prop = torch.zeros((num_envs, sequence_len, prop_hist_len * num_prop), dtype=torch.float32, device=self.device)
        self.seq_depth = torch.zeros((num_envs, sequence_len, depth_hist_len, *depth_shape), dtype=torch.float32, device=self.device)

        self.seq_action = None
        self.seq_done = torch.zeros((num_envs, sequence_len), dtype=torch.bool, device=self.device)
        self.seq_info = torch.zeros((num_envs, sequence_len, extra_info_dim), dtype=torch.float32, device=self.device)
        self.seq_valid = torch.ones((num_envs, sequence_len), dtype=torch.bool, device=self.device)

        self.current_seq_step = 0

    def reset(self) -> None:
        self.prop_hist.zero_()
        self.depth_hist.zero_()

        self.seq_prop.zero_()
        self.seq_depth.zero_()
        self.seq_action = None
        self.seq_done.zero_()
        self.seq_info.zero_()
        self.seq_valid.fill_(True)

        self.current_seq_step = 0

    def push_step(
        self,
        obs_prop: torch.Tensor,
        depth_frame: torch.Tensor,
        teacher_actions: torch.Tensor,
        done: torch.Tensor,
        extra_info: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """
        接收 Tensor 数据（需保证已在正确的 device 上），更新历史 buffer 和 sequence buffer。
        """
        if valid_mask is None:
            valid_mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            valid_mask = valid_mask.to(device=self.device, dtype=torch.bool)

        # --- 1. 更新历史 (整体左移) ---
        self.prop_hist = torch.roll(self.prop_hist, -1, dims=1)
        self.depth_hist = torch.roll(self.depth_hist, -1, dims=1)

        # 填入最新数据
        self.prop_hist[:, -1, :] = obs_prop
        self.depth_hist[:, -1, :, :] = depth_frame

        # --- 2. 填入 Sequence Buffer ---
        idx = self.current_seq_step

        # Lazy Init for actions
        if self.seq_action is None:
            action_dim = teacher_actions.shape[-1]
            self.seq_action = torch.zeros((self.num_envs, self.sequence_len, action_dim), dtype=torch.float32, device=self.device)

        # Flatten Proprio: [Num_Envs, Hist_Len, num_prop] -> [Num_Envs, Hist_Len * num_prop]
        current_prop_flat = self.prop_hist.reshape(self.num_envs, -1)

        self.seq_prop[:, idx] = current_prop_flat
        self.seq_depth[:, idx] = self.depth_hist
        self.seq_action[:, idx] = teacher_actions
        self.seq_done[:, idx] = done
        self.seq_info[:, idx] = extra_info
        self.seq_valid[:, idx] = valid_mask

        # --- 3. 处理 Done (批量清零) ---
        if done.any():
            """ 对于done掉的环境, 不能将它们的seq_buffer置0,因为transformer网络在训练时是sequencely output的!!! """
            self.prop_hist[done] = 0.0
            self.depth_hist[done] = 0.0

        # --- 4. 检查 Batch 是否完成 ---
        self.current_seq_step += 1
        if self.current_seq_step == self.sequence_len:
            batch = self._pack_batch()
            self.current_seq_step = 0
            return batch

        return None

    def _pack_batch(self):
        # 返回 Tensor 副本，防止下一轮循环修改 buffer 影响 dataloader 队列
        return {
            "proprio": self.seq_prop.clone(),
            "depth": self.seq_depth.clone(),
            "actions": self.seq_action.clone(),
            "dones": self.seq_done.clone(),
            "extra_infos": self.seq_info.clone(),
            "valid_mask": self.seq_valid.clone(),
        }


class ResetSettleManager:
    """Tracks reset settling as TXL-only pseudo episodes."""

    def __init__(self, num_envs: int, settle_steps: int, device: torch.device) -> None:
        self.settle_steps = max(int(settle_steps), 0)
        self.remaining = torch.full(
            (num_envs,),
            self.settle_steps,
            dtype=torch.long,
            device=device,
        )
        self.prev_txl_done = self.remaining > 0

    def active_mask(self) -> torch.Tensor:
        return self.remaining > 0

    def step(self, dones_bool: torch.Tensor, txl_done: torch.Tensor) -> None:
        settling = self.active_mask()
        if settling.any():
            self.remaining[settling] = torch.clamp(self.remaining[settling] - 1, min=0)

        if dones_bool.any():
            self.remaining[dones_bool] = self.settle_steps

        self.prev_txl_done = txl_done.detach().clone()