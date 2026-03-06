#!/usr/bin/env python3
"""
Phase 5: Student RL Fine-tuning 训练脚本

对 DAgger 训练的 Student Policy 进行 PPO 强化学习微调。
通过 Domain Randomization 增强策略的鲁棒性。

主要功能：
1. 加载 DAgger 预训练的 Student Policy
2. 创建 StudentActorCritic（添加 ValueHead）
3. 初始化 PPOStudent 算法
4. 集成 Domain Randomization
5. 实现训练循环
6. 日志记录和检查点保存
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

# 确保项目路径可导入
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "parkour_isaaclab"))
sys.path.insert(0, str(PROJECT_ROOT / "parkour_tasks"))
# 确保 scripts/rsl_rl 在最前面，避免与 parkour_isaaclab/utils.py 冲突
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))

# 导入核心模块
from modules.student_actor_critic import StudentActorCritic
from modules.ppo_student import PPOStudent
from modules.student_rollout_storage import StudentRolloutStorage

# 尝试导入 Domain Randomization（可能需要 Isaac Lab 环境）
try:
    from parkour_isaaclab.envs.mdp.domain_randomization import (
        DepthNoiseAugmentation,
        LatencySimulation,
        LightingAugmentation,
        DomainRandCurriculum,
        CameraDropoutManagerWithCurriculum,
    )
    DOMAIN_RAND_AVAILABLE = True
except ImportError:
    # 如果无法导入，使用本地实现
    DOMAIN_RAND_AVAILABLE = False
    print("[WARNING] Could not import domain_randomization from parkour_isaaclab.")
    print("[WARNING] Using local implementation for testing.")

    # 本地实现的 Domain Randomization 类（用于测试）
    from collections import deque

    class DepthNoiseAugmentation:
        """本地深度噪声增强实现"""
        def __init__(self, cfg):
            self.gaussian_std = getattr(cfg, 'gaussian_std', 0.02)
            self.salt_pepper_prob = getattr(cfg, 'salt_pepper_prob', 0.01)
            self.missing_pixel_prob = getattr(cfg, 'missing_pixel_prob', 0.01)
            self.quantization_levels = getattr(cfg, 'quantization_levels', 0)
            self.scale_range = getattr(cfg, 'scale_range', (0.95, 1.05))

        def apply(self, depth: Tensor) -> Tensor:
            augmented = depth.clone()
            if self.gaussian_std > 0:
                noise = torch.randn_like(augmented) * self.gaussian_std
                augmented = augmented + noise
            return augmented

    class LatencySimulation:
        """本地延迟模拟实现"""
        def __init__(self, depth_delay_frames: int = 3, drop_prob: float = 0.0):
            self.depth_delay = depth_delay_frames
            self.drop_prob = drop_prob
            self.depth_buffer = deque(maxlen=depth_delay_frames + 1)

        def reset(self):
            self.depth_buffer.clear()

        def apply(self, depth: Tensor) -> Tensor:
            self.depth_buffer.append(depth.clone())
            if len(self.depth_buffer) <= self.depth_delay:
                return self.depth_buffer[0].clone()
            return self.depth_buffer[0].clone()

    class LightingAugmentation:
        """本地光照增强实现"""
        def __init__(self, cfg):
            self.brightness_range = getattr(cfg, 'brightness_range', (0.9, 1.1))
            self.contrast_range = getattr(cfg, 'contrast_range', (0.9, 1.1))

        def apply(self, depth: Tensor) -> Tensor:
            augmented = depth.clone()
            if self.brightness_range != (1.0, 1.0):
                brightness_factor = torch.empty(1).uniform_(
                    self.brightness_range[0], self.brightness_range[1]
                ).item()
                augmented = augmented * brightness_factor
            return augmented

    class DomainRandCurriculum:
        """本地课程学习调度器实现"""
        SCHEDULE = [
            (0,    0.01, 0.005, 0.1, 1),
            (1000, 0.02, 0.01,  0.2, 2),
            (3000, 0.03, 0.015, 0.3, 3),
            (5000, 0.04, 0.02,  0.5, 3),
        ]

        def get_params(self, iteration: int) -> Dict[str, float]:
            for i in range(len(self.SCHEDULE) - 1):
                iter_start, noise_start, salt_start, dropout_start, latency_start = self.SCHEDULE[i]
                iter_end, noise_end, salt_end, dropout_end, latency_end = self.SCHEDULE[i + 1]
                if iter_start <= iteration < iter_end:
                    alpha = (iteration - iter_start) / (iter_end - iter_start)
                    return {
                        'noise_std': noise_start + (noise_end - noise_start) * alpha,
                        'salt_pepper': salt_start + (salt_end - salt_start) * alpha,
                        'dropout': dropout_start + (dropout_end - dropout_start) * alpha,
                        'latency': int(latency_start + (latency_end - latency_start) * alpha),
                    }
            _, noise, salt, dropout, latency = self.SCHEDULE[-1]
            return {'noise_std': noise, 'salt_pepper': salt, 'dropout': dropout, 'latency': latency}

    class CameraDropoutManagerWithCurriculum:
        """本地相机丢失管理器实现"""
        def __init__(self, num_envs, device, **kwargs):
            self.num_envs = num_envs
            self.device = device


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置数据类"""

    # 环境配置
    num_envs: int = 256
    num_steps_per_env: int = 64  # 匹配 TXL sequence length

    # 训练配置
    max_iterations: int = 10000
    save_interval: int = 100
    log_interval: int = 10

    # PPO 超参数（针对 DAgger 模型微调优化）
    learning_rate: float = 3e-5  # 降低学习率，避免破坏预训练权重
    clip_param: float = 0.1  # 更保守的 clip，限制策略更新幅度
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.001  # 降低熵系数，避免鼓励增加噪声
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5  # 更严格的梯度裁剪
    num_learning_epochs: int = 3  # 减少 epoch 数，避免过度更新
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    desired_kl: float = 0.01  # 单维度平均 KL 目标，与原生 PPO 一致

    # 编码器冻结策略
    freeze_proprio_encoder: bool = False
    freeze_depth_encoder: bool = False
    freeze_fusion_transformer: bool = False
    freeze_temporal_transformer: bool = False

    # Domain Randomization
    domain_rand_enabled: bool = True
    domain_rand_curriculum: bool = True

    # 模型配置
    proprio_dim: int = 53
    action_dim: int = 12
    depth_shape: Tuple[int, int, int] = (4, 58, 87)
    token_dim: int = 128

    # 路径配置
    dagger_checkpoint: str = ""
    log_dir: str = ""
    experiment_name: str = "student_finetune"

    # 设备
    device: str = "cuda"

    # TensorBoard 配置（替代 wandb）
    use_tensorboard: bool = True
    tensorboard_flush_secs: int = 10

    def validate(self) -> None:
        """验证配置参数"""
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.clip_param < 0:
            raise ValueError(f"clip_param must be non-negative, got {self.clip_param}")
        if not (0 <= self.gamma <= 1):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not (0 <= self.lam <= 1):
            raise ValueError(f"lam must be in [0, 1], got {self.lam}")


def parse_training_config(**kwargs) -> TrainingConfig:
    """解析训练配置

    Args:
        **kwargs: 配置参数覆盖

    Returns:
        TrainingConfig: 验证后的配置对象
    """
    config = TrainingConfig(**kwargs)
    config.validate()
    return config


# ============================================================
# Domain Randomization 包装器
# ============================================================

@dataclass
class DepthNoiseCfg:
    """深度噪声配置"""
    gaussian_std: float = 0.02
    salt_pepper_prob: float = 0.01
    missing_pixel_prob: float = 0.01
    quantization_levels: int = 0
    scale_range: Tuple[float, float] = (0.95, 1.05)


@dataclass
class LightingCfg:
    """光照增强配置"""
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)


class DomainRandomizationWrapper:
    """Domain Randomization 包装器

    整合所有增强模块，提供统一接口。
    """

    def __init__(
        self,
        enabled: bool = True,
        use_curriculum: bool = True,
        num_envs: int = 256,
        device: str = "cuda",
    ) -> None:
        self.enabled = enabled
        self.use_curriculum = use_curriculum
        self.num_envs = num_envs
        self.device = device

        # 初始化课程学习调度器
        self.curriculum = DomainRandCurriculum() if use_curriculum else None

        # 当前参数
        self._current_params = {
            "noise_std": 0.01,
            "salt_pepper": 0.005,
            "dropout": 0.1,
            "latency": 1,
        }

        # 初始化增强模块
        self._init_augmentation_modules()

    def _init_augmentation_modules(self) -> None:
        """初始化增强模块"""
        # 深度噪声
        noise_cfg = DepthNoiseCfg(
            gaussian_std=self._current_params["noise_std"],
            salt_pepper_prob=self._current_params["salt_pepper"],
        )
        self.depth_noise = DepthNoiseAugmentation(noise_cfg)

        # 光照增强
        lighting_cfg = LightingCfg()
        self.lighting = LightingAugmentation(lighting_cfg)

        # 延迟模拟
        self.latency = LatencySimulation(
            depth_delay_frames=self._current_params["latency"]
        )

    def update_curriculum(self, iteration: int) -> None:
        """更新课程学习参数

        Args:
            iteration: 当前训练迭代次数
        """
        if not self.use_curriculum or self.curriculum is None:
            return

        self._current_params = self.curriculum.get_params(iteration)

        # 更新增强模块参数
        self.depth_noise.gaussian_std = self._current_params["noise_std"]
        self.depth_noise.salt_pepper_prob = self._current_params["salt_pepper"]
        self.latency.depth_delay = self._current_params["latency"]

    def get_current_params(self) -> Dict[str, float]:
        """获取当前增强参数"""
        return self._current_params.copy()

    def apply_augmentation(self, depth: Tensor) -> Tensor:
        """应用深度图像增强

        Args:
            depth: 深度图像 [B, depth_hist_len, H, W]

        Returns:
            增强后的深度图像
        """
        if not self.enabled:
            return depth

        # 应用深度噪声
        augmented = self.depth_noise.apply(depth)

        # 应用光照增强
        augmented = self.lighting.apply(augmented)

        return augmented


def create_domain_randomization(
    enabled: bool = True,
    use_curriculum: bool = True,
    num_envs: int = 256,
    device: str = "cuda",
) -> DomainRandomizationWrapper:
    """创建 Domain Randomization 模块

    Args:
        enabled: 是否启用
        use_curriculum: 是否使用课程学习
        num_envs: 环境数量
        device: 设备

    Returns:
        DomainRandomizationWrapper 实例
    """
    return DomainRandomizationWrapper(
        enabled=enabled,
        use_curriculum=use_curriculum,
        num_envs=num_envs,
        device=device,
    )


# ============================================================
# TensorBoard 集成（替代 wandb）
# ============================================================

# 导入 TensorBoard 日志工具
from utils.tensorboard_logger import TensorBoardLogger


def init_tensorboard(
    config: TrainingConfig,
    checkpoint_meta: Dict[str, Any],
    run_dir: Path,
) -> Optional[TensorBoardLogger]:
    """初始化 TensorBoard 日志器

    Args:
        config: 训练配置
        checkpoint_meta: checkpoint 元数据
        run_dir: 运行目录

    Returns:
        TensorBoardLogger 对象，如果未启用则返回 None
    """
    if not config.use_tensorboard:
        return None

    try:
        # TensorBoard 日志目录
        tb_log_dir = run_dir / "tensorboard"

        # 初始化日志器
        logger = TensorBoardLogger(
            log_dir=str(tb_log_dir),
            flush_secs=config.tensorboard_flush_secs,
        )

        # 记录配置
        tb_config = {
            # 训练配置
            "num_envs": config.num_envs,
            "num_steps_per_env": config.num_steps_per_env,
            "max_iterations": config.max_iterations,

            # PPO 超参数
            "learning_rate": config.learning_rate,
            "clip_param": config.clip_param,
            "gamma": config.gamma,
            "lam": config.lam,
            "entropy_coef": config.entropy_coef,
            "value_loss_coef": config.value_loss_coef,
            "max_grad_norm": config.max_grad_norm,
            "num_learning_epochs": config.num_learning_epochs,
            "num_mini_batches": config.num_mini_batches,
            "schedule": config.schedule,
            "desired_kl": config.desired_kl,

            # 编码器冻结
            "freeze_proprio_encoder": config.freeze_proprio_encoder,
            "freeze_depth_encoder": config.freeze_depth_encoder,
            "freeze_fusion_transformer": config.freeze_fusion_transformer,
            "freeze_temporal_transformer": config.freeze_temporal_transformer,

            # Domain Randomization
            "domain_rand_enabled": config.domain_rand_enabled,
            "domain_rand_curriculum": config.domain_rand_curriculum,

            # 模型配置（从 checkpoint 元数据）
            "proprio_dim": checkpoint_meta.get("num_prop", config.proprio_dim),
            "action_dim": checkpoint_meta.get("action_dim", config.action_dim),
            "depth_shape": config.depth_shape,
            "token_dim": config.token_dim,

            # 路径
            "dagger_checkpoint": config.dagger_checkpoint,
        }
        logger.log_config(tb_config)

        return logger
    except Exception as e:
        print(f"[tensorboard] Warning: Failed to initialize TensorBoard: {e}")
        print("[tensorboard] Training will continue without TensorBoard logging")
        return None


def log_training_metrics(
    logger: Optional[TensorBoardLogger],
    iteration: int,
    train_info: Dict[str, float],
    domain_rand_params: Optional[Dict[str, float]] = None,
    episode_stats: Optional[Dict[str, float]] = None,
) -> None:
    """记录训练指标到 TensorBoard

    Args:
        logger: TensorBoardLogger 对象
        iteration: 当前迭代次数
        train_info: PPO 更新返回的训练信息
        domain_rand_params: Domain Randomization 当前参数
        episode_stats: Episode 统计信息
    """
    if logger is None or not logger.enabled:
        return

    try:
        metrics = {
            # PPO 损失
            "train/value_loss": train_info.get("value_loss", 0.0),
            "train/policy_loss": train_info.get("surrogate_loss", 0.0),
            "train/entropy": train_info.get("entropy", 0.0),
            "train/kl_divergence": train_info.get("kl", 0.0),
            "train/learning_rate": train_info.get("learning_rate", 0.0),

            # 迭代
            "iteration": iteration,
        }

        # 添加 Domain Randomization 参数
        if domain_rand_params:
            for key, value in domain_rand_params.items():
                metrics[f"domain_rand/{key}"] = value

        # 添加 Episode 统计
        if episode_stats:
            for key, value in episode_stats.items():
                metrics[f"episode/{key}"] = value

        # 记录日志
        logger.log(metrics, step=iteration)
    except Exception as e:
        # 记录错误但不中断训练
        print(f"[tensorboard] Warning: Failed to log metrics: {e}")


# ============================================================
# Student Policy 加载
# ============================================================

def load_student_policy_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """从 DAgger checkpoint 加载 Student Policy

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 设备

    Returns:
        student_policy: 加载的 Student Policy
        meta: checkpoint 元数据
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 获取元数据
    meta = checkpoint.get("meta", {})
    proprio_dim = int(meta.get("num_prop", 53))
    action_dim = int(meta.get("action_dim", 12))
    camera_resolution = tuple(meta.get("camera_resolution", [58, 87]))
    prop_hist_len = int(meta.get("prop_hist_len", 1))
    depth_hist_len = int(meta.get("depth_hist_len", 1))

    # 动态加载 MultiModalStudentPolicy
    MODULES_ROOT = PROJECT_ROOT / "parkour_tasks" / "parkour_tasks" / "extreme_parkour_task" / "modules"

    def load_symbol(module_path: Path, symbol: str):
        spec = importlib.util.spec_from_file_location(f"student_policy.{symbol}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, symbol)

    # 加载必要的类
    JointPoseActionHead = load_symbol(MODULES_ROOT / "actionheads" / "joint_action_head.py", "JointPoseActionHead")
    MultiModalFusionTransformer = load_symbol(
        MODULES_ROOT / "encoders" / "fusion_transformer.py", "MultiModalFusionTransformer"
    )
    TransformerXLTemporal = load_symbol(MODULES_ROOT / "temperal" / "txl.py", "TransformerXLTemporal")
    DepthEncoder = load_symbol(MODULES_ROOT / "tokenizers" / "depth_encoder.py", "DepthEncoder")
    ProprioEncoder = load_symbol(MODULES_ROOT / "tokenizers" / "proprio_encoder.py", "ProprioEncoder")

    # 从 train_student_from_dataset.py 导入 MultiModalStudentPolicy
    train_student_path = PROJECT_ROOT / "scripts" / "rsl_rl" / "train_student_from_dataset.py"
    spec = importlib.util.spec_from_file_location("train_student", train_student_path)
    train_student_module = importlib.util.module_from_spec(spec)

    # 手动设置必要的符号
    train_student_module.JointPoseActionHead = JointPoseActionHead
    train_student_module.MultiModalFusionTransformer = MultiModalFusionTransformer
    train_student_module.TransformerXLTemporal = TransformerXLTemporal
    train_student_module.DepthEncoder = DepthEncoder
    train_student_module.ProprioEncoder = ProprioEncoder

    spec.loader.exec_module(train_student_module)
    MultiModalStudentPolicy = train_student_module.MultiModalStudentPolicy

    # 创建模型
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
        "action_scale": 1,
    }

    student_policy = MultiModalStudentPolicy(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        camera_resolution=camera_resolution,
        prop_hist_len=prop_hist_len,
        depth_hist_len=depth_hist_len,
        fusion_cfg=fusion_cfg,
        temporal_cfg=temporal_cfg,
        action_head_cfg=action_head_cfg,
        token_dim=128,
    )

    # 加载权重
    if "model_state_dict" in checkpoint:
        student_policy.load_state_dict(checkpoint["model_state_dict"])

    student_policy.to(device)
    return student_policy, meta


def create_student_actor_critic(
    checkpoint_path: str,
    freeze_encoders: bool = False,
    freeze_fusion: bool = False,
    freeze_temporal: bool = False,
    device: str = "cuda",
) -> Tuple[StudentActorCritic, Dict[str, Any]]:
    """创建 StudentActorCritic

    Args:
        checkpoint_path: DAgger checkpoint 路径
        freeze_encoders: 是否冻结编码器
        freeze_fusion: 是否冻结融合 transformer
        freeze_temporal: 是否冻结时序模型
        device: 设备

    Returns:
        Tuple of (StudentActorCritic 实例, checkpoint 元数据)
    """
    # 加载 Student Policy
    student_policy, meta = load_student_policy_from_checkpoint(checkpoint_path, device)

    # 创建 StudentActorCritic
    actor_critic = StudentActorCritic(
        student_policy=student_policy,
        value_hidden_dims=(256, 256),
        init_noise_std=0.5,  # 平衡探索和稳定性，避免熵为负数
        freeze_encoders=freeze_encoders,
        freeze_fusion=freeze_fusion,
        freeze_temporal=freeze_temporal,
    )

    actor_critic.to(device)
    return actor_critic, meta


# ============================================================
# PPO 初始化
# ============================================================

def initialize_ppo_student(
    actor_critic: StudentActorCritic,
    learning_rate: float = 1e-4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 0.01,
    value_loss_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    device: str = "cuda",
) -> PPOStudent:
    """初始化 PPOStudent 算法

    Args:
        actor_critic: StudentActorCritic 模型
        learning_rate: 学习率
        clip_param: PPO clip 参数
        gamma: 折扣因子
        lam: GAE lambda
        entropy_coef: 熵系数
        value_loss_coef: 价值损失系数
        max_grad_norm: 梯度裁剪
        num_learning_epochs: 学习 epoch 数
        num_mini_batches: mini-batch 数
        schedule: 学习率调度
        desired_kl: 期望 KL 散度
        device: 设备

    Returns:
        PPOStudent 实例
    """
    ppo = PPOStudent(
        actor_critic=actor_critic,
        num_learning_epochs=num_learning_epochs,
        num_mini_batches=num_mini_batches,
        clip_param=clip_param,
        gamma=gamma,
        lam=lam,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        use_clipped_value_loss=True,
        schedule=schedule,
        desired_kl=desired_kl,
        device=device,
    )

    return ppo


# ============================================================
# 训练循环函数
# ============================================================

def collect_rollouts(
    env: Any,
    ppo: PPOStudent,
    domain_rand: Optional[DomainRandomizationWrapper],
    num_steps: int,
    num_prop: int,
) -> Dict[str, Any]:
    """收集 rollout 数据并追踪 episode 统计

    Args:
        env: 环境（ParkourRslRlVecEnvWrapper）
        ppo: PPOStudent 算法
        domain_rand: Domain Randomization 模块
        num_steps: 收集步数
        num_prop: proprio 维度

    Returns:
        字典包含：
        - "proprio": 最后一步的 proprio 观测
        - "depth": 最后一步的 depth 观测
        - "episode_stats": episode 统计信息
            - "mean_return": 完成的 episode 的平均总奖励
            - "mean_length": 完成的 episode 的平均长度
            - "count": 完成的 episode 数量
            - "rollout_mean_reward": rollout 期间的平均 step reward
    """
    # 获取初始观测
    obs_tensor, extras = env.get_observations()
    num_envs = obs_tensor.shape[0]
    device = obs_tensor.device

    # Episode 追踪状态
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int)

    # 完成的 episode 统计
    completed_returns: List[float] = []
    completed_lengths: List[int] = []

    # Rollout 统计（保留原有功能）
    step_rewards: List[float] = []

    for _ in range(num_steps):
        # 提取 proprio 和 depth
        proprio = obs_tensor[:, :num_prop]
        depth = extras["observations"]["depth_camera"]

        # 添加 depth_hist_len 维度：[B, H, W] -> [B, 1, H, W]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)

        # 应用 Domain Randomization
        if domain_rand is not None:
            depth = domain_rand.apply_augmentation(depth)

        # 采样动作
        actions = ppo.act(proprio, depth)

        # 执行动作
        obs_tensor, rewards, dones, infos = env.step(actions)

        # 处理 rewards 和 dones 的维度（可能是 [N] 或 [N, 1]）
        rewards_flat = rewards.squeeze(-1) if rewards.dim() == 2 else rewards
        dones_flat = dones.squeeze(-1) if dones.dim() == 2 else dones

        # 累积 episode 奖励和长度
        episode_rewards += rewards_flat
        episode_lengths += 1

        # Rollout 统计
        step_rewards.append(rewards_flat.mean().item())

        # 检查完成的 episode
        done_indices = dones_flat.nonzero(as_tuple=True)[0]
        if len(done_indices) > 0:
            # 记录完成的 episode
            completed_returns.extend(episode_rewards[done_indices].cpu().tolist())
            completed_lengths.extend(episode_lengths[done_indices].cpu().tolist())

            # 重置完成的 episode
            episode_rewards[done_indices] = 0
            episode_lengths[done_indices] = 0

        # 更新 extras
        if "observations" in infos:
            extras = infos

        # 处理环境步骤（PPO storage 和 TXL memory reset）
        ppo.process_env_step(rewards, dones, infos)

    # 计算统计
    rollout_mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0

    if completed_returns:
        episode_stats = {
            "mean_return": sum(completed_returns) / len(completed_returns),
            "mean_length": sum(completed_lengths) / len(completed_lengths),
            "count": len(completed_returns),
            "rollout_mean_reward": rollout_mean_reward,
        }
    else:
        # 没有完成的 episode（可能 episode 很长）
        episode_stats = {
            "mean_return": 0.0,
            "mean_length": 0.0,
            "count": 0,
            "rollout_mean_reward": rollout_mean_reward,
        }

    # 返回最后一步的观测
    final_depth = extras["observations"]["depth_camera"]
    if final_depth.dim() == 3:
        final_depth = final_depth.unsqueeze(1)

    return {
        "proprio": obs_tensor[:, :num_prop],
        "depth": final_depth,
        "episode_stats": episode_stats,
    }


def compute_returns_and_update(
    ppo: PPOStudent,
    actor_critic: StudentActorCritic,
    last_obs: Dict[str, Tensor],
) -> Dict[str, float]:
    """计算 returns 并执行 PPO 更新

    Args:
        ppo: PPOStudent 算法
        actor_critic: StudentActorCritic 模型
        last_obs: 最后一步观测

    Returns:
        训练信息字典
    """
    # 计算最后一步的价值
    with torch.no_grad():
        last_values = actor_critic.evaluate(
            last_obs["proprio"],
            last_obs["depth"],
        )

    # 计算 returns
    ppo.compute_returns(last_values)

    # PPO 更新
    train_info = ppo.update()

    return train_info


def run_training_iteration(
    env: Any,
    ppo: PPOStudent,
    actor_critic: StudentActorCritic,
    domain_rand: Optional[DomainRandomizationWrapper],
    num_steps: int,
    iteration: int,
    num_prop: int,
) -> Dict[str, Any]:
    """运行单次训练迭代

    Args:
        env: 环境
        ppo: PPOStudent 算法
        actor_critic: StudentActorCritic 模型
        domain_rand: Domain Randomization 模块
        num_steps: 每次迭代的步数
        iteration: 当前迭代次数
        num_prop: proprio 维度

    Returns:
        训练信息字典，包含 PPO 损失和 episode 统计
    """
    # 更新 Domain Randomization 课程
    if domain_rand is not None:
        domain_rand.update_curriculum(iteration)

    # 收集 rollouts
    rollout_result = collect_rollouts(env, ppo, domain_rand, num_steps, num_prop)

    # 提取 episode 统计
    episode_stats = rollout_result.get("episode_stats", {})

    # 准备 last_obs 用于 compute_returns_and_update
    last_obs = {
        "proprio": rollout_result["proprio"],
        "depth": rollout_result["depth"],
    }

    # 计算 returns 并更新
    train_info = compute_returns_and_update(ppo, actor_critic, last_obs)

    # 添加 episode 统计到训练信息
    train_info["episode_stats"] = episode_stats

    return train_info


# ============================================================
# 检查点保存和加载
# ============================================================

def save_training_checkpoint(
    path: Path,
    actor_critic: StudentActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: Dict[str, Any],
) -> None:
    """保存训练检查点

    Args:
        path: 保存路径
        actor_critic: StudentActorCritic 模型
        optimizer: 优化器
        iteration: 当前迭代次数
        config: 配置字典
    """
    checkpoint = {
        "actor_critic_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"[checkpoint] Saved to {path}")


def load_training_checkpoint(
    path: Path,
    actor_critic: StudentActorCritic,
    optimizer: torch.optim.Optimizer,
) -> int:
    """加载训练检查点

    Args:
        path: 检查点路径
        actor_critic: StudentActorCritic 模型
        optimizer: 优化器

    Returns:
        恢复的迭代次数
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint.get("iteration", 0)

    print(f"[checkpoint] Loaded from {path} (iteration={iteration})")
    return iteration


# ============================================================
# 命令行参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Student RL Fine-tuning: PPO 微调 DAgger 训练的 Student Policy"
    )

    # 必需参数
    parser.add_argument(
        "--dagger_checkpoint",
        type=str,
        required=True,
        help="DAgger 训练的 Student Policy checkpoint 路径",
    )

    # 环境配置
    parser.add_argument("--num_envs", type=int, default=256, help="环境数量")
    parser.add_argument("--num_steps_per_env", type=int, default=64, help="每环境步数")
    parser.add_argument("--max_iterations", type=int, default=10000, help="最大迭代次数")

    # PPO 超参数
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip 参数")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵系数")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="价值损失系数")

    # 编码器冻结
    parser.add_argument("--freeze_encoders", action="store_true", help="冻结编码器")
    parser.add_argument("--freeze_fusion", action="store_true", help="冻结融合 transformer")
    parser.add_argument("--freeze_temporal", action="store_true", help="冻结时序模型")

    # Domain Randomization
    parser.add_argument("--no_domain_rand", action="store_true", help="禁用 Domain Randomization")
    parser.add_argument("--no_curriculum", action="store_true", help="禁用课程学习")

    # 日志和检查点
    parser.add_argument("--log_dir", type=str, default="logs/student_finetune", help="日志目录")
    parser.add_argument("--experiment_name", type=str, default="student_finetune", help="实验名称")
    parser.add_argument("--save_interval", type=int, default=100, help="保存间隔")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")

    # 恢复训练
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")

    # 设备
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> TrainingConfig:
    """将命令行参数转换为配置对象"""
    return TrainingConfig(
        num_envs=args.num_envs,
        num_steps_per_env=args.num_steps_per_env,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        clip_param=args.clip_param,
        gamma=args.gamma,
        lam=args.lam,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        freeze_proprio_encoder=args.freeze_encoders,
        freeze_depth_encoder=args.freeze_encoders,
        freeze_fusion_transformer=args.freeze_fusion,
        freeze_temporal_transformer=args.freeze_temporal,
        domain_rand_enabled=not args.no_domain_rand,
        domain_rand_curriculum=not args.no_curriculum,
        dagger_checkpoint=args.dagger_checkpoint,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=args.device,
    )


# ============================================================
# 主训练函数（不依赖 Isaac Lab 环境）
# ============================================================

def run_training_loop(
    env: Any,
    config: TrainingConfig,
    resume_path: Optional[str] = None,
) -> None:
    """运行训练循环

    Args:
        env: Isaac Lab 环境
        config: 训练配置
        resume_path: 恢复训练的检查点路径
    """
    # 验证配置
    config.validate()

    # 设置日志目录
    log_dir = Path(config.log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = log_dir / config.experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Logging to: {run_dir}")

    # 创建 StudentActorCritic
    print(f"[INFO] Loading DAgger checkpoint: {config.dagger_checkpoint}")
    actor_critic, checkpoint_meta = create_student_actor_critic(
        checkpoint_path=config.dagger_checkpoint,
        freeze_encoders=config.freeze_proprio_encoder or config.freeze_depth_encoder,
        freeze_fusion=config.freeze_fusion_transformer,
        freeze_temporal=config.freeze_temporal_transformer,
        device=config.device,
    )

    # 从 checkpoint 元数据更新配置（覆盖硬编码的默认值）
    depth_hist_len = int(checkpoint_meta.get("depth_hist_len", 1))
    camera_resolution = tuple(checkpoint_meta.get("camera_resolution", [58, 87]))
    config.depth_shape = (depth_hist_len, camera_resolution[0], camera_resolution[1])
    config.proprio_dim = int(checkpoint_meta.get("num_prop", config.proprio_dim))
    config.action_dim = int(checkpoint_meta.get("action_dim", config.action_dim))
    print(f"[INFO] Updated config from checkpoint: depth_shape={config.depth_shape}, "
          f"proprio_dim={config.proprio_dim}, action_dim={config.action_dim}")

    # 初始化 PPOStudent
    ppo = initialize_ppo_student(
        actor_critic=actor_critic,
        learning_rate=config.learning_rate,
        clip_param=config.clip_param,
        gamma=config.gamma,
        lam=config.lam,
        entropy_coef=config.entropy_coef,
        value_loss_coef=config.value_loss_coef,
        max_grad_norm=config.max_grad_norm,
        num_learning_epochs=config.num_learning_epochs,
        num_mini_batches=config.num_mini_batches,
        schedule=config.schedule,
        desired_kl=config.desired_kl,
        device=config.device,
    )

    # 初始化 storage
    ppo.init_storage(
        num_envs=config.num_envs,
        num_steps=config.num_steps_per_env,
        proprio_dim=config.proprio_dim,
        depth_shape=config.depth_shape,
        action_dim=config.action_dim,
    )

    # 创建 Domain Randomization
    domain_rand = None
    if config.domain_rand_enabled:
        domain_rand = create_domain_randomization(
            enabled=True,
            use_curriculum=config.domain_rand_curriculum,
            num_envs=config.num_envs,
            device=config.device,
        )
        print("[INFO] Domain Randomization: ENABLED")
        if config.domain_rand_curriculum:
            print("[INFO] Domain Randomization Curriculum: ENABLED")

    # 恢复训练
    start_iteration = 0
    if resume_path is not None:
        start_iteration = load_training_checkpoint(
            Path(resume_path),
            actor_critic,
            ppo.optimizer,
        )

    # 初始化 TensorBoard 日志器
    tb_logger = init_tensorboard(
        config=config,
        checkpoint_meta=checkpoint_meta,
        run_dir=run_dir,
    )

    # 训练循环
    print(f"[INFO] Starting training from iteration {start_iteration}")
    print(f"[INFO] Max iterations: {config.max_iterations}")

    for iteration in range(start_iteration, config.max_iterations):
        # 运行训练迭代
        train_info = run_training_iteration(
            env=env,
            ppo=ppo,
            actor_critic=actor_critic,
            domain_rand=domain_rand,
            num_steps=config.num_steps_per_env,
            iteration=iteration,
            num_prop=config.proprio_dim,
        )

        # 日志记录
        if iteration % config.log_interval == 0:
            # 提取 episode 统计
            episode_stats = train_info.get("episode_stats", {})
            ep_return = episode_stats.get("mean_return", 0.0)
            ep_length = episode_stats.get("mean_length", 0.0)
            ep_count = episode_stats.get("count", 0)

            print(
                f"[Iter {iteration}] "
                f"ep_ret={ep_return:.2f} ep_len={ep_length:.0f} ep_cnt={ep_count} | "
                f"v_loss={train_info['value_loss']:.4f} "
                f"p_loss={train_info['surrogate_loss']:.4f} "
                f"ent={train_info['entropy']:.4f} "
                f"kl={train_info['kl']:.4f} "
                f"lr={train_info['learning_rate']:.2e}"
            )

            # 记录到 TensorBoard
            if config.use_tensorboard and tb_logger is not None:
                # 获取 Domain Randomization 参数
                domain_rand_params = None
                if domain_rand is not None and config.domain_rand_curriculum:
                    curriculum = domain_rand.curriculum
                    if curriculum is not None:
                        domain_rand_params = curriculum.get_params(iteration)

                # 记录训练指标
                log_training_metrics(
                    logger=tb_logger,
                    iteration=iteration,
                    train_info=train_info,
                    domain_rand_params=domain_rand_params,
                    episode_stats=episode_stats,
                )

        # 保存检查点
        if iteration % config.save_interval == 0 and iteration > 0:
            checkpoint_path = run_dir / f"checkpoint_{iteration:06d}.pt"
            save_training_checkpoint(
                path=checkpoint_path,
                actor_critic=actor_critic,
                optimizer=ppo.optimizer,
                iteration=iteration,
                config=vars(config),
            )

    # 保存最终检查点
    final_checkpoint_path = run_dir / "checkpoint_final.pt"
    save_training_checkpoint(
        path=final_checkpoint_path,
        actor_critic=actor_critic,
        optimizer=ppo.optimizer,
        iteration=config.max_iterations,
        config=vars(config),
    )

    # 关闭 TensorBoard 日志器
    if config.use_tensorboard and tb_logger is not None and tb_logger.enabled:
        try:
            tb_logger.finish()
            print("[tensorboard] 日志记录已完成")
        except Exception as e:
            print(f"[tensorboard] 警告: 完成日志记录时出错: {e}")

    print(f"[INFO] Training completed. Final checkpoint: {final_checkpoint_path}")


# ============================================================
# 主函数入口（需要 Isaac Lab 环境）
# ============================================================

if __name__ == "__main__":
    # 这部分需要 Isaac Lab 环境，在实际运行时启用
    # 测试时可以单独导入模块中的函数

    print("=" * 60)
    print("Student RL Fine-tuning Training Script")
    print("=" * 60)
    print()
    print("This script requires Isaac Lab environment.")
    print("Please run with proper Isaac Lab setup.")
    print()
    print("Usage:")
    print("  python train_student_rl_finetune.py --dagger_checkpoint <path>")
    print()
    print("For testing, import functions directly:")
    print("  from train_student_rl_finetune import parse_training_config")
    print("=" * 60)
