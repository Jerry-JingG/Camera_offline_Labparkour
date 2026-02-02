# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Student RL Fine-tuning 配置

用于对 DAgger 训练的 Student Policy 进行 PPO 强化学习微调的配置文件。
包含 PPO 超参数、环境配置、Domain Randomization 调度和编码器冻结策略。
"""

from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlOnPolicyRunnerCfg,
    ParkourRslRlPpoActorCriticCfg,
    ParkourRslRlActorCfg,
    ParkourRslRlStateHistEncoderCfg,
    ParkourRslRlEstimatorCfg,
    ParkourRslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


@configclass
class StudentFinetuneAlgorithmCfg(ParkourRslRlPpoAlgorithmCfg):
    """Student Fine-tuning PPO 算法配置

    使用保守的超参数以避免灾难性遗忘。
    学习率低于 teacher (1e-4 vs 2e-4)。
    """

    # PPO 核心超参数
    learning_rate: float = 1e-4  # 低于 teacher，避免灾难性遗忘
    clip_param: float = 0.2  # 标准 PPO clip 参数
    gamma: float = 0.99  # 折扣因子
    lam: float = 0.95  # GAE lambda
    entropy_coef: float = 0.01  # 熵系数，鼓励探索
    value_loss_coef: float = 0.5  # 价值损失系数
    max_grad_norm: float = 1.0  # 梯度裁剪

    # 训练配置
    num_learning_epochs: int = 5  # 每次更新的 epoch 数
    num_mini_batches: int = 4  # Mini-batch 数量
    schedule: str = "adaptive"  # 自适应学习率调度
    desired_kl: float = 0.01  # 期望的 KL 散度
    use_clipped_value_loss: bool = True  # 使用裁剪的价值损失


@configclass
class UnitreeGo2StudentFinetunePPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    """Unitree Go2 Student Fine-tuning PPO Runner 配置

    用于 RL 微调 DAgger 训练的 Student Policy。
    包含环境配置、编码器冻结策略和 Domain Randomization 设置。
    """

    # 实验配置
    experiment_name: str = "unitree_go2_student_finetune"
    run_name: str = ""

    # 环境配置
    num_steps_per_env: int = 64  # 匹配 TXL sequence length
    max_iterations: int = 10000  # ~10M steps (256 envs * 64 steps * 10000)

    # 日志和检查点配置
    save_interval: int = 100  # 每 100 次迭代保存一次
    log_interval: int = 10  # 每 10 次迭代记录一次
    empirical_normalization: bool = False

    # 编码器冻结策略配置
    # 默认不冻结（端到端微调），如果出现灾难性遗忘可以启用
    freeze_proprio_encoder: bool = False
    freeze_depth_encoder: bool = False
    freeze_fusion_transformer: bool = False
    freeze_temporal_transformer: bool = False

    # Domain Randomization 配置
    domain_rand_enabled: bool = True  # 启用 domain randomization
    domain_rand_curriculum: bool = True  # 启用 curriculum 调度

    # Policy 配置（继承自 teacher，但使用 student 架构）
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=ParkourRslRlActorCfg(
            class_name="Actor",
            state_history_encoder=ParkourRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder"
            ),
        ),
    )

    # Estimator 配置
    estimator = ParkourRslRlEstimatorCfg(hidden_dims=[128, 64])

    # Depth encoder 配置（student 使用深度相机）
    depth_encoder = None  # 将在训练脚本中从 DAgger checkpoint 加载

    # Algorithm 配置
    algorithm = StudentFinetuneAlgorithmCfg()

