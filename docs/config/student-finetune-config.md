# Student RL Fine-tuning 配置文档

## 概述

`rsl_student_finetune_cfg.py` 包含用于对 DAgger 训练的 Student Policy 进行 PPO 强化学习微调的配置。

## 配置文件位置

```
parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py
```

## 主要配置类

### 1. StudentFinetuneAlgorithmCfg

PPO 算法配置，使用保守的超参数以避免灾难性遗忘。

#### PPO 核心超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 1e-4 | 学习率，低于 teacher (2e-4) 以避免灾难性遗忘 |
| `clip_param` | 0.2 | PPO clip 参数，标准值 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda |
| `entropy_coef` | 0.01 | 熵系数，鼓励探索 |
| `value_loss_coef` | 0.5 | 价值损失系数 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |

#### 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_learning_epochs` | 5 | 每次更新的 epoch 数 |
| `num_mini_batches` | 4 | Mini-batch 数量 |
| `schedule` | "adaptive" | 自适应学习率调度 |
| `desired_kl` | 0.01 | 期望的 KL 散度 |
| `use_clipped_value_loss` | True | 使用裁剪的价值损失 |

### 2. UnitreeGo2StudentFinetunePPORunnerCfg

主配置类，包含所有训练相关的设置。

#### 实验配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `experiment_name` | "unitree_go2_student_finetune" | 实验名称 |
| `run_name` | "" | 运行名称（可选） |

#### 环境配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_steps_per_env` | 64 | 每环境步数，匹配 TXL sequence length |
| `max_iterations` | 10000 | 最大迭代次数，约 10M steps |

**计算总步数：**
```
总步数 = num_envs × num_steps_per_env × max_iterations
      = 256 × 64 × 10000
      = 163,840,000 steps (~164M steps)
```

#### 日志和检查点配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `save_interval` | 100 | 每 100 次迭代保存一次检查点 |
| `log_interval` | 10 | 每 10 次迭代记录一次日志 |
| `empirical_normalization` | False | 不使用经验归一化 |

#### 编码器冻结策略配置

默认不冻结（端到端微调），如果出现灾难性遗忘可以启用。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `freeze_proprio_encoder` | False | 是否冻结 proprio encoder |
| `freeze_depth_encoder` | False | 是否冻结 depth encoder |
| `freeze_fusion_transformer` | False | 是否冻结 fusion transformer |
| `freeze_temporal_transformer` | False | 是否冻结 temporal transformer |

**使用建议：**
- 默认使用端到端微调（所有编码器都不冻结）
- 如果 DAgger 验证损失增加 >20%，考虑冻结编码器
- 可以逐步冻结：先冻结 depth encoder，再冻结 fusion，最后冻结 temporal

#### Domain Randomization 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `domain_rand_enabled` | True | 启用 domain randomization |
| `domain_rand_curriculum` | True | 启用 curriculum 调度 |

**Domain Randomization Curriculum 调度：**

| 迭代范围 | Gaussian Noise | Salt-Pepper | Camera Dropout | Latency |
|----------|----------------|-------------|----------------|---------|
| 0-1000 | 0.01 | 0.005 | 0.1 | 1 frame |
| 1000-3000 | 0.02 | 0.01 | 0.2 | 2 frames |
| 3000-5000 | 0.03 | 0.015 | 0.3 | 3 frames |
| 5000+ | 0.04 | 0.02 | 0.5 | 3 frames |

## 使用方法

### 基本使用

```python
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
    UnitreeGo2StudentFinetunePPORunnerCfg
)

# 创建配置实例
cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

# 访问配置参数
print(f"学习率: {cfg.algorithm.learning_rate}")
print(f"最大迭代: {cfg.max_iterations}")
print(f"Domain Randomization: {cfg.domain_rand_enabled}")
```

### 命令行覆盖参数

在训练脚本中可以通过命令行参数覆盖配置：

```bash
python scripts/rsl_rl/train_student_rl_finetune.py \
    --dagger_checkpoint path/to/checkpoint.pth \
    --max_iterations 15000 \
    --learning_rate 5e-5 \
    --freeze_encoders
```

### 自定义配置

```python
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
    UnitreeGo2StudentFinetunePPORunnerCfg
)

# 创建自定义配置
cfg = UnitreeGo2StudentFinetunePPORunnerCfg()

# 修改参数
cfg.max_iterations = 15000
cfg.algorithm.learning_rate = 5e-5
cfg.freeze_depth_encoder = True  # 冻结 depth encoder

# 禁用 domain randomization
cfg.domain_rand_enabled = False
```

## 与 Teacher 配置的对比

| 参数 | Teacher | Student Fine-tune | 说明 |
|------|---------|-------------------|------|
| `learning_rate` | 2e-4 | 1e-4 | Student 使用更低的学习率 |
| `num_steps_per_env` | 24 | 64 | Student 需要更长的序列 |
| `observation` | Height scan | Depth camera | 不同的观测空间 |
| `architecture` | MLP | Transformer-XL | 不同的网络架构 |

## 监控指标

### 训练指标

- `episode_return_mean`: 平均回合回报（应增加）
- `episode_length_mean`: 平均回合长度（应增加）
- `goal_progress`: 目标进度百分比（应增加）
- `policy_loss`: 策略损失（应先降后稳定）
- `value_loss`: 价值损失（应先降后稳定）
- `entropy`: 策略熵（应缓慢降低）
- `kl_divergence`: KL 散度（应 < desired_kl）

### 鲁棒性指标

- `return_clean`: 无增强的回报（基线）
- `return_noisy`: 有噪声的回报（应 >80% clean）
- `return_dropout`: 相机掉线的回报（应 >85% clean）
- `return_latency`: 有延迟的回报（应 >80% clean）
- `dagger_loss`: DAgger 验证损失（应保持稳定）

## 故障排除

### 问题 1: 灾难性遗忘

**症状：** DAgger 验证损失增加 >20%，性能下降

**解决方案：**
1. 降低学习率：`cfg.algorithm.learning_rate = 5e-5`
2. 冻结编码器：`cfg.freeze_depth_encoder = True`
3. 回滚到最佳检查点

### 问题 2: 训练不稳定

**症状：** 损失震荡，KL 散度过大

**解决方案：**
1. 降低学习率
2. 增加梯度裁剪：`cfg.algorithm.max_grad_norm = 0.5`
3. 减少 mini-batch 数量：`cfg.algorithm.num_mini_batches = 2`

### 问题 3: Domain Randomization 过强

**症状：** 性能在增强下严重下降

**解决方案：**
1. 禁用 curriculum：`cfg.domain_rand_curriculum = False`
2. 降低增强强度（修改 curriculum 调度）
3. 延长 curriculum 阶段

## 参考文档

- [Implementation Plan](/home/droplet/IsaacLab/Camera_offline_Labparkour/docs/plans/2026-02-02-rl-finetuning-design.md)
- [Architecture Design](/home/droplet/IsaacLab/Camera_offline_Labparkour/docs/architecture/2026-02-02-rl-finetuning-architecture.md)

## 版本历史

- **v1.0** (2026-02-03): 初始版本，包含基本 PPO 配置和 Domain Randomization 设置
