# wandb 集成使用文档

## 概述

本项目已集成 [Weights & Biases (wandb)](https://wandb.ai/) 用于 RL Fine-tuning 训练过程的实验跟踪和可视化。wandb 提供了强大的实验管理、指标可视化和模型版本控制功能。

## 功能特性

### 1. 训练指标记录

自动记录以下训练指标：

- **PPO 损失**
  - `train/value_loss`: 价值函数损失
  - `train/policy_loss`: 策略损失
  - `train/entropy`: 策略熵
  - `train/kl_divergence`: KL 散度
  - `train/learning_rate`: 学习率

- **Domain Randomization 参数**（如果启用课程学习）
  - `domain_rand/noise_std`: 深度噪声标准差
  - `domain_rand/salt_pepper`: 椒盐噪声概率
  - `domain_rand/dropout`: 相机丢失概率
  - `domain_rand/latency`: 延迟帧数

### 2. 超参数配置记录

自动记录所有训练配置：
- 环境配置（环境数量、步数等）
- PPO 超参数（学习率、clip 参数、gamma 等）
- 编码器冻结策略
- Domain Randomization 配置
- 模型架构参数

### 3. 模型检查点管理

自动上传训练检查点到 wandb Artifacts：
