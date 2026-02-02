# Phase 2 快速参考

**版本：** 1.0
**日期：** 2026-02-03
**完整文档：** [Phase 2 Summary](PHASE2_SUMMARY.md)

---

## 概览

Phase 2 实现了适配 Transformer-XL 的 PPO 算法：
- **StudentRolloutStorage**：支持深度图像和 TXL 内存的存储
- **PPOStudent**：适配图像观测和序列处理的 PPO 算法

---

## StudentRolloutStorage

### 基本用法

```python
from scripts.rsl_rl.modules import StudentRolloutStorage

# 创建存储
storage = StudentRolloutStorage(
    num_steps=64,                    # 每个 rollout 的步数
    num_envs=256,                    # 并行环境数
    proprio_dim=53,                  # Proprio 观测维度
    action_dim=12,                   # 动作维度
    depth_shape=(4, 58, 87),         # 深度图像形状 (hist_len, H, W)
    device="cuda",
)

# 添加转换
for step in range(num_steps):
    storage.add_transition(
        step=step,
        proprio=proprio,              # [num_envs, proprio_dim]
        depth=depth,                  # [num_envs, hist_len, H, W]
        actions=actions,              # [num_envs, action_dim]
        rewards=rewards,              # [num_envs, 1]
        values=values,                # [num_envs, 1]
        log_probs=log_probs,          # [num_envs, 1]
        dones=dones,                  # [num_envs, 1]
        mu=mu,                        # [num_envs, action_dim]
        sigma=sigma,                  # [num_envs, action_dim]
        mems=mems,                    # List[Tensor] (可选)
    )

# 计算回报和优势
storage.compute_returns(
    last_values=last_values,         # [num_envs, 1]
    gamma=0.99,                      # 折扣因子
    lam=0.95,                        # GAE lambda
)

# 生成 mini-batch
for batch in storage.sequence_mini_batch_generator(
    num_batches=4,
    num_epochs=5,
):
    proprio = batch["proprio"]       # [num_steps, batch_size, proprio_dim]
    depth = batch["depth"]           # [num_steps, batch_size, hist_len, H, W]
    actions = batch["actions"]       # [num_steps, batch_size, action_dim]
    returns = batch["returns"]       # [num_steps, batch_size, 1]
    advantages = batch["advantages"] # [num_steps, batch_size, 1]
    # ... 训练代码

# 清空存储
storage.clear()
```

### 关键特性

- ✅ uint8 深度存储（节省 75% 内存）
- ✅ 预分配缓冲区（避免动态分配）
- ✅ GAE 计算
- ✅ 序列感知的 mini-batch 生成
- ✅ TXL 内存状态管理

### 内存占用估算

对于 `num_envs=256`, `num_steps=64`, `depth_shape=(4, 58, 87)`:

| 缓冲区 | 形状 | 大小 (uint8/float32) |
|--------|------|---------------------|
| proprio | [64, 256, 53] | 3.5 MB |
| depth | [64, 256, 4, 58, 87] | 330 MB (uint8) |
| actions | [64, 256, 12] | 0.8 MB |
| values | [65, 256, 1] | 0.07 MB |
| **总计** | - | **~350 MB** |

---

## PPOStudent

### 基本用法

```python
from scripts.rsl_rl.modules import PPOStudent, StudentActorCritic

# 创建 Actor-Critic
actor_critic = StudentActorCritic(student_policy)

# 创建 PPO
ppo = PPOStudent(
    actor_critic=actor_critic,
    num_learning_epochs=5,           # 每次更新的 epoch 数
    num_mini_batches=4,              # 每个 epoch 的 mini-batch 数
    clip_param=0.2,                  # PPO 裁剪参数
    gamma=0.99,                      # 折扣因子
    lam=0.95,                        # GAE lambda
    value_loss_coef=1.0,             # 价值损失系数
    entropy_coef=0.01,               # 熵奖励系数
    learning_rate=1e-4,              # 学习率
    max_grad_norm=1.0,               # 梯度裁剪
    use_clipped_value_loss=True,     # 使用裁剪的价值损失
    schedule="adaptive",             # 学习率调度 ('fixed' 或 'adaptive')
    desired_kl=0.01,                 # 目标 KL 散度
    device="cuda",
)

# 初始化存储
ppo.init_storage(
    num_envs=256,
    num_steps=64,
    proprio_dim=53,
    depth_shape=(4, 58, 87),
    action_dim=12,
)

# Rollout 循环
for step in range(num_steps):
    # 采样动作
    actions = ppo.act(proprio, depth)

    # 环境步进
    obs_next, rewards, dones, infos = env.step(actions)

    # 处理环境步骤
    ppo.process_env_step(rewards, dones, infos)

    # 更新观测
    proprio, depth = obs_next["proprio"], obs_next["depth"]

# 计算回报
last_values = actor_critic.evaluate(proprio, depth)
ppo.compute_returns(last_values)

# PPO 更新
loss_dict = ppo.update()
print(f"Policy Loss: {loss_dict['policy_loss']:.4f}")
print(f"Value Loss: {loss_dict['value_loss']:.4f}")
print(f"Entropy: {loss_dict['entropy']:.4f}")
print(f"KL: {loss_dict['kl']:.4f}")
print(f"LR: {loss_dict['learning_rate']:.6f}")
```

### 完整训练循环

```python
# 训练循环
for iteration in range(max_iterations):
    # Rollout 阶段
    for step in range(num_steps):
        actions = ppo.act(proprio, depth)
        obs_next, rewards, dones, infos = env.step(actions)
        ppo.process_env_step(rewards, dones, infos)
        proprio, depth = obs_next["proprio"], obs_next["depth"]

    # 计算回报
    last_values = actor_critic.evaluate(proprio, depth)
    ppo.compute_returns(last_values)

    # 更新策略
    loss_dict = ppo.update()

    # 日志
    if iteration % 10 == 0:
        print(f"Iter {iteration}: {loss_dict}")

    # 保存检查点
    if iteration % 100 == 0:
        torch.save({
            "iteration": iteration,
            "actor_critic": actor_critic.state_dict(),
            "optimizer": ppo.optimizer.state_dict(),
        }, f"checkpoint_{iteration}.pth")
```

### 关键特性

- ✅ 序列感知的 mini-batch 生成
- ✅ 自动内存管理（重置 done 环境）
- ✅ 梯度裁剪
- ✅ 裁剪的价值损失
- ✅ 自适应学习率
- ✅ KL 散度监控

---

## 序列感知批处理

### 标准 PPO vs 序列感知 PPO

```
标准 PPO:
+---+---+---+---+---+---+---+---+
| t0| t1| t2| t3| t4| t5| t6| t7|  <- 随机转换
+---+---+---+---+---+---+---+---+

序列感知 PPO:
+---+---+---+---+---+---+---+---+
| e0| e0| e0| e0| e0| e0| e0| e0|  <- 环境 0 的完整序列
+---+---+---+---+---+---+---+---+
| e1| e1| e1| e1| e1| e1| e1| e1|  <- 环境 1 的完整序列
+---+---+---+---+---+---+---+---+
```

### 为什么需要序列感知？

- Transformer-XL 需要保持时序顺序
- 内存状态在序列中传递
- 随机打乱会破坏时序依赖

### 如何补偿样本效率损失？

- 使用更多环境（256 vs 128）
- 更长的 rollout（64 steps vs 32）
- 更多学习 epoch（5 vs 3）

---

## GAE 计算

### 公式

```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
R_t = A_t + V(s_t)
```

### 参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| gamma | 0.99 | 折扣因子，越大越重视长期奖励 |
| lam | 0.95 | GAE lambda，越大越重视长期优势 |

---

## 超参数调优

### PPO 超参数

| 参数 | 默认值 | 调优范围 | 说明 |
|------|--------|---------|------|
| clip_param | 0.2 | 0.1-0.3 | PPO 裁剪参数 |
| learning_rate | 1e-4 | 1e-5 - 1e-3 | 学习率 |
| num_learning_epochs | 5 | 3-10 | 每次更新的 epoch 数 |
| num_mini_batches | 4 | 2-8 | mini-batch 数量 |
| value_loss_coef | 1.0 | 0.5-2.0 | 价值损失系数 |
| entropy_coef | 0.01 | 0.001-0.1 | 熵奖励系数 |
| max_grad_norm | 1.0 | 0.5-2.0 | 梯度裁剪阈值 |

### 自适应学习率

```python
# 基于 KL 散度自动调整
if mean_kl > desired_kl * 2.0:
    learning_rate /= 1.5  # 策略变化太快
elif mean_kl < desired_kl / 2.0:
    learning_rate *= 1.5  # 策略变化太慢
```

---

## 测试

### 运行测试

```bash
# 激活环境
conda activate parkour

# StudentRolloutStorage 测试
cd scripts/rsl_rl/modules/tests
python -m pytest test_student_rollout_storage.py -v

# PPOStudent 测试
python -m pytest test_ppo_student.py -v

# 所有测试
python -m pytest -v
```

### 测试覆盖率

```bash
pytest --cov=scripts.rsl_rl.modules \
       --cov-report=html
```

---

## 常见问题

### Q: 为什么使用 uint8 存储深度图像？

**A:**
- 节省 75% 内存（1 byte vs 4 bytes）
- 支持更大的 batch size
- 转换开销可忽略不计

### Q: 如何选择 num_steps 和 num_envs？

**A:**
- `num_steps * num_envs` 应该足够大（通常 > 10000）
- 更多环境 → 更好的并行化
- 更长 rollout → 更好的长期依赖学习
- 推荐：`num_envs=256`, `num_steps=64`

### Q: 自适应学习率何时有用？

**A:**
- 训练不稳定时
- 策略变化过快或过慢时
- 不确定最佳学习率时
- 推荐：始终启用

### Q: 如何监控训练进度？

**A:**
监控以下指标：
- `policy_loss`: 应该逐渐减小
- `value_loss`: 应该逐渐减小
- `entropy`: 应该缓慢减小（保持探索）
- `kl`: 应该接近 `desired_kl`
- `learning_rate`: 应该稳定或缓慢变化

### Q: 训练不稳定怎么办？

**A:**
1. 降低学习率（1e-5）
2. 增加梯度裁剪（0.5）
3. 减少 clip_param（0.1）
4. 增加 value_loss_coef（2.0）
5. 启用自适应学习率

---

## 性能优化技巧

### 1. 内存优化

```python
# 使用 uint8 存储深度
storage = StudentRolloutStorage(..., device="cuda")

# 预分配缓冲区
# 避免动态分配
```

### 2. 计算优化

```python
# 批量处理
total_samples = num_steps * batch_size
proprio_flat = proprio_batch.reshape(total_samples, -1)

# 分离张量
actions = actions.detach()
```

### 3. GPU 利用率

```python
# 使用更大的 batch size
num_envs = 256  # 而非 128

# 使用混合精度（可选）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## 文件位置

```
scripts/rsl_rl/modules/
├── student_rollout_storage.py           # StudentRolloutStorage 实现
├── ppo_student.py                       # PPOStudent 实现
└── tests/
    ├── test_student_rollout_storage.py  # StudentRolloutStorage 测试
    └── test_ppo_student.py              # PPOStudent 测试
```

---

## 下一步

- 阅读 [Phase 1 Quick Reference](PHASE1_QUICKREF.md)
- 查看 [完整 Phase 2 文档](PHASE2_SUMMARY.md)
- 查看 [架构设计文档](architecture/2026-02-02-rl-finetuning-architecture.md)

---

**最后更新：** 2026-02-03
