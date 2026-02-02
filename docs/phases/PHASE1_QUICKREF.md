# Phase 1 快速参考

**版本：** 1.0
**日期：** 2026-02-03
**完整文档：** [Phase 1 Summary](PHASE1_SUMMARY.md)

---

## 概览

Phase 1 实现了 RL fine-tuning 的核心架构组件：
- **ValueHead**：状态价值估计模块
- **StudentActorCritic**：Actor-Critic 包装器

---

## ValueHead

### 基本用法

```python
from parkour_tasks.extreme_parkour_task.modules.actionheads import ValueHead

# 创建价值头
value_head = ValueHead(
    d_model=128,              # 输入特征维度
    hidden_dims=(256, 256),   # 隐藏层维度
)

# 单步推理
h = torch.randn(32, 128)  # [batch_size, d_model]
values = value_head.forward_step(h)  # [32, 1]

# 序列推理
h_seq = torch.randn(32, 10, 128)  # [batch_size, seq_len, d_model]
values_seq = value_head.forward_sequence(h_seq)  # [32, 10, 1]

# 自动分发
values = value_head(h)  # 自动选择 forward_step 或 forward_sequence
```

### 关键特性

- ✅ 支持单步和序列输入
- ✅ 简单的 MLP 架构（2 层隐藏层）
- ✅ 可配置隐藏层维度
- ✅ 输入验证

---

## StudentActorCritic

### 基本用法

```python
from scripts.rsl_rl.modules import StudentActorCritic

# 加载预训练的学生策略
student_policy = load_student_policy("checkpoint.pth")

# 创建 Actor-Critic
actor_critic = StudentActorCritic(
    student_policy=student_policy,
    value_hidden_dims=(256, 256),
    init_noise_std=1.0,
    freeze_encoders=False,      # 是否冻结编码器
    freeze_fusion=False,        # 是否冻结融合层
    freeze_temporal=False,      # 是否冻结时序模型
)

# 采样动作
proprio = torch.randn(32, 53)        # [batch_size, proprio_dim]
depth = torch.randn(32, 4, 58, 87)   # [batch_size, hist_len, H, W]

actions, log_probs, values, mems = actor_critic.act(proprio, depth)
# actions: [32, 12]
# log_probs: [32]
# values: [32, 1]
# mems: List[Tensor]

# 仅计算价值
values = actor_critic.evaluate(proprio, depth, mems)

# 计算给定动作的对数概率
log_probs = actor_critic.get_actions_log_prob(actions)

# 重置指定环境的内存
done_env_ids = torch.tensor([0, 2, 5])
actor_critic.reset_memory(done_env_ids)

# 分离内存（在训练段之间）
actor_critic.detach_memory()
```

### 冻结策略

| 策略 | 参数设置 | 适用场景 |
|------|---------|---------|
| 端到端微调 | 全部 False | 最大适应能力 |
| 冻结编码器 | `freeze_encoders=True` | 保留 DAgger 表示 |
| 冻结融合 | `freeze_fusion=True` | 保留多模态融合 |
| 冻结时序 | `freeze_temporal=True` | 保留时序建模 |

### 属性访问

```python
# 必须在 act() 之后访问
mean = actor_critic.action_mean      # [batch_size, action_dim]
std = actor_critic.action_std        # [batch_size, action_dim]
entropy = actor_critic.entropy       # [batch_size]
dist = actor_critic.distribution     # Normal distribution
```

---

## 内存管理

### 重置内存

```python
# Episode 终止时调用
dones = torch.tensor([True, False, True, False])
done_env_ids = dones.nonzero().squeeze()
actor_critic.reset_memory(done_env_ids)
```

### 分离内存

```python
# 在训练段之间调用，防止梯度流过时间边界
actor_critic.detach_memory()
```

---

## 测试

### 运行测试

```bash
# 激活环境
conda activate parkour

# ValueHead 测试
cd parkour_tasks/parkour_tasks/extreme_parkour_task/modules/tests
python -m pytest test_value_head.py -v

# StudentActorCritic 测试
cd scripts/rsl_rl/modules/tests
python -m pytest test_student_actor_critic.py -v
```

### 测试覆盖率

```bash
pytest --cov=parkour_tasks.extreme_parkour_task.modules.actionheads \
       --cov=scripts.rsl_rl.modules \
       --cov-report=html
```

---

## 常见问题

### Q: 如何选择冻结策略？

**A:**
- 开始时使用端到端微调（全部 False）
- 如果 DAgger 损失增加 >20%，冻结编码器
- 如果训练不稳定，逐步冻结更多组件

### Q: 何时调用 reset_memory()？

**A:**
- 在 episode 终止时（done=True）
- 防止跨 episode 的信息泄漏
- 在 process_env_step() 中自动处理

### Q: 何时调用 detach_memory()？

**A:**
- 在训练段之间
- 防止梯度流过整个 rollout 历史
- 在 PPO update() 开始时调用

### Q: ValueHead 的隐藏层维度如何选择？

**A:**
- 默认 (256, 256) 适用于大多数情况
- 更复杂的任务可以增加到 (512, 512)
- 更简单的任务可以减少到 (128, 128)

---

## 文件位置

```
parkour_tasks/parkour_tasks/extreme_parkour_task/modules/
├── actionheads/
│   └── value_head.py                    # ValueHead 实现
└── tests/
    └── test_value_head.py               # ValueHead 测试

scripts/rsl_rl/modules/
├── student_actor_critic.py              # StudentActorCritic 实现
└── tests/
    └── test_student_actor_critic.py     # StudentActorCritic 测试
```

---

## 下一步

- 阅读 [Phase 2 Quick Reference](PHASE2_QUICKREF.md)
- 查看 [完整 Phase 1 文档](PHASE1_SUMMARY.md)
- 查看 [架构设计文档](architecture/2026-02-02-rl-finetuning-architecture.md)

---

**最后更新：** 2026-02-03
