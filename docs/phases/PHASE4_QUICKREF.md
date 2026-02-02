# Phase 4 快速参考

## 配置文件位置
```
parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py
```

## 快速导入
```python
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_finetune_cfg import (
    UnitreeGo2StudentFinetunePPORunnerCfg
)

cfg = UnitreeGo2StudentFinetunePPORunnerCfg()
```

## 关键参数速查

### PPO 超参数
```python
cfg.algorithm.learning_rate = 1e-4      # 学习率
cfg.algorithm.clip_param = 0.2          # PPO clip
cfg.algorithm.gamma = 0.99              # 折扣因子
cfg.algorithm.lam = 0.95                # GAE lambda
cfg.algorithm.entropy_coef = 0.01       # 熵系数
cfg.algorithm.value_loss_coef = 0.5     # 价值损失系数
cfg.algorithm.max_grad_norm = 1.0       # 梯度裁剪
```

### 环境配置
```python
cfg.num_steps_per_env = 64              # 每环境步数
cfg.max_iterations = 10000              # 最大迭代
cfg.save_interval = 100                 # 保存间隔
cfg.log_interval = 10                   # 日志间隔
```

### 编码器冻结
```python
cfg.freeze_proprio_encoder = False      # Proprio encoder
cfg.freeze_depth_encoder = False        # Depth encoder
cfg.freeze_fusion_transformer = False   # Fusion transformer
cfg.freeze_temporal_transformer = False # Temporal transformer
```

### Domain Randomization
```python
cfg.domain_rand_enabled = True          # 启用 DR
cfg.domain_rand_curriculum = True       # 启用 curriculum
```

## 常用配置模式

### 保守训练（避免遗忘）
```python
cfg.algorithm.learning_rate = 5e-5
cfg.freeze_depth_encoder = True
cfg.freeze_fusion_transformer = True
```

### 激进训练（最大性能）
```python
cfg.algorithm.learning_rate = 2e-4
cfg.max_iterations = 20000
cfg.algorithm.entropy_coef = 0.02
```

### 禁用 Domain Randomization
```python
cfg.domain_rand_enabled = False
cfg.domain_rand_curriculum = False
```

## 测试命令
```bash
# 运行简化测试
python tests/test_student_finetune_cfg_simple.py

# 查看使用示例
python examples/config_usage_examples.py
```

## 文档位置
- 详细文档: `docs/config/student-finetune-config.md`
- 完成报告: `docs/phase4-completion-report.md`
- 总结: `docs/PHASE4_SUMMARY.md`

## 状态
✅ 完成 | ✅ 测试通过 | ✅ 文档完整
