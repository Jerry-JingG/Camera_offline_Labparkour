# 配置文档

**最后更新：** 2026-02-03

---

## 概述

本目录包含项目的配置文件说明文档，描述各种配置参数的含义、用法和最佳实践。

---

## 文档列表

### Student Fine-tune 配置

- **[student-finetune-config.md](student-finetune-config.md)**
  - Student 模型 fine-tuning 配置说明
  - PPO 超参数配置
  - 环境配置
  - 训练配置
  - 领域随机化配置

**主要内容：**
- 配置文件结构
- 参数详细说明
- 推荐配置值
- 配置示例
- 调优建议

---

## 配置文件位置

### 训练配置
```
scripts/rsl_rl/configs/
└── rsl_student_finetune_cfg.py    # Student fine-tuning 配置
```

### 环境配置
```
parkour_isaaclab/envs/
└── [环境配置文件]
```

---

## 使用建议

### 配置优先级

1. **命令行参数** - 最高优先级
   ```bash
   python train.py --num_envs 512 --learning_rate 3e-4
   ```

2. **配置文件** - 中等优先级
   ```python
   # rsl_student_finetune_cfg.py
   num_envs = 256
   learning_rate = 1e-4
   ```

3. **默认值** - 最低优先级
   ```python
   # 代码中的默认值
   ```

### 配置最佳实践

1. **使用配置文件管理复杂配置**
   - 便于版本控制
   - 易于复现实验
   - 方便团队协作

2. **使用命令行参数进行快速实验**
   - 快速调整单个参数
   - 不需要修改配置文件
   - 适合超参数搜索

3. **记录配置变更**
   - 在实验日志中记录配置
   - 使用 git 跟踪配置变更
   - 文档化重要配置决策

---

## 常用配置场景

### 快速测试
```python
num_envs = 64          # 少量环境
num_steps = 32         # 短 rollout
max_iterations = 100   # 少量迭代
```

### 正式训练
```python
num_envs = 256         # 标准环境数
num_steps = 64         # 标准 rollout
max_iterations = 5000  # 完整训练
```

### 大规模训练
```python
num_envs = 512         # 大量环境
num_steps = 128        # 长 rollout
max_iterations = 10000 # 长时间训练
```

---

## 相关文档

- [项目文档索引](../README.md) - 返回主文档索引
- [Phase 4 文档](../phases/PHASE4_SUMMARY.md) - 训练配置详细说明
- [Phase 4 快速参考](../phases/PHASE4_QUICKREF.md) - 配置快速指南
- [使用指南](../guides/) - 模块使用指南

---

## 配置模板

### 新配置文件模板

```python
"""
[配置名称] 配置文件

描述：[配置用途]
作者：[作者]
日期：[日期]
"""

from dataclasses import dataclass

@dataclass
class Config:
    # 环境配置
    num_envs: int = 256

    # 训练配置
    max_iterations: int = 5000

    # PPO 超参数
    learning_rate: float = 1e-4

    # ... 其他配置
```

---

**维护者：** RL Fine-tuning Team
**最后更新：** 2026-02-03
