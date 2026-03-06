# RL Fine-tuning 快速入门指南

## 5 分钟快速开始

### 步骤 1: 配置 Checkpoint 路径 (1 分钟)

```bash
# 找到你的 DAgger checkpoint
ls -la logs/rsl_rl/parkour_student/

# 编辑启动脚本
vim scripts/launch_rl_finetune.sh

# 修改第 39 行左右的 DAGGER_CHECKPOINT 变量
DAGGER_CHECKPOINT="logs/rsl_rl/parkour_student/YOUR_RUN/model_10000.pt"

# 保存并退出 (:wq)
```

### 步骤 2: 测试配置 (1 分钟)

```bash
# 查看将要执行的命令（不实际执行）
./scripts/launch_rl_finetune.sh --dry-run
```

### 步骤 3: 快速验证 (2 分钟)

```bash
# 使用少量环境和迭代快速测试
./scripts/launch_rl_finetune.sh fast-test
```

### 步骤 4: 正式训练 (1 分钟启动)

```bash
# 使用默认配置开始训练
./scripts/launch_rl_finetune.sh baseline
```

## 常用命令

```bash
# 查看帮助
./scripts/launch_rl_finetune.sh --help

# 使用不同预设
./scripts/launch_rl_finetune.sh baseline          # 标准训练
./scripts/launch_rl_finetune.sh freeze-encoders   # 冻结编码器
./scripts/launch_rl_finetune.sh no-domain-rand    # 禁用 DR
./scripts/launch_rl_finetune.sh fast-test         # 快速测试
./scripts/launch_rl_finetune.sh high-lr           # 高学习率
./scripts/launch_rl_finetune.sh conservative      # 保守训练

# Dry-run 模式（只显示命令）
./scripts/launch_rl_finetune.sh [preset] --dry-run
```

## 预设速查

| 预设 | 环境数 | 迭代数 | 学习率 | 冻结编码器 | DR | 用途 |
|-----|--------|--------|--------|-----------|----|----|
| baseline | 4096 | 1500 | 1e-5 | ❌ | ✅ | 标准训练 |
| freeze-encoders | 4096 | 1500 | 5e-6 | ✅ | ✅ | 保护特征 |
| no-domain-rand | 4096 | 1500 | 1e-5 | ❌ | ❌ | 消融实验 |
| fast-test | 512 | 100 | 1e-5 | ❌ | ✅ | 快速验证 |
| high-lr | 4096 | 1500 | 5e-5 | ❌ | ✅ | 激进训练 |
| conservative | 4096 | 1500 | 5e-6 | ❌ | ✅ | 保守训练 |

## 修改配置

编辑脚本顶部的配置区域（第 30-150 行）：

```bash
vim scripts/launch_rl_finetune.sh
```

关键配置项：

```bash
# 环境配置
NUM_ENVS=4096              # 并行环境数
MAX_ITERATIONS=1500        # 训练迭代数

# PPO 超参数
LEARNING_RATE=1e-5         # 学习率

# 编码器
FREEZE_DEPTH_ENCODER=false # 是否冻结深度编码器

# Domain Randomization
ENABLE_DOMAIN_RAND=true    # 是否启用 DR

# Wandb
USE_WANDB=true             # 是否使用 wandb
EXPERIMENT_NAME=""         # 实验名称
```

## 故障排除

### 问题 1: Checkpoint 不存在

```bash
# 查找 checkpoint
find logs/rsl_rl/parkour_student -name "model_*.pt"

# 更新脚本中的路径
vim scripts/launch_rl_finetune.sh
```

### 问题 2: Conda 环境不存在

```bash
# 创建环境
conda create -n parkour python=3.8
conda activate parkour
pip install -r requirements.txt
```

### 问题 3: GPU 内存不足

```bash
# 减少环境数量
vim scripts/launch_rl_finetune.sh
# 修改: NUM_ENVS=2048

# 或使用 fast-test
./scripts/launch_rl_finetune.sh fast-test
```

## 下一步

### 学习更多

- **完整文档**: `docs/RL_FINETUNE_LAUNCHER.md`
- **配置示例**: `docs/RL_FINETUNE_CONFIG_EXAMPLES.md`
- **使用示例**: `docs/RL_FINETUNE_USAGE_EXAMPLES.md`
- **文档索引**: `docs/RL_FINETUNE_LAUNCHER_README.md`

### 进行实验

```bash
# 消融实验
./scripts/launch_rl_finetune.sh baseline
./scripts/launch_rl_finetune.sh freeze-encoders
./scripts/launch_rl_finetune.sh no-domain-rand

# 在 wandb 中对比结果
open https://wandb.ai/your-team/parkour-rl-finetune
```

### 自定义配置

1. 编辑脚本添加新预设
2. 修改配置参数
3. 测试新配置
4. 运行实验

## 最佳实践

1. ✅ 首次使用先 dry-run
2. ✅ 用 fast-test 验证环境
3. ✅ 设置有意义的 EXPERIMENT_NAME
4. ✅ 使用 wandb 跟踪实验
5. ✅ 定期保存配置快照

## 获取帮助

- 查看文档: `docs/RL_FINETUNE_LAUNCHER_README.md`
- 查看示例: `docs/RL_FINETUNE_USAGE_EXAMPLES.md`
- 查看故障排除: `docs/RL_FINETUNE_LAUNCHER.md` (故障排除部分)

---

**提示**: 这是一个快速入门指南。完整功能和详细说明请参考其他文档。
