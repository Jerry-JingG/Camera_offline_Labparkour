# RL Fine-tuning 启动脚本创建总结

## 创建时间
2026-02-03

## 创建的文件

### 1. 核心脚本

#### `scripts/launch_rl_finetune.sh` (13KB, 481 行)
- **功能**: 一键启动 RL fine-tuning 训练的 bash 脚本
- **特性**:
  - 集中配置管理（所有参数在脚本顶部）
  - 6 种预设配置（baseline, freeze-encoders, no-domain-rand, fast-test, high-lr, conservative）
  - 自动环境检查（conda 环境、文件存在性）
  - Dry-run 模式（只显示命令不执行）
  - 配置摘要显示
  - 交互式确认
  - 自动激活 conda 环境
- **权限**: 可执行 (chmod +x)

### 2. 文档文件

#### `docs/RL_FINETUNE_LAUNCHER.md` (9.3KB)
- **内容**: 完整的使用文档
- **章节**:
  - 概述和基本使用
  - 可用预设详解
  - 配置参数说明
  - 典型工作流程
  - 高级用法（自定义预设）
  - 故障排除指南
  - 与直接调用 Python 的对比
  - 最佳实践

#### `docs/RL_FINETUNE_LAUNCHER_QUICKREF.md` (3.8KB)
- **内容**: 快速参考卡片
- **章节**:
  - 快速开始命令
  - 预设速查表
  - 关键配置项
  - 典型工作流
  - 预设详细配置
  - 故障排除速查
  - 自定义预设模板

#### `docs/RL_FINETUNE_CONFIG_EXAMPLES.md` (9.7KB)
- **内容**: 8 种场景的配置示例
- **场景**:
  1. 标准训练（推荐起点）
  2. 快速原型验证
  3. 保守微调（保护预训练知识）
  4. 激进微调（大幅改进性能）
  5. 消融实验 - 冻结编码器
  6. 消融实验 - 无 Domain Randomization
  7. 高性能训练（多 GPU）
  8. 长期稳定训练
- **附加内容**:
  - 参数调优指南
  - 实验对比建议
  - 配置检查清单

#### `docs/RL_FINETUNE_USAGE_EXAMPLES.md` (11KB)
- **内容**: 实际使用示例
- **示例**:
  1. 首次使用流程（5 个步骤）
  2. 消融实验流程（3 个实验）
  3. 参数调优流程（学习率调优）
  4. 快速迭代开发
  5. 批量实验
  6. 自定义预设
  7. 故障排除（3 个常见问题）
  8. 长期训练监控

#### `docs/RL_FINETUNE_LAUNCHER_README.md` (6.4KB)
- **内容**: 文档索引和导航
- **章节**:
  - 文件列表
  - 快速开始
  - 可用预设
  - 文档导航（"我想..."）
  - 典型使用场景
  - 脚本架构
  - 配置参数分类
  - 与其他文档的关系
  - 最佳实践
  - 更新日志

## 脚本功能详解

### 配置参数（50+ 个）

#### 基础配置
- `DAGGER_CHECKPOINT`: DAgger checkpoint 路径
- `CONDA_ENV`: Conda 环境名称
- `PYTHON_SCRIPT`: Python 脚本路径

#### 环境配置
- `NUM_ENVS`: 并行环境数量（默认 4096）
- `NUM_STEPS_PER_ENV`: 每环境步数（默认 24）
- `MAX_ITERATIONS`: 最大迭代次数（默认 1500）

#### PPO 超参数
- `LEARNING_RATE`: 学习率（默认 1e-5）
- `CLIP_PARAM`: PPO clip 参数（默认 0.2）
- `GAMMA`: 折扣因子（默认 0.99）
- `GAE_LAMBDA`: GAE lambda（默认 0.95）
- `VALUE_LOSS_COEF`: 价值损失系数（默认 1.0）
- `ENTROPY_COEF`: 熵损失系数（默认 0.01）
- `NUM_MINI_BATCHES`: Mini-batch 数量（默认 4）
- `NUM_LEARNING_EPOCHS`: 学习 epoch 数（默认 5）

#### 编码器配置
- `FREEZE_DEPTH_ENCODER`: 冻结深度编码器（默认 false）
- `FREEZE_PROPRIO_ENCODER`: 冻结本体感知编码器（默认 false）

#### Domain Randomization
- `ENABLE_DOMAIN_RAND`: 启用 DR（默认 true）
- `PUSH_ROBOT_PROB`: 推送概率（默认 0.1）
- `RANDOMIZE_FRICTION`: 随机化摩擦力（默认 true）
- `RANDOMIZE_MASS`: 随机化质量（默认 true）

#### 日志配置
- `EXPERIMENT_NAME`: 实验名称（默认自动生成）
- `SAVE_INTERVAL`: 保存间隔（默认 100）
- `LOG_INTERVAL`: 日志间隔（默认 10）

#### Wandb 配置
- `USE_WANDB`: 启用 wandb（默认 true）
- `WANDB_PROJECT`: Wandb 项目名称
- `WANDB_RUN_NAME`: Wandb 运行名称

### 预设配置（6 种）

| 预设 | 主要特点 | 适用场景 |
|-----|---------|---------|
| `baseline` | 默认配置 | 标准训练 |
| `freeze-encoders` | 冻结编码器，降低学习率 | 保护视觉特征 |
| `no-domain-rand` | 禁用所有 DR | 消融实验 |
| `fast-test` | 少量环境和迭代 | 快速验证 |
| `high-lr` | 高学习率 (5e-5) | 激进训练 |
| `conservative` | 低学习率，小 clip | 保守训练 |

### 脚本架构

```
launch_rl_finetune.sh (481 行)
│
├── 头部注释 (1-28 行)
│   ├── 功能说明
│   ├── 使用方法
│   └── 示例
│
├── 配置区域 (30-150 行)
│   ├── 基础配置
│   ├── 环境配置
│   ├── PPO 超参数
│   ├── 编码器配置
│   ├── Domain Randomization
│   ├── 日志配置
│   └── Wandb 配置
│
├── 预设配置函数 (152-220 行)
│   └── apply_preset()
│       ├── baseline
│       ├── freeze-encoders
│       ├── no-domain-rand
│       ├── fast-test
│       ├── high-lr
│       └── conservative
│
├── 辅助函数 (222-350 行)
│   ├── show_help()
│   ├── check_file_exists()
│   ├── check_conda_env()
│   ├── show_config_summary()
│   └── build_command()
│
└── 主程序 (352-481 行)
    └── main()
        ├── 解析命令行参数
        ├── 应用预设
        ├── 显示配置摘要
        ├── 检查环境
        ├── 构建命令
        ├── Dry-run 或执行
        └── 激活 conda 并运行
```

## 使用方式

### 基本命令

```bash
# 查看帮助
./scripts/launch_rl_finetune.sh --help

# 使用默认配置
./scripts/launch_rl_finetune.sh

# 使用预设
./scripts/launch_rl_finetune.sh [preset]

# Dry-run 模式
./scripts/launch_rl_finetune.sh --dry-run
./scripts/launch_rl_finetune.sh [preset] --dry-run
```

### 预设使用

```bash
# 标准训练
./scripts/launch_rl_finetune.sh baseline

# 冻结编码器
./scripts/launch_rl_finetune.sh freeze-encoders

# 禁用 DR
./scripts/launch_rl_finetune.sh no-domain-rand

# 快速测试
./scripts/launch_rl_finetune.sh fast-test

# 高学习率
./scripts/launch_rl_finetune.sh high-lr

# 保守训练
./scripts/launch_rl_finetune.sh conservative
```

## 文档导航

### 我想...

| 需求 | 推荐文档 |
|-----|---------|
| 快速开始使用 | `RL_FINETUNE_LAUNCHER_QUICKREF.md` |
| 了解所有功能 | `RL_FINETUNE_LAUNCHER.md` |
| 查看配置示例 | `RL_FINETUNE_CONFIG_EXAMPLES.md` |
| 学习实际使用 | `RL_FINETUNE_USAGE_EXAMPLES.md` |
| 浏览文档索引 | `RL_FINETUNE_LAUNCHER_README.md` |
| 解决问题 | `RL_FINETUNE_LAUNCHER.md` (故障排除) |

## 优势对比

### 使用启动脚本 vs 直接调用 Python

#### 使用启动脚本
```bash
./scripts/launch_rl_finetune.sh freeze-encoders
```

**优势**:
- 一行命令
- 配置集中管理
- 预设快速切换
- 自动环境检查
- 配置摘要显示
- 易于修改和扩展

#### 直接调用 Python
```bash
conda activate parkour && python scripts/rsl_rl/run_student_rl_finetune.py \
    --checkpoint logs/rsl_rl/parkour_student/2024-01-01_00-00-00/model_10000.pt \
    --num_envs 4096 \
    --num_steps_per_env 24 \
    --max_iterations 1500 \
    --learning_rate 5e-6 \
    --clip_param 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --value_loss_coef 1.0 \
    --entropy_coef 0.01 \
    --num_mini_batches 4 \
    --num_learning_epochs 5 \
    --freeze_depth_encoder \
    --freeze_proprio_encoder \
    --push_robot_prob 0.1 \
    --experiment_name freeze_encoders \
    --save_interval 100 \
    --log_interval 10 \
    --wandb \
    --wandb_project parkour-rl-finetune \
    --verbose
```

**劣势**:
- 命令冗长（20+ 个参数）
- 容易出错
- 难以记忆
- 不易修改
- 无配置摘要

## 典型工作流

### 1. 首次使用
```bash
# 步骤 1: 查看帮助
./scripts/launch_rl_finetune.sh --help

# 步骤 2: 配置 checkpoint
vim scripts/launch_rl_finetune.sh

# 步骤 3: Dry-run 测试
./scripts/launch_rl_finetune.sh --dry-run

# 步骤 4: 快速验证
./scripts/launch_rl_finetune.sh fast-test

# 步骤 5: 正式训练
./scripts/launch_rl_finetune.sh baseline
```

### 2. 消融实验
```bash
# 实验 1: Baseline
./scripts/launch_rl_finetune.sh baseline

# 实验 2: Freeze Encoders
./scripts/launch_rl_finetune.sh freeze-encoders

# 实验 3: No Domain Rand
./scripts/launch_rl_finetune.sh no-domain-rand
```

### 3. 参数调优
```bash
# 编辑配置
vim scripts/launch_rl_finetune.sh

# 测试新配置
./scripts/launch_rl_finetune.sh --dry-run

# 运行实验
./scripts/launch_rl_finetune.sh
```

## 扩展性

### 添加新预设

在 `apply_preset()` 函数中添加新 case：

```bash
my-preset)
    echo "应用预设: my-preset (我的自定义配置)"
    NUM_ENVS=2048
    LEARNING_RATE=2e-5
    FREEZE_DEPTH_ENCODER=true
    EXPERIMENT_NAME="my_preset"
    ;;
```

### 添加新参数

1. 在配置区域添加变量
2. 在 `build_command()` 函数中添加参数构建逻辑
3. 更新文档

## 最佳实践

1. **首次使用**: 先 dry-run，再 fast-test，最后正式训练
2. **配置管理**: 使用 git 跟踪配置变化
3. **实验管理**: 设置有意义的 EXPERIMENT_NAME
4. **故障排除**: 检查配置摘要，使用 dry-run 调试
5. **批量实验**: 创建批量脚本自动运行多个预设

## 相关文件

### 依赖的 Python 脚本
- `scripts/rsl_rl/run_student_rl_finetune.py`: 主启动脚本
- `scripts/rsl_rl/train_student_rl_finetune.py`: 训练器实现
- `scripts/rsl_rl/evaluate_student_robustness.py`: 评估脚本

### 相关文档
- `docs/phases/PHASE5_SUMMARY.md`: RL Fine-tuning 架构文档
- `docs/phases/PHASE5_QUICKREF.md`: RL Fine-tuning 快速参考
- `docs/phases/PHASE6_SUMMARY.md`: 评估和部署文档

## 测试状态

### 已测试功能
- ✅ 帮助信息显示
- ✅ Dry-run 模式
- ✅ 配置摘要显示
- ✅ 预设切换（baseline, freeze-encoders, fast-test）
- ✅ 环境检查（conda 环境、文件存在性）
- ✅ 命令构建

### 待测试功能
- ⏳ 实际训练执行（需要有效的 DAgger checkpoint）
- ⏳ Wandb 集成
- ⏳ 多 GPU 支持
- ⏳ 长期训练稳定性

## 未来改进

### 短期
1. 添加更多预设（如 multi-gpu, debug 等）
2. 支持从配置文件加载参数
3. 添加训练进度监控
4. 支持断点续训

### 长期
1. 图形化配置界面
2. 自动超参数调优
3. 实验结果自动分析
4. 与 wandb sweep 集成

## 总结

### 创建的价值

1. **简化使用**: 从 20+ 个参数的命令行简化为一行命令
2. **提高效率**: 预设配置快速切换，节省配置时间
3. **减少错误**: 集中配置管理，自动检查，降低出错概率
4. **易于维护**: 配置集中，易于修改和扩展
5. **完善文档**: 5 个文档文件，覆盖所有使用场景

### 适用场景

- ✅ 日常训练
- ✅ 消融实验
- ✅ 参数调优
- ✅ 快速原型验证
- ✅ 批量实验
- ✅ 教学演示

### 文件统计

- **脚本**: 1 个（481 行，13KB）
- **文档**: 5 个（共 39KB）
- **总计**: 6 个文件，52KB

### 文档覆盖

- ✅ 快速开始指南
- ✅ 完整使用文档
- ✅ 配置示例（8 种场景）
- ✅ 实际使用示例（8 个示例）
- ✅ 文档索引和导航
- ✅ 故障排除指南
- ✅ 最佳实践

## 下一步

1. **测试脚本**: 使用实际的 DAgger checkpoint 测试完整流程
2. **收集反馈**: 在实际使用中收集用户反馈
3. **迭代改进**: 根据反馈改进脚本和文档
4. **扩展功能**: 添加更多预设和高级功能

## 维护

### 更新频率
- 脚本: 根据需求更新
- 文档: 与脚本同步更新

### 版本控制
- 使用 git 跟踪所有变更
- 在文档中记录更新日志

### 责任人
- 脚本维护: 开发团队
- 文档维护: 开发团队

---

**创建日期**: 2026-02-03  
**版本**: 1.0  
**状态**: 已完成
