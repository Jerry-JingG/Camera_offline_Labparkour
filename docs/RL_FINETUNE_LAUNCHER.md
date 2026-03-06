# RL Fine-tuning 一键启动脚本使用指南

## 概述

`launch_rl_finetune.sh` 是一个用于快速启动 RL fine-tuning 训练的便捷脚本。它提供了集中的配置管理、多种预设配置、以及完善的错误检查功能。

## 脚本位置

```bash
scripts/launch_rl_finetune.sh
```

## 基本使用

### 1. 查看帮助信息

```bash
./scripts/launch_rl_finetune.sh --help
```

### 2. 使用默认配置启动

```bash
./scripts/launch_rl_finetune.sh
```

### 3. 使用预设配置

```bash
# 冻结编码器
./scripts/launch_rl_finetune.sh freeze-encoders

# 禁用 Domain Randomization
./scripts/launch_rl_finetune.sh no-domain-rand

# 快速测试（少量环境和迭代）
./scripts/launch_rl_finetune.sh fast-test

# 高学习率
./scripts/launch_rl_finetune.sh high-lr

# 保守训练
./scripts/launch_rl_finetune.sh conservative
```

### 4. Dry-run 模式（只显示命令不执行）

```bash
./scripts/launch_rl_finetune.sh --dry-run
./scripts/launch_rl_finetune.sh freeze-encoders --dry-run
```

## 可用预设

| 预设名称 | 说明 | 主要修改 |
|---------|------|---------|
| `baseline` | 默认配置 | 使用脚本中定义的默认值 |
| `freeze-encoders` | 冻结所有编码器 | 冻结深度和本体感知编码器，降低学习率 |
| `no-domain-rand` | 禁用 Domain Randomization | 关闭所有随机化选项 |
| `fast-test` | 快速测试 | 少量环境（512）、少量迭代（100）、禁用 wandb |
| `high-lr` | 高学习率 | 学习率设为 5e-5 |
| `conservative` | 保守训练 | 低学习率、小 clip 参数、低熵系数 |

## 配置参数

### 修改配置

编辑脚本顶部的配置区域（第 30-150 行左右）来修改参数：

```bash
# 打开脚本编辑
vim scripts/launch_rl_finetune.sh

# 或使用其他编辑器
nano scripts/launch_rl_finetune.sh
```

### 主要配置项

#### 1. 基础配置

```bash
# DAgger checkpoint 路径（必须存在）
DAGGER_CHECKPOINT="logs/rsl_rl/parkour_student/2024-01-01_00-00-00/model_10000.pt"

# Conda 环境名称
CONDA_ENV="parkour"
```

#### 2. 环境配置

```bash
# 并行环境数量
NUM_ENVS=4096

# 每个环境的步数
NUM_STEPS_PER_ENV=24

# 最大训练迭代次数
MAX_ITERATIONS=1500
```

#### 3. PPO 超参数

```bash
# 学习率
LEARNING_RATE=1e-5

# PPO clip 参数
CLIP_PARAM=0.2

# 折扣因子
GAMMA=0.99

# GAE lambda
GAE_LAMBDA=0.95

# Mini-batch 数量
NUM_MINI_BATCHES=4

# 每次迭代的学习 epoch 数
NUM_LEARNING_EPOCHS=5
```

#### 4. 编码器配置

```bash
# 是否冻结深度编码器
FREEZE_DEPTH_ENCODER=false

# 是否冻结本体感知编码器
FREEZE_PROPRIO_ENCODER=false
```

#### 5. Domain Randomization 配置

```bash
# 是否启用 Domain Randomization
ENABLE_DOMAIN_RAND=true

# 推送机器人的概率
PUSH_ROBOT_PROB=0.1

# 随机化摩擦力
RANDOMIZE_FRICTION=true

# 随机化质量
RANDOMIZE_MASS=true
```

#### 6. 日志和检查点配置

```bash
# 实验名称（留空则自动生成）
EXPERIMENT_NAME=""

# 保存检查点的间隔（迭代次数）
SAVE_INTERVAL=100

# 日志记录间隔（迭代次数）
LOG_INTERVAL=10
```

#### 7. Wandb 配置

```bash
# 是否启用 wandb
USE_WANDB=true

# Wandb 项目名称
WANDB_PROJECT="parkour-rl-finetune"

# Wandb 运行名称（留空则自动生成）
WANDB_RUN_NAME=""
```

## 工作流程

### 典型使用流程

1. **首次使用：配置 checkpoint 路径**

```bash
# 编辑脚本，修改 DAGGER_CHECKPOINT 路径
vim scripts/launch_rl_finetune.sh

# 找到这一行并修改为实际路径
DAGGER_CHECKPOINT="logs/rsl_rl/parkour_student/YOUR_ACTUAL_PATH/model_10000.pt"
```

2. **测试配置（dry-run）**

```bash
# 查看将要执行的命令
./scripts/launch_rl_finetune.sh --dry-run
```

3. **快速测试**

```bash
# 使用 fast-test 预设进行快速验证
./scripts/launch_rl_finetune.sh fast-test
```

4. **正式训练**

```bash
# 使用合适的预设开始训练
./scripts/launch_rl_finetune.sh baseline
```

### 实验对比流程

如果要进行消融实验，可以依次运行不同预设：

```bash
# 实验 1: 基线
./scripts/launch_rl_finetune.sh baseline

# 实验 2: 冻结编码器
./scripts/launch_rl_finetune.sh freeze-encoders

# 实验 3: 无 Domain Randomization
./scripts/launch_rl_finetune.sh no-domain-rand

# 实验 4: 高学习率
./scripts/launch_rl_finetune.sh high-lr
```

## 脚本特性

### 1. 自动环境检查

脚本会自动检查：
- Python 脚本是否存在
- Conda 环境是否存在
- DAgger checkpoint 是否存在（警告但可继续）

### 2. 配置摘要显示

启动前会显示完整的配置摘要：

```
================================================================================
配置摘要
================================================================================

基础配置:
  DAgger Checkpoint:     logs/rsl_rl/parkour_student/2024-01-01_00-00-00/model_10000.pt
  Conda 环境:            parkour
  实验名称:              自动生成

环境配置:
  并行环境数:            4096
  每环境步数:            24
  最大迭代次数:          1500

PPO 超参数:
  学习率:                1e-5
  Clip 参数:             0.2
  ...

================================================================================
```

### 3. 交互式确认

在执行训练前会要求确认：

```
是否开始训练? (Y/n)
```

### 4. 自动激活 Conda 环境

脚本会自动激活 `parkour` conda 环境，无需手动激活。

## 高级用法

### 创建自定义预设

编辑脚本中的 `apply_preset()` 函数，添加新的预设：

```bash
apply_preset() {
    local preset=$1
    
    case $preset in
        # ... 现有预设 ...
        
        my-custom-preset)
            echo "应用预设: my-custom-preset (我的自定义配置)"
            NUM_ENVS=2048
            LEARNING_RATE=2e-5
            FREEZE_DEPTH_ENCODER=true
            EXPERIMENT_NAME="my_custom"
            ;;
            
        *)
            echo "错误: 未知的预设 '$preset'"
            exit 1
            ;;
    esac
}
```

### 临时修改参数

如果只是临时修改某个参数，可以：

1. 复制脚本到临时文件
2. 修改配置
3. 运行临时脚本

```bash
cp scripts/launch_rl_finetune.sh /tmp/my_launch.sh
vim /tmp/my_launch.sh  # 修改配置
chmod +x /tmp/my_launch.sh
/tmp/my_launch.sh
```

### 批量实验

创建一个包装脚本来运行多个实验：

```bash
#!/bin/bash

# 运行多个预设的批量实验
for preset in baseline freeze-encoders no-domain-rand high-lr; do
    echo "开始实验: $preset"
    ./scripts/launch_rl_finetune.sh $preset
    
    # 等待一段时间或检查训练完成
    sleep 10
done
```

## 故障排除

### 问题 1: Conda 环境不存在

```
错误: Conda 环境 'parkour' 不存在
```

**解决方案**：
```bash
# 创建 conda 环境
conda create -n parkour python=3.8

# 或修改脚本中的 CONDA_ENV 变量
```

### 问题 2: DAgger checkpoint 不存在

```
警告: DAgger checkpoint 不存在: logs/rsl_rl/parkour_student/...
```

**解决方案**：
```bash
# 1. 检查实际的 checkpoint 路径
ls -la logs/rsl_rl/parkour_student/

# 2. 修改脚本中的 DAGGER_CHECKPOINT 变量
vim scripts/launch_rl_finetune.sh
```

### 问题 3: Python 脚本不存在

```
错误: Python 脚本 不存在: .../run_student_rl_finetune.py
```

**解决方案**：
```bash
# 确保在项目根目录运行脚本
cd /path/to/Camera_offline_Labparkour
./scripts/launch_rl_finetune.sh
```

### 问题 4: 权限问题

```
bash: ./scripts/launch_rl_finetune.sh: Permission denied
```

**解决方案**：
```bash
chmod +x scripts/launch_rl_finetune.sh
```

## 与直接调用 Python 脚本的对比

### 使用启动脚本

```bash
./scripts/launch_rl_finetune.sh freeze-encoders
```

### 直接调用 Python（等效命令）

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

**优势**：
- 启动脚本更简洁易用
- 配置集中管理
- 支持预设快速切换
- 自动环境检查
- 配置摘要显示

## 最佳实践

1. **首次使用前先 dry-run**
   ```bash
   ./scripts/launch_rl_finetune.sh --dry-run
   ```

2. **使用 fast-test 验证环境**
   ```bash
   ./scripts/launch_rl_finetune.sh fast-test
   ```

3. **为不同实验使用不同的 EXPERIMENT_NAME**
   - 编辑脚本设置 `EXPERIMENT_NAME`
   - 或使用预设（自动设置名称）

4. **定期备份配置**
   ```bash
   cp scripts/launch_rl_finetune.sh scripts/launch_rl_finetune.sh.backup
   ```

5. **使用版本控制跟踪配置变化**
   ```bash
   git diff scripts/launch_rl_finetune.sh
   ```

## 参考

- Python 脚本文档: `scripts/rsl_rl/run_student_rl_finetune.py`
- RL Fine-tuning 架构文档: `docs/phases/PHASE5_SUMMARY.md`
- 训练配置文档: `docs/phases/PHASE5_QUICKREF.md`
