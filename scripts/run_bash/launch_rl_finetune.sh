#!/bin/bash

################################################################################
# RL Fine-tuning 一键启动脚本
# 
# 功能：
#   - 集中配置所有训练参数
#   - 支持配置预设快速切换
#   - 自动激活 conda 环境
#   - 参数验证和错误检查
#   - Dry-run 模式支持
#
# 使用方法：
#   ./scripts/launch_rl_finetune.sh [preset] [--dry-run]
#
# 预设选项：
#   baseline          - 默认配置
#   freeze-encoders   - 冻结编码器
#   no-domain-rand    - 禁用 Domain Randomization
#   fast-test         - 快速测试配置（少量环境和迭代）
#
# 示例：
#   ./scripts/launch_rl_finetune.sh
#   ./scripts/launch_rl_finetune.sh freeze-encoders
#   ./scripts/launch_rl_finetune.sh --dry-run
################################################################################

set -e  # 遇到错误立即退出

################################################################################
# 配置区域 - 在这里修改所有训练参数
################################################################################

# ============================================================================
# 基础配置
# ============================================================================

# DAgger checkpoint 路径（必须存在）
DAGGER_CHECKPOINT="outputs/DAgger_ckpt/26_0116/student_epoch_final.pt"

# Conda 环境名称
CONDA_ENV="parkour"

# 项目根目录（脚本所在目录的上一级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Python 脚本路径
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/rsl_rl/run_student_rl_finetune.py"

# ============================================================================
# 环境配置
# ============================================================================

# 并行环境数量
NUM_ENVS=128

# 每个环境的步数
NUM_STEPS_PER_ENV=24

# 最大训练迭代次数
MAX_ITERATIONS=1500

# ============================================================================
# PPO 超参数
# ============================================================================

# 学习率
LEARNING_RATE=1e-5

# PPO clip 参数
CLIP_PARAM=0.2

# 折扣因子
GAMMA=0.99

# GAE lambda
GAE_LAMBDA=0.95

# 价值函数损失系数
VALUE_LOSS_COEF=1.0

# 熵损失系数
ENTROPY_COEF=0.01

# Mini-batch 数量
NUM_MINI_BATCHES=4

# 每次迭代的学习 epoch 数
NUM_LEARNING_EPOCHS=5

# ============================================================================
# 编码器配置
# ============================================================================

# 是否冻结深度编码器
FREEZE_DEPTH_ENCODER=false

# 是否冻结本体感知编码器
FREEZE_PROPRIO_ENCODER=false

# ============================================================================
# Domain Randomization 配置
# ============================================================================

# 是否启用 Domain Randomization
ENABLE_DOMAIN_RAND=true

# 推送机器人的概率
PUSH_ROBOT_PROB=0.1

# 随机化摩擦力
RANDOMIZE_FRICTION=true

# 随机化质量
RANDOMIZE_MASS=true

# ============================================================================
# 日志和检查点配置
# ============================================================================

# 实验名称（留空则自动生成）
EXPERIMENT_NAME=""

# 保存检查点的间隔（迭代次数）
SAVE_INTERVAL=100

# 日志记录间隔（迭代次数）
LOG_INTERVAL=10

# ============================================================================
# TensorBoard 配置（替代 wandb）
# ============================================================================

# 是否启用 TensorBoard
USE_TENSORBOARD=true

# TensorBoard 刷新间隔（秒）
TENSORBOARD_FLUSH_SECS=10

# ============================================================================
# 其他配置
# ============================================================================

# 是否使用 GPU
USE_GPU=true

# 随机种子（-1 表示随机）
SEED=-1

# 是否显示详细日志
VERBOSE=true

# 是否使用 headless 模式（无 GUI）
HEADLESS=true

# 是否录制视频
RECORD_VIDEO=false

################################################################################
# 预设配置
################################################################################

apply_preset() {
    local preset=$1
    
    case $preset in
        baseline)
            echo "应用预设: baseline (默认配置)"
            # 使用上面定义的默认值
            ;;
            
        freeze-encoders)
            echo "应用预设: freeze-encoders (冻结所有编码器)"
            FREEZE_DEPTH_ENCODER=true
            FREEZE_PROPRIO_ENCODER=true
            LEARNING_RATE=5e-6  # 降低学习率
            EXPERIMENT_NAME="freeze_encoders"
            ;;
            
        no-domain-rand)
            echo "应用预设: no-domain-rand (禁用 Domain Randomization)"
            ENABLE_DOMAIN_RAND=false
            PUSH_ROBOT_PROB=0.0
            RANDOMIZE_FRICTION=false
            RANDOMIZE_MASS=false
            EXPERIMENT_NAME="no_domain_rand"
            ;;
            
        fast-test)
            echo "应用预设: fast-test (快速测试配置)"
            NUM_ENVS=512
            NUM_STEPS_PER_ENV=12
            MAX_ITERATIONS=100
            SAVE_INTERVAL=20
            LOG_INTERVAL=5
            USE_TENSORBOARD=false
            EXPERIMENT_NAME="fast_test"
            ;;
            
        high-lr)
            echo "应用预设: high-lr (高学习率)"
            LEARNING_RATE=5e-5
            EXPERIMENT_NAME="high_lr"
            ;;
            
        conservative)
            echo "应用预设: conservative (保守训练 - 针对 DAgger 模型微调优化)"
            LEARNING_RATE=3e-5  # 降低学习率
            CLIP_PARAM=0.1  # 更保守的 clip
            ENTROPY_COEF=0.001  # 降低熵系数
            NUM_LEARNING_EPOCHS=3  # 减少 epoch 数
            EXPERIMENT_NAME="conservative"
            ;;

        finetune)
            echo "应用预设: finetune (DAgger 模型微调专用配置)"
            LEARNING_RATE=3e-5  # 降低学习率，避免破坏预训练权重
            CLIP_PARAM=0.1  # 更保守的 clip，限制策略更新幅度
            ENTROPY_COEF=0.001  # 降低熵系数，避免鼓励增加噪声
            NUM_LEARNING_EPOCHS=3  # 减少 epoch 数，避免过度更新
            # 注意: desired_kl=0.12 和 init_noise_std=0.1 已在代码中设置
            EXPERIMENT_NAME="finetune"
            ;;
            
        *)
            echo "错误: 未知的预设 '$preset'"
            echo "可用预设: baseline, freeze-encoders, no-domain-rand, fast-test, high-lr, conservative, finetune"
            exit 1
            ;;
    esac
}

################################################################################
# 辅助函数
################################################################################

# 显示使用帮助
show_help() {
    cat << EOF
RL Fine-tuning 一键启动脚本

使用方法:
    $0 [preset] [options]

预设选项:
    baseline          - 默认配置
    freeze-encoders   - 冻结编码器
    no-domain-rand    - 禁用 Domain Randomization
    fast-test         - 快速测试配置
    high-lr           - 高学习率
    conservative      - 保守训练（针对 DAgger 模型微调优化）
    finetune          - DAgger 模型微调专用配置

选项:
    --dry-run         - 只显示命令，不执行
    --help, -h        - 显示此帮助信息

示例:
    $0                              # 使用默认配置
    $0 freeze-encoders              # 使用 freeze-encoders 预设
    $0 fast-test --dry-run          # Dry-run 模式
    
配置文件位置:
    编辑此脚本顶部的配置区域来修改参数

EOF
}

# 检查文件是否存在
check_file_exists() {
    local file=$1
    local description=$2
    
    if [ ! -f "$file" ]; then
        echo "错误: $description 不存在: $file"
        exit 1
    fi
}

# 检查 conda 环境是否存在
check_conda_env() {
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        echo "错误: Conda 环境 '$CONDA_ENV' 不存在"
        echo "请先创建环境: conda create -n $CONDA_ENV python=3.8"
        exit 1
    fi
}

# 显示配置摘要
show_config_summary() {
    cat << EOF

================================================================================
配置摘要
================================================================================

基础配置:
  DAgger Checkpoint:     $DAGGER_CHECKPOINT
  Conda 环境:            $CONDA_ENV
  实验名称:              ${EXPERIMENT_NAME:-自动生成}

环境配置:
  并行环境数:            $NUM_ENVS
  每环境步数:            $NUM_STEPS_PER_ENV
  最大迭代次数:          $MAX_ITERATIONS

PPO 超参数:
  学习率:                $LEARNING_RATE
  Clip 参数:             $CLIP_PARAM
  Gamma:                 $GAMMA
  GAE Lambda:            $GAE_LAMBDA
  Mini-batch 数:         $NUM_MINI_BATCHES
  Learning Epochs:       $NUM_LEARNING_EPOCHS

编码器配置:
  冻结深度编码器:        $FREEZE_DEPTH_ENCODER
  冻结本体感知编码器:    $FREEZE_PROPRIO_ENCODER

Domain Randomization:
  启用 DR:               $ENABLE_DOMAIN_RAND
  推送机器人概率:        $PUSH_ROBOT_PROB
  随机化摩擦力:          $RANDOMIZE_FRICTION
  随机化质量:            $RANDOMIZE_MASS

日志配置:
  保存间隔:              $SAVE_INTERVAL
  日志间隔:              $LOG_INTERVAL
  使用 TensorBoard:      $USE_TENSORBOARD

其他配置:
  Headless 模式:         $HEADLESS
  录制视频:              $RECORD_VIDEO
  使用 GPU:              $USE_GPU

================================================================================

EOF
}

# 构建命令行参数
build_command() {
    local cmd="python $PYTHON_SCRIPT train"

    # 基础参数
    cmd="$cmd --dagger_checkpoint $DAGGER_CHECKPOINT"
    cmd="$cmd --num_envs $NUM_ENVS"
    cmd="$cmd --num_steps_per_env $NUM_STEPS_PER_ENV"
    cmd="$cmd --max_iterations $MAX_ITERATIONS"

    # PPO 超参数
    cmd="$cmd --learning_rate $LEARNING_RATE"
    cmd="$cmd --clip_param $CLIP_PARAM"
    cmd="$cmd --gamma $GAMMA"
    cmd="$cmd --lam $GAE_LAMBDA"
    cmd="$cmd --value_loss_coef $VALUE_LOSS_COEF"
    cmd="$cmd --entropy_coef $ENTROPY_COEF"
    cmd="$cmd --num_mini_batches $NUM_MINI_BATCHES"
    cmd="$cmd --num_learning_epochs $NUM_LEARNING_EPOCHS"

    # 编码器配置
    if [ "$FREEZE_DEPTH_ENCODER" = true ] || [ "$FREEZE_PROPRIO_ENCODER" = true ]; then
        cmd="$cmd --freeze_encoders"
    fi

    # Domain Randomization
    if [ "$ENABLE_DOMAIN_RAND" = false ]; then
        cmd="$cmd --no_domain_rand"
    fi

    # 日志配置
    if [ -n "$EXPERIMENT_NAME" ]; then
        cmd="$cmd --experiment_name $EXPERIMENT_NAME"
    fi
    cmd="$cmd --save_interval $SAVE_INTERVAL"
    cmd="$cmd --log_interval $LOG_INTERVAL"

    # TensorBoard 配置
    if [ "$USE_TENSORBOARD" = true ]; then
        cmd="$cmd --tensorboard"
        cmd="$cmd --tensorboard_flush_secs $TENSORBOARD_FLUSH_SECS"
    fi

    # 其他配置
    if [ "$USE_GPU" = true ]; then
        cmd="$cmd --device cuda"
    else
        cmd="$cmd --device cpu"
    fi
    if [ $SEED -ge 0 ]; then
        cmd="$cmd --seed $SEED"
    fi

    # Headless 模式和视频录制
    if [ "$HEADLESS" = true ]; then
        cmd="$cmd --headless"
    fi
    if [ "$RECORD_VIDEO" = true ]; then
        cmd="$cmd --video"
    fi

    echo "$cmd"
}

################################################################################
# 主程序
################################################################################

main() {
    # 解析命令行参数
    local preset="baseline"
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            -*)
                echo "错误: 未知选项 $1"
                show_help
                exit 1
                ;;
            *)
                preset=$1
                shift
                ;;
        esac
    done
    
    # 应用预设
    apply_preset "$preset"
    
    # 显示配置摘要
    show_config_summary
    
    # 检查必要的文件和环境
    echo "检查环境..."
    check_file_exists "$PYTHON_SCRIPT" "Python 脚本"
    
    # 注意: DAgger checkpoint 可能还不存在，这里只是警告
    if [ ! -f "$DAGGER_CHECKPOINT" ]; then
        echo "警告: DAgger checkpoint 不存在: $DAGGER_CHECKPOINT"
        echo "请确保路径正确，或者训练完成后再运行"
        read -p "是否继续? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    check_conda_env
    
    # 构建命令
    local full_command
    full_command=$(build_command)
    
    echo "执行命令:"
    echo "  conda activate $CONDA_ENV && cd $PROJECT_ROOT && $full_command"
    echo
    
    # Dry-run 模式
    if [ "$dry_run" = true ]; then
        echo "Dry-run 模式: 不执行命令"
        exit 0
    fi
    
    # 确认执行
    read -p "是否开始训练? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "已取消"
        exit 0
    fi
    
    # 切换到项目根目录并执行
    cd "$PROJECT_ROOT"
    
    # 激活 conda 环境并执行命令
    echo "激活 conda 环境并开始训练..."
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    
    # 执行训练命令
    eval "$full_command"
}

# 运行主程序
main "$@"
