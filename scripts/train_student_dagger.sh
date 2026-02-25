#!/usr/bin/env bash
# 用途：启动基于多模态 Transformer 学生策略的在线 DAgger 训练。
#       学生策略结构与 train_student_from_dataset.py 中保持一致。
#       本脚本会调用 scripts/rsl_rl/train_student_dagger.py。

set -euo pipefail

# 防止显存碎片化导致 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------- 核心训练参数 ---------------------------------
TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Collect-v0"  # --task
NUM_ENVS=64                                                 # --num_envs：并行环境数量
NUM_ITERS=50000                                              # --num_iters：DAGGER 总迭代次数
NUM_PRETRAIN_ITERS=1000                                      # --num_pretrain_iters：预热迭代，前若干迭代由 Teacher 全程驾驶

SEQUENCE_LENGTH=128                                          # --sequence_length：TXL 序列长度 / mem_len
PROP_HIST_LEN=1                                             # --prop_hist_len：ProprioEncoder 的历史步数
DEPTH_HIST_LEN=1                                            # --depth_hist_len：DepthEncoder 的帧堆叠数

TEACHER_CHECKPOINT="logs/rsl_rl/unitree_go2_parkour/2026-01-28_22-35-55_try2/model_74000.pt"    # 教师 PPO 权重路径
STUDENT_CHECKPOINT=""                                           # 可选：已有学生模型 checkpoint
SAVE_DIR="outputs/students/trainxl_from_dataset/dagger0219"          # 输出目录

LEARNING_RATE=3e-4                                              # --learning_rate
WEIGHT_DECAY=1e-4                                               # --weight_decay
GRAD_CLIP=1.0                                                   # --grad_clip

# -------------------------- 教师-学生混合策略参数 -----------------------------
# USE_MIXTURE=true 开启 mixture；false 关闭。beta 线性从 start 衰减到 end。
USE_MIXTURE=true
BETA_START=0.8
BETA_END=0.0
BETA_DECAY_ITERS=3000

# -------------------------- 数据增强 (Dropout) -------------------------------
# 模拟摄像头/传感器掉线，强制学生学习鲁棒性
USE_DROPOUT=true

# ------------------------------- 日志 / W&B 参数 -------------------------------
USE_WANDB=true                                                 # --wandb：是否开启 W&B
WANDB_PROJECT="camera-offline-parkour"                          # --wandb_project
WANDB_RUN_NAME="dagger-txl-dropout-0219"                        # --wandb_run_name

# --------------------------- AppLauncher / Isaac 参数 -------------------------
DEVICE_ARG="cuda:0"                                             # --device
HEADLESS_FLAG=true                                              # --headless：是否无头运行
DISABLE_FABRIC_FLAG=false                                       # --disable_fabric

# ------------------------------- 运行前准备 -----------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ------------------------------- 构建命令行 -----------------------------------
DAGGER_CMD=("${PYTHON_BIN}" "scripts/rsl_rl/train_student_dagger.py"
    "--task" "${TASK_ID}"
    "--num_envs" "${NUM_ENVS}"
    "--teacher_checkpoint" "${TEACHER_CHECKPOINT}"
    "--num_iters" "${NUM_ITERS}"
    "--num_pretrain_iters" "${NUM_PRETRAIN_ITERS}"
    "--sequence_length" "${SEQUENCE_LENGTH}"
    "--prop_hist_len" "${PROP_HIST_LEN}"
    "--depth_hist_len" "${DEPTH_HIST_LEN}"
    "--learning_rate" "${LEARNING_RATE}"
    "--weight_decay" "${WEIGHT_DECAY}"
    "--grad_clip" "${GRAD_CLIP}"
    "--save_dir" "${SAVE_DIR}"
)

if [[ -n "${STUDENT_CHECKPOINT}" ]]; then
    DAGGER_CMD+=("--student_checkpoint" "${STUDENT_CHECKPOINT}")
fi

# Mixture 参数映射
if [[ "${USE_MIXTURE}" == true ]]; then
    DAGGER_CMD+=(
        "--teacher_mixture"
        "--beta_start" "${BETA_START}"
        "--beta_end" "${BETA_END}"
        "--beta_decay_iters" "${BETA_DECAY_ITERS}"
    )
fi

# Dropout 参数
if [[ "${USE_DROPOUT}" == true ]]; then
    DAGGER_CMD+=("--use_dropout")
fi

# WandB 参数映射
if [[ "${USE_WANDB}" == true ]]; then
    DAGGER_CMD+=("--wandb")
    if [[ -n "${WANDB_PROJECT}" ]]; then
        DAGGER_CMD+=("--wandb_project" "${WANDB_PROJECT}")
    fi
    if [[ -n "${WANDB_RUN_NAME}" ]]; then
        DAGGER_CMD+=("--wandb_run_name" "${WANDB_RUN_NAME}")
    fi
fi

# 通用参数
if [[ -n "${DEVICE_ARG}" ]]; then
    DAGGER_CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
    DAGGER_CMD+=("--headless")
fi

if [[ "${DISABLE_FABRIC_FLAG}" == true ]]; then
    DAGGER_CMD+=("--disable_fabric")
fi

# --------------------------------- 执行命令 -----------------------------------
echo "[INFO] Running student DAGGER training..."
echo "[CMD] ${DAGGER_CMD[*]}"
"${DAGGER_CMD[@]}"