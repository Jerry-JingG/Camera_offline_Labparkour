#!/usr/bin/env bash
# 用途：配置并启动 scripts/rsl_rl/train_student_from_dataset.py，方便改参数后直接运行。

set -euo pipefail

# ------------------------------- 核心训练参数 ---------------------------------
DATASET_DIR="outputs/datasets/teacher_cam/noised_collection_1127"   # --dataset：collect.py 输出目录
DEVICE="cuda:0"                                     # --device：训练设备（如 cuda:0 / cpu）
NUM_EPOCHS=20                                        # --num_epochs：训练轮数
BATCH_SIZE=8                                        # --batch_size：每步优化的序列数
SEQUENCE_LENGTH=16                                  # --sequence_length：时间窗口长度
PROP_HIST_LEN=3                                     # --prop_hist_len：proprio 历史长度
DEPTH_HIST_LEN=4                                    # --depth_hist_len：depth 堆叠帧数
LEARNING_RATE=3e-4                                  # --learning_rate：学习率
WEIGHT_DECAY=1e-4                                   # --weight_decay：AdamW 权重衰减
GRAD_CLIP=1.0                                       # --grad_clip：梯度裁剪阈值（L2 norm）
LOG_INTERVAL=100                                    # --log_interval：打印/记录间隔（更新步）
MAX_SEQS_PER_EPOCH=                                 # --max_sequences_per_epoch：可选，限制每轮序列数
SAVE_DIR=""                                         # --save_dir：输出 checkpoint 目录（为空则写到 dataset/student_policy）
RESUME_CKPT=""                                      # --resume：可选，继续训练的 checkpoint 路径
USE_WANDB=true                                     # --use_wandb：是否启用 W&B

# ------------------------------- 可执行与路径 ---------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"
PYTHON_BIN="python"

# ------------------------------- 构建命令行 -----------------------------------
TRAIN_CMD=("${PYTHON_BIN}" "scripts/rsl_rl/train_student_from_dataset.py"
    "--dataset" "${DATASET_DIR}"
    "--device" "${DEVICE}"
    "--num_epochs" "${NUM_EPOCHS}"
    "--batch_size" "${BATCH_SIZE}"
    "--sequence_length" "${SEQUENCE_LENGTH}"
    "--prop_hist_len" "${PROP_HIST_LEN}"
    "--depth_hist_len" "${DEPTH_HIST_LEN}"
    "--learning_rate" "${LEARNING_RATE}"
    "--weight_decay" "${WEIGHT_DECAY}"
    "--grad_clip" "${GRAD_CLIP}"
    "--log_interval" "${LOG_INTERVAL}"
)

if [[ -n "${MAX_SEQS_PER_EPOCH}" ]]; then
    TRAIN_CMD+=("--max_sequences_per_epoch" "${MAX_SEQS_PER_EPOCH}")
fi

if [[ -n "${SAVE_DIR}" ]]; then
    TRAIN_CMD+=("--save_dir" "${SAVE_DIR}")
fi

if [[ -n "${RESUME_CKPT}" ]]; then
    TRAIN_CMD+=("--resume" "${RESUME_CKPT}")
fi

if [[ "${USE_WANDB}" == true ]]; then
    TRAIN_CMD+=("--use_wandb")
fi

# --------------------------------- 执行命令 -----------------------------------
echo "[INFO] Running: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
