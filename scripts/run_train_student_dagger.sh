#!/usr/bin/env bash
# 用途：启动 DAGGER 学生训练，可在顶部切换教师-学生 mixture。

set -euo pipefail

# ========= 基本配置 =========
TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0"
TEACHER_CKPT="/path/to/teacher.pt"
STUDENT_INIT="/home/jing/Datasets/student_epoch_012300.pt"
NUM_ENVS=16
PROP_HIST_LEN=3
DEPTH_HIST_LEN=4
SEQUENCE_LENGTH=64
NUM_ITERS=2000
NUM_PRETRAIN_ITERS=200
LR=3e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_DIR="outputs/dagger_runs/run1"
DEVICE_ARG="cuda:0"

# ========= Mixture 开关与超参 =========
USE_MIXTURE=0             # 1 开启；0 关闭
MIX_BETA_START=0.6
MIX_BETA_END=0.1
MIX_DECAY_ITERS=800

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${USE_MIXTURE}" -eq 1 ]]; then
  MIXTURE_FLAG=(--teacher_mixture --teacher_mixture_beta_start "${MIX_BETA_START}" --teacher_mixture_beta_end "${MIX_BETA_END}" --teacher_mixture_decay_iters "${MIX_DECAY_ITERS}")
else
  MIXTURE_FLAG=()
fi

CMD=("${PYTHON_BIN}" "scripts/rsl_rl/train_student_dagger.py"
  "--task" "${TASK_ID}"
  "--num_envs" "${NUM_ENVS}"
  "--teacher_checkpoint" "${TEACHER_CKPT}"
  "--student_checkpoint" "${STUDENT_INIT}"
  "--num_iters" "${NUM_ITERS}"
  "--sequence_length" "${SEQUENCE_LENGTH}"
  "--prop_hist_len" "${PROP_HIST_LEN}"
  "--depth_hist_len" "${DEPTH_HIST_LEN}"
  "--num_pretrain_iters" "${NUM_PRETRAIN_ITERS}"
  "--learning_rate" "${LR}"
  "--weight_decay" "${WEIGHT_DECAY}"
  "--grad_clip" "${GRAD_CLIP}"
  "--save_dir" "${SAVE_DIR}"
  "--device" "${DEVICE_ARG}"
  "${MIXTURE_FLAG[@]}"
)

echo "[INFO] Running train_student_dagger: ${CMD[*]}"
"${CMD[@]}" "$@"
