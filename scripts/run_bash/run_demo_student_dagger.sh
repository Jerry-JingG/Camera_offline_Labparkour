#!/usr/bin/env bash
# 用途：使用 DAGGER 学生策略开启 IsaacLab demo UI（带手柄/键盘/相机控制）。

set -euo pipefail

TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0"
STUDENT_CKPT="logs/rsl_rl/student_dagger_transformer/student-dagger-dropout-1-6/student_epoch_final.pt"
NUM_ENVS=1
PROP_HIST_LEN=3
DEPTH_HIST_LEN=4
SEQUENCE_LENGTH=64
MAX_STEPS=0          # 0 表示一直运行直到关闭窗口
DEVICE_ARG="cuda:0"
HEADLESS_FLAG=false  # GUI 模式设为 false；无界面设为 true
INPUT_DEVICE="keyboard"  # gamepad|keyboard：选择控制输入源

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=("${PYTHON_BIN}" "scripts/rsl_rl/demo_dagger.py"
  "--task" "${TASK_ID}"
  "--student_checkpoint" "${STUDENT_CKPT}"
  "--num_envs" "${NUM_ENVS}"
  "--prop_hist_len" "${PROP_HIST_LEN}"
  "--depth_hist_len" "${DEPTH_HIST_LEN}"
  "--sequence_length" "${SEQUENCE_LENGTH}"
  "--max_steps" "${MAX_STEPS}"
  "--input_device" "${INPUT_DEVICE}"
)

if [[ -n "${DEVICE_ARG}" ]]; then
  CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
  CMD+=("--headless")
else
  CMD+=("--enable_cameras")
fi

echo "[INFO] Running DAGGER demo: ${CMD[*]}"
"${CMD[@]}" "$@"
