#!/usr/bin/env bash
# 用途：以指定学生权重运行 play_dagger，便于快速验证在线推理表现。

set -euo pipefail

TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0"
STUDENT_CKPT="logs/rsl_rl/student_dagger_transformer/student-dagger-1-2/student_epoch_final.pt"
NUM_ENVS=16
PROP_HIST_LEN=3
DEPTH_HIST_LEN=4
SEQUENCE_LENGTH=64
MAX_STEPS=2000          # 0 表示跑到窗口关闭
DEVICE_ARG="cuda:0"
HEADLESS_FLAG=false     # GUI 模式设为 false；无界面设为 true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=("${PYTHON_BIN}" "scripts/rsl_rl/play_dagger.py"
  "--task" "${TASK_ID}"
  "--student_checkpoint" "${STUDENT_CKPT}"
  "--num_envs" "${NUM_ENVS}"
  "--prop_hist_len" "${PROP_HIST_LEN}"
  "--depth_hist_len" "${DEPTH_HIST_LEN}"
  "--sequence_length" "${SEQUENCE_LENGTH}"
  "--max_steps" "${MAX_STEPS}"
)

if [[ -n "${DEVICE_ARG}" ]]; then
  CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
  CMD+=("--headless")
else
  CMD+=("--enable_cameras")
fi

echo "[INFO] Running play_dagger: ${CMD[*]}"
"${CMD[@]}" "$@"
