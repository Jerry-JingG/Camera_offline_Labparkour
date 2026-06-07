#!/usr/bin/env bash
# 用途：以指定学生权重运行 play_dagger，便于快速验证在线推理表现。

set -euo pipefail

TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0"
STUDENT_CKPT="logs/rsl_rl/student_dagger_transformer/student-dagger-dropout-1-32-pred-yaw/student_epoch_035100.pt"
NUM_ENVS=16
PROP_HIST_LEN=1
DEPTH_HIST_LEN=1
SEQUENCE_LENGTH=64
MAX_STEPS=1000          # 0 表示跑到窗口关闭
DEVICE_ARG="cuda:0"
HEADLESS_FLAG=false     # GUI 模式设为 false；无界面设为 true

# ========== 录制验证数据配置 ==========
# 设为 true 时，会录制 env_id=0 的 (proprio, depth, action) 数据
# 用于与 MuJoCo C++ 推理结果进行对齐验证
RECORD_DATA=true
# 录制数据的输出路径 (二进制格式)
RECORD_OUTPUT_PATH="/home/jing/IsaacLab/Camera_offline_Labparkour/obs_output/pred_yaw_verify_data.bin"
# =====================================

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

# 录制数据参数
if [[ "${RECORD_DATA}" == true ]]; then
  CMD+=("--record_data" "--record_output" "${RECORD_OUTPUT_PATH}")
fi

echo "[INFO] Running play_dagger: ${CMD[*]}"
"${CMD[@]}" "$@"
