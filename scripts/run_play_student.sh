#!/usr/bin/env bash
# 用途：一键启动 play_student.py 运行学生策略推理

set -euo pipefail

# ========= 基本配置 =========
TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0"
# 二选一：指定检查点文件或检查点目录
STUDENT_CKPT="outputs/DAgger_ckpt/26_0109/student_epoch_final.pt"   # 具体检查点文件
# CKPT_DIR="outputs/student_runs/run1"                       # 或指定目录（自动选择最新）

NUM_ENVS=4
PROP_HIST_LEN=3
DEPTH_HIST_LEN=4
MEM_LEN=128
MAX_STEPS=5000
DEVICE_ARG="cuda:0"

# ========= 可视化配置 =========
HEADLESS=0   # 1 无头模式；0 带可视化

# ========= 脚本路径处理 =========
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ========= 构建命令行参数 =========
if [[ "${HEADLESS}" -eq 1 ]]; then
  HEADLESS_FLAG=("--headless")
else
  HEADLESS_FLAG=()
fi

# 选择检查点参数
if [[ -n "${STUDENT_CKPT:-}" ]]; then
  CKPT_FLAG=("--student_checkpoint" "${STUDENT_CKPT}")
elif [[ -n "${CKPT_DIR:-}" ]]; then
  CKPT_FLAG=("--checkpoint_dir" "${CKPT_DIR}")
else
  echo "[ERROR] 请设置 STUDENT_CKPT 或 CKPT_DIR"
  exit 1
fi

CMD=("${PYTHON_BIN}" "scripts/rsl_rl/play_student.py"
  "--task" "${TASK_ID}"
  "${CKPT_FLAG[@]}"
  "--num_envs" "${NUM_ENVS}"
  "--prop_hist_len" "${PROP_HIST_LEN}"
  "--depth_hist_len" "${DEPTH_HIST_LEN}"
  "--mem_len" "${MEM_LEN}"
  "--max_steps" "${MAX_STEPS}"
  "--device" "${DEVICE_ARG}"
  "${HEADLESS_FLAG[@]}"
)

echo "[INFO] Running play_student: ${CMD[*]}"
"${CMD[@]}" "$@"
