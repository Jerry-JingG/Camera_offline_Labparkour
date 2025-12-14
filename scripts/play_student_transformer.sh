#!/usr/bin/env bash
# 用途：加载 Transformer 学生策略，在仿真环境中跑一段，观测行为 / 打印简单统计。

set -euo pipefail

TASK_ID="Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0"   # 目标验证环境
NUM_ENVS=1                                                    # 并行环境数，GUI 建议 1
STUDENT_CKPT="logs/rsl_rl/student_dagger_transformer/student_dagger_017200.pt"  # 学生模型权重

DEVICE_ARG="cuda:0"                                           # 推理设备
HEADLESS_FLAG=false                                        # GUI 运行设为 false；无界面设为 true
DISABLE_DEPTH_DEBUG_VIS_FLAG=true                             # 关闭 cv2 depth debug 窗口

# 学生模型输入配置（需与训练保持一致）
PROP_HIST_LEN=3
DEPTH_HIST_LEN=4
SEQUENCE_LENGTH=64

# 运行控制
MAX_STEPS=0          # 0 表示跑到窗口关闭；>0 表示跑固定步数后退出
PRINT_INTERVAL=200   # 每隔多少步打印一次 ep_return_mean / ep_len_mean

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=("${PYTHON_BIN}" "scripts/rsl_rl/play_student_transformer.py"
    "--task" "${TASK_ID}"
    "--num_envs" "${NUM_ENVS}"
    "--student_checkpoint" "${STUDENT_CKPT}"
    "--prop_hist_len" "${PROP_HIST_LEN}"
    "--depth_hist_len" "${DEPTH_HIST_LEN}"
    "--sequence_length" "${SEQUENCE_LENGTH}"
    "--print_interval" "${PRINT_INTERVAL}"
    )

if [[ -n "${DEVICE_ARG}" ]]; then
    CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
    CMD+=("--headless")
fi

if [[ "${DISABLE_DEPTH_DEBUG_VIS_FLAG}" == true ]]; then
    CMD+=("--disable_depth_debug_vis")
fi

if [[ "${MAX_STEPS}" -gt 0 ]]; then
    CMD+=("--max_steps" "${MAX_STEPS}")
fi

echo "[INFO] Running student play: ${CMD[*]}"
"${CMD[@]}"
