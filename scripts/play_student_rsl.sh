#!/usr/bin/env bash
# 用途：调用现有的 scripts/rsl_rl/play.py 跑学生策略（RSL-RL 版本）。
# 默认参数与手动运行：
#   python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16
# 并指定学生策略 checkpoint。

set -euo pipefail

TASK_ID="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0"
NUM_ENVS=16
CHECKPOINT="/home/jing/IsaacLab/Camera_offline_Labparkour/logs/rsl_rl/unitree_go2_parkour/2026-01-29_00-35-28_1-29-teacher-1/model_25900.pt"

DEVICE_ARG="cuda:0"    # 可改为 cpu
HEADLESS_FLAG=false    # 无界面可设为 true
FREE_CAM=true          # 设置为 true 使用自由视角，false 则跟随机器人
REAL_TIME_FLAG=false   # 如需实时播放设为 true
VIDEO_FLAG=false       # 如需录制视频设为 true
VIDEO_LENGTH=500

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# 确保本地包可被找到（parkour_isaaclab 等未安装为 site-package 时）
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/parkour_tasks:${PYTHONPATH:-}"

CMD=("${PYTHON_BIN}" "scripts/rsl_rl/play.py"
    "--task" "${TASK_ID}"
    "--num_envs" "${NUM_ENVS}"
    "--checkpoint" "${CHECKPOINT}"
)

if [[ -n "${DEVICE_ARG}" ]]; then
    CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
    CMD+=("--headless")
fi

if [[ "${FREE_CAM}" == true ]]; then
    CMD+=("--free_cam")
fi

if [[ "${REAL_TIME_FLAG}" == true ]]; then
    CMD+=("--real-time")
fi

if [[ "${VIDEO_FLAG}" == true ]]; then
    CMD+=("--video" "--video_length" "${VIDEO_LENGTH}")
fi

echo "[INFO] Running student play (RSL-RL): ${CMD[*]}"
"${CMD[@]}"
