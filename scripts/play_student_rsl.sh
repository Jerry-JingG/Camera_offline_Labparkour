#!/usr/bin/env bash
# 用途：调用现有的 scripts/rsl_rl/play.py 跑学生策略（RSL-RL 版本）。
# 默认参数与手动运行：
#   python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16
# 并指定学生策略 checkpoint。

set -euo pipefail

TASK_ID="Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0"
NUM_ENVS=1
CHECKPOINT="/home/jing/IsaacLab/Camera_offline_Labparkour/logs/rsl_rl/unitree_go2_parkour/2026-05-12_21-23-50/model_99998.pt"

DEVICE_ARG="cuda:0"    # 可改为 cpu
HEADLESS_FLAG="${HEADLESS_FLAG:-false}"    # 无界面可设为 true
FREE_CAM="${FREE_CAM:-true}"               # 设置为 true 使用自由视角，false 则跟随机器人
REAL_TIME_FLAG="${REAL_TIME_FLAG:-false}"  # 如需实时播放设为 true
VIDEO_FLAG="${VIDEO_FLAG:-false}"          # 如需录制视频设为 true
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"

# 默认录制 env0 的 obs/action/depth，格式对齐 go2_parkour_deploy 的 parkour_obs_action + depth 目录。
# TERRAIN_TYPE 支持 parkour_hurdle / hurdle / parkour_gap / gap / parkour_step / step / parkour_flat / flat 等。
TERRAIN_TYPE="${TERRAIN_TYPE:-parkour_hurdle}"
RECORD_TRACE="${RECORD_TRACE:-true}"
RECORD_STEPS="${RECORD_STEPS:-300}"
RECORD_DEPTH_EVERY="${RECORD_DEPTH_EVERY:-5}"
RECORD_ENV_ID="${RECORD_ENV_ID:-0}"
DISABLE_DEPTH_DEBUG_VIS="${DISABLE_DEPTH_DEBUG_VIS:-true}"
STATIONARY_RECORD="${STATIONARY_RECORD:-false}"
STAND_WARMUP_STEPS="${STAND_WARMUP_STEPS:-100}"

if [[ "${STATIONARY_RECORD}" == true ]]; then
    RECORD_TRACE=true
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# 确保本地包可被找到（parkour_isaaclab 等未安装为 site-package 时）
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/parkour_tasks:${PYTHONPATH:-}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TERRAIN_LABEL="${TERRAIN_TYPE#parkour_}"
if [[ "${STATIONARY_RECORD}" == true ]]; then
    DEFAULT_RECORD_ROOT="${PROJECT_ROOT}/logs/isaacsim_stationary_zero_action"
else
    DEFAULT_RECORD_ROOT="${PROJECT_ROOT}/logs/isaacsim_${TERRAIN_LABEL}_play"
fi
RECORD_ROOT="${RECORD_ROOT:-${DEFAULT_RECORD_ROOT}}"
RECORD_CSV="${RECORD_CSV:-${RECORD_ROOT}/isaacsim_obs_action_${RUN_STAMP}.csv}"
RECORD_DEPTH_DIR="${RECORD_DEPTH_DIR:-${RECORD_ROOT}/isaacsim_depth_${RUN_STAMP}}"

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

if [[ -n "${TERRAIN_TYPE}" ]]; then
    CMD+=("--terrain_type" "${TERRAIN_TYPE}")
fi

if [[ "${DISABLE_DEPTH_DEBUG_VIS}" == true ]]; then
    CMD+=("--disable_depth_debug_vis")
fi

if [[ "${RECORD_TRACE}" == true ]]; then
    CMD+=(
        "--record_trace"
        "--record_csv" "${RECORD_CSV}"
        "--record_depth_dir" "${RECORD_DEPTH_DIR}"
        "--record_depth_every" "${RECORD_DEPTH_EVERY}"
        "--record_env_id" "${RECORD_ENV_ID}"
        "--record_steps" "${RECORD_STEPS}"
        "--record_foot_force_threshold" "2.0"
    )
fi

if [[ "${STATIONARY_RECORD}" == true ]]; then
    CMD+=("--stationary_record" "--stand_warmup_steps" "${STAND_WARMUP_STEPS}")
fi

echo "[INFO] Running student play (RSL-RL): ${CMD[*]}"
if [[ "${RECORD_TRACE}" == true ]]; then
    echo "[INFO] Recording CSV: ${RECORD_CSV}"
    echo "[INFO] Recording depth dir: ${RECORD_DEPTH_DIR}"
fi
if [[ "${STATIONARY_RECORD}" == true ]]; then
    echo "[INFO] Stationary zero-action recording enabled after ${STAND_WARMUP_STEPS} warmup steps"
fi
"${CMD[@]}"
