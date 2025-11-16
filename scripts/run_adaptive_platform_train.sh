#!/usr/bin/env bash
# 用途：以统一的方式启动自定义的 Adaptive Platform 训练任务（底层仍调用 scripts/rsl_rl/train.py）。
#       修改下方变量即可快速切换任务、环境数量、运行名等参数。

set -euo pipefail

# ------------------------------- 核心训练参数 ---------------------------------
TASK_ID="Isaac-Adaptive-Platform-Teacher-Unitree-Go2-v0"  # --task：自定义 Gym 任务 ID
NUM_ENVS=""                                              # --num_envs：覆盖环境数（留空使用配置默认值）
MAX_ITERS=""                                             # --max_iterations：迭代次数（留空使用配置默认值）
SEED=42                                                  # --seed：环境与策略随机种子（留空不指定）
RUN_NAME="adaptive_platform_baseline"                    # --run_name：可选的运行名
DISTRIBUTED_FLAG=false                                   # --distributed：是否多 GPU/多节点训练

# ------------------------------- 录像相关参数 ---------------------------------
VIDEO_FLAG=false                                         # --video：是否录制训练视频
VIDEO_LENGTH=200                                         # --video_length：录制的视频长度
VIDEO_INTERVAL=2000                                      # --video_interval：录制间隔

# ------------------------------- RSL-RL 额外参数 --------------------------------
# 示例：RSL_RL_ARGS=("--empirical_normalization" "--resume")
RSL_RL_ARGS=()

# --------------------------- AppLauncher/Isaac 参数 -----------------------------
DEVICE_ARG="cuda:0"                                      # --device：指定 GPU/CPU（留空则使用默认值）
HEADLESS_FLAG=true                                       # --headless：是否无头模式
DISABLE_FABRIC_FLAG=false                                # --disable_fabric：是否关闭 Fabric
ENABLE_CAMERAS_FLAG=false                                # --enable_cameras：是否总是渲染相机

# ------------------------------- 运行前检查 -------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="scripts/rsl_rl/train.py"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] 找不到训练脚本：${TRAIN_SCRIPT}"
    exit 1
fi

# ------------------------------- 构建命令行 -------------------------------------
TRAIN_CMD=("${PYTHON_BIN}" "${TRAIN_SCRIPT}" "--task" "${TASK_ID}")

if [[ -n "${NUM_ENVS}" ]]; then
    TRAIN_CMD+=("--num_envs" "${NUM_ENVS}")
fi

if [[ -n "${MAX_ITERS}" ]]; then
    TRAIN_CMD+=("--max_iterations" "${MAX_ITERS}")
fi

if [[ -n "${SEED}" ]]; then
    TRAIN_CMD+=("--seed" "${SEED}")
fi

if [[ -n "${RUN_NAME}" ]]; then
    TRAIN_CMD+=("--run_name" "${RUN_NAME}")
fi

if [[ "${DISTRIBUTED_FLAG}" == true ]]; then
    TRAIN_CMD+=("--distributed")
fi

if [[ "${VIDEO_FLAG}" == true ]]; then
    TRAIN_CMD+=("--video")
fi

if [[ -n "${VIDEO_LENGTH}" ]]; then
    TRAIN_CMD+=("--video_length" "${VIDEO_LENGTH}")
fi

if [[ -n "${VIDEO_INTERVAL}" ]]; then
    TRAIN_CMD+=("--video_interval" "${VIDEO_INTERVAL}")
fi

if [[ ${#RSL_RL_ARGS[@]} -gt 0 ]]; then
    TRAIN_CMD+=("${RSL_RL_ARGS[@]}")
fi

if [[ -n "${DEVICE_ARG}" ]]; then
    TRAIN_CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
    TRAIN_CMD+=("--headless")
fi

if [[ "${DISABLE_FABRIC_FLAG}" == true ]]; then
    TRAIN_CMD+=("--disable_fabric")
fi

if [[ "${ENABLE_CAMERAS_FLAG}" == true ]]; then
    TRAIN_CMD+=("--enable_cameras")
fi

# --------------------------------- 执行命令 -------------------------------------
echo "[INFO] Running Adaptive Platform training: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
