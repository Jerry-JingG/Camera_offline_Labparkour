#!/usr/bin/env bash
# 用途：配置并启动 scripts/rsl_rl/collect.py，统一管理采集所需的全部参数。

set -euo pipefail

# ------------------------------- 核心采集参数 ---------------------------------
TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0"  # --task：需要采集的 Gym 任务名
# 如果要使用域随机化， TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Collect-v0"
NUM_ENVS=16                                                    # --num_envs：并行环境数量


""" shard_size需要为train_student中sequence_length的整数倍! """

TOTAL_STEPS=5120                                               # --total_steps：总采集步数（一次 step 全部 env 同步计数）
SHARD_SIZE=1024                                                # --shard_size：每个数据分片包含的 step 数

OUTPUT_DIR="outputs/datasets/teacher_cam/noised_collect"                      # --out：数据输出目录
DEPTH_ENCODER_CKPT=""                                          # --depth-encoder-checkpoint：学生深度编码器权重（可为空）
LATENT_INTERVAL=5                                              # --latent-interval：深度 latent 更新间隔
DATASET_FORMAT="npz"                                           # --dataset-format：数据格式，目前仅支持 npz
DEPTH_DTYPE="uint16"                                          # --depth-dtype：深度图保存精度（float32 或 uint16）
DEPTH_SCALE=1000.0                                             # --depth-scale：当保存为 uint16 时的缩放倍数
VIDEO_FLAG=false                                               # --video：是否录制视频（true/false）
VIDEO_LENGTH=500                                               # --video_length：录制的视频长度
REALTIME_FLAG=false                                            # --real-time：是否按真实时间节奏采集
USE_PRETRAINED_FLAG=false                                      # --use_pretrained_checkpoint：是否改用官方预训练模型
CHECKPOINT_PATH="logs/rsl_rl/unitree_go2_parkour/251114_ckpt/model_49999.pt"  # --checkpoint：本地 checkpoint 路径
RESUME_DATASET_FLAG=false                                      # --resume_dataset：若目录存在是否继续追加采集

# ------------------------------- RSL-RL 额外参数 --------------------------------
# 示例：RSL_RL_ARGS=("--seed" "123" "--run_name" "collect_debug")
RSL_RL_ARGS=()                                                 # RSL-RL 相关可选参数列表

# --------------------------- AppLauncher/Isaac 参数 -----------------------------
DEVICE_ARG="cuda:0"                                           # --device：指定使用的 GPU / CPU 设备（留空则使用默认）
HEADLESS_FLAG=true                                             # --headless：是否启用无头模式运行 Isaac
DISABLE_FABRIC_FLAG=false                                      # --disable_fabric：是否关闭 Fabric（一般保持 false）
ENABLE_CAMERAS_FLAG=false                                      # --enable_cameras：是否强制启用摄像头渲染

# ------------------------------- 运行前检查 -------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="python"

# Isaac Sim's pip package bundles its own wheels (numpy==1.26.0, etc.) inside
# `isaacsim/extscache/.../pip_prebundle`. When a newer numpy is installed in the
# active conda env, importing Isaac extensions may mix both builds and trigger
# the "numpy.dtype size changed" error seen during collection. Force Python to
# prioritize the pip_prebundle path so that Isaac's wheels (and their compiled
# extensions) stay self-consistent.
ISAACSIM_PIP_PREBUNDLE="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec("isaacsim")
if spec is None or spec.origin is None:
    sys.exit(0)
isaacsim_dir = pathlib.Path(spec.origin).resolve().parent
extscache = isaacsim_dir / "extscache"
if not extscache.is_dir():
    sys.exit(0)
for candidate in sorted(extscache.glob("omni.kit.pip_archive*/pip_prebundle")):
    if candidate.is_dir():
        print(candidate)
        break
PY
)"
if [[ -n "${ISAACSIM_PIP_PREBUNDLE}" ]]; then
    if [[ ":${PYTHONPATH:-}:" != *":${ISAACSIM_PIP_PREBUNDLE}:"* ]]; then
        if [[ -n "${PYTHONPATH:-}" ]]; then
            export PYTHONPATH="${ISAACSIM_PIP_PREBUNDLE}:${PYTHONPATH}"
        else
            export PYTHONPATH="${ISAACSIM_PIP_PREBUNDLE}"
        fi
    fi
    echo "[INFO] Using Isaac Sim pip_prebundle: ${ISAACSIM_PIP_PREBUNDLE}"
else
    echo "[WARN] Unable to locate Isaac Sim pip_prebundle directory; proceeding without it."
fi

# ------------------------------- 构建命令行 -------------------------------------
COLLECT_CMD=("${PYTHON_BIN}" "scripts/rsl_rl/collect.py"
    "--task" "${TASK_ID}"
    "--num_envs" "${NUM_ENVS}"
    "--total_steps" "${TOTAL_STEPS}"
    "--shard_size" "${SHARD_SIZE}"
    "--out" "${OUTPUT_DIR}"
    "--latent-interval" "${LATENT_INTERVAL}"
    "--dataset-format" "${DATASET_FORMAT}"
    "--depth-dtype" "${DEPTH_DTYPE}"
    "--depth-scale" "${DEPTH_SCALE}"
    "--video_length" "${VIDEO_LENGTH}"
)

if [[ -n "${DEPTH_ENCODER_CKPT}" ]]; then
    COLLECT_CMD+=("--depth-encoder-checkpoint" "${DEPTH_ENCODER_CKPT}")
fi

if [[ -n "${CHECKPOINT_PATH}" ]]; then
    COLLECT_CMD+=("--checkpoint" "${CHECKPOINT_PATH}")
fi

if [[ "${VIDEO_FLAG}" == true ]]; then
    COLLECT_CMD+=("--video")
fi

if [[ "${REALTIME_FLAG}" == true ]]; then
    COLLECT_CMD+=("--real-time")
fi

if [[ "${USE_PRETRAINED_FLAG}" == true ]]; then
    COLLECT_CMD+=("--use_pretrained_checkpoint")
fi

if [[ "${RESUME_DATASET_FLAG}" == true ]]; then
    COLLECT_CMD+=("--resume_dataset")
fi

if [[ ${#RSL_RL_ARGS[@]} -gt 0 ]]; then
    COLLECT_CMD+=("${RSL_RL_ARGS[@]}")
fi

if [[ -n "${DEVICE_ARG}" ]]; then
    COLLECT_CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
    COLLECT_CMD+=("--headless")
fi

if [[ "${DISABLE_FABRIC_FLAG}" == true ]]; then
    COLLECT_CMD+=("--disable_fabric")
fi

if [[ "${ENABLE_CAMERAS_FLAG}" == true ]]; then
    COLLECT_CMD+=("--enable_cameras")
fi

# --------------------------------- 执行命令 -------------------------------------
echo "[INFO] Running: ${COLLECT_CMD[*]}"
"${COLLECT_CMD[@]}"
