#!/usr/bin/env bash
# 用途：配置并启动 Parkour Teacher 训练任务。
#       该脚本会调用 scripts/rsl_rl/train.py
#
#       参考命令：python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless

set -euo pipefail

# ------------------------------- 核心训练参数 ---------------------------------
TASK_ID="Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0"   # --task
SEED=1                                                     # --seed
HEADLESS_FLAG=true                                         # --headless

# --------------------------- 其他参数 (可按需修改) ----------------------------
# 如果需要指定并行环境数量，可以取消注释并设置数值
NUM_ENVS=4096                                           # --num_envs

# 如果需要指定设备，可以取消注释
DEVICE_ARG="cuda:0"                                      # --device

# ------------------------------- 日志 / W&B 参数 -------------------------------
LOGGER="wandb"                                                  # --logger：设置为 wandb 开启 W&B 记录，留空则关闭
LOG_PROJECT_NAME="parkour-teacher"                              # --log_project_name：W&B Project 名
RUN_NAME="1-29-teacher-2"                                           # --run_name：W&B run 名称前缀，可自定义/留空
# WANDB_API_KEY 如果此处不填，请确保环境变量中已设置
WANDB_API_KEY="85897bb211dff1da90eca7244d836724804604d2" 
WANDB_ENTITY="${WANDB_ENTITY:-}"                                # 可选：指定团队/空间

# 如需额外的 RSL-RL 配置（如 run_name / logger 等），可以在下方通过
# RSL_RL_ARGS 数组追加
RSL_RL_ARGS=()

# ------------------------------- 运行前准备 -----------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# 为避免 Isaac Sim 自带 wheel 与当前环境中的 numpy 等冲突，
# 复用 run_collect.sh 中的 pip_prebundle 逻辑，优先使用 Isaac Sim 的预打包依赖。
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

# ------------------------------- 构建命令行 -----------------------------------
TRAIN_CMD=("${PYTHON_BIN}" "scripts/rsl_rl/train.py"
    "--task" "${TASK_ID}"
    "--seed" "${SEED}"
)

if [[ "${HEADLESS_FLAG}" == true ]]; then
    TRAIN_CMD+=("--headless")
fi

if [[ -n "${NUM_ENVS:-}" ]]; then
    TRAIN_CMD+=("--num_envs" "${NUM_ENVS}")
fi

if [[ -n "${DEVICE_ARG:-}" ]]; then
    TRAIN_CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ ${#RSL_RL_ARGS[@]} -gt 0 ]]; then
    TRAIN_CMD+=("${RSL_RL_ARGS[@]}")
fi

if [[ -n "${LOGGER}" ]]; then
    TRAIN_CMD+=("--logger" "${LOGGER}")
fi
if [[ -n "${LOG_PROJECT_NAME}" ]]; then
    TRAIN_CMD+=("--log_project_name" "${LOG_PROJECT_NAME}")
fi
if [[ -n "${RUN_NAME}" ]]; then
    TRAIN_CMD+=("--run_name" "${RUN_NAME}")
fi

# 可选：如果设置了 WANDB_API_KEY 且不在 offline 模式，则尝试自动登录
if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "offline" ]]; then
    if command -v wandb >/dev/null 2>&1; then
        echo "[INFO] Attempting wandb login via CLI (WANDB_ENTITY=${WANDB_ENTITY:-unset})"
        # wandb CLI 用法：wandb login [KEY] [--entity xxx]
        if ! wandb login --relogin "${WANDB_API_KEY}" ${WANDB_ENTITY:+--entity "${WANDB_ENTITY}"}; then
            echo "[WARN] wandb login failed; continuing without CLI login."
        fi
    else
        echo "[WARN] wandb CLI not found; skipping wandb login."
    fi
fi

# --------------------------------- 执行命令 -----------------------------------
echo "[INFO] Running parkour teacher training: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
