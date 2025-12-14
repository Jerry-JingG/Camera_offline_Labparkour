#!/usr/bin/env bash
# 用途：配置并启动“泛环境越野”训练任务。
#       该脚本会调用 scripts/rsl_rl/train.py，并使用一个全新的 Gym 任务 ID，
#       目前内部复用跑酷 Teacher 的配置，后续可以在 offroad 配置中逐步替换。

set -euo pipefail

# ------------------------------- 核心训练参数 ---------------------------------
TASK_ID="Isaac-Generic-Offroad-Teacher-Unitree-Go2-v0"  # --task：泛环境越野 Teacher 训练任务 ID
NUM_ENVS=10                                          # --num_envs：并行环境数量（默认与跑酷一致）
SEED=1                                                 # --seed：随机种子

# 如需额外的 RSL-RL 配置（如 run_name / logger 等），可以在下方通过
# RSL_RL_ARGS 数组追加，例如：
# RSL_RL_ARGS=("--run_name" "debug_offroad" "--logger" "tensorboard")
RSL_RL_ARGS=()

# --------------------------- AppLauncher / Isaac 参数 -------------------------
DEVICE_ARG="cuda:0"                                    # --device：使用的 GPU / CPU 设备
HEADLESS_FLAG=false                                     # --headless：是否无头运行
DISABLE_FABRIC_FLAG=false                              # --disable_fabric：一般保持 false
ENABLE_CAMERAS_FLAG=false                              # --enable_cameras：是否强制启用摄像头

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
    "--num_envs" "${NUM_ENVS}"
    "--seed" "${SEED}"
)

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

# --------------------------------- 执行命令 -----------------------------------
echo "[INFO] Running generic offroad training: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"

