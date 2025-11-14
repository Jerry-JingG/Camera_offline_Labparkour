#!/usr/bin/env bash
# 用途：在 IsaacLab Parkour 工程中启动多模态 Transformer 的冒烟测试。
#       该脚本会调用 extreme_parkour_task 下的 pipeline_smoketest.py，
#       以便快速验证 tokenizer + fusion + temporal 模块是否工作正常。

set -euo pipefail

# ------------------------------- 可选自定义参数 ---------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"  # 可通过环境变量覆盖使用的 Python 解释器
EXTRA_ARGS=()                       # 预留给 pipeline_smoketest.py 的额外参数（当前脚本未使用）

# ------------------------------- 路径准备 ---------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

SMOKETEST_SCRIPT="/home/droplet/IsaacLab/Camera_offline_Labparkour/parkour_tasks/parkour_tasks/extreme_parkour_task/modules/tests/pipeline_smoketest.py"

if [[ ! -f "${SMOKETEST_SCRIPT}" ]]; then
    echo "[ERROR] 找不到冒烟测试脚本：${SMOKETEST_SCRIPT}"
    exit 1
fi

# ------------------------------- 运行命令 ---------------------------------------
CMD=("${PYTHON_BIN}" "${SMOKETEST_SCRIPT}")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] Running transformer smoketest: ${CMD[*]}"
"${CMD[@]}"
