#!/usr/bin/env bash
# 用途：将 IsaacLab 的跑酷地形导出为 STL Mesh 文件，以便在 MuJoCo 等环境中使用。

set -euo pipefail

# --- 参数配置 ---

# 地形类型可选: gap, hurdle, step, beam, parkour, demo, flat
TERRAIN_TYPE="step"

# 难度系数: 0.0 到 1.0 之间
DIFFICULTY=0.5

# 保存文件名或路径
# 默认保存在当前目录下的 terrain.stl
OUTPUT_PATH="/home/jing/IsaacLab/mujoco_terrain/terrain_step.stl"

# --- 环境设置 ---

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

# 使用默认的 python 解释器
# 使用默认的 python 解释器
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- 执行命令 ---

CMD=("${PYTHON_BIN}" "scripts/export_terrain_mesh.py"
  "--type" "${TERRAIN_TYPE}"
  "--difficulty" "${DIFFICULTY}"
  "--output" "${OUTPUT_PATH}"
  "--headless"
)

echo "[INFO] 正在导出地形..."
echo "[INFO] 类型: ${TERRAIN_TYPE}"
echo "[INFO] 难度: ${DIFFICULTY}"
echo "[INFO] 路径: ${OUTPUT_PATH}"
echo "[INFO] 执行命令: ${CMD[*]}"

# 执行并透传额外的命令行参数
"${CMD[@]}" "$@"

echo "[SUCCESS] 地形导出完成！"
