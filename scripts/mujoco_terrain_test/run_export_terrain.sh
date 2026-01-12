#!/bin/bash
# ============================================================================
# Multi-Env Terrain Generator for MuJoCo (with Beam Terrain)
# ============================================================================
# 快速调整参数生成跑酷地形
# Usage: ./run_export_terrain.sh
# ============================================================================

# ==================== 可调参数 ====================
# 网格配置
NUM_ROWS=3          # 行数（难度级别数）
NUM_COLS=5          # 列数（每种地形一列，按比例分配）

# 难度范围 (0.0 - 1.0)
DIFFICULTY_MIN=0.3  # 最低难度（第 0 行）
DIFFICULTY_MAX=0.8  # 最高难度（最后一行）

# 随机种子（改变种子会生成不同的地形）
SEED=42

# 是否自动打开 MuJoCo 查看器
AUTO_VIEW=true
# =================================================

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  Multi-Env Parkour Terrain Generator"
echo "  (with Beam Terrain - Jingg branch)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Grid:       ${NUM_ROWS} rows × ${NUM_COLS} cols = $((NUM_ROWS * NUM_COLS)) envs"
echo "  Difficulty: ${DIFFICULTY_MIN} - ${DIFFICULTY_MAX}"
echo "  Seed:       ${SEED}"
echo ""
echo "Terrain types: gap, hurdle, flat, step, beam (20% each)"
echo ""

# 生成地形
python export_multi_env_terrain.py \
    --num_rows "$NUM_ROWS" \
    --num_cols "$NUM_COLS" \
    --difficulty_min "$DIFFICULTY_MIN" \
    --difficulty_max "$DIFFICULTY_MAX" \
    --seed "$SEED"

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Terrain generation complete!"
    
    if [ "$AUTO_VIEW" = true ]; then
        echo ""
        echo "Opening MuJoCo viewer..."
        python -m mujoco.viewer output/multi_env_terrain_scene.xml
    fi
else
    echo ""
    echo "❌ Terrain generation failed!"
    exit 1
fi
