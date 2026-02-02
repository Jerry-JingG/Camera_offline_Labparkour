#!/usr/bin/env bash
# 一键运行完整的 TorchScript 模型验证流程
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "   TorchScript Model Verification Flow"
echo "=========================================="

# 配置
STUDENT_CKPT="${STUDENT_CKPT:-logs/rsl_rl/student_dagger_transformer/student-dagger-dropout-1-16/student_epoch_final.pt}"
TORCHSCRIPT_OUT="${TORCHSCRIPT_OUT:-/tmp/torchscript_verify}"
VERIFY_DATA="${VERIFY_DATA:-obs_output/play_student_verify_data.bin}"
VERIFY_FRAMES="${VERIFY_FRAMES:-100}"
RL_SAR_ROOT="${RL_SAR_ROOT:-../rl_sar_yky/rl_sar}"

echo ""
echo "步骤 1/4: 导出 TorchScript 模型"
echo "----------------------------------------"
python scripts/rsl_rl/export_student_torchscript.py \
  --student_checkpoint "${STUDENT_CKPT}" \
  --out "${TORCHSCRIPT_OUT}"

echo ""
echo "步骤 2/4: 复制模型到 rl_sar 项目"
echo "----------------------------------------"
mkdir -p "${RL_SAR_ROOT}/policy/go2/parkour_student_torchscript"
cp "${TORCHSCRIPT_OUT}/student_policy.pt" \
   "${RL_SAR_ROOT}/policy/go2/parkour_student_torchscript/"
echo "已复制到: ${RL_SAR_ROOT}/policy/go2/parkour_student_torchscript/student_policy.pt"

echo ""
echo "步骤 3/4: 录制验证数据"
echo "----------------------------------------"
# 临时修改 run_play_student.sh
cp scripts/run_play_student.sh scripts/run_play_student.sh.tmp
sed -i 's/NUM_ENVS=4/NUM_ENVS=1/' scripts/run_play_student.sh
sed -i 's/MAX_STEPS=5000/MAX_STEPS='"${VERIFY_FRAMES}"'/' scripts/run_play_student.sh
sed -i 's/RECORD_DATA=0/RECORD_DATA=1/' scripts/run_play_student.sh

bash scripts/run_play_student.sh

# 恢复配置
mv scripts/run_play_student.sh.tmp scripts/run_play_student.sh

echo ""
echo "步骤 4/4: 运行 C++ LibTorch 验证"
echo "----------------------------------------"
cd "${RL_SAR_ROOT}"
MODEL_PATH="policy/go2/parkour_student_torchscript/student_policy.pt" \
VERIFY_DATA="${PROJECT_ROOT}/${VERIFY_DATA}" \
MAX_FRAMES="${VERIFY_FRAMES}" \
BACKEND="torch" \
bash scripts/verify_model_alignment.sh

echo ""
echo "=========================================="
echo "   验证流程完成"
echo "=========================================="
