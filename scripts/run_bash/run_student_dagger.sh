#!/usr/bin/env bash
# 用途：启动基于多模态 Transformer 学生策略的在线 DAGGER 训练。
#       学生策略结构与 train_student_from_dataset.py 中保持一致：
#       ProprioEncoder + DepthEncoder + MultiModalFusionTransformer + TransformerXLTemporal + JointPoseActionHead
#       本脚本会调用 scripts/rsl_rl/train_student_dagger.py。

set -euo pipefail

# 防止显存碎片化导致 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------- 核心训练参数 ---------------------------------
TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-v0"  # --task：带相机的跑酷 Teacher 任务
NUM_ENVS=256                                                # --num_envs：并行环境数量（先用较小并行数稳定调试）
NUM_ITERS=50000                                             # --num_iters：DAGGER 迭代次数（短程实验，确认策略再加大）
NUM_PRETRAIN_ITERS=1000                                     # --num_pretrain_iters：预热迭代，前若干迭代由 Teacher 全程驾驶

SEQUENCE_LENGTH=64                                          # --sequence_length：TXL 序列长度 / mem_len
PROP_HIST_LEN=1                                             # --prop_hist_len：ProprioEncoder 的历史步数
DEPTH_HIST_LEN=1                                            # --depth_hist_len：DepthEncoder 的帧堆叠数

TEACHER_CHECKPOINT="/home/jing/IsaacLab/Camera_offline_Labparkour/logs/rsl_rl/unitree_go2_parkour/2025-12-30_22-14-25_12-30-teacher-1/model_49999.pt"  # 预训练 Teacher PPO 权重路径
STUDENT_CHECKPOINT=""                                           # 可选：已有学生模型 checkpoint，用于继续 DAGGER 训练
BASE_SAVE_DIR="logs/rsl_rl/student_dagger_transformer"          # 学生模型基础输出目录

LEARNING_RATE=3e-4                                              # --learning_rate：学生优化器学习率
WEIGHT_DECAY=1e-4                                               # --weight_decay：AdamW 的 weight decay
GRAD_CLIP=1.0                                                   # --grad_clip：梯度裁剪阈值（L2 范数）

# -------------------------- 教师-学生混合策略参数 -----------------------------
# USE_MIXTURE=1 开启 mixture；0 关闭（传统 dagger）。beta 线性从 start 衰减到 end。
USE_MIXTURE=1
MIX_BETA_START=0.8
MIX_BETA_END=0.1
MIX_DECAY_ITERS=5000

# -------------------------- 相机掉线模拟参数 -----------------------------------
# 仅对学生施加随机相机掉线（全黑屏），教师始终看到干净深度。
# 设为 0.0 关闭掉线模拟；设为 >0 的概率值开启（如 0.3 表示 30%）。
CAMERA_DROPOUT_PROB=0.5

# 教师是否使用历史编码（hist_encoding）。开启后 teacher 标签使用 TXL 历史，贴近 train.py/distill 行为。
TEACHER_HIST_ENCODING=true                                      # --teacher_hist_encoding

# ------------------------------- 日志 / W&B 参数 -------------------------------
LOGGER="wandb"                                                  # --logger：设置为 wandb 开启 W&B 记录，留空则关闭
LOG_PROJECT_NAME="parkour-dagger"                              # --log_project_name：W&B Project 名（需先在网页创建）
RUN_NAME="student-dagger-dropout-2-1-pred-yaw"                                      # --run_name：W&B run 名称前缀，可自定义/留空
# 如需离线记录，可在运行前手动 export WANDB_MODE=offline；如需指定实体，可 export WANDB_ENTITY=your_team
# 如果只在本机使用且希望写死 Key，可在此填写；为空则使用环境变量或跳过。
WANDB_API_KEY="85897bb211dff1da90eca7244d836724804604d2"                             # 示例：WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
WANDB_ENTITY="${WANDB_ENTITY:-}"                               # 可选：指定团队/空间，留空走个人默认

# 将保存目录加上 run_name 子目录，便于分 run 归档
if [[ -n "${RUN_NAME}" ]]; then
    SAVE_DIR="${BASE_SAVE_DIR}/${RUN_NAME}"
else
    SAVE_DIR="${BASE_SAVE_DIR}"
fi

# --------------------------- AppLauncher / Isaac 参数 -------------------------
DEVICE_ARG="cuda:0"                                             # --device：使用的 GPU / CPU 设备
HEADLESS_FLAG=true                                              # --headless：是否无头运行
DISABLE_FABRIC_FLAG=false                                       # --disable_fabric：一般保持 false
ENABLE_CAMERAS_FLAG=false                                       # --enable_cameras：通常由任务配置自动开启相机
DISABLE_DEPTH_DEBUG_VIS_FLAG=true                               # 是否关闭深度相机调试窗口（cv2.imshow）

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
DAGGER_CMD=("${PYTHON_BIN}" "scripts/rsl_rl/train_student_dagger.py"
    "--task" "${TASK_ID}"
    "--num_envs" "${NUM_ENVS}"
    "--teacher_checkpoint" "${TEACHER_CHECKPOINT}"
    "--num_iters" "${NUM_ITERS}"
    "--num_pretrain_iters" "${NUM_PRETRAIN_ITERS}"
    "--sequence_length" "${SEQUENCE_LENGTH}"
    "--prop_hist_len" "${PROP_HIST_LEN}"
    "--depth_hist_len" "${DEPTH_HIST_LEN}"
    "--learning_rate" "${LEARNING_RATE}"
    "--weight_decay" "${WEIGHT_DECAY}"
    "--grad_clip" "${GRAD_CLIP}"
    "--save_dir" "${SAVE_DIR}"
)

if [[ -n "${STUDENT_CHECKPOINT}" ]]; then
    DAGGER_CMD+=("--student_checkpoint" "${STUDENT_CHECKPOINT}")
fi

if [[ "${USE_MIXTURE}" -eq 1 ]]; then
    DAGGER_CMD+=(
        "--teacher_mixture"
        "--teacher_mixture_beta_start" "${MIX_BETA_START}"
        "--teacher_mixture_beta_end" "${MIX_BETA_END}"
        "--teacher_mixture_decay_iters" "${MIX_DECAY_ITERS}"
    )
fi

if [[ -n "${DEVICE_ARG}" ]]; then
    DAGGER_CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${DISABLE_DEPTH_DEBUG_VIS_FLAG}" == true ]]; then
    DAGGER_CMD+=("--disable_depth_debug_vis")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
    DAGGER_CMD+=("--headless")
fi

if [[ "${DISABLE_FABRIC_FLAG}" == true ]]; then
    DAGGER_CMD+=("--disable_fabric")
fi

if [[ "${ENABLE_CAMERAS_FLAG}" == true ]]; then
    DAGGER_CMD+=("--enable_cameras")
fi

if [[ "${TEACHER_HIST_ENCODING:-false}" == true ]]; then
    DAGGER_CMD+=("--teacher_hist_encoding")
fi

if [[ -n "${LOGGER}" ]]; then
    DAGGER_CMD+=("--logger" "${LOGGER}")
fi
if [[ -n "${LOG_PROJECT_NAME}" ]]; then
    DAGGER_CMD+=("--log_project_name" "${LOG_PROJECT_NAME}")
fi
if [[ -n "${RUN_NAME}" ]]; then
    DAGGER_CMD+=("--run_name" "${RUN_NAME}")
fi

# 相机掉线模拟（仅对学生生效）
if [[ -n "${CAMERA_DROPOUT_PROB}" ]] && (( $(echo "${CAMERA_DROPOUT_PROB} > 0" | bc -l) )); then
    DAGGER_CMD+=("--camera_dropout_prob" "${CAMERA_DROPOUT_PROB}")
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
echo "[INFO] Running student DAGGER training: ${DAGGER_CMD[*]}"
"${DAGGER_CMD[@]}"
