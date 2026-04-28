#!/usr/bin/env bash
# 用途：启动基于多模态 Transformer-XL 学生策略的在线 PPO (RL) 微调训练。
#       该脚本会调用 scripts/txl_student/train_student_rl.py。

set -euo pipefail

# 防止显存碎片化导致 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------- 核心环境与网络参数 ---------------------------------
TASK_ID="Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Collect-v0"  # --task
NUM_ENVS=64                                                 # --num_envs：并行环境数量
NUM_ITERS=10000                                             # --num_iters：RL 总迭代次数

# HIST_LEN 必须 DAgger 预训练时完全一致
SEQUENCE_LENGTH=128                                         # --sequence_length：PPO Rollout 长度 & TXL mem_len
PROP_HIST_LEN=1                                             # --prop_hist_len：ProprioEncoder 的历史步数
DEPTH_HIST_LEN=1                                            # --depth_hist_len：DepthEncoder 的帧堆叠数

# ------------------------------- 权重与路径配置 -----------------------------------
TEACHER_CHECKPOINT="logs/rsl_rl/unitree_go2_parkour/260128_ckpt/model_74000.pt"
STUDENT_CHECKPOINT="outputs/students/train_from_dagger/xl0413/student_dagger_49999.pt"
STUDENT_IS_DAGGER=true                                      # --student_is_dagger：指明只加载 Actor，不加载 DAgger 的优化器
LOAD_TEACHER_CRITIC=true                                    # --load_teacher_critic：复用老师的 Critic 权重加速收敛

SAVE_DIR="outputs/students/rl_finetune/txl0426"                      # 输出目录
SAVE_INTERVAL=1000                                          # --save_interval

# ------------------------------- RL超参数 ---------------------------------------
ACTOR_LR=5e-6                                               # --actor_lr
CRITIC_LR=3e-5                                              # --critic_lr
WEIGHT_DECAY=1e-4                                           # --weight_decay
GRAD_CLIP=1.0                                               # --grad_clip

ENTROPY_COEF=0.01                                           # --entropy_coef：初期可稍微调大(如 0.01)鼓励探索
VALUE_LOSS_COEF=0.5                                         # --value_loss_coef
YAW_LOSS_COEF=0.2                                           # --yaw_loss_coef
GAMMA=0.99                                                  # --gamma
LAM=0.95                                                    # --lam

FREEZE_CRITIC_ITERS=0                                     # --freeze_critic_iters：冻结 Critic 前 N 轮

# -------------------------- 数据增强 (Dropout) -------------------------------
# 模拟摄像头/传感器掉线，强制学生学习鲁棒性
USE_DROPOUT=true

# ------------------------------- 日志 / W&B 参数 -------------------------------
USE_WANDB=true                                              # --wandb：是否开启 W&B
WANDB_PROJECT="camera-offline-parkour"                      # --wandb_project
WANDB_RUN_NAME="rl-txl-0426"                                # --wandb_run_name
LOG_INTERVAL=10                                             # --log_interval

# --------------------------- AppLauncher / Isaac 参数 -------------------------
DEVICE_ARG="cuda:0"                                         # --device
HEADLESS_FLAG=true                                          # --headless：是否无头运行
DISABLE_FABRIC_FLAG=false                                   # --disable_fabric

# ------------------------------- 运行前准备 -----------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ------------------------------- 构建命令行 -----------------------------------
RL_CMD=("${PYTHON_BIN}" "scripts/txl_student/train_student_rl.py"
    "--task" "${TASK_ID}"
    "--num_envs" "${NUM_ENVS}"
    "--teacher_checkpoint" "${TEACHER_CHECKPOINT}"
    "--num_iters" "${NUM_ITERS}"
    "--sequence_length" "${SEQUENCE_LENGTH}"
    "--prop_hist_len" "${PROP_HIST_LEN}"
    "--depth_hist_len" "${DEPTH_HIST_LEN}"
    
    "--actor_lr" "${ACTOR_LR}"
    "--critic_lr" "${CRITIC_LR}"
    "--weight_decay" "${WEIGHT_DECAY}"
    "--grad_clip" "${GRAD_CLIP}"
    
    "--entropy_coef" "${ENTROPY_COEF}"
    "--value_loss_coef" "${VALUE_LOSS_COEF}"
    "--yaw_loss_coef" "${YAW_LOSS_COEF}"
    "--gamma" "${GAMMA}"
    "--lam" "${LAM}"
    
    "--freeze_critic_iters" "${FREEZE_CRITIC_ITERS}"
    
    "--save_dir" "${SAVE_DIR}"
    "--save_interval" "${SAVE_INTERVAL}"
    "--log_interval" "${LOG_INTERVAL}"
    "--normalize_adv"
)

# 加载 Checkpoint 与 Teacher Critic
if [[ -n "${STUDENT_CHECKPOINT}" ]]; then
    RL_CMD+=("--student_checkpoint" "${STUDENT_CHECKPOINT}")
fi

if [[ "${STUDENT_IS_DAGGER}" == true ]]; then
    RL_CMD+=("--student_is_dagger")
fi

if [[ "${LOAD_TEACHER_CRITIC}" == true ]]; then
    RL_CMD+=("--load_teacher_critic")
fi

# Dropout 参数
if [[ "${USE_DROPOUT}" == true ]]; then
    RL_CMD+=("--use_dropout")
fi

# WandB 参数映射
if [[ "${USE_WANDB}" == true ]]; then
    RL_CMD+=("--wandb")
    if [[ -n "${WANDB_PROJECT}" ]]; then
        RL_CMD+=("--wandb_project" "${WANDB_PROJECT}")
    fi
    if [[ -n "${WANDB_RUN_NAME}" ]]; then
        RL_CMD+=("--wandb_run_name" "${WANDB_RUN_NAME}")
    fi
fi

# 通用参数
if [[ -n "${DEVICE_ARG}" ]]; then
    RL_CMD+=("--device" "${DEVICE_ARG}")
fi

if [[ "${HEADLESS_FLAG}" == true ]]; then
    RL_CMD+=("--headless")
fi

if [[ "${DISABLE_FABRIC_FLAG}" == true ]]; then
    RL_CMD+=("--disable_fabric")
fi

# --------------------------------- 执行命令 -----------------------------------
echo "[INFO] Running student PPO (RL) finetuning..."
echo "[CMD] ${RL_CMD[*]}"
"${RL_CMD[@]}"