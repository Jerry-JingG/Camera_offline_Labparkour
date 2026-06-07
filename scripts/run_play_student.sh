#!/usr/bin/env bash
# Start Isaac play with the latest Camera_offline TXL student and record a MuJoCo-compatible trace.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("${PYTHON_BIN}")
else
    PYTHON_CMD=()
    for candidate in \
        "/home/jing/miniconda3/envs/Isaaclab/bin/python" \
        "/home/jing/miniconda3/envs/isaaclab_v2/bin/python" \
        "python"; do
        if [[ -x "${candidate}" ]] || command -v "${candidate}" >/dev/null 2>&1; then
            if "${candidate}" - <<'PY' >/dev/null 2>&1
import isaacsim  # noqa: F401
import isaaclab  # noqa: F401
PY
            then
                PYTHON_CMD=("${candidate}")
                break
            fi
        fi
    done
    if [[ "${#PYTHON_CMD[@]}" -eq 0 ]]; then
        PYTHON_CMD=("python")
    fi
fi

TASK_ID="${TASK_ID:-Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0}"
CAMERA_OFFLINE_ROOT="${CAMERA_OFFLINE_ROOT:-/home/jing/Camera_offline_Labparkour_new}"
STUDENT_CKPT="${STUDENT_CKPT:-${CAMERA_OFFLINE_ROOT}/outputs/students/train_from_dagger/xl0505_pro/student_dagger_49999.pt}"
DEVICE_ARG="${DEVICE_ARG:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
MEM_LEN="${MEM_LEN:-128}"

# Match go2_parkour_deploy/scripts/start_go2_camera_offline_mujoco.sh defaults.
COMMAND_X="${COMMAND_X:-0.8}"
TERRAIN_TYPE="${TERRAIN_TYPE:-parkour_flat}"
TERRAIN_FLAG="${TERRAIN_FLAG:-auto}"
SINGLE_SUBTERRAIN="${SINGLE_SUBTERRAIN:-1}"
SINGLE_SUBTERRAIN_BORDER_WIDTH="${SINGLE_SUBTERRAIN_BORDER_WIDTH:-0.0}"
SINGLE_SUBTERRAIN_DIFFICULTY="${SINGLE_SUBTERRAIN_DIFFICULTY:-0.8}"
USE_DROPOUT="${USE_DROPOUT:-0}"
CAMERA_OFFLINE_HOLD_STEPS="${CAMERA_OFFLINE_HOLD_STEPS:-0}"
CAMERA_OFFLINE_RAMP_STEPS="${CAMERA_OFFLINE_RAMP_STEPS:-0}"
CAMERA_OFFLINE_WARMUP_DELTA_LIMIT="${CAMERA_OFFLINE_WARMUP_DELTA_LIMIT:-0.0}"
CAMERA_OFFLINE_WARMUP_TAIL_STEPS="${CAMERA_OFFLINE_WARMUP_TAIL_STEPS:-0}"
ACTION_LPF_ALPHA="${ACTION_LPF_ALPHA:-1.0}"
ACTION_DELTA_LIMIT="${ACTION_DELTA_LIMIT:-0.0}"
ACTION_CLIP="${ACTION_CLIP:-0.0}"

HEADLESS="${HEADLESS:-0}"
FREE_CAM="${FREE_CAM:-0}"
DISABLE_DEPTH_DEBUG_VIS="${DISABLE_DEPTH_DEBUG_VIS:-1}"
USE_AGENT_CLIP_ACTIONS="${USE_AGENT_CLIP_ACTIONS:-0}"

RECORD_TRACE="${RECORD_TRACE:-1}"
RECORD_DEPTH_EVERY="${RECORD_DEPTH_EVERY:-5}"
RECORD_ENV_ID="${RECORD_ENV_ID:-0}"
RECORD_STEPS="${RECORD_STEPS:-500}"
RECORD_FOOT_FORCE_THRESHOLD="${RECORD_FOOT_FORCE_THRESHOLD:-2.0}"
RECORD_OBS_MODE="${RECORD_OBS_MODE:-mujoco_raw}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RECORD_ROOT="${RECORD_ROOT:-${PROJECT_ROOT}/logs/isaacsim_camera_offline_play}"
RECORD_CSV="${RECORD_CSV:-${RECORD_ROOT}/isaacsim_obs_action_${RUN_STAMP}.csv}"
RECORD_DEPTH_DIR="${RECORD_DEPTH_DIR:-${RECORD_ROOT}/isaacsim_depth_${RUN_STAMP}}"

ISAACLAB_ROOT=$(cd "${PROJECT_ROOT}/.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/parkour_tasks:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_rl:${ISAACLAB_ROOT}/source/isaaclab_mimic:${PYTHONPATH:-}"

CMD=("${PYTHON_CMD[@]}" "scripts/rsl_rl/play_camera_offline_student.py"
    "--task" "${TASK_ID}"
    "--camera_offline_root" "${CAMERA_OFFLINE_ROOT}"
    "--student_checkpoint" "${STUDENT_CKPT}"
    "--num_envs" "${NUM_ENVS}"
    "--max_steps" "${MAX_STEPS}"
    "--mem_len" "${MEM_LEN}"
    "--device" "${DEVICE_ARG}"
    "--command_x" "${COMMAND_X}"
    "--camera_offline_hold_steps" "${CAMERA_OFFLINE_HOLD_STEPS}"
    "--camera_offline_ramp_steps" "${CAMERA_OFFLINE_RAMP_STEPS}"
    "--camera_offline_warmup_delta_limit" "${CAMERA_OFFLINE_WARMUP_DELTA_LIMIT}"
    "--camera_offline_warmup_tail_steps" "${CAMERA_OFFLINE_WARMUP_TAIL_STEPS}"
    "--action_lpf_alpha" "${ACTION_LPF_ALPHA}"
    "--action_delta_limit" "${ACTION_DELTA_LIMIT}"
    "--action_clip" "${ACTION_CLIP}"
)

if [[ -n "${TERRAIN_TYPE}" ]]; then
    CMD+=("--terrain_type" "${TERRAIN_TYPE}")
fi
if [[ -n "${TERRAIN_FLAG}" ]]; then
    CMD+=("--terrain_flag" "${TERRAIN_FLAG}")
fi
if [[ "${SINGLE_SUBTERRAIN}" == "1" ]]; then
    CMD+=(
        "--single_subterrain"
        "--single_subterrain_border_width" "${SINGLE_SUBTERRAIN_BORDER_WIDTH}"
        "--single_subterrain_difficulty" "${SINGLE_SUBTERRAIN_DIFFICULTY}"
    )
fi

if [[ "${HEADLESS}" == "1" ]]; then
    CMD+=("--headless")
fi

if [[ "${FREE_CAM}" == "1" ]]; then
    CMD+=("--free_cam")
fi

if [[ "${DISABLE_DEPTH_DEBUG_VIS}" == "1" ]]; then
    CMD+=("--disable_depth_debug_vis")
fi

if [[ "${USE_DROPOUT}" == "1" ]]; then
    CMD+=("--use_dropout")
fi

if [[ "${USE_AGENT_CLIP_ACTIONS}" == "1" ]]; then
    CMD+=("--use_agent_clip_actions")
fi

if [[ "${RECORD_TRACE}" == "1" ]]; then
    CMD+=(
        "--record_trace"
        "--record_csv" "${RECORD_CSV}"
        "--record_depth_dir" "${RECORD_DEPTH_DIR}"
        "--record_depth_every" "${RECORD_DEPTH_EVERY}"
        "--record_env_id" "${RECORD_ENV_ID}"
        "--record_steps" "${RECORD_STEPS}"
        "--record_foot_force_threshold" "${RECORD_FOOT_FORCE_THRESHOLD}"
        "--record_obs_mode" "${RECORD_OBS_MODE}"
    )
fi

echo "[INFO] Running latest Camera_offline TXL student in Isaac:"
echo "[INFO] ${CMD[*]}"
if [[ "${RECORD_TRACE}" == "1" ]]; then
    echo "[INFO] Recording CSV: ${RECORD_CSV}"
    echo "[INFO] Recording depth dir: ${RECORD_DEPTH_DIR}"
fi

"${CMD[@]}" "$@"
