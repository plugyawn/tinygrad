#!/usr/bin/env bash
set -euo pipefail

CALLER_CWD="${PWD}"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../../../../../" && pwd)"

resolve_from_caller() {
  local path="$1"
  if [[ -z "${path}" || "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${CALLER_CWD}" "${path}"
  fi
}

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DEV="${DEV:-AMD}"
export MODEL="flux"

export CHECK_OOB="${CHECK_OOB:-0}"
export REWRITE_STACK_LIMIT="${REWRITE_STACK_LIMIT:-5000000}"
export HCQDEV_WAIT_TIMEOUT_MS="${HCQDEV_WAIT_TIMEOUT_MS:-300000}"
export AMD_LLVM="${AMD_LLVM:-0}"

export DEFAULT_FLOAT="${DEFAULT_FLOAT:-bfloat16}"
export GPUS="${GPUS:-8}"
export BS="${BS:-128}"
export EVAL_BS="${EVAL_BS:-$BS}"
export LR="${LR:-1e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-1000}"
export MAX_NORM="${MAX_NORM:-1.0}"
export SEED="${SEED:-$RANDOM}"

export BEAM="${BEAM:-2}"
export TRAIN_BEAM="${TRAIN_BEAM:-$BEAM}"
export EVAL_BEAM="${EVAL_BEAM:-$BEAM}"
export BEAM_UOPS_MAX="${BEAM_UOPS_MAX:-8000}"
export BEAM_UPCAST_MAX="${BEAM_UPCAST_MAX:-256}"
export BEAM_LOCAL_MAX="${BEAM_LOCAL_MAX:-1024}"
export BEAM_MIN_PROGRESS="${BEAM_MIN_PROGRESS:-5}"
export IGNORE_JIT_FIRST_BEAM="${IGNORE_JIT_FIRST_BEAM:-1}"

export DATADIR="${DATADIR:-/raid/datasets/flux}"
export TRAIN_DATASET="${TRAIN_DATASET:-${DATADIR}/cc12m_preprocessed/*}"
export VAL_DATASET="${VAL_DATASET:-${DATADIR}/coco_preprocessed/*}"
export PRETRAINED="${PRETRAINED:-}"

export SEED="${SEED:-$RANDOM}"
RUN_NAME="${RUN_NAME:-$(date "+%m%d%H%M%S")_${SEED}}"
export CKPT_ROOT="${CKPT_ROOT:-/raid/weights/flux}"
export SAVE_CKPT_DIR="${SAVE_CKPT_DIR:-${CKPT_ROOT}/training_checkpoints/${RUN_NAME}}"
export EVAL_CKPT_DIR="${EVAL_CKPT_DIR:-$SAVE_CKPT_DIR}"

# Mirror stable diffusion's offline submission flow: train only writes checkpoints, eval scans them afterwards.
export CKPT="${CKPT:-1}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
export STOP_IF_CONVERGED="${STOP_IF_CONVERGED:-1}"
export TOTAL_CKPTS="${TOTAL_CKPTS:-7}"

if [[ -z "${CKPT_INTERVAL:-}" ]]; then
  export CKPT_INTERVAL=$(((512000 + BS - 1) / BS))
fi
if [[ -z "${TRAIN_STEPS:-}" && -n "${TOTAL_CKPTS}" ]]; then
  export TRAIN_STEPS=$((CKPT_INTERVAL * TOTAL_CKPTS))
fi

export DATADIR="$(resolve_from_caller "${DATADIR}")"
export TRAIN_DATASET="$(resolve_from_caller "${TRAIN_DATASET}")"
export VAL_DATASET="$(resolve_from_caller "${VAL_DATASET}")"
export PRETRAINED="$(resolve_from_caller "${PRETRAINED}")"
export CKPT_ROOT="$(resolve_from_caller "${CKPT_ROOT}")"
export SAVE_CKPT_DIR="$(resolve_from_caller "${SAVE_CKPT_DIR}")"
export EVAL_CKPT_DIR="$(resolve_from_caller "${EVAL_CKPT_DIR}")"

mkdir -p "${SAVE_CKPT_DIR}"

echo "running flux train with checkpoints in ${SAVE_CKPT_DIR}"
echo "train_dataset=${TRAIN_DATASET}"
echo "val_dataset=${VAL_DATASET}"
echo "BS=${BS} EVAL_BS=${EVAL_BS} CKPT_INTERVAL=${CKPT_INTERVAL} TRAIN_STEPS=${TRAIN_STEPS:-default}"

python3 "${REPO_ROOT}/examples/mlperf/model_train.py"

if [[ "${EVAL_INTERVAL}" != "0" ]]; then
  exit 0
fi

shopt -s nullglob
flux_ckpts=("${EVAL_CKPT_DIR}"/flux_step*.safetensors)
if (( ${#flux_ckpts[@]} == 0 )); then
  echo "no flux_step checkpoints found in ${EVAL_CKPT_DIR}; adjust CKPT, CKPT_INTERVAL, TOTAL_CKPTS, or TRAIN_STEPS" >&2
  exit 1
fi

echo "evaluating checkpoints from ${EVAL_CKPT_DIR}"
BS="${EVAL_BS}" python3 "${REPO_ROOT}/examples/mlperf/model_eval.py"
