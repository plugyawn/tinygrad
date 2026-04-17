#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SUBMISSION_PLATFORM="${SUBMISSION_PLATFORM:-tinybox_8xMI350X}"
export LOGMLPERF="${LOGMLPERF:-1}"
export RUNMLPERF="${RUNMLPERF:-1}"
export SEED="${SEED:-$RANDOM}"
export RUN_NAME="${RUN_NAME:-$(date "+%m%d%H%M")_${SEED}}"

LOGFILE="${LOGFILE:-flux_8xMI350x_${RUN_NAME}.log}"

"${SCRIPT_DIR}/dev_run.sh" | tee "${LOGFILE}"
