#!/usr/bin/env bash
# Reproduce the optimization ablation behind Table tab:scheduling_ablation.
#
# Builds both binaries, installs each in turn (restarting PostgreSQL), and
# measures year_predict, slice, and swarm at low, medium, and high CPU
# contention. The whole run takes roughly an hour.
#
# Usage:
#   bash run_experiment.sh                 # full run
#   bash run_experiment.sh --dry-run       # print the steps, change nothing
#   bash run_experiment.sh --workloads slice,swarm
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

WORKLOADS="year_predict,slice,swarm"
DRY=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workloads) WORKLOADS="$2"; shift 2;;
        --dry-run)   DRY=(--dry-run); shift;;
        *) echo "unknown argument: $1" >&2; exit 2;;
    esac
done

echo "[1/3] compiling the load generators"
if [[ ${#DRY[@]} -eq 0 ]]; then
    gcc -O2 -o load/spin_load load/spin_load.c
    gcc -O2 -pthread -o load/wake_load load/wake_load.c
fi

echo "[2/3] building both pgdl binaries (source is restored afterwards)"
if [[ ${#DRY[@]} -eq 0 ]]; then
    python3 prepare_binary.py --which both
fi

echo "[3/3] measuring ${WORKLOADS}"
IFS=',' read -ra WANTED <<< "$WORKLOADS"
for optimization in on off; do
    python3 install_and_run.py \
        --optimization "$optimization" \
        --workloads "$(IFS=,; echo "${WANTED[*]}")" \
        --cooldown 0 \
        --repetitions 5 \
        "${DRY[@]}"
done

if [[ ${#DRY[@]} -eq 0 ]]; then
    LEFT="$(ps -eo comm | awk '$1=="spin_load" || $1=="wake_load"' | wc -l)"
    echo "leftover load processes: ${LEFT}"
    python3 fill_table_row.py || true
fi
