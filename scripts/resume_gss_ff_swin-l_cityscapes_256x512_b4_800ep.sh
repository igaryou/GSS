#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/igarashi_25/GSS"
CONFIG="$REPO_ROOT/configs/gss/cityscapes/gss-ff_swin-l_256x512_b4_800ep_cityscapes.py"
WORK_DIR="$REPO_ROOT/work_dirs/gss-ff_swin-l_256x512_b4_800ep_cityscapes"
PYTHON_BIN="${GSS_PYTHON:-/home/igarashi_25/anaconda3/envs/gss-py38/bin/python}"
CHECKPOINT="${1:-}"

if [[ -z "$CHECKPOINT" && -d "$WORK_DIR" ]]; then
    CHECKPOINT="$(find "$WORK_DIR" -maxdepth 1 -type f \
        -name 'iter_*.pth' -print | sort -V | tail -n 1)"
fi
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found. Pass one explicitly or train Stage 1 first." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
bash tools/dist_train.sh "$CONFIG" 1 \
    --work-dir "$WORK_DIR" \
    --resume-from "$CHECKPOINT"
