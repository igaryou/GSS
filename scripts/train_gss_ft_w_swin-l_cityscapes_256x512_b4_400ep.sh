#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/igarashi_25/GSS"
CONFIG="$REPO_ROOT/configs/gss/cityscapes/gss-ft-w_swin-l_256x512_b4_400ep_cityscapes.py"
WORK_DIR="$REPO_ROOT/work_dirs/gss-ft-w_swin-l_256x512_b4_400ep_cityscapes"
PYTHON_BIN="${GSS_PYTHON:-/home/igarashi_25/anaconda3/envs/gss-py38/bin/python}"
STAGE1_CHECKPOINT="${1:-}"

if [[ -z "$STAGE1_CHECKPOINT" || ! -f "$STAGE1_CHECKPOINT" ]]; then
    echo "Usage: bash $0 /path/to/gss_ff_stage1_checkpoint.pth" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
bash tools/dist_train.sh "$CONFIG" 1 \
    --work-dir "$WORK_DIR" \
    --load-from "$STAGE1_CHECKPOINT"
