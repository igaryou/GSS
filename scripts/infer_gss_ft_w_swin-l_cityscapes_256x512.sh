#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/igarashi_25/GSS"
CONFIG="$REPO_ROOT/configs/gss/cityscapes/gss-ft-w_swin-l_256x512_b4_400ep_cityscapes.py"
OUTPUT_DIR="$REPO_ROOT/inference_results/gss-ft-w_swin-l_256x512"
PYTHON_BIN="${GSS_PYTHON:-/home/igarashi_25/anaconda3/envs/gss-py38/bin/python}"
CHECKPOINT="${1:-}"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Usage: bash $0 /path/to/stage2_or_merged_checkpoint.pth" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" -c \
    "from mmcv import Config; \
from mmseg.datasets import build_dataloader, build_dataset; \
cfg=Config.fromfile('$CONFIG'); \
dataset=build_dataset(cfg.data.test, dict(test_mode=True)); \
loader=build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, \
num_gpus=1, dist=False, shuffle=False, persistent_workers=False); \
shape=tuple(next(iter(loader))['img'][0].shape); \
print('first inference batch image shape:', shape); \
assert shape == (1, 3, 256, 512), shape"

"$PYTHON_BIN" tools/test.py "$CONFIG" "$CHECKPOINT" \
    --show-dir "$OUTPUT_DIR" \
    --opacity 0.5
