#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/igarashi_25/GSS"
CONFIG="$REPO_ROOT/configs/gss/cityscapes/gss-ff_r101_256x512_b4_800ep_cityscapes.py"
SMOKE_DIR="$REPO_ROOT/work_dirs/smoke_gss-ff_r101_256x512_b4"
SHOW_DIR="$SMOKE_DIR/visualization"
PYTHON_BIN="${GSS_PYTHON:-/home/igarashi_25/anaconda3/envs/gss-py38/bin/python}"
SMOKE_SPLIT="$(mktemp /tmp/gss-cityscapes-val-one.XXXXXX.txt)"
trap 'rm -f "$SMOKE_SPLIT"' EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
find /home/igarashi_25/datasets/cityscapes/leftImg8bit/val \
    -type f -name '*_leftImg8bit.png' | sort | sed -n '1p' | \
    sed -e 's#^.*/val/##' -e 's#_leftImg8bit\.png$##' > "$SMOKE_SPLIT"

"$PYTHON_BIN" -c \
    "from mmcv import Config; \
from mmseg.datasets import build_dataloader, build_dataset; \
cfg=Config.fromfile('$CONFIG'); \
dataset=build_dataset(cfg.data.train); \
loader=build_dataloader(dataset, samples_per_gpu=cfg.data.samples_per_gpu, \
workers_per_gpu=0, num_gpus=1, dist=True, shuffle=True, drop_last=True, \
persistent_workers=False); \
batch=next(iter(loader)); \
img_shape=tuple(batch['img'].data[0].shape); \
gt_shape=tuple(batch['gt_semantic_seg'].data[0].shape); \
print('dataset length:', len(dataset)); \
print('train dataloader length:', len(loader)); \
print('first train image shape:', img_shape); \
print('first train GT shape:', gt_shape); \
assert len(dataset) == 2975; \
assert len(loader) == 743; \
assert img_shape == (4, 3, 256, 512), img_shape; \
assert gt_shape == (4, 1, 256, 512), gt_shape"

bash tools/dist_train.sh "$CONFIG" 1 \
    --work-dir "$SMOKE_DIR" \
    --cfg-options \
    runner.max_iters=2 \
    checkpoint_config.interval=2 \
    checkpoint_config.max_keep_ckpts=1 \
    evaluation.interval=2 \
    log_config.interval=1 \
    data.workers_per_gpu=2 \
    data.val.split="$SMOKE_SPLIT"

CHECKPOINT="$SMOKE_DIR/iter_2.pth"
test -f "$CHECKPOINT"
mkdir -p "$SHOW_DIR"

"$PYTHON_BIN" tools/test.py "$CONFIG" "$CHECKPOINT" \
    --eval mIoU \
    --show-dir "$SHOW_DIR" \
    --work-dir "$SMOKE_DIR" \
    --cfg-options \
    data.test.split="$SMOKE_SPLIT" \
    data.workers_per_gpu=1

echo "Smoke test checkpoint: $CHECKPOINT"
echo "Smoke test visualization: $SHOW_DIR"
