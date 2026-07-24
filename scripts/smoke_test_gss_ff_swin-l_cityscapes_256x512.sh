#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/igarashi_25/GSS"
CONFIG="$REPO_ROOT/configs/gss/cityscapes/gss-ff_swin-l_256x512_b4_800ep_cityscapes.py"
PYTHON_BIN="${GSS_PYTHON:-/home/igarashi_25/anaconda3/envs/gss-py38/bin/python}"
BATCH_SIZE="${GSS_SMOKE_BATCH_SIZE:-4}"
SMOKE_DIR="$REPO_ROOT/work_dirs/smoke_gss-ff_swin-l_256x512_b${BATCH_SIZE}"
SHOW_DIR="$SMOKE_DIR/visualization"
SMOKE_SPLIT="$(mktemp /tmp/gss-swin-cityscapes-val-one.XXXXXX.txt)"
trap 'rm -f "$SMOKE_SPLIT"' EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GSS_SMOKE_CONFIG="$CONFIG"
export GSS_SMOKE_DIR="$SMOKE_DIR"
export GSS_SMOKE_SPLIT="$SMOKE_SPLIT"
export GSS_SMOKE_BATCH_SIZE="$BATCH_SIZE"

cd "$REPO_ROOT"
find /home/igarashi_25/datasets/cityscapes/leftImg8bit/val \
    -type f -name '*_leftImg8bit.png' | sort | sed -n '1p' | \
    sed -e 's#^.*/val/##' -e 's#_leftImg8bit\.png$##' > "$SMOKE_SPLIT"

"$PYTHON_BIN" -c '
import os
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset

cfg = Config.fromfile(os.environ["GSS_SMOKE_CONFIG"])
batch_size = int(os.environ["GSS_SMOKE_BATCH_SIZE"])
dataset = build_dataset(cfg.data.train)
loader = build_dataloader(
    dataset,
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
    num_gpus=1,
    dist=True,
    shuffle=True,
    drop_last=True,
    persistent_workers=False)
batch = next(iter(loader))
img_shape = tuple(batch["img"].data[0].shape)
gt_shape = tuple(batch["gt_semantic_seg"].data[0].shape)
print("dataset length:", len(dataset))
print("train dataloader length:", len(loader))
print("first train image shape:", img_shape)
print("first train GT shape:", gt_shape)
assert len(dataset) == 2975
assert len(loader) == 2975 // batch_size
assert img_shape == (batch_size, 3, 256, 512), img_shape
assert gt_shape == (batch_size, 1, 256, 512), gt_shape
if batch_size == 4:
    assert len(loader) == 743
    assert cfg.runner.max_iters == 594400
    assert cfg.checkpoint_config.interval == 37150
    assert cfg.evaluation.interval == 37150
'

# Run the existing MMSegmentation train entry point and runner for two
# iterations. The wrapper only adds process-local CUDA peak-memory reporting.
"$PYTHON_BIN" -c '
import atexit
import os
import runpy
import sys
import torch

def report_cuda_peak():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        mib = 1024 ** 2
        print(
            "peak allocated GPU memory (MiB):",
            f"{torch.cuda.max_memory_allocated() / mib:.2f}")
        print(
            "peak reserved GPU memory (MiB):",
            f"{torch.cuda.max_memory_reserved() / mib:.2f}")

atexit.register(report_cuda_peak)
sys.argv = [
    "tools/train.py",
    os.environ["GSS_SMOKE_CONFIG"],
    "--work-dir",
    os.environ["GSS_SMOKE_DIR"],
    "--cfg-options",
    "runner.max_iters=2",
    "checkpoint_config.interval=2",
    "checkpoint_config.max_keep_ckpts=1",
    "evaluation.interval=2",
    "log_config.interval=1",
    "data.samples_per_gpu=" + os.environ["GSS_SMOKE_BATCH_SIZE"],
    "data.workers_per_gpu=2",
    "data.val.split=" + os.environ["GSS_SMOKE_SPLIT"],
]
runpy.run_path("tools/train.py", run_name="__main__")
'

CHECKPOINT="$SMOKE_DIR/iter_2.pth"
test -f "$CHECKPOINT"
mkdir -p "$SHOW_DIR"

export GSS_SMOKE_CHECKPOINT="$CHECKPOINT"
"$PYTHON_BIN" -c '
import glob
import json
import math
import os

logs = sorted(
    glob.glob(os.path.join(os.environ["GSS_SMOKE_DIR"], "*.log.json")),
    key=os.path.getmtime)
assert logs, "No JSON training log was written"
records = []
with open(logs[-1], encoding="utf-8") as stream:
    for line in stream:
        record = json.loads(line)
        if record.get("mode") == "train" and "loss" in record:
            records.append({
                "iter": record.get("iter"),
                "loss": record["loss"],
                "loss_ce": record.get("decode.loss_ce"),
                "loss_ce_pixel": record.get("decode.loss_ce_pixel"),
            })
assert len(records) >= 2, records
for record in records[-2:]:
    for key, value in record.items():
        if key != "iter" and value is not None:
            assert math.isfinite(value), (key, value)
print("last two finite training loss records:", records[-2:])
'

"$PYTHON_BIN" tools/test.py "$CONFIG" "$CHECKPOINT" \
    --eval mIoU \
    --show-dir "$SHOW_DIR" \
    --work-dir "$SMOKE_DIR" \
    --cfg-options \
    data.test.split="$SMOKE_SPLIT" \
    data.workers_per_gpu=1

echo "Smoke test checkpoint: $CHECKPOINT"
echo "Smoke test visualization: $SHOW_DIR"
