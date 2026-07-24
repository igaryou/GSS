#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/igarashi_25/GSS"
CONFIG="$REPO_ROOT/configs/gss/cityscapes/gss-ft-w_swin-l_256x512_b4_400ep_cityscapes.py"
PYTHON_BIN="${GSS_PYTHON:-/home/igarashi_25/anaconda3/envs/gss-py38/bin/python}"
BATCH_SIZE="${GSS_SMOKE_BATCH_SIZE:-4}"
STAGE1_CHECKPOINT="${1:-$REPO_ROOT/work_dirs/smoke_gss-ff_swin-l_256x512_b4/iter_2.pth}"
SMOKE_DIR="$REPO_ROOT/work_dirs/smoke_gss-ft-w_swin-l_256x512_b${BATCH_SIZE}"
SHOW_DIR="$SMOKE_DIR/visualization"
SMOKE_SPLIT="$(mktemp /tmp/gss-ftw-swin-cityscapes-val-one.XXXXXX.txt)"
trap 'rm -f "$SMOKE_SPLIT"' EXIT

if [[ ! -f "$STAGE1_CHECKPOINT" ]]; then
    echo "Stage 1 smoke checkpoint not found: $STAGE1_CHECKPOINT" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GSS_SMOKE_CONFIG="$CONFIG"
export GSS_SMOKE_DIR="$SMOKE_DIR"
export GSS_SMOKE_SPLIT="$SMOKE_SPLIT"
export GSS_SMOKE_BATCH_SIZE="$BATCH_SIZE"
export GSS_STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"

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
    assert cfg.runner.max_iters == 297200
    assert cfg.checkpoint_config.interval == 37150
    assert cfg.evaluation.interval == 37150
'

# --load-from initializes model weights only. A fresh optimizer, scheduler, and
# iteration counter are created by the Stage 2 runner.
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
    "--load-from",
    os.environ["GSS_STAGE1_CHECKPOINT"],
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

export GSS_STAGE2_SMOKE_CHECKPOINT="$CHECKPOINT"
"$PYTHON_BIN" -c '
import glob
import json
import math
import os
import torch
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import build_optimizer, load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

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
            })
assert len(records) >= 2, records
for record in records[-2:]:
    for key, value in record.items():
        if key != "iter" and value is not None:
            assert math.isfinite(value), (key, value)
print("last two finite training loss records:", records[-2:])

cfg = Config.fromfile(os.environ["GSS_SMOKE_CONFIG"])
model = build_segmentor(
    cfg.model,
    train_cfg=cfg.get("train_cfg"),
    test_cfg=cfg.get("test_cfg"))
model = revert_sync_batchnorm(model)
stage1 = torch.load(os.environ["GSS_STAGE1_CHECKPOINT"], map_location="cpu")
model_keys = set(model.state_dict())
stage1_keys = set(stage1["state_dict"])
missing = sorted(model_keys - stage1_keys)
unexpected = sorted(stage1_keys - model_keys)
print("Stage 1 -> Stage 2 missing keys:", missing)
print("Stage 1 -> Stage 2 unexpected keys:", unexpected)

load_checkpoint(
    model,
    os.environ["GSS_STAGE2_SMOKE_CHECKPOINT"],
    map_location="cpu",
    strict=False)
model.cuda()
model.train()

backbone_trainable = [
    name for name, parameter in model.named_parameters()
    if name.startswith("backbone.") and parameter.requires_grad
]
print("backbone parameters with requires_grad=True:", backbone_trainable)
assert backbone_trainable == []

# Use one real sample for a mechanical backward/gradient check. The two-iter
# runner above is the batch-size smoke test; this extra check identifies which
# official FT-W parameters actually receive gradients.
dataset = build_dataset(cfg.data.train)
loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,
    num_gpus=1,
    dist=False,
    shuffle=True,
    drop_last=True,
    persistent_workers=False)
batch = next(iter(loader))
img = batch["img"].data[0].cuda()
img_metas = batch["img_metas"].data[0]
gt = batch["gt_semantic_seg"].data[0].cuda()
optimizer = build_optimizer(model, cfg.optimizer)
optimizer.zero_grad()
loss_dict = model.forward_train(img, img_metas, gt)
loss = sum(value for key, value in loss_dict.items() if "loss" in key)
assert torch.isfinite(loss).item(), loss
loss.backward()

backbone_with_grad = [
    name for name, parameter in model.named_parameters()
    if name.startswith("backbone.") and parameter.grad is not None
]
ftw_with_grad = [
    name for name, parameter in model.named_parameters()
    if name.startswith("decode_head.") and parameter.grad is not None
]
print("backbone parameters with gradients:", backbone_with_grad)
print("FT-W decode-head parameters with gradients:", ftw_with_grad)
assert backbone_with_grad == []
assert any("projection." in name for name in ftw_with_grad)
assert any("post_transformer_block." in name for name in ftw_with_grad)
assert any("cls_segmap." in name for name in ftw_with_grad)
optimizer.step()
print("FT-W diagnostic loss:", float(loss.detach().cpu()))
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
