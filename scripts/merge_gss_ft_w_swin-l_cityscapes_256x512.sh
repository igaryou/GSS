#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/igarashi_25/GSS"
CONFIG="$REPO_ROOT/configs/gss/cityscapes/gss-ft-w_swin-l_256x512_b4_400ep_cityscapes.py"
PYTHON_BIN="${GSS_PYTHON:-/home/igarashi_25/anaconda3/envs/gss-py38/bin/python}"
STAGE1_CHECKPOINT="${1:-}"
STAGE2_CHECKPOINT="${2:-}"
TARGET_CHECKPOINT="${3:-}"

if [[ -z "$STAGE1_CHECKPOINT" || -z "$STAGE2_CHECKPOINT" || \
      -z "$TARGET_CHECKPOINT" || ! -f "$STAGE1_CHECKPOINT" || \
      ! -f "$STAGE2_CHECKPOINT" ]]; then
    echo "Usage: bash $0 STAGE1.pth STAGE2.pth MERGED.pth" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
"$PYTHON_BIN" tools/merge_checkpoints.py \
    --model_path "$STAGE1_CHECKPOINT" \
    --post_model_path "$STAGE2_CHECKPOINT" \
    --target_path "$TARGET_CHECKPOINT" \
    --backbone_type swin

export GSS_MERGE_CONFIG="$CONFIG"
export GSS_STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT"
export GSS_STAGE2_CHECKPOINT="$STAGE2_CHECKPOINT"
export GSS_TARGET_CHECKPOINT="$TARGET_CHECKPOINT"
"$PYTHON_BIN" -c '
import os
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor

stage1 = torch.load(os.environ["GSS_STAGE1_CHECKPOINT"], map_location="cpu")
stage2 = torch.load(os.environ["GSS_STAGE2_CHECKPOINT"], map_location="cpu")
merged = torch.load(os.environ["GSS_TARGET_CHECKPOINT"], map_location="cpu")
stage1_keys = set(stage1["state_dict"])
stage2_keys = set(stage2["state_dict"])
merged_keys = set(merged["state_dict"])
stage2_added = stage2_keys - stage1_keys
print("merged checkpoint keys:", len(merged_keys))
print("Stage 1-derived keys:", len(stage1_keys))
print("Stage 2-derived added keys:", len(stage2_added))

cfg = Config.fromfile(os.environ["GSS_MERGE_CONFIG"])
model = build_segmentor(
    cfg.model,
    train_cfg=cfg.get("train_cfg"),
    test_cfg=cfg.get("test_cfg"))
model_keys = set(model.state_dict())
print("merged missing keys:", sorted(model_keys - merged_keys))
print("merged unexpected keys:", sorted(merged_keys - model_keys))
load_checkpoint(
    model,
    os.environ["GSS_TARGET_CHECKPOINT"],
    map_location="cpu",
    strict=False)
'
