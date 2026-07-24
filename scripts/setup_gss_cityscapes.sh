#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/igarashi_25/GSS"
ENV_NAME="${GSS_CONDA_ENV:-gss-py38}"
CONDA_BIN="${CONDA_BIN:-/home/igarashi_25/anaconda3/bin/conda}"
MSEG_ROOT="${MSEG_ROOT:-/home/igarashi_25/mseg-api}"
MSEG_COMMIT="2bd00ae8c224d94cbc7ff9b47fde6ae8420adc27"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$CONDA_BIN" env create \
        --name "$ENV_NAME" \
        --file "$REPO_ROOT/environment_gss_cityscapes.yml"
fi

ENV_PREFIX="$("$CONDA_BIN" env list | awk -v name="$ENV_NAME" \
    '$1 == name {print $NF; exit}')"
if [[ -z "$ENV_PREFIX" || ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Could not resolve conda environment: $ENV_NAME" >&2
    exit 1
fi
PYTHON_BIN="$ENV_PREFIX/bin/python"

"$PYTHON_BIN" -m pip install --upgrade \
    "setuptools==59.5.0" \
    "torch==1.12.1+cu116" \
    "torchvision==0.13.1+cu116" \
    "torchaudio==0.12.1+cu116" \
    --extra-index-url https://download.pytorch.org/whl/cu116

"$PYTHON_BIN" -m pip install \
    "mmcv-full==1.6.2" \
    -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html \
    "numpy==1.23.5" \
    "opencv-python==4.7.0.72" \
    "DALL-E==0.1" \
    "einops==0.6.1" \
    "attrs==23.1.0" \
    "scipy==1.10.1" \
    "mmcls==0.25.0" \
    "cityscapesscripts==2.2.2" \
    "matplotlib==3.7.5" \
    "tensorboard==2.14.0" \
    "pandas==2.0.3" \
    "imageio==2.35.1" \
    "yapf==0.32.0" \
    "future==1.0.0" \
    "prettytable==3.11.0" \
    "packaging==26.2"

if [[ ! -d "$MSEG_ROOT/.git" ]]; then
    git clone https://github.com/mseg-dataset/mseg-api.git "$MSEG_ROOT"
fi
git -C "$MSEG_ROOT" checkout "$MSEG_COMMIT"
"$PYTHON_BIN" -m pip install --no-deps -e "$MSEG_ROOT"
"$PYTHON_BIN" -m pip install --no-deps -e "$REPO_ROOT"

cd "$REPO_ROOT"
if [[ ! -s ckp/encoder.pkl || ! -s ckp/decoder.pkl ]]; then
    bash tools/download_pretrain_vqvae.sh
fi

"$PYTHON_BIN" -c \
    "import torch, torchvision, mmcv, mmseg; \
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); \
print('torchvision:', torchvision.__version__); \
print('mmcv:', mmcv.__version__); \
print('mmseg:', mmseg.__version__); \
print('CUDA available:', torch.cuda.is_available())"
