#!/bin/bash

export WANDB_API_KEY=338a1a94f799ce9a1470532e392f97fe330af917
export LD_LIBRARY_PATH=/root/.tensornvme/lib:$LD_LIBRARY_PATH

# bs=1 时，多卡并行没有用
GPUS_PER_NODE=1
MASTER_ADDR=xx.xx.xx.xx:xxxx
NNODES=1
RANK=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

CONFIG_PATH="config"
CONFIG_NAME="val_config"

# 单卡运行
python scripts/inference_libero.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    "$@"

