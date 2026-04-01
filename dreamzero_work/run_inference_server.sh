#!/bin/bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /workspace/dreamzero_work

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    inference_server.py \
    --port 5000 \
    --enable-dit-cache \
    --embodiment-tag libero_sim \
    --model-path /workspace/dreamzero_work/output/libero_smoke_test/checkpoint-10
