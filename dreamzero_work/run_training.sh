#!/bin/bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /workspace/dreamzero_work

export MAX_STEPS=10000
export REPORT_TO=tensorboard
export OUTPUT_DIR=/workspace/dreamzero_work/output/dreamzero_libero_lora

bash scripts/train/libero_training.sh
