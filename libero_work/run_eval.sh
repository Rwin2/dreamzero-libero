#!/bin/bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate libero
cd /workspace/LIBERO_work

# Headless EGL rendering for MuJoCo
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0

python eval_dreamzero.py \
    --host localhost \
    --port 5000 \
    --suite libero_spatial \
    --num-trials 3 \
    --max-steps 300 \
    --open-loop-horizon 8 \
    --save-videos \
    --output-dir /workspace/dreamzero_work/output/eval_zeroshot \
    --log-level INFO
