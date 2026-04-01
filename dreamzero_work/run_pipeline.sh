#!/bin/bash
# Pipeline: train → reload server → eval
set -e

LOG="/workspace/dreamzero_work/output/overnight.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== Started: $(date) ==="

# 1. Kill inference server (needs GPUs for training)
echo "=== Killing inference server ==="
pkill -f inference_server || true
sleep 5

# 2. Fine-tune (~30 min)
echo "=== Starting fine-tuning: $(date) ==="
source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate dreamzero
cd /workspace/dreamzero_work
PRETRAINED_PATH=/workspace/dreamzero/checkpoints/DreamZero-DROID \
MAX_STEPS=200 \
OUTPUT_DIR=/workspace/dreamzero_work/output/dreamzero_libero_lora_droid \
bash run_training.sh
echo "=== Fine-tuning done: $(date) ==="

# 3. Relaunch inference server with fine-tuned model (~10 min to load)
echo "=== Launching fine-tuned server: $(date) ==="
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    inference_server.py \
    --port 5000 --enable-dit-cache \
    --embodiment-tag libero_sim \
    --model-path /workspace/dreamzero_work/output/dreamzero_libero_lora_droid/checkpoint-200 \
    > /workspace/dreamzero_work/output/inference_server_finetuned.log 2>&1 &

# Wait for server to be ready
echo "Waiting for server to load..."
for i in $(seq 1 120); do
    if grep -q "WebSocket server listening" /workspace/dreamzero_work/output/inference_server_finetuned.log 2>/dev/null; then
        echo "Server ready after ${i}0 seconds"
        break
    fi
    sleep 10
done

# 4. Run eval with OSC_POSE (delta EE — matches fine-tuned output)
echo "=== Starting eval: $(date) ==="
conda activate libero
cd /workspace/LIBERO_work
export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl && export EGL_DEVICE_ID=0
python eval_dreamzero.py \
    --host localhost --port 5000 --suite libero_spatial \
    --num-trials 3 --max-steps 300 --open-loop-horizon 8 \
    --save-videos --output-dir /workspace/dreamzero_work/output/eval_finetuned

echo "=== All done: $(date) ==="
echo "Results at: /workspace/dreamzero_work/output/eval_finetuned/"
cat /workspace/dreamzero_work/output/eval_finetuned/eval_progress.jsonl
