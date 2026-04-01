# DreamZero-LIBERO Quick Reference

## Files

| File | Purpose |
|---|---|
| `dreamzero_work/socket_test_optimized_AR.py` | Inference server — loads 14B model, serves actions via WebSocket |
| `dreamzero_work/launch_server_agibot.sh` | Server launch script (now uses DROID, name kept for compat) |
| `dreamzero_work/scripts/train/libero_training.sh` | Training launcher — LoRA fine-tuning with DeepSpeed ZeRO-2 |
| `dreamzero_work/run_training.sh` | Training wrapper (sets env vars, calls libero_training.sh) |
| `dreamzero_work/scripts/data/convert_libero_to_lerobot.py` | LIBERO HDF5 → LeRobot v2 format (parquet + MP4) |
| `LIBERO_work/eval_dreamzero.py` | Eval client — LIBERO env ↔ WebSocket ↔ DreamZero server |
| `LIBERO_work/run_eval.sh` | Eval launch script (sets EGL vars, calls eval_dreamzero.py) |

## Terminal Commands

### Zero-Shot Evaluation

```bash
# Terminal 1: Launch inference server (~10 min to load model)
source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate dreamzero
cd /workspace/dreamzero_work
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 5000 --enable-dit-cache \
    --embodiment-tag oxe_droid \
    --model-path /workspace/dreamzero/checkpoints/DreamZero-DROID

# Terminal 2: Run eval (wait for server to say "WebSocket server listening")
source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate libero
cd /workspace/LIBERO_work
export MUJOCO_GL=egl && export PYOPENGL_PLATFORM=egl && export EGL_DEVICE_ID=0
python eval_dreamzero.py \
    --host localhost --port 5000 --suite libero_spatial \
    --num-trials 3 --max-steps 300 --open-loop-horizon 8 \
    --save-videos --output-dir /workspace/dreamzero_work/output/eval_zeroshot
```

### Fine-Tuning (after zero-shot)

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate dreamzero
cd /workspace/dreamzero_work
PRETRAINED_PATH=/workspace/dreamzero/checkpoints/DreamZero-DROID \
MAX_STEPS=10000 \
bash run_training.sh
```

### Monitoring

```bash
# Server loading:
tail -f /workspace/dreamzero_work/output/inference_server.log

# Eval progress:
tail -f /workspace/dreamzero_work/output/eval_zeroshot/eval_progress.jsonl

# Training loss:
tensorboard --logdir /workspace/dreamzero_work/output/dreamzero_libero_lora/runs --port 6006
```

## Key Modifications for Adaptation

1. **Checkpoint: AgiBot → DROID** — AgiBot's metadata.json lacks `oxe_droid` key; DROID has it. DROID's action space (7-DOF Panda) is also closer to LIBERO's.

2. **Camera key mapping** — LIBERO has 2 cameras (agentview + wrist), DreamZero expects 3. We duplicate agentview as the 2nd exterior camera. Key names use DROID convention: `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`.

3. **Action space reuse** — LIBERO's 6-dim delta EE actions are mapped to `action.joint_position` (6-dim) + `action.gripper_position` (1-dim) during fine-tuning. The key name "joint_position" is misleading but the model learns the correct action semantics from the data.

4. **Output paths** — All writable output goes to `/workspace/dreamzero_work/output/` (user-owned), never to `/workspace/dreamzero/checkpoints/` (root-owned, read-only).

5. **Image resize** — LIBERO images (128x128) are resized to 180x320 to match DreamZero's expected resolution.

## Main Reflexion Axes

**Why zero-shot will fail (~0% expected):**
The DROID model outputs delta joint angles (7-dim), but LIBERO's OSC_POSE controller expects delta end-effector commands (6-dim). These are different coordinate spaces (joint space vs Cartesian space). No remapping is possible because the model doesn't output `cartesian_position` — only `joint_position` is in the action output.

**Why fine-tuning should work:**
After fine-tuning on LIBERO data, the model learns to output delta EE actions under the `action.joint_position` key. The values become semantically correct even though the key name doesn't change. The entire eval pipeline then works end-to-end.

**Baseline value:**
The zero-shot 0% baseline is still valuable for the paper — it quantifies the gap that LIBERO-specific fine-tuning bridges, demonstrating that DreamZero's generalist pretraining alone is insufficient for a new embodiment with different action conventions.

## Output Locations

| Output | Path |
|---|---|
| Zero-shot results | `/workspace/dreamzero_work/output/eval_zeroshot/eval_results.json` |
| Zero-shot plots | `/workspace/dreamzero_work/output/eval_zeroshot/success_rates.png` |
| Zero-shot videos | `/workspace/dreamzero_work/output/eval_zeroshot/videos/` |
| Training checkpoints | `/workspace/dreamzero_work/output/dreamzero_libero_lora/` |
| Training TensorBoard | `/workspace/dreamzero_work/output/dreamzero_libero_lora/runs/` |
| Server log | `/workspace/dreamzero_work/output/inference_server.log` |
