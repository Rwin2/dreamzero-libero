# DreamZero on LIBERO Benchmark

Adapting [DreamZero](https://arxiv.org/abs/2602.15922) (14B World Action Model) on the [LIBERO](https://arxiv.org/abs/2306.03310) tabletop manipulation benchmark. Zero-shot baselines, action space analysis, LoRA fine-tuning, and closed-loop evaluation.

## Repository Structure

```
dreamzero_work/
  groot/vla/configs/data/dreamzero/
    libero_relative.yaml                # LIBERO data config
    base_48_wan_fine_aug_relative.yaml   # Base config (added libero_sim entries)
  groot/vla/configs/model/dreamzero/
    transform/base.yaml                 # Transform config (added libero_sim)
  groot/vla/model/dreamzero/
    transform/dreamzero_cotrain.py      # Transform class (added LIBERO transform)
  scripts/
    data/convert_libero_to_lerobot.py   # LIBERO HDF5 → LeRobot v2 conversion
    data/convert_lerobot_to_gear.py     # LeRobot → GEAR metadata
    train/libero_training.sh            # LoRA fine-tuning launcher
  inference_server.py                   # WebSocket inference server (2-GPU distributed)
  run_training.sh                       # Training wrapper
  run_pipeline.sh                       # Automated train → eval pipeline
  validate_transforms.py               # Transform validation (no model load)

libero_work/
  eval_dreamzero.py                     # Eval client: LIBERO sim ↔ WebSocket ↔ DreamZero
  offline_comparison.py                 # Offline action comparison (pred vs expert)
  action_diagnostic.py                  # Single-task action diagnostic
  multitask_diagnostic.py               # Cross-task action diagnostic
  run_eval.sh                           # Eval launcher with EGL setup
```

## Action Space

**DreamZero-DROID and LIBERO use different action representations. This mismatch is the core challenge.**

| Property | DreamZero-DROID | LIBERO |
|---|---|---|
| Action type | Delta joint positions | Delta end-effector pose |
| Dimensions | 7 joints + 1 gripper = 8 | 6 DoF (xyz + euler) + 1 gripper = 7 |
| Coordinate space | Joint space (radians) | Cartesian space (meters, radians) |
| Controller | Direct joint position | OSC_POSE (operational space control) |

Both robots are 7-DoF Franka Pandas, but the action semantics are fundamentally different. LIBERO only provides delta EE actions collected with OSC_POSE — there is no joint position action data in the dataset. Computing joint position actions from the EE actions would require inverse kinematics via the Jacobian J(q) and the recorded joint states. The Franka Panda URDF is available and standard IK libraries (pinocchio, roboticstoolbox) could handle this — similar to classic Puma arm derivations — but it was not done here.

### Fine-Tuning Approach

During fine-tuning, LIBERO's 6-dim delta EE actions are stored under the key `action.joint_position` (reusing the DROID key name for compatibility with the action head, but the actual values are Cartesian). After training, the model outputs delta EE actions and we evaluate with OSC_POSE. The pipeline is self-consistent, but the model is learning a different action representation than what the DROID pretrained weights encode. Future work should switch to joint space via inverse kinematics to better leverage the pretrained representations.

### Zero-Shot Controller Experiments

Before fine-tuning, I tested the pretrained DROID model directly on LIBERO to understand the action space gap. Since the pretrained model natively outputs delta joint positions (not delta EE), I tried three different simulator controller configurations:

1. **OSC_POSE (LIBERO's default controller)**: The model outputs delta joint positions, but OSC_POSE expects delta Cartesian poses. The action semantics are completely mismatched — the robot moves erratically. 0% success.
2. **JOINT_POSITION controller, raw output**: This controller matches the model's native action space. However, robosuite's JOINT_POSITION controller internally scales inputs by 0.05, so the model's ~0.5 rad deltas become ~0.025 rad at the joints. The robot barely moves. 0% success.
3. **JOINT_POSITION controller, rescaled (÷ 0.05)**: Dividing by the internal scaling factor restores the correct magnitude. The robot moves at a reasonable speed, but the actions are not task-directed — the model was trained on DROID kitchen scenes, not LIBERO tabletop tasks. 0% success.

These experiments confirmed that even with the correct controller and action scaling, zero-shot transfer fails due to the visual domain gap between DROID and LIBERO. Fine-tuning is necessary. After fine-tuning (where the model learns delta EE actions from LIBERO's OSC_POSE data), all evaluation uses OSC_POSE.

## Training

### Loss Function

DreamZero jointly trains a video prediction model and an action prediction model through flow matching:

- **Dynamics loss**: MSE between predicted and target noise in the video latent space — measures how well the model predicts future visual observations.
- **Action loss**: MSE between predicted and target noise in the action space — measures how well the model maps observations to actions.

Both use flow matching (continuous-time diffusion) with a timestep-dependent schedule. Total loss = dynamics_loss + action_loss.

### Configuration

| Parameter | Value |
|---|---|
| Base checkpoint | DreamZero-DROID (14B params) |
| Fine-tuning | LoRA (rank 128, attention layers) |
| Batch size | 1 per GPU × 2 GPUs |
| Learning rate | 1e-5 (cosine decay) |
| Action horizon | 24 steps at 20 Hz (1.2s chunks) |
| Video context | 33 frames |
| Hardware | 2× H100 80GB |
| Training time | ~9 sec/step, ~30 min for 200 steps |
| Server load time | ~10 min (14B model across 2 GPUs) |
| Eval time | ~80 sec/task/trial |

For reference, DreamZero-DROID was trained with 80K steps on **32 H100s over ~3 days**. This LoRA fine-tuning on 500 LIBERO demos is **0.25%** of that compute.

### Training Results (200 steps)

| Step | Total Loss | Dynamics Loss | Action Loss |
|------|-----------|---------------|-------------|
| 0 | — | 0.063 | 0.191 |
| 50 | 0.143 | 0.048 | 0.114 |
| 100 | 0.113 | 0.042 | 0.064 |
| 150 | 0.103 | 0.041 | 0.071 |
| 200 | 0.096 | 0.037 | 0.068 |

Action loss decreased 64% (0.191 → 0.068). Dynamics loss is roughly stable, suggesting the visual domain shift is partially absorbed without catastrophic forgetting.

## Evaluation Results

### Closed-Loop: LIBERO-Spatial (10 tasks, 1 trial each)

| Metric | Zero-Shot | Fine-Tuned (200 steps) |
|---|---|---|
| Success rate | 0% (0/10) | 0% (0/10) |
| Eval time | ~13 min | ~13 min |

### Offline Action Comparison

To measure improvement beyond the binary success signal, I compared model-predicted actions against expert actions on the training data (first 3 episodes per task, first 8 steps each, 240 total comparisons).

**Baselines comparison:**

| Method | Mean L2 | Mean Cosine Sim | Median Cosine Sim |
|---|---|---|---|
| Random uniform [-1,1] | 1.56 | +0.00 | +0.02 |
| **Fine-tuned (200 steps)** | **1.16** | **-0.28** | **-0.56** |
| Dataset mean (best constant policy) | 0.49 | +0.65 | +0.87 |

**Training loss (flow-matching noise space) before/after:**

| Stage | Action Loss | Dynamics Loss |
|---|---|---|
| Pretrained (step 0) | 0.191 | 0.063 |
| Fine-tuned (step 200) | 0.068 (3x better) | 0.037 |

The training loss improved 3x in noise space, but the offline action comparison shows the fine-tuned model still predicts actions in the opposite direction from the expert (negative cosine similarity). The model learned a constant prediction closer to the data distribution center, but the DROID pretrained prior — trained on delta joint positions — still dominates the output, even though the fine-tuning targets are delta EE actions in a completely different coordinate space.

**Per-dimension breakdown (fine-tuned model vs expert):**

| Dim | Pred Mean | Expert Mean | Correlation | MAE | Order of magnitude |
|---|---|---|---|---|---|
| dx (m) | -0.33 | +0.42 | -0.23 | 0.85 | ~10^-1 |
| dy (m) | -0.15 | +0.15 | -0.03 | 0.45 | ~10^-1 |
| dz (m) | -0.25 | +0.11 | +0.02 | 0.44 | ~10^-1 |
| droll (rad) | +0.09 | -0.01 | +0.03 | 0.10 | ~10^-2 |
| dpitch (rad) | -0.06 | +0.01 | -0.06 | 0.09 | ~10^-2 |
| dyaw (rad) | +0.00 | +0.00 | -0.07 | 0.05 | ~10^-2 |

Position errors (~0.5-0.9 MAE) are an order of magnitude larger than rotation errors (~0.05-0.10 MAE). Cross-task prediction similarity is 0.996 — the model outputs essentially the same trajectory regardless of the input scene.

## Discussion

### Why 0% Success

The model does not condition on the visual input after 200 steps — it outputs the same action trajectory for every scene. The training loss decreased because the model learned to predict a constant vector closer to the dataset action distribution, but this is not sufficient for closed-loop control where actions must be reactive to the current state.

### Action Space Limitation

LIBERO only stores delta end-effector actions (OSC_POSE). There is no joint position data. The fine-tuned model therefore learned a different action representation than DROID's native joint space. With inverse kinematics — computing J(q)^{-1} * dx using the Franka Panda's kinematic model and the recorded joint configurations — one could recover the corresponding joint position actions and train in the model's native space. This would better leverage the pretrained weights. The Franka URDF is publicly available and standard robotics libraries (pinocchio, roboticstoolbox-python) provide the IK solver. This is analogous to classic Puma arm IK derivations.

### Scaling Up

The original DreamZero uses 80K steps on a much larger dataset. A realistic fine-tuning run would be:

| Steps | ETA (2×H100) |
|---|---|
| 200 | 30 min |
| 2K | 5 hours |
| 10K | 25 hours |
| 50K | 5 days |

### DAgger

Distribution shift during closed-loop rollouts could be addressed by running the fine-tuned model in LIBERO, collecting failure trajectories, and re-training on the aggregated dataset.

## Quick Start

### Prerequisites

- 2× H100 GPUs (80GB each)
- DreamZero-DROID checkpoint
- LIBERO benchmark datasets
- Conda environments: `dreamzero` (training/inference), `libero` (evaluation)

### Data Conversion

```bash
conda activate dreamzero
python scripts/data/convert_libero_to_lerobot.py \
    --src-path /path/to/libero_spatial \
    --tgt-path ./data/libero_spatial_lerobot --fps 20
python scripts/data/convert_lerobot_to_gear.py \
    --src-path ./data/libero_spatial_lerobot
```

### Training

```bash
conda activate dreamzero
cd dreamzero_work
PRETRAINED_PATH=/path/to/DreamZero-DROID \
MAX_STEPS=200 \
OUTPUT_DIR=./output/dreamzero_libero_lora_droid \
bash run_training.sh
```

### Inference Server

```bash
conda activate dreamzero
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    inference_server.py --port 5000 --enable-dit-cache \
    --embodiment-tag libero_sim \
    --model-path ./output/dreamzero_libero_lora_droid/checkpoint-200
```

### Evaluation

```bash
conda activate libero
cd libero_work
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python eval_dreamzero.py --port 5000 --suite libero_spatial \
    --num-trials 3 --max-steps 300 --save-videos
```

### Offline Action Comparison

```bash
conda activate dreamzero
cd libero_work
python offline_comparison.py  # requires server running
```
