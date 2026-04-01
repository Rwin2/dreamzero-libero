# DreamZero-on-LIBERO Benchmark Setup: Findings & Reflexion

**Date**: 2026-03-31
**Machine**: RunPod, 2x H100 SXM 80GB, CUDA Driver 580.126.09, CUDA Version 13.0
**Goal**: Get a fully working DreamZero-on-LIBERO benchmark pipeline

---

## Phase 1: Repo Audit — Key Findings

### DreamZero Architecture
- **Model**: 14B param World Action Model built on Wan2.1-I2V-14B-480P video backbone + umt5-xxl tokenizer
- **Python**: 3.11, torch==2.8.0, CUDA 12.9+
- **Data format**: LeRobot v2 (parquet + MP4 videos) with GEAR metadata overlay
- **Inference**: Distributed WebSocket server across 2+ GPUs (~3s/step on H100)
- **Training**: DeepSpeed ZeRO-2, LoRA fine-tuning from DreamZero-DROID checkpoint
- **Existing embodiments**: DROID (oxe_droid), AgiBot, YAM
- **Repo**: https://github.com/dreamzero0/dreamzero

### LIBERO Architecture
- **Benchmark**: 130 tasks across 5 suites (Spatial/Object/Goal/90/10)
- **Env**: robosuite-based MuJoCo simulation, OSC_POSE controller (Panda robot)
- **Repo**: https://github.com/Lifelong-Robot-Learning/LIBERO
- **Action space**: 7-DOF delta actions (3 pos + 3 rot + 1 gripper), range [-1, 1]
  - Controller: OSC_POSE with output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
- **Observation space** (verified from HDF5):
  - `agentview_rgb`: (128, 128, 3) uint8 — third-person camera
  - `eye_in_hand_rgb`: (128, 128, 3) uint8 — wrist camera
  - `joint_states`: (7,) float64 — arm joint positions
  - `gripper_states`: (2,) float64 — gripper finger positions
  - `ee_pos`: (3,) float64 — end-effector position
  - `ee_ori`: (3,) float64 — end-effector orientation
  - `ee_states`: (6,) float64 — combined ee state
- **Demo format**: HDF5 with 50 demos per task, 75-168 steps per demo (mean ~101)
- **Eval protocol**: 50 fixed init states per task, max 1000 steps, sparse reward (+1 on success)
- **Control frequency**: 20 Hz

### Critical Finding: No Direct LIBERO Support in DreamZero
DreamZero does **NOT** have a ready-made LIBERO evaluation pipeline. However:
- `LIBERO_SIM` is already registered in `embodiment_tags.py` (value: `"libero_sim"`) — the tag exists
- No `modality_config_libero_sim`, no `transform_libero_sim`, no `libero_relative.yaml` exists
- No LIBERO-specific eval client exists
- The DROID eval client (`socket_test_optimized_AR.py`) uses a specific camera key mapping and roboarena interface

### The Benchmark Path (Adapted, Not Official)

```
LIBERO HDF5 demos (50 demos × 10 tasks × ~100 steps)
    → Convert to LeRobot v2 format (custom script: parquet + MP4)
    → Run convert_lerobot_to_gear.py (generates GEAR metadata)
    → Add modality_config_libero_sim + transform_libero_sim to base YAML
    → Create libero_relative.yaml dataset config
    → Fine-tune DreamZero-AgiBot with LoRA on LIBERO data (2× H100)
    → Launch DreamZero inference server with fine-tuned checkpoint
    → Build custom LIBERO eval client (LIBERO env → WebSocket → DreamZero)
    → Run evaluation loop across all tasks × init states
```

---

## Phase 2: Environment Setup — COMPLETED

### System Info
- CUDA Driver: 580.126.09, CUDA Version: 13.0
- No sudo/root access — all installs to user-writable paths
- Miniconda: `/workspace/miniconda3`
- Rust: installed via rustup for tokenizers compilation
- Original repos root-owned at `/workspace/dreamzero` and `/workspace/LIBERO`
- Writable copies at `/workspace/dreamzero_work` and `/workspace/LIBERO_work`
- Checkpoints symlinked: `/workspace/dreamzero_work/checkpoints` → `/workspace/dreamzero/checkpoints`

### Checkpoints (Pre-existing, All Verified)
| Checkpoint | Path | Size | Status |
|---|---|---|---|
| DreamZero-AgiBot | `/workspace/dreamzero/checkpoints/DreamZero-AgiBot/` | 43 GB | PASS |
| DreamZero-DROID | `/workspace/dreamzero/checkpoints/DreamZero-DROID/` | 61 GB | PASS |
| Wan2.1-I2V-14B-480P | `/workspace/dreamzero/checkpoints/Wan2.1-I2V-14B-480P/` | 77 GB | PASS |
| umt5-xxl | `/workspace/dreamzero/checkpoints/umt5-xxl/` | 49 GB | PASS |

### Conda Environments

#### `dreamzero` env — WORKING
- Python 3.11.15
- torch==2.8.0+cu129
- flash-attn==2.8.3 (compiled from source, MAX_JOBS=8)
- deepspeed==0.18.9
- transformers==4.51.3
- wandb, peft==0.5.0, hydra-core, ray==2.47.1
- Editable install from `/workspace/dreamzero_work`
- **Modified**: pyproject.toml — commented out tensorrt, nvidia-modelopt (GB200-only), relaxed numpy pin

#### `libero` env — WORKING
- Python 3.8.13
- torch==2.4.1+cu121 (replaces unavailable torch==1.11.0+cu113)
- robosuite==1.4.0, robomimic==0.2.0
- egl_probe==1.0.2 (cmake installed via conda)
- huggingface_hub for dataset downloads
- Editable install from `/workspace/LIBERO_work`
- LIBERO config at `~/.libero/config.yaml` (auto-created to avoid interactive prompt)

### Issues Encountered & Resolved
| Issue | Resolution |
|---|---|
| No sudo/root | All installed to /workspace/miniconda3, user-writable copies of repos |
| Conda ToS not accepted | `conda tos accept --override-channels --channel ...` for pkgs/main and pkgs/r |
| Repos owned by root | Copied to `_work` suffixed dirs, symlinked checkpoints |
| tensorrt/nvidia-modelopt in pyproject.toml | Commented out (GB200-only, not needed for H100) |
| numpy==1.26.4 pin conflicts with torch 2.8 | Relaxed to numpy>=1.26.4 |
| egl_probe needs cmake | `conda install cmake` in libero env |
| LIBERO __init__.py interactive prompt | Pre-created `~/.libero/config.yaml` |
| LIBERO dataset download interactive prompt | Used direct `snapshot_download()` API |

---

## Phase 3: Assets & Data — COMPLETED

### LIBERO-Spatial Dataset — DOWNLOADED
- **Location**: `/workspace/LIBERO_work/datasets/libero_spatial/`
- **Size**: 5.9 GB (10 HDF5 files, 50 demos each)
- **Tasks**: All 10 spatial reasoning tasks downloaded and verified

### Verified HDF5 Data Structure
```
demo_file.hdf5
├── data/
│   ├── demo_0/  ... demo_49/
│   │   ├── actions: (T, 7) float64     # 7-DOF delta actions [-1, 1]
│   │   ├── obs/
│   │   │   ├── agentview_rgb: (T, 128, 128, 3) uint8
│   │   │   ├── eye_in_hand_rgb: (T, 128, 128, 3) uint8
│   │   │   ├── joint_states: (T, 7) float64
│   │   │   ├── gripper_states: (T, 2) float64
│   │   │   ├── ee_pos: (T, 3) float64
│   │   │   ├── ee_ori: (T, 3) float64
│   │   │   └── ee_states: (T, 6) float64
│   │   ├── states: (T, 92) float64      # full MuJoCo state
│   │   ├── robot_states/                 # robot-specific states
│   │   ├── dones: (T,)
│   │   └── rewards: (T,)
│   └── attrs: {env_args, bddl_file_name, num_demos=50, ...}
```

---

## Phase 4: Sanity Checks — ALL PASSED

| Check | Status | Details |
|---|---|---|
| `import groot` (dreamzero package) | PASS | groot module imports cleanly |
| `import flash_attn` | PASS | v2.8.3 |
| `import deepspeed` | PASS | v0.18.9 |
| `import libero` | PASS | All suites available |
| All 4 checkpoints exist & sized correctly | PASS | 43G + 61G + 77G + 49G = 230G |
| LIBERO task suite enumeration | PASS | 6 suites, 10 tasks in libero_spatial |
| CUDA available in dreamzero env | PASS | 2× H100 detected, torch 2.8.0+cu129 |
| CUDA available in libero env | PASS | 2× H100 detected, torch 2.4.1+cu121 |
| EmbodimentTag.LIBERO_SIM exists | PASS | Value: "libero_sim" |

---

## Phase 5: Benchmark Path Design

### Official vs Adapted Paths
| Component | Official Support | Adaptation Needed |
|---|---|---|
| DreamZero inference server | YES (DROID/AgiBot) | Minor: camera key mapping for LIBERO |
| DreamZero LoRA fine-tuning | YES (new embodiment guide) | YES: need LIBERO config files |
| LIBERO → LeRobot conversion | NO | YES: custom HDF5 → parquet+MP4 script |
| GEAR metadata generation | YES (convert_lerobot_to_gear.py) | Just run with LIBERO params |
| LIBERO eval client | NO | YES: custom WebSocket client |
| LIBERO env stepping | YES (LIBERO native) | Use from libero env |

### Action Space Analysis (CRITICAL)

**The core challenge**: DreamZero trains on **absolute joint positions** and computes relative actions internally. LIBERO uses **delta end-effector actions** through an OSC_POSE controller.

**Options**:
1. **Store delta actions directly**: Train DreamZero to predict delta actions. Set `relative_action: false` since the actions are already deltas. This is simpler but may not align with DreamZero's typical training setup.
2. **Store absolute joint positions as "actions"**: Reframe the problem so DreamZero predicts next joint positions, then compute deltas in the eval client. More aligned with DreamZero's DROID training.
3. **Store the raw delta actions + joint states**: Let DreamZero learn the relationship. Use `relative_action: true` with appropriate keys.

**Recommended approach**: Option 1 — store the 7-DOF delta actions as-is. DreamZero's architecture is flexible enough to predict any action representation. The key is consistent training and eval. 



### Data Conversion Plan: LIBERO HDF5 → LeRobot v2

**Script needed**: `/workspace/dreamzero_work/scripts/data/convert_libero_to_lerobot.py`

**Mapping**:
```
LIBERO HDF5 key                    → LeRobot parquet column
obs/agentview_rgb                   → observation.images.agentview (video)
obs/eye_in_hand_rgb                 → observation.images.eye_in_hand (video)
obs/joint_states (7,)               → observation.state (packed [0:7])
obs/gripper_states (2,)             → observation.state (packed [7:9])
actions (7,)                        → action (packed [0:7])
language_instruction (from attrs)   → annotation.language.language_instruction
```

**State/Action key mapping for GEAR conversion**:
```json
{
  "state": {
    "joint_states": {"start": 0, "end": 7},
    "gripper_states": {"start": 7, "end": 9}
  },
  "action": {
    "delta_pos": {"start": 0, "end": 3},
    "delta_rot": {"start": 3, "end": 6},
    "gripper": {"start": 6, "end": 7}
  }
}
```

### DreamZero Config Files Needed

1. **Modality config** in `base_48_wan_fine_aug_relative.yaml`:
   - `modality_config_libero_sim`: 2 cameras, joint_states + gripper_states for state, 7-dim action
   - `transform_libero_sim`: video transforms + state/action normalization
   - Register in `modality_configs`, `transforms`, `metadata_versions`, `fps` dicts

2. **Dataset YAML**: `groot/vla/configs/data/dreamzero/libero_relative.yaml`
   - Points to converted dataset
   - Sets `relative_action: true` or `false` based on chosen approach

3. **Training script**: `scripts/train/libero_training.sh`
   - Similar to `yam_training.sh` but with LIBERO-specific params
   - `num_views=2` (agentview + wrist, pad with black for 3rd view or adjust)

### Camera View Count Issue
DreamZero expects **3 camera views** (based on DROID/AgiBot/YAM configs). LIBERO has **2 cameras**.

**Options**:
- Pad with a black frame for the 3rd camera (wasteful but compatible)
- Set `num_views=2` and adjust the video modality config (may require code changes)
- Duplicate one camera view (e.g., duplicate agentview)

**Recommendation**: Start with `num_views=3` and duplicate the agentview image as a "third camera". This is the least invasive approach.

### Training Configuration
```bash
DATA_ROOT=/workspace/dreamzero_work/data/libero_spatial_lerobot
OUTPUT_DIR=/workspace/dreamzero_work/checkpoints/dreamzero_libero_lora
NUM_GPUS=2

torchrun --nproc_per_node 2 --standalone \
    groot/vla/experiment/experiment.py \
    data=dreamzero/libero_relative \
    train_architecture=lora \
    num_frames=33 action_horizon=24 num_views=3 \
    model=dreamzero/vla \
    training_args.learning_rate=1e-5 \
    per_device_train_batch_size=1 \
    max_steps=50000 \
    image_resolution_width=320 image_resolution_height=176 \
    save_lora_only=true bf16=true \
    libero_data_root=$DATA_ROOT \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    dit_version=./checkpoints/Wan2.1-I2V-14B-480P \
    output_dir=$OUTPUT_DIR
```

### Inference Server Launch (post fine-tune)
```bash
conda activate dreamzero
cd /workspace/dreamzero_work
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 5000 --enable-dit-cache \
    --model-path ./checkpoints/dreamzero_libero_lora
```

### Custom LIBERO Eval Client
**Script needed**: `/workspace/LIBERO_work/eval_dreamzero.py`

```python
# Pseudocode
for task_id in range(10):
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file=task.bddl_file, ...)
    init_states = suite.get_task_init_states(task_id)

    for init_state in init_states:
        env.reset()
        env.set_init_state(init_state)
        client.reset({"session_id": f"task{task_id}_init{init_id}"})

        for step in range(1000):
            obs = env.get_observation()
            action_chunk = client.infer({
                "observation/agentview_rgb": obs["agentview_rgb"],
                "observation/eye_in_hand_rgb": obs["eye_in_hand_rgb"],
                "observation/joint_states": obs["joint_states"],
                "observation/gripper_states": obs["gripper_states"],
                "prompt": task.language,
            })

            for action in action_chunk:
                obs, reward, done, info = env.step(action[:7])
                if done or reward > 0:
                    break
```

### Baselines (Literature)
| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-10 | Avg |
|---|---|---|---|---|---|
| OpenVLA-OFT | 96.7% | 95.0% | 84.0% | 94.7% | ~97.1% |
| Dream-VLA | — | — | — | — | 97.2% |
| π0 | — | — | — | — | ~90% |
| DreamZero | **Not reported** | — | — | — | — |

DreamZero has NOT reported LIBERO numbers. This would be a **novel benchmark contribution**.

---

## Phase 6: Concrete Next Steps (Priority Order)

### Step 1: Write LIBERO → LeRobot Conversion Script (est. ~2-3 hrs coding)
- Read all 10 task HDF5 files
- For each demo: write parquet (state + action) + MP4 videos (2 cameras, duplicate for 3rd)
- Generate `info.json` with LeRobot v2 metadata
- Output to `/workspace/dreamzero_work/data/libero_spatial_lerobot/`

### Step 2: Run GEAR Metadata Conversion (~minutes)
```bash
cd /workspace/dreamzero_work
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path ./data/libero_spatial_lerobot \
    --embodiment-tag libero_sim \
    --state-keys '{"joint_states": [0, 7], "gripper_states": [7, 9]}' \
    --action-keys '{"delta_pos": [0, 3], "delta_rot": [3, 6], "gripper": [6, 7]}' \
    --relative-action-keys joint_states \
    --task-key annotation.language.language_instruction
```

### Step 3: Add DreamZero Config Files (~1 hr)
- Add `modality_config_libero_sim` and `transform_libero_sim` to base YAML
- Create `libero_relative.yaml`
- Create `libero_training.sh`

### Step 4: LoRA Fine-Tune (~4-12 hrs on 2× H100)
- Start with small max_steps (100) for smoke test
- Then full training (~50K steps)

### Step 5: Write Eval Client (~2-3 hrs)
- WebSocket client that bridges LIBERO env ↔ DreamZero server
- Handle camera key mapping, action format conversion

### Step 6: Evaluate
- Smoke test: 1 task, 5 init states
- Full eval: 10 tasks, 50 init states each

---

## Final Status Table

| Component | Status | Notes |
|---|---|---|
| DreamZero install | **PASS** | All imports work, flash-attn compiled |
| LIBERO install | **PASS** | All suites enumerable, env stepping works |
| Checkpoints | **PASS** | All 4 present, 230GB total |
| LIBERO datasets | **PASS** | libero_spatial downloaded (5.9GB, 10 tasks) |
| DreamZero server launchability | **READY** | Env works, need to test actual server launch |
| Data conversion (HDF5→LeRobot) | **DONE** | `convert_libero_to_lerobot.py` — direct parquet+MP4 output, no lerobot dep |
| GEAR metadata conversion | **READY** | `libero_sim` added to VALID_EMBODIMENT_TAGS |
| Modality config for LIBERO | **DONE** | `modality_config_libero_sim` + `transform_libero_sim` in base YAML |
| Dataset YAML | **DONE** | `libero_relative.yaml` created |
| Training script | **DONE** | `scripts/train/libero_training.sh` created |
| Training readiness | **READY** | Awaiting GEAR conversion, then smoke test |
| Custom eval client | **DONE** | Local matplotlib + JSONL (no W&B) |
| Smoke test (10 steps) | **PASS** | Loss 0.169, ~9s/step, 208MB LoRA checkpoint |
| Full LoRA training | **PENDING** | 10K steps, ~25 hrs, using DreamZero-DROID base |
| Inference server | **READY** | DreamZero-DROID + oxe_droid tag, fixed output paths |
| Zero-shot eval | **IN PROGRESS** | DROID checkpoint → expect ~0% (action space mismatch) |

### What's Working Right Now
- Both conda environments fully functional
- LIBERO can enumerate tasks, load demos, create environments
- DreamZero can import all modules, including flash-attn
- All checkpoints accessible via symlink
- Data conversion script tested (debug mode: 20 episodes OK)
- Full pipeline config chain: base YAML → libero_relative.yaml → libero_training.sh

### Remaining Blockers (in order)
1. ~~Data conversion script~~ **DONE** (500 episodes, 62,250 frames)
2. ~~DreamZero LIBERO config~~ **DONE**
3. ~~GEAR metadata conversion~~ **DONE** (10 tasks, modality.json + stats.json + embodiment.json)
4. ~~Training smoke test~~ **PASS** (10 steps, loss=0.169, 9s/step, 208MB LoRA checkpoint)
5. **Full LoRA training** — IN PROGRESS: 10K steps @ ~9s/step ≈ 25 hrs with TensorBoard
6. ~~Eval client~~ **DONE** — local matplotlib plots + JSONL logging (no W&B)
7. **Run benchmark evaluation** — after training completes

---

## Reflexion Log

### 2026-03-31 23:10 — Initial Assessment
- The task is more complex than "install and run". DreamZero has zero LIBERO integration.
- The `LIBERO_SIM` tag in embodiment_tags.py suggests the authors planned for it but haven't shipped configs.
- The full pipeline requires: data conversion, config authoring, fine-tuning, and custom eval client code.
- Good news: all checkpoints are already downloaded (230GB saved), CUDA 13.0 driver is excellent.
- Risk: action space mismatch between LIBERO (delta OSC) and DreamZero's typical absolute joint pos training.
- Risk: pyproject.toml pins tensorrt which is GB200-only — need to handle during install.

### 2026-03-31 23:15 — Environment Strategy
- No root access, so can't install to /opt/miniconda3 or use apt-get.
- Using /workspace/miniconda3 instead.
- Repos root-owned — solution: copy to `_work` dirs, symlink checkpoints.
- DreamZero's pyproject.toml has GB200-specific deps — commented out tensorrt, nvidia-modelopt.
- Conda ToS needed accepting before env creation worked.

### 2026-03-31 23:30 — Both Envs Working
- DreamZero env: all sanity checks pass including flash-attn 2.8.3.
- LIBERO env: all sanity checks pass, task enumeration works.
- LIBERO datasets downloaded and inspected — HDF5 format fully understood.
- Identified the key remaining work: data conversion, configs, eval client.

### Key Insight: Camera Count Mismatch
- DreamZero expects 3 cameras (all existing embodiments use 3).
- LIBERO has only 2 (agentview + wrist).
- Simplest fix: duplicate agentview as "third camera" during conversion.
- This is a common trick in VLA fine-tuning — verified that DROID also does camera mapping.

### Key Insight: Action Representation
- LIBERO uses **delta end-effector** actions through OSC_POSE controller.
- DreamZero typically trains on **absolute joint positions**.
- For LIBERO, we should store the delta actions as-is and train DreamZero to predict them.
- At eval time, the predicted actions go directly to `env.step()` — no conversion needed.
- This is cleaner than trying to convert everything to absolute joint positions.

### 2026-03-31 — Session 2: Implementation Phase

#### Key Decision: No LeRobot Dependency for Conversion
- `lerobot` is NOT in DreamZero's pyproject.toml — it's optional, only used by convert_agibot.py
- Wrote `convert_libero_to_lerobot.py` to directly output LeRobot v2 format using pyarrow + av
- This is cleaner: no API version issues, full control over output format
- Debug test (20 episodes) passed in 9 seconds

#### Key Decision: State/Action Key Naming
- State packed as: `joint_states` [0:7] + `gripper_states` [7:9] = 9-dim
- Action: 7-dim delta EE actions stored as-is
- YAML config maps: `state.joint_states`, `state.gripper_states`, `action.joint_states`, `action.gripper_states`
- `relative_action: false` — actions are already deltas, no need for DreamZero to compute differences
- Normalization mode: `q99` for all state/action keys (matches DROID/AgiBot/YAM pattern)

#### Key Finding: DreamZero Paper Training Recipe
- YAM fine-tuning: 55 trajectories (~30 min play data), 100K steps, lr=1e-5, batch=4/GPU
- LoRA on all DiT blocks + state/action encoders/decoders (text/image/VAE frozen)
- Pre-train base: DreamZero-AgiBot checkpoint
- Action horizon: 24, input frames: 33, resolution: 320x176
- LIBERO has ~500 demos × ~100 steps = ~50K frames — comparable to YAM's data scale

#### Key Finding: W&B Integration
- Training already logs to W&B via `report_to=wandb` (HuggingFace Trainer)
- Metrics: loss, dynamics_loss_avg, action_loss_avg, learning_rate per step
- Also writes JSONL loss log to `{output_dir}/loss_log.jsonl`
- No eval/validation during training — need to add benchmark W&B logging in eval client

#### Smoke Test Results (2026-04-01 00:50)
- **10-step LoRA training completed successfully** on 2× H100
- Initial losses: dynamics_loss=0.068, action_loss=0.233 → combined loss=0.169 at step 10
- Timing: ~9s per training step (2.5s forward + 6.5s backward/optimizer)
- Model loading takes ~10 min (14B params loaded to CPU first, then transferred to GPU)
- GPU memory: 72.6GB + 75.0GB (out of 81.5GB each) — fits with batch_size=1
- LoRA checkpoint: 208MB (`save_lora_only=true`)
- DeepSpeed ZeRO-2 full state: 87GB (not needed for inference, deleted to free disk)
- TensorBoard events written successfully to `runs/` directory
- Extrapolation: 100K steps ≈ 10.4 days, 10K steps ≈ 25 hrs

#### Visualization Switch: W&B → TensorBoard + Local
- User lost W&B free trial — all visualization now uses TensorBoard (training) and matplotlib (eval)
- Training: `report_to=tensorboard` in training script, events in `output_dir/runs/`
- Training also writes `loss_log.jsonl` for programmatic access
- Eval: per-task results to `eval_progress.jsonl`, summary bar chart to `success_rates_{suite}.png`
- View TensorBoard: `tensorboard --logdir /workspace/dreamzero_work/output/dreamzero_libero_lora/runs`

### 2026-04-01 — Session 3: Checkpoint Selection & Action Space Analysis

#### Why DreamZero-DROID, Not DreamZero-AgiBot

We switched from DreamZero-AgiBot to DreamZero-DROID as the base checkpoint. Reasons:

1. **Action space compatibility**: DROID was trained on 7-DOF single-arm Panda data (same robot family as LIBERO). AgiBot is a 32-DOF bimanual humanoid — its action space is fundamentally different.

2. **metadata.json mismatch**: AgiBot checkpoint's `metadata.json` only has `agibot` key. The inference server (`GrootSimPolicy`) needs `oxe_droid` metadata for normalization stats. DROID checkpoint has it. Attempting `--embodiment-tag agibot` fails because the socket server's observation mapping is hardcoded for DROID-style keys (`video.exterior_image_1_left`), while agibot expects `video.top_head`, `video.hand_left`, `video.hand_right`.

3. **Camera keys**: DROID uses `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left` — exactly what our eval client and data conversion already output. AgiBot uses `top_head`, `hand_left`, `hand_right`.

4. **PermissionError**: Server tried to write eval output to `/workspace/dreamzero/checkpoints/` (root-owned). Fixed by redirecting output to relative `output/` dir under the writable working directory.

| Feature | DreamZero-DROID | DreamZero-AgiBot |
|---|---|---|
| Robot type | 7-DOF Panda arm | 32-DOF bimanual humanoid |
| Action dim (output) | 8 (7 joint + 1 gripper) | 32 (22 joints + 10 hand) |
| Camera keys | exterior_image_1/2_left, wrist_image_left | top_head, hand_left, hand_right |
| metadata.json has `oxe_droid` | YES | NO |
| Socket server compatible | YES (hardcoded for DROID keys) | NO (would need agibot key mapping) |
| Checkpoint size | 61 GB | 43 GB |

#### Action Space Mismatch (Critical for Zero-Shot)

**The problem**: The DreamZero-DROID model outputs `action.joint_position` (7-dim) + `action.gripper_position` (1-dim). These are **delta joint angles** (mean≈0, std≈0.1-0.2). LIBERO's controller expects **delta end-effector** actions (3 position + 3 rotation + 1 gripper).

These are fundamentally different coordinate spaces:
- **Delta joint position**: change in each of the 7 joint angles (θ₁, θ₂, ..., θ₇)
- **Delta end-effector**: change in Cartesian pose (Δx, Δy, Δz, Δax, Δay, Δaz)
- Relationship: Δx = J(q) × Δq (through the Jacobian, which is configuration-dependent)

**Why we can't remap**: The model's `action_concat_order` for oxe_droid only includes `[joint_position, gripper_position]`. It does NOT output `cartesian_position` — that key exists in the metadata/statistics but isn't in the inference transform's output split. The model was trained to only predict 8 dims for this embodiment.

**Zero-shot implication**: Feeding delta joint angles to LIBERO's delta-EE controller will produce random/meaningless behavior. Expected success rate: ~0%. This is still a valuable baseline — it quantifies the gap that fine-tuning must bridge.

**After fine-tuning**: The LIBERO config maps the 6-dim delta EE actions to `action.joint_position` (reusing the key name). So after fine-tuning on LIBERO data, the model learns to output delta EE under the `action.joint_position` key. The name is misleading but the values will be correct. The eval pipeline then works end-to-end.

### 2026-04-01 — Session 3 (continued): Training Results & Inference Setup

#### Smoke Test Training PASSED (10 steps)
- Initial losses: dynamics_loss=0.068, action_loss=0.233
- Step 10: combined loss=0.169, lr=3.0e-7 (warmup), grad_norm=1.89
- Forward pass: ~2.5s, total step: ~9s
- GPU memory: 72.6GB + 75.0GB out of 81.5GB each
- LoRA checkpoint: 208MB, DeepSpeed full state: 87GB (deleted)
- TensorBoard events written successfully

#### Training Time Analysis
The "30 minutes" mentioned in the README is **DATA COLLECTION** time, not training time:
- YAM: 100K steps, batch=4/GPU, 8 GPUs default → days of training
- AgiBot: 5K steps, batch=1/GPU, 8 GPUs → ~12.5 hours
- Our setup: 2 GPUs, batch=1, ~9s/step
- 10K steps ≈ 25 hours, 5K steps ≈ 12.5 hours

#### Inference Server Setup
- Added `--embodiment-tag` CLI arg to `socket_test_optimized_AR.py` (was hardcoded to `oxe_droid`)
- Server loads model from checkpoint's `experiment_cfg/conf.yaml`
- DreamZero-AgiBot checkpoint has NO `libero_sim` config → can't do true zero-shot
- Smoke test checkpoint (10 steps) used as quasi-zero-shot baseline
- Server command: `python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --embodiment-tag libero_sim --model-path <checkpoint_path>`

#### Visualization: W&B → TensorBoard + Local
- Training: `report_to=tensorboard`, view with `tensorboard --logdir output/runs --port 6006`
- Also writes `loss_log.jsonl` for programmatic access
- Eval: per-task JSONL + matplotlib bar chart (no W&B dependency)

#### Files Created
| File | Purpose |
|---|---|
| `scripts/data/convert_libero_to_lerobot.py` | LIBERO HDF5 → LeRobot v2 (parquet + MP4) |
| `configs/data/dreamzero/libero_relative.yaml` | Dataset config for LIBERO |
| `scripts/train/libero_training.sh` | LoRA fine-tuning launch script |
| `base_48_wan_fine_aug_relative.yaml` (modified) | Added `modality_config_libero_sim` + `transform_libero_sim` |
| `convert_lerobot_to_gear.py` (modified) | Added `libero_sim` to VALID_EMBODIMENT_TAGS |
| `LIBERO_work/eval_dreamzero.py` | Eval client — local matplotlib + JSONL (no W&B) |
| `DREAMZERO_LIBERO_ARCHITECTURE.md` | Full pipeline architecture reference |
| `base.yaml` (modified) | Added `libero_sim: 17` to embodiment_tag_to_projector_index |
| `dreamzero_cotrain.py` (modified) | Added LIBERO_SIM text template in collate() |

---

## Appendix: Working Directory Reference

```
/workspace/
├── miniconda3/           # Conda installation
│   └── envs/
│       ├── dreamzero/    # Python 3.11, torch 2.8.0+cu129
│       └── libero/       # Python 3.8.13, torch 2.4.1+cu121
├── dreamzero/            # Original repo (root-owned, read-only)
│   └── checkpoints/      # 230GB of model weights
├── dreamzero_work/       # Writable copy for development
│   └── checkpoints → /workspace/dreamzero/checkpoints  # symlink
├── LIBERO/               # Original repo (root-owned)
├── LIBERO_work/          # Writable copy for development
│   └── datasets/
│       └── libero_spatial/  # 10 HDF5 files, 5.9GB
├── hf_cache/             # HuggingFace cache
└── DREAMZERO_LIBERO_FINDINGS.md  # This document
```

## Appendix: Activation Commands

```bash
# Always set before any pip/HF command
export TMPDIR=/tmp
export HF_HOME=/workspace/hf_cache

# Activate dreamzero
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate dreamzero
cd /workspace/dreamzero_work

# Activate libero
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate libero
cd /workspace/LIBERO_work
```
