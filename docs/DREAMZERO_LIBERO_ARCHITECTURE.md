# DreamZero-on-LIBERO Benchmark Pipeline Architecture

This document describes the full code structure, configuration hierarchy, data flow,
and runtime behavior of the DreamZero pipeline when applied to the LIBERO simulation
benchmark.

---

## 1. Pipeline Flow Diagram (ASCII)

```
LIBERO HDF5 files (one per task, 50 demos each)
       |
       | convert_libero_to_lerobot.py
       |   --src-path  /path/to/libero_spatial
       |   --tgt-path  /path/to/libero_spatial_lerobot
       |   --fps 20
       v
LeRobot v2 Dataset
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.{cam}/episode_000000.mp4
  meta/info.json, tasks.jsonl, episodes.jsonl
       |
       | convert_lerobot_to_gear.py
       |   --dataset-path /path/to/libero_spatial_lerobot
       |   --embodiment-tag libero_sim
       |   --state-keys '{"joint_states":[0,7],"gripper_states":[7,9]}'
       |   --action-keys '{"joint_states":[0,6],"gripper_states":[6,7]}'
       |   --relative-action-keys joint_states gripper_states
       v
GEAR-augmented Dataset (adds to meta/):
  meta/modality.json              (key mapping for training)
  meta/embodiment.json            ({"embodiment_tag": "libero_sim"})
  meta/stats.json                 (mean/std/min/max/q01/q99)
  meta/relative_stats_dreamzero.json  (relative action stats)
       |
       | Hydra config resolution
       |   conf.yaml
       |     -> data=dreamzero/libero_relative (libero_relative.yaml)
       |          -> defaults: dreamzero/base_48_wan_fine_aug_relative
       |               (base_48_wan_fine_aug_relative.yaml)
       v
Training (torchrun)
  groot/vla/experiment/experiment.py  (@hydra.main)
    -> VLAExperiment(BaseExperiment)
       -> VLATrainer(BaseTrainer) via HuggingFace Transformers
       -> W&B logging (report_to=wandb)
       -> DeepSpeed ZeRO-2
       -> LoRA fine-tuning from DreamZero-AgiBot checkpoint
       |
       v
Checkpoint (LoRA weights)
  checkpoints/dreamzero_libero_lora/checkpoint-{step}/
       |
       v
Inference / Evaluation
  model.get_action(inputs)  ->  7-dim delta EE actions
```

---

## 2. File-by-File Breakdown

### 2.1 `scripts/data/convert_libero_to_lerobot.py`

**Purpose:** Convert raw LIBERO HDF5 demo files into the LeRobot v2 directory layout
(parquet + MP4 videos) that DreamZero's GEAR pipeline expects.

**Key constants:**

| Constant | Value | Purpose |
|---|---|---|
| `CAMERA_KEYS` | 3 entries (see below) | LeRobot video keys; DreamZero requires exactly 3 cameras |
| `LIBERO_CAM_MAP` | Maps LeRobot key to HDF5 obs key | `agentview_2` duplicates `agentview_rgb` |
| `CHUNKS_SIZE` | 1000 | Episodes per chunk directory |

**Key functions:**

| Function | Arguments | Description |
|---|---|---|
| `get_language_instruction(hdf5_path)` | `Path` to HDF5 file | Extracts NL task string from `data.attrs["problem_info"]["language_instruction"]`; falls back to filename-derived text |
| `encode_video(frames, output_path, fps)` | `np.ndarray (T,H,W,3)`, `Path`, `int` | Encodes RGB frames to H.264 MP4 using PyAV with CRF 23, YUV420P |
| `convert_libero_to_lerobot(src_path, tgt_path, fps, debug)` | `str`, `str`, `int=20`, `bool=False` | Main conversion loop; iterates HDF5 files, extracts state/action/video, writes parquet + MP4 + meta files |

**CLI arguments:**

```
--src-path   DIR   Directory containing LIBERO .hdf5 files (required)
--tgt-path   DIR   Output LeRobot v2 dataset directory (required)
--fps        INT   Control frequency, default 20
--debug            Only convert 2 demos per task (for testing)
```

**Output structure:**

```
tgt-path/
  data/chunk-000/episode_000000.parquet   # columns: observation.state, action, episode_index,
                                          #          frame_index, index, timestamp, task_index
  videos/chunk-000/
    observation.images.agentview/episode_000000.mp4
    observation.images.eye_in_hand/episode_000000.mp4
    observation.images.agentview_2/episode_000000.mp4
  meta/
    info.json       # codebase_version, robot_type, features, fps, splits, etc.
    tasks.jsonl     # {"task_index": 0, "task": "pick up the ..."}
    episodes.jsonl  # {"episode_index": 0, "tasks": ["..."], "length": T}
```

**State/action packing:**

- `observation.state` = `[joint_states(7), gripper_states(2)]` = 9 dimensions
- `action` = raw LIBERO actions = 7 dimensions (delta end-effector)
- `info.json` declares `robot_type: "libero_sim"`

---

### 2.2 `scripts/data/convert_lerobot_to_gear.py`

**Purpose:** Generate/augment metadata files required by DreamZero's training pipeline
on top of an existing LeRobot v2 dataset. Does NOT modify parquet or video files.

**Key functions:**

| Function | Signature | Description |
|---|---|---|
| `load_info(dataset_path)` | `Path -> dict` | Reads `meta/info.json` |
| `get_parquet_paths(dataset_path, info)` | `Path, dict -> list[Path]` | Enumerates all parquet files using `data_path` template and `total_episodes` |
| `detect_features(info)` | `dict -> dict` | Categorises features from info.json into state/action/video/annotation lists |
| `build_modality_json(info, detected, state_mapping, action_mapping, task_key)` | various | Constructs the `modality.json` dict with `state`, `action`, `video`, `annotation` sections |

**CLI arguments (relevant subset):**

```
--dataset-path      DIR       Path to LeRobot v2 dataset (required)
--output-path       DIR       Optional output dir (else modifies in-place)
--embodiment-tag    STR       Embodiment tag (must be in VALID_EMBODIMENT_TAGS list)
--state-keys        JSON_STR  e.g. '{"joint_states":[0,7],"gripper_states":[7,9]}'
--action-keys       JSON_STR  e.g. '{"joint_states":[0,6],"gripper_states":[6,7]}'
--relative-action-keys  STR+  Keys that should get relative-action stats
--task-key          STR       Annotation key name (auto-detected if omitted)
--force                       Overwrite existing metadata files
--fps               INT       Override dataset FPS in info.json
```

**Files created in `meta/`:**

| File | Content |
|---|---|
| `modality.json` | Key mapping: state/action sub-key names, index ranges (`start`/`end`), dtypes, `original_key` back-references |
| `embodiment.json` | `{"robot_type": "libero_sim", "embodiment_tag": "libero_sim"}` |
| `stats.json` | Per-column statistics: mean, std, min, max, q01, q99 |
| `relative_stats_dreamzero.json` | Statistics computed on `action[t] - state[t]` for relative action keys |

**modality.json structure for LIBERO:**

```json
{
  "state": {
    "joint_states":   {"original_key": "observation.state", "start": 0, "end": 7, ...},
    "gripper_states": {"original_key": "observation.state", "start": 7, "end": 9, ...}
  },
  "action": {
    "joint_states":   {"original_key": "action", "start": 0, "end": 6, ...},
    "gripper_states": {"original_key": "action", "start": 6, "end": 7, ...}
  },
  "video": {
    "agentview":    {"original_key": "observation.images.agentview"},
    "eye_in_hand":  {"original_key": "observation.images.eye_in_hand"},
    "agentview_2":  {"original_key": "observation.images.agentview_2"}
  },
  "annotation": {
    "language.language_instruction": {"original_key": "annotation.language.language_instruction"}
  }
}
```

---

### 2.3 Config Files

#### 2.3.1 `groot/vla/configs/conf.yaml` (Root Hydra Config)

**Purpose:** Top-level configuration. All other configs compose into or override fields
defined here.

**Key fields:**

| Field | Default | Description |
|---|---|---|
| `model` | `dreamzero/vla` | Model config group |
| `data` | `dreamzero/droid_horizon_relative` | Data config group (overridden to `dreamzero/libero_relative` for LIBERO) |
| `trainer._target_` | `groot.vla.experiment.VLATrainer` | Trainer class |
| `report_to` | `wandb` | Reporting backend |
| `wandb_project` | `???` (must be set) | W&B project name |
| `output_dir` | `???` (must be set) | Checkpoint output directory |
| `training_args._target_` | `transformers.TrainingArguments` | HuggingFace TrainingArguments |
| `save_lora_only` | `false` | Whether to save only LoRA adapter weights |
| `pretrained_model_path` | `null` | Path to pre-trained checkpoint to load |

The `training_args` section mirrors all standard HF Trainer parameters (lr, batch size,
deepspeed, etc.) and references top-level Hydra variables via `${}` interpolation.

#### 2.3.2 `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`

**Purpose:** Base data config shared across all embodiments. Defines horizons,
image resolutions, video/state/action transforms (with YAML anchors), and per-embodiment
`modality_config_*` / `transform_*` blocks.

**Scope:** Uses `# @package _global_` to merge into the root config namespace.

**Key parameters:**

| Parameter | Value | Description |
|---|---|---|
| `num_frames` | 49 | Total video frames per sample |
| `action_horizon` | 48 | Action prediction horizon |
| `state_horizon` | 1 | Number of state frames to condition on |
| `image_resolution_width` | 480 | Multi-frame video width |
| `image_resolution_height` | 256 | Multi-frame video height |
| `image_resolution_width_single_frame` | 256 | Single-frame width |
| `image_resolution_height_single_frame` | 256 | Single-frame height |

**LIBERO-specific section (`modality_config_libero_sim`):**

```yaml
modality_config_libero_sim:
  video:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, ..., 24]     # 25 video frame indices
    eval_delta_indices: [0]                # single frame at eval
    modality_keys:
      - video.agentview
      - video.eye_in_hand
      - video.agentview_2
  state:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - state.joint_states
      - state.gripper_states
  action:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, ..., 23]     # 24-step action chunk
    modality_keys:
      - action.joint_states
      - action.gripper_states
  language:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - annotation.language.language_instruction
```

**Transform pipeline (`transform_libero_sim`):**

The transform is a `ComposedModalityTransform` with these stages applied in order:

1. **VideoToTensor** -- convert video frames to tensors
2. **VideoCrop** -- random crop with scale 0.95
3. **VideoResize** -- resize to `(256, 480)` (H, W)
4. **VideoColorJitter** -- brightness/contrast/saturation/hue augmentation
5. **VideoToNumpy** -- convert back to numpy for the model
6. **StateActionToTensor** -- convert state arrays to tensors
7. **StateActionTransform** -- normalize state using `q99` mode
8. **StateActionToTensor** -- convert action arrays to tensors
9. **StateActionTransform** -- normalize action using `q99` mode
10. **ConcatTransform** -- concatenate multi-key state/action/video into single tensors
11. **`${model_specific_transform}`** -- injected by model config

**Registry maps at file bottom:**

```yaml
modality_configs:
  libero_sim: ${modality_config_libero_sim}

transforms:
  libero_sim: ${transform_libero_sim}

metadata_versions:
  libero_sim: '0221'

fps:
  libero_sim: 20
```

These maps are keyed by embodiment tag. The dataset loader looks up the correct config
by matching `embodiment.json["embodiment_tag"]` against these maps.

#### 2.3.3 `groot/vla/configs/data/dreamzero/libero_relative.yaml`

**Purpose:** LIBERO-specific overrides on top of the base config.

**Hydra defaults chain:**

```yaml
defaults:
  - dreamzero/base_48_wan_fine_aug_relative  # loads the base config first
  - _self_                                    # then this file overrides
```

**Key overrides:**

| Parameter | Value | Rationale |
|---|---|---|
| `max_state_dim` | 64 | Maximum state vector dimensionality |
| `relative_action` | `false` | LIBERO actions are already delta EE -- no subtraction needed |
| `relative_action_per_horizon` | `false` | Same reason |
| `relative_action_keys` | `[joint_states, gripper_states]` | Still listed for stats computation, but `relative_action: false` prevents subtraction |
| `max_chunk_size` | 5 | Maximum action chunk size for chunked sampling |
| `dataset_shard_sampling_rate` | 0.1 | Fraction of shards sampled per epoch |
| `mixture_dataset_cls` | `...ShardedLeRobotMixtureDataset.from_mixture_spec` | Dataset class for multi-dataset mixture |
| `single_dataset_cls` | `...ShardedLeRobotSubLangSingleActionChunkDatasetDROID` | Per-shard dataset class |
| `libero_sim_data_root` | `???` | Must be set via CLI or env var |

**Dataset mixture spec:**

```yaml
train_dataset:
  _target_: ${mixture_dataset_cls}
  mixture_spec:
    - dataset_path:
        libero_sim:
          - ${libero_sim_data_root}
      dataset_weight: 1.0
      distribute_weights: true
  dataset_class: ${single_dataset_cls}
  all_modality_configs: ${modality_configs}     # -> embodiment tag -> ModalityConfig
  all_transforms: ${transforms}                 # -> embodiment tag -> transform pipeline
  metadata_versions: ${metadata_versions}
  fps: ${fps}
  dataset_kwargs:
    video_backend: decord
    use_global_metadata: false
    max_chunk_size: 5
    relative_action: false
    relative_action_keys: [joint_states, gripper_states]
    relative_action_per_horizon: false
  mixture_kwargs:
    training: true
    balance_dataset_weights: false
    seed: 42
    shard_sampling_rate: 0.1
```

The `mixture_spec` is a list of dataset entries. Each entry has:
- `dataset_path`: a dict keyed by embodiment tag, mapping to a list of data root paths
- `dataset_weight`: sampling weight in the mixture
- `distribute_weights`: whether to split weight across paths

---

### 2.4 `scripts/train/libero_training.sh`

**Purpose:** Shell script that launches DreamZero LoRA fine-tuning on LIBERO data.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `LIBERO_DATA_ROOT` | `./data/libero_spatial_lerobot` | Path to converted LIBERO dataset |
| `OUTPUT_DIR` | `./checkpoints/dreamzero_libero_lora` | Checkpoint output directory |
| `NUM_GPUS` | auto-detected via `nvidia-smi` | Number of GPUs for torchrun |
| `WAN_CKPT_DIR` | `./checkpoints/Wan2.1-I2V-14B-480P` | Wan2.1 video model checkpoint |
| `TOKENIZER_DIR` | `./checkpoints/umt5-xxl` | UMT5-XXL tokenizer |
| `MAX_STEPS` | 100000 | Training step limit |

**Hydra overrides passed via CLI:**

| Override | Value | Purpose |
|---|---|---|
| `data=dreamzero/libero_relative` | selects LIBERO data config | |
| `report_to=wandb` | enables W&B logging | |
| `wandb_project=dreamzero-libero` | W&B project name | |
| `train_architecture=lora` | LoRA adapter training | |
| `num_frames=33` | total video frames (overrides base 49) | |
| `action_horizon=24` | action chunk length (overrides base 48) | |
| `num_views=3` | 3 camera views | |
| `num_frame_per_block=2` | video frames per DiT block | |
| `num_action_per_block=24` | actions per DiT block | |
| `num_state_per_block=1` | state frames per DiT block | |
| `training_args.learning_rate=1e-5` | learning rate | |
| `training_args.deepspeed=".../zero2.json"` | DeepSpeed ZeRO-2 config | |
| `training_args.warmup_ratio=0.05` | LR warmup fraction | |
| `per_device_train_batch_size=4` | batch size per GPU | |
| `image_resolution_width=320` | video width (overrides base 480) | |
| `image_resolution_height=176` | video height (overrides base 256) | |
| `save_lora_only=true` | only save LoRA adapter weights | |
| `max_chunk_size=4` | action chunk size (overrides YAML 5) | |
| `frame_seqlen=880` | sequence length for frame tokens | |
| `libero_sim_data_root=$LIBERO_DATA_ROOT` | dataset path injection | |
| `pretrained_model_path=./checkpoints/DreamZero-AgiBot` | base model to fine-tune | |
| `++action_head_cfg.config.skip_component_loading=true` | skip loading action head weights from checkpoint | |
| `++action_head_cfg.config.defer_lora_injection=true` | defer LoRA injection until after model init | |

---

### 2.5 `groot/vla/experiment/experiment.py`

**Purpose:** Main training entry point. Uses `@hydra.main` to load configs and
orchestrate the experiment.

**Classes:**

| Class | Parent | Role |
|---|---|---|
| `VLATrainer` | `BaseTrainer` (which extends HF `Trainer`) | Custom training step with step timing, force-restart logic, and action head `global_step` sync |
| `VLATrainerInferenceBenchmark` | `VLATrainer` | Replaces `compute_loss` with inference benchmarking (warmup + timed loop) |
| `VLAExperiment` | `BaseExperiment` | Initializes experiment, dumps initial actions (for real robot), calls `train()` |

**`VLATrainer.training_step(model, inputs)`:**

1. Increments `micro_global_step`
2. Syncs `model.action_head.global_step` with trainer state
3. Optional benchmark timing (every 100 steps)
4. Force-restart check at save boundaries (`restart_max_seconds`)
5. Delegates to `super().training_step()`

**`main(cfg)` function:**

1. Calls `apply_action_overrides(cfg)` to update action dim/horizon from config
2. Instantiates `VLAExperiment(cfg)`
3. Calls `experiment.train()`

---

### 2.6 `groot/vla/experiment/base.py`

**Purpose:** Base experiment and trainer infrastructure. Handles model creation,
dataset creation, W&B setup, checkpoint management, and callbacks.

**Callbacks:**

| Callback | Purpose |
|---|---|
| `LossLoggerCallback` | Writes per-step loss metrics (`loss`, `dynamics_loss_avg`, `action_loss_avg`, `learning_rate`) to a JSONL file at `output_path` |
| `CheckpointFormatCallback` | On save: copies `experiment_cfg/` dir, `wandb_config.json` into each checkpoint directory for standalone reproducibility |
| `ProfCallback` | Optional PyTorch profiler with configurable start step, warmup, active window; auto-removes itself after profiling completes |

**`BaseExperiment.__init__(cfg)` (key operations, lines ~600+):**

1. Validates embodiment tags against `EmbodimentTag` enum
2. Instantiates `transformers.TrainingArguments` from Hydra config
3. Sets `WANDB_PROJECT`, `WANDB_RUN_ID`, `WANDB_DIR` environment variables
4. Creates `experiment_cfg/` directory, saves resolved `conf.yaml`
5. Writes `wandb_config.json` with project and run ID
6. Checks for existing checkpoints (resume logic)
7. Instantiates model via `create_model(cfg, training_args)`
8. Creates train dataset via `create_train_dataset(cfg, model)`
9. Dumps merged metadata to `experiment_cfg/metadata.json`

---

### 2.7 `groot/vla/data/schema/embodiment_tags.py`

**Purpose:** Enum of all recognized embodiment tags. The training pipeline uses this
to validate that a dataset's `embodiment.json` tag is known.

**LIBERO entry:**

```python
LIBERO_SIM = "libero_sim"
"""
The Libero Sim dataset.
"""
```

The string value `"libero_sim"` must match:
- `embodiment.json["embodiment_tag"]` in the dataset
- Keys in `modality_configs`, `transforms`, `metadata_versions`, `fps` maps in the base YAML
- The key under `dataset_path` in `mixture_spec`

---

## 3. Config File Relationships

```
conf.yaml (root)
  |
  |-- defaults:
  |     model: dreamzero/vla          (model architecture config)
  |     data:  dreamzero/libero_relative  (overridden by training script)
  |
  |-- training_args, trainer, wandb_project, etc.
  |
  +-- libero_relative.yaml (@package _global_)
        |
        |-- defaults:
        |     dreamzero/base_48_wan_fine_aug_relative  (loaded first)
        |     _self_                                    (then overrides)
        |
        |-- Overrides: relative_action, max_chunk_size, dataset classes, etc.
        |-- train_dataset.mixture_spec -> points to libero_sim_data_root
        |-- References: ${modality_configs}, ${transforms}, ${fps}
        |
        +-- base_48_wan_fine_aug_relative.yaml (@package _global_)
              |
              |-- Defines all transform anchors (&totensor_cfg, &crop_cfg, etc.)
              |-- Defines modality_config_libero_sim (video/state/action/language keys)
              |-- Defines transform_libero_sim (composed transform pipeline)
              |-- Registry maps: modality_configs, transforms, metadata_versions, fps
```

**Override precedence (highest to lowest):**

1. CLI overrides in `libero_training.sh` (e.g., `num_frames=33`)
2. `libero_relative.yaml` (`_self_` overrides)
3. `base_48_wan_fine_aug_relative.yaml` (base defaults)
4. `conf.yaml` (root defaults)

---

## 4. Data Flow

### 4.1 HDF5 to LeRobot v2

```
LIBERO HDF5                          LeRobot v2
-----------                          ----------
data/demo_0/obs/agentview_rgb    ->  videos/.../observation.images.agentview/ep_000000.mp4
data/demo_0/obs/eye_in_hand_rgb  ->  videos/.../observation.images.eye_in_hand/ep_000000.mp4
data/demo_0/obs/agentview_rgb    ->  videos/.../observation.images.agentview_2/ep_000000.mp4  (DUPLICATE)
data/demo_0/obs/joint_states(7)  \
                                  +->  parquet: observation.state = [joint(7), gripper(2)] = 9-dim
data/demo_0/obs/gripper_states(2)/
data/demo_0/actions(7)           ->  parquet: action = 7-dim (delta EE)
data.attrs.problem_info          ->  meta/tasks.jsonl
```

### 4.2 LeRobot v2 to GEAR Metadata

```
meta/info.json (already exists)
  + features dict (state shape, action shape, video keys)
    |
    v
convert_lerobot_to_gear.py
    |
    +-> meta/modality.json       (splits observation.state[0:7] -> "joint_states",
    |                              observation.state[7:9] -> "gripper_states", etc.)
    +-> meta/embodiment.json     (tag = "libero_sim")
    +-> meta/stats.json          (aggregated statistics over all parquet files)
    +-> meta/relative_stats_dreamzero.json  (delta statistics)
```

### 4.3 GEAR Metadata to Training

```
modality.json keys              YAML modality_config keys
-------------------             -------------------------
state.joint_states       <-->   state.joint_states       (modality_config_libero_sim.state.modality_keys)
state.gripper_states     <-->   state.gripper_states
action.joint_states      <-->   action.joint_states      (modality_config_libero_sim.action.modality_keys)
action.gripper_states    <-->   action.gripper_states
video.agentview          <-->   video.agentview           (modality_config_libero_sim.video.modality_keys)
video.eye_in_hand        <-->   video.eye_in_hand
video.agentview_2        <-->   video.agentview_2
annotation.language.language_instruction <--> annotation.language.language_instruction
```

The dataset loader (`ShardedLeRobotMixtureDataset`) reads each dataset's
`embodiment.json` to determine the embodiment tag, then looks up the corresponding
`ModalityConfig` and `ComposedModalityTransform` from the `all_modality_configs`
and `all_transforms` dicts (keyed by embodiment tag string).

### 4.4 Training to Inference

The model outputs 7-dim delta end-effector actions per timestep, chunked into
groups of up to `action_horizon` (24 in the training script). At inference:

```python
model.eval()
with torch.inference_mode():
    action = model.module.get_action(inputs)
```

The `metadata.json` saved in `experiment_cfg/` contains normalization stats
needed to un-normalize predicted actions back to the original action space.

---

## 5. Key Naming Conventions

### 5.1 modality.json Keys to YAML modality_config Keys

The naming convention uses a **prefix.suffix** pattern:

| Layer | modality.json key | YAML modality_keys entry | Parquet / video original_key |
|---|---|---|---|
| State | `joint_states` | `state.joint_states` | `observation.state` (indices 0-7) |
| State | `gripper_states` | `state.gripper_states` | `observation.state` (indices 7-9) |
| Action | `joint_states` | `action.joint_states` | `action` (indices 0-6) |
| Action | `gripper_states` | `action.gripper_states` | `action` (indices 6-7) |
| Video | `agentview` | `video.agentview` | `observation.images.agentview` |
| Video | `eye_in_hand` | `video.eye_in_hand` | `observation.images.eye_in_hand` |
| Video | `agentview_2` | `video.agentview_2` | `observation.images.agentview_2` |
| Language | `language.language_instruction` | `annotation.language.language_instruction` | task text from `tasks.jsonl` |

**Rule:** The YAML `modality_keys` prepend the modality type prefix (`state.`, `action.`,
`video.`, `annotation.`) to the short key name from `modality.json`. The dataset loader
uses `modality.json` to map these back to raw parquet columns and index ranges.

### 5.2 Embodiment Tag Consistency

The string `"libero_sim"` must appear identically in:
- `embodiment_tags.py`: `LIBERO_SIM = "libero_sim"`
- `convert_lerobot_to_gear.py`: `VALID_EMBODIMENT_TAGS` list
- `embodiment.json`: `{"embodiment_tag": "libero_sim"}`
- Base YAML: keys in `modality_configs`, `transforms`, `metadata_versions`, `fps`
- `libero_relative.yaml`: `mixture_spec[0].dataset_path.libero_sim`

---

## 6. Camera Mapping: 2-to-3 Duplication Strategy

LIBERO provides only 2 camera views:

| LIBERO HDF5 Key | Description |
|---|---|
| `obs/agentview_rgb` | Third-person view of the workspace |
| `obs/eye_in_hand_rgb` | Wrist-mounted camera |

DreamZero's architecture expects exactly 3 camera views (matching its pre-training
on 3-view robot datasets like AGIBot and DROID). The solution is to **duplicate**
the agentview as a third camera:

```python
# In convert_libero_to_lerobot.py
LIBERO_CAM_MAP = {
    "observation.images.agentview":   "agentview_rgb",      # cam0
    "observation.images.eye_in_hand": "eye_in_hand_rgb",    # cam1
    "observation.images.agentview_2": "agentview_rgb",      # cam2 = DUPLICATE of cam0
}
```

This produces three identical-format MP4 streams. The model processes all three through
its video encoder. In the YAML, they appear as three entries in
`modality_config_libero_sim.video.modality_keys`.

Video frame sampling uses `delta_indices: [0..24]` (25 frames for training) but only
`eval_delta_indices: [0]` (single frame) at evaluation time.

---

## 7. Action Space Handling

### 7.1 Dimensions

| Component | Dimensions | Source |
|---|---|---|
| State: joint_states | 7 | `obs/joint_states` in HDF5 |
| State: gripper_states | 2 | `obs/gripper_states` in HDF5 |
| Action: joint_states | 6 | `actions[0:6]` in HDF5 |
| Action: gripper_states | 1 | `actions[6:7]` in HDF5 |
| Total state | 9 | Concatenated in parquet `observation.state` |
| Total action | 7 | Stored in parquet `action` |

### 7.2 Relative vs. Absolute Actions

LIBERO uses **delta end-effector** actions natively (the actions are already relative
displacements). Therefore:

```yaml
# libero_relative.yaml
relative_action: false              # DO NOT subtract state from action
relative_action_per_horizon: false  # same
relative_action_keys:               # listed for stats computation only
  - joint_states
  - gripper_states
```

The `relative_stats_dreamzero.json` is still computed by `convert_lerobot_to_gear.py`
for consistency, but `relative_action: false` means the training loop will NOT apply
the `action[t] - state[t]` transformation at runtime.

### 7.3 Normalization

Both state and action use `q99` normalization mode (99th percentile scaling):

```yaml
normalization_modes:
  state.joint_states: q99
  state.gripper_states: q99
  action.joint_states: q99
  action.gripper_states: q99
```

Statistics come from `meta/stats.json` (computed by `convert_lerobot_to_gear.py`).

### 7.4 Action Chunking

| Parameter | Base YAML | Training Script Override |
|---|---|---|
| `action_horizon` | 48 | 24 |
| `max_chunk_size` | 5 (libero_relative.yaml) | 4 |
| `num_action_per_block` | N/A | 24 |

The model predicts `action_horizon` steps at once. `max_chunk_size` controls how many
consecutive action chunks are sampled during training.

---

## 8. W&B Logging Integration Points

### 8.1 Setup (`base.py`, `BaseExperiment.__init__`)

```python
# Lines ~615-641
os.environ["WANDB_PROJECT"] = cfg.wandb_project      # "dreamzero-libero"
os.environ["WANDB_RUN_ID"]  = os.environ.get("RUNTIME_ID", <auto>)
os.environ["WANDB_DIR"]     = training_args.output_dir

# Persisted to checkpoint for reproducibility:
wandb_config.json = {"project": ..., "run_id": ...}
```

### 8.2 Reporting (`conf.yaml`)

```yaml
report_to: wandb           # passed to transformers.TrainingArguments
logging_steps: 10.0         # log metrics every 10 steps
```

HuggingFace Trainer's built-in W&B integration handles:
- Training loss per `logging_steps`
- Learning rate schedule
- GPU memory usage
- Gradient norms (if enabled)

### 8.3 Custom Loss Logging (`LossLoggerCallback`)

In addition to W&B, `LossLoggerCallback` writes a JSONL file with per-step entries:

```json
{"step": 100, "loss": 0.45, "dynamics_loss_avg": 0.12, "action_loss_avg": 0.33, "learning_rate": 9.5e-6}
```

### 8.4 Checkpoint Callback (`CheckpointFormatCallback`)

On every checkpoint save, copies `wandb_config.json` into the checkpoint directory
so that evaluation runs can link back to the same W&B run.

---

## 9. How to Run Each Step

### Step 1: Convert LIBERO HDF5 to LeRobot v2

```bash
python scripts/data/convert_libero_to_lerobot.py \
    --src-path /workspace/LIBERO_work/datasets/libero_spatial \
    --tgt-path /workspace/dreamzero_work/data/libero_spatial_lerobot \
    --fps 20

# Debug mode (2 demos per task):
python scripts/data/convert_libero_to_lerobot.py \
    --src-path /workspace/LIBERO_work/datasets/libero_spatial \
    --tgt-path /workspace/dreamzero_work/data/libero_spatial_lerobot \
    --fps 20 --debug
```

### Step 2: Generate GEAR Metadata

```bash
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path /workspace/dreamzero_work/data/libero_spatial_lerobot \
    --embodiment-tag libero_sim \
    --state-keys '{"joint_states": [0, 7], "gripper_states": [7, 9]}' \
    --action-keys '{"joint_states": [0, 6], "gripper_states": [6, 7]}' \
    --relative-action-keys joint_states gripper_states \
    --task-key annotation.language.language_instruction
```

### Step 3: Launch Training

```bash
# Using the provided script (auto-detects GPUs):
bash scripts/train/libero_training.sh

# Or with custom overrides:
LIBERO_DATA_ROOT=/workspace/dreamzero_work/data/libero_spatial_lerobot \
OUTPUT_DIR=./checkpoints/my_libero_run \
NUM_GPUS=4 \
MAX_STEPS=50000 \
bash scripts/train/libero_training.sh
```

### Step 4: Inference (from trained checkpoint)

```python
# Pseudocode -- actual inference script depends on evaluation harness
model.eval()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.inference_mode():
        action = model.module.get_action(inputs)
        # action contains un-normalized 7-dim delta EE commands
```

### Prerequisites Checklist

1. LIBERO HDF5 dataset files in `--src-path`
2. DreamZero-AgiBot checkpoint at `./checkpoints/DreamZero-AgiBot`
3. Wan2.1-I2V-14B-480P model at `./checkpoints/Wan2.1-I2V-14B-480P`
4. UMT5-XXL tokenizer at `./checkpoints/umt5-xxl`
5. Python packages: `h5py`, `av` (PyAV), `pandas`, `pyarrow`, `tqdm`, `numpy`,
   `hydra-core`, `omegaconf`, `transformers`, `deepspeed`, `torch`, `decord`, `wandb`

---

## 10. Summary of File Locations

| File | Path |
|---|---|
| HDF5-to-LeRobot converter | `scripts/data/convert_libero_to_lerobot.py` |
| LeRobot-to-GEAR converter | `scripts/data/convert_lerobot_to_gear.py` |
| Base data config (all embodiments) | `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` |
| LIBERO data config (overrides) | `groot/vla/configs/data/dreamzero/libero_relative.yaml` |
| Root Hydra config | `groot/vla/configs/conf.yaml` |
| Training shell script | `scripts/train/libero_training.sh` |
| Experiment entry point | `groot/vla/experiment/experiment.py` |
| Base experiment / trainer | `groot/vla/experiment/base.py` |
| Embodiment tag enum | `groot/vla/data/schema/embodiment_tags.py` |

All paths above are relative to `/workspace/dreamzero_work/`.

---

## 11. Inference Pipeline Architecture

### 11.1 Server Launch

The inference server is launched via `socket_test_optimized_AR.py`, which instantiates
a `GrootSimPolicy` that loads the model from a checkpoint directory.

**Checkpoint requirements:**

| File | Purpose |
|---|---|
| `experiment_cfg/conf.yaml` | Full Hydra config (transforms, modality configs, normalization stats, etc.) |
| `config.json` | Model architecture metadata |
| `model.safetensors` | LoRA adapter weights (when `save_lora_only=true`) |

**Model loading sequence:**

1. `GrootSimPolicy.__init__` reads `experiment_cfg/conf.yaml` to get transforms, modality configs, normalization stats, and other runtime parameters.
2. If `save_lora_only=true` in config: calls `load_lora()` which creates the full base model (14B DiT) and then loads LoRA adapter weights on top.
3. The base model (14B DiT) is loaded from paths specified in config (`WAN_CKPT_DIR`).

**Server launch command:**

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    socket_test_optimized_AR.py \
    --port 5000 \
    --enable-dit-cache \
    --embodiment-tag libero_sim \
    --model-path <checkpoint_path>
```

The `--embodiment-tag` CLI argument was added to support LIBERO (previously hardcoded to `oxe_droid`).

### 11.2 Observation Flow

Client sends observations via WebSocket to `ARDroidRoboarenaPolicy._convert_observation()`,
which performs the following key mapping:

```
Client observation key                  -> Internal model key (frame buffer)
---------------------------------------   ------------------------------------
observation/exterior_image_0_left       -> video.exterior_image_1_left
observation/exterior_image_1_left       -> video.exterior_image_2_left
observation/wrist_image_left            -> video.wrist_image_left
```

**Frame accumulation behavior:**

- The server accumulates frames across calls (buffer size = `FRAMES_PER_CHUNK=4`).
- First call processes a single frame.
- Subsequent calls accumulate frames into the buffer until the chunk is full.
- The model processes the accumulated frame buffer to generate actions.

### 11.3 Action Flow

The model predicts a 24-action chunk in a single forward pass. The server returns
the full chunk to the client.

**Client-side execution:**

- The client uses **open-loop execution** (default 8 steps) before re-querying the server.
- For LIBERO: each action is 7-dim delta end-effector commands `[pos(3), rot(3), gripper(1)]`.
- The client executes actions sequentially via `env.step(action[:7])`.

### 11.4 Eval Client (`eval_dreamzero.py`)

The eval client bridges the LIBERO simulation environment (running in the `libero` conda
env) with the DreamZero inference server (running in the `dreamzero` conda env).

**Observation pipeline:**

1. LIBERO env produces: `agentview_rgb` (128x128), `eye_in_hand_rgb` (128x128), joint/gripper states.
2. Eval client maps to DreamZero expected observation keys.
3. Agentview image is **duplicated** as the second exterior camera (LIBERO has 2 cameras, DreamZero expects 3).
4. Images are **resized** from 128x128 to 180x320 to match DreamZero's expected input resolution.

**Results output:**

| Output | Format | Description |
|---|---|---|
| `eval_progress.jsonl` | JSONL | Per-task progress log with episode results |
| `success_rates_{suite}.png` | PNG | Matplotlib bar chart of per-task success rates |
| `eval_summary.json` | JSON | Overall summary with aggregate statistics |

No W&B dependency -- all visualization is local (TensorBoard for training, matplotlib for eval).
