#!/bin/bash
# DreamZero LIBERO Fine-Tuning Script
#
# Usage:
#   bash scripts/train/libero_training.sh
#
# Prerequisites:
#   - LIBERO dataset converted to LeRobot v2 + GEAR format at LIBERO_DATA_ROOT
#   - DreamZero-AgiBot checkpoint at ./checkpoints/DreamZero-AgiBot
#   - Wan2.1-I2V-14B-480P and umt5-xxl at ./checkpoints/

export HYDRA_FULL_ERROR=1

# ============ CHANGE THESE VARIABLES ============
LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-"/workspace/dreamzero_work/data/libero_spatial_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/dreamzero_work/output/dreamzero_libero_lora"}

if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-2}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"/workspace/dreamzero/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/workspace/dreamzero/checkpoints/umt5-xxl"}
PRETRAINED_PATH=${PRETRAINED_PATH:-"/workspace/dreamzero/checkpoints/DreamZero-DROID"}
MAX_STEPS=${MAX_STEPS:-100000}
# Visualization: tensorboard (local) or wandb
REPORT_TO=${REPORT_TO:-"tensorboard"}
# =============================================

# Validate dataset exists
if [ ! -d "$LIBERO_DATA_ROOT" ]; then
    echo "ERROR: LIBERO dataset not found at $LIBERO_DATA_ROOT"
    echo "Run convert_libero_to_lerobot.py + convert_lerobot_to_gear.py first."
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=$REPORT_TO \
    data=dreamzero/libero_relative \
    wandb_project=dreamzero-libero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=50 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    libero_sim_data_root=$LIBERO_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_PATH \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
