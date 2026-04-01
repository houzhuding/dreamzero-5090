#!/bin/bash
export HYDRA_FULL_ERROR=1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dreamzero

# ============ CONFIGURATION ============
DATA_ROOT=${DATA_ROOT:?"Set DATA_ROOT to your GEAR-converted dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_robocasa_bimanual_panda_parallel_gripper_lora"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}
WAN_CKPT_DIR="/team/models/Wan2.1-I2V-14B-480P"
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_REPO=${TOKENIZER_REPO:-"google/umt5-xxl"}
# =======================================

# Auto-download Wan2.1 weights if missing (optional - can be pre-downloaded)
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "INFO: Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR"
    echo "      It will be downloaded on-demand from HuggingFace during training"
    echo "      Or pre-download with: huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir \"$WAN_CKPT_DIR\""
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing — run convert_lerobot_to_gear.py first"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/robocasa_bimanual_panda_parallel_gripper_relative \
    wandb_project=dreamzero \
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
    save_steps=10000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=10 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=0 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    robocasa_bimanual_panda_parallel_gripper_data_root=$DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_REPO
    pretrained_model_path=/teams/DreamZero-AgiBot
