#!/usr/bin/env bash
set -Eeuo pipefail

# DreamZero LoRA training for the ANT Isaac OXE-DROID collection.
#
# Expected input:
#   A GEAR-converted dataset, e.g.
#     /team/datasets/ant_oxe_droid_isaac/_isaac_gello_dataset_320x176_gear
#
# Common usage on the training server:
#   bash scripts/train/droid_training_lora_ant.sh

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${DREAMZERO_ROOT:-}" && -d "$DREAMZERO_ROOT/groot" ]]; then
    :
elif [[ -d "$SCRIPT_REPO_ROOT/groot" ]]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
elif [[ -d "/root/yejink/dreamzero/groot" ]]; then
    DREAMZERO_ROOT="/root/yejink/dreamzero"
elif [[ -d "/root/dreamzero/groot" ]]; then
    DREAMZERO_ROOT="/root/dreamzero"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-$SCRIPT_REPO_ROOT}"
fi

if [[ ! -d "$DREAMZERO_ROOT/groot" ]]; then
    echo "ERROR: No groot/ under DREAMZERO_ROOT=$DREAMZERO_ROOT"
    echo "Set DREAMZERO_ROOT to the dreamzero repo root."
    exit 1
fi

DROID_DATA_ROOT="${DROID_DATA_ROOT:-/team/datasets/ant_oxe_droid_isaac/_isaac_gello_dataset_320x176_gear}"
if [[ -z "${DATA_CONFIG:-}" && "$DROID_DATA_ROOT" == *"_wrench"* ]]; then
    DATA_CONFIG="dreamzero/droid_relative_wrench"
else
    DATA_CONFIG="${DATA_CONFIG:-dreamzero/droid_relative}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DREAMZERO_ROOT/checkpoints/dreamzero_droid_lora_ant_320x176}"

WAN_CKPT_DIR="${WAN_CKPT_DIR:-/team/models/Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/team/models/Wan2.1-I2V-14B-480P/google/umt5-xxl}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/team/datasets/DreamZero/DreamZero-DROID}"

NUM_GPUS="${NUM_GPUS:-8}"
if [[ "$NUM_GPUS" -lt 1 ]]; then
    NUM_GPUS=1
fi

PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
STEP_BATCH_SIZE="$((NUM_GPUS * PER_DEVICE_BS))"
if (( GLOBAL_BATCH_SIZE % STEP_BATCH_SIZE != 0 )); then
    echo "ERROR: GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS*PER_DEVICE_BS=$STEP_BATCH_SIZE"
    exit 1
fi

# Conservative defaults for a small 107-episode sim dataset.
LR="${LR:-5e-5}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
REPORT_TO="${REPORT_TO:-wandb}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-zero2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-2}"
VIDEO_BACKEND="${VIDEO_BACKEND:-ffmpeg}"

REQUIRED_DATA_FILES=(
    "$DROID_DATA_ROOT/meta/info.json"
    "$DROID_DATA_ROOT/meta/modality.json"
    "$DROID_DATA_ROOT/meta/embodiment.json"
    "$DROID_DATA_ROOT/meta/stats.json"
    "$DROID_DATA_ROOT/meta/relative_stats_dreamzero.json"
)

if [[ ! -d "$DROID_DATA_ROOT" ]]; then
    echo "ERROR: GEAR dataset not found at DROID_DATA_ROOT=$DROID_DATA_ROOT"
    echo "Set DROID_DATA_ROOT to your uploaded _isaac_gello_dataset_320x176_gear folder."
    exit 1
fi

for required_file in "${REQUIRED_DATA_FILES[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: Missing required converted dataset file: $required_file"
        echo "Run scripts/data/convert_lerobot_to_gear.py before training."
        exit 1
    fi
done

if [[ ! -d "$WAN_CKPT_DIR" ]]; then
    echo "ERROR: WAN_CKPT_DIR not found: $WAN_CKPT_DIR"
    exit 1
fi
if [[ ! -f "$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" ]]; then
    echo "ERROR: Missing text encoder: $WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
    exit 1
fi
if [[ ! -f "$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]]; then
    echo "ERROR: Missing image encoder: $WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    exit 1
fi
if [[ ! -f "$WAN_CKPT_DIR/Wan2.1_VAE.pth" ]]; then
    echo "ERROR: Missing VAE: $WAN_CKPT_DIR/Wan2.1_VAE.pth"
    exit 1
fi
if [[ ! -d "$TOKENIZER_DIR" ]]; then
    echo "ERROR: TOKENIZER_DIR not found: $TOKENIZER_DIR"
    exit 1
fi
if [[ ! -f "$PRETRAINED_MODEL_PATH/model.safetensors" && ! -f "$PRETRAINED_MODEL_PATH/model.safetensors.index.json" ]]; then
    echo "ERROR: PRETRAINED_MODEL_PATH must contain model.safetensors or model.safetensors.index.json"
    echo "Current PRETRAINED_MODEL_PATH=$PRETRAINED_MODEL_PATH"
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
if [[ ! -f "$EXPERIMENT_PY" ]]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
elif [[ -x "/usr/bin/python3.11" ]]; then
    PYTHON_BIN="/usr/bin/python3.11"
else
    PYTHON_BIN="$(command -v python3)"
fi

RUN_CMD=(
    "$PYTHON_BIN" -m torch.distributed.run
    --nproc_per_node "$NUM_GPUS"
    --standalone
    "$EXPERIMENT_PY"
)

echo "[droid-lora-ant] DREAMZERO_ROOT=$DREAMZERO_ROOT"
echo "[droid-lora-ant] DATA_CONFIG=$DATA_CONFIG"
echo "[droid-lora-ant] DROID_DATA_ROOT=$DROID_DATA_ROOT"
echo "[droid-lora-ant] OUTPUT_DIR=$OUTPUT_DIR"
echo "[droid-lora-ant] WAN_CKPT_DIR=$WAN_CKPT_DIR"
echo "[droid-lora-ant] TOKENIZER_DIR=$TOKENIZER_DIR"
echo "[droid-lora-ant] PRETRAINED_MODEL_PATH=$PRETRAINED_MODEL_PATH"
echo "[droid-lora-ant] NUM_GPUS=$NUM_GPUS PER_DEVICE_BS=$PER_DEVICE_BS GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "[droid-lora-ant] LR=$LR MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS NUM_FRAME_PER_BLOCK=$NUM_FRAME_PER_BLOCK"
echo "[droid-lora-ant] VIDEO_BACKEND=$VIDEO_BACKEND"
echo "[droid-lora-ant] Python=$PYTHON_BIN"

cd "$DREAMZERO_ROOT"

"${RUN_CMD[@]}" \
    report_to="$REPORT_TO" \
    data="$DATA_CONFIG" \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block="$NUM_FRAME_PER_BLOCK" \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate="$LR" \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps="$SAVE_STEPS" \
    training_args.warmup_ratio="$WARMUP_RATIO" \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size="$PER_DEVICE_BS" \
    global_batch_size="$GLOBAL_BATCH_SIZE" \
    max_steps="$MAX_STEPS" \
    weight_decay="$WEIGHT_DECAY" \
    save_total_limit="$SAVE_TOTAL_LIMIT" \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    video_backend="$VIDEO_BACKEND" \
    droid_data_root="$DROID_DATA_ROOT" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    pretrained_model_path="$PRETRAINED_MODEL_PATH"
