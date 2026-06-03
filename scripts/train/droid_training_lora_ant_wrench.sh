#!/usr/bin/env bash
set -Eeuo pipefail

# LoRA training entrypoint for the ANT Isaac OXE-DROID wrench-state experiment.
# Uses the copied dataset variant whose meta/modality.json exposes state.wrist_wrench.

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
export DREAMZERO_ROOT

export DATA_CONFIG="${DATA_CONFIG:-dreamzero/droid_relative_wrench}"
export DROID_DATA_ROOT="${DROID_DATA_ROOT:-/team/datasets/ant_oxe_droid_isaac/_isaac_gello_dataset_320x176_gear_wrench}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DREAMZERO_ROOT/checkpoints/dreamzero_droid_lora_ant_320x176_wrench}"

exec bash "$SCRIPT_DIR/droid_training_lora_ant.sh" "$@"
