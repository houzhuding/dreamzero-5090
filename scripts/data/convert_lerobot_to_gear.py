"""
Convert a standard LeRobot v2 dataset to the GEAR/DreamZero training format.

This script takes a dataset collected with LeRobot v2 and generates/augments the
metadata files required by DreamZero's training pipeline:

  - meta/modality.json    (state/action/video/annotation key mapping)
  - meta/embodiment.json  (embodiment tag for the training pipeline)
  - meta/stats.json       (dataset-level statistics: mean, std, min, max, q01, q99)
  - meta/relative_stats_dreamzero.json  (relative action statistics)
  - meta/tasks.jsonl      (task descriptions)
  - meta/episodes.jsonl   (episode-level metadata)

The script does NOT modify parquet files or videos -- it only creates metadata.

Usage:
  # Auto-detect state/action structure, default embodiment tag 'xdof':
  python scripts/data/convert_lerobot_to_gear.py --dataset-path ./Dataset/my_robot_data

  # Explicit modality mapping via JSON:
  python scripts/data/convert_lerobot_to_gear.py \\
      --dataset-path ./Dataset/my_robot_data \\
      --embodiment-tag xdof \\
      --state-keys '{"joint_pos": [0, 6], "gripper_pos": [6, 7]}' \\
      --action-keys '{"joint_pos": [0, 6], "gripper_pos": [6, 7]}' \\
      --relative-action-keys joint_pos \\
      --task-key annotation.task

  # Copy to a new output directory instead of modifying in-place:
  python scripts/data/convert_lerobot_to_gear.py \\
      --dataset-path ./Dataset/my_robot_data \\
      --output-path ./Dataset/my_robot_data_gear
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

VALID_EMBODIMENT_TAGS = [
    "real_gr1_arms_only", "real_gr1_arms_only_annotated",
    "real_gr1_arms_waist", "real_gr1_arms_waist_annotated",
    "dexmg_gr1_arms_only_inspire", "dexmg_gr1_arms_only_fourier",
    "dexmg_gr1_arms_waist_fourier",
    "robocasa_single_arm", "onex_eve_gripper",
    "robocasa_gr1_arms_only_inspire_hands", "robocasa_gr1_arms_only_fourier_hands",
    "robocasa_gr1_fixed_lower_body_inspire_hands", "robocasa_gr1_fixed_lower_body_fourier_hands",
    "robocasa_panda_omron",
    "robocasa_bimanual_panda_parallel_gripper", "robocasa_bimanual_panda_inspire_hand",
    "oxe_droid", "oxe_fractal", "oxe_language_table", "oxe_bridge",
    "real_panda_single_arm", "hot3d_hands_only",
    "gr1_unified", "robocasa_gr1_arms_waist_fourier_hands",
    "agibot", "lapa", "oxe_mutex", "oxe_roboset", "oxe_plex",
    "dream", "yam", "xdof",
    "gr1_unified_segmentation", "language_table_sim", "gr1_isaac",
    "sim_behavior_r1_pro", "mecka_hands", "real_r1_pro_sharpa",
    "ant_bimanual_fr3_parallel_gripper",
]

EMBODIMENT_MODALITY_PRESETS = {
    # Canonical DROID / OXE-DROID LeRobot schema:
    # observation.state = cartesian_position(6), gripper_position(1), joint_position(7)
    # action = cartesian_position(6), cartesian_velocity(6), gripper_position(1),
    #          gripper_velocity(1), joint_position(7), joint_velocity(7)
    "oxe_droid": {
        "state": {
            "cartesian_position": [0, 6],
            "gripper_position": [6, 7],
            "joint_position": [7, 14],
        },
        "action": {
            "cartesian_position": [0, 6],
            "cartesian_velocity": [6, 12],
            "gripper_position": [12, 13],
            "gripper_velocity": [13, 14],
            "joint_position": [14, 21],
            "joint_velocity": [21, 28],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        log.error("meta/info.json not found at %s", info_path)
        sys.exit(1)
    with open(info_path) as f:
        return json.load(f)


def episode_index_from_path(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return -1


def load_existing_tasks(dataset_path: Path) -> tuple[dict[int, str], dict[str, int]]:
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    by_index: dict[int, str] = {}
    by_text: dict[str, int] = {}
    if not tasks_path.exists():
        return by_index, by_text
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = int(row["task_index"])
            text = str(row["task"])
            by_index[idx] = text
            by_text[text] = idx
    return by_index, by_text


def resolve_task_value(value, task_index_to_text: dict[int, str]) -> str:
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        idx = int(value)
    except Exception:
        return str(value)
    return task_index_to_text.get(idx, str(value))


def get_parquet_paths(dataset_path: Path, info: dict) -> list[Path]:
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    total_episodes = int(info.get("total_episodes", 0))
    chunks_size = int(info.get("chunks_size", 1000))
    episode_ids: list[int] = []
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        with open(episodes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                episode_ids.append(int(row["episode_index"]))

    if not episode_ids and total_episodes > 0:
        episode_ids = list(range(total_episodes))

    paths: list[Path] = []
    for ep_idx in episode_ids:
        chunk_idx = ep_idx // chunks_size
        p = dataset_path / pattern.format(episode_chunk=chunk_idx, episode_index=ep_idx)
        if p.exists():
            paths.append(p)

    if not paths:
        paths = sorted((dataset_path / "data").glob("chunk-*/episode_*.parquet"), key=episode_index_from_path)

    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(paths, key=episode_index_from_path):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def detect_features(info: dict) -> dict:
    """Return categorised feature names from info.json."""
    features = info.get("features", {})
    state_keys = [k for k in features if k.startswith("observation.state")]
    action_keys = [k for k in features if k == "action" or k.startswith("action.")]
    video_keys = [k for k in features if features[k].get("dtype") == "video"]
    annotation_keys = [k for k in features if k.startswith("annotation")]
    return {
        "state": state_keys,
        "action": action_keys,
        "video": video_keys,
        "annotation": annotation_keys,
        "features": features,
    }


def parse_key_mapping(raw: str | None) -> dict[str, list[int]] | None:
    """Parse a JSON string like '{"joint_pos": [0, 6], "gripper": [6, 7]}'."""
    if raw is None:
        return None
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("Invalid JSON for key mapping: %s", e)
        sys.exit(1)
    for name, bounds in mapping.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            log.error("Each entry must be [start, end]. Got %s for '%s'", bounds, name)
            sys.exit(1)
    return mapping


def parse_layout_mapping(info: dict, layout_key: str) -> dict[str, list[int]] | None:
    """Parse a mapping from info.json collection_config.<layout_key>.

    Expected form:
      info["collection_config"][layout_key] = {"name": [start, end], ...}
    """
    collection_cfg = info.get("collection_config", {})
    raw_layout = collection_cfg.get(layout_key)
    if raw_layout is None:
        return None
    if not isinstance(raw_layout, dict):
        log.warning("collection_config.%s is not a dict, ignoring", layout_key)
        return None

    parsed: dict[str, list[int]] = {}
    for name, bounds in raw_layout.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            log.warning("Invalid %s entry for '%s': %s", layout_key, name, bounds)
            continue
        start, end = bounds
        try:
            parsed[name] = [int(start), int(end)]
        except Exception:
            log.warning("Non-integer bounds in %s for '%s': %s", layout_key, name, bounds)

    return parsed or None


def preset_mapping(embodiment_tag: str, modality_type: str) -> dict[str, list[int]] | None:
    preset = EMBODIMENT_MODALITY_PRESETS.get(embodiment_tag)
    if not preset:
        return None
    mapping = preset.get(modality_type)
    return dict(mapping) if mapping else None


def mapping_from_existing_modality(modality: dict | None, modality_type: str) -> dict[str, list[int]] | None:
    if not modality or modality_type not in modality:
        return None
    out: dict[str, list[int]] = {}
    for name, cfg in modality[modality_type].items():
        if "start" in cfg and "end" in cfg:
            out[name] = [int(cfg["start"]), int(cfg["end"])]
    return out or None


def default_vector_mapping(info: dict, detected: dict, modality_type: str) -> dict[str, list[int]] | None:
    cols = detected[modality_type]
    if not cols:
        return None
    col = cols[0]
    shape = info.get("features", {}).get(col, {}).get("shape", [1])
    dim = int(shape[0] if isinstance(shape, list) else shape)
    return {modality_type: [0, dim]}


def choose_mapping(
    *,
    explicit: dict[str, list[int]] | None,
    existing_modality: dict | None,
    info: dict,
    detected: dict,
    modality_type: str,
    embodiment_tag: str,
) -> tuple[dict[str, list[int]] | None, str]:
    if explicit is not None:
        return explicit, "explicit CLI"

    existing = mapping_from_existing_modality(existing_modality, modality_type)
    if existing is not None:
        return existing, "existing modality.json"

    layout = parse_layout_mapping(info, f"{modality_type}_vector_layout")
    if layout is not None:
        return layout, f"collection_config.{modality_type}_vector_layout"

    preset = preset_mapping(embodiment_tag, modality_type)
    if preset is not None:
        return preset, f"{embodiment_tag} preset"

    return default_vector_mapping(info, detected, modality_type), "generic full-vector fallback"


def parse_relative_key_map(raw: list[str] | None) -> dict[str, str] | None:
    """Parse action->state key mapping entries like:

    ["left_target_pose9d_rot6d=left_ee_pose9d_rot6d", "right_target_pose9d_rot6d=right_ee_pose9d_rot6d"]
    """
    if not raw:
        return None

    mapping: dict[str, str] = {}
    for entry in raw:
        if "=" not in entry:
            log.error("Invalid --relative-action-state-map entry '%s' (expected action_key=state_key)", entry)
            sys.exit(1)
        action_key, state_key = entry.split("=", 1)
        action_key = action_key.strip()
        state_key = state_key.strip()
        if not action_key or not state_key:
            log.error("Invalid --relative-action-state-map entry '%s' (empty action/state key)", entry)
            sys.exit(1)
        mapping[action_key] = state_key
    return mapping


# ---------------------------------------------------------------------------
# Modality JSON
# ---------------------------------------------------------------------------

def build_modality_json(
    info: dict,
    detected: dict,
    state_mapping: dict[str, list[int]] | None,
    action_mapping: dict[str, list[int]] | None,
    task_key: str | None,
) -> dict:
    """Build the modality.json structure expected by GEAR/DreamZero."""
    features = detected["features"]
    modality: dict = {"state": {}, "action": {}, "video": {}, "annotation": {}}

    # --- State ---
    state_col = detected["state"][0] if detected["state"] else None
    if state_col and state_mapping:
        for name, (start, end) in state_mapping.items():
            dtype = features[state_col].get("dtype", "float64")
            modality["state"][name] = {
                "original_key": state_col,
                "start": start,
                "end": end,
                "rotation_type": None,
                "absolute": True,
                "dtype": dtype,
                "range": None,
            }
    elif state_col:
        shape = features[state_col].get("shape", [1])
        dim = shape[0] if isinstance(shape, list) else shape
        dtype = features[state_col].get("dtype", "float64")
        modality["state"]["state"] = {
            "original_key": state_col,
            "start": 0,
            "end": dim,
            "rotation_type": None,
            "absolute": True,
            "dtype": dtype,
            "range": None,
        }

    # --- Action ---
    action_col = detected["action"][0] if detected["action"] else None
    if action_col and action_mapping:
        for name, (start, end) in action_mapping.items():
            dtype = features[action_col].get("dtype", "float64")
            modality["action"][name] = {
                "original_key": action_col,
                "start": start,
                "end": end,
                "rotation_type": None,
                "absolute": True,
                "dtype": dtype,
                "range": None,
            }
    elif action_col:
        shape = features[action_col].get("shape", [1])
        dim = shape[0] if isinstance(shape, list) else shape
        dtype = features[action_col].get("dtype", "float64")
        modality["action"]["action"] = {
            "original_key": action_col,
            "start": 0,
            "end": dim,
            "rotation_type": None,
            "absolute": True,
            "dtype": dtype,
            "range": None,
        }

    # --- Video ---
    for vk in detected["video"]:
        short_name = vk.replace("observation.images.", "")
        modality["video"][short_name] = {"original_key": vk}

    # --- Annotation ---
    annotation_keys = list(detected["annotation"])
    if task_key and task_key not in annotation_keys:
        annotation_keys.insert(0, task_key)
    for ak in annotation_keys:
        short = ak.replace("annotation.", "")
        modality["annotation"][short] = {"original_key": ak}

    return modality


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats(parquet_paths: list[Path], columns: list[str]) -> dict:
    """Compute mean/std/min/max/q01/q99 for numeric columns across all episodes."""
    all_data: dict[str, list] = {col: [] for col in columns}
    for pp in tqdm(parquet_paths, desc="Computing stats"):
        df = pd.read_parquet(pp)
        for col in columns:
            if col not in df.columns:
                continue
            arr = np.stack(df[col].values)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            all_data[col].append(arr)

    stats = {}
    for col in columns:
        if not all_data[col]:
            continue
        data = np.concatenate(all_data[col], axis=0).astype(np.float64)
        stats[col] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }
    return stats



def compute_stats_by_modality(parquet_paths: list[Path], modality: dict, columns: list[str]) -> dict:
    """Compute mean/std/min/max/q01/q99 keyed by modality field names.
    
    Instead of keying by 'observation.state' (which has all state fields),
    split it into individual keys like 'left_q', 'right_q', etc. based on
    the modality.json index ranges.
    """
    # First pass: collect raw data by column name
    all_data: dict[str, list] = {col: [] for col in columns}
    for pp in tqdm(parquet_paths, desc="Computing stats"):
        df = pd.read_parquet(pp)
        for col in columns:
            if col not in df.columns:
                continue
            arr = np.stack(df[col].values)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            all_data[col].append(arr)
    
    # Consolidate raw data
    consolidated: dict[str, np.ndarray] = {}
    for col in columns:
        if not all_data[col]:
            continue
        data = np.concatenate(all_data[col], axis=0).astype(np.float64)
        consolidated[col] = data
    
    # Second pass: extract stats by modality field
    stats = {}
    for modality_type in ['state', 'action']:
        stats[modality_type] = {}
        if modality_type not in modality:
            continue
        
        for field_name, field_info in modality[modality_type].items():
            original_col = field_info['original_key']
            start = field_info['start']
            end = field_info['end']
            
            if original_col not in consolidated:
                log.warning(f"{original_col} not found in data, skipping {field_name}")
                continue
            
            # Extract the slice for this field
            full_data = consolidated[original_col]  # shape: (N, total_dim)
            field_data = full_data[:, start:end]     # shape: (N, field_dim)
            
            # Compute statistics
            stats[modality_type][field_name] = {
                "mean": np.mean(field_data, axis=0).tolist(),
                "std": np.std(field_data, axis=0).tolist(),
                "min": np.min(field_data, axis=0).tolist(),
                "max": np.max(field_data, axis=0).tolist(),
                "q01": np.quantile(field_data, 0.01, axis=0).tolist(),
                "q99": np.quantile(field_data, 0.99, axis=0).tolist(),
            }
    
    return stats


def compute_relative_stats(
    parquet_paths: list[Path],
    modality: dict,
    relative_action_keys: list[str],
    action_to_state_key_map: dict[str, str] | None = None,
    action_horizon: int = 24,
) -> dict:
    """Compute relative-action statistics: (action - reference_state) for each key.

    This replicates the logic in groot/vla/data/dataset/lerobot.py
    _calculate_relative_stats_for_key.
    """
    stats: dict = {}
    action_to_state_key_map = action_to_state_key_map or {}

    for rel_action_key in relative_action_keys:
        if rel_action_key not in modality["action"]:
            log.warning("Relative action key '%s' not found in action modality, skipping", rel_action_key)
            continue
        rel_state_key = action_to_state_key_map.get(rel_action_key, rel_action_key)
        if rel_state_key not in modality["state"]:
            log.warning(
                "Relative action key '%s' has no matching state key '%s'. Skipping.",
                rel_action_key,
                rel_state_key,
            )
            continue

        action_meta = modality["action"][rel_action_key]
        state_meta = modality["state"][rel_state_key]

        all_relative = []
        for pp in tqdm(parquet_paths, desc=f"Relative stats [{rel_action_key}->{rel_state_key}]"):
            df = pd.read_parquet(pp)
            action_col = action_meta["original_key"]
            state_col = state_meta["original_key"]
            if action_col not in df.columns or state_col not in df.columns:
                continue

            action_data = np.stack(df[action_col].values).astype(np.float64)
            state_data = np.stack(df[state_col].values).astype(np.float64)
            if action_data.ndim == 1:
                action_data = action_data.reshape(-1, 1)
            if state_data.ndim == 1:
                state_data = state_data.reshape(-1, 1)

            a_start, a_end = action_meta["start"], action_meta["end"]
            s_start, s_end = state_meta["start"], state_meta["end"]

            action_slice = action_data[:, a_start:a_end]
            state_slice = state_data[:, s_start:s_end]

            traj_len = len(df)
            usable = traj_len - action_horizon
            for i in range(max(usable, 0)):
                ref_state = state_slice[i]
                chunk_end = min(i + action_horizon, traj_len)
                actions = action_slice[i:chunk_end]
                relative = actions - ref_state
                all_relative.extend(relative)

        if not all_relative:
            log.warning("No relative actions computed for '%s'", rel_action_key)
            continue

        data = np.array(all_relative)
        stats[rel_action_key] = {
            "max": np.max(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

    return stats


# ---------------------------------------------------------------------------
# Tasks & episodes
# ---------------------------------------------------------------------------

def build_tasks(
    parquet_paths: list[Path],
    task_key: str | None,
    task_index_to_text: dict[int, str] | None = None,
) -> list[dict]:
    """Build tasks.jsonl entries from the dataset."""
    task_index_to_text = task_index_to_text or {}
    if task_key is None:
        if task_index_to_text:
            return [
                {"task_index": idx, "task": text}
                for idx, text in sorted(task_index_to_text.items())
            ]
        return [{"task_index": 0, "task": ""}]

    task_set: dict[str, int] = {text: idx for idx, text in task_index_to_text.items()}
    for pp in tqdm(parquet_paths, desc="Extracting tasks"):
        df = pd.read_parquet(pp)
        if task_key not in df.columns:
            continue
        for val in df[task_key].unique():
            text = resolve_task_value(val, task_index_to_text)
            if text not in task_set:
                next_idx = max(task_set.values(), default=-1) + 1
                task_set[text] = next_idx

    if not task_set:
        return [{"task_index": 0, "task": ""}]

    return [{"task_index": idx, "task": text} for text, idx in sorted(task_set.items(), key=lambda x: x[1])]


def build_episodes(
    parquet_paths: list[Path],
    info: dict,
    task_key: str | None,
    tasks: list[dict],
    task_index_to_text: dict[int, str] | None = None,
) -> list[dict]:
    """Build episodes.jsonl entries."""
    task_index_to_text = task_index_to_text or {}
    task_text_to_idx = {t["task"]: t["task_index"] for t in tasks}
    episodes = []
    for pp in tqdm(parquet_paths, desc="Building episodes"):
        df = pd.read_parquet(pp)
        length = len(df)
        if "episode_index" in df.columns and len(df) > 0:
            ep_idx = int(df["episode_index"].iloc[0])
        else:
            ep_idx = episode_index_from_path(pp)
        if ep_idx < 0:
            ep_idx = len(episodes)

        ep_tasks: list[str] = []
        if task_key and task_key in df.columns:
            unique_tasks = df[task_key].unique()
            for t in unique_tasks:
                text = resolve_task_value(t, task_index_to_text)
                if text and text in task_text_to_idx:
                    ep_tasks.append(text)
        if not ep_tasks:
            ep_tasks = [""]

        episodes.append({
            "episode_index": ep_idx,
            "tasks": ep_tasks,
            "length": length,
        })

    return episodes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def probe_video_header(path: Path) -> dict:
    try:
        import av
    except Exception as exc:
        return {"error": f"PyAV unavailable: {exc}"}

    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return {"error": "no video stream"}
            return {
                "codec": stream.codec_context.name,
                "width": int(stream.codec_context.width),
                "height": int(stream.codec_context.height),
                "pix_fmt": (
                    str(stream.codec_context.format.name)
                    if stream.codec_context.format is not None
                    else ""
                ),
                "fps": float(stream.average_rate) if stream.average_rate is not None else 0.0,
            }
    except Exception as exc:
        return {"error": str(exc)}


def validate_dataset(dataset_path: Path, info: dict, modality: dict) -> list[str]:
    """Run basic validation and return a list of warnings."""
    warnings = []

    # Check required directories
    for subdir in ["data", "videos", "meta"]:
        if not (dataset_path / subdir).exists():
            warnings.append(f"Missing directory: {subdir}/")

    # Check at least one video key exists
    if not modality["video"]:
        warnings.append("No video features detected -- DreamZero requires at least one camera view")

    # Check state/action exist
    if not modality["state"]:
        warnings.append("No state modality keys defined")
    if not modality["action"]:
        warnings.append("No action modality keys defined")

    # Check total_episodes > 0
    if info.get("total_episodes", 0) == 0:
        warnings.append("total_episodes is 0 in info.json")

    # Check FPS
    if info.get("fps") is None:
        warnings.append("fps not set in info.json")

    parquet_paths = get_parquet_paths(dataset_path, info)
    if parquet_paths:
        info_total = int(info.get("total_episodes", 0))
        if info_total and info_total != len(parquet_paths):
            warnings.append(
                f"info.json total_episodes={info_total}, but found {len(parquet_paths)} parquet files"
            )
        episode_ids = [episode_index_from_path(p) for p in parquet_paths]
        for key, video_cfg in modality.get("video", {}).items():
            original_key = video_cfg.get("original_key", f"observation.images.{key}")
            feature_meta = info.get("features", {}).get(original_key, {})
            expected_shape = feature_meta.get("shape")
            expected_codec = feature_meta.get("video_info", {}).get("video.codec")
            expected_pix_fmt = feature_meta.get("video_info", {}).get("video.pix_fmt")
            expected_fps = feature_meta.get("video_info", {}).get("video.fps", info.get("fps"))
            missing = []
            probe_failed = []
            codec_mismatch = []
            shape_mismatch = []
            pix_fmt_mismatch = []
            fps_mismatch = []
            for p in parquet_paths:
                ep_idx = episode_index_from_path(p)
                if ep_idx < 0:
                    continue
                chunk_idx = ep_idx // int(info.get("chunks_size", 1000))
                video_rel = info.get(
                    "video_path",
                    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                ).format(episode_chunk=chunk_idx, episode_index=ep_idx, video_key=original_key)
                video_path = dataset_path / video_rel
                if not video_path.exists():
                    missing.append(ep_idx)
                    continue
                header = probe_video_header(video_path)
                if "error" in header:
                    probe_failed.append((ep_idx, header["error"]))
                    continue
                if expected_codec and header.get("codec") != expected_codec:
                    codec_mismatch.append((ep_idx, header.get("codec")))
                if expected_shape and [header.get("height"), header.get("width"), 3] != list(expected_shape):
                    shape_mismatch.append((ep_idx, [header.get("height"), header.get("width"), 3]))
                if expected_pix_fmt and header.get("pix_fmt") != expected_pix_fmt:
                    pix_fmt_mismatch.append((ep_idx, header.get("pix_fmt")))
                if expected_fps is not None and abs(float(header.get("fps", 0.0)) - float(expected_fps)) > 1e-3:
                    fps_mismatch.append((ep_idx, header.get("fps")))
            if missing:
                warnings.append(
                    f"video key '{key}' is missing {len(missing)} videos; first missing ids: {missing[:5]}"
                )
            if probe_failed:
                warnings.append(
                    f"video key '{key}' failed header probe for {len(probe_failed)} videos; "
                    f"first failures: {probe_failed[:3]}"
                )
            if codec_mismatch:
                warnings.append(
                    f"video key '{key}' codec mismatches metadata for {len(codec_mismatch)} videos; "
                    f"expected {expected_codec}, first mismatches: {codec_mismatch[:5]}"
                )
            if shape_mismatch:
                warnings.append(
                    f"video key '{key}' resolution mismatches metadata for {len(shape_mismatch)} videos; "
                    f"expected {expected_shape}, first mismatches: {shape_mismatch[:5]}"
                )
            if pix_fmt_mismatch:
                warnings.append(
                    f"video key '{key}' pix_fmt mismatches metadata for {len(pix_fmt_mismatch)} videos; "
                    f"expected {expected_pix_fmt}, first mismatches: {pix_fmt_mismatch[:5]}"
                )
            if fps_mismatch:
                warnings.append(
                    f"video key '{key}' fps mismatches metadata for {len(fps_mismatch)} videos; "
                    f"expected {expected_fps}, first mismatches: {fps_mismatch[:5]}"
                )
        if len(episode_ids) != len(set(episode_ids)):
            warnings.append("duplicate episode ids found in parquet filenames")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot v2 dataset to GEAR/DreamZero training format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the LeRobot v2 dataset")
    parser.add_argument("--output-path", type=str, default=None, help="Output path (default: modify in-place)")
    parser.add_argument(
        "--embodiment-tag", type=str, default="xdof",
        help=f"Embodiment tag (default: xdof). Valid: {', '.join(sorted(set(VALID_EMBODIMENT_TAGS)))}"
    )
    parser.add_argument(
        "--state-keys", type=str, default=None,
        help='JSON mapping of state sub-keys to [start, end] index ranges, '
             'e.g. \'{"joint_pos": [0, 6], "gripper_pos": [6, 7]}\''
    )
    parser.add_argument(
        "--action-keys", type=str, default=None,
        help='JSON mapping of action sub-keys to [start, end] index ranges'
    )
    parser.add_argument(
        "--relative-action-keys", type=str, nargs="*", default=None,
        help="Action sub-key names to compute relative stats for (e.g. joint_pos gripper_pos). "
             "By default each action key is matched to a state key with the same name; use "
             "--relative-action-state-map for explicit action->state matching."
    )
    parser.add_argument(
        "--relative-action-state-map", type=str, nargs="*", default=None,
        help="Optional explicit action->state key mapping for relative stats, entries as "
             "action_key=state_key (e.g. left_target_pose9d_rot6d=left_ee_pose9d_rot6d)."
    )
    parser.add_argument("--task-key", type=str, default=None, help="Column name for language annotations (auto-detected if not set)")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default: use dataset FPS from info.json)")
    parser.add_argument("--action-horizon", type=int, default=24, help="Action horizon for relative stats (default: 24)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing GEAR metadata files")

    args = parser.parse_args()
    relative_action_state_map = parse_relative_key_map(args.relative_action_state_map)

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        log.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)

    # Validate embodiment tag
    if args.embodiment_tag not in VALID_EMBODIMENT_TAGS:
        log.error(
            "Invalid embodiment tag '%s'. Valid tags:\n  %s",
            args.embodiment_tag,
            "\n  ".join(sorted(set(VALID_EMBODIMENT_TAGS))),
        )
        sys.exit(1)

    # Output path handling
    if args.output_path:
        output_path = Path(args.output_path).resolve()
        if output_path != dataset_path:
            log.info("Copying dataset to %s", output_path)
            if output_path.exists():
                if not args.force:
                    log.error("Output path already exists. Use --force to overwrite.")
                    sys.exit(1)
                shutil.rmtree(output_path)
            shutil.copytree(dataset_path, output_path)
            dataset_path = output_path
    else:
        output_path = dataset_path

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load info.json
    info = load_info(dataset_path)
    detected = detect_features(info)
    existing_modality = load_json_if_exists(output_path / "meta" / "modality.json")
    task_index_to_text, _ = load_existing_tasks(output_path)

    log.info("Dataset: %s", dataset_path.name)
    log.info("  Episodes: %d", info.get("total_episodes", 0))
    log.info("  FPS: %s", info.get("fps", "not set"))
    log.info("  State columns: %s", detected["state"])
    log.info("  Action columns: %s", detected["action"])
    log.info("  Video features: %d camera(s)", len(detected["video"]))
    log.info("  Annotation columns: %s", detected["annotation"])

    if args.fps is not None:
        info["fps"] = args.fps
        with open(output_path / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=4)
        log.info("  Overriding FPS to %s", args.fps)

    # Parse user-provided key mappings. Precedence:
    # explicit CLI > existing modality.json > collection_config layout > embodiment preset > generic full vector.
    explicit_state_mapping = parse_key_mapping(args.state_keys)
    explicit_action_mapping = parse_key_mapping(args.action_keys)
    state_mapping, state_mapping_source = choose_mapping(
        explicit=explicit_state_mapping,
        existing_modality=existing_modality,
        info=info,
        detected=detected,
        modality_type="state",
        embodiment_tag=args.embodiment_tag,
    )
    action_mapping, action_mapping_source = choose_mapping(
        explicit=explicit_action_mapping,
        existing_modality=existing_modality,
        info=info,
        detected=detected,
        modality_type="action",
        embodiment_tag=args.embodiment_tag,
    )
    log.info("  State mapping source: %s (%d keys)", state_mapping_source, len(state_mapping or {}))
    log.info("  Action mapping source: %s (%d keys)", action_mapping_source, len(action_mapping or {}))

    # Auto-detect task key if not provided
    task_key = args.task_key
    if task_key is None and detected["annotation"]:
        for candidate in ["annotation.task", "annotation.language.language_instruction"]:
            if candidate in detected["annotation"]:
                task_key = candidate
                break
        if task_key is None:
            task_key = detected["annotation"][0]
        log.info("  Auto-detected task key: %s", task_key)

    # 2. Build modality.json
    modality = build_modality_json(info, detected, state_mapping, action_mapping, task_key)

    modality_path = meta_dir / "modality.json"
    modality_override_requested = explicit_state_mapping is not None or explicit_action_mapping is not None
    if modality_path.exists() and not args.force and not modality_override_requested:
        log.info("  modality.json already exists, skipping (use --force to overwrite)")
    else:
        with open(modality_path, "w") as f:
            json.dump(modality, f, indent=4)
        log.info("  Wrote modality.json (%d state keys, %d action keys, %d video keys)",
                 len(modality["state"]), len(modality["action"]), len(modality["video"]))

    # 3. Write embodiment.json
    embodiment = {"robot_type": args.embodiment_tag, "embodiment_tag": args.embodiment_tag}
    embodiment_path = meta_dir / "embodiment.json"
    if embodiment_path.exists() and not args.force:
        log.info("  embodiment.json already exists, skipping")
    else:
        with open(embodiment_path, "w") as f:
            json.dump(embodiment, f, indent=4)
        log.info("  Wrote embodiment.json (tag=%s)", args.embodiment_tag)

    # 4. Get parquet file paths
    parquet_paths = get_parquet_paths(output_path, info)
    if not parquet_paths:
        log.error("No parquet files found. Check dataset structure.")
        sys.exit(1)
    log.info("  Found %d parquet files", len(parquet_paths))

    # 5. Compute stats.json
    stats_path = meta_dir / "stats.json"
    numeric_cols = detected["state"] + detected["action"]
    if "timestamp" in info.get("features", {}):
        numeric_cols.append("timestamp")

    if stats_path.exists() and not args.force:
        log.info("  stats.json already exists, skipping")
    else:
        log.info("  Computing dataset statistics...")
        stats = compute_stats_by_modality(parquet_paths, modality, numeric_cols)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        total_keys = len(stats.get('state', {})) + len(stats.get('action', {}))
        log.info("  Wrote stats.json (%d keys)", total_keys)

    # 6. Compute relative_stats_dreamzero.json
    rel_stats_path = meta_dir / "relative_stats_dreamzero.json"
    if args.relative_action_keys:
        if rel_stats_path.exists() and not args.force:
            log.info("  relative_stats_dreamzero.json already exists, skipping")
        else:
            log.info("  Computing relative action statistics for keys: %s", args.relative_action_keys)
            rel_stats = compute_relative_stats(
                parquet_paths, modality, args.relative_action_keys,
                action_to_state_key_map=relative_action_state_map,
                action_horizon=args.action_horizon,
            )
            if rel_stats:
                with open(rel_stats_path, "w") as f:
                    json.dump(rel_stats, f, indent=4)
                log.info("  Wrote relative_stats_dreamzero.json (%d keys)", len(rel_stats))
            else:
                log.warning("  No relative stats computed (check key names match between state and action)")
    else:
        log.info("  Skipping relative stats (no --relative-action-keys provided)")

    # 7. Build tasks.jsonl
    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.exists() and not args.force:
        log.info("  tasks.jsonl already exists, skipping")
    else:
        tasks = build_tasks(parquet_paths, task_key, task_index_to_text=task_index_to_text)
        with open(tasks_path, "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
        log.info("  Wrote tasks.jsonl (%d tasks)", len(tasks))

    # 8. Build episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    if episodes_path.exists() and not args.force:
        log.info("  episodes.jsonl already exists, skipping")
    else:
        tasks = []
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    tasks.append(json.loads(line.strip()))
        if not tasks:
            tasks = [{"task_index": 0, "task": ""}]
        episodes = build_episodes(
            parquet_paths,
            info,
            task_key,
            tasks,
            task_index_to_text=task_index_to_text,
        )
        with open(episodes_path, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        log.info("  Wrote episodes.jsonl (%d episodes)", len(episodes))

    # 9. Validation
    warnings = validate_dataset(output_path, info, modality)
    if warnings:
        log.warning("Validation warnings:")
        for w in warnings:
            log.warning("  - %s", w)
    else:
        log.info("Validation passed -- no warnings")

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Embodiment tag: {args.embodiment_tag}")
    print(f"  State keys: {list(modality['state'].keys())}")
    print(f"  Action keys: {list(modality['action'].keys())}")
    print(f"  Video keys: {list(modality['video'].keys())}")
    print(f"  Task key: {task_key or '(none)'}")
    if args.relative_action_keys:
        print(f"  Relative action keys: {args.relative_action_keys}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Create a YAML data config in groot/vla/configs/data/dreamzero/")
    print("  2. Add modality configs to base_48_wan_fine_aug_relative.yaml")
    print("  3. Create a training script in scripts/train/")
    print("  See docs/CUSTOM_EMBODIMENT_TRAINING.md for the full guide.")


if __name__ == "__main__":
    main()
