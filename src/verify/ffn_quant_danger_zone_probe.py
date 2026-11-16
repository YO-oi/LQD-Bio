from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset

try:
    from src.analysis.build_fp32_oer_oracle import (
        ORACLE_STORAGE_FORMAT as FP32_OER_STORAGE_FORMAT,
        iter_fp32_oer_oracle_shards,
        load_fp32_oer_oracle,
    )
    from src.analysis.quant_danger_zone_summary import (
        COMPARISON_SUMMARY_OUTPUT_FILENAME,
        CORRELATION_OUTPUT_FILENAME,
        DEFAULT_QUANT_MODE,
        SAMPLE_COLUMNS,
        SAMPLE_OUTPUT_FILENAME,
        SUMMARY_OUTPUT_FILENAME,
        build_summary_table,
    )
    from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from src.export.export_fp32_core_cache import (
        CORE_STORAGE_FORMAT as FP32_CORE_STORAGE_FORMAT,
        load_fp32_core_cache,
    )
    from src.metrics.activation_distribution_metrics import compute_basic_distribution_stats
    from src.metrics.metric_config import (
        DEFAULT_EPS,
        DEFAULT_SCALAR_OER_TOP_RATIO,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    )
    from src.metrics.quant_danger_zone_metrics import (
        compute_direct_integer_grid_danger_metrics,
        compute_projection_damage_metrics,
        compute_single_activation_danger_metrics,
    )
    from src.plots.quant_danger_zone_plotter import plot_all_quant_danger_zone_figures
    from src.quant.calibration import freeze_weight_qparams_from_tensor
    from src.quant.ffn_quant_injector import (
        make_single_ffn_quant_controller,
        normalize_quant_mode,
        quant_mode_uses_activation,
        quant_mode_uses_weight,
        replace_ffn_module_with_quant,
        restore_clean_model_from_checkpoint,
    )
    from src.utils.activation_hook import ActivationHookManager
    from src.utils.group_utils import ANALYSIS_GROUPS, group_protocol_metadata, read_canonical_bucket, tail_included
    from src.utils.layout_utils import (
        canonicalize_activation_btc,
        canonicalize_padding_mask_bt,
        layout_info_to_public_fields,
        require_valid_token_policy,
        validate_valid_token_policy,
    )
except ImportError:  # pragma: no cover - package-relative debugging fallback.
    from ..analysis.build_fp32_oer_oracle import (
        ORACLE_STORAGE_FORMAT as FP32_OER_STORAGE_FORMAT,
        iter_fp32_oer_oracle_shards,
        load_fp32_oer_oracle,
    )
    from ..analysis.quant_danger_zone_summary import (
        COMPARISON_SUMMARY_OUTPUT_FILENAME,
        CORRELATION_OUTPUT_FILENAME,
        DEFAULT_QUANT_MODE,
        SAMPLE_COLUMNS,
        SAMPLE_OUTPUT_FILENAME,
        SUMMARY_OUTPUT_FILENAME,
        build_summary_table,
    )
    from ..data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from ..export.export_fp32_core_cache import (
        CORE_STORAGE_FORMAT as FP32_CORE_STORAGE_FORMAT,
        load_fp32_core_cache,
    )
    from ..metrics.activation_distribution_metrics import compute_basic_distribution_stats
    from ..metrics.metric_config import (
        DEFAULT_EPS,
        DEFAULT_SCALAR_OER_TOP_RATIO,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    )
    from ..metrics.quant_danger_zone_metrics import (
        compute_direct_integer_grid_danger_metrics,
        compute_projection_damage_metrics,
        compute_single_activation_danger_metrics,
    )
    from ..plots.quant_danger_zone_plotter import plot_all_quant_danger_zone_figures
    from ..quant.calibration import freeze_weight_qparams_from_tensor
    from ..quant.ffn_quant_injector import (
        make_single_ffn_quant_controller,
        normalize_quant_mode,
        quant_mode_uses_activation,
        quant_mode_uses_weight,
        replace_ffn_module_with_quant,
        restore_clean_model_from_checkpoint,
    )
    from ..utils.activation_hook import ActivationHookManager
    from ..utils.group_utils import ANALYSIS_GROUPS, group_protocol_metadata, read_canonical_bucket, tail_included
    from ..utils.layout_utils import (
        canonicalize_activation_btc,
        canonicalize_padding_mask_bt,
        layout_info_to_public_fields,
        require_valid_token_policy,
        validate_valid_token_policy,
    )


DEFAULT_FFN_MODULE_NAMES = ("fc1", "fc2")
DEFAULT_SELECTED_FFN_TARGET_MODULES = (
    (1, "fc1"),
    (3, "fc1"),
    (4, "fc1"),
    (9, "fc1"),
    (0, "fc2"),
    (3, "fc2"),
    (9, "fc2"),
    (11, "fc2"),
)
TARGET_MODULES_ALL_SENTINELS = {"all", "*", "all_encoder_layers_fc1_fc2"}
ACTIVATION_POSITIONS = ("input_activation", "output_activation")
DANGER_ZONE_POSITIONS = ("input_activation", "linear_output_projection")
DANGER_ZONE_QUANT_MODES = ("A4-only", "W4A4")
OPTIONAL_REFERENCE_MODES = ("W4-only",)
RESULT_DIR_NAME = "quant_danger_zone"
MANIFEST_FILENAME = "ffn_quant_danger_zone_manifest.json"
TOPK_INDICES_FILENAME = "quant_danger_zone_topk_indices.pt"
PROTOCOL_VERSION = "ffn_quant_danger_zone.v3"
QPARAMS_DIRNAME = "qparams"
SHARD_DIRNAME = "shards"
TOPK_COMPACT_DIRNAME = "topk_compact"
DONE_DIRNAME = "done"
SUMMARY_PARTIAL_DIRNAME = "summary_partial"
FP32_REFERENCE_CORE_FILENAME = "core_eval.pt"
FP32_REFERENCE_OER_ORACLE_FILENAME = "oer_oracle_eval.pt"
FP32_CORE_CACHE_TYPE = "fp32_core_cache"
FP32_CORE_STAGE = "quant_injection_pre_fp32_reference"
FP32_OER_CACHE_TYPE = "fp32_oer_oracle"
FP32_OER_STAGE = "quant_injection_pre_fp32_oer_oracle"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}


def _quant_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "QUANT", {})
    return value if isinstance(value, Mapping) else {}


def _metrics_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "METRICS", {})
    return value if isinstance(value, Mapping) else {}


def _probe_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FFN_QUANT_DANGER_ZONE_PROBE", {})
    return value if isinstance(value, Mapping) else {}


def _legacy_probe_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "QUANT_DANGER_ZONE", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    probe = _probe_cfg(cfg)
    if key in probe:
        return probe[key]
    legacy = _legacy_probe_cfg(cfg)
    if key in legacy:
        return legacy[key]
    return default


def _dataset_name(cfg: Any) -> str:
    dataset_cfg = _dataset_cfg(cfg)
    for key in ("NAME", "name", "DATASET_NAME", "dataset_name"):
        value = _nested_get(dataset_cfg, key, None)
        if value:
            return str(value)
    active_dataset = _cfg_get(cfg, "ACTIVE_DATASET", None)
    if active_dataset:
        return str(active_dataset)
    dataset_root = _nested_get(dataset_cfg, "ROOT", None)
    if dataset_root is not None:
        return Path(dataset_root).name
    current_result_dir = _cfg_get(cfg, "CURRENT_RESULT_DIR", None)
    if current_result_dir is not None:
        return Path(current_result_dir).name
    raise AttributeError("Cannot determine dataset name: set ACTIVE_DATASET or CURRENT_DATASET['ROOT'].")


def _result_dir(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / RESULT_DIR_NAME


def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", _cfg_get(cfg, "DEVICE", None))
    if configured is None:
        raise AttributeError("ffn_quant_danger_zone_probe requires DEVICE from config.py.")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py DEVICE={configured!r}, but CUDA is not available.")
    return device


def _cache_device(cfg: Any) -> torch.device | str | None:
    configured = _setting_get(cfg, "CACHE_DEVICE", "cpu")
    if configured is None:
        return None
    text = str(configured).strip().lower()
    if text in {"", "none", "model", "same"}:
        return None
    return torch.device(str(configured))


def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


def _batch_size(cfg: Any) -> int:
    return int(_setting_get(cfg, "BATCH_SIZE", 4))


def _num_workers(cfg: Any) -> int:
    return int(_setting_get(cfg, "NUM_WORKERS", 0))


def _pin_memory(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "PIN_MEMORY", _device(cfg).type == "cuda"))


def _percentile(cfg: Any) -> float:
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    if not isinstance(activation_cfg, Mapping):
        activation_cfg = {}
    return float(_setting_get(cfg, "PERCENTILE", activation_cfg.get("PERCENTILE", 0.999)))


def _topk_ratio(cfg: Any) -> float:
    return float(
        _setting_get(
            cfg,
            "TOPK_RATIO",
            _metrics_cfg(cfg).get(
                "SCALAR_OER_TOP_RATIO",
                _cfg_get(cfg, "OER_SCALAR_TOP_RATIO", DEFAULT_SCALAR_OER_TOP_RATIO),
            ),
        )
    )


def _qrange(cfg: Any) -> tuple[int, int]:
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    if not isinstance(activation_cfg, Mapping):
        activation_cfg = {}
    qmin = int(_setting_get(cfg, "QMIN", activation_cfg.get("QMIN", DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN)))
    qmax = int(_setting_get(cfg, "QMAX", activation_cfg.get("QMAX", DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX)))
    if qmin >= qmax:
        raise ValueError(f"qmin 必须小于 qmax，当前为 qmin={qmin}, qmax={qmax}")
    return qmin, qmax


def _make_audio_config(cfg: Any) -> AudioInputConfig:
    audio = _cfg_get(cfg, "AUDIO", {})
    if not isinstance(audio, Mapping):
        audio = {}
    return AudioInputConfig(
        sample_rate=int(audio.get("TARGET_SR", 16000)),
        target_seconds=float(audio.get("TARGET_SECONDS", 10.0)),
        normalization=str(audio.get("NORMALIZATION", "peak")),
        eps=float(audio.get("EPS", 1e-8)),
        deterministic_crop=str(audio.get("CROP_MODE", "center")),  # type: ignore[arg-type]
        fixed_crop_start=int(audio.get("FIXED_CROP_START", 0)),
    )


def _probe_csv(cfg: Any) -> Path:
    dataset_cfg = _dataset_cfg(cfg)
    value = _setting_get(cfg, "EVAL_CSV", None)
    if value is not None:
        return Path(value)
    value = _nested_get(dataset_cfg, "EVAL_CSV", None)
    if value is not None:
        return Path(value)
    raise AttributeError("Cannot find eval CSV: set EVAL_CSV or CURRENT_DATASET['EVAL_CSV'].")


def _optional_probe_csv(cfg: Any) -> Path | None:
    try:
        return _probe_csv(cfg)
    except AttributeError:
        return None


def _build_eval_loader(cfg: Any) -> DataLoader:
    override = _cfg_get(cfg, "build_quant_danger_zone_probe_loader", None)
    if callable(override):
        return override(cfg)

    dataset: Any = AudioInputDataset(
        _probe_csv(cfg),
        mode=str(_setting_get(cfg, "MODE", "eval")),
        dataset_root=_nested_get(_dataset_cfg(cfg), "ROOT", None),
        project_root=_cfg_get(cfg, "BASE_DIR", None),
        audio_config=_make_audio_config(cfg),
        global_seed=_seed(cfg),
        epoch=0,
        return_dict=True,
        require_label_id=True,
        include_audio_stats=False,
    )
    max_samples = _setting_get(cfg, "MAX_SAMPLES", None)
    if max_samples is not None:
        limit = min(int(max_samples), len(dataset))
        if limit <= 0:
            raise ValueError(f"MAX_SAMPLES must be positive, got {max_samples}")
        dataset = Subset(dataset, range(limit))

    return DataLoader(
        dataset,
        batch_size=_batch_size(cfg),
        shuffle=False,
        num_workers=_num_workers(cfg),
        pin_memory=_pin_memory(cfg),
        collate_fn=partial(audio_input_collate_fn, include_audio_stats=False),
    )


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _extract_tensor_and_mask(output: Any) -> tuple[Tensor, Tensor | None]:
    if isinstance(output, Tensor):
        return output, None
    if isinstance(output, Mapping):
        tensor = None
        for key in ("embedding", "embeddings", "features", "feature", "x", "logits", "lprobs"):
            value = output.get(key)
            if isinstance(value, Tensor):
                tensor = value
                break
        if tensor is None:
            raise TypeError("模型输出 dict 中没有可用的 Tensor 字段")
        mask = output.get("padding_mask")
        return tensor, mask if isinstance(mask, Tensor) else None
    if isinstance(output, (tuple, list)) and output:
        tensor = output[0]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"模型输出序列的第一个元素必须是 Tensor，当前类型为 {type(tensor)!r}")
        mask = output[1] if len(output) > 1 and isinstance(output[1], Tensor) else None
        return tensor, mask
    raise TypeError(f"不支持的模型输出类型：{type(output)!r}")


def _forward_model(model: nn.Module, batch: Mapping[str, Any], cfg: Any) -> tuple[Tensor, Tensor | None]:
    forward_fn = _cfg_get(cfg, "ffn_quant_danger_zone_forward", None)
    if callable(forward_fn):
        return _extract_tensor_and_mask(forward_fn(model, batch, cfg))
    fallback_forward_fn = _cfg_get(cfg, "ffn_activation_distribution_forward", None)
    if callable(fallback_forward_fn):
        return _extract_tensor_and_mask(fallback_forward_fn(model, batch, cfg))

    waveform = batch["waveform"]
    padding_mask = batch.get("padding_mask")
    if hasattr(model, "extract_features"):
        return _extract_tensor_and_mask(model.extract_features(waveform, padding_mask=padding_mask))

    try:
        return _extract_tensor_and_mask(model(waveform, padding_mask=padding_mask))
    except TypeError:
        return _extract_tensor_and_mask(model(waveform))


def _autocast_context(cfg: Any, device: torch.device) -> Any:
    use_amp = bool(_setting_get(cfg, "USE_AMP", False))
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def _sequence_to_list(value: Any, *, batch_size: int, field_name: str) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value] * int(batch_size)
    if len(values) != int(batch_size):
        raise ValueError(f"{field_name} count must equal batch size, got {len(values)} vs {batch_size}")
    return values


def _first_existing_column(df: Any, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _metadata_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() not in {"", "nan", "none", "null", "<na>"}


def _parse_bool_metadata(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and int(value) in {0, 1}:
        return bool(int(value))
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer() and int(value) in {0, 1}:
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{context}: tail_included must be bool-like, got {value!r}")


def _sample_id_from_metadata_row(row: Mapping[str, Any]) -> str | None:
    for key in ("sample_id", "audio_id", "file_id"):
        value = row.get(key)
        if _metadata_value_present(value):
            return str(value).strip()
    return None


def _full_dataset_csv(cfg: Any) -> Path | None:
    dataset_cfg = _dataset_cfg(cfg)
    for key in ("FULL_CSV", "FULL_DATASET_CSV"):
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
        value = _nested_get(dataset_cfg, key, None)
        if value is not None:
            return Path(value)
    return None


def _metadata_csv_paths(cfg: Any) -> list[Path]:
    paths: list[Path] = []
    probe_csv = _optional_probe_csv(cfg)
    if probe_csv is not None:
        paths.append(probe_csv)
    full_csv = _full_dataset_csv(cfg)
    if full_csv is not None:
        paths.append(full_csv)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(Path(path))
        if key not in seen:
            seen.add(key)
            deduped.append(Path(path))
    return deduped


def _load_sample_metadata_lookup(cfg: Any) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for csv_path in _metadata_csv_paths(cfg):
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader, start=2):
                sample_id = _sample_id_from_metadata_row(row)
                if sample_id is None:
                    continue
                has_bucket = _metadata_value_present(row.get("bucket"))
                has_shot_group = _metadata_value_present(row.get("shot_group"))
                has_train_label_n = _metadata_value_present(row.get("train_label_n"))
                has_tail_flag = _metadata_value_present(row.get("tail_included"))
                if not (has_bucket or has_shot_group or has_train_label_n or has_tail_flag):
                    continue
                if not has_bucket:
                    raise ValueError(f"{csv_path}:{row_idx}: sample-level bucket is required; shot_group fallback is forbidden")
                result = read_canonical_bucket(
                    bucket=[row.get("bucket")],
                    shot_group=[row.get("shot_group")] if has_shot_group else None,
                    train_label_n=[row.get("train_label_n")] if has_train_label_n else None,
                    context=f"{csv_path}:{row_idx}",
                    allow_shot_group_fallback=False,
                )
                bucket = result.bucket[0]
                computed_tail = bool(tail_included([bucket])[0])
                if has_tail_flag:
                    observed_tail = _parse_bool_metadata(row.get("tail_included"), context=f"{csv_path}:{row_idx}")
                    if observed_tail != computed_tail:
                        raise ValueError(f"{csv_path}:{row_idx}: tail_included does not match bucket={bucket!r}")
                previous = lookup.get(sample_id)
                if previous is not None and previous["bucket"] != bucket:
                    raise ValueError(
                        f"Conflicting bucket metadata for sample_id={sample_id!r}: "
                        f"{previous['bucket']!r} vs {bucket!r}"
                    )
                item: dict[str, Any] = {
                    "bucket": bucket,
                    "shot_group": bucket,
                    "tail_included": computed_tail,
                }
                for label_key in ("label_id", "label", "label_name"):
                    if _metadata_value_present(row.get(label_key)):
                        item["label"] = row.get(label_key)
                        break
                for audio_key in ("audio_path", "path", "absolute_path", "filepath", "rel_path"):
                    if _metadata_value_present(row.get(audio_key)):
                        item["audio_path"] = row.get(audio_key)
                        break
                lookup[sample_id] = item
    return lookup


def _batch_metadata(batch: Mapping[str, Any], metadata_lookup: Mapping[str, Mapping[str, Any]], cfg: Any) -> dict[str, list[Any]]:
    del cfg
    sample_ids = [str(item) for item in batch["sample_id"]]
    batch_size = len(sample_ids)

    labels = _sequence_to_list(batch.get("label_id"), batch_size=batch_size, field_name="label_id")
    if labels is None:
        labels = [metadata_lookup.get(sample_id, {}).get("label", None) for sample_id in sample_ids]
    labels = [None if item is None else item for item in labels]

    audio_paths = _sequence_to_list(batch.get("audio_path"), batch_size=batch_size, field_name="audio_path")
    if audio_paths is None:
        audio_paths = [metadata_lookup.get(sample_id, {}).get("audio_path", "") for sample_id in sample_ids]
    audio_paths = ["" if item is None else str(item) for item in audio_paths]

    bucket_values = _sequence_to_list(batch.get("bucket"), batch_size=batch_size, field_name="bucket")
    shot_group_values = _sequence_to_list(batch.get("shot_group"), batch_size=batch_size, field_name="shot_group")
    train_label_n_values = _sequence_to_list(batch.get("train_label_n"), batch_size=batch_size, field_name="train_label_n")
    tail_flag_values = _sequence_to_list(batch.get("tail_included"), batch_size=batch_size, field_name="tail_included")

    if bucket_values is None:
        missing = [sample_id for sample_id in sample_ids if sample_id not in metadata_lookup]
        if missing:
            raise ValueError(
                "Danger Zone probe requires sample-level bucket metadata; "
                f"missing sample_id values={missing[:8]}"
            )
        bucket_values = [metadata_lookup[sample_id]["bucket"] for sample_id in sample_ids]
        if shot_group_values is None:
            shot_group_values = [metadata_lookup[sample_id].get("shot_group", metadata_lookup[sample_id]["bucket"]) for sample_id in sample_ids]
        if tail_flag_values is None:
            tail_flag_values = [metadata_lookup[sample_id].get("tail_included") for sample_id in sample_ids]

    result = read_canonical_bucket(
        bucket=bucket_values,
        shot_group=shot_group_values,
        train_label_n=train_label_n_values,
        context="quant_danger_zone batch",
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in result.bucket]
    computed_tail_flags = [bool(item) for item in tail_included(buckets)]
    if tail_flag_values is not None:
        observed_tail_flags = [
            _parse_bool_metadata(value, context=f"quant_danger_zone batch idx={idx}")
            for idx, value in enumerate(tail_flag_values)
        ]
        if observed_tail_flags != computed_tail_flags:
            raise ValueError("quant_danger_zone batch: tail_included must match bucket-derived tail membership")

    return {
        "sample_id": sample_ids,
        "label": labels,
        "bucket": buckets,
        "shot_group": list(buckets),
        "tail_included": computed_tail_flags,
        "audio_path": audio_paths,
    }

def _activation_to_btc(
    activation: Tensor,
    batch_size: int,
    target_name: str,
    *,
    padding_mask: Tensor | None = None,
    raw_layout: str | None = None,
) -> Tensor:
    """Strict wrapper around the shared BTC layout adapter."""

    return canonicalize_activation_btc(
        activation.detach().float(),
        raw_layout=raw_layout,
        padding_mask=padding_mask,
        batch_size=int(batch_size),
        context=target_name,
    ).tensor


def _activation_to_btc_with_layout(
    activation: Tensor,
    *,
    batch_size: int,
    target_name: str,
    padding_mask: Tensor | None,
    raw_layout: str | None = None,
):
    return canonicalize_activation_btc(
        activation.detach().float(),
        raw_layout=raw_layout,
        padding_mask=padding_mask,
        batch_size=int(batch_size),
        context=target_name,
    )


def _padding_mask_true_means_padding(cfg: Any) -> bool:
    value = _nested_get(_cfg_get(cfg, "AUDIO", {}), "PADDING_MASK_TRUE_MEANS", "padding")
    text = str(value).strip().lower()
    if text in {"padding", "pad", "masked", "invalid", "padded"}:
        return True
    if text in {"valid", "keep", "non_padding", "non-padding"}:
        return False
    raise ValueError(f"AUDIO.PADDING_MASK_TRUE_MEANS must be 'padding' or 'valid', got {value!r}")


def _valid_token_policy(cfg: Any) -> str:
    return validate_valid_token_policy(_setting_get(cfg, "VALID_TOKEN_POLICY", "require_mask"))


def _hidden_valid_token_mask(forward_mask: Tensor | None, *, batch_size: int, token_count: int, cfg: Any, context: str) -> Tensor:
    policy = _valid_token_policy(cfg)
    if forward_mask is None:
        if policy == "explicit_no_padding":
            return torch.ones((int(batch_size), int(token_count)), dtype=torch.bool)
        require_valid_token_policy(padding_mask=None, valid_token_policy=policy, context=context)
    mask = canonicalize_padding_mask_bt(
        forward_mask.detach(),
        batch_size=int(batch_size),
        token_count=int(token_count),
        context=context,
    )
    valid = ~mask if _padding_mask_true_means_padding(cfg) else mask
    if not bool(valid.any().item()):
        raise ValueError(f"{context}: aligned valid-token mask contains no valid tokens")
    return valid.contiguous()


def _get_encoder_layers(model: nn.Module) -> nn.ModuleList:
    if not hasattr(model, "encoder"):
        raise AttributeError("model is missing encoder; cannot enumerate BEATs Transformer FFN modules.")
    encoder = model.encoder
    if not hasattr(encoder, "layers"):
        raise AttributeError("model.encoder is missing layers; cannot enumerate BEATs Transformer FFN modules.")
    layers = encoder.layers
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"model.encoder.layers must be nn.ModuleList, got {type(layers)!r}")
    return layers


def _all_ffn_target_modules(model: nn.Module) -> list[tuple[int, str]]:
    layers = _get_encoder_layers(model)
    targets: list[tuple[int, str]] = []
    for layer_idx, layer in enumerate(layers):
        for module_name in DEFAULT_FFN_MODULE_NAMES:
            if not hasattr(layer, module_name):
                raise AttributeError(f"encoder.layers.{layer_idx}.{module_name} 不存在，无法用于 FFN danger probe")
            targets.append((int(layer_idx), module_name))
    if not targets:
        raise ValueError("model.encoder.layers is empty; cannot enumerate FFN fc1/fc2 modules.")
    return targets


def _parse_target_module_string(value: str) -> tuple[int, str]:
    text = str(value).strip()
    if not text:
        raise ValueError("target module string cannot be empty")
    if text in TARGET_MODULES_ALL_SENTINELS:
        raise ValueError("all-target sentinel is only valid for the full TARGET_MODULES value")

    parts = [part for part in text.split(".") if part]
    if len(parts) == 2:
        layer_text, module_name = parts
        if layer_text.startswith("L") and layer_text[1:].isdigit():
            return int(layer_text[1:]), str(module_name)
        if layer_text.isdigit():
            return int(layer_text), str(module_name)
    if len(parts) >= 4 and parts[-2].isdigit():
        return int(parts[-2]), str(parts[-1])
    raise ValueError(f"target module must look like 'L3.fc1' or (3, 'fc1'), got {value!r}")


def _parse_target_module_entry(item: Any) -> tuple[int, str]:
    if isinstance(item, str):
        return _parse_target_module_string(item)
    if isinstance(item, Mapping):
        layer_value = item.get("layer_idx", item.get("layer_index", item.get("layer")))
        module_value = item.get("module_name", item.get("module", item.get("name")))
        if layer_value is None or module_value is None:
            raise ValueError(f"target module mapping must include layer_idx and module_name, got {item!r}")
        return int(layer_value), str(module_value)
    try:
        layer_idx, module_name = item
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target_modules entries must be 'Lx.fc1/fc2', mapping, or (layer_idx, module_name), got {item!r}") from exc
    return int(layer_idx), str(module_name)


def _normalize_target_modules(target_modules: Sequence[Any]) -> list[tuple[int, str]]:
    if isinstance(target_modules, str):
        entries: Sequence[Any] = [part.strip() for part in target_modules.split(",") if part.strip()]
    else:
        entries = target_modules

    normalized: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for item in entries:
        layer_idx, module_name = _parse_target_module_entry(item)
        if int(layer_idx) < 0:
            raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")
        module_name = str(module_name)
        if module_name not in set(DEFAULT_FFN_MODULE_NAMES):
            raise ValueError(f"module_name must be fc1 or fc2, got {module_name!r}")
        target = (int(layer_idx), module_name)
        if target in seen:
            raise ValueError(f"duplicate target module: L{target[0]}.{target[1]}")
        seen.add(target)
        normalized.append(target)
    if not normalized:
        raise ValueError("target_modules cannot be empty")
    return normalized


def _target_modules_value_is_all(value: Any) -> bool:
    return isinstance(value, str) and str(value).strip() in TARGET_MODULES_ALL_SENTINELS


def _configured_target_modules(cfg: Any) -> Any:
    stage_value = _setting_get(cfg, "TARGET_MODULES", None)
    if stage_value is not None:
        return stage_value
    return _cfg_get(cfg, "FFN_SELECTED_TARGET_MODULES", None)


def _validate_target_modules_exist(model: nn.Module, target_modules: Sequence[tuple[int, str]]) -> None:
    layers = _get_encoder_layers(model)
    missing: list[str] = []
    non_module: list[str] = []
    for layer_idx, module_name in target_modules:
        if int(layer_idx) >= len(layers):
            missing.append(f"L{int(layer_idx)}.{module_name} (model has {len(layers)} layers)")
            continue
        layer = layers[int(layer_idx)]
        if not hasattr(layer, str(module_name)):
            missing.append(f"L{int(layer_idx)}.{module_name}")
            continue
        if not isinstance(getattr(layer, str(module_name)), nn.Module):
            non_module.append(f"L{int(layer_idx)}.{module_name}")
    if missing:
        raise AttributeError("Configured quant danger-zone target modules do not exist: " + ", ".join(missing))
    if non_module:
        raise TypeError("Configured quant danger-zone targets are not nn.Module instances: " + ", ".join(non_module))


def _resolve_target_modules(
    model: nn.Module,
    target_modules: Sequence[Any] | None,
    cfg: Any,
) -> tuple[list[tuple[int, str]], str]:
    if target_modules is not None:
        resolved = _all_ffn_target_modules(model) if _target_modules_value_is_all(target_modules) else _normalize_target_modules(target_modules)
        source = "explicit_argument"
    else:
        configured = _configured_target_modules(cfg)
        if configured is not None:
            resolved = _all_ffn_target_modules(model) if _target_modules_value_is_all(configured) else _normalize_target_modules(configured)
            source = "config_target_modules"
        else:
            resolved = _normalize_target_modules(DEFAULT_SELECTED_FFN_TARGET_MODULES)
            source = "default_selected_target_modules"
    _validate_target_modules_exist(model, resolved)
    return resolved, source


def _target_modules(model: nn.Module, target_modules: Sequence[Any] | None, cfg: Any) -> list[tuple[int, str]]:
    return _resolve_target_modules(model, target_modules, cfg)[0]


def _format_target_modules(target_modules: Sequence[tuple[int, str]]) -> str:
    return ", ".join(f"L{int(layer_idx)}.{module_name}" for layer_idx, module_name in target_modules)


def _tensor_to_float(value: Any) -> float:
    if isinstance(value, Tensor):
        detached = value.detach().float().cpu()
        if detached.numel() != 1:
            raise ValueError(f"expected a scalar Tensor, got numel={detached.numel()}")
        return float(detached.item())
    return float(value)


def _safe_ratio(numerator: Any, denominator: Any, *, eps: float = DEFAULT_EPS) -> float:
    num = _tensor_to_float(numerator)
    den = _tensor_to_float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= float(eps):
        return float("nan")
    return float(num / den)


def _scale_summary(scale: Any, granularity: str) -> tuple[float, float, float]:
    scale_t = torch.as_tensor(scale).detach().float().cpu().reshape(-1)
    if int(scale_t.numel()) <= 0:
        raise ValueError("scale 不能为空")
    if not torch.isfinite(scale_t).all().item():
        raise ValueError("scale values must all be finite")
    if str(granularity) == "per_tensor":
        return float(scale_t.mean().item()), float("nan"), float("nan")
    return (
        float(scale_t.mean().item()),
        float(torch.quantile(scale_t, 0.5).item()),
        float(torch.quantile(scale_t, 0.95).item()),
    )


def _danger_rows_for_activation(
    activation: Tensor,
    *,
    sample_id: str,
    label: Any,
    bucket: str,
    shot_group: str,
    tail_flag: bool,
    layer_idx: int,
    module_name: str,
    tensor_position: str,
    cfg: Any,
    valid_token_mask: Tensor | None = None,
    topk_index_records: list[dict[str, Any]] | None = None,
    layout_info: Mapping[str, Any] | None = None,
    valid_token_policy: str = "require_mask",
) -> list[dict[str, Any]]:
    bucket_result = read_canonical_bucket(
        bucket=[bucket],
        shot_group=[shot_group] if _metadata_value_present(shot_group) else None,
        context=f"quant_danger_zone row sample_id={sample_id!r}",
        allow_shot_group_fallback=False,
    )
    bucket = bucket_result.bucket[0]
    shot_group = bucket
    computed_tail = bool(tail_included([bucket])[0])
    if bool(tail_flag) != computed_tail:
        raise ValueError(f"sample_id={sample_id!r}: tail_included does not match bucket={bucket!r}")

    qmin, qmax = _qrange(cfg)
    percentile = _percentile(cfg)
    topk_ratio = _topk_ratio(cfg)
    public_layout_info = dict(layout_info or {})
    activation_shape_for_row = public_layout_info.get("canonical_shape") or tuple(int(item) for item in activation.shape)
    metrics = compute_single_activation_danger_metrics(
        activation,
        percentile=percentile,
        topk_ratio=topk_ratio,
        qmin=qmin,
        qmax=qmax,
        zero_point=0.0,
        valid_token_mask=valid_token_mask,
        valid_token_policy=valid_token_policy,
        return_intermediates=True,
    )
    stats_activation = activation
    if valid_token_mask is not None:
        valid_stats_mask = valid_token_mask.detach().bool().to(device=activation.device)
        if valid_stats_mask.ndim == 1:
            valid_stats_mask = valid_stats_mask.reshape(1, -1)
        if valid_stats_mask.ndim != 2 or tuple(valid_stats_mask.shape) != tuple(activation.shape[:2]):
            raise ValueError(
                f"sample_id={sample_id!r}: valid_token_mask must align with per-sample activation [1,T,C], "
                f"got mask_shape={tuple(valid_stats_mask.shape)}, activation_shape={tuple(activation.shape)}"
            )
        stats_activation = activation[valid_stats_mask]
    basic = compute_basic_distribution_stats(stats_activation)
    p99 = basic["p99"]
    activation_shape = tuple(int(item) for item in metrics["activation_shape"])
    if len(activation_shape) != 3:
        raise ValueError(f"Danger-zone activation_shape must be [B,T,C], got {activation_shape}")

    rows: list[dict[str, Any]] = []
    key_map = {
        "per_tensor": {
            "scale": "per_tensor_scale",
            "upper": "clip_high_rate_per_tensor",
            "lower": "clip_low_rate_per_tensor",
            "total": "clip_rate_per_tensor",
            "boundary_hit": "boundary_hit_rate_per_tensor",
            "strict_overflow": "strict_overflow_rate_per_tensor",
            "saturation_boundary": "saturation_boundary_hit_rate_per_tensor",
            "saturation_strict_overflow": "saturation_strict_overflow_rate_per_tensor",
            "saturation_boundary_energy": "saturation_boundary_energy_share_per_tensor",
            "overflow_energy_proxy": "overflow_energy_proxy_per_tensor",
            "exact_zero": "exact_zero_rate_per_tensor",
            "near_zero_pm1": "near_zero_pm1_rate_per_tensor",
            "near_zero": "near_zero_rate_per_tensor",
            "topk_token_exact_zero": "topk_token_exact_zero_rate_per_tensor",
            "topk_token_near_zero_pm1": "topk_token_near_zero_pm1_rate_per_tensor",
            "topk_token_boundary_hit": "topk_token_boundary_hit_rate_per_tensor",
            "topk_token_strict_overflow": "topk_token_strict_overflow_rate_per_tensor",
            "high_energy_scalar_exact_zero": "high_energy_scalar_exact_zero_rate_per_tensor",
            "high_energy_scalar_near_zero_pm1": "high_energy_scalar_near_zero_pm1_rate_per_tensor",
            "high_energy_scalar_boundary_hit": "high_energy_scalar_boundary_hit_rate_per_tensor",
            "high_energy_scalar_strict_overflow": "high_energy_scalar_strict_overflow_rate_per_tensor",
            "topk_clip": "topk_token_clip_rate_per_tensor",
            "topk_near_zero": "topk_token_near_zero_rate_per_tensor",
            "saturation": "saturation_clip_rate_per_tensor",
            "saturation_overflow": "saturation_overflow_rate_per_tensor",
            "near_zero_swallow": "key_token_near_zero_swallow_rate_per_tensor",
            "high_energy_near_zero": "high_energy_near_zero_rate_per_tensor",
            "high_energy_scalar_clip": "high_energy_scalar_clip_rate_per_tensor",
            "oer_scalar_clip": "oer_scalar_clip_rate_per_tensor",
            "key_token_near_zero": "key_token_near_zero_swallow_rate_per_tensor",
            "saturation_energy": "saturation_clip_energy_share_per_tensor",
            "near_zero_energy": "near_zero_swallow_energy_share_per_tensor",
            "near_zero_loss": "near_zero_swallow_loss_share_per_tensor",
            "key_token_energy": "key_token_clip_energy_share_per_tensor",
            "key_token_near_zero_energy": "key_token_near_zero_energy_share_per_tensor",
            "key_token_near_zero_loss": "key_token_near_zero_loss_share_per_tensor",
            "oer_retention": "oer_retention_per_tensor",
            "oer_loss_total": "oer_loss_total_per_tensor",
            "oer_loss": "oer_loss_per_tensor",
        },
        "per_channel": {
            "scale": "per_channel_scale",
            "upper": "clip_high_rate_per_channel",
            "lower": "clip_low_rate_per_channel",
            "total": "clip_rate_per_channel",
            "boundary_hit": "boundary_hit_rate_per_channel",
            "strict_overflow": "strict_overflow_rate_per_channel",
            "saturation_boundary": "saturation_boundary_hit_rate_per_channel",
            "saturation_strict_overflow": "saturation_strict_overflow_rate_per_channel",
            "saturation_boundary_energy": "saturation_boundary_energy_share_per_channel",
            "overflow_energy_proxy": "overflow_energy_proxy_per_channel",
            "exact_zero": "exact_zero_rate_per_channel",
            "near_zero_pm1": "near_zero_pm1_rate_per_channel",
            "near_zero": "near_zero_rate_per_channel",
            "topk_token_exact_zero": "topk_token_exact_zero_rate_per_channel",
            "topk_token_near_zero_pm1": "topk_token_near_zero_pm1_rate_per_channel",
            "topk_token_boundary_hit": "topk_token_boundary_hit_rate_per_channel",
            "topk_token_strict_overflow": "topk_token_strict_overflow_rate_per_channel",
            "high_energy_scalar_exact_zero": "high_energy_scalar_exact_zero_rate_per_channel",
            "high_energy_scalar_near_zero_pm1": "high_energy_scalar_near_zero_pm1_rate_per_channel",
            "high_energy_scalar_boundary_hit": "high_energy_scalar_boundary_hit_rate_per_channel",
            "high_energy_scalar_strict_overflow": "high_energy_scalar_strict_overflow_rate_per_channel",
            "topk_clip": "topk_token_clip_rate_per_channel",
            "topk_near_zero": "topk_token_near_zero_rate_per_channel",
            "saturation": "saturation_clip_rate_per_channel",
            "saturation_overflow": "saturation_overflow_rate_per_channel",
            "near_zero_swallow": "key_token_near_zero_swallow_rate_per_channel",
            "high_energy_near_zero": "high_energy_near_zero_rate_per_channel",
            "high_energy_scalar_clip": "high_energy_scalar_clip_rate_per_channel",
            "oer_scalar_clip": "oer_scalar_clip_rate_per_channel",
            "key_token_near_zero": "key_token_near_zero_swallow_rate_per_channel",
            "saturation_energy": "saturation_clip_energy_share_per_channel",
            "near_zero_energy": "near_zero_swallow_energy_share_per_channel",
            "near_zero_loss": "near_zero_swallow_loss_share_per_channel",
            "key_token_energy": "key_token_clip_energy_share_per_channel",
            "key_token_near_zero_energy": "key_token_near_zero_energy_share_per_channel",
            "key_token_near_zero_loss": "key_token_near_zero_loss_share_per_channel",
            "oer_retention": "oer_retention_per_channel",
            "oer_loss_total": "oer_loss_total_per_channel",
            "oer_loss": "oer_loss_per_channel",
        },
    }
    for granularity, keys in key_map.items():
        scale_mean, scale_median, scale_p95 = _scale_summary(metrics[keys["scale"]], granularity)
        if topk_index_records is not None:
            topk_index_records.append({
                "sample_id": str(sample_id),
                "label": label,
                "bucket": bucket,
                "shot_group": shot_group,
                "tail_included": computed_tail,
                "target": f"L{int(layer_idx)}_{module_name}",
                "layer_idx": int(layer_idx),
                "module_name": str(module_name),
                "tensor_position": str(tensor_position),
                "granularity": granularity,
                "quant_mode": DEFAULT_QUANT_MODE,
                "topk_token_indices": metrics["topk_token_indices"].detach().cpu(),
                "oer_topk_scalar_indices": metrics[f"oer_topk_indices_{granularity}"].detach().cpu(),
                "valid_token_mask": None if valid_token_mask is None else valid_token_mask.detach().cpu(),
                "valid_token_count": int(metrics["valid_token_count"]),
                "total_token_count": int(metrics["total_token_count"]),
                "padding_token_ratio": float(metrics["padding_token_ratio"]),
            })
        rows.append(
            {
                "target": f"L{int(layer_idx)}_{module_name}",
                "sample_id": str(sample_id),
                "label": label,
                "bucket": bucket,
                "shot_group": shot_group,
                "tail_included": computed_tail,
                "layer_idx": int(layer_idx),
                "module_name": str(module_name),
                "tensor_position": str(tensor_position),
                "granularity": granularity,
                "quant_mode": DEFAULT_QUANT_MODE,
                "scale_type": granularity,
                "activation_shape": activation_shape_for_row,
                "original_layout": public_layout_info.get("original_layout", ""),
                "original_shape": public_layout_info.get("original_shape", ""),
                "canonical_layout": public_layout_info.get("canonical_layout", "BTC"),
                "canonical_shape": activation_shape_for_row,
                "layout_inference_source": public_layout_info.get("layout_inference_source", ""),
                "padding_mask_available": public_layout_info.get("padding_mask_available", False),
                "padding_mask_aligned": public_layout_info.get("padding_mask_aligned", False),
                "mask_status": public_layout_info.get("mask_status", ""),
                "valid_token_policy": valid_token_policy,
                "num_tokens": int(activation_shape[1]),
                "hidden_dim": int(activation_shape[2]),
                "valid_token_count": int(metrics["valid_token_count"]),
                "total_token_count": int(metrics["total_token_count"]),
                "padding_token_ratio": float(metrics["padding_token_ratio"]),
                "valid_token_mask_applied": bool(metrics["valid_token_mask_applied"]),
                "qmin": int(qmin),
                "qmax": int(qmax),
                "scale_mean": scale_mean,
                "scale_median": scale_median,
                "scale_p95": scale_p95,
                "upper_clip_rate": _tensor_to_float(metrics[keys["upper"]]),
                "lower_clip_rate": _tensor_to_float(metrics[keys["lower"]]),
                "total_clip_rate": _tensor_to_float(metrics[keys["total"]]),
                "boundary_hit_rate": _tensor_to_float(metrics[keys["boundary_hit"]]),
                "strict_overflow_rate": _tensor_to_float(metrics[keys["strict_overflow"]]),
                "saturation_boundary_hit_rate": _tensor_to_float(metrics[keys["saturation_boundary"]]),
                "saturation_strict_overflow_rate": _tensor_to_float(metrics[keys["saturation_strict_overflow"]]),
                "saturation_boundary_energy_share": _tensor_to_float(metrics[keys["saturation_boundary_energy"]]),
                "overflow_energy_proxy": _tensor_to_float(metrics[keys["overflow_energy_proxy"]]),
                "exact_zero_rate": _tensor_to_float(metrics[keys["exact_zero"]]),
                "near_zero_pm1_rate": _tensor_to_float(metrics[keys["near_zero_pm1"]]),
                "near_zero_rate": _tensor_to_float(metrics[keys["near_zero"]]),
                "topk_token_exact_zero_rate": _tensor_to_float(metrics[keys["topk_token_exact_zero"]]),
                "topk_token_near_zero_pm1_rate": _tensor_to_float(metrics[keys["topk_token_near_zero_pm1"]]),
                "topk_token_boundary_hit_rate": _tensor_to_float(metrics[keys["topk_token_boundary_hit"]]),
                "topk_token_strict_overflow_rate": _tensor_to_float(metrics[keys["topk_token_strict_overflow"]]),
                "high_energy_scalar_exact_zero_rate": _tensor_to_float(metrics[keys["high_energy_scalar_exact_zero"]]),
                "high_energy_scalar_near_zero_pm1_rate": _tensor_to_float(metrics[keys["high_energy_scalar_near_zero_pm1"]]),
                "high_energy_scalar_boundary_hit_rate": _tensor_to_float(metrics[keys["high_energy_scalar_boundary_hit"]]),
                "high_energy_scalar_strict_overflow_rate": _tensor_to_float(metrics[keys["high_energy_scalar_strict_overflow"]]),
                "topk_token_clip_rate": _tensor_to_float(metrics[keys["topk_clip"]]),
                "topk_token_near_zero_rate": _tensor_to_float(metrics[keys["topk_near_zero"]]),
                "saturation_clip_rate": _tensor_to_float(metrics[keys["saturation"]]),
                "saturation_overflow_rate": _tensor_to_float(metrics[keys["saturation_overflow"]]),
                "near_zero_swallow_rate": _tensor_to_float(metrics[keys["near_zero_swallow"]]),
                "high_energy_near_zero_rate": _tensor_to_float(metrics[keys["high_energy_near_zero"]]),
                "high_energy_scalar_clip_rate": _tensor_to_float(metrics[keys["high_energy_scalar_clip"]]),
                "near_zero_rate_deprecated_alias_of": "near_zero_pm1_rate",
                "total_clip_rate_deprecated_alias_of": "boundary_hit_rate",
                "topk_token_clip_rate_deprecated_alias_of": "topk_token_boundary_hit_rate",
                "topk_token_near_zero_rate_deprecated_alias_of": "topk_token_near_zero_pm1_rate",
                "high_energy_scalar_clip_rate_deprecated_alias_of": "high_energy_scalar_boundary_hit_rate",
                "oer_scalar_clip_rate": _tensor_to_float(metrics[keys["oer_scalar_clip"]]),
                "key_token_near_zero_swallow_rate": _tensor_to_float(metrics[keys["key_token_near_zero"]]),
                "saturation_clip_energy_share": _tensor_to_float(metrics[keys["saturation_energy"]]),
                "near_zero_swallow_energy_share": _tensor_to_float(metrics[keys["near_zero_energy"]]),
                "near_zero_swallow_loss_share": _tensor_to_float(metrics[keys["near_zero_loss"]]),
                "key_token_clip_energy_share": _tensor_to_float(metrics[keys["key_token_energy"]]),
                "key_token_near_zero_energy_share": _tensor_to_float(metrics[keys["key_token_near_zero_energy"]]),
                "key_token_near_zero_loss_share": _tensor_to_float(metrics[keys["key_token_near_zero_loss"]]),
                "oer_retention": _tensor_to_float(metrics[keys["oer_retention"]]),
                "oer_loss_total": _tensor_to_float(metrics[keys["oer_loss_total"]]),
                "oer_loss": _tensor_to_float(metrics[keys["oer_loss"]]),
                "max_abs_over_p99": _safe_ratio(basic["max_abs"], p99),
                "p999_over_p99": _safe_ratio(basic["p999"], p99),
            }
        )
    return rows

def _assert_summary_sample_protocol() -> None:
    """Ensure this probe and the summary module agree on required sample columns."""

    required = {"target", "sample_id", "bucket", "shot_group", "tail_included", "layer_idx", "module_name", "tensor_position", "granularity", "quant_mode"}
    missing = sorted(required - set(SAMPLE_COLUMNS))
    if missing:
        raise RuntimeError(
            "quant_danger_zone_summary.SAMPLE_COLUMNS is missing columns required by ffn_quant_danger_zone_probe: "
            f"{', '.join(missing)}; update the summary protocol before running this probe"
        )


def _validate_sample_row(row: Mapping[str, Any], *, source: str) -> None:
    expected = set(SAMPLE_COLUMNS)
    actual = set(row.keys())
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing fields: {', '.join(missing)}")
    if extra:
        details.append(f"extra fields: {', '.join(extra)}")
    raise ValueError(f"{source} does not match danger-zone sample row contract: " + "; ".join(details))


def _csv_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return json.dumps(value.tolist(), ensure_ascii=False)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _open_writer(csv_path: Path | None) -> tuple[Any | None, csv.DictWriter | None]:
    if csv_path is None:
        return None, None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    handle = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
    writer.writeheader()
    return handle, writer


def _emit_row(row: Mapping[str, Any], *, writer: csv.DictWriter | None, rows: list[dict[str, Any]]) -> None:
    _validate_sample_row(row, source="danger-zone sample row")
    if writer is not None:
        writer.writerow({key: _csv_value(row[key]) for key in SAMPLE_COLUMNS})
    else:
        rows.append(dict(row))


def collect_quant_danger_zone_metrics(
    model: nn.Module,
    loader: DataLoader,
    cfg: Any,
    target_modules: Sequence[Any] | None = None,
    *,
    sample_csv_path: Path | None = None,
) -> dict[str, Any]:
    """Stream FP32 BEATs, capture FFN activations, and emit danger-zone sample rows."""

    _assert_summary_sample_protocol()
    active_target_modules = _target_modules(model, target_modules, cfg)
    metadata_lookup = _load_sample_metadata_lookup(cfg)
    device = _device(cfg)
    model.to(device)
    model.eval()

    current_metadata: dict[str, list[Any]] | None = None
    current_batch_capture_counts: dict[tuple[int, str], int] | None = None
    current_batch_payloads: list[dict[str, Any]] | None = None
    capture_counts = {target: 0 for target in active_target_modules}
    sample_rows: list[dict[str, Any]] = []
    topk_index_records: list[dict[str, Any]] = []
    row_count = 0
    batch_count = 0
    sample_count = 0

    handle, writer = _open_writer(sample_csv_path)

    def on_capture(payload: dict[str, Any]) -> None:
        if current_metadata is None or current_batch_payloads is None:
            raise RuntimeError("hook captured activation without current batch metadata")
        layer_idx = int(payload["layer_idx"])
        module_name = str(payload["module_name"])
        target_key = (layer_idx, module_name)
        if target_key not in capture_counts:
            raise RuntimeError(f"hook captured an unregistered target module: L{layer_idx}.{module_name}")
        capture_counts[target_key] += 1
        if current_batch_capture_counts is not None:
            current_batch_capture_counts[target_key] = current_batch_capture_counts.get(target_key, 0) + 1
        current_batch_payloads.append(dict(payload))

    hook_manager = ActivationHookManager(
        model,
        active_target_modules,
        cache_activations=False,
        cache_device=_cache_device(cfg),
        on_capture=on_capture,
    )

    try:
        hook_manager.register_hooks()
        with torch.no_grad():
            for batch in loader:
                current_metadata = _batch_metadata(batch, metadata_lookup, cfg)
                current_batch_capture_counts = {target: 0 for target in active_target_modules}
                current_batch_payloads = []
                batch_count += 1
                sample_count += len(current_metadata["sample_id"])
                moved = _move_batch_to_device(batch, device)
                with _autocast_context(cfg, device):
                    forward_output, forward_mask = _forward_model(model, moved, cfg)
                del forward_output

                missing_targets = [target for target, count in current_batch_capture_counts.items() if count <= 0]
                if missing_targets:
                    raise RuntimeError(
                        "model forward did not pass through target FFN modules: "
                        f"{_format_target_modules(missing_targets)}; "
                        f"current batch sample_id={current_metadata['sample_id']}; "
                        "check ffn_quant_danger_zone_forward / model.extract_features / model.forward path"
                    )

                batch_size = len(current_metadata["sample_id"])
                for payload in current_batch_payloads:
                    layer_idx = int(payload["layer_idx"])
                    module_name = str(payload["module_name"])
                    for tensor_position in ACTIVATION_POSITIONS:
                        context = f"L{layer_idx} {module_name} {tensor_position}"
                        canonical = _activation_to_btc_with_layout(
                            payload[tensor_position],
                            batch_size=batch_size,
                            target_name=context,
                            padding_mask=forward_mask,
                        )
                        normalized = canonical.tensor
                        hidden_valid_mask = _hidden_valid_token_mask(
                            forward_mask,
                            batch_size=batch_size,
                            token_count=int(normalized.shape[1]),
                            cfg=cfg,
                            context=context,
                        ).to(device=normalized.device)
                        for sample_idx in range(batch_size):
                            sample_layout_info = layout_info_to_public_fields({
                                **canonical.layout_info,
                                "canonical_shape": tuple(int(item) for item in normalized[sample_idx: sample_idx + 1].shape),
                            })
                            rows = _danger_rows_for_activation(
                                normalized[sample_idx: sample_idx + 1],
                                sample_id=str(current_metadata["sample_id"][sample_idx]),
                                label=current_metadata["label"][sample_idx],
                                bucket=str(current_metadata["bucket"][sample_idx]),
                                shot_group=str(current_metadata["shot_group"][sample_idx]),
                                tail_flag=bool(current_metadata["tail_included"][sample_idx]),
                                layer_idx=layer_idx,
                                module_name=module_name,
                                tensor_position=tensor_position,
                                cfg=cfg,
                                valid_token_mask=hidden_valid_mask[sample_idx: sample_idx + 1],
                                topk_index_records=topk_index_records,
                                layout_info=sample_layout_info,
                                valid_token_policy=_valid_token_policy(cfg),
                            )
                            for row in rows:
                                _emit_row(row, writer=writer, rows=sample_rows)
                                row_count += 1

                del forward_mask
                current_batch_payloads = None
                current_batch_capture_counts = None
                current_metadata = None
    finally:
        hook_manager.clear_hooks()
        if handle is not None:
            handle.close()

    if batch_count <= 0:
        raise RuntimeError("fixed eval loader produced no batches")

    return {
        "sample_rows": sample_rows,
        "topk_index_records": topk_index_records,
        "num_topk_index_records": int(len(topk_index_records)),
        "num_rows": int(row_count),
        "num_samples": int(sample_count),
        "num_batches": int(batch_count),
        "target_modules": active_target_modules,
        "hook_capture_counts": {
            f"L{int(layer_idx)}.{module_name}": int(capture_counts[(layer_idx, module_name)])
            for layer_idx, module_name in active_target_modules
        },
    }


def _optional_checkpoint_path(cfg: Any) -> Path | None:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    if isinstance(beats_cfg, Mapping):
        for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT"):
            value = beats_cfg.get(key)
            if value is not None:
                return Path(value)
    for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT"):
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return Path(value)
    return None


def _target_modules_manifest(target_modules: Sequence[tuple[int, str]]) -> list[dict[str, Any]]:
    return [
        {
            "layer_idx": int(layer_idx),
            "module_name": str(module_name),
            "target_name": f"encoder.layers.{int(layer_idx)}.{module_name}",
        }
        for layer_idx, module_name in target_modules
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        return detached.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _build_manifest(
    *,
    cfg: Any,
    result: Mapping[str, Any],
    target_modules: Sequence[tuple[int, str]],
    target_modules_source: str,
    model_was_provided: bool,
    loader_was_provided: bool,
    sample_csv_path: Path,
    summary_csv_path: Path,
    comparison_summary_csv_path: Path,
    correlation_csv_path: Path,
    topk_indices_path: Path,
    manifest_path: Path,
    plot_paths: Mapping[str, Any],
) -> dict[str, Any]:
    audio_config = _make_audio_config(cfg)
    qmin, qmax = _qrange(cfg)
    dataset_cfg = _dataset_cfg(cfg)
    return {
        "experiment": "ffn_quant_danger_zone_probe",
        "model_source": "provided_model" if model_was_provided else "restore_clean_model_from_checkpoint",
        "loader_source": "provided_loader" if loader_was_provided else "AudioInputDataset",
        "checkpoint_path": _optional_checkpoint_path(cfg),
        "active_dataset": _cfg_get(cfg, "ACTIVE_DATASET", None),
        "dataset_name": _dataset_name(cfg),
        "dataset_root": _nested_get(dataset_cfg, "ROOT", None),
        "eval_csv": _optional_probe_csv(cfg),
        "audio_protocol": {
            "mode": str(_setting_get(cfg, "MODE", "eval")),
            "sample_rate": int(audio_config.sample_rate),
            "target_seconds": float(audio_config.target_seconds),
            "target_num_samples": int(audio_config.target_num_samples),
            "normalization": str(audio_config.normalization),
            "normalization_eps": float(audio_config.eps),
            "deterministic_crop": str(audio_config.deterministic_crop),
            "fixed_crop_start": int(audio_config.fixed_crop_start),
            "padding_mask_true_means": _nested_get(_cfg_get(cfg, "AUDIO", {}), "PADDING_MASK_TRUE_MEANS", "padding"),
        },
        "probe_params": {
            "device": str(_device(cfg)),
            "cache_device": None if _cache_device(cfg) is None else str(_cache_device(cfg)),
            "qmin": int(qmin),
            "qmax": int(qmax),
            "percentile": float(_percentile(cfg)),
            "topk_ratio": float(_topk_ratio(cfg)),
            "valid_token_policy": _valid_token_policy(cfg),
            "batch_size": int(_batch_size(cfg)),
            "num_workers": int(_num_workers(cfg)),
            "pin_memory": bool(_pin_memory(cfg)),
            "use_amp": bool(_setting_get(cfg, "USE_AMP", False)),
        },
        "target_modules_source": str(target_modules_source),
        "target_module_count": int(len(target_modules)),
        "target_modules": _target_modules_manifest(target_modules),
        "row_protocol": {
            "one_row_per": "sample_id x layer_idx x module_name x tensor_position x granularity x quant_mode",
            "tensor_positions": list(ACTIVATION_POSITIONS),
            "granularities": ["per_tensor", "per_channel"],
            "quant_mode": DEFAULT_QUANT_MODE,
            "activation_layout": "raw original_layout must be BTC or TBC; public metric input is canonical [B,T,C] via src.utils.layout_utils",
            "valid_token_mask": "controlled by VALID_TOKEN_POLICY; default require_mask; padding_mask must align exactly to canonical [B,T]",
            "padding_mask_mismatch_policy": "raise_error_stop_no_transpose_guessing",
        },
        "danger_zone_protocol": {
            "main_loss_field": "oer_loss_total = 1 - oer_retention",
            "abnormal_energy_loss": "deprecated_not_emitted_as_main_explanation_field",
            "decomposition": "explanatory diagnostics, not additive attribution",
            "exact_zero_rate": "q_int == zero_point over valid hidden tokens only",
            "near_zero_pm1_rate": "abs(q_int - zero_point) <= 1 over valid hidden tokens only",
            "boundary_hit_rate": "q_int <= qmin or q_int >= qmax over valid hidden tokens only",
            "strict_overflow_rate": "q_preclip < qmin or q_preclip > qmax over valid hidden tokens only",
            "near_zero_rate": "deprecated alias of near_zero_pm1_rate in this probe",
            "total_clip_rate": "deprecated alias of boundary_hit_rate in this probe",
            "topk_token_exact_zero_rate": "FP32 token-norm top-k token is exact-zero affected if any scalar/channel is exact zero",
            "topk_token_near_zero_pm1_rate": "FP32 token-norm top-k token is pm1-near-zero affected if any scalar/channel is pm1 near-zero",
            "topk_token_boundary_hit_rate": "FP32 token-norm top-k token has any scalar/channel hitting qmin/qmax",
            "topk_token_strict_overflow_rate": "FP32 token-norm top-k token has any scalar/channel overflowing before clamp",
            "high_energy_scalar_exact_zero_rate": "scalar exact-zero rate inside FP32 token-norm top-k tokens",
            "high_energy_scalar_near_zero_pm1_rate": "scalar pm1-near-zero rate inside FP32 token-norm top-k tokens",
            "high_energy_scalar_boundary_hit_rate": "scalar boundary-hit rate inside FP32 token-norm top-k tokens",
            "high_energy_scalar_strict_overflow_rate": "scalar strict-overflow rate inside FP32 token-norm top-k tokens",
            "high_energy_near_zero_rate": "FP32 OER-oracle scalar top-k positions quantized to pm1 near-zero",
            "oer_scalar_clip_rate": "diagnostic-only FP32 OER-oracle scalar top-k boundary hit rate",
            "qparams_source": "per-activation percentile probe, not fixed calibration qparams",
            "tail_amplification": "global/group-derived comparison summary only",
        },
        **group_protocol_metadata(
            tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
        ),
        "grouping_protocol": group_protocol_metadata(
            tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
        ),
        "counts": {
            "num_rows": int(result["num_rows"]),
            "num_samples": int(result["num_samples"]),
            "num_batches": int(result["num_batches"]),
            "num_topk_index_records": int(result.get("num_topk_index_records", 0)),
            "hook_capture_counts": result.get("hook_capture_counts", {}),
        },
        "outputs": {
            "sample_csv": sample_csv_path,
            "summary_csv": summary_csv_path,
            "comparison_summary_csv": comparison_summary_csv_path,
            "correlation_v2_csv": correlation_csv_path,
            "topk_indices_pt": topk_indices_path,
            "manifest": manifest_path,
            "plots": plot_paths,
        },
    }

def _target_name(layer_idx: int, module_name: str) -> str:
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _target_stem(layer_idx: int, module_name: str) -> str:
    return f"layer_{int(layer_idx):02d}_{module_name}"


def _mode_label(mode: str) -> str:
    normalized = normalize_quant_mode(mode, None)
    return {"w4": "W4-only", "a4": "A4-only", "w4a4": "W4A4", "none": "FP32"}[normalized]


def _mode_slug(mode: str) -> str:
    return _mode_label(mode).lower().replace("-", "_")


def _mode_uses_weight(mode: str) -> bool:
    return quant_mode_uses_weight(mode, None)


def _mode_uses_activation(mode: str) -> bool:
    return quant_mode_uses_activation(mode, None)


def _position_slug(position: str) -> str:
    return str(position).lower().replace("/", "_").replace(" ", "_")


def _danger_quant_modes(cfg: Any) -> tuple[str, ...]:
    configured = _setting_get(cfg, "DANGER_ZONE_QUANT_MODES", None)
    if configured is None:
        modes: list[str] = list(DANGER_ZONE_QUANT_MODES)
    else:
        modes = [str(item) for item in configured]
    if bool(_setting_get(cfg, "INCLUDE_OPTIONAL_REFERENCE_MODES", False)):
        modes.extend(str(item) for item in OPTIONAL_REFERENCE_MODES)
    return tuple(dict.fromkeys(_mode_label(mode) for mode in modes))


def _danger_positions(cfg: Any) -> tuple[str, ...]:
    configured = _setting_get(cfg, "POSITIONS", None)
    if configured is None:
        configured = _setting_get(cfg, "DANGER_ZONE_POSITIONS", DANGER_ZONE_POSITIONS)
    raw_positions = (configured,) if isinstance(configured, str) else configured
    positions = tuple(str(item) for item in raw_positions)
    allowed = set(DANGER_ZONE_POSITIONS)
    invalid = [item for item in positions if item not in allowed]
    if invalid:
        raise ValueError(f"unsupported danger-zone position(s): {invalid}; allowed={sorted(allowed)}")
    return positions


def _activation_granularity_for_position(position: str) -> str:
    return "per_tensor" if str(position) == "input_activation" else "projection"


def _split_csv(cfg: Any, name: str) -> Path:
    name_u = str(name).upper()
    if name_u == "EVAL":
        return _probe_csv(cfg)
    keys = (f"{name_u}_CSV", "CALIB_CSV", "TRAIN_CSV") if name_u == "CALIBRATION" else (f"{name_u}_CSV",)
    dataset_cfg = _dataset_cfg(cfg)
    for key in keys:
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
        value = _nested_get(dataset_cfg, key, None)
        if value is not None:
            return Path(value)
    raise AttributeError(f"missing {name_u}_CSV; configure it in CURRENT_DATASET or FFN_QUANT_DANGER_ZONE_PROBE")


def _loader_batch_size(cfg: Any, name: str) -> int:
    return int(_setting_get(cfg, f"{str(name).upper()}_BATCH_SIZE", _batch_size(cfg)))


def _build_calibration_loader(cfg: Any) -> DataLoader:
    override = _cfg_get(cfg, "build_quant_danger_zone_calibration_loader", None)
    if callable(override):
        return override(cfg)
    dataset: Any = AudioInputDataset(
        _split_csv(cfg, "calibration"),
        mode=str(_setting_get(cfg, "CALIBRATION_MODE", "train")),
        dataset_root=_nested_get(_dataset_cfg(cfg), "ROOT", None),
        project_root=_cfg_get(cfg, "BASE_DIR", None),
        audio_config=_make_audio_config(cfg),
        global_seed=_seed(cfg),
        epoch=0,
        return_dict=True,
        require_label_id=True,
        include_audio_stats=False,
    )
    max_samples = _setting_get(cfg, "MAX_CALIBRATION_SAMPLES", None)
    if max_samples is not None:
        limit = min(int(max_samples), len(dataset))
        if limit <= 0:
            raise ValueError(f"MAX_CALIBRATION_SAMPLES must be positive, got {max_samples}")
        dataset = Subset(dataset, range(limit))
    return DataLoader(
        dataset,
        batch_size=_loader_batch_size(cfg, "calibration"),
        shuffle=False,
        num_workers=_num_workers(cfg),
        pin_memory=_pin_memory(cfg),
        collate_fn=partial(audio_input_collate_fn, include_audio_stats=False),
    )


def _qparams_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _result_dir(cfg) / QPARAMS_DIRNAME / f"{_target_stem(layer_idx, module_name)}_qparams.pt"


def _shard_stem(layer_idx: int, module_name: str, quant_mode: str, position: str) -> str:
    return "__".join([
        "eval",
        _mode_slug(quant_mode),
        _activation_granularity_for_position(position),
        f"L{int(layer_idx):02d}",
        str(module_name),
        _position_slug(position),
        "shard0000",
    ])


def _sample_shard_path(cfg: Any, layer_idx: int, module_name: str, quant_mode: str, position: str) -> Path:
    return _result_dir(cfg) / SHARD_DIRNAME / f"{_shard_stem(layer_idx, module_name, quant_mode, position)}.sample_metrics.csv"


def _topk_compact_path(cfg: Any, layer_idx: int, module_name: str, quant_mode: str, position: str) -> Path:
    return _result_dir(cfg) / TOPK_COMPACT_DIRNAME / f"{_shard_stem(layer_idx, module_name, quant_mode, position)}.topk_compact.pt"


def _summary_partial_path(cfg: Any, layer_idx: int, module_name: str, quant_mode: str, position: str) -> Path:
    return _result_dir(cfg) / SUMMARY_PARTIAL_DIRNAME / f"{_shard_stem(layer_idx, module_name, quant_mode, position)}.summary_partial.json"


def _done_marker_path(cfg: Any, layer_idx: int, module_name: str, quant_mode: str, position: str) -> Path:
    return _result_dir(cfg) / DONE_DIRNAME / f"{_shard_stem(layer_idx, module_name, quant_mode, position)}.done.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive resume path.
        print(f"[resume] ignore invalid marker {path}: {exc}")
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path_value: Any) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path_value: Any) -> dict[str, Any]:
    if path_value is None:
        return {"path": None, "sha256": None}
    path = Path(path_value)
    return {"path": str(path.expanduser().resolve(strict=False)), "sha256": _sha256_file(path)}


def _fp32_reference_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_REFERENCE_CACHE", {})
    return value if isinstance(value, Mapping) else {}


def _fp32_reference_cache_root(cfg: Any) -> Path:
    configured = _setting_get(cfg, "FP32_REFERENCE_CACHE_ROOT", None)
    if configured is None:
        configured = _nested_get(_fp32_reference_cfg(cfg), "CACHE_ROOT", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "cache" / "fp32_reference"


def _fp32_core_eval_path(cfg: Any) -> Path:
    for key in ("CORE_CACHE_PATH", "CORE_EVAL_PATH", "FP32_CORE_CACHE_PATH"):
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
    reference_cfg = _fp32_reference_cfg(cfg)
    for key in ("CORE_CACHE_PATH", "CORE_EVAL_PATH", "FP32_CORE_CACHE_PATH"):
        value = _nested_get(reference_cfg, key, None)
        if value is not None:
            return Path(value)
    return _fp32_reference_cache_root(cfg) / FP32_REFERENCE_CORE_FILENAME


def _fp32_oer_oracle_path(cfg: Any) -> Path:
    for key in ("OER_ORACLE_PATH", "OER_ORACLE_EVAL_PATH", "FP32_OER_ORACLE_PATH"):
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
    reference_cfg = _fp32_reference_cfg(cfg)
    for key in ("OER_ORACLE_PATH", "OER_ORACLE_EVAL_PATH", "FP32_OER_ORACLE_PATH"):
        value = _nested_get(reference_cfg, key, None)
        if value is not None:
            return Path(value)
    return _fp32_reference_cache_root(cfg) / FP32_REFERENCE_OER_ORACLE_FILENAME


def _artifact_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _read_json_mapping_required(path: Path, *, context: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{context} manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} manifest must be a mapping: {path}")
    return dict(payload)


def _normalise_sample_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Tensor):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def _read_eval_csv_sample_ids(cfg: Any) -> list[str]:
    csv_path = _probe_csv(cfg)
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"eval CSV does not exist: {csv_path}")
    sample_ids: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"eval CSV has no header: {csv_path}")
        for row_idx, row in enumerate(reader, start=2):
            sample_id = _sample_id_from_metadata_row(row)
            if sample_id is None:
                raise ValueError(f"{csv_path}:{row_idx}: missing sample_id/audio_id/file_id")
            sample_ids.append(str(sample_id))
            if str(sample_id) in seen:
                duplicates.append(str(sample_id))
            seen.add(str(sample_id))
    if not sample_ids:
        raise ValueError(f"eval CSV has no samples: {csv_path}")
    if duplicates:
        raise ValueError(f"eval CSV has duplicate sample_id values: {sorted(set(duplicates))[:8]}")
    return sample_ids


def _effective_reference_sample_ids(cfg: Any, sample_ids: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    configured = _setting_get(cfg, "MAX_SAMPLES", None)
    full_count = len(sample_ids)
    if configured is None:
        return list(sample_ids), {
            "sample_order_policy": "full_eval_order",
            "max_samples_configured": None,
            "effective_sample_count": int(full_count),
            "full_sample_count": int(full_count),
        }
    requested = int(configured)
    limit = min(requested, full_count)
    if limit <= 0:
        raise ValueError(f"MAX_SAMPLES must be positive, got {configured}")
    return list(sample_ids[:limit]), {
        "sample_order_policy": "prefix_of_eval_order",
        "max_samples_configured": int(requested),
        "effective_sample_count": int(limit),
        "full_sample_count": int(full_count),
    }


def _resolve_reference_relative_path(root_file: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root_file.parent / path


def _validate_sharded_reference_manifest(
    artifact_path: Path,
    manifest: Mapping[str, Any],
    *,
    expected_cache_type: str,
    expected_stage: str,
    expected_storage_format: str,
    context: str,
) -> None:
    mismatches: list[str] = []
    required_pairs = {
        "cache_type": expected_cache_type,
        "stage": expected_stage,
        "storage_format": expected_storage_format,
    }
    for key, expected in required_pairs.items():
        actual = manifest.get(key)
        if actual != expected:
            mismatches.append(f"manifest[{key}] expected {expected!r}, got {actual!r}")
    if not bool(manifest.get("sharded", False)):
        mismatches.append("manifest[sharded] expected True")
    shard_paths = manifest.get("shard_paths")
    if not isinstance(shard_paths, list) or not shard_paths:
        mismatches.append("manifest[shard_paths] must be a non-empty list")
    declared_shards = manifest.get("num_shards")
    if declared_shards is None:
        mismatches.append("manifest[num_shards] is required")
    elif isinstance(shard_paths, list) and int(declared_shards) != len(shard_paths):
        mismatches.append(f"manifest[num_shards]={declared_shards} but shard_paths has {len(shard_paths)} entries")
    if isinstance(shard_paths, list):
        missing = [str(item) for item in shard_paths if not _resolve_reference_relative_path(artifact_path, item).is_file()]
        if missing:
            mismatches.append(f"missing shard files: {missing[:8]}")
    if mismatches:
        preview = "; ".join(mismatches[:8])
        raise RuntimeError(f"{context} is not a valid sharded FP32 reference artifact: {artifact_path}; {preview}")


def _validate_reference_identity(cfg: Any, manifest: Mapping[str, Any], *, context: str) -> tuple[str, str]:
    eval_csv = _probe_csv(cfg)
    eval_csv_hash = _sha256_file(eval_csv)
    if manifest.get("eval_csv_sha256") != eval_csv_hash:
        raise RuntimeError(
            f"{context} eval_csv_sha256 mismatch: "
            f"manifest={manifest.get('eval_csv_sha256')!r}, current={eval_csv_hash!r}, eval_csv={eval_csv}"
        )
    checkpoint_path = _optional_checkpoint_path(cfg)
    checkpoint_hash = None if checkpoint_path is None else _sha256_file(checkpoint_path)
    if checkpoint_hash is None:
        raise RuntimeError(f"{context} checkpoint_sha256 cannot be validated because checkpoint path is missing or unreadable")
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(
            f"{context} checkpoint_sha256 mismatch: "
            f"manifest={manifest.get('checkpoint_sha256')!r}, current={checkpoint_hash!r}, checkpoint={checkpoint_path}"
        )
    return str(eval_csv_hash), str(checkpoint_hash)


def _load_required_core_eval_reference(cfg: Any) -> dict[str, Any]:
    core_path = _fp32_core_eval_path(cfg)
    manifest_path = _artifact_manifest_path(core_path)
    manifest = _read_json_mapping_required(manifest_path, context="core_eval.pt")
    _validate_sharded_reference_manifest(
        core_path,
        manifest,
        expected_cache_type=FP32_CORE_CACHE_TYPE,
        expected_stage=FP32_CORE_STAGE,
        expected_storage_format=FP32_CORE_STORAGE_FORMAT,
        context="core_eval.pt",
    )
    eval_csv_hash, checkpoint_hash = _validate_reference_identity(cfg, manifest, context="core_eval.pt")
    eval_sample_ids = _read_eval_csv_sample_ids(cfg)
    eval_sample_hash = _sample_order_hash(eval_sample_ids)
    if manifest.get("sample_order_hash") != eval_sample_hash:
        raise RuntimeError(
            "core_eval.pt sample_order_hash does not match eval CSV order: "
            f"manifest={manifest.get('sample_order_hash')!r}, current={eval_sample_hash!r}"
        )
    payload = load_fp32_core_cache(core_path, materialize=False)
    core_sample_ids = _normalise_sample_ids(payload.get("sample_id"))
    if core_sample_ids != eval_sample_ids:
        first_mismatches = [
            {"index": idx, "core_eval": observed, "eval_csv": expected}
            for idx, (observed, expected) in enumerate(zip(core_sample_ids, eval_sample_ids))
            if observed != expected
        ][:8]
        raise RuntimeError(
            "core_eval.pt sample_id order must match eval CSV order; "
            f"core_count={len(core_sample_ids)}, eval_count={len(eval_sample_ids)}, first_mismatches={first_mismatches}"
        )
    effective_sample_ids, sample_policy = _effective_reference_sample_ids(cfg, core_sample_ids)
    effective_hash = _sample_order_hash(effective_sample_ids)
    return {
        "required": True,
        "status": "validated",
        "path": core_path,
        "manifest_path": manifest_path,
        "identity": _path_identity(core_path),
        "manifest_identity": _path_identity(manifest_path),
        "manifest": dict(manifest),
        "cache_type": manifest.get("cache_type"),
        "stage": manifest.get("stage"),
        "storage_format": manifest.get("storage_format"),
        "sharded": bool(manifest.get("sharded", False)),
        "num_shards": int(manifest.get("num_shards", 0)),
        "eval_csv_sha256": eval_csv_hash,
        "checkpoint_sha256": checkpoint_hash,
        "sample_order_hash": manifest.get("sample_order_hash"),
        "effective_sample_order_hash": effective_hash,
        "sample_count": len(effective_sample_ids),
        "full_sample_count": len(core_sample_ids),
        "sample_policy": sample_policy,
        "sample_id": effective_sample_ids,
        "full_sample_id": core_sample_ids,
        "sample_id_preview": effective_sample_ids[:5],
    }


def _target_manifest_set(value: Any) -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return result
    for item in value:
        if isinstance(item, Mapping):
            result.add((int(item.get("layer_idx", -1)), str(item.get("module_name", ""))))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            result.add((int(item[0]), str(item[1])))
    return result


def _oracle_activation_position_for_probe_position(position: str) -> str:
    if str(position) == "input_activation":
        return "input"
    if str(position) == "linear_output_projection":
        return "output"
    raise ValueError(f"unsupported danger-zone position for OER oracle: {position!r}")


def _required_oracle_activation_positions(cfg: Any) -> set[str]:
    return {_oracle_activation_position_for_probe_position(position) for position in _danger_positions(cfg)}


def _load_required_oer_oracle_reference(
    cfg: Any,
    *,
    core_reference: Mapping[str, Any],
    target_modules: Sequence[tuple[int, str]],
) -> dict[str, Any]:
    oracle_path = _fp32_oer_oracle_path(cfg)
    manifest_path = _artifact_manifest_path(oracle_path)
    manifest = _read_json_mapping_required(manifest_path, context="oer_oracle_eval.pt")
    _validate_sharded_reference_manifest(
        oracle_path,
        manifest,
        expected_cache_type=FP32_OER_CACHE_TYPE,
        expected_stage=FP32_OER_STAGE,
        expected_storage_format=FP32_OER_STORAGE_FORMAT,
        context="oer_oracle_eval.pt",
    )
    eval_csv_hash, checkpoint_hash = _validate_reference_identity(cfg, manifest, context="oer_oracle_eval.pt")
    if eval_csv_hash != core_reference.get("eval_csv_sha256"):
        raise RuntimeError("oer_oracle_eval.pt eval_csv_sha256 differs from core_eval.pt")
    if checkpoint_hash != core_reference.get("checkpoint_sha256"):
        raise RuntimeError("oer_oracle_eval.pt checkpoint_sha256 differs from core_eval.pt")
    if manifest.get("sample_order_hash") != core_reference.get("sample_order_hash"):
        raise RuntimeError(
            "oer_oracle_eval.pt sample_order_hash differs from core_eval.pt: "
            f"oer={manifest.get('sample_order_hash')!r}, core={core_reference.get('sample_order_hash')!r}"
        )
    cached_ratio = manifest.get("topk_ratio")
    if cached_ratio is None or abs(float(cached_ratio) - float(_topk_ratio(cfg))) > 1e-12:
        raise RuntimeError(f"oer_oracle_eval.pt topk_ratio mismatch: manifest={cached_ratio!r}, current={_topk_ratio(cfg)!r}")
    positions = {str(item) for item in manifest.get("activation_positions", [])}
    required_positions = _required_oracle_activation_positions(cfg)
    missing_positions = sorted(required_positions - positions)
    if missing_positions:
        raise RuntimeError(f"oer_oracle_eval.pt missing activation positions required by this experiment: {missing_positions}")
    cached_targets = _target_manifest_set(manifest.get("target_modules"))
    missing_targets = [
        {"layer_idx": int(layer_idx), "module_name": str(module_name)}
        for layer_idx, module_name in target_modules
        if (int(layer_idx), str(module_name)) not in cached_targets
    ]
    if missing_targets:
        raise RuntimeError(f"oer_oracle_eval.pt missing target modules required by this experiment: {missing_targets[:8]}")
    oracle = load_fp32_oer_oracle(oracle_path, materialize=False)
    oracle_sample_ids = _normalise_sample_ids(oracle.get("sample_id_order", oracle.get("sample_id")))
    core_full_sample_ids = _normalise_sample_ids(core_reference.get("full_sample_id"))
    if oracle_sample_ids and oracle_sample_ids != core_full_sample_ids:
        raise RuntimeError("oer_oracle_eval.pt sample_id_order must exactly match core_eval.pt full sample_id order")
    return {
        "required": True,
        "status": "validated",
        "path": oracle_path,
        "manifest_path": manifest_path,
        "identity": _path_identity(oracle_path),
        "manifest_identity": _path_identity(manifest_path),
        "manifest": dict(manifest),
        "oracle": oracle,
        "cache_type": manifest.get("cache_type"),
        "stage": manifest.get("stage"),
        "storage_format": manifest.get("storage_format"),
        "sharded": bool(manifest.get("sharded", False)),
        "num_shards": int(manifest.get("num_shards", 0)),
        "sample_order_hash": manifest.get("sample_order_hash"),
        "effective_sample_order_hash": core_reference.get("effective_sample_order_hash"),
        "sample_count": int(core_reference.get("sample_count", 0)),
        "topk_ratio": float(cached_ratio),
        "activation_positions": list(manifest.get("activation_positions", [])),
        "required_activation_positions": sorted(required_positions),
    }


def _reference_public_state(reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in reference.items()
        if key not in {"oracle", "sample_id", "full_sample_id"}
    }


def _build_required_reference_cache(cfg: Any, target_modules: Sequence[tuple[int, str]]) -> dict[str, Any]:
    core_reference = _load_required_core_eval_reference(cfg)
    oer_reference = _load_required_oer_oracle_reference(
        cfg,
        core_reference=core_reference,
        target_modules=target_modules,
    )
    reference_state = {
        "required": {
            "core_eval": _reference_public_state(core_reference),
            "oer_oracle_eval": _reference_public_state(oer_reference),
        },
    }
    return {
        "core_reference": core_reference,
        "oer_oracle_reference": oer_reference,
        "reference_state": reference_state,
        "manifest": reference_state,
        "sample_id": list(core_reference["sample_id"]),
        "sample_order_hash": core_reference["effective_sample_order_hash"],
    }


def _fp32_reference_signature_payload(reference_state: Mapping[str, Any]) -> dict[str, Any]:
    required = _nested_get(reference_state, "required", {})
    core = _nested_get(required, "core_eval", {})
    oer = _nested_get(required, "oer_oracle_eval", {})
    return {
        "required_core_eval": {
            "path": _nested_get(core, "identity", {}).get("path") if isinstance(_nested_get(core, "identity", {}), Mapping) else None,
            "sha256": _nested_get(core, "identity", {}).get("sha256") if isinstance(_nested_get(core, "identity", {}), Mapping) else None,
            "manifest_sha256": _nested_get(core, "manifest_identity", {}).get("sha256") if isinstance(_nested_get(core, "manifest_identity", {}), Mapping) else None,
            "storage_format": core.get("storage_format"),
            "sample_order_hash": core.get("sample_order_hash"),
            "effective_sample_order_hash": core.get("effective_sample_order_hash"),
            "sample_count": core.get("sample_count"),
            "sample_policy": core.get("sample_policy"),
            "eval_csv_sha256": core.get("eval_csv_sha256"),
            "checkpoint_sha256": core.get("checkpoint_sha256"),
        },
        "required_oer_oracle_eval": {
            "path": _nested_get(oer, "identity", {}).get("path") if isinstance(_nested_get(oer, "identity", {}), Mapping) else None,
            "sha256": _nested_get(oer, "identity", {}).get("sha256") if isinstance(_nested_get(oer, "identity", {}), Mapping) else None,
            "manifest_sha256": _nested_get(oer, "manifest_identity", {}).get("sha256") if isinstance(_nested_get(oer, "manifest_identity", {}), Mapping) else None,
            "storage_format": oer.get("storage_format"),
            "sample_order_hash": oer.get("sample_order_hash"),
            "effective_sample_order_hash": oer.get("effective_sample_order_hash"),
            "sample_count": oer.get("sample_count"),
            "topk_ratio": oer.get("topk_ratio"),
            "required_activation_positions": oer.get("required_activation_positions"),
        },
    }

def _experiment_signature_payload(
    cfg: Any,
    target_modules: Sequence[tuple[int, str]],
    fp32_reference_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "ffn_quant_danger_zone_probe",
        "dataset_name": _dataset_name(cfg),
        "active_dataset": _cfg_get(cfg, "ACTIVE_DATASET", None),
        "checkpoint": _path_identity(_optional_checkpoint_path(cfg)),
        "calibration_csv": _path_identity(_split_csv(cfg, "calibration")),
        "eval_csv": _path_identity(_probe_csv(cfg)),
        "quant": _json_safe(_quant_cfg(cfg)),
        "probe": _json_safe(_probe_cfg(cfg)),
        "legacy_probe": _json_safe(_legacy_probe_cfg(cfg)),
        "audio": _json_safe(_cfg_get(cfg, "AUDIO", {})),
        "use_amp": bool(_setting_get(cfg, "USE_AMP", False)),
        "topk_ratio": float(_topk_ratio(cfg)),
        "quant_modes": list(_danger_quant_modes(cfg)),
        "optional_reference_modes": list(OPTIONAL_REFERENCE_MODES),
        "positions": list(_danger_positions(cfg)),
        "fp32_reference": _fp32_reference_signature_payload(fp32_reference_state),
        "activation_metric_scope": "input_activation_only",
        "projection_metric_scope": "linear_output_projection_only",
        "integer_grid_source": "direct_quantlinear_forward_grid",
        "target_modules": [
            {"layer_idx": int(layer_idx), "module_name": str(module_name)}
            for layer_idx, module_name in target_modules
        ],
        "save_full_activation": False,
    }


def _experiment_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _protocol_matches(payload: Mapping[str, Any], protocol_version: str, experiment_signature: str) -> bool:
    return str(payload.get("protocol_version", "")) == str(protocol_version) and str(payload.get("experiment_signature", "")) == str(experiment_signature)


def _save_qparams_atomic(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    qparams: Mapping[str, Any],
    *,
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> Path:
    path = _qparams_path(cfg, layer_idx, module_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    payload = dict(qparams)
    payload["protocol_version"] = protocol_version
    payload["experiment_signature"] = experiment_signature
    payload["experiment_signature_payload"] = _json_safe(signature_payload)
    payload["target"] = {"layer_idx": int(layer_idx), "module_name": str(module_name)}
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def _load_qparams_if_valid(path: Path, protocol_version: str, experiment_signature: str) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        value = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive resume path.
        print(f"[resume] ignore invalid qparams {path}: {exc}")
        return None
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    if not _protocol_matches(payload, protocol_version, experiment_signature):
        print(f"[resume] ignore stale qparams {path}: protocol/signature mismatch")
        return None
    if "weight" not in payload or "activation" not in payload:
        print(f"[resume] ignore invalid qparams {path}: missing weight/activation")
        return None
    return payload


def _run_forward_only(model: nn.Module, loader: DataLoader, cfg: Any) -> None:
    device = _device(cfg)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            with _autocast_context(cfg, device):
                output, _mask = _forward_model(model, moved, cfg)
            del output, moved


def _run_two_pass_calibration(controller: Any, model: nn.Module, loader: DataLoader, cfg: Any) -> None:
    if controller.activation_quant_enabled:
        controller.prepare_calibration_range(reset_observers=True)
        _run_forward_only(model, loader, cfg)
        controller.prepare_calibration_histogram()
        _run_forward_only(model, loader, cfg)
        controller.freeze_qparams()
    controller.qparams_frozen = True
    controller.prepare_quant_forward()


def _qrange_from_config(qconfig: Any, *, kind: str) -> tuple[int, int]:
    if kind == "weight":
        return int(qconfig.weight_qmin), int(qconfig.weight_qmax)
    return int(qconfig.activation_qmin), int(qconfig.activation_qmax)


def _calibrate_target_qparams(cfg: Any, layer_idx: int, module_name: str, calibration_loader: DataLoader) -> dict[str, Any]:
    device = _device(cfg)
    model = restore_clean_model_from_checkpoint(cfg).to(device)
    quant_module = replace_ffn_module_with_quant(
        model,
        int(layer_idx),
        module_name,
        _quant_cfg(cfg),
        quant_mode="W4A4",
        enable_fake_quant=False,
    )
    controller = make_single_ffn_quant_controller(
        model,
        quant_module,
        layer_idx=int(layer_idx),
        module_name=module_name,
        quant_config=_quant_cfg(cfg),
        quant_mode="W4A4",
    )
    try:
        _run_two_pass_calibration(controller, model, calibration_loader, cfg)
        w_qmin, w_qmax = _qrange_from_config(quant_module.config, kind="weight")
        a_qmin, a_qmax = _qrange_from_config(quant_module.config, kind="activation")
        weight_qparams = freeze_weight_qparams_from_tensor(
            quant_module.linear.weight.detach().cpu(),
            qmin=w_qmin,
            qmax=w_qmax,
            symmetric=True,
            eps=float(quant_module.config.eps),
            channel_axis=0,
        )
        activation_qparams = {
            "scale": quant_module.activation_scale.detach().cpu().clone(),
            "zero_point": quant_module.activation_zero_point.detach().cpu().clone(),
            "qmin": a_qmin,
            "qmax": a_qmax,
            "symmetric": bool(quant_module.config.activation_symmetric),
            "granularity": "per_tensor",
            "observer_type": "activation",
            "method": "calibration_two_pass_percentile",
            "channel_axis": -1,
        }
        return {
            "layer_idx": int(layer_idx),
            "module_name": module_name,
            "target_name": _target_name(layer_idx, module_name),
            "weight": weight_qparams,
            "activation": activation_qparams,
            "observer_summary": controller.observer_summary(),
            "weight_observer_summary": controller.weight_observer_summary(),
            "controller_summary": controller.state_summary(),
        }
    finally:
        controller.remove_padding_mask_hooks()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_or_create_qparams(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    calibration_loader: DataLoader,
    *,
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = _qparams_path(cfg, layer_idx, module_name)
    cached = _load_qparams_if_valid(path, protocol_version, experiment_signature)
    if cached is not None:
        return cached, path
    qparams = _calibrate_target_qparams(cfg, layer_idx, module_name, calibration_loader)
    saved_path = _save_qparams_atomic(
        cfg,
        layer_idx,
        module_name,
        qparams,
        protocol_version=protocol_version,
        experiment_signature=experiment_signature,
        signature_payload=signature_payload,
    )
    payload = torch.load(saved_path, map_location="cpu")
    return dict(payload), saved_path


def _complete_sample_row(row: Mapping[str, Any]) -> dict[str, Any]:
    completed = {name: row.get(name, "") for name in SAMPLE_COLUMNS}
    _validate_sample_row(completed, source="danger-zone shard row")
    return completed


def _write_rows_atomic(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            completed = _complete_sample_row(row)
            writer.writerow({key: _csv_value(completed.get(key, "")) for key in SAMPLE_COLUMNS})
    os.replace(tmp_path, csv_path)


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    encoded = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shard_done_valid(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    position: str,
    *,
    protocol_version: str,
    experiment_signature: str,
    qparams_hash: str | None,
) -> bool:
    payload = _read_json(_done_marker_path(cfg, layer_idx, module_name, quant_mode, position))
    if payload is None or not _protocol_matches(payload, protocol_version, experiment_signature):
        return False
    target = payload.get("target", {})
    if int(_nested_get(target, "layer_idx", -1)) != int(layer_idx):
        return False
    if str(_nested_get(target, "module_name", "")) != str(module_name):
        return False
    if str(payload.get("quant_mode", "")) != _mode_label(quant_mode):
        return False
    if str(payload.get("position", "")) != str(position):
        return False
    if str(payload.get("qparams_hash", "")) != str(qparams_hash):
        return False
    for key in ("sample_shard", "topk_compact", "summary_partial"):
        value = payload.get(key)
        if value is None:
            return False
        path = Path(str(value))
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return True


def _write_shard_done(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    position: str,
    *,
    protocol_version: str,
    experiment_signature: str,
    sample_shard: Path,
    topk_compact: Path,
    summary_partial: Path,
    qparams_path: Path,
    sample_ids: Sequence[str],
    row_count: int,
) -> None:
    _write_json_atomic(
        _done_marker_path(cfg, layer_idx, module_name, quant_mode, position),
        {
            "protocol_version": protocol_version,
            "experiment_signature": experiment_signature,
            "status": "done",
            "target": {"layer_idx": int(layer_idx), "module_name": str(module_name)},
            "quant_mode": _mode_label(quant_mode),
            "position": str(position),
            "activation_granularity": _activation_granularity_for_position(position),
            "integer_grid_source": "direct_quantlinear_forward_grid" if position == "input_activation" else "not_applicable_projection",
            "sample_shard": str(sample_shard),
            "topk_compact": str(topk_compact),
            "summary_partial": str(summary_partial),
            "qparams_path": str(qparams_path),
            "qparams_hash": _sha256_file(qparams_path),
            "sample_order_hash": _sample_order_hash(sample_ids),
            "num_samples_done": int(len(sample_ids)),
            "num_rows": int(row_count),
            "layout_protocol_version": 2,
            "group_protocol": group_protocol_metadata(tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}"),
            "valid_token_policy": _valid_token_policy(cfg),
        },
    )


def _slice_grid(grid: Mapping[str, Any], sample_idx: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in grid.items():
        if isinstance(value, Tensor) and value.ndim >= 1 and int(value.shape[0]) > sample_idx:
            result[key] = value[sample_idx : sample_idx + 1]
        else:
            result[key] = value
    return result


def _row_layout_fields(layout_info: Mapping[str, Any] | None, shape: Sequence[int]) -> dict[str, Any]:
    public = layout_info_to_public_fields(dict(layout_info or {}))
    shape_tuple = tuple(int(item) for item in shape)
    return {
        "activation_shape": shape_tuple,
        "original_layout": public.get("original_layout", ""),
        "original_shape": public.get("original_shape", ""),
        "canonical_layout": public.get("canonical_layout", "BTC"),
        "canonical_shape": shape_tuple,
        "layout_inference_source": public.get("layout_inference_source", ""),
        "padding_mask_available": public.get("padding_mask_available", False),
        "padding_mask_aligned": public.get("padding_mask_aligned", False),
        "mask_status": public.get("mask_status", ""),
    }


def _stats_ratios(activation: Tensor, valid_token_mask: Tensor | None) -> tuple[float, float]:
    stats_activation = activation
    if valid_token_mask is not None:
        valid = valid_token_mask.detach().bool().to(device=activation.device)
        if valid.ndim == 1:
            valid = valid.reshape(1, -1)
        stats_activation = activation[valid]
    basic = compute_basic_distribution_stats(stats_activation)
    p99 = basic["p99"]
    return _safe_ratio(basic["max_abs"], p99), _safe_ratio(basic["p999"], p99)


def _scale_summary_from_metric(metric: Mapping[str, Any]) -> tuple[float, float, float]:
    return _scale_summary(metric.get("scale", torch.tensor(float("nan"))), "per_tensor")


def _metadata_for_sample(metadata: Mapping[str, list[Any]], sample_idx: int) -> dict[str, Any]:
    return {
        "sample_id": str(metadata["sample_id"][sample_idx]),
        "label": metadata["label"][sample_idx],
        "bucket": str(metadata["bucket"][sample_idx]),
        "shot_group": str(metadata["shot_group"][sample_idx]),
        "tail_included": bool(metadata["tail_included"][sample_idx]),
    }


def _base_row(
    *,
    sample_meta: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
    position: str,
    quant_mode: str,
    shape: Sequence[int],
    layout_info: Mapping[str, Any] | None,
    valid_token_policy: str,
) -> dict[str, Any]:
    row = {
        "target": f"L{int(layer_idx)}_{module_name}",
        "sample_id": str(sample_meta["sample_id"]),
        "label": sample_meta.get("label", ""),
        "bucket": str(sample_meta["bucket"]),
        "shot_group": str(sample_meta["shot_group"]),
        "tail_included": bool(sample_meta["tail_included"]),
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "tensor_position": str(position),
        "granularity": _activation_granularity_for_position(position),
        "quant_mode": _mode_label(quant_mode),
        "scale_type": _activation_granularity_for_position(position),
        "valid_token_policy": valid_token_policy,
    }
    row.update(_row_layout_fields(layout_info, shape))
    return row


def _activation_metric_row(
    *,
    cfg: Any,
    sample_meta: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    fp32_sample: Tensor,
    valid_mask: Tensor | None,
    metric: Mapping[str, Any],
    layout_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scale_mean, scale_median, scale_p95 = _scale_summary_from_metric(metric)
    max_abs_over_p99, p999_over_p99 = _stats_ratios(fp32_sample, valid_mask)
    shape = tuple(int(item) for item in metric["activation_shape"])
    row = _base_row(
        sample_meta=sample_meta,
        layer_idx=layer_idx,
        module_name=module_name,
        position="input_activation",
        quant_mode=quant_mode,
        shape=shape,
        layout_info=layout_info,
        valid_token_policy=_valid_token_policy(cfg),
    )
    row.update(
        {
            "metric_status": "ok",
            "key_token_source": metric.get("key_token_source", "fp32_token_energy_topk"),
            "aux_token_source": metric.get("aux_token_source", "scalar_oer_topk_aggregated_to_token"),
            "integer_grid_source": metric.get("integer_grid_source", "quantlinear_forward"),
            "is_direct_quantlinear_forward_grid": bool(metric.get("is_direct_quantlinear_forward_grid", True)),
            "num_tokens": int(shape[1]),
            "hidden_dim": int(shape[2]),
            "valid_token_count": int(metric["valid_token_count"]),
            "total_token_count": int(metric["total_token_count"]),
            "padding_token_ratio": float(metric["padding_token_ratio"]),
            "valid_token_mask_applied": bool(metric["valid_token_mask_applied"]),
            "qmin": int(metric["qmin"]),
            "qmax": int(metric["qmax"]),
            "scale_mean": scale_mean,
            "scale_median": scale_median,
            "scale_p95": scale_p95,
            "upper_clip_rate": _tensor_to_float(metric["clip_high_rate"]),
            "lower_clip_rate": _tensor_to_float(metric["clip_low_rate"]),
            "total_clip_rate": _tensor_to_float(metric["clip_rate"]),
            "boundary_hit_rate": _tensor_to_float(metric["boundary_hit_rate"]),
            "strict_overflow_rate": _tensor_to_float(metric["strict_overflow_rate"]),
            "saturation_boundary_energy_share": _tensor_to_float(metric["saturation_boundary_energy_share"]),
            "overflow_energy_proxy": _tensor_to_float(metric["overflow_energy_proxy"]),
            "exact_zero_rate": _tensor_to_float(metric["exact_zero_rate"]),
            "near_zero_pm1_rate": _tensor_to_float(metric["near_zero_pm1_rate"]),
            "near_zero_rate": _tensor_to_float(metric["near_zero_rate"]),
            "topk_token_exact_zero_rate": _tensor_to_float(metric["topk_token_exact_zero_rate"]),
            "topk_token_near_zero_pm1_rate": _tensor_to_float(metric["topk_token_near_zero_pm1_rate"]),
            "topk_token_boundary_hit_rate": _tensor_to_float(metric["topk_token_boundary_hit_rate"]),
            "topk_token_strict_overflow_rate": _tensor_to_float(metric["topk_token_strict_overflow_rate"]),
            "high_energy_scalar_exact_zero_rate": _tensor_to_float(metric["high_energy_scalar_exact_zero_rate"]),
            "high_energy_scalar_near_zero_pm1_rate": _tensor_to_float(metric["high_energy_scalar_near_zero_pm1_rate"]),
            "high_energy_scalar_boundary_hit_rate": _tensor_to_float(metric["high_energy_scalar_boundary_hit_rate"]),
            "high_energy_scalar_strict_overflow_rate": _tensor_to_float(metric["high_energy_scalar_strict_overflow_rate"]),
            "topk_token_clip_rate": _tensor_to_float(metric["topk_token_boundary_hit_rate"]),
            "high_energy_scalar_clip_rate": _tensor_to_float(metric["high_energy_scalar_boundary_hit_rate"]),
            "topk_token_near_zero_rate": _tensor_to_float(metric["topk_token_near_zero_pm1_rate"]),
            "near_zero_swallow_rate": _tensor_to_float(metric["high_energy_near_zero_rate"]),
            "high_energy_near_zero_rate": _tensor_to_float(metric["high_energy_near_zero_rate"]),
            "oer_scalar_clip_rate": _tensor_to_float(metric["oer_scalar_clip_rate"]),
            "key_token_near_zero_swallow_rate": _tensor_to_float(metric["high_energy_near_zero_rate"]),
            "saturation_clip_energy_share": _tensor_to_float(metric["saturation_clip_energy_share"]),
            "near_zero_swallow_energy_share": _tensor_to_float(metric["near_zero_swallow_energy_share"]),
            "near_zero_swallow_loss_share": _tensor_to_float(metric["near_zero_swallow_loss_share"]),
            "key_token_clip_energy_share": _tensor_to_float(metric["key_token_clip_energy_share"]),
            "key_token_near_zero_energy_share": _tensor_to_float(metric["key_token_near_zero_energy_share"]),
            "key_token_near_zero_loss_share": _tensor_to_float(metric["key_token_near_zero_loss_share"]),
            "oer_retention": _tensor_to_float(metric["oer_retention"]),
            "oer_loss_total": _tensor_to_float(metric["oer_loss_total"]),
            "oer_loss": _tensor_to_float(metric["oer_loss"]),
            "max_abs_over_p99": max_abs_over_p99,
            "p999_over_p99": p999_over_p99,
            "near_zero_rate_deprecated_alias_of": "near_zero_pm1_rate",
            "total_clip_rate_deprecated_alias_of": "boundary_hit_rate",
            "topk_token_clip_rate_deprecated_alias_of": "topk_token_boundary_hit_rate",
            "topk_token_near_zero_rate_deprecated_alias_of": "topk_token_near_zero_pm1_rate",
            "high_energy_scalar_clip_rate_deprecated_alias_of": "high_energy_scalar_boundary_hit_rate",
        }
    )
    return row


def _projection_metric_row(
    *,
    cfg: Any,
    sample_meta: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    fp32_sample: Tensor,
    valid_mask: Tensor | None,
    metric: Mapping[str, Any],
    layout_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    max_abs_over_p99, p999_over_p99 = _stats_ratios(fp32_sample, valid_mask)
    shape = tuple(int(item) for item in metric["activation_shape"])
    row = _base_row(
        sample_meta=sample_meta,
        layer_idx=layer_idx,
        module_name=module_name,
        position="linear_output_projection",
        quant_mode=quant_mode,
        shape=shape,
        layout_info=layout_info,
        valid_token_policy=_valid_token_policy(cfg),
    )
    row.update(
        {
            "metric_status": "projection_detail",
            "key_token_source": "not_applicable_projection",
            "aux_token_source": metric.get("oer_oracle_source", "scalar_oer_topk_projection"),
            "integer_grid_source": "not_applicable_projection",
            "is_direct_quantlinear_forward_grid": False,
            "num_tokens": int(shape[1]),
            "hidden_dim": int(shape[2]),
            "valid_token_count": int(metric["valid_token_count"]),
            "total_token_count": int(metric["total_token_count"]),
            "padding_token_ratio": float(metric["padding_token_ratio"]),
            "valid_token_mask_applied": bool(metric["valid_token_mask_applied"]),
            "oer_retention": _tensor_to_float(metric["projection_oer_retention"]),
            "oer_loss_total": _tensor_to_float(metric["projection_oer_loss"]),
            "oer_loss": _tensor_to_float(metric["projection_oer_loss"]),
            "max_abs_over_p99": max_abs_over_p99,
            "p999_over_p99": p999_over_p99,
            "projection_rel_l2_error": _tensor_to_float(metric["projection_rel_l2_error"]),
            "projection_cosine_drift": _tensor_to_float(metric["projection_cosine_drift"]),
            "projection_mse": _tensor_to_float(metric["projection_mse"]),
            "projection_oer_shift": _tensor_to_float(metric["projection_oer_shift"]),
            "projection_oer_retention": _tensor_to_float(metric["projection_oer_retention"]),
            "projection_oer_loss": _tensor_to_float(metric["projection_oer_loss"]),
            "projection_off_oracle_energy_ratio": _tensor_to_float(metric["projection_off_oracle_energy_ratio"]),
            "output_energy_distortion": _tensor_to_float(metric["output_energy_distortion"]),
        }
    )
    return row


def _tensor_payload_1d(value: Any) -> list[Any]:
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    if tensor.dtype == torch.bool:
        return [bool(item) for item in tensor.tolist()]
    if tensor.dtype.is_floating_point:
        return [float(item) for item in tensor.tolist()]
    return [int(item) for item in tensor.tolist()]


def _input_topk_record(
    *,
    sample_meta: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    metric: Mapping[str, Any],
    layout_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    compact = metric["compact_topk"]
    token_ids = _tensor_payload_1d(compact["topk_token_indices"])
    return {
        "sample_id": str(sample_meta["sample_id"]),
        "label_id": sample_meta.get("label", ""),
        "bucket": str(sample_meta["bucket"]),
        "tail_included": bool(sample_meta["tail_included"]),
        "quant_mode": _mode_label(quant_mode),
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "position": "input_activation",
        "topk_token_ids": token_ids,
        "topk_token_energy_fp32": _tensor_payload_1d(compact["topk_token_energy_fp32"]),
        "topk_token_energy_quant": _tensor_payload_1d(compact["topk_token_energy_quant"]),
        "topk_token_strict_overflow_flags": _tensor_payload_1d(compact["topk_token_strict_overflow_flags"]),
        "topk_token_near_zero_pm1_flags": _tensor_payload_1d(compact["topk_token_near_zero_pm1_flags"]),
        "topk_token_boundary_hit_flags": _tensor_payload_1d(compact["topk_token_boundary_hit_flags"]),
        "high_energy_scalar_strict_overflow_rate": _tensor_to_float(metric["high_energy_scalar_strict_overflow_rate"]),
        "high_energy_scalar_near_zero_pm1_rate": _tensor_to_float(metric["high_energy_scalar_near_zero_pm1_rate"]),
        "valid_token_count": int(metric["valid_token_count"]),
        "topk_k": len(token_ids),
        "topk_source": metric.get("key_token_source", "fp32_token_energy_topk"),
        "padding_mask_hash": None,
        "layout_info": layout_info_to_public_fields(dict(layout_info or {})),
        "debug_summary": {
            "scale": _tensor_payload_1d(metric.get("scale", torch.tensor(float("nan"))))[:8],
            "zero_point": _tensor_payload_1d(metric.get("zero_point", torch.tensor(0.0)))[:8],
            "qmin": int(metric["qmin"]),
            "qmax": int(metric["qmax"]),
            "strict_overflow_rate": _tensor_to_float(metric["strict_overflow_rate"]),
            "near_zero_pm1_rate": _tensor_to_float(metric["near_zero_pm1_rate"]),
            "oer_retention": _tensor_to_float(metric["oer_retention"]),
            "oer_loss": _tensor_to_float(metric["oer_loss"]),
        },
    }


def _projection_topk_record(
    *,
    sample_meta: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    metric: Mapping[str, Any],
    layout_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "sample_id": str(sample_meta["sample_id"]),
        "label_id": sample_meta.get("label", ""),
        "bucket": str(sample_meta["bucket"]),
        "tail_included": bool(sample_meta["tail_included"]),
        "quant_mode": _mode_label(quant_mode),
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "position": "linear_output_projection",
        "topk_source": metric.get("oer_oracle_source", "scalar_oer_topk_projection"),
        "oer_topk_scalar_indices": _tensor_payload_1d(metric["oer_topk_indices"]),
        "valid_token_count": int(metric["valid_token_count"]),
        "layout_info": layout_info_to_public_fields(dict(layout_info or {})),
        "debug_summary": {
            "projection_rel_l2_error": _tensor_to_float(metric["projection_rel_l2_error"]),
            "projection_cosine_drift": _tensor_to_float(metric["projection_cosine_drift"]),
            "projection_mse": _tensor_to_float(metric["projection_mse"]),
            "projection_oer_shift": _tensor_to_float(metric["projection_oer_shift"]),
            "output_energy_distortion": _tensor_to_float(metric["output_energy_distortion"]),
        },
    }


def _probe_position_from_oracle_activation_position(position: str) -> str:
    name = str(position).strip().lower()
    if name in {"input", "input_activation", "linear_input"}:
        return "input_activation"
    if name in {"output", "output_activation", "linear_output", "linear_output_projection"}:
        return "linear_output_projection"
    raise ValueError(f"unsupported OER oracle activation_position: {position!r}")


def _load_target_oer_oracle(
    fp32_reference_cache: Mapping[str, Any],
    *,
    layer_idx: int,
    module_name: str,
    positions: Sequence[str],
) -> dict[str, Any]:
    reference = fp32_reference_cache.get("oer_oracle_reference")
    if not isinstance(reference, Mapping) or not bool(reference.get("required", False)):
        raise RuntimeError("oer_oracle_eval.pt is required but was not validated")
    oracle = reference.get("oracle")
    if not isinstance(oracle, Mapping):
        raise RuntimeError("validated OER oracle reference is missing its top-level shard manifest")
    oracle_path = Path(reference["path"])
    expected_sample_ids = _normalise_sample_ids(fp32_reference_cache.get("sample_id"))
    expected_count = len(expected_sample_ids)
    if expected_count <= 0:
        raise RuntimeError("validated FP32 reference has no effective sample_id rows")
    requested_positions = tuple(str(item) for item in positions)
    rows_by_position: dict[str, dict[str, Any]] = {
        position: {
            "sample_id": [],
            "topk_scalar_indices": [],
            "oer_topk_token_indices_aux": [],
            "topk_ratio": [],
            "layout_debug": [],
        }
        for position in requested_positions
    }
    target_name = _target_name(layer_idx, module_name)

    for shard in iter_fp32_oer_oracle_shards(oracle, oracle_path, layer_idx=int(layer_idx)):
        shard_sample_ids = [str(item) for item in shard.get("sample_id", [])]
        row_count = len(shard_sample_ids)
        if row_count <= 0:
            continue
        layer_values = torch.as_tensor(shard.get("layer_idx"), dtype=torch.long).reshape(-1)
        module_values = [str(item) for item in shard.get("module_name", [])]
        position_values = [str(item) for item in shard.get("activation_position", [])]
        scalar_values = shard.get("topk_scalar_indices")
        token_values = shard.get("oer_topk_token_indices_aux", shard.get("topk_token_indices"))
        ratio_values = torch.as_tensor(shard.get("topk_ratio"), dtype=torch.float32).reshape(-1)
        if len(module_values) != row_count or len(position_values) != row_count or int(layer_values.numel()) != row_count:
            raise RuntimeError(f"{target_name}: OER shard row metadata length mismatch")
        if not isinstance(scalar_values, Sequence) or isinstance(scalar_values, (str, bytes)) or len(scalar_values) != row_count:
            raise RuntimeError(f"{target_name}: OER shard topk_scalar_indices length mismatch")
        if not isinstance(token_values, Sequence) or isinstance(token_values, (str, bytes)) or len(token_values) != row_count:
            raise RuntimeError(f"{target_name}: OER shard oer_topk_token_indices_aux length mismatch")
        if int(ratio_values.numel()) != row_count:
            raise RuntimeError(f"{target_name}: OER shard topk_ratio length mismatch")
        for row_idx, sample_id in enumerate(shard_sample_ids):
            if int(layer_values[row_idx].item()) != int(layer_idx):
                continue
            if module_values[row_idx] != str(module_name):
                continue
            probe_position = _probe_position_from_oracle_activation_position(position_values[row_idx])
            if probe_position not in rows_by_position:
                continue
            current = rows_by_position[probe_position]
            if len(current["sample_id"]) >= expected_count:
                continue
            current["sample_id"].append(str(sample_id))
            current["topk_scalar_indices"].append(torch.as_tensor(scalar_values[row_idx], dtype=torch.long).cpu().reshape(-1).contiguous())
            current["oer_topk_token_indices_aux"].append(torch.as_tensor(token_values[row_idx], dtype=torch.long).cpu().reshape(-1).contiguous())
            current["topk_ratio"].append(float(ratio_values[row_idx].item()))
            if not current["layout_debug"]:
                activation_shape = shard.get("activation_shape", [])[row_idx] if isinstance(shard.get("activation_shape"), Sequence) else ""
                current["layout_debug"].append(
                    {
                        "activation_position": str(position_values[row_idx]),
                        "activation_shape": activation_shape,
                        "original_layout": shard.get("original_layout", [""] * row_count)[row_idx],
                        "original_shape": shard.get("original_shape", [""] * row_count)[row_idx],
                        "canonical_layout": shard.get("canonical_layout", [""] * row_count)[row_idx],
                        "canonical_shape": shard.get("canonical_shape", [""] * row_count)[row_idx],
                        "layout_inference_source": shard.get("layout_inference_source", [""] * row_count)[row_idx],
                        "padding_mask_available": shard.get("padding_mask_available", [False] * row_count)[row_idx],
                        "padding_mask_aligned": shard.get("padding_mask_aligned", [False] * row_count)[row_idx],
                        "mask_status": shard.get("mask_status", [""] * row_count)[row_idx],
                    }
                )

    for position, current in rows_by_position.items():
        observed_sample_ids = [str(item) for item in current["sample_id"]]
        if observed_sample_ids != expected_sample_ids:
            first_mismatches = [
                {"index": idx, "oracle": observed, "core_eval": expected}
                for idx, (observed, expected) in enumerate(zip(observed_sample_ids, expected_sample_ids))
                if observed != expected
            ][:8]
            raise RuntimeError(
                f"{target_name} {position}: OER oracle sample_id order must match core_eval.pt effective sample order; "
                f"oracle_count={len(observed_sample_ids)}, core_count={len(expected_sample_ids)}, first_mismatches={first_mismatches}"
            )
        if len(set(observed_sample_ids)) != len(observed_sample_ids):
            raise RuntimeError(f"{target_name} {position}: OER oracle contains duplicate sample_id rows")
        ratios = [float(item) for item in current["topk_ratio"]]
        if ratios and any(abs(float(item) - float(ratios[0])) > 1e-12 for item in ratios):
            raise RuntimeError(f"{target_name} {position}: OER oracle topk_ratio changed across samples")
        current["topk_ratio"] = float(ratios[0]) if ratios else float(reference.get("topk_ratio", 0.0))
        current["layout_debug"] = current["layout_debug"][0] if current["layout_debug"] else {}

    return {
        "oracle_source": "oer_oracle_eval.pt",
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "target_name": target_name,
        "sample_id": expected_sample_ids,
        "sample_order_hash": _sample_order_hash(expected_sample_ids),
        "topk_ratio": float(reference.get("topk_ratio", 0.0)),
        "positions": rows_by_position,
    }


def _target_oracle_position_rows(target_oer_oracle: Mapping[str, Any], position: str) -> Mapping[str, Any]:
    positions = target_oer_oracle.get("positions", {})
    if not isinstance(positions, Mapping) or position not in positions:
        raise KeyError(f"OER oracle for {target_oer_oracle.get('target_name', '<unknown>')} missing position={position!r}")
    rows = positions[position]
    if not isinstance(rows, Mapping):
        raise TypeError(f"OER oracle position payload must be a mapping, got {type(rows)!r}")
    return rows

def _run_quant_danger_shard(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    position: str,
    qparams: Mapping[str, Any],
    qparams_path: Path,
    eval_loader: DataLoader,
    target_oer_oracle: Mapping[str, Any],
    protocol_version: str,
    experiment_signature: str,
) -> tuple[list[dict[str, Any]], Path, Path, Path]:
    device = _device(cfg)
    sample_shard = _sample_shard_path(cfg, layer_idx, module_name, quant_mode, position)
    topk_path = _topk_compact_path(cfg, layer_idx, module_name, quant_mode, position)
    summary_partial = _summary_partial_path(cfg, layer_idx, module_name, quant_mode, position)
    model = restore_clean_model_from_checkpoint(cfg).to(device)
    target_name = _target_name(layer_idx, module_name)
    weight_qparams = qparams["weight"] if _mode_uses_weight(quant_mode) else None
    activation_qparams = qparams["activation"] if _mode_uses_activation(quant_mode) else None
    quant_module = replace_ffn_module_with_quant(
        model,
        int(layer_idx),
        module_name,
        _quant_cfg(cfg),
        quant_mode=quant_mode,
        weight_qparams=weight_qparams,
        activation_qparams=activation_qparams,
        require_calibrated_qparams=True,
        enable_fake_quant=False,
    )
    quant_module.store_oer_activation = True
    if position == "input_activation":
        quant_module.set_integer_grid_capture(True, save_device=str(_setting_get(cfg, "INTEGER_GRID_SAVE_DEVICE", "cpu")))
    controller = make_single_ffn_quant_controller(
        model,
        quant_module,
        layer_idx=int(layer_idx),
        module_name=module_name,
        quant_config=_quant_cfg(cfg),
        quant_mode=quant_mode,
    )
    metadata_lookup = _load_sample_metadata_lookup(cfg)
    oracle_position_rows = _target_oracle_position_rows(target_oer_oracle, position)
    expected_sample_ids = [str(item) for item in target_oer_oracle.get("sample_id", [])]
    rows: list[dict[str, Any]] = []
    topk_records: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    batch_count = 0
    oracle_cursor = 0
    try:
        controller.qparams_frozen = True
        controller.prepare_quant_forward()
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                metadata = _batch_metadata(batch, metadata_lookup, cfg)
                batch_sample_ids = [str(item) for item in metadata["sample_id"]]
                batch_size = len(batch_sample_ids)
                expected_batch_sample_ids = expected_sample_ids[oracle_cursor : oracle_cursor + batch_size]
                if batch_sample_ids != expected_batch_sample_ids:
                    first_mismatches = [
                        {"index": oracle_cursor + idx, "loader": observed, "oracle": expected}
                        for idx, (observed, expected) in enumerate(zip(batch_sample_ids, expected_batch_sample_ids))
                        if observed != expected
                    ][:8]
                    raise RuntimeError(
                        f"{target_name} {_mode_label(quant_mode)} {position}: eval loader sample_id order does not match FP32 OER oracle; "
                        f"loader_batch={batch_sample_ids[:5]}, oracle_batch={expected_batch_sample_ids[:5]}, first_mismatches={first_mismatches}"
                    )
                sample_ids.extend(batch_sample_ids)
                moved = _move_batch_to_device(batch, device)
                with _autocast_context(cfg, device):
                    output, _mask = _forward_model(model, moved, cfg)
                del output
                valid_mask = getattr(quant_module, "last_activation_valid_mask_for_oer", None)
                layout_info = getattr(quant_module, "last_activation_layout_info_for_oer", {}) or {}
                if position == "input_activation":
                    if not _mode_uses_activation(quant_mode):
                        raise RuntimeError(f"{target_name} {quant_mode}: input_activation danger metrics require activation quantization")
                    fp32_activation, quant_activation = quant_module.get_oer_activation_tensors()
                    integer_grid = quant_module.get_integer_grid_diagnostics()
                    if fp32_activation is None or quant_activation is None or integer_grid is None:
                        raise RuntimeError(f"{target_name} {quant_mode}: missing direct QuantLinear integer-grid diagnostics")
                    for sample_idx in range(batch_size):
                        sample_meta = _metadata_for_sample(metadata, sample_idx)
                        sample_valid = None if valid_mask is None else valid_mask[sample_idx : sample_idx + 1]
                        oracle_index = oracle_cursor + sample_idx
                        metric = compute_direct_integer_grid_danger_metrics(
                            fp32_activation[sample_idx : sample_idx + 1],
                            quant_activation[sample_idx : sample_idx + 1],
                            _slice_grid(integer_grid, sample_idx),
                            topk_ratio=_topk_ratio(cfg),
                            topk_token_indices=oracle_position_rows["oer_topk_token_indices_aux"][oracle_index],
                            oer_topk_indices=oracle_position_rows["topk_scalar_indices"][oracle_index],
                            valid_token_mask=sample_valid,
                            eps=DEFAULT_EPS,
                        )
                        rows.append(_activation_metric_row(
                            cfg=cfg,
                            sample_meta=sample_meta,
                            layer_idx=layer_idx,
                            module_name=module_name,
                            quant_mode=quant_mode,
                            fp32_sample=fp32_activation[sample_idx : sample_idx + 1],
                            valid_mask=sample_valid,
                            metric=metric,
                            layout_info=layout_info,
                        ))
                        topk_records.append(_input_topk_record(
                            sample_meta=sample_meta,
                            layer_idx=layer_idx,
                            module_name=module_name,
                            quant_mode=quant_mode,
                            metric=metric,
                            layout_info=layout_info,
                        ))
                else:
                    fp32_projection, quant_projection = quant_module.get_oer_projection_tensors()
                    if fp32_projection is None or quant_projection is None:
                        raise RuntimeError(f"{target_name} {quant_mode}: missing projection tensors")
                    for sample_idx in range(batch_size):
                        sample_meta = _metadata_for_sample(metadata, sample_idx)
                        sample_valid = None if valid_mask is None else valid_mask[sample_idx : sample_idx + 1]
                        oracle_index = oracle_cursor + sample_idx
                        metric = compute_projection_damage_metrics(
                            fp32_projection[sample_idx : sample_idx + 1],
                            quant_projection[sample_idx : sample_idx + 1],
                            topk_ratio=_topk_ratio(cfg),
                            oer_topk_indices=oracle_position_rows["topk_scalar_indices"][oracle_index],
                            valid_token_mask=sample_valid,
                            eps=DEFAULT_EPS,
                        )
                        rows.append(_projection_metric_row(
                            cfg=cfg,
                            sample_meta=sample_meta,
                            layer_idx=layer_idx,
                            module_name=module_name,
                            quant_mode=quant_mode,
                            fp32_sample=fp32_projection[sample_idx : sample_idx + 1],
                            valid_mask=sample_valid,
                            metric=metric,
                            layout_info=layout_info,
                        ))
                        topk_records.append(_projection_topk_record(
                            sample_meta=sample_meta,
                            layer_idx=layer_idx,
                            module_name=module_name,
                            quant_mode=quant_mode,
                            metric=metric,
                            layout_info=layout_info,
                        ))
                batch_count += 1
                oracle_cursor += batch_size
                quant_module.clear_oer_activation_tensors()
                del moved
    finally:
        controller.remove_padding_mask_hooks()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if batch_count <= 0:
        raise RuntimeError("eval loader produced no batches")
    if oracle_cursor != len(expected_sample_ids):
        raise RuntimeError(
            f"{target_name} {_mode_label(quant_mode)} {position}: eval loader consumed {oracle_cursor} samples, "
            f"but FP32 OER oracle expected {len(expected_sample_ids)}"
        )
    _write_rows_atomic(sample_shard, rows)
    topk_path.parent.mkdir(parents=True, exist_ok=True)
    topk_tmp = topk_path.with_name(f"{topk_path.name}.tmp")
    torch.save({
        "protocol_version": protocol_version,
        "experiment_signature": experiment_signature,
        "target": {"layer_idx": int(layer_idx), "module_name": str(module_name)},
        "quant_mode": _mode_label(quant_mode),
        "position": position,
        "topk_source": "fp32_oer_oracle.oer_topk_token_indices_aux" if position == "input_activation" else "fp32_oer_oracle.topk_scalar_indices",
        "records": topk_records,
    }, topk_tmp)
    os.replace(topk_tmp, topk_path)
    _write_json_atomic(summary_partial, {
        "protocol_version": protocol_version,
        "experiment_signature": experiment_signature,
        "target": {"layer_idx": int(layer_idx), "module_name": str(module_name)},
        "quant_mode": _mode_label(quant_mode),
        "position": position,
        "num_batches": int(batch_count),
        "num_samples": int(len(sample_ids)),
        "num_rows": int(len(rows)),
        "sample_order_hash": _sample_order_hash(sample_ids),
        "oer_oracle_sample_order_hash": target_oer_oracle.get("sample_order_hash"),
        "oer_oracle_source": target_oer_oracle.get("oracle_source"),
        "qparams_path": str(qparams_path),
        "qparams_hash": _sha256_file(qparams_path),
        "save_full_activation": False,
    })
    _write_shard_done(
        cfg,
        layer_idx,
        module_name,
        quant_mode,
        position,
        protocol_version=protocol_version,
        experiment_signature=experiment_signature,
        sample_shard=sample_shard,
        topk_compact=topk_path,
        summary_partial=summary_partial,
        qparams_path=qparams_path,
        sample_ids=sample_ids,
        row_count=len(rows),
    )
    return rows, sample_shard, topk_path, summary_partial

def _cleanup_incomplete_shard(cfg: Any, layer_idx: int, module_name: str, quant_mode: str, position: str) -> None:
    candidates = [
        _sample_shard_path(cfg, layer_idx, module_name, quant_mode, position),
        _topk_compact_path(cfg, layer_idx, module_name, quant_mode, position),
        _summary_partial_path(cfg, layer_idx, module_name, quant_mode, position),
        _done_marker_path(cfg, layer_idx, module_name, quant_mode, position),
    ]
    for path in candidates:
        tmp_path = path.with_name(f"{path.name}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        if path.exists():
            path.unlink()


def run_quant_danger_zone_probe(
    cfg: Any,
    *,
    model: nn.Module | None = None,
    loader: DataLoader | None = None,
    target_modules: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Run the sharded FFN Quant Danger Zone experiment."""

    model_was_provided = model is not None
    loader_was_provided = loader is not None
    enum_model = model if model is not None else restore_clean_model_from_checkpoint(cfg)
    try:
        active_target_modules, target_modules_source = _resolve_target_modules(enum_model, target_modules, cfg)
    finally:
        if model is None:
            del enum_model
    output_dir = _result_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv_path = output_dir / SAMPLE_OUTPUT_FILENAME
    summary_csv_path = output_dir / SUMMARY_OUTPUT_FILENAME
    comparison_summary_csv_path = output_dir / COMPARISON_SUMMARY_OUTPUT_FILENAME
    correlation_csv_path = output_dir / CORRELATION_OUTPUT_FILENAME
    topk_indices_path = output_dir / TOPK_INDICES_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    protocol_version = PROTOCOL_VERSION
    quant_modes = _danger_quant_modes(cfg)
    positions = _danger_positions(cfg)
    fp32_reference_cache = _build_required_reference_cache(cfg, active_target_modules)
    reference_manifest = fp32_reference_cache.get("manifest", {})
    reference_state = fp32_reference_cache.get("reference_state", reference_manifest)
    signature_payload = _experiment_signature_payload(
        cfg,
        active_target_modules,
        reference_state if isinstance(reference_state, Mapping) else {},
    )
    experiment_signature = _experiment_signature(signature_payload)
    eval_loader = loader if loader is not None else _build_eval_loader(cfg)
    calibration_loader = _build_calibration_loader(cfg)

    all_rows: list[dict[str, Any]] = []
    topk_shards: list[str] = []
    partial_summaries: list[str] = []
    completed_shards = 0
    rerun_shards = 0
    skipped_reference_shards = 0

    for layer_idx, module_name in active_target_modules:
        target_oer_oracle = _load_target_oer_oracle(
            fp32_reference_cache,
            layer_idx=int(layer_idx),
            module_name=module_name,
            positions=positions,
        )
        qparams, qparams_path = _load_or_create_qparams(
            cfg,
            int(layer_idx),
            module_name,
            calibration_loader,
            protocol_version=protocol_version,
            experiment_signature=experiment_signature,
            signature_payload=signature_payload,
        )
        qparams_hash = _sha256_file(qparams_path)
        for quant_mode in quant_modes:
            for position in positions:
                if position == "input_activation" and not _mode_uses_activation(quant_mode):
                    skipped_reference_shards += 1
                    continue
                sample_shard = _sample_shard_path(cfg, int(layer_idx), module_name, quant_mode, position)
                topk_path = _topk_compact_path(cfg, int(layer_idx), module_name, quant_mode, position)
                summary_partial = _summary_partial_path(cfg, int(layer_idx), module_name, quant_mode, position)
                if _shard_done_valid(
                    cfg,
                    int(layer_idx),
                    module_name,
                    quant_mode,
                    position,
                    protocol_version=protocol_version,
                    experiment_signature=experiment_signature,
                    qparams_hash=qparams_hash,
                ):
                    rows = _read_csv_rows(sample_shard)
                    all_rows.extend(rows)
                    topk_shards.append(str(topk_path))
                    partial_summaries.append(str(summary_partial))
                    completed_shards += 1
                    print(f"[resume] skip completed danger-zone shard {_target_name(layer_idx, module_name)} {_mode_label(quant_mode)} {position}")
                    continue
                _cleanup_incomplete_shard(cfg, int(layer_idx), module_name, quant_mode, position)
                rows, sample_shard, topk_path, summary_partial = _run_quant_danger_shard(
                    cfg,
                    layer_idx=int(layer_idx),
                    module_name=module_name,
                    quant_mode=quant_mode,
                    position=position,
                    qparams=qparams,
                    qparams_path=qparams_path,
                    eval_loader=eval_loader,
                    target_oer_oracle=target_oer_oracle,
                    protocol_version=protocol_version,
                    experiment_signature=experiment_signature,
                )
                all_rows.extend(rows)
                topk_shards.append(str(topk_path))
                partial_summaries.append(str(summary_partial))
                rerun_shards += 1

    if not all_rows:
        raise RuntimeError("quant danger-zone produced no sample rows")
    _write_rows_atomic(sample_csv_path, all_rows)
    topk_tmp = topk_indices_path.with_name(f"{topk_indices_path.name}.tmp")
    torch.save(
        {
            "protocol_version": protocol_version,
            "experiment_signature": experiment_signature,
            "fp32_reference_manifest": _json_safe(reference_manifest),
            "topk_shards": topk_shards,
            "note": "Per-shard compact top-k records only; no full activation/projection tensors are stored.",
        },
        topk_tmp,
    )
    os.replace(topk_tmp, topk_indices_path)
    sample_table, summary_table, comparison_table, correlation_table = build_summary_table(sample_csv_path, output_dir=output_dir, save=True)
    plot_paths = plot_all_quant_danger_zone_figures(
        summary_table,
        output_dir,
        comparison_df=comparison_table,
        correlation_df=correlation_table,
    )
    _write_json_atomic(
        manifest_path,
        {
            "experiment": "ffn_quant_danger_zone_probe",
            "protocol_version": protocol_version,
            "experiment_signature": experiment_signature,
            "experiment_signature_payload": signature_payload,
            "model_source": "provided_model_for_target_enumeration" if model_was_provided else "restore_clean_model_from_checkpoint_per_shard",
            "loader_source": "provided_eval_loader" if loader_was_provided else "AudioInputDataset",
            "checkpoint_path": _optional_checkpoint_path(cfg),
            "fp32_reference_core_path": _fp32_core_eval_path(cfg),
            "fp32_reference_oer_oracle_path": _fp32_oer_oracle_path(cfg),
            "fp32_reference_manifest": _json_safe(reference_manifest),
            "quant_modes": list(quant_modes),
            "optional_reference_modes": list(OPTIONAL_REFERENCE_MODES),
            "positions": list(positions),
            "target_modules_source": str(target_modules_source),
            "target_module_count": int(len(active_target_modules)),
            "target_modules": _target_modules_manifest(active_target_modules),
            "resume_protocol": {
                "unit": "split_plus_quant_mode_plus_activation_granularity_plus_target_plus_position_plus_sample_shard_id",
                "shard_by_position": True,
                "done_marker_requires": ["protocol_version", "experiment_signature", "sample_order_hash", "qparams_hash"],
                "incomplete_shard_policy": "delete_tmp_and_partial_outputs_then_rerun",
                "parallel_policy": "do_not_parallelize_same_result_dir; merge from shards in one process",
            },
            "metric_scope": {
                "input_activation": "formal activation integer-grid danger metrics from direct QuantLinear forward grid",
                "linear_output_projection": "projection damage detail only; no activation q_int clip/near-zero semantics",
                "w4_only": "skipped for input_activation by default unless used as projection reference",
            },
            "integer_grid_source": "direct_quantlinear_forward_grid",
            "save_full_activation": False,
            "counts": {
                "num_rows": int(len(all_rows)),
                "completed_shards_loaded": int(completed_shards),
                "rerun_shards": int(rerun_shards),
                "skipped_reference_shards": int(skipped_reference_shards),
                "num_topk_shards": int(len(topk_shards)),
            },
            "outputs": {
                "sample_csv": sample_csv_path,
                "summary_csv": summary_csv_path,
                "comparison_summary_csv": comparison_summary_csv_path,
                "correlation_v2_csv": correlation_csv_path,
                "topk_index_manifest": topk_indices_path,
                "topk_compact_shards": topk_shards,
                "summary_partials": partial_summaries,
                "manifest": manifest_path,
                "plots": plot_paths,
            },
            **group_protocol_metadata(tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}"),
        },
    )
    return {
        "output_dir": output_dir,
        "sample_csv_path": sample_csv_path,
        "summary_csv_path": summary_csv_path,
        "comparison_summary_csv_path": comparison_summary_csv_path,
        "correlation_csv_path": correlation_csv_path,
        "topk_indices_path": topk_indices_path,
        "manifest_path": manifest_path,
        "fp32_reference_core_path": _fp32_core_eval_path(cfg),
        "fp32_reference_oer_oracle_path": _fp32_oer_oracle_path(cfg),
        "plot_paths": plot_paths,
        "num_rows": int(len(all_rows)),
        "num_targets": int(len(active_target_modules)),
        "num_quant_modes": int(len(quant_modes)),
        "num_positions": int(len(positions)),
        "completed_shards_loaded": int(completed_shards),
        "rerun_shards": int(rerun_shards),
        "skipped_reference_shards": int(skipped_reference_shards),
        "target_modules": active_target_modules,
        "sample_table_shape": tuple(sample_table.shape),
        "summary_table_shape": tuple(summary_table.shape),
        "comparison_table_shape": tuple(comparison_table.shape),
        "correlation_table_shape": tuple(correlation_table.shape),
    }

def main(cfg: Any, target_modules: Sequence[Any] | None = None) -> dict[str, Any]:
    seed_everything = _cfg_get(cfg, "seed_everything", None)
    if callable(seed_everything):
        seed_everything(_seed(cfg))
    return run_quant_danger_zone_probe(cfg, target_modules=target_modules)


if __name__ == "__main__":
    import config as project_config

    summary = main(project_config)
    print(f"FFN 量化危险区探针完成：{summary['sample_csv_path']}")


__all__ = [
    "ACTIVATION_POSITIONS",
    "MANIFEST_FILENAME",
    "TOPK_INDICES_FILENAME",
    "RESULT_DIR_NAME",
    "collect_quant_danger_zone_metrics",
    "main",
    "run_quant_danger_zone_probe",
]



