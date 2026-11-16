from __future__ import annotations

import csv
import hashlib
import json
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
    from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from src.metrics.activation_distribution_metrics import (
        activation_distribution_metric_metadata,
        activation_distribution_metric_names,
        compute_activation_distribution_metrics,
    )
    from src.plots.activation_distribution_plotter import (
        plot_activation_histograms,
        plot_adaptive_clipping_rate_comparison,
        plot_adaptive_oer_retention_comparison,
        plot_fixed_clipping_rate_comparison,
        plot_fixed_oer_retention_comparison,
        plot_activation_range_bar,
    )
    from src.quant.ffn_quant_injector import restore_clean_model_from_checkpoint
    from src.quant.observer import ObserverConfig, TensorDistributionObserver
    from src.utils.activation_hook import ActivationHookManager
    from src.utils.group_utils import ANALYSIS_GROUPS, compute_tail_and_few_gaps, analysis_group_mask, group_protocol_metadata, read_canonical_bucket, tail_included
    from src.utils.layout_utils import (
        canonicalize_activation_btc,
        canonicalize_already_filtered_tokens_btc,
        canonicalize_padding_mask_bt,
        layout_info_to_public_fields,
    )
except ImportError:  # pragma: no cover - support package-relative debugging.
    from ..data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from ..metrics.activation_distribution_metrics import (
        activation_distribution_metric_metadata,
        activation_distribution_metric_names,
        compute_activation_distribution_metrics,
    )
    from ..plots.activation_distribution_plotter import (
        plot_activation_histograms,
        plot_adaptive_clipping_rate_comparison,
        plot_adaptive_oer_retention_comparison,
        plot_fixed_clipping_rate_comparison,
        plot_fixed_oer_retention_comparison,
        plot_activation_range_bar,
    )
    from ..quant.ffn_quant_injector import restore_clean_model_from_checkpoint
    from ..quant.observer import ObserverConfig, TensorDistributionObserver
    from ..utils.activation_hook import ActivationHookManager
    from ..utils.group_utils import ANALYSIS_GROUPS, compute_tail_and_few_gaps, analysis_group_mask, group_protocol_metadata, read_canonical_bucket, tail_included
    from ..utils.layout_utils import (
        canonicalize_activation_btc,
        canonicalize_already_filtered_tokens_btc,
        canonicalize_padding_mask_bt,
        layout_info_to_public_fields,
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

CSV_FILENAME = "ffn_activation_distribution_probe.csv"
PLOT_SUMMARY_CSV_FILENAME = "ffn_activation_distribution_plot_summary.csv"
MANIFEST_FILENAME = "ffn_activation_distribution_manifest.json"
RESULT_DIR_NAME = "ffn_activation_distribution_probe"
ACTIVATION_KINDS = ("input_activation", "output_activation")
PROTOCOL_VERSION = "ffn_activation_distribution_probe.v3"
FP32_REFERENCE_CORE_FILENAME = "core_eval.pt"
FP32_REFERENCE_OER_ORACLE_FILENAME = "oer_oracle_eval.pt"
FP32_CORE_STORAGE_FORMAT = "sample_sharded_v1"
FP32_CORE_CACHE_TYPE = "fp32_core_cache"
FP32_CORE_STAGE = "quant_injection_pre_fp32_reference"

CSV_COLUMNS = [
    "row_type",
    "sample_id",
    "label_id",
    "bucket",
    "shot_group",
    "tail_included",
    "analysis_group",
    "group_protocol_version",
    "main_comparison_group",
    "supplementary_extreme_group",
    "metric_name",
    "metric_value",
    "metric_direction",
    "aggregation_unit",
    "aggregation_fn",
    "valid_count",
    "total_count",
    "invalid_count",
    "valid_ratio",
    "many_value",
    "medium_value",
    "few_value",
    "tail_value",
    "tail_vs_many_gap",
    "few_vs_many_gap",
    "sample_count",
    "class_count",
    "valid_token_count",
    "total_token_count",
    "padding_token_count",
    "padding_token_ratio",
    "many_sample_count",
    "medium_sample_count",
    "few_sample_count",
    "tail_sample_count",
    "many_class_count",
    "medium_class_count",
    "few_class_count",
    "tail_class_count",
    "audio_path",
    "layer_idx",
    "module_name",
    "activation_kind",
    "target_name",
    "activation_shape",
    "original_layout",
    "original_shape",
    "canonical_layout",
    "canonical_shape",
    "layout_inference_source",
    "padding_mask_available",
    "padding_mask_aligned",
    "mask_status",
    "valid_token_policy",
    "qmin",
    "qmax",
    "percentile",
    "topk_ratio",
    "max_abs",
    "p99",
    "p999",
    "mean",
    "std",
    "kurtosis",
    "heavy_tail_ratio",
    "max_abs_over_p99",
    "p999_over_p99",
    "sparsity",
    "sparsity_ratio",
    "near_zero_rate",
    "near_zero_ratio",
    "outlier_ratio",
    "outlier_energy",
    "outlier_energy_ratio",
    "scale_mean",
    "scale_median",
    "scale_p95",
    "scale_max",
    "clipping_rate",
    "oer_retention",
    "adaptive_per_tensor_scale",
    "adaptive_per_channel_scale_mean",
    "adaptive_per_channel_scale_std",
    "adaptive_scale_ratio_mean",
    "adaptive_scale_ratio_std",
    "adaptive_clipping_rate_per_tensor",
    "adaptive_clipping_rate_per_channel",
    "adaptive_oer_retention_per_tensor",
    "adaptive_oer_retention_per_channel",
    "fixed_per_tensor_scale",
    "fixed_per_channel_scale_mean",
    "fixed_per_channel_scale_std",
    "fixed_scale_ratio_mean",
    "fixed_scale_ratio_std",
    "fixed_clipping_rate_per_tensor",
    "fixed_clipping_rate_per_channel",
    "fixed_oer_retention_per_tensor",
    "fixed_oer_retention_per_channel",
    "hist_edges",
    "hist_counts",
    "hist_density",
    "hist_total_count",
]

SCALAR_METRIC_COLUMNS = [
    "max_abs",
    "p99",
    "p999",
    "mean",
    "std",
    "kurtosis",
    "heavy_tail_ratio",
    "max_abs_over_p99",
    "p999_over_p99",
    "sparsity",
    "sparsity_ratio",
    "near_zero_rate",
    "near_zero_ratio",
    "outlier_ratio",
    "outlier_energy",
    "outlier_energy_ratio",
    "scale_mean",
    "scale_median",
    "scale_p95",
    "scale_max",
    "clipping_rate",
    "oer_retention",
    "adaptive_per_tensor_scale",
    "adaptive_per_channel_scale_mean",
    "adaptive_per_channel_scale_std",
    "adaptive_scale_ratio_mean",
    "adaptive_scale_ratio_std",
    "adaptive_clipping_rate_per_tensor",
    "adaptive_clipping_rate_per_channel",
    "adaptive_oer_retention_per_tensor",
    "adaptive_oer_retention_per_channel",
    "fixed_per_tensor_scale",
    "fixed_per_channel_scale_mean",
    "fixed_per_channel_scale_std",
    "fixed_scale_ratio_mean",
    "fixed_scale_ratio_std",
    "fixed_clipping_rate_per_tensor",
    "fixed_clipping_rate_per_channel",
    "fixed_oer_retention_per_tensor",
    "fixed_oer_retention_per_channel",
]

RANGE_DIAGNOSTIC_GAP_METRICS = {
    "max_abs",
    "p99",
    "p999",
    "heavy_tail_ratio",
    "max_abs_over_p99",
    "p999_over_p99",
}


_GROUP_PROTOCOL_METADATA = group_protocol_metadata(
    tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
)
GROUP_PROTOCOL_VERSION = int(_GROUP_PROTOCOL_METADATA["group_protocol_version"])
MAIN_COMPARISON_GROUP = "tail"
SUPPLEMENTARY_EXTREME_GROUP = "few"
HISTOGRAM_METRIC_NAME = "histogram_density"
SUMMARY_VALUE_COLUMNS = ("many_value", "medium_value", "few_value", "tail_value")
GAP_COLUMNS = ("tail_vs_many_gap", "few_vs_many_gap")
GROUP_SAMPLE_COUNT_COLUMNS = (
    "many_sample_count",
    "medium_sample_count",
    "few_sample_count",
    "tail_sample_count",
)
GROUP_CLASS_COUNT_COLUMNS = (
    "many_class_count",
    "medium_class_count",
    "few_class_count",
    "tail_class_count",
)


def _blank_protocol_metric_fields() -> dict[str, Any]:
    return {
        "group_protocol_version": GROUP_PROTOCOL_VERSION,
        "main_comparison_group": MAIN_COMPARISON_GROUP,
        "supplementary_extreme_group": SUPPLEMENTARY_EXTREME_GROUP,
        "metric_name": "",
        "metric_value": "",
        "metric_direction": "",
        "aggregation_unit": "",
        "aggregation_fn": "",
        "valid_count": "",
        "total_count": "",
        "invalid_count": "",
        "valid_ratio": "",
        "many_value": "",
        "medium_value": "",
        "few_value": "",
        "tail_value": "",
        "tail_vs_many_gap": "",
        "few_vs_many_gap": "",
        "sample_count": "",
        "class_count": "",
        "valid_token_count": "",
        "total_token_count": "",
        "padding_token_count": "",
        "padding_token_ratio": "",
        "many_sample_count": "",
        "medium_sample_count": "",
        "few_sample_count": "",
        "tail_sample_count": "",
        "many_class_count": "",
        "medium_class_count": "",
        "few_class_count": "",
        "tail_class_count": "",
    }


def _empty_csv_row() -> dict[str, Any]:
    return {key: "" for key in CSV_COLUMNS}


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
    value = _cfg_get(cfg, "FFN_ACTIVATION_DISTRIBUTION_PROBE", {})
    return value if isinstance(value, Mapping) else {}


def _legacy_distribution_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "ACTIVATION_DISTRIBUTION", {})
    return value if isinstance(value, Mapping) else {}


def _probe_get(cfg: Any, key: str, default: Any = None) -> Any:
    probe = _probe_cfg(cfg)
    if key in probe:
        return probe[key]
    legacy = _legacy_distribution_cfg(cfg)
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
    configured = _probe_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / RESULT_DIR_NAME


def _device(cfg: Any) -> torch.device:
    configured = _probe_get(cfg, "DEVICE", None)
    if configured is None:
        configured = _cfg_get(cfg, "DEVICE", None)
    if configured is None:
        raise AttributeError("ffn_activation_distribution_probe requires DEVICE from config.py.")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py DEVICE={configured!r}, but CUDA is not available.")
    return device


def _seed(cfg: Any) -> int:
    return int(_probe_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


def _batch_size(cfg: Any) -> int:
    return int(_probe_get(cfg, "BATCH_SIZE", 4))


def _num_workers(cfg: Any) -> int:
    return int(_probe_get(cfg, "NUM_WORKERS", 0))


def _pin_memory(cfg: Any) -> bool:
    return bool(_probe_get(cfg, "PIN_MEMORY", _device(cfg).type == "cuda"))


def _persistent_workers(cfg: Any) -> bool:
    return bool(_probe_get(cfg, "PERSISTENT_WORKERS", _num_workers(cfg) > 0))


def _prefetch_factor(cfg: Any) -> int | None:
    value = _probe_get(cfg, "PREFETCH_FACTOR", 2)
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"PREFETCH_FACTOR must be positive or None, got {value}")
    return value


def _hist_bins(cfg: Any) -> int:
    value = int(_probe_get(cfg, "HIST_BINS", _probe_get(cfg, "HISTOGRAM_BINS", 128)))
    if value <= 0:
        raise ValueError(f"HIST_BINS must be positive, got {value}")
    return value


def _percentile(cfg: Any) -> float:
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    if not isinstance(activation_cfg, Mapping):
        activation_cfg = {}
    return float(_probe_get(cfg, "PERCENTILE", activation_cfg.get("PERCENTILE", 0.999)))


def _topk_ratio(cfg: Any) -> float:
    return float(
        _probe_get(
            cfg,
            "TOPK_RATIO",
            _metrics_cfg(cfg).get("SCALAR_OER_TOP_RATIO", _cfg_get(cfg, "OER_SCALAR_TOP_RATIO", 0.001)),
        )
    )


def _qrange(cfg: Any) -> tuple[int, int]:
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    if not isinstance(activation_cfg, Mapping):
        activation_cfg = {}
    qmin = int(_probe_get(cfg, "QMIN", activation_cfg.get("QMIN", -7)))
    qmax = int(_probe_get(cfg, "QMAX", activation_cfg.get("QMAX", 7)))
    if qmin >= qmax:
        raise ValueError(f"qmin 蹇呴』灏忎簬 qmax锛屽綋鍓嶄负 {qmin}, {qmax}")
    if qmax <= 0:
        raise ValueError(f"qmax must be positive to build activation scale, got {qmax}")
    return qmin, qmax


def _percentile_for_observer(cfg: Any) -> float:
    value = float(_percentile(cfg))
    if value <= 1.0:
        value *= 100.0
    if not (0.0 < value <= 100.0):
        raise ValueError(f"percentile must be in (0, 1] or (0, 100], got {value}")
    return value


def _fixed_qparam_hist_bins(cfg: Any) -> int:
    observer_cfg = _cfg_get(cfg, "OBSERVER", {})
    if not isinstance(observer_cfg, Mapping):
        observer_cfg = {}
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    if not isinstance(activation_cfg, Mapping):
        activation_cfg = {}
    default_bins = observer_cfg.get("HISTOGRAM_BINS", activation_cfg.get("OBSERVER_BINS", _hist_bins(cfg)))
    value = int(_probe_get(cfg, "FIXED_QPARAM_HIST_BINS", _probe_get(cfg, "CALIBRATION_HIST_BINS", default_bins)))
    if value <= 1:
        raise ValueError(f"FIXED_QPARAM_HIST_BINS 蹇呴』澶т簬 1锛屽綋鍓嶄负 {value}")
    return value


def _fixed_qparam_target_chunk_size(cfg: Any) -> int:
    value = int(_probe_get(cfg, "FIXED_QPARAM_TARGET_CHUNK_SIZE", _probe_get(cfg, "CALIBRATION_TARGET_CHUNK_SIZE", 2)))
    if value <= 0:
        raise ValueError(f"FIXED_QPARAM_TARGET_CHUNK_SIZE must be positive, got {value}")
    return value


def _calibration_batch_size(cfg: Any) -> int:
    return int(_probe_get(cfg, "CALIBRATION_BATCH_SIZE", _batch_size(cfg)))


def _calibration_num_workers(cfg: Any) -> int:
    return int(_probe_get(cfg, "CALIBRATION_NUM_WORKERS", _num_workers(cfg)))


def _calibration_pin_memory(cfg: Any) -> bool:
    return bool(_probe_get(cfg, "CALIBRATION_PIN_MEMORY", _pin_memory(cfg)))


def _calibration_persistent_workers(cfg: Any) -> bool:
    return bool(_probe_get(cfg, "CALIBRATION_PERSISTENT_WORKERS", _calibration_num_workers(cfg) > 0))


def _calibration_prefetch_factor(cfg: Any) -> int | None:
    value = _probe_get(cfg, "CALIBRATION_PREFETCH_FACTOR", _prefetch_factor(cfg))
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"CALIBRATION_PREFETCH_FACTOR must be positive or None, got {value}")
    return value


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
    value = _probe_get(cfg, "EVAL_CSV", None)
    if value is not None:
        return Path(value)
    value = _nested_get(dataset_cfg, "EVAL_CSV", None)
    if value is not None:
        return Path(value)
    raise AttributeError("Cannot find eval CSV: set EVAL_CSV or CURRENT_DATASET['EVAL_CSV'].")


def _calibration_csv(cfg: Any) -> Path:
    dataset_cfg = _dataset_cfg(cfg)
    for key in ("CALIBRATION_CSV", "CALIB_CSV"):
        value = _probe_get(cfg, key, None)
        if value is not None:
            return Path(value)
    value = _nested_get(dataset_cfg, "CALIBRATION_CSV", None)
    if value is not None:
        return Path(value)
    raise AttributeError("Cannot find calibration CSV: set CALIBRATION_CSV or CURRENT_DATASET['CALIBRATION_CSV'].")


def _build_probe_loader(cfg: Any) -> DataLoader:
    override = _cfg_get(cfg, "build_activation_distribution_probe_loader", None)
    if callable(override):
        return override(cfg)

    dataset: Any = AudioInputDataset(
        _probe_csv(cfg),
        mode=str(_probe_get(cfg, "MODE", "eval")),
        dataset_root=_nested_get(_dataset_cfg(cfg), "ROOT", None),
        project_root=_cfg_get(cfg, "BASE_DIR", None),
        audio_config=_make_audio_config(cfg),
        global_seed=_seed(cfg),
        epoch=0,
        return_dict=True,
        require_label_id=True,
        include_audio_stats=False,
    )
    max_samples = _probe_get(cfg, "MAX_SAMPLES", None)
    if max_samples is not None:
        limit = min(int(max_samples), len(dataset))
        if limit <= 0:
            raise ValueError(f"MAX_SAMPLES must be positive, got {max_samples}")
        dataset = Subset(dataset, range(limit))

    num_workers = _num_workers(cfg)
    loader_kwargs: dict[str, Any] = {
        "batch_size": _batch_size(cfg),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": _pin_memory(cfg),
        "collate_fn": partial(audio_input_collate_fn, include_audio_stats=False),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = _persistent_workers(cfg)
        prefetch_factor = _prefetch_factor(cfg)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _build_calibration_loader(cfg: Any) -> DataLoader:
    override = _cfg_get(cfg, "build_activation_distribution_calibration_loader", None)
    if callable(override):
        return override(cfg)

    dataset: Any = AudioInputDataset(
        _calibration_csv(cfg),
        mode=str(_probe_get(cfg, "CALIBRATION_MODE", "calibration")),
        dataset_root=_nested_get(_dataset_cfg(cfg), "ROOT", None),
        project_root=_cfg_get(cfg, "BASE_DIR", None),
        audio_config=_make_audio_config(cfg),
        global_seed=_seed(cfg),
        epoch=0,
        return_dict=True,
        require_label_id=False,
        include_audio_stats=False,
    )
    max_samples = _probe_get(cfg, "CALIBRATION_MAX_SAMPLES", None)
    if max_samples is not None:
        limit = min(int(max_samples), len(dataset))
        if limit <= 0:
            raise ValueError(f"CALIBRATION_MAX_SAMPLES must be positive, got {max_samples}")
        dataset = Subset(dataset, range(limit))

    num_workers = _calibration_num_workers(cfg)
    loader_kwargs: dict[str, Any] = {
        "batch_size": _calibration_batch_size(cfg),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": _calibration_pin_memory(cfg),
        "collate_fn": partial(audio_input_collate_fn, include_audio_stats=False),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = _calibration_persistent_workers(cfg)
        prefetch_factor = _calibration_prefetch_factor(cfg)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


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
            raise TypeError("妯″瀷杈撳嚭 dict 涓病鏈夊彲鐢ㄧ殑 tensor 瀛楁")
        mask = output.get("padding_mask")
        return tensor, mask if isinstance(mask, Tensor) else None
    if isinstance(output, (tuple, list)) and output:
        tensor = output[0]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"妯″瀷杈撳嚭绗竴涓厓绱犲繀椤绘槸 Tensor锛屽綋鍓嶄负 {type(tensor)!r}")
        mask = output[1] if len(output) > 1 and isinstance(output[1], Tensor) else None
        return tensor, mask
    raise TypeError(f"涓嶆敮鎸佺殑妯″瀷杈撳嚭绫诲瀷: {type(output)!r}")


def _forward_model(model: nn.Module, batch: Mapping[str, Any], cfg: Any) -> tuple[Tensor, Tensor | None]:
    forward_fn = _cfg_get(cfg, "ffn_activation_distribution_forward", None)
    if callable(forward_fn):
        return _extract_tensor_and_mask(forward_fn(model, batch, cfg))

    waveform = batch["waveform"]
    padding_mask = batch.get("padding_mask")
    if hasattr(model, "extract_features"):
        return _extract_tensor_and_mask(model.extract_features(waveform, padding_mask=padding_mask))

    try:
        return _extract_tensor_and_mask(model(waveform, padding_mask=padding_mask))
    except TypeError:
        return _extract_tensor_and_mask(model(waveform))


def _autocast_context(cfg: Any, device: torch.device) -> Any:
    use_amp = bool(_nested_get(_cfg_get(cfg, "BEATS", {}), "USE_AMP", False))
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def _metadata_value_present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def _parse_bool_metadata(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{context}: tail_included must be bool-like, got {value!r}")


def _batch_sequence(batch: Mapping[str, Any], key: str, batch_size: int) -> list[Any] | None:
    if key not in batch:
        return None
    value = batch.get(key)
    if isinstance(value, Tensor):
        result = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
    else:
        result = [value] * int(batch_size)
    if len(result) != int(batch_size):
        raise ValueError(f"{key} length={len(result)} must match batch_size={batch_size}")
    return result


def _sample_id_from_metadata_row(row: Mapping[str, Any]) -> str | None:
    for key in ("sample_id", "audio_id", "file_id"):
        value = row.get(key)
        if _metadata_value_present(value):
            return str(value).strip()
    return None


def _full_dataset_csv(cfg: Any) -> Path | None:
    dataset_cfg = _dataset_cfg(cfg)
    for key in ("FULL_CSV", "FULL_DATASET_CSV"):
        value = _probe_get(cfg, key, None)
        if value is not None:
            return Path(value)
        value = _nested_get(dataset_cfg, key, None)
        if value is not None:
            return Path(value)
    return None


def _bucket_lookup_paths(cfg: Any) -> list[Path]:
    paths: list[Path] = []
    try:
        paths.append(_probe_csv(cfg))
    except AttributeError:
        pass
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


def _load_bucket_lookup(cfg: Any) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in _bucket_lookup_paths(cfg):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
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
                result = read_canonical_bucket(
                    bucket=[row.get("bucket")] if has_bucket else None,
                    shot_group=[row.get("shot_group")] if has_shot_group else None,
                    train_label_n=[row.get("train_label_n")] if has_train_label_n else None,
                    context=f"{path}:{row_idx}",
                    allow_shot_group_fallback=False,
                )
                bucket = result.bucket[0]
                computed_tail = bool(tail_included([bucket])[0])
                if has_tail_flag:
                    observed_tail = _parse_bool_metadata(row.get("tail_included"), context=f"{path}:{row_idx}")
                    if observed_tail != computed_tail:
                        raise ValueError(
                            f"{path}:{row_idx}: tail_included={observed_tail!r} does not match bucket={bucket!r}"
                        )
                previous = lookup.get(sample_id)
                if previous is not None and previous["bucket"] != bucket:
                    raise ValueError(
                        f"Conflicting bucket metadata for sample_id={sample_id!r}: "
                        f"{previous['bucket']!r} vs {bucket!r}"
                    )
                lookup[sample_id] = {
                    "bucket": bucket,
                    "shot_group": bucket,
                    "tail_included": computed_tail,
                }
    return lookup


def _batch_metadata(
    batch: Mapping[str, Any],
    cfg: Any,
    *,
    bucket_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[Any]]:
    del cfg
    sample_ids = [str(item) for item in batch["sample_id"]]
    batch_size = len(sample_ids)

    label_value = batch.get("label_id")
    if isinstance(label_value, Tensor):
        labels: list[Any] = [int(item) for item in label_value.detach().cpu().reshape(-1).tolist()]
    elif isinstance(label_value, Sequence) and not isinstance(label_value, (str, bytes)):
        labels = [None if item is None else int(item) for item in label_value]
    else:
        labels = [None] * batch_size
    if len(labels) != batch_size:
        raise ValueError(f"label_id count must match batch_size, got {len(labels)} vs {batch_size}")

    audio_path = batch.get("audio_path", None)
    if isinstance(audio_path, Sequence) and not isinstance(audio_path, (str, bytes)):
        audio_paths = [str(item) for item in audio_path]
    elif audio_path is None:
        audio_paths = [""] * batch_size
    else:
        audio_paths = [str(audio_path)] * batch_size
    if len(audio_paths) != batch_size:
        raise ValueError(f"audio_path count must match batch_size, got {len(audio_paths)} vs {batch_size}")

    bucket_values = _batch_sequence(batch, "bucket", batch_size)
    shot_group_values = _batch_sequence(batch, "shot_group", batch_size)
    train_label_n_values = _batch_sequence(batch, "train_label_n", batch_size)
    tail_flag_values = _batch_sequence(batch, "tail_included", batch_size)

    if bucket_values is None:
        missing = [sample_id for sample_id in sample_ids if sample_id not in bucket_lookup]
        if missing:
            raise ValueError(
                "Activation Distribution probe requires sample-level bucket metadata; "
                f"missing sample_id values={missing[:8]}"
            )
        bucket_values = [bucket_lookup[sample_id]["bucket"] for sample_id in sample_ids]
        if shot_group_values is None:
            shot_group_values = [bucket_lookup[sample_id].get("shot_group", bucket_lookup[sample_id]["bucket"]) for sample_id in sample_ids]
        if tail_flag_values is None:
            tail_flag_values = [bucket_lookup[sample_id].get("tail_included") for sample_id in sample_ids]

    result = read_canonical_bucket(
        bucket=bucket_values,
        shot_group=shot_group_values,
        train_label_n=train_label_n_values,
        context="activation_distribution batch",
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in result.bucket]
    computed_tail_flags = [bool(item) for item in tail_included(buckets)]
    if tail_flag_values is not None:
        observed_tail_flags = [
            _parse_bool_metadata(value, context=f"activation_distribution batch idx={idx}")
            for idx, value in enumerate(tail_flag_values)
        ]
        if observed_tail_flags != computed_tail_flags:
            raise ValueError("activation_distribution batch: tail_included must match bucket-derived tail membership")

    return {
        "sample_id": sample_ids,
        "label_id": labels,
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
        activation.detach().float().cpu(),
        raw_layout=raw_layout,
        padding_mask=None if padding_mask is None else padding_mask.detach().cpu(),
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
        activation.detach().float().cpu(),
        raw_layout=raw_layout,
        padding_mask=None if padding_mask is None else padding_mask.detach().cpu(),
        batch_size=int(batch_size),
        context=target_name,
    )


def _hidden_padding_mask_to_bt(
    padding_mask: Tensor | None,
    *,
    batch_size: int,
    token_count: int,
    target_name: str,
) -> Tensor:
    if padding_mask is None:
        raise RuntimeError(
            f"{target_name}: forward output must provide hidden padding_mask [B,T]; "
            "activation distribution metrics refuse to include padding tokens"
        )
    return canonicalize_padding_mask_bt(
        padding_mask.detach().cpu(),
        batch_size=int(batch_size),
        token_count=int(token_count),
        context=target_name,
    )


def _select_valid_hidden_tokens(
    activation_bt: Tensor,
    padding_mask_t: Tensor,
    *,
    target_name: str,
) -> tuple[Tensor, dict[str, Any]]:
    if activation_bt.ndim != 2:
        raise ValueError(f"{target_name}: per-sample activation must be [T,C], got shape={tuple(activation_bt.shape)}")
    mask = padding_mask_t.detach().to(dtype=torch.bool, device=activation_bt.device).reshape(-1)
    if int(mask.numel()) != int(activation_bt.shape[0]):
        raise ValueError(
            f"{target_name}: sample padding mask length must match token count, "
            f"got {int(mask.numel())} vs {int(activation_bt.shape[0])}"
        )
    total_tokens = int(mask.numel())
    padding_tokens = int(mask.sum().item())
    valid_tokens = total_tokens - padding_tokens
    if total_tokens <= 0:
        raise ValueError(f"{target_name}: hidden token dimension is empty")
    if valid_tokens <= 0:
        raise ValueError(f"{target_name}: all hidden tokens are padding; cannot compute activation metrics")
    valid_activation = activation_bt[~mask].contiguous()
    canonical = canonicalize_already_filtered_tokens_btc(valid_activation, context=target_name)
    return canonical.tensor, {
        "valid_token_count": valid_tokens,
        "total_token_count": total_tokens,
        "padding_token_count": padding_tokens,
        "padding_token_ratio": float(padding_tokens) / float(total_tokens),
        "valid_token_policy": "explicit_no_padding",
        **layout_info_to_public_fields(canonical.layout_info),
    }


def _histogram_density_from_counts(counts: Any) -> tuple[list[float], int]:
    count_array = np.asarray(counts, dtype=np.float64).reshape(-1)
    total_count = int(np.sum(count_array))
    if total_count <= 0:
        return [float("nan") for _ in count_array], 0
    density = count_array / float(total_count)
    return density.astype(float).tolist(), total_count


def _histogram_payload(x: Tensor, *, bins: int) -> tuple[list[float], list[int], list[float], int]:
    flat = x.detach().float().cpu().reshape(-1).numpy()
    if flat.size == 0:
        raise ValueError("涓嶈兘瀵圭┖ activation 璁＄畻 histogram")
    if not np.isfinite(flat).all():
        raise ValueError("activation contains NaN or Inf; cannot compute histogram")
    min_value = float(np.min(flat))
    max_value = float(np.max(flat))
    if min_value == max_value:
        delta = max(abs(min_value), 1.0) * 1e-6
        min_value -= delta
        max_value += delta
    counts, edges = np.histogram(flat, bins=int(bins), range=(min_value, max_value))
    count_list = counts.astype(int).tolist()
    density, total_count = _histogram_density_from_counts(count_list)
    return edges.astype(float).tolist(), count_list, density, total_count


def _tensor_to_scalar(value: Any) -> Any:
    if isinstance(value, Tensor):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.item()
        return detached.tolist()
    return value


def _get_encoder_layers_for_probe(model: nn.Module) -> nn.ModuleList:
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
    """Cover all Transformer FFN fc1/fc2 modules by default."""

    layers = _get_encoder_layers_for_probe(model)
    targets: list[tuple[int, str]] = []
    for layer_idx, layer in enumerate(layers):
        for module_name in DEFAULT_FFN_MODULE_NAMES:
            if not hasattr(layer, module_name):
                raise AttributeError(f"encoder.layers.{layer_idx}.{module_name} 涓嶅瓨鍦紝鏃犳硶鎵ц鍏ㄩ噺 FFN probe")
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
    """Validate supplied target FFN modules without silently skipping missing targets."""

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
    stage_value = _probe_get(cfg, "TARGET_MODULES", None)
    if stage_value is not None:
        return stage_value
    return _cfg_get(cfg, "FFN_SELECTED_TARGET_MODULES", None)


def _validate_target_modules_exist(model: nn.Module, target_modules: Sequence[tuple[int, str]]) -> None:
    layers = _get_encoder_layers_for_probe(model)
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
        raise AttributeError("Configured activation distribution target modules do not exist: " + ", ".join(missing))
    if non_module:
        raise TypeError("Configured activation distribution targets are not nn.Module instances: " + ", ".join(non_module))


def _resolve_target_modules(
    model: nn.Module,
    cfg: Any,
    target_modules: Sequence[tuple[int, str]] | None,
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


def _format_target_modules(target_modules: Sequence[tuple[int, str]]) -> str:
    return ", ".join(f"L{int(layer_idx)}.{module_name}" for layer_idx, module_name in target_modules)


def _batch_sample_count(batch: Mapping[str, Any]) -> int:
    sample_ids = batch.get("sample_id", None)
    if isinstance(sample_ids, Sequence) and not isinstance(sample_ids, (str, bytes)):
        return len(sample_ids)
    waveform = batch.get("waveform", None)
    if isinstance(waveform, Tensor) and waveform.ndim >= 1:
        return int(waveform.shape[0])
    label = batch.get("label_id", None)
    if isinstance(label, Tensor) and label.ndim >= 1:
        return int(label.shape[0])
    raise ValueError("Cannot infer batch size from calibration batch.")


def _make_fixed_qparam_observer(cfg: Any, *, granularity: str) -> TensorDistributionObserver:
    qmin, qmax = _qrange(cfg)
    percentile = _percentile_for_observer(cfg)
    observer = TensorDistributionObserver(
        ObserverConfig(
            observer_type="activation",
            granularity=granularity,  # type: ignore[arg-type]
            qmin=int(qmin),
            qmax=int(qmax),
            symmetric=True,
            percentiles=(float(percentile),),
            percentile_low=None,
            percentile_high=float(percentile),
            channel_axis=-1,
            eps=float(_probe_get(cfg, "FIXED_QPARAM_EPS", 1e-8)),
            histogram_bins=_fixed_qparam_hist_bins(cfg),
        ),
        enabled=True,
    )
    observer.start_range_pass(reset=True)
    return observer


def _make_fixed_qparam_observers(
    cfg: Any,
    target_modules: Sequence[tuple[int, str]],
) -> dict[tuple[int, str, str], dict[str, TensorDistributionObserver]]:
    observers: dict[tuple[int, str, str], dict[str, TensorDistributionObserver]] = {}
    for layer_idx, module_name in target_modules:
        for activation_kind in ACTIVATION_KINDS:
            observers[(int(layer_idx), str(module_name), activation_kind)] = {
                "per_tensor": _make_fixed_qparam_observer(cfg, granularity="per_tensor"),
                "per_channel": _make_fixed_qparam_observer(cfg, granularity="per_channel"),
            }
    return observers


def _run_fixed_qparam_calibration_pass(
    model: nn.Module,
    loader: DataLoader,
    cfg: Any,
    target_modules: Sequence[tuple[int, str]],
    observers: Mapping[tuple[int, str, str], Mapping[str, TensorDistributionObserver]],
) -> int:
    device = _device(cfg)
    model.to(device)
    model.eval()
    current_batch_size: int | None = None
    current_batch_capture_counts: dict[tuple[int, str], int] | None = None
    current_batch_payloads: list[dict[str, Any]] | None = None
    batch_count = 0

    def on_capture(payload: dict[str, Any]) -> None:
        if current_batch_size is None or current_batch_payloads is None:
            raise RuntimeError("calibration hook captured activation without active batch context")
        layer_idx = int(payload["layer_idx"])
        module_name = str(payload["module_name"])
        target_key = (layer_idx, module_name)
        if current_batch_capture_counts is not None:
            current_batch_capture_counts[target_key] = current_batch_capture_counts.get(target_key, 0) + 1
        current_batch_payloads.append(dict(payload))

    def consume_payloads(payloads: Sequence[Mapping[str, Any]], hidden_padding_mask: Tensor | None) -> None:
        if current_batch_size is None:
            raise RuntimeError("calibration payload processing lost batch size")
        for payload in payloads:
            layer_idx = int(payload["layer_idx"])
            module_name = str(payload["module_name"])
            for activation_kind in ACTIVATION_KINDS:
                target_name = f"L{layer_idx} {module_name} {activation_kind} calibration"
                canonical = _activation_to_btc_with_layout(
                    payload[activation_kind],
                    batch_size=current_batch_size,
                    target_name=target_name,
                    padding_mask=hidden_padding_mask,
                )
                normalized = canonical.tensor
                hidden_padding_mask_bt = canonical.padding_mask
                if hidden_padding_mask_bt is None:
                    hidden_padding_mask_bt = _hidden_padding_mask_to_bt(
                        hidden_padding_mask,
                        batch_size=current_batch_size,
                        token_count=int(normalized.shape[1]),
                        target_name=target_name,
                    )
                observer_input = normalized.detach().float().cpu()
                observer_pair = observers[(layer_idx, module_name, activation_kind)]
                observer_pair["per_tensor"].update(
                    observer_input,
                    padding_mask=hidden_padding_mask_bt,
                    num_samples=current_batch_size,
                )
                observer_pair["per_channel"].update(
                    observer_input,
                    padding_mask=hidden_padding_mask_bt,
                    num_samples=current_batch_size,
                )

    hook_manager = ActivationHookManager(
        model,
        target_modules,
        cache_activations=False,
        cache_device="cpu",
        on_capture=on_capture,
    )
    try:
        hook_manager.register_hooks()
        with torch.no_grad():
            for batch in loader:
                current_batch_size = _batch_sample_count(batch)
                current_batch_capture_counts = {target: 0 for target in target_modules}
                current_batch_payloads = []
                moved = _move_batch_to_device(batch, device)
                with _autocast_context(cfg, device):
                    forward_output, forward_mask = _forward_model(model, moved, cfg)
                missing_targets = [target for target, count in current_batch_capture_counts.items() if count <= 0]
                if missing_targets:
                    raise RuntimeError(
                        "calibration forward ?????? FFN ??: "
                        f"{_format_target_modules(missing_targets)}"
                    )
                consume_payloads(current_batch_payloads, forward_mask)
                del forward_output, forward_mask, moved
                current_batch_capture_counts = None
                current_batch_payloads = None
                current_batch_size = None
                batch_count += 1
    finally:
        hook_manager.clear_hooks()
    return batch_count


def collect_calibration_fixed_qparams(
    model: nn.Module,
    loader: DataLoader,
    cfg: Any,
    target_modules: Sequence[tuple[int, str]],
) -> dict[tuple[int, str, str], dict[str, Any]]:
    """Collect calibration-fixed activation qparams without storing full activations."""

    targets = list(target_modules)
    if not targets:
        raise ValueError("target_modules cannot be empty; cannot generate fixed qparams")
    chunk_size = min(_fixed_qparam_target_chunk_size(cfg), len(targets))
    percentile = _percentile_for_observer(cfg)
    _qmin, qmax = _qrange(cfg)
    eps = float(_probe_get(cfg, "FIXED_QPARAM_EPS", 1e-8))
    fixed: dict[tuple[int, str, str], dict[str, Any]] = {}

    for start in range(0, len(targets), chunk_size):
        chunk = targets[start : start + chunk_size]
        observers = _make_fixed_qparam_observers(cfg, chunk)
        range_batches = _run_fixed_qparam_calibration_pass(model, loader, cfg, chunk, observers)
        if range_batches <= 0:
            raise RuntimeError("calibration loader produced no batches; cannot generate fixed qparams")

        for observer_pair in observers.values():
            observer_pair["per_tensor"].start_histogram_pass()
            observer_pair["per_channel"].start_histogram_pass()

        histogram_batches = _run_fixed_qparam_calibration_pass(model, loader, cfg, chunk, observers)
        if histogram_batches <= 0:
            raise RuntimeError("calibration histogram pass produced no batches; cannot generate fixed qparams")

        for key, observer_pair in observers.items():
            per_tensor_observer = observer_pair["per_tensor"]
            per_channel_observer = observer_pair["per_channel"]
            per_tensor_observer.finalize()
            per_channel_observer.finalize()
            per_tensor_scale = torch.clamp(
                per_tensor_observer.percentile_value(percentile, absolute=True) / float(qmax),
                min=eps,
            ).detach().cpu().float().reshape(())
            per_channel_scale = torch.clamp(
                per_channel_observer.percentile_value(percentile, absolute=True) / float(qmax),
                min=eps,
            ).detach().cpu().float().reshape(-1)
            per_tensor_summary = per_tensor_observer.summary()
            per_channel_summary = per_channel_observer.summary()
            valid_tokens = int(per_tensor_summary.get("histogram_num_valid_tokens", per_tensor_summary.get("num_valid_tokens", 0)))
            padding_tokens = int(per_tensor_summary.get("histogram_num_padding_tokens", per_tensor_summary.get("num_padding_tokens", 0)))
            total_tokens = valid_tokens + padding_tokens
            fixed[key] = {
                "per_tensor_scale": per_tensor_scale,
                "per_channel_scale": per_channel_scale,
                "num_calibration_elements_per_tensor": int(per_tensor_observer.histogram_num_elements),
                "num_calibration_elements_per_channel": int(per_channel_observer.histogram_num_elements),
                "calibration_range_batches": int(range_batches),
                "calibration_histogram_batches": int(histogram_batches),
                "calibration_valid_token_count": int(valid_tokens),
                "calibration_total_token_count": int(total_tokens),
                "calibration_padding_token_count": int(padding_tokens),
                "calibration_padding_token_ratio": float(padding_tokens) / float(total_tokens) if total_tokens > 0 else float("nan"),
                "calibration_padding_mask_provided": bool(per_tensor_summary.get("padding_mask_provided", False)),
                "per_tensor_observer_summary": per_tensor_summary,
                "per_channel_observer_summary": per_channel_summary,
            }
        del observers

    return fixed

def _fixed_qparams_for_activation(
    fixed_qparams: Mapping[tuple[int, str, str], Mapping[str, Any]] | None,
    *,
    layer_idx: int,
    module_name: str,
    activation_kind: str,
) -> Mapping[str, Any] | None:
    if fixed_qparams is None:
        return None
    return fixed_qparams.get((int(layer_idx), str(module_name), str(activation_kind)))


def _fixed_qparams_manifest(fixed_qparams: Mapping[tuple[int, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (layer_idx, module_name, activation_kind), qparams in sorted(fixed_qparams.items()):
        per_tensor = torch.as_tensor(qparams["per_tensor_scale"]).detach().cpu().float().reshape(-1)
        per_channel = torch.as_tensor(qparams["per_channel_scale"]).detach().cpu().float().reshape(-1)
        rows.append(
            {
                "layer_idx": int(layer_idx),
                "module_name": str(module_name),
                "activation_kind": str(activation_kind),
                "per_tensor_scale": float(per_tensor.reshape(-1)[0].item()),
                "per_channel_scale_mean": float(per_channel.mean().item()),
                "per_channel_scale_std": float(per_channel.std(unbiased=False).item()),
                "per_channel_count": int(per_channel.numel()),
                "num_calibration_elements_per_tensor": int(qparams.get("num_calibration_elements_per_tensor", 0)),
                "num_calibration_elements_per_channel": int(qparams.get("num_calibration_elements_per_channel", 0)),
                "calibration_valid_token_count": int(qparams.get("calibration_valid_token_count", 0)),
                "calibration_total_token_count": int(qparams.get("calibration_total_token_count", 0)),
                "calibration_padding_token_count": int(qparams.get("calibration_padding_token_count", 0)),
                "calibration_padding_token_ratio": float(qparams.get("calibration_padding_token_ratio", float("nan"))),
                "calibration_padding_mask_provided": bool(qparams.get("calibration_padding_mask_provided", False)),
            }
        )
    return rows


def _metrics_row(
    activation: Tensor,
    *,
    row_type: str,
    sample_id: str,
    label_id: Any,
    bucket: str,
    shot_group: str,
    tail_flag: bool,
    analysis_group: str,
    audio_path: str,
    layer_idx: int,
    module_name: str,
    activation_kind: str,
    cfg: Any,
    token_coverage: Mapping[str, Any],
    fixed_qparams: Mapping[tuple[int, str, str], Mapping[str, Any]] | None = None,
    layout_info: Mapping[str, Any] | None = None,
    valid_token_policy: str = "explicit_no_padding",
) -> dict[str, Any]:
    if row_type == "sample":
        if _metadata_value_present(analysis_group):
            raise ValueError("sample rows must not set analysis_group; tail is aggregate-only")
        result = read_canonical_bucket(
            bucket=[bucket],
            shot_group=[shot_group] if _metadata_value_present(shot_group) else None,
            context=f"activation_distribution row sample_id={sample_id!r}",
            allow_shot_group_fallback=False,
        )
        bucket = result.bucket[0]
        shot_group = bucket if _metadata_value_present(shot_group) else ""
        computed_tail = bool(tail_included([bucket])[0])
        if bool(tail_flag) != computed_tail:
            raise ValueError(f"sample_id={sample_id!r}: tail_included does not match bucket={bucket!r}")
        analysis_group = ""

    required_coverage = ("valid_token_count", "total_token_count", "padding_token_count", "padding_token_ratio")
    missing_coverage = [key for key in required_coverage if key not in token_coverage]
    if missing_coverage:
        raise KeyError(f"token_coverage is missing fields: {missing_coverage}")

    qmin, qmax = _qrange(cfg)
    percentile = _percentile(cfg)
    topk_ratio = _topk_ratio(cfg)
    fixed = _fixed_qparams_for_activation(
        fixed_qparams,
        layer_idx=int(layer_idx),
        module_name=module_name,
        activation_kind=activation_kind,
    )
    public_layout_info = dict(layout_info or {key: token_coverage.get(key, "") for key in (
        "original_layout",
        "original_shape",
        "canonical_layout",
        "canonical_shape",
        "layout_inference_source",
        "padding_mask_available",
        "padding_mask_aligned",
        "mask_status",
    )})
    canonical_shape = public_layout_info.get("canonical_shape") or list(activation.shape)
    metrics = compute_activation_distribution_metrics(
        activation,
        percentile=percentile,
        topk_ratio=topk_ratio,
        channel_dim=-1,
        qmin=qmin,
        qmax=qmax,
        fixed_per_tensor_scale=None if fixed is None else fixed.get("per_tensor_scale"),
        fixed_per_channel_scale=None if fixed is None else fixed.get("per_channel_scale"),
    )
    hist_edges, hist_counts, hist_density, hist_total_count = _histogram_payload(activation, bins=_hist_bins(cfg))
    row = _empty_csv_row()
    row.update(_blank_protocol_metric_fields())
    row.update(
        {
            "row_type": row_type,
            "sample_id": sample_id,
            "label_id": label_id,
            "bucket": bucket,
            "shot_group": shot_group,
            "tail_included": bool(tail_flag),
            "analysis_group": analysis_group,
            "audio_path": audio_path,
            "layer_idx": int(layer_idx),
            "module_name": module_name,
            "activation_kind": activation_kind,
            "target_name": f"L{int(layer_idx)} {module_name} {activation_kind.replace('_activation', '')}",
            "activation_shape": canonical_shape,
            "original_layout": public_layout_info.get("original_layout", ""),
            "original_shape": public_layout_info.get("original_shape", ""),
            "canonical_layout": public_layout_info.get("canonical_layout", "BTC"),
            "canonical_shape": canonical_shape,
            "layout_inference_source": public_layout_info.get("layout_inference_source", ""),
            "padding_mask_available": public_layout_info.get("padding_mask_available", False),
            "padding_mask_aligned": public_layout_info.get("padding_mask_aligned", False),
            "mask_status": public_layout_info.get("mask_status", ""),
            "valid_token_policy": token_coverage.get("valid_token_policy", valid_token_policy),
            "valid_token_count": int(token_coverage["valid_token_count"]),
            "total_token_count": int(token_coverage["total_token_count"]),
            "padding_token_count": int(token_coverage["padding_token_count"]),
            "padding_token_ratio": float(token_coverage["padding_token_ratio"]),
            "qmin": qmin,
            "qmax": qmax,
            "percentile": percentile,
            "topk_ratio": topk_ratio,
            "hist_edges": hist_edges,
            "hist_counts": hist_counts,
            "hist_density": hist_density,
            "hist_total_count": hist_total_count,
        }
    )
    for key, value in metrics.items():
        if key not in row:
            raise KeyError(f"activation metric {key!r} is not listed in CSV_COLUMNS")
        row[key] = _tensor_to_scalar(value)
    return row

class _PlotSummaryAccumulator:
    """Stream protocol-v2 plot summary rows by public analysis_group semantics."""

    def __init__(self, *, bins: int) -> None:
        self.bins = int(bins)
        self._states: dict[tuple[int, str, str, str], dict[str, Any]] = {}

    def update(self, row: Mapping[str, Any], activation: Tensor) -> None:
        bucket = row.get("bucket")
        if not _metadata_value_present(bucket):
            raise ValueError("sample row is missing bucket; cannot build analysis_group summaries")
        for analysis_group in ANALYSIS_GROUPS:
            mask = list(analysis_group_mask([bucket], analysis_group))
            if mask and bool(mask[0]):
                self._update_group_state(row, activation, analysis_group=analysis_group)

    def _new_state(self, row: Mapping[str, Any], activation: Tensor, *, analysis_group: str) -> dict[str, Any]:
        return {
            "sample_id": "__aggregate__",
            "label_id": "",
            "bucket": "",
            "shot_group": "",
            "tail_included": "",
            "analysis_group": str(analysis_group),
            "audio_path": "",
            "layer_idx": int(row["layer_idx"]),
            "module_name": str(row["module_name"]),
            "activation_kind": str(row["activation_kind"]),
            "target_name": str(row["target_name"]),
            "activation_shape": [0, 0, int(activation.shape[-1])],
            "original_layout": "",
            "original_shape": "",
            "canonical_layout": "BTC",
            "canonical_shape": [0, 0, int(activation.shape[-1])],
            "layout_inference_source": "group_aggregate_from_canonical_samples",
            "padding_mask_available": False,
            "padding_mask_aligned": False,
            "mask_status": "already_filtered_valid_tokens_group_aggregate",
            "valid_token_policy": "explicit_no_padding",
            "qmin": row["qmin"],
            "qmax": row["qmax"],
            "percentile": row["percentile"],
            "topk_ratio": row["topk_ratio"],
            "num_samples": 0,
            "label_ids": set(),
            "valid_token_count": 0,
            "total_token_count": 0,
            "padding_token_count": 0,
            "metric_sum": {column: 0.0 for column in SCALAR_METRIC_COLUMNS if column != "max_abs"},
            "metric_count": {column: 0 for column in SCALAR_METRIC_COLUMNS},
            "metric_total_count": {column: 0 for column in SCALAR_METRIC_COLUMNS},
            "max_abs": float("nan"),
            "hist_edges": None,
            "hist_counts": None,
        }

    def _update_group_state(self, row: Mapping[str, Any], activation: Tensor, *, analysis_group: str) -> None:
        key = (
            int(row["layer_idx"]),
            str(row["module_name"]),
            str(row["activation_kind"]),
            str(analysis_group),
        )
        state = self._states.get(key)
        if state is None:
            state = self._new_state(row, activation, analysis_group=analysis_group)
            self._states[key] = state

        state["num_samples"] += 1
        state["activation_shape"][0] = int(state["num_samples"])
        label_id = row.get("label_id")
        if _metadata_value_present(label_id):
            state["label_ids"].add(str(label_id))
        state["valid_token_count"] += int(row.get("valid_token_count") or 0)
        state["total_token_count"] += int(row.get("total_token_count") or 0)
        state["padding_token_count"] += int(row.get("padding_token_count") or 0)
        state["activation_shape"][1] = int(state["valid_token_count"])
        state["canonical_shape"] = list(state["activation_shape"])
        self._update_metrics(state, row)
        self._update_histogram(state, activation)

    def _update_metrics(self, state: dict[str, Any], row: Mapping[str, Any]) -> None:
        for column in SCALAR_METRIC_COLUMNS:
            state["metric_total_count"][column] += 1
            value = row.get(column, None)
            if value is None or value == "":
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            state["metric_count"][column] += 1
            if column == "max_abs":
                current = float(state["max_abs"])
                state["max_abs"] = numeric if not np.isfinite(current) else max(current, numeric)
            else:
                state["metric_sum"][column] += numeric

    def _update_histogram(self, state: dict[str, Any], activation: Tensor) -> None:
        flat = activation.detach().float().cpu().reshape(-1).numpy()
        if flat.size == 0:
            raise ValueError("cannot aggregate histogram for an empty activation")
        if not np.isfinite(flat).all():
            raise ValueError("activation contains NaN or Inf; cannot aggregate histogram")
        min_value = float(np.min(flat))
        max_value = float(np.max(flat))
        if min_value == max_value:
            delta = max(abs(min_value), 1.0) * 1e-6
            min_value -= delta
            max_value += delta

        old_edges = state["hist_edges"]
        old_counts = state["hist_counts"]
        if old_edges is None or old_counts is None:
            counts, edges = np.histogram(flat, bins=self.bins, range=(min_value, max_value))
            state["hist_edges"] = edges.astype(float)
            state["hist_counts"] = counts.astype(np.int64)
            return

        assert isinstance(old_edges, np.ndarray)
        assert isinstance(old_counts, np.ndarray)
        if min_value < float(old_edges[0]) or max_value > float(old_edges[-1]):
            new_min = min(min_value, float(old_edges[0]))
            new_max = max(max_value, float(old_edges[-1]))
            new_edges = np.linspace(new_min, new_max, self.bins + 1, dtype=float)
            old_centers = (old_edges[:-1] + old_edges[1:]) * 0.5
            rebinned_old, _ = np.histogram(old_centers, bins=new_edges, weights=old_counts)
            new_counts, _ = np.histogram(flat, bins=new_edges)
            state["hist_edges"] = new_edges
            state["hist_counts"] = rebinned_old.astype(np.int64) + new_counts.astype(np.int64)
            return

        new_counts, _ = np.histogram(flat, bins=old_edges)
        state["hist_counts"] = old_counts + new_counts.astype(np.int64)

    @staticmethod
    def _metric_value_from_state(state: Mapping[str, Any] | None, metric_name: str) -> float:
        if state is None:
            return float("nan")
        if metric_name == "max_abs":
            return float(state["max_abs"])
        count = int(state["metric_count"].get(metric_name, 0))
        if count <= 0:
            return float("nan")
        return float(state["metric_sum"][metric_name]) / count

    @staticmethod
    def _metric_count_fields_from_state(state: Mapping[str, Any] | None, metric_name: str) -> dict[str, Any]:
        if state is None:
            return {"valid_count": 0, "total_count": 0, "invalid_count": 0, "valid_ratio": float("nan")}
        total_count = int(state["metric_total_count"].get(metric_name, 0))
        valid_count = int(state["metric_count"].get(metric_name, 0))
        invalid_count = max(total_count - valid_count, 0)
        valid_ratio = float(valid_count) / float(total_count) if total_count > 0 else float("nan")
        return {
            "valid_count": valid_count,
            "total_count": total_count,
            "invalid_count": invalid_count,
            "valid_ratio": valid_ratio,
        }

    @staticmethod
    def _sample_count_from_state(state: Mapping[str, Any] | None) -> int:
        return 0 if state is None else int(state["num_samples"])

    @staticmethod
    def _class_count_from_state(state: Mapping[str, Any] | None) -> int:
        if state is None:
            return 0
        return int(len(state["label_ids"]))

    @staticmethod
    def _base_key(key: tuple[int, str, str, str]) -> tuple[int, str, str]:
        return (int(key[0]), str(key[1]), str(key[2]))

    def _summary_base_row(self, state: Mapping[str, Any], *, row_type: str) -> dict[str, Any]:
        row = _empty_csv_row()
        row.update(_blank_protocol_metric_fields())
        row.update(
            {
                "row_type": row_type,
                "sample_id": state["sample_id"],
                "label_id": state["label_id"],
                "bucket": state["bucket"],
                "shot_group": state["shot_group"],
                "tail_included": state["tail_included"],
                "analysis_group": state["analysis_group"],
                "audio_path": state["audio_path"],
                "layer_idx": state["layer_idx"],
                "module_name": state["module_name"],
                "activation_kind": state["activation_kind"],
                "target_name": state["target_name"],
                "activation_shape": state["canonical_shape"],
                "original_layout": state["original_layout"],
                "original_shape": state["original_shape"],
                "canonical_layout": state["canonical_layout"],
                "canonical_shape": state["canonical_shape"],
                "layout_inference_source": state["layout_inference_source"],
                "padding_mask_available": state["padding_mask_available"],
                "padding_mask_aligned": state["padding_mask_aligned"],
                "mask_status": state["mask_status"],
                "valid_token_policy": state["valid_token_policy"],
                "qmin": state["qmin"],
                "qmax": state["qmax"],
                "percentile": state["percentile"],
                "topk_ratio": state["topk_ratio"],
                "sample_count": self._sample_count_from_state(state),
                "class_count": self._class_count_from_state(state),
                "valid_token_count": int(state.get("valid_token_count", 0)),
                "total_token_count": int(state.get("total_token_count", 0)),
                "padding_token_count": int(state.get("padding_token_count", 0)),
                "padding_token_ratio": (
                    float(state.get("padding_token_count", 0)) / float(state.get("total_token_count", 0))
                    if int(state.get("total_token_count", 0)) > 0
                    else float("nan")
                ),
            }
        )
        return row

    def _group_count_fields(self, group_states: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
        fields: dict[str, int] = {}
        for group in ANALYSIS_GROUPS:
            state = group_states.get(group)
            fields[f"{group}_sample_count"] = self._sample_count_from_state(state)
            fields[f"{group}_class_count"] = self._class_count_from_state(state)
        return fields

    def _metric_rows_for_base(
        self,
        group_states: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        count_fields = self._group_count_fields(group_states)
        for metric_name in SCALAR_METRIC_COLUMNS:
            metadata = activation_distribution_metric_metadata(metric_name)
            group_values = {
                group: self._metric_value_from_state(group_states.get(group), metric_name)
                for group in ANALYSIS_GROUPS
            }
            diagnostic_gap_direction = (
                "lower_is_better" if metric_name in RANGE_DIAGNOSTIC_GAP_METRICS else None
            )
            gaps = compute_tail_and_few_gaps(
                group_values,
                metric_direction=str(metadata["metric_direction"]),
                diagnostic_gap_direction=diagnostic_gap_direction,
            )
            for group in ANALYSIS_GROUPS:
                state = group_states.get(group)
                if state is None:
                    continue
                metric_value = group_values[group]
                row = self._summary_base_row(state, row_type="metric_summary")
                row.update(count_fields)
                row.update(self._metric_count_fields_from_state(state, metric_name))
                row.update(
                    {
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "metric_direction": metadata["metric_direction"],
                        "aggregation_unit": metadata["aggregation_unit"],
                        "aggregation_fn": metadata["aggregation_fn"],
                        "many_value": group_values["many"],
                        "medium_value": group_values["medium"],
                        "few_value": group_values["few"],
                        "tail_value": group_values["tail"],
                        "tail_vs_many_gap": gaps["tail_vs_many_gap"],
                        "few_vs_many_gap": gaps["few_vs_many_gap"],
                        metric_name: metric_value,
                    }
                )
                rows.append(row)
        return rows

    def _histogram_row(self, state: Mapping[str, Any]) -> dict[str, Any]:
        metadata = activation_distribution_metric_metadata(HISTOGRAM_METRIC_NAME)
        row = self._summary_base_row(state, row_type="histogram_summary")
        hist_counts = state["hist_counts"].astype(int).tolist()
        hist_density, hist_total_count = _histogram_density_from_counts(hist_counts)
        row.update(
            {
                "metric_name": HISTOGRAM_METRIC_NAME,
                "metric_direction": metadata["metric_direction"],
                "aggregation_unit": metadata["aggregation_unit"],
                "aggregation_fn": metadata["aggregation_fn"],
                "hist_edges": state["hist_edges"].astype(float).tolist(),
                "hist_counts": hist_counts,
                "hist_density": hist_density,
                "hist_total_count": hist_total_count,
            }
        )
        return row

    def rows(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = {}
        for key, state in self._states.items():
            grouped.setdefault(self._base_key(key), {})[str(key[3])] = state

        result: list[dict[str, Any]] = []
        for base_key in sorted(grouped):
            group_states = grouped[base_key]
            result.extend(self._metric_rows_for_base(group_states))
            for group in ANALYSIS_GROUPS:
                state = group_states.get(group)
                if state is not None:
                    result.append(self._histogram_row(state))
        return result

def _open_sample_writer(csv_path: Path | None) -> tuple[Any | None, csv.DictWriter | None]:
    if csv_path is None:
        return None, None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    handle = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    return handle, writer


def _emit_sample_row(
    row: Mapping[str, Any],
    *,
    writer: csv.DictWriter | None,
    rows: list[dict[str, Any]],
) -> None:
    _validate_csv_row_contract(row, source="sample row")
    if writer is not None:
        writer.writerow(_csv_row(row))
    else:
        rows.append(dict(row))


def collect_activation_statistics(
    model: nn.Module,
    loader: DataLoader,
    cfg: Any,
    target_modules: Sequence[tuple[int, str]] | None = None,
    *,
    sample_csv_path: Path | None = None,
    fixed_qparams: Mapping[tuple[int, str, str], Mapping[str, Any]] | None = None,
    layout_info: Mapping[str, Any] | None = None,
    valid_token_policy: str = "explicit_no_padding",
) -> dict[str, Any]:
    """Stream FFN input/output activations and compute per-sample valid-token statistics."""

    active_target_modules, _target_modules_source = _resolve_target_modules(model, cfg, target_modules)
    device = _device(cfg)
    model.to(device)
    model.eval()

    current_metadata: dict[str, list[Any]] | None = None
    current_batch_payloads: list[dict[str, Any]] | None = None
    sample_rows: list[dict[str, Any]] = []
    sample_row_count = 0
    batch_count = 0
    sample_count = 0
    accumulator = _PlotSummaryAccumulator(bins=_hist_bins(cfg))
    bucket_lookup = _load_bucket_lookup(cfg)
    capture_counts = {target: 0 for target in active_target_modules}
    current_batch_capture_counts: dict[tuple[int, str], int] | None = None

    handle, writer = _open_sample_writer(sample_csv_path)

    def on_capture(payload: dict[str, Any]) -> None:
        if current_metadata is None or current_batch_payloads is None:
            raise RuntimeError("hook captured activation without current batch metadata")
        layer_idx = int(payload["layer_idx"])
        module_name = str(payload["module_name"])
        target_key = (layer_idx, module_name)
        if target_key not in capture_counts:
            raise RuntimeError(f"hook ??????????: L{layer_idx}.{module_name}")
        capture_counts[target_key] += 1
        if current_batch_capture_counts is not None:
            current_batch_capture_counts[target_key] = current_batch_capture_counts.get(target_key, 0) + 1
        current_batch_payloads.append(dict(payload))

    def consume_payloads(payloads: Sequence[Mapping[str, Any]], hidden_padding_mask: Tensor | None) -> None:
        nonlocal sample_row_count
        if current_metadata is None:
            raise RuntimeError("activation payload processing lost batch metadata")
        batch_size = len(current_metadata["sample_id"])
        for payload in payloads:
            layer_idx = int(payload["layer_idx"])
            module_name = str(payload["module_name"])
            for activation_kind in ACTIVATION_KINDS:
                target_name = f"L{layer_idx} {module_name} {activation_kind}"
                canonical = _activation_to_btc_with_layout(
                    payload[activation_kind],
                    batch_size=batch_size,
                    target_name=target_name,
                    padding_mask=hidden_padding_mask,
                )
                normalized = canonical.tensor
                hidden_padding_mask_bt = canonical.padding_mask
                if hidden_padding_mask_bt is None:
                    hidden_padding_mask_bt = _hidden_padding_mask_to_bt(
                        hidden_padding_mask,
                        batch_size=batch_size,
                        token_count=int(normalized.shape[1]),
                        target_name=target_name,
                    )
                for sample_idx in range(batch_size):
                    sample_target_name = f"{target_name} sample_id={current_metadata['sample_id'][sample_idx]}"
                    valid_activation, token_coverage = _select_valid_hidden_tokens(
                        normalized[sample_idx],
                        hidden_padding_mask_bt[sample_idx],
                        target_name=sample_target_name,
                    )
                    row = _metrics_row(
                        valid_activation,
                        row_type="sample",
                        sample_id=str(current_metadata["sample_id"][sample_idx]),
                        label_id=current_metadata["label_id"][sample_idx],
                        bucket=str(current_metadata["bucket"][sample_idx]),
                        shot_group=str(current_metadata["shot_group"][sample_idx]),
                        tail_flag=bool(current_metadata["tail_included"][sample_idx]),
                        analysis_group="",
                        audio_path=str(current_metadata["audio_path"][sample_idx]),
                        layer_idx=layer_idx,
                        module_name=module_name,
                        activation_kind=activation_kind,
                        cfg=cfg,
                        token_coverage=token_coverage,
                        fixed_qparams=fixed_qparams,
                        layout_info=token_coverage,
                        valid_token_policy="explicit_no_padding",
                    )
                    _emit_sample_row(row, writer=writer, rows=sample_rows)
                    accumulator.update(row, valid_activation)
                    sample_row_count += 1

    hook_manager = ActivationHookManager(
        model,
        active_target_modules,
        cache_activations=False,
        cache_device="cpu",
        on_capture=on_capture,
    )

    try:
        hook_manager.register_hooks()
        with torch.no_grad():
            for batch in loader:
                current_metadata = _batch_metadata(batch, cfg, bucket_lookup=bucket_lookup)
                current_batch_capture_counts = {target: 0 for target in active_target_modules}
                current_batch_payloads = []
                batch_count += 1
                sample_count += len(current_metadata["sample_id"])
                moved = _move_batch_to_device(batch, device)
                with _autocast_context(cfg, device):
                    # Metrics come from hooks; forward only ensures the target path is executed.
                    forward_output, forward_mask = _forward_model(model, moved, cfg)
                missing_targets = [target for target, count in current_batch_capture_counts.items() if count <= 0]
                if missing_targets:
                    raise RuntimeError(
                        "model forward did not pass through target FFN modules: "
                        f"{_format_target_modules(missing_targets)}; "
                        f"current batch sample_id={current_metadata['sample_id']}; "
                        "check ffn_activation_distribution_forward / model.extract_features / model.forward path"
                    )
                consume_payloads(current_batch_payloads, forward_mask)
                del forward_output, forward_mask, moved
                current_batch_capture_counts = None
                current_batch_payloads = None
                current_metadata = None
    finally:
        hook_manager.clear_hooks()
        if handle is not None:
            handle.close()

    if batch_count <= 0:
        raise RuntimeError("?????? loader ?????? batch")

    plot_rows = accumulator.rows()
    return {
        "sample_rows": sample_rows,
        "plot_rows": plot_rows,
        "num_rows": sample_row_count,
        "num_samples": sample_count,
        "num_batches": batch_count,
        "target_modules": active_target_modules,
        "hook_capture_counts": {
            f"L{int(layer_idx)}.{module_name}": int(capture_counts[(layer_idx, module_name)])
            for layer_idx, module_name in active_target_modules
        },
    }


def _to_csv_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        value = _tensor_to_scalar(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _validate_csv_row_contract(row: Mapping[str, Any], *, source: str) -> None:
    """Validate the CSV row contract so new metrics are not silently truncated."""

    expected = set(CSV_COLUMNS)
    actual = set(row.keys())
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if not extra and not missing:
        return

    details: list[str] = []
    if extra:
        details.append(f"extra fields: {', '.join(extra)}")
    if missing:
        details.append(f"missing fields: {', '.join(missing)}")
    raise ValueError(f"{source} does not match CSV_COLUMNS output contract: " + "; ".join(details))


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _to_csv_value(row[key]) for key in CSV_COLUMNS}


def _write_rows(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row_idx, row in enumerate(rows):
            _validate_csv_row_contract(row, source=f"{csv_path.name} row {row_idx}")
            writer.writerow(_csv_row(row))
    os.replace(tmp_path, csv_path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Tensor):
        return _tensor_to_scalar(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _optional_probe_csv(cfg: Any) -> Path | None:
    try:
        return _probe_csv(cfg)
    except AttributeError:
        return None


def _optional_calibration_csv(cfg: Any) -> Path | None:
    try:
        return _calibration_csv(cfg)
    except AttributeError:
        return None


def _safe_dataset_name(cfg: Any) -> str | None:
    try:
        return _dataset_name(cfg)
    except Exception:
        return None


def _target_stem(layer_idx: int, module_name: str) -> str:
    return f"layer_{int(layer_idx):02d}_{module_name}"


def _sample_shard_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _result_dir(cfg) / "sample_shards" / f"{_target_stem(layer_idx, module_name)}.samples.csv"


def _summary_shard_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _result_dir(cfg) / "summary_shards" / f"{_target_stem(layer_idx, module_name)}.summary.csv"


def _fixed_qparams_shard_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _result_dir(cfg) / "fixed_qparams" / f"{_target_stem(layer_idx, module_name)}_fixed_qparams.pt"


def _done_marker_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _result_dir(cfg) / "done" / f"{_target_stem(layer_idx, module_name)}.done.json"


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
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
    return {
        "path": str(path.expanduser().resolve(strict=False)),
        "sha256": _sha256_file(path),
    }



def _fp32_reference_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_REFERENCE_CACHE", {})
    return value if isinstance(value, Mapping) else {}


def _fp32_reference_cache_root(cfg: Any) -> Path:
    configured = _probe_get(cfg, "FP32_REFERENCE_CACHE_ROOT", None)
    if configured is None:
        configured = _nested_get(_fp32_reference_cfg(cfg), "CACHE_ROOT", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "cache" / "fp32_reference"


def _fp32_core_eval_path(cfg: Any) -> Path:
    for key in ("CORE_CACHE_PATH", "CORE_EVAL_PATH", "FP32_CORE_CACHE_PATH"):
        value = _probe_get(cfg, key, None)
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
        value = _probe_get(cfg, key, None)
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


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_probe_csv_sample_ids(cfg: Any) -> list[str]:
    csv_path = _probe_csv(cfg)
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"probe eval CSV does not exist: {csv_path}")
    sample_ids: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"probe eval CSV has no header: {csv_path}")
        for row_idx, row in enumerate(reader, start=2):
            sample_id = _sample_id_from_metadata_row(row)
            if sample_id is None:
                raise ValueError(f"{csv_path}:{row_idx}: missing sample_id/audio_id/file_id")
            sample_ids.append(str(sample_id))
    if not sample_ids:
        raise ValueError(f"probe eval CSV has no samples: {csv_path}")
    duplicates = sorted({item for item in sample_ids if sample_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"probe eval CSV has duplicate sample_id values: {duplicates[:8]}")
    return sample_ids


def _probe_max_samples(cfg: Any) -> int | None:
    value = _probe_get(cfg, "MAX_SAMPLES", None)
    if value is None:
        return None
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"MAX_SAMPLES must be positive, got {value}")
    return limit


def _resolve_reference_relative_path(root_file: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root_file.parent / path


def _validate_sharded_core_manifest(core_path: Path, manifest: Mapping[str, Any]) -> None:
    required_pairs = {
        "cache_type": FP32_CORE_CACHE_TYPE,
        "stage": FP32_CORE_STAGE,
        "storage_format": FP32_CORE_STORAGE_FORMAT,
    }
    mismatches: list[str] = []
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
        missing = [str(item) for item in shard_paths if not _resolve_reference_relative_path(core_path, item).is_file()]
        if missing:
            mismatches.append(f"missing core_eval shard files: {missing[:8]}")
    if mismatches:
        preview = "; ".join(mismatches[:8])
        raise RuntimeError(
            f"core_eval.pt is not a reusable sharded FP32 reference cache: {core_path}; "
            f"{preview}; rebuild with python -m src.verify.build_fp32_reference_cache --force"
        )


def _load_required_core_eval_reference(cfg: Any) -> dict[str, Any]:
    core_path = _fp32_core_eval_path(cfg)
    manifest_path = _artifact_manifest_path(core_path)
    manifest = _read_json_mapping_required(manifest_path, context="core_eval.pt")
    _validate_sharded_core_manifest(core_path, manifest)

    eval_csv = _probe_csv(cfg)
    eval_csv_hash = _sha256_file(eval_csv)
    if manifest.get("eval_csv_sha256") != eval_csv_hash:
        raise RuntimeError(
            "core_eval.pt eval_csv_sha256 mismatch: "
            f"manifest={manifest.get('eval_csv_sha256')!r}, current={eval_csv_hash!r}, eval_csv={eval_csv}"
        )

    checkpoint_path = _optional_checkpoint_path(cfg)
    checkpoint_hash = None if checkpoint_path is None else _sha256_file(checkpoint_path)
    if checkpoint_hash is None:
        raise RuntimeError("core_eval.pt checkpoint_sha256 cannot be validated because checkpoint path is missing or unreadable")
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(
            "core_eval.pt checkpoint_sha256 mismatch: "
            f"manifest={manifest.get('checkpoint_sha256')!r}, current={checkpoint_hash!r}, checkpoint={checkpoint_path}"
        )

    payload = _torch_load_cpu(core_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"core_eval.pt top-level payload must be a mapping: {core_path}")
    core_sample_ids = _normalise_sample_ids(payload.get("sample_id"))
    if not core_sample_ids:
        raise KeyError(f"core_eval.pt missing top-level sample_id order: {core_path}")

    eval_sample_ids = _read_probe_csv_sample_ids(cfg)
    full_eval_hash = _sample_order_hash(eval_sample_ids)
    manifest_sample_hash = manifest.get("sample_order_hash")
    if manifest_sample_hash != full_eval_hash:
        raise RuntimeError(
            "core_eval.pt sample_order_hash does not match probe eval CSV order: "
            f"manifest={manifest_sample_hash!r}, current={full_eval_hash!r}"
        )

    max_samples = _probe_max_samples(cfg)
    if max_samples is None:
        expected_sample_ids = eval_sample_ids
        sample_order_policy = "full_eval_exact"
    else:
        expected_count = min(int(max_samples), len(eval_sample_ids))
        expected_sample_ids = eval_sample_ids[:expected_count]
        sample_order_policy = "full_eval_prefix_subset"

    if len(core_sample_ids) < len(expected_sample_ids):
        raise RuntimeError(
            f"core_eval.pt has {len(core_sample_ids)} samples, fewer than required probe samples {len(expected_sample_ids)}"
        )
    observed_prefix = core_sample_ids[: len(expected_sample_ids)]
    if observed_prefix != expected_sample_ids:
        first_mismatches = [
            {"index": idx, "core_eval": observed, "probe_eval": expected}
            for idx, (observed, expected) in enumerate(zip(observed_prefix, expected_sample_ids))
            if observed != expected
        ][:8]
        raise RuntimeError(
            "core_eval.pt sample_id order does not match required probe subset; "
            f"policy={sample_order_policy}, first_mismatches={first_mismatches}"
        )

    return {
        "required": True,
        "status": "validated",
        "path": core_path,
        "manifest_path": manifest_path,
        "identity": _path_identity(core_path),
        "manifest_identity": _path_identity(manifest_path),
        "cache_type": manifest.get("cache_type"),
        "stage": manifest.get("stage"),
        "storage_format": manifest.get("storage_format"),
        "sharded": bool(manifest.get("sharded", False)),
        "num_shards": int(manifest.get("num_shards", 0)),
        "eval_csv_sha256": eval_csv_hash,
        "checkpoint_sha256": checkpoint_hash,
        "sample_order_hash": manifest_sample_hash,
        "sample_order_policy": sample_order_policy,
        "max_samples": max_samples,
        "probe_sample_count": len(expected_sample_ids),
        "core_sample_count": len(core_sample_ids),
        "sample_id_preview": expected_sample_ids[:5],
    }


def _describe_optional_oer_oracle_reference(cfg: Any) -> dict[str, Any]:
    path = _fp32_oer_oracle_path(cfg)
    manifest_path = _artifact_manifest_path(path)
    result: dict[str, Any] = {
        "required": False,
        "path": path,
        "manifest_path": manifest_path,
        "exists": bool(path.exists()),
        "status": "missing",
    }
    if not path.exists():
        return result
    result["status"] = "present_without_manifest"
    if not manifest_path.exists():
        return result
    try:
        manifest = _read_json_mapping_required(manifest_path, context="oer_oracle_eval.pt")
    except Exception as exc:  # noqa: BLE001 - optional diagnostic only.
        result["status"] = "invalid_manifest"
        result["error"] = str(exc)
        return result
    result.update(
        {
            "status": "present",
            "manifest_identity": _path_identity(manifest_path),
            "cache_type": manifest.get("cache_type"),
            "oracle_source": manifest.get("oracle_source"),
            "stage": manifest.get("stage"),
            "storage_format": manifest.get("storage_format"),
            "sharded": bool(manifest.get("sharded", False)),
            "num_shards": manifest.get("num_shards"),
            "sample_order_hash": manifest.get("sample_order_hash"),
            "topk_ratio": manifest.get("topk_ratio"),
            "activation_positions": manifest.get("activation_positions"),
        }
    )
    return result


def _validate_fp32_reference_contract(cfg: Any) -> dict[str, Any]:
    core_reference = _load_required_core_eval_reference(cfg)
    return {
        "required": {"core_eval": core_reference},
        "optional": {"oer_oracle_eval": _describe_optional_oer_oracle_reference(cfg)},
    }


def _fp32_reference_signature_payload(reference_state: Mapping[str, Any]) -> dict[str, Any]:
    core = _nested_get(_nested_get(reference_state, "required", {}), "core_eval", {})
    return {
        "required_core_eval": {
            "path": _nested_get(_nested_get(core, "identity", {}), "path", None),
            "sha256": _nested_get(_nested_get(core, "identity", {}), "sha256", None),
            "manifest_sha256": _nested_get(_nested_get(core, "manifest_identity", {}), "sha256", None),
            "storage_format": _nested_get(core, "storage_format", None),
            "eval_csv_sha256": _nested_get(core, "eval_csv_sha256", None),
            "checkpoint_sha256": _nested_get(core, "checkpoint_sha256", None),
            "sample_order_hash": _nested_get(core, "sample_order_hash", None),
            "sample_order_policy": _nested_get(core, "sample_order_policy", None),
            "max_samples": _nested_get(core, "max_samples", None),
            "probe_sample_count": _nested_get(core, "probe_sample_count", None),
        }
    }

def _experiment_signature_payload(
    cfg: Any,
    target_modules: Sequence[tuple[int, str]],
    *,
    fp32_reference_state: Mapping[str, Any],
) -> dict[str, Any]:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "ffn_activation_distribution_probe",
        "dataset_name": _safe_dataset_name(cfg),
        "active_dataset": _cfg_get(cfg, "ACTIVE_DATASET", None),
        "checkpoint": _path_identity(_optional_checkpoint_path(cfg)),
        "eval_csv": _path_identity(_optional_probe_csv(cfg)),
        "calibration_csv": _path_identity(_optional_calibration_csv(cfg)),
        "audio": _json_safe(_cfg_get(cfg, "AUDIO", {})),
        "quant": _json_safe(_quant_cfg(cfg)),
        "observer": _json_safe(_cfg_get(cfg, "OBSERVER", {})),
        "probe": _json_safe(_probe_cfg(cfg)),
        "legacy_activation_distribution": _json_safe(_legacy_distribution_cfg(cfg)),
        "fp32_reference": _fp32_reference_signature_payload(fp32_reference_state),
        "beats_use_amp": bool(_nested_get(beats_cfg, "USE_AMP", False)),
        "params": {
            "qmin": int(_qrange(cfg)[0]),
            "qmax": int(_qrange(cfg)[1]),
            "percentile": float(_percentile(cfg)),
            "observer_percentile": float(_percentile_for_observer(cfg)),
            "topk_ratio": float(_topk_ratio(cfg)),
            "hist_bins": int(_hist_bins(cfg)),
            "fixed_qparam_hist_bins": int(_fixed_qparam_hist_bins(cfg)),
            "fixed_qparam_target_chunk_size": int(_fixed_qparam_target_chunk_size(cfg)),
            "batch_size": int(_batch_size(cfg)),
            "num_workers": int(_num_workers(cfg)),
            "pin_memory": bool(_pin_memory(cfg)),
            "persistent_workers": bool(_persistent_workers(cfg)),
            "prefetch_factor": _prefetch_factor(cfg),
            "calibration_batch_size": int(_calibration_batch_size(cfg)),
            "calibration_num_workers": int(_calibration_num_workers(cfg)),
            "calibration_pin_memory": bool(_calibration_pin_memory(cfg)),
            "calibration_persistent_workers": bool(_calibration_persistent_workers(cfg)),
            "calibration_prefetch_factor": _calibration_prefetch_factor(cfg),
        },
        "activation_kinds": list(ACTIVATION_KINDS),
        "target_modules": [
            {"layer_idx": int(layer_idx), "module_name": str(module_name)}
            for layer_idx, module_name in target_modules
        ],
        **group_protocol_metadata(
            tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
        ),
    }


def _experiment_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility.
        return torch.load(path, map_location="cpu")


def _cache_protocol_matches(payload: Mapping[str, Any], protocol_version: str, experiment_signature: str) -> bool:
    return (
        str(payload.get("protocol_version", "")) == str(protocol_version)
        and str(payload.get("experiment_signature", "")) == str(experiment_signature)
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _target_fixed_qparams(
    fixed_qparams: Mapping[tuple[int, str, str], Mapping[str, Any]],
    layer_idx: int,
    module_name: str,
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    return {
        key: value
        for key, value in fixed_qparams.items()
        if int(key[0]) == int(layer_idx) and str(key[1]) == str(module_name)
    }


def _load_fixed_qparams_shard_if_valid(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    protocol_version: str,
    experiment_signature: str,
) -> dict[tuple[int, str, str], Mapping[str, Any]] | None:
    path = _fixed_qparams_shard_path(cfg, layer_idx, module_name)
    if not path.exists():
        return None
    try:
        payload = _torch_load_cpu(path)
        if not isinstance(payload, Mapping):
            raise TypeError("fixed qparams shard must be a mapping")
        if not _cache_protocol_matches(payload, protocol_version, experiment_signature):
            raise ValueError("fixed qparams protocol/signature mismatch")
        target = payload.get("target", {})
        if int(_nested_get(target, "layer_idx", -1)) != int(layer_idx):
            raise ValueError("fixed qparams layer_idx mismatch")
        if str(_nested_get(target, "module_name", "")) != str(module_name):
            raise ValueError("fixed qparams module_name mismatch")
        fixed = payload.get("fixed_qparams")
        if not isinstance(fixed, Mapping):
            raise TypeError("fixed_qparams missing or not a mapping")
        result = dict(fixed)
        for activation_kind in ACTIVATION_KINDS:
            key = (int(layer_idx), str(module_name), str(activation_kind))
            if key not in result:
                raise KeyError(f"fixed qparams missing {key}")
        return result
    except Exception as exc:  # pragma: no cover - defensive resume path.
        print(f"[resume] ignore invalid fixed qparams shard {path}: {exc}")
        return None


def _save_fixed_qparams_shard(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    fixed_qparams: Mapping[tuple[int, str, str], Mapping[str, Any]],
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> Path:
    path = _fixed_qparams_shard_path(cfg, layer_idx, module_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(
        {
            "protocol_version": protocol_version,
            "experiment_signature": experiment_signature,
            "experiment_signature_payload": _json_safe(signature_payload),
            "target": {"layer_idx": int(layer_idx), "module_name": str(module_name)},
            "activation_kinds": list(ACTIVATION_KINDS),
            "fixed_qparams": dict(fixed_qparams),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)
    return path


def _done_marker_valid(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    protocol_version: str,
    experiment_signature: str,
) -> bool:
    payload = _read_json(_done_marker_path(cfg, layer_idx, module_name))
    if payload is None or not _cache_protocol_matches(payload, protocol_version, experiment_signature):
        return False
    target = payload.get("target", {})
    if int(_nested_get(target, "layer_idx", -1)) != int(layer_idx):
        return False
    if str(_nested_get(target, "module_name", "")) != str(module_name):
        return False
    for path in (
        _sample_shard_path(cfg, layer_idx, module_name),
        _summary_shard_path(cfg, layer_idx, module_name),
        _fixed_qparams_shard_path(cfg, layer_idx, module_name),
    ):
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return True


def _write_done_marker(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Path:
    path = _done_marker_path(cfg, layer_idx, module_name)
    payload = {
        "protocol_version": protocol_version,
        "experiment_signature": experiment_signature,
        "experiment_signature_payload": _json_safe(signature_payload),
        "target": {"layer_idx": int(layer_idx), "module_name": str(module_name)},
        "paths": {
            "sample_shard": _sample_shard_path(cfg, layer_idx, module_name),
            "summary_shard": _summary_shard_path(cfg, layer_idx, module_name),
            "fixed_qparams_shard": _fixed_qparams_shard_path(cfg, layer_idx, module_name),
        },
        "counts": {
            "num_rows": int(result.get("num_rows", 0)),
            "num_plot_rows": len(result.get("plot_rows", [])),
            "num_samples": int(result.get("num_samples", 0)),
            "num_batches": int(result.get("num_batches", 0)),
            "hook_capture_counts": result.get("hook_capture_counts", {}),
        },
    }
    _write_json_atomic(path, payload)
    return path


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def _merge_csv_shards(shard_paths: Sequence[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    row_count = 0
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for shard_path in shard_paths:
            if not shard_path.exists():
                raise FileNotFoundError(f"missing CSV shard: {shard_path}")
            with shard_path.open("r", newline="", encoding="utf-8") as shard_handle:
                reader = csv.DictReader(shard_handle)
                if reader.fieldnames is None:
                    raise RuntimeError(f"CSV shard has no header: {shard_path}")
                missing = [column for column in CSV_COLUMNS if column not in reader.fieldnames]
                if missing:
                    raise RuntimeError(f"CSV shard {shard_path} missing columns: {missing}")
                for row_idx, row in enumerate(reader):
                    normalized = {key: row.get(key, "") for key in CSV_COLUMNS}
                    _validate_csv_row_contract(normalized, source=f"{shard_path.name} row {row_idx}")
                    writer.writerow(_csv_row(normalized))
                    row_count += 1
    os.replace(tmp_path, output_path)
    return row_count

def _target_modules_manifest(target_modules: Sequence[tuple[int, str]]) -> list[dict[str, Any]]:
    return [
        {
            "layer_idx": int(layer_idx),
            "module_name": str(module_name),
            "target_name": f"encoder.layers.{int(layer_idx)}.{module_name}",
        }
        for layer_idx, module_name in target_modules
    ]


def _build_experiment_manifest(
    *,
    cfg: Any,
    target_modules: Sequence[tuple[int, str]],
    target_modules_source: str,
    result: Mapping[str, Any],
    csv_path: Path,
    plot_summary_csv_path: Path,
    manifest_path: Path,
    plot_paths: Mapping[str, Any],
    model_was_provided: bool,
    loader_was_provided: bool,
    calibration_loader_was_provided: bool,
    fixed_qparams: Mapping[tuple[int, str, str], Mapping[str, Any]],
    fp32_reference_state: Mapping[str, Any],
) -> dict[str, Any]:
    audio_config = _make_audio_config(cfg)
    qmin, qmax = _qrange(cfg)
    dataset_cfg = _dataset_cfg(cfg)
    checkpoint_path = _optional_checkpoint_path(cfg)
    eval_csv = _optional_probe_csv(cfg)
    calibration_csv = _optional_calibration_csv(cfg)
    return {
        "experiment": "ffn_activation_distribution_probe",
        "model_source": "provided_model" if model_was_provided else "restore_clean_model_from_checkpoint",
        "loader_source": "provided_loader" if loader_was_provided else "AudioInputDataset",
        "calibration_loader_source": "provided_loader" if calibration_loader_was_provided else "AudioInputDataset",
        "checkpoint_path": checkpoint_path,
        "active_dataset": _cfg_get(cfg, "ACTIVE_DATASET", None),
        "dataset_name": _safe_dataset_name(cfg),
        "dataset_root": _nested_get(dataset_cfg, "ROOT", None),
        "eval_csv": eval_csv,
        "calibration_csv": calibration_csv,
        "fp32_reference": fp32_reference_state,
        "audio_protocol": {
            "mode": str(_probe_get(cfg, "MODE", "eval")),
            "calibration_mode": str(_probe_get(cfg, "CALIBRATION_MODE", "calibration")),
            "sample_rate": int(audio_config.sample_rate),
            "target_seconds": float(audio_config.target_seconds),
            "target_num_samples": int(audio_config.target_num_samples),
            "target_samples_rule": _nested_get(_cfg_get(cfg, "AUDIO", {}), "TARGET_SAMPLES_RULE", "round"),
            "normalization": str(audio_config.normalization),
            "normalization_eps": float(audio_config.eps),
            "deterministic_crop": str(audio_config.deterministic_crop),
            "fixed_crop_start": int(audio_config.fixed_crop_start),
            "padding_value": _nested_get(_cfg_get(cfg, "AUDIO", {}), "PADDING_VALUE", 0.0),
            "padding_mask_true_means": _nested_get(_cfg_get(cfg, "AUDIO", {}), "PADDING_MASK_TRUE_MEANS", "padding"),
        },
        "probe_params": {
            "device": str(_device(cfg)),
            "qmin": int(qmin),
            "qmax": int(qmax),
            "percentile": float(_percentile(cfg)),
            "observer_percentile": float(_percentile_for_observer(cfg)),
            "topk_ratio": float(_topk_ratio(cfg)),
            "hist_bins": int(_hist_bins(cfg)),
            "fixed_qparam_hist_bins": int(_fixed_qparam_hist_bins(cfg)),
            "fixed_qparam_target_chunk_size": int(_fixed_qparam_target_chunk_size(cfg)),
            "batch_size": int(_batch_size(cfg)),
            "num_workers": int(_num_workers(cfg)),
            "pin_memory": bool(_pin_memory(cfg)),
            "calibration_batch_size": int(_calibration_batch_size(cfg)),
            "calibration_num_workers": int(_calibration_num_workers(cfg)),
            "calibration_pin_memory": bool(_calibration_pin_memory(cfg)),
        },
        "target_modules_source": str(target_modules_source),
        "target_module_count": len(target_modules),
        "target_modules": _target_modules_manifest(target_modules),
        "fixed_qparams": {
            "source": "calibration_two_pass_histogram",
            "granularities": ["per_tensor", "per_channel"],
            "items": _fixed_qparams_manifest(fixed_qparams),
        },
        **group_protocol_metadata(
            tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
        ),
        "grouping_protocol": group_protocol_metadata(
            tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
        ),
        "metric_protocol": {
            "summary_row_types": ["metric_summary", "histogram_summary"],
            "metric_specs": {
                name: activation_distribution_metric_metadata(name)
                for name in activation_distribution_metric_names()
            },
            "sample_rows": "raw per-sample activation tensors converted to scalar descriptors; no sample-level tail bucket",
            "plot_summary_rows": "group aggregates recomputed from raw sample rows for many/medium/few/tail",
            "scalar_summary_aggregation": "finite-only aggregation: valid_count finite records over total_count records; non-finite values are excluded",
            "histogram_summary_aggregation": "hist_counts are raw summed counts; hist_density is hist_counts normalized by sum(hist_counts) for histogram_density/sum_then_normalize",
            "hidden_activation_shape_contract": "hook captures are normalized to [B,T,C]; rank-2 activations are rejected",
            "hidden_padding_mask_semantics": "forward output padding_mask must be [B,T] with True meaning padding; all activation metrics, histograms, OER, clipping, and fixed-qparam observers exclude padding tokens",
            "token_coverage_fields": ["valid_token_count", "total_token_count", "padding_token_count", "padding_token_ratio"],
            "diagnostic_gap_direction": "diagnostic activation metrics do not compute tail_vs_many_gap/few_vs_many_gap by default; range diagnostic whitelist explicitly uses lower_is_better gap direction",
            "range_diagnostic_gap_metrics": sorted(RANGE_DIAGNOSTIC_GAP_METRICS),
            "adaptive_metrics": "per-sample qparams from the same activation tensor; intrinsic upper-bound pressure",
            "fixed_metrics": "calibration-fixed qparams applied to each probe activation tensor; deployment-like pressure",
            "oer_definition": "top-k energy retention on activation tensor positions selected from fp32 activation energy",
            "primary_gap": "tail_vs_many_gap",
            "supplementary_gap": "few_vs_many_gap",
        },
        "counts": {
            "num_rows": int(result["num_rows"]),
            "num_plot_rows": len(result["plot_rows"]),
            "num_samples": int(result["num_samples"]),
            "num_batches": int(result["num_batches"]),
            "fixed_qparam_count": len(fixed_qparams),
            "hook_capture_counts": result.get("hook_capture_counts", {}),
        },
        "outputs": {
            "sample_csv": csv_path,
            "plot_summary_csv": plot_summary_csv_path,
            "manifest": manifest_path,
            "plots": plot_paths,
        },
    }

def _write_manifest(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_activation_distribution_probe(
    cfg: Any,
    *,
    model: nn.Module | None = None,
    loader: DataLoader | None = None,
    calibration_loader: DataLoader | None = None,
    target_modules: Sequence[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    """Run the FFN activation distribution explanation experiment target by target."""

    model_was_provided = model is not None
    loader_was_provided = loader is not None
    calibration_loader_was_provided = calibration_loader is not None
    fp32_reference_state = _validate_fp32_reference_contract(cfg)
    if model is None:
        model = restore_clean_model_from_checkpoint(cfg)

    active_target_modules, target_modules_source = _resolve_target_modules(model, cfg, target_modules)
    if not active_target_modules:
        raise ValueError("activation distribution probe has no target modules")

    output_dir = _result_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_FILENAME
    plot_summary_csv_path = output_dir / PLOT_SUMMARY_CSV_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    signature_payload = _experiment_signature_payload(
        cfg,
        active_target_modules,
        fp32_reference_state=fp32_reference_state,
    )
    protocol_version = PROTOCOL_VERSION
    experiment_signature = _experiment_signature(signature_payload)

    fixed_qparams: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    target_count_rows: list[dict[str, Any]] = []

    for layer_idx, module_name in active_target_modules:
        target = (int(layer_idx), str(module_name))
        target_modules_for_run = [target]
        target_label = f"L{target[0]}.{target[1]}"
        cached_fixed_qparams = _load_fixed_qparams_shard_if_valid(
            cfg,
            target[0],
            target[1],
            protocol_version,
            experiment_signature,
        )
        done_is_valid = _done_marker_valid(
            cfg,
            target[0],
            target[1],
            protocol_version,
            experiment_signature,
        )
        done_payload = _read_json(_done_marker_path(cfg, target[0], target[1])) if done_is_valid else None
        if done_payload is not None and cached_fixed_qparams is not None:
            fixed_qparams.update(cached_fixed_qparams)
            counts = done_payload.get("counts", {})
            if isinstance(counts, Mapping):
                target_count_rows.append(dict(counts))
            print(f"[resume] skip completed activation distribution target {target_label}")
            continue

        if cached_fixed_qparams is None:
            if calibration_loader is None:
                calibration_loader = _build_calibration_loader(cfg)
            cached_fixed_qparams = collect_calibration_fixed_qparams(
                model,
                calibration_loader,
                cfg,
                target_modules_for_run,
            )
            _save_fixed_qparams_shard(
                cfg,
                target[0],
                target[1],
                cached_fixed_qparams,
                protocol_version,
                experiment_signature,
                signature_payload,
            )
        fixed_qparams.update(cached_fixed_qparams)

        if loader is None:
            loader = _build_probe_loader(cfg)
        result = collect_activation_statistics(
            model,
            loader,
            cfg,
            target_modules_for_run,
            sample_csv_path=_sample_shard_path(cfg, target[0], target[1]),
            fixed_qparams=cached_fixed_qparams,
        )
        _write_rows(_summary_shard_path(cfg, target[0], target[1]), result["plot_rows"])
        _write_done_marker(
            cfg,
            target[0],
            target[1],
            protocol_version,
            experiment_signature,
            signature_payload,
            result,
        )
        target_count_rows.append(
            {
                "num_rows": int(result.get("num_rows", 0)),
                "num_plot_rows": len(result.get("plot_rows", [])),
                "num_samples": int(result.get("num_samples", 0)),
                "num_batches": int(result.get("num_batches", 0)),
                "hook_capture_counts": result.get("hook_capture_counts", {}),
            }
        )
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_shards = [_sample_shard_path(cfg, layer_idx, module_name) for layer_idx, module_name in active_target_modules]
    summary_shards = [_summary_shard_path(cfg, layer_idx, module_name) for layer_idx, module_name in active_target_modules]
    num_rows = _merge_csv_shards(sample_shards, csv_path)
    num_plot_rows = _merge_csv_shards(summary_shards, plot_summary_csv_path)
    plot_rows = _read_csv_rows(plot_summary_csv_path)
    if len(plot_rows) != int(num_plot_rows):
        raise RuntimeError(
            f"merged plot summary row count mismatch: counted {num_plot_rows}, loaded {len(plot_rows)}"
        )

    def max_count(name: str) -> int:
        values: list[int] = []
        for counts in target_count_rows:
            try:
                values.append(int(counts.get(name, 0)))
            except (TypeError, ValueError):
                continue
        return max(values) if values else 0

    hook_capture_counts: dict[str, int] = {}
    for counts in target_count_rows:
        raw_counts = counts.get("hook_capture_counts", {})
        if not isinstance(raw_counts, Mapping):
            continue
        for key, value in raw_counts.items():
            try:
                hook_capture_counts[str(key)] = int(value)
            except (TypeError, ValueError):
                hook_capture_counts[str(key)] = 0

    aggregate_result = {
        "sample_rows": [],
        "plot_rows": plot_rows,
        "num_rows": int(num_rows),
        "num_samples": max_count("num_samples"),
        "num_batches": max_count("num_batches"),
        "target_modules": active_target_modules,
        "hook_capture_counts": hook_capture_counts,
    }

    import pandas as pd

    plot_df = pd.DataFrame(plot_rows)
    plot_paths = {
        "activation_histogram": plot_activation_histograms(plot_df, output_dir),
        "activation_range": plot_activation_range_bar(plot_df, output_dir),
        "adaptive_clipping_rate": plot_adaptive_clipping_rate_comparison(plot_df, output_dir),
        "fixed_clipping_rate": plot_fixed_clipping_rate_comparison(plot_df, output_dir),
        "adaptive_oer_retention": plot_adaptive_oer_retention_comparison(plot_df, output_dir),
        "fixed_oer_retention": plot_fixed_oer_retention_comparison(plot_df, output_dir),
    }
    manifest = _build_experiment_manifest(
        cfg=cfg,
        target_modules=active_target_modules,
        target_modules_source=target_modules_source,
        result=aggregate_result,
        csv_path=csv_path,
        plot_summary_csv_path=plot_summary_csv_path,
        manifest_path=manifest_path,
        plot_paths=plot_paths,
        model_was_provided=model_was_provided,
        loader_was_provided=loader_was_provided,
        calibration_loader_was_provided=calibration_loader_was_provided,
        fixed_qparams=fixed_qparams,
        fp32_reference_state=fp32_reference_state,
    )
    manifest["protocol_version"] = protocol_version
    manifest["experiment_signature"] = experiment_signature
    manifest["experiment_signature_payload"] = signature_payload
    manifest["resume_protocol"] = {
        "unit": "target_module",
        "done_marker_requires": ["sample_shard", "summary_shard", "fixed_qparams_shard"],
        "sample_shard_dir": output_dir / "sample_shards",
        "summary_shard_dir": output_dir / "summary_shards",
        "fixed_qparams_dir": output_dir / "fixed_qparams",
        "done_dir": output_dir / "done",
    }
    _write_manifest(manifest_path, manifest)

    return {
        "result_dir": output_dir,
        "csv_path": csv_path,
        "plot_summary_csv_path": plot_summary_csv_path,
        "manifest_path": manifest_path,
        "plot_paths": plot_paths,
        "num_rows": int(num_rows),
        "num_plot_rows": int(num_plot_rows),
        "num_samples": aggregate_result["num_samples"],
        "num_batches": aggregate_result["num_batches"],
        "target_modules": active_target_modules,
        "fixed_qparam_count": len(fixed_qparams),
        "hook_capture_counts": hook_capture_counts,
        "protocol_version": protocol_version,
        "experiment_signature": experiment_signature,
        "fp32_reference": fp32_reference_state,
    }

def main(cfg: Any, target_modules: Sequence[tuple[int, str]] | None = None) -> dict[str, Any]:
    seed_everything = _cfg_get(cfg, "seed_everything", None)
    if callable(seed_everything):
        seed_everything(_seed(cfg))
    return run_activation_distribution_probe(cfg, target_modules=target_modules)


if __name__ == "__main__":
    import config as project_config

    summary = main(project_config)
    print(f"FFN activation distribution probe 瀹屾垚: {summary['csv_path']}")








