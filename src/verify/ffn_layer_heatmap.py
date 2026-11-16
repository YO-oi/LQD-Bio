from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sanitize_positive_int_env(name: str, default: int) -> None:
    value = os.environ.get(name)
    try:
        if value is not None and int(value) > 0:
            return
    except ValueError:
        pass
    os.environ[name] = str(int(default))


_sanitize_positive_int_env("OMP_NUM_THREADS", 4)
_sanitize_positive_int_env("MKL_NUM_THREADS", 4)

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from src.analysis.build_fp32_oer_oracle import is_sharded_fp32_oer_oracle, iter_fp32_oer_oracle_shards, load_fp32_oer_oracle
    from src.export.export_fp32_core_cache import load_fp32_core_cache
    from src.metrics.common_metrics import scalar_oer_topk_index_metadata, scalar_oer_topk_indices
    from src.metrics.geometry_metrics import compute_group_margin
    from src.metrics.ffn_heatmap_metrics import (
        HEATMAP_CSV_COLUMNS,
        HEATMAP_SUMMARY_CSV_COLUMNS,
        PROJECTION_DETAIL_CSV_COLUMNS,
        build_heatmap_metric_summary_rows,
        compute_effective_class_count,
        compute_group_embedding_distortion,
        compute_group_heatmap_row_summary,
        compute_group_fisher,
        compute_group_oer_metrics_from_energy_cache,
        compute_group_projection_detail_row_summary,
        compute_group_scalar_mean,
        compute_group_tensor_distortion_metrics_from_cache,
        compute_teacher_anchor_center_shift,
        compute_teacher_anchor_embedding_drift,
        compute_teacher_center_margin_quant,
        compute_tensor_distortion_metrics,
    )
    from src.plots.ffn_heatmap_plotter import plot_all_heatmaps
    from src.quant.ffn_quant_injector import make_single_ffn_quant_controller, replace_ffn_module_with_quant, restore_clean_model_from_checkpoint
    from src.utils.group_utils import ANALYSIS_GROUPS, analysis_group_mask, group_protocol_metadata, normalize_analysis_group, read_canonical_bucket
    from src.utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
except ImportError:  # pragma: no cover - direct script debugging fallback.
    from ..data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from ..analysis.build_fp32_oer_oracle import is_sharded_fp32_oer_oracle, iter_fp32_oer_oracle_shards, load_fp32_oer_oracle
    from ..export.export_fp32_core_cache import load_fp32_core_cache
    from ..metrics.common_metrics import scalar_oer_topk_index_metadata, scalar_oer_topk_indices
    from ..metrics.geometry_metrics import compute_group_margin
    from ..metrics.ffn_heatmap_metrics import (
        HEATMAP_CSV_COLUMNS,
        HEATMAP_SUMMARY_CSV_COLUMNS,
        PROJECTION_DETAIL_CSV_COLUMNS,
        build_heatmap_metric_summary_rows,
        compute_effective_class_count,
        compute_group_embedding_distortion,
        compute_group_heatmap_row_summary,
        compute_group_fisher,
        compute_group_oer_metrics_from_energy_cache,
        compute_group_projection_detail_row_summary,
        compute_group_scalar_mean,
        compute_group_tensor_distortion_metrics_from_cache,
        compute_teacher_anchor_center_shift,
        compute_teacher_anchor_embedding_drift,
        compute_teacher_center_margin_quant,
        compute_tensor_distortion_metrics,
    )
    from ..plots.ffn_heatmap_plotter import plot_all_heatmaps
    from ..quant.ffn_quant_injector import make_single_ffn_quant_controller, replace_ffn_module_with_quant, restore_clean_model_from_checkpoint
    from ..utils.group_utils import ANALYSIS_GROUPS, analysis_group_mask, group_protocol_metadata, normalize_analysis_group, read_canonical_bucket
    from ..utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields


TARGET_LAYER_INDICES = tuple(range(12))
TARGET_MODULE_NAMES = ("fc1", "fc2")
FP32_REFERENCE_BUNDLE_FILENAME = "fp32_reference_bundle.json"
FP32_REFERENCE_CORE_FILENAME = "core_eval.pt"
FP32_REFERENCE_OER_ORACLE_FILENAME = "oer_oracle_eval.pt"
REQUIRED_FP32_OER_ORACLE_POSITIONS = ("input", "output")
CSV_FILENAME = "ffn_layer_heatmap.csv"
SUMMARY_CSV_FILENAME = "ffn_layer_heatmap_summary.csv"
PROJECTION_DETAIL_CSV_FILENAME = "ffn_layer_projection_detail.csv"
BASELINE_EMBEDDING_FILENAME = "fp32_baseline_embeddings.pt"
BASELINE_OER_ORACLE_FILENAME = "fp32_oer_oracle_indices.pt"
OER_FEATURE_LAYOUT = "B,T,C"
ANALYSIS_GROUP_NAMES = tuple(ANALYSIS_GROUPS)
COMPLETED_GROUP_NAMES = frozenset(ANALYSIS_GROUP_NAMES)
PROTOCOL_VERSION = "ffn_layer_heatmap.v7"
GROUPING_PROTOCOL_VERSION = 2
PROTOCOL_CSV_COLUMNS = frozenset(("protocol_version", "experiment_signature", "group_protocol_version"))
INPUT_OER_FEATURE_KIND = "input_activation"
PROJECTION_OUTPUT_OER_FEATURE_KIND = "projection_output"
FFN_HIDDEN_ACTIVATION_FEATURE_KIND = "ffn_hidden_activation"
OER_FEATURE_KINDS = (
    INPUT_OER_FEATURE_KIND,
    PROJECTION_OUTPUT_OER_FEATURE_KIND,
    FFN_HIDDEN_ACTIVATION_FEATURE_KIND,
)
OER_ACTIVATION_KIND = PROJECTION_OUTPUT_OER_FEATURE_KIND
MANIFEST_FILENAME = "ffn_layer_heatmap_manifest.json"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def _heatmap_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FFN_LAYER_HEATMAP", {})
    return value if isinstance(value, Mapping) else {}


def _fp32_reference_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_REFERENCE_CACHE", {})
    return value if isinstance(value, Mapping) else {}

def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}


def _quant_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "QUANT", {})
    if not isinstance(value, Mapping):
        raise TypeError("cfg.QUANT must be a mapping for FFN INT4 injection.")
    return value


def _metrics_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "METRICS", {})
    return value if isinstance(value, Mapping) else {}


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
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "ffn_layer_heatmap"


def _fp32_reference_cache_root(cfg: Any) -> Path:
    configured = _nested_get(_fp32_reference_cfg(cfg), "CACHE_ROOT", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "cache" / "fp32_reference"


def _fp32_reference_bundle_path(cfg: Any) -> Path:
    configured = _nested_get(_fp32_reference_cfg(cfg), "BUNDLE_PATH", None)
    if configured is not None:
        return Path(configured)
    return _fp32_reference_cache_root(cfg) / FP32_REFERENCE_BUNDLE_FILENAME


def _resolve_reference_artifact_path(cache_root: Path, value: Any, *, default_filename: str, field_name: str) -> Path:
    raw = default_filename if value is None or str(value).strip() == "" else str(value)
    path = Path(raw)
    if not path.is_absolute():
        path = Path(cache_root) / path
    if not path.exists():
        raise FileNotFoundError(f"FP32 reference bundle {field_name} does not exist: {path}")
    return path

def _device(cfg: Any) -> torch.device:
    heatmap_cfg = _heatmap_cfg(cfg)
    configured = _nested_get(heatmap_cfg, "DEVICE", None)
    if configured is None:
        configured = _cfg_get(cfg, "DEVICE", None)
    if configured is None:
        raise AttributeError("ffn_layer_heatmap requires DEVICE from config.py or FFN_LAYER_HEATMAP['DEVICE'].")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py configured DEVICE={configured!r}, but CUDA is not available")
    return device


def _seed(cfg: Any) -> int:
    heatmap_cfg = _heatmap_cfg(cfg)
    return int(_nested_get(heatmap_cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


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


def _split_csv(cfg: Any, name: str) -> Path:
    heatmap_cfg = _heatmap_cfg(cfg)
    dataset_cfg = _dataset_cfg(cfg)
    explicit_key = f"{name.upper()}_CSV"
    explicit = _nested_get(heatmap_cfg, explicit_key, None)
    if explicit is not None:
        return Path(explicit)
    dataset_value = _nested_get(dataset_cfg, explicit_key, None)
    if dataset_value is not None:
        return Path(dataset_value)
    raise AttributeError(f"Missing {explicit_key}; configure it in cfg.CURRENT_DATASET or cfg.FFN_LAYER_HEATMAP")


def _full_dataset_csv(cfg: Any) -> Path | None:
    heatmap_cfg = _heatmap_cfg(cfg)
    dataset_cfg = _dataset_cfg(cfg)
    for key in ("FULL_CSV", "FULL_DATASET_CSV"):
        explicit = _nested_get(heatmap_cfg, key, None)
        if explicit is not None:
            return Path(explicit)
        dataset_value = _nested_get(dataset_cfg, key, None)
        if dataset_value is not None:
            return Path(dataset_value)
    split_dir = _nested_get(dataset_cfg, "SPLIT_DIR", None)
    if split_dir is not None:
        return Path(split_dir) / "full_dataset.csv"
    return None

def _batch_size(cfg: Any, name: str) -> int:
    heatmap_cfg = _heatmap_cfg(cfg)
    key = f"{name.upper()}_BATCH_SIZE"
    return int(_nested_get(heatmap_cfg, key, _nested_get(heatmap_cfg, "BATCH_SIZE", 4)))


def _num_workers(cfg: Any) -> int:
    return int(_nested_get(_heatmap_cfg(cfg), "NUM_WORKERS", 0))


def _pin_memory(cfg: Any) -> bool:
    default = _device(cfg).type == "cuda"
    return bool(_nested_get(_heatmap_cfg(cfg), "PIN_MEMORY", default))


def _persistent_workers(cfg: Any) -> bool:
    default = _num_workers(cfg) > 0
    return bool(_nested_get(_heatmap_cfg(cfg), "PERSISTENT_WORKERS", default))


def _prefetch_factor(cfg: Any) -> int | None:
    value = _nested_get(_heatmap_cfg(cfg), "PREFETCH_FACTOR", 2)
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("FFN_LAYER_HEATMAP['PREFETCH_FACTOR'] must be positive or None")
    return value


def _resume_enabled(cfg: Any) -> bool:
    return bool(_nested_get(_heatmap_cfg(cfg), "RESUME", True))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value.keys(), key=lambda item: str(item))}
    if isinstance(value, Path):
        return str(value.expanduser().resolve(strict=False))
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() <= 1024:
            return tensor.tolist()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    return repr(value)


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_optional_file(path_value: Any) -> str | None:
    if path_value is None:
        return None
    return _sha256_file(Path(path_value))


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _audio_config_hash(cfg: Any) -> str:
    audio_config = _make_audio_config(cfg)
    payload = getattr(audio_config, "__dict__", audio_config)
    return _stable_json_hash(payload)


def _checkpoint_path_for_reference(cfg: Any) -> Path | None:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    if isinstance(beats_cfg, Mapping):
        for key in ("CHECKPOINT", "checkpoint", "CHECKPOINT_PATH", "checkpoint_path"):
            value = beats_cfg.get(key)
            if value is not None:
                return Path(value)
    for key in ("CHECKPOINT_PATH", "checkpoint_path", "FP32_CHECKPOINT", "TEACHER_CHECKPOINT"):
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return Path(value)
    return None


def _label_mapping_path_for_reference(cfg: Any) -> Path | None:
    for mapping in (_fp32_reference_cfg(cfg), _dataset_cfg(cfg)):
        for key in ("LABEL_MAPPING", "LABEL_MAPPING_PATH", "label_mapping"):
            value = _nested_get(mapping, key, None)
            if value is not None:
                return Path(value)
    return None

def _path_identity(path_value: Any) -> dict[str, Any]:
    if path_value is None:
        return {"path": None, "sha256": None}
    path = Path(path_value)
    return {
        "path": str(path.expanduser().resolve(strict=False)),
        "sha256": _sha256_file(path),
    }


def _experiment_signature_payload(cfg: Any) -> dict[str, Any]:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    group_protocol = group_protocol_metadata(
        tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_name": _dataset_name(cfg),
        "oer": {
            "kind": OER_ACTIVATION_KIND,
            "primary_feature_kind": PROJECTION_OUTPUT_OER_FEATURE_KIND,
            "feature_kinds": list(OER_FEATURE_KINDS),
            "layout": OER_FEATURE_LAYOUT,
            "topk_ratio": _oer_top_ratio(cfg),
        },
        "group_protocol_version": group_protocol["group_protocol_version"],
        "grouping_protocol": dict(group_protocol),
        "grouping": {
            **group_protocol,
            "source_priority": ["eval_split_bucket", "full_dataset_bucket"],
            "sample_bucket_required": True,
            "full_csv": _path_identity(_full_dataset_csv(cfg)),
        },
        "checkpoint": _path_identity(_nested_get(beats_cfg, "CHECKPOINT", None)),
        "eval_csv": _path_identity(_split_csv(cfg, "eval")),
        "calibration_csv": _path_identity(_split_csv(cfg, "calibration")),
        "quant": _jsonable(_quant_cfg(cfg)),
        "observer": _jsonable(_cfg_get(cfg, "OBSERVER", {})),
        "audio": _jsonable(_cfg_get(cfg, "AUDIO", {})),
        "amp_enabled": False,
        "autocast_enabled": False,
        "reference_dtype": "float32",
        "model_dtype": "float32",
        "input_dtype": "float32",
        "beats_use_amp": False,
        "beats_use_amp_config": bool(_nested_get(beats_cfg, "USE_AMP", False)),
        "target_layer_indices": list(TARGET_LAYER_INDICES),
        "target_module_names": list(TARGET_MODULE_NAMES),
    }


def _experiment_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_experiment_manifest(
    output_dir: Path,
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> Path:
    manifest_path = output_dir / MANIFEST_FILENAME
    payload = {
        "protocol_version": protocol_version,
        "experiment_signature": experiment_signature,
        "amp_enabled": False,
        "autocast_enabled": False,
        "reference_dtype": "float32",
        "model_dtype": "float32",
        "input_dtype": "float32",
        "experiment_signature_payload": _jsonable(signature_payload),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path



def _tensor_json_values(value: Any) -> Any:
    tensor = torch.as_tensor(value).detach().cpu().float()
    if int(tensor.numel()) == 0:
        return []
    if int(tensor.numel()) == 1:
        return float(tensor.reshape(-1)[0].item())
    return [float(item) for item in tensor.reshape(-1).tolist()]


def _tensor_shape(value: Any) -> list[int]:
    return [int(item) for item in torch.as_tensor(value).shape]


def _finite_check(value: Any, *, positive: bool = False) -> bool:
    tensor = torch.as_tensor(value).detach().float()
    if int(tensor.numel()) == 0:
        return False
    finite = torch.isfinite(tensor).all()
    if positive:
        finite = finite & (tensor > 0).all()
    return bool(finite.item())


def _observer_count(summary: Mapping[str, Any], key: str) -> int:
    try:
        value = summary.get(key, 0)
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _activation_qparams_summary(quant_module: Any) -> dict[str, Any]:
    scale = getattr(quant_module, "activation_scale", torch.tensor(float("nan"))).detach().cpu().float()
    zero_point = getattr(quant_module, "activation_zero_point", torch.tensor(float("nan"))).detach().cpu().float()
    config = getattr(quant_module, "config", None)
    qmin = int(getattr(config, "activation_qmin", 0)) if config is not None else None
    qmax = int(getattr(config, "activation_qmax", 0)) if config is not None else None
    return {
        "activation_scale": _tensor_json_values(scale),
        "activation_zero_point": _tensor_json_values(zero_point),
        "scale_shape": _tensor_shape(scale),
        "zero_point_shape": _tensor_shape(zero_point),
        "scale_finite_check": _finite_check(scale, positive=True),
        "zero_point_finite_check": _finite_check(zero_point),
        "qmin": qmin,
        "qmax": qmax,
        "qparams_frozen": bool(getattr(quant_module, "qparams_frozen", False)),
        "activation_quant_enabled": bool(getattr(quant_module, "activation_quant_enabled", False)),
    }


def _weight_qparams_for_summary(quant_module: Any) -> tuple[Tensor, Tensor]:
    weight = quant_module.linear.weight.detach()
    if bool(getattr(quant_module, "fixed_weight_qparams_enabled", False)):
        scale = getattr(quant_module, "weight_scale", torch.empty(0, device=weight.device)).detach()
        zero_point = getattr(quant_module, "weight_zero_point", torch.empty(0, device=weight.device)).detach()
        if int(scale.numel()) > 0 and int(zero_point.numel()) > 0:
            return scale.float(), zero_point.float()
    qparams_fn = getattr(quant_module, "_weight_qparams_for", None)
    if not callable(qparams_fn):
        return torch.empty(0, device=weight.device), torch.empty(0, device=weight.device)
    scale, zero_point = qparams_fn(weight)
    return scale.detach().float(), zero_point.detach().float()


def _weight_clip_summary(quant_module: Any, scale: Tensor, zero_point: Tensor) -> dict[str, Any]:
    if int(scale.numel()) == 0 or int(zero_point.numel()) == 0:
        return {
            "weight_clip_rate": float("nan"),
            "weight_clip_rate_by_channel": [],
            "weight_saturation_proxy": float("nan"),
        }
    config = getattr(quant_module, "config", None)
    qmin = int(getattr(config, "weight_qmin", 0)) if config is not None else 0
    qmax = int(getattr(config, "weight_qmax", 0)) if config is not None else 0
    weight = quant_module.linear.weight.detach().float().to(device=scale.device)
    scale_b = scale.to(device=weight.device, dtype=weight.dtype)
    zero_point_b = zero_point.to(device=weight.device, dtype=weight.dtype)
    if scale_b.ndim == 1 and int(scale_b.numel()) == int(weight.shape[0]):
        scale_b = scale_b.reshape(int(weight.shape[0]), 1)
    if zero_point_b.ndim == 1 and int(zero_point_b.numel()) == int(weight.shape[0]):
        zero_point_b = zero_point_b.reshape(int(weight.shape[0]), 1)
    q = weight / torch.clamp(scale_b, min=float(getattr(config, "eps", 1e-8))) + zero_point_b
    clipped = (q < float(qmin)) | (q > float(qmax))
    saturated = (q <= float(qmin)) | (q >= float(qmax))
    per_channel = clipped.float().reshape(int(weight.shape[0]), -1).mean(dim=1)
    return {
        "weight_clip_rate": float(clipped.float().mean().detach().cpu().item()),
        "weight_clip_rate_by_channel": [float(item) for item in per_channel.detach().cpu().reshape(-1).tolist()],
        "weight_saturation_proxy": float(saturated.float().mean().detach().cpu().item()),
    }


def _weight_qparams_summary(quant_module: Any) -> dict[str, Any]:
    scale, zero_point = _weight_qparams_for_summary(quant_module)
    config = getattr(quant_module, "config", None)
    qmin = int(getattr(config, "weight_qmin", 0)) if config is not None else None
    qmax = int(getattr(config, "weight_qmax", 0)) if config is not None else None
    return {
        "weight_scale": _tensor_json_values(scale.detach().cpu()),
        "weight_zero_point": _tensor_json_values(zero_point.detach().cpu()),
        "scale_shape": _tensor_shape(scale),
        "zero_point_shape": _tensor_shape(zero_point),
        "scale_finite_check": _finite_check(scale, positive=True),
        "zero_point_finite_check": _finite_check(zero_point),
        "qmin": qmin,
        "qmax": qmax,
        "weight_quant_enabled": bool(getattr(quant_module, "weight_quant_enabled", False)),
        "fixed_weight_qparams_enabled": bool(getattr(quant_module, "fixed_weight_qparams_enabled", False)),
        **_weight_clip_summary(quant_module, scale, zero_point),
    }


def _target_quant_state_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _result_dir(cfg) / "quant_state" / f"layer_{int(layer_idx):02d}_{module_name}_quant_state.json"


def _save_target_quant_state(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    quant_module: Any,
    quant_mode: str,
) -> Path:
    observer = getattr(quant_module, "activation_observer", None)
    observer_summary = observer.summary() if observer is not None and callable(getattr(observer, "summary", None)) else {}
    activation_qparams = _activation_qparams_summary(quant_module)
    weight_qparams = _weight_qparams_summary(quant_module)
    calibration_sample_count = _observer_count(observer_summary, "range_num_samples")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "target_name": _target_name(int(layer_idx), module_name),
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "quant_mode": str(quant_mode),
        "granularity": _granularity_label(cfg),
        "calibration_sample_count": calibration_sample_count,
        "observer_num_batches": _observer_count(observer_summary, "num_batches"),
        "observer_num_samples": _observer_count(observer_summary, "num_samples"),
        "observer_range_num_batches": _observer_count(observer_summary, "range_num_batches"),
        "observer_range_num_samples": _observer_count(observer_summary, "range_num_samples"),
        "observer_histogram_num_batches": _observer_count(observer_summary, "histogram_num_batches"),
        "observer_histogram_num_samples": _observer_count(observer_summary, "histogram_num_samples"),
        "activation_min": observer_summary.get("min"),
        "activation_max": observer_summary.get("max"),
        "activation_abs_max": observer_summary.get("abs_max"),
        "activation_histogram_summary": {
            "histogram_built": observer_summary.get("histogram_built"),
            "histogram_bins": observer_summary.get("histogram_bins"),
            "histogram_num_elements": observer_summary.get("histogram_num_elements"),
            "percentiles": observer_summary.get("percentiles", {}),
            "abs_percentiles": observer_summary.get("abs_percentiles", {}),
            "percentile_low": observer_summary.get("percentile_low"),
            "percentile_high": observer_summary.get("percentile_high"),
        },
        "observer_summary": _jsonable(observer_summary),
        "activation_qparams": activation_qparams,
        "weight_qparams": weight_qparams,
        "scale_finite_check": {
            "activation": activation_qparams["scale_finite_check"],
            "weight": weight_qparams["scale_finite_check"],
        },
        "zero_point_finite_check": {
            "activation": activation_qparams["zero_point_finite_check"],
            "weight": weight_qparams["zero_point_finite_check"],
        },
        "qmin_qmax": {
            "activation": {"qmin": activation_qparams["qmin"], "qmax": activation_qparams["qmax"]},
            "weight": {"qmin": weight_qparams["qmin"], "qmax": weight_qparams["qmax"]},
        },
    }
    path = _target_quant_state_path(cfg, int(layer_idx), module_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _protocol_group_version(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def _cache_protocol_matches(cache: Mapping[str, Any], protocol_version: str, experiment_signature: str) -> bool:
    return (
        str(cache.get("protocol_version", "")) == str(protocol_version)
        and str(cache.get("experiment_signature", "")) == str(experiment_signature)
        and _protocol_group_version(cache.get("group_protocol_version", GROUPING_PROTOCOL_VERSION)) >= GROUPING_PROTOCOL_VERSION
    )


def _row_protocol_matches(row: Mapping[str, Any], protocol_version: str, experiment_signature: str) -> bool:
    return (
        str(row.get("protocol_version", "")).strip() == str(protocol_version)
        and str(row.get("experiment_signature", "")).strip() == str(experiment_signature)
        and _protocol_group_version(row.get("group_protocol_version", "")) >= GROUPING_PROTOCOL_VERSION
    )


def _filter_rows_for_current_protocol(
    rows: Sequence[Mapping[str, Any]],
    protocol_version: str,
    experiment_signature: str,
) -> list[dict[str, Any]]:
    filtered = [dict(row) for row in rows if _row_protocol_matches(row, protocol_version, experiment_signature)]
    ignored = len(rows) - len(filtered)
    if ignored:
        print(f"[resume] ignored {ignored} CSV rows with mismatched protocol/signature")
    return filtered


def _attach_protocol_fields(
    rows: Sequence[Mapping[str, Any]],
    protocol_version: str,
    experiment_signature: str,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["protocol_version"] = protocol_version
        item["experiment_signature"] = experiment_signature
        item["group_protocol_version"] = GROUPING_PROTOCOL_VERSION
        updated.append(item)
    return updated

def _build_audio_loader(cfg: Any, *, name: str, mode: str) -> DataLoader:
    override_name = f"build_{name}_loader"
    override = _cfg_get(cfg, override_name, None)
    if callable(override):
        return override(cfg)

    dataset = AudioInputDataset(
        _split_csv(cfg, name),
        mode=mode,
        dataset_root=_nested_get(_dataset_cfg(cfg), "ROOT", None),
        project_root=_cfg_get(cfg, "BASE_DIR", None),
        audio_config=_make_audio_config(cfg),
        global_seed=_seed(cfg),
        epoch=0,
        return_dict=True,
        require_label_id=str(name).lower() != "calibration",
        include_audio_stats=False,
    )
    num_workers = _num_workers(cfg)
    loader_kwargs: dict[str, Any] = {
        "batch_size": _batch_size(cfg, name),
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


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
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
            raise TypeError("model output dict does not contain a usable tensor field")
        mask = output.get("padding_mask")
        return tensor, mask if isinstance(mask, Tensor) else None
    if isinstance(output, (tuple, list)) and output:
        tensor = output[0]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"first model output element must be a Tensor, got {type(tensor)!r}")
        mask = output[1] if len(output) > 1 and isinstance(output[1], Tensor) else None
        return tensor, mask
    raise TypeError(f"unsupported model output type: {type(output)!r}")


def _forward_features(model: nn.Module, batch: Mapping[str, Any], cfg: Any) -> tuple[Tensor, Tensor | None]:
    forward_fn = _cfg_get(cfg, "ffn_heatmap_forward", None)
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


def _pool_embedding(features: Tensor, padding_mask: Tensor | None) -> Tensor:
    if features.ndim == 2:
        return features.detach()
    if features.ndim == 3:
        if padding_mask is not None and tuple(padding_mask.shape) == tuple(features.shape[:2]):
            valid = (~padding_mask.to(device=features.device, dtype=torch.bool)).unsqueeze(-1)
            summed = features.masked_fill(~valid, 0.0).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1).to(dtype=features.dtype)
            return (summed / denom).detach()
        return features.mean(dim=1).detach()
    if features.ndim > 3:
        return features.flatten(start_dim=1).detach()
    return features.reshape(features.shape[0], -1).detach()


def _autocast_context(cfg: Any, device: torch.device) -> Any:
    del cfg
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def _collect_embeddings(model: nn.Module, loader: DataLoader, cfg: Any) -> dict[str, Any]:
    device = _device(cfg)
    model.eval()
    embeddings: list[Tensor] = []
    labels: list[Tensor] = []
    sample_ids: list[str] = []
    batch_sizes: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(cfg, device):
                features, feature_padding_mask = _forward_features(model, batch, cfg)
            embeddings.append(_pool_embedding(features.float(), feature_padding_mask).cpu())
            batch_label = batch["label_id"].detach().cpu().long()
            labels.append(batch_label)
            batch_sizes.append(int(batch_label.numel()))
            sample_ids.extend(str(item) for item in batch["sample_id"])

    if not embeddings:
        raise RuntimeError("eval_loader produced no batches")
    return {
        "sample_ids": sample_ids,
        "labels": torch.cat(labels, dim=0),
        "embeddings": torch.cat(embeddings, dim=0),
        "batch_sizes": batch_sizes,
    }



def _first_tensor_from_hook_inputs(args: tuple[Any, ...], kwargs: Mapping[str, Any], target_name: Any) -> Tensor:
    if args and isinstance(args[0], Tensor):
        return args[0]
    for key in ("x", "input"):
        value = kwargs.get(key)
        if isinstance(value, Tensor):
            return value
    for value in kwargs.values():
        if isinstance(value, Tensor):
            return value
    raise RuntimeError(f"{target_name}: unable to capture Tensor input from hook")


def _fp32_oer_oracle_chunk(
    feature: Tensor,
    sample_ids: Sequence[str],
    layer_idx: int,
    module_name: str,
    topk_ratio: float,
    feature_kind: str,
) -> dict[str, Any]:
    batch_size = len(sample_ids)
    target_name = _target_name(int(layer_idx), module_name)
    canonical = _activation_to_btc_with_layout(feature, batch_size, f"{target_name}.fp32_{feature_kind}")
    x = canonical.tensor
    public_layout = layout_info_to_public_fields(canonical.layout_info)
    flat = x.reshape(batch_size, -1)
    topk_indices = scalar_oer_topk_indices(x, topk_ratio=topk_ratio, batch_dim=0).detach().long()
    selected = torch.gather(flat, dim=1, index=topk_indices.to(device=flat.device))
    metadata = scalar_oer_topk_index_metadata(
        x,
        sample_id=None,
        layer_idx=int(layer_idx),
        module_name=module_name,
        topk_indices=topk_indices,
        batch_dim=0,
        token_dim=1,
        channel_dim=2,
    )
    return {
        "feature_kind": str(feature_kind),
        "activation_kind": str(feature_kind),
        "topk_indices": topk_indices.detach().cpu().long(),
        "flat_index": topk_indices.detach().cpu().long(),
        "token_idx": metadata["token_idx"].detach().cpu().long(),
        "channel_idx": metadata["channel_idx"].detach().cpu().long(),
        "sample_index": metadata["sample_index"].detach().cpu().long(),
        "fp32_oracle_energy": selected.square().sum(dim=1).detach().cpu(),
        "fp32_total_energy": flat.square().sum(dim=1).detach().cpu(),
        "activation_shape": tuple(int(item) for item in x.shape),
        "original_layout": public_layout["original_layout"],
        "original_shape": public_layout["original_shape"],
        "canonical_layout": public_layout["canonical_layout"],
        "canonical_shape": public_layout["canonical_shape"],
        "layout_inference_source": public_layout["layout_inference_source"],
        "padding_mask_available": public_layout["padding_mask_available"],
        "padding_mask_aligned": public_layout["padding_mask_aligned"],
        "mask_status": public_layout["mask_status"],
        "topk_ratio": float(topk_ratio),
    }


def _merge_oer_oracle_chunks(
    chunks: Sequence[Mapping[str, Any]],
    layer_idx: int,
    module_name: str,
    feature_kind: str,
) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError(f"L{int(layer_idx)}.{module_name}: no FP32 OER oracle chunks captured for {feature_kind}")
    activation_shapes = [tuple(chunk["activation_shape"]) for chunk in chunks]
    nonbatch_shapes = {shape[1:] for shape in activation_shapes}
    if len(nonbatch_shapes) != 1:
        raise RuntimeError(f"L{int(layer_idx)}.{module_name}: OER {feature_kind} non-batch shape mismatch: {activation_shapes}")
    topk_indices = torch.cat([chunk["topk_indices"] for chunk in chunks], dim=0)
    total_batch = int(topk_indices.shape[0])
    sample_index = torch.arange(total_batch, dtype=torch.long).reshape(-1, 1).expand_as(topk_indices)
    first_shape = activation_shapes[0]
    return {
        "layout": OER_FEATURE_LAYOUT,
        "contains_full_activation": False,
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "target_name": _target_name(int(layer_idx), module_name),
        "activation_kind": str(feature_kind),
        "feature_kind": str(feature_kind),
        "activation_shape": (total_batch, *first_shape[1:]),
        "original_layout": chunks[0].get("original_layout", ""),
        "original_shape": chunks[0].get("original_shape", ""),
        "canonical_layout": "BTC",
        "canonical_shape": (total_batch, *first_shape[1:]),
        "layout_inference_source": chunks[0].get("layout_inference_source", ""),
        "padding_mask_available": bool(chunks[0].get("padding_mask_available", False)),
        "padding_mask_aligned": bool(chunks[0].get("padding_mask_aligned", False)),
        "mask_status": chunks[0].get("mask_status", ""),
        "topk_ratio": float(chunks[0]["topk_ratio"]),
        "topk_indices": topk_indices,
        "flat_index": topk_indices,
        "sample_index": sample_index,
        "token_idx": torch.cat([chunk["token_idx"] for chunk in chunks], dim=0),
        "channel_idx": torch.cat([chunk["channel_idx"] for chunk in chunks], dim=0),
        "batch_dim": 0,
        "token_dim": 1,
        "channel_dim": 2,
        "fp32_oracle_energy": torch.cat([chunk["fp32_oracle_energy"] for chunk in chunks], dim=0),
        "fp32_total_energy": torch.cat([chunk["fp32_total_energy"] for chunk in chunks], dim=0),
    }


def _merge_target_oer_oracles(
    chunks_by_kind: Mapping[str, Sequence[Mapping[str, Any]]],
    layer_idx: int,
    module_name: str,
) -> dict[str, Any]:
    feature_oracles = {
        feature_kind: _merge_oer_oracle_chunks(chunks_by_kind[feature_kind], int(layer_idx), module_name, feature_kind)
        for feature_kind in OER_FEATURE_KINDS
    }
    primary = dict(feature_oracles[PROJECTION_OUTPUT_OER_FEATURE_KIND])
    primary["feature_oracles"] = feature_oracles
    primary["feature_kinds"] = list(OER_FEATURE_KINDS)
    primary["primary_feature_kind"] = PROJECTION_OUTPUT_OER_FEATURE_KIND
    return primary


def _register_target_fp32_oer_hooks(
    model: nn.Module,
    layer_idx: int,
    module_name: str,
    chunks_by_kind: dict[str, list[dict[str, Any]]],
    state: dict[str, Any],
    topk_ratio: float,
) -> list[Any]:
    layer = model.encoder.layers[int(layer_idx)]
    module = getattr(layer, module_name)
    fc2_module = getattr(layer, "fc2")

    def sample_ids_for_hook(feature_kind: str) -> Sequence[str]:
        sample_ids = state.get("sample_ids")
        if sample_ids is None:
            raise RuntimeError(f"{_target_name(layer_idx, module_name)}: missing batch sample_id for {feature_kind} OER hook")
        return sample_ids

    def record_capture(feature_kind: str) -> None:
        capture_counts = state.setdefault("capture_counts", {})
        capture_counts[feature_kind] = int(capture_counts.get(feature_kind, 0)) + 1

    def target_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        del _module
        target_name = _target_name(layer_idx, module_name)
        sample_ids = sample_ids_for_hook(INPUT_OER_FEATURE_KIND)
        input_activation = _first_tensor_from_hook_inputs(args, kwargs, f"{target_name}.input_activation")
        chunks_by_kind[INPUT_OER_FEATURE_KIND].append(
            _fp32_oer_oracle_chunk(input_activation, sample_ids, int(layer_idx), module_name, topk_ratio, INPUT_OER_FEATURE_KIND)
        )
        record_capture(INPUT_OER_FEATURE_KIND)
        if not isinstance(output, Tensor):
            raise RuntimeError(f"{target_name}: unable to capture FFN projection output")
        chunks_by_kind[PROJECTION_OUTPUT_OER_FEATURE_KIND].append(
            _fp32_oer_oracle_chunk(output, sample_ids, int(layer_idx), module_name, topk_ratio, PROJECTION_OUTPUT_OER_FEATURE_KIND)
        )
        record_capture(PROJECTION_OUTPUT_OER_FEATURE_KIND)

    def hidden_pre_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del _module
        target_name = _target_name(layer_idx, module_name)
        sample_ids = sample_ids_for_hook(FFN_HIDDEN_ACTIVATION_FEATURE_KIND)
        hidden_activation = _first_tensor_from_hook_inputs(args, kwargs, f"{target_name}.ffn_hidden_activation")
        chunks_by_kind[FFN_HIDDEN_ACTIVATION_FEATURE_KIND].append(
            _fp32_oer_oracle_chunk(hidden_activation, sample_ids, int(layer_idx), module_name, topk_ratio, FFN_HIDDEN_ACTIVATION_FEATURE_KIND)
        )
        record_capture(FFN_HIDDEN_ACTIVATION_FEATURE_KIND)

    return [
        module.register_forward_hook(target_hook, with_kwargs=True),
        fc2_module.register_forward_pre_hook(hidden_pre_hook, with_kwargs=True),
    ]


def _remove_hooks(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


def _release_memory(device: torch.device | None = None) -> None:
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

def _activation_to_btc(activation: Tensor, batch_size: int, target_name: Any) -> Tensor:
    """Strict wrapper around the shared BTC layout adapter."""

    x = canonicalize_activation_btc(
        activation.detach().float(),
        batch_size=int(batch_size),
        context=str(target_name),
    ).tensor
    return _assert_btc_feature_layout(x, batch_size=batch_size, target_name=target_name)


def _activation_to_btc_with_layout(activation: Tensor, batch_size: int, target_name: Any):
    return canonicalize_activation_btc(
        activation.detach().float(),
        batch_size=int(batch_size),
        context=str(target_name),
    )


def _assert_btc_feature_layout(feature: Tensor, *, batch_size: int | None, target_name: Any) -> Tensor:
    """Validate the canonical [B, T, C] OER feature layout."""

    if feature.ndim != 3:
        raise ValueError(f"{target_name}: OER feature must use {OER_FEATURE_LAYOUT}, got {tuple(feature.shape)}")
    if batch_size is not None and int(feature.shape[0]) != int(batch_size):
        raise ValueError(
            f"{target_name}: OER feature first dim must equal batch/sample count, got {int(feature.shape[0])} vs {int(batch_size)}"
        )
    return feature.contiguous()


def _collect_target_fp32_oer_oracle(
    cfg: Any,
    eval_loader: DataLoader,
    layer_idx: int,
    module_name: str,
) -> dict[str, Any]:
    device = _device(cfg)
    model = restore_clean_model_from_checkpoint(cfg).to(device)
    model.float()
    model.eval()
    chunks_by_kind: dict[str, list[dict[str, Any]]] = {feature_kind: [] for feature_kind in OER_FEATURE_KINDS}
    sample_ids: list[str] = []
    labels: list[Tensor] = []
    state: dict[str, Any] = {"sample_ids": None, "capture_counts": {}}
    handles = _register_target_fp32_oer_hooks(
        model,
        int(layer_idx),
        module_name,
        chunks_by_kind,
        state,
        _oer_top_ratio(cfg),
    )
    try:
        with torch.no_grad():
            for batch in eval_loader:
                batch_sample_ids = [str(item) for item in batch["sample_id"]]
                batch_label = batch["label_id"].detach().cpu().long().reshape(-1)
                if len(batch_sample_ids) != int(batch_label.numel()):
                    raise RuntimeError("sample_id count must match label_id count for FP32 OER oracle")
                state["sample_ids"] = batch_sample_ids
                state["capture_counts"] = {}
                moved = _move_batch_to_device(batch, device)
                with _autocast_context(cfg, device):
                    _forward_features(model, moved, cfg)
                capture_counts = state.get("capture_counts", {})
                for feature_kind in OER_FEATURE_KINDS:
                    if int(capture_counts.get(feature_kind, 0)) != 1:
                        raise RuntimeError(
                            f"{_target_name(layer_idx, module_name)}: expected one FP32 {feature_kind} OER capture per batch, "
                            f"got {capture_counts.get(feature_kind, 0)}"
                        )
                sample_ids.extend(batch_sample_ids)
                labels.append(batch_label)
                state["sample_ids"] = None
                state["capture_counts"] = {}
    finally:
        _remove_hooks(handles)
        del model
        _release_memory(device)

    if not labels:
        raise RuntimeError("eval_loader produced no batches for FP32 OER oracle")
    oracle = _merge_target_oer_oracles(chunks_by_kind, int(layer_idx), module_name)
    oracle["sample_id_order"] = sample_ids
    oracle["label_id"] = torch.cat(labels, dim=0)
    for feature_oracle in oracle["feature_oracles"].values():
        feature_oracle["sample_id_order"] = sample_ids
        feature_oracle["label_id"] = oracle["label_id"]
    return oracle



def _oer_feature_oracle(oracle_cache: Mapping[str, Any], feature_kind: str) -> Mapping[str, Any]:
    feature_oracles = oracle_cache.get("feature_oracles")
    if isinstance(feature_oracles, Mapping) and feature_kind in feature_oracles:
        feature_oracle = feature_oracles[feature_kind]
        if not isinstance(feature_oracle, Mapping):
            raise TypeError(f"OER feature oracle {feature_kind!r} must be a mapping")
        return feature_oracle
    if feature_kind == PROJECTION_OUTPUT_OER_FEATURE_KIND and "topk_indices" in oracle_cache:
        return oracle_cache
    raise KeyError(f"OER oracle missing feature_kind={feature_kind!r}")


def _quant_oer_energy_chunk(
    quant_feature: Tensor,
    oracle_cache: Mapping[str, Any],
    start_index: int,
    batch_size: int,
    target_name: Any,
) -> dict[str, Tensor]:
    x = _activation_to_btc(quant_feature, int(batch_size), target_name)
    expected_shape = tuple(oracle_cache.get("activation_shape", ()))
    if expected_shape and tuple(int(item) for item in x.shape[1:]) != expected_shape[1:]:
        raise ValueError(
            f"{target_name}: quant feature non-batch shape {tuple(x.shape[1:])} "
            f"does not match FP32 oracle shape {expected_shape[1:]}"
        )
    flat = x.reshape(int(batch_size), -1)
    indices_all = oracle_cache["topk_indices"]
    start = int(start_index)
    stop = start + int(batch_size)

    if isinstance(indices_all, Tensor):
        indices = indices_all[start:stop].to(device=flat.device, dtype=torch.long).reshape(int(batch_size), -1)
        if int(indices.numel()) == 0:
            raise ValueError(f"{target_name}: empty OER top-k indices")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(flat.shape[1]):
            raise ValueError(f"{target_name}: OER top-k indices out of range for quant feature")
        selected = torch.gather(flat, dim=1, index=indices)
        quant_oracle_energy = selected.square().sum(dim=1).detach().cpu()
    else:
        if not isinstance(indices_all, Sequence) or isinstance(indices_all, (str, bytes)):
            raise TypeError(f"{target_name}: OER top-k indices must be a Tensor or per-sample sequence")
        rows = list(indices_all)[start:stop]
        if len(rows) != int(batch_size):
            raise ValueError(f"{target_name}: OER top-k index slice length {len(rows)} does not match batch_size={batch_size}")
        energies: list[Tensor] = []
        for row_idx, row_indices in enumerate(rows):
            indices = torch.as_tensor(row_indices, device=flat.device, dtype=torch.long).reshape(-1)
            if int(indices.numel()) == 0:
                raise ValueError(f"{target_name}: empty OER top-k indices for batch row {row_idx}")
            if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(flat.shape[1]):
                raise ValueError(f"{target_name}: OER top-k indices out of range for batch row {row_idx}")
            selected = flat[row_idx].index_select(0, indices)
            energies.append(selected.square().sum().detach().cpu().reshape(()))
        quant_oracle_energy = torch.stack(energies, dim=0)

    return {
        "quant_oracle_energy": quant_oracle_energy,
        "quant_total_energy": flat.square().sum(dim=1).detach().cpu(),
    }

def _csv_value_present(row: Mapping[str, Any], key: str) -> bool:
    if key not in row:
        return False
    value = row.get(key)
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "null", "<na>"}


def _sample_id_from_metadata_row(row: Mapping[str, Any]) -> str | None:
    for key in ("sample_id", "audio_id", "file_id"):
        if _csv_value_present(row, key):
            return str(row.get(key)).strip()
    return None


def _bucket_lookup_from_csv(path: Path | None, *, context: str) -> dict[str, str]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"{context}: metadata CSV has no header: {path}")
        fields = set(reader.fieldnames)
        if "bucket" not in fields:
            if "shot_group" in fields:
                raise RuntimeError(f"{context}: shot_group is present but sample-level bucket is missing: {path}")
            raise RuntimeError(f"{context}: sample-level bucket is required: {path}")
        for row_idx, row in enumerate(reader, start=2):
            sample_id = _sample_id_from_metadata_row(row)
            if sample_id is None:
                continue
            shot_group = [row.get("shot_group")] if _csv_value_present(row, "shot_group") else None
            train_label_n = [row.get("train_label_n")] if _csv_value_present(row, "train_label_n") else None
            result = read_canonical_bucket(
                bucket=[row.get("bucket")],
                shot_group=shot_group,
                train_label_n=train_label_n,
                context=f"{context}:{path}:{row_idx}:sample_id={sample_id}",
                allow_shot_group_fallback=False,
            )
            bucket = result.bucket[0]
            previous = lookup.get(sample_id)
            if previous is not None and previous != bucket:
                raise RuntimeError(
                    f"{context}: sample_id={sample_id!r} has conflicting bucket values "
                    f"{previous!r} vs {bucket!r}"
                )
            lookup[sample_id] = bucket
    return lookup


def _load_buckets_for_sample_order(sample_ids: Sequence[str], cfg: Any) -> list[str]:
    eval_lookup = _bucket_lookup_from_csv(_split_csv(cfg, "eval"), context="eval")
    buckets: list[str | None] = [eval_lookup.get(str(sample_id)) for sample_id in sample_ids]
    missing_indices = [idx for idx, bucket in enumerate(buckets) if bucket is None]
    if missing_indices:
        full_path = _full_dataset_csv(cfg)
        if full_path is not None and Path(full_path) != _split_csv(cfg, "eval"):
            full_lookup = _bucket_lookup_from_csv(full_path, context="full_dataset")
            for idx in missing_indices:
                buckets[idx] = full_lookup.get(str(sample_ids[idx]))
        missing_indices = [idx for idx, bucket in enumerate(buckets) if bucket is None]
    if missing_indices:
        missing = [str(sample_ids[idx]) for idx in missing_indices[:8]]
        raise RuntimeError(f"missing sample-level bucket for eval samples: {missing}")
    return [str(bucket) for bucket in buckets]


def _sample_bucket_mask(buckets: Sequence[str], analysis_group: str) -> Tensor:
    mask = analysis_group_mask(buckets, analysis_group)
    return torch.as_tensor(mask, dtype=torch.bool).reshape(-1)


def _analysis_group_class_labels(labels: Tensor, buckets: Sequence[str]) -> dict[str, Tensor]:
    label_list = [int(item) for item in labels.detach().cpu().long().reshape(-1).tolist()]
    if len(label_list) != len(buckets):
        raise RuntimeError(f"bucket length={len(buckets)} must match label count={len(label_list)}")
    class_to_bucket: dict[int, str] = {}
    for label_id, bucket in zip(label_list, buckets):
        previous = class_to_bucket.get(int(label_id))
        if previous is not None and previous != str(bucket):
            raise RuntimeError(
                f"label_id={label_id} has conflicting sample-level buckets: {previous!r} vs {bucket!r}"
            )
        class_to_bucket[int(label_id)] = str(bucket)

    class_items = sorted(class_to_bucket.items())
    class_ids = [label_id for label_id, _bucket in class_items]
    class_buckets = [bucket for _label_id, bucket in class_items]

    groups: dict[str, Tensor] = {}
    for analysis_group in ANALYSIS_GROUP_NAMES:
        group_mask = analysis_group_mask(class_buckets, analysis_group)
        selected = [label_id for label_id, keep in zip(class_ids, group_mask) if bool(keep)]
        groups[analysis_group] = torch.tensor(selected, dtype=torch.long)
    return groups


def _analysis_group_sample_counts(buckets: Sequence[str]) -> dict[str, int]:
    return {group: int(_sample_bucket_mask(buckets, group).sum().item()) for group in ANALYSIS_GROUP_NAMES}


def _nan_scalar_like(reference: Tensor) -> Tensor:
    return reference.detach().float().new_tensor(float("nan"))


def _compute_group_margin_or_nan(embeddings: Tensor, labels: Tensor, group_labels: Tensor) -> Tensor:
    if int(group_labels.numel()) == 0:
        return _nan_scalar_like(embeddings)
    return compute_group_margin(embeddings, labels, group_labels)


def _compute_group_margin_samples_or_empty(embeddings: Tensor, labels: Tensor, group_labels: Tensor) -> Tensor:
    if int(group_labels.numel()) == 0:
        return embeddings.detach().float().new_empty((0,))
    return compute_group_margin(embeddings, labels, group_labels, reduction="none").reshape(-1)


def _compute_group_fisher_or_nan(embeddings: Tensor, labels: Tensor, group_labels: Tensor) -> Tensor:
    if int(group_labels.numel()) < 2:
        return _nan_scalar_like(embeddings)
    return compute_group_fisher(embeddings, labels, group_labels)


def _oer_top_ratio(cfg: Any) -> float:
    metrics = _metrics_cfg(cfg)
    return float(metrics.get("SCALAR_OER_TOP_RATIO", _cfg_get(cfg, "OER_SCALAR_TOP_RATIO", 0.001)))


def run_fp32_baseline(cfg: Any, model: nn.Module, eval_loader: DataLoader) -> dict[str, Any]:
    """Run FP32 embedding baseline once; OER oracle is built per target."""

    device = _device(cfg)
    model.to(device)
    model.float()
    model.eval()
    cache = _collect_embeddings(model, eval_loader, cfg)

    labels = cache["labels"]
    buckets = _load_buckets_for_sample_order(cache["sample_ids"], cfg)
    analysis_group_labels = _analysis_group_class_labels(labels, buckets)
    sample_counts = _analysis_group_sample_counts(buckets)
    margin_fp32_by_group: dict[str, Tensor] = {}
    fisher_fp32_by_group: dict[str, Tensor] = {}
    effective_class_count_by_group: dict[str, int] = {}
    for analysis_group in ANALYSIS_GROUP_NAMES:
        group_labels = analysis_group_labels[analysis_group]
        margin_fp32_by_group[analysis_group] = _compute_group_margin_samples_or_empty(cache["embeddings"], labels, group_labels)
        fisher_fp32_by_group[analysis_group] = _compute_group_fisher_or_nan(cache["embeddings"], labels, group_labels)
        effective_class_count_by_group[analysis_group] = compute_effective_class_count(labels, group_labels)

    cache.update(
        {
            "bucket": buckets,
            "analysis_group_labels": analysis_group_labels,
            "analysis_group_sample_counts": sample_counts,
            "effective_class_count_by_group": effective_class_count_by_group,
            "margin_fp32_by_group": margin_fp32_by_group,
            "fisher_fp32_by_group": fisher_fp32_by_group,
        }
    )
    return cache

def _target_name(layer_idx: int, module_name: str) -> str:
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _run_forward_only(model: nn.Module, loader: DataLoader, cfg: Any) -> None:
    device = _device(cfg)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(cfg, device):
                _forward_features(model, batch, cfg)


def _run_two_pass_calibration(controller: Any, model: nn.Module, loader: DataLoader, cfg: Any) -> None:
    if controller.activation_quant_enabled:
        controller.prepare_calibration_range(reset_observers=True)
        _run_forward_only(model, loader, cfg)
        controller.prepare_calibration_histogram()
        _run_forward_only(model, loader, cfg)
        controller.freeze_qparams()
    controller.prepare_quant_forward()


def _clear_quant_module_oer_cache(quant_module: Any) -> None:
    clear_fn = getattr(quant_module, "clear_oer_projection_tensors", None)
    if not callable(clear_fn):
        clear_fn = getattr(quant_module, "clear_oer_activation_tensors", None)
    if callable(clear_fn):
        clear_fn()
        return
    for name in (
        "last_fp32_activation_for_oer",
        "last_quant_activation_for_oer",
        "last_activation_valid_mask_for_oer",
        "last_fp32_projection_for_oer",
        "last_quant_projection_for_oer",
    ):
        if hasattr(quant_module, name):
            setattr(quant_module, name, None)


def _append_metric_chunks(accumulator: dict[str, list[Tensor]], metrics: Mapping[str, Tensor]) -> None:
    for key in accumulator:
        value = torch.as_tensor(metrics[key]).detach().cpu().float()
        if key == "error_histogram":
            accumulator[key].append(value.reshape(int(value.shape[0]), -1))
        else:
            accumulator[key].append(value.reshape(-1))


def _finalize_metric_chunks(accumulator: Mapping[str, Sequence[Tensor]]) -> dict[str, Tensor]:
    return {key: torch.cat(list(chunks), dim=0) for key, chunks in accumulator.items()}


def _append_oer_energy_chunk(accumulator: dict[str, list[Tensor]], chunk: Mapping[str, Tensor]) -> None:
    accumulator["quant_oracle_energy"].append(chunk["quant_oracle_energy"].detach().cpu().float().reshape(-1))
    accumulator["quant_total_energy"].append(chunk["quant_total_energy"].detach().cpu().float().reshape(-1))


def _finalize_oer_energy_chunks(accumulator: Mapping[str, Sequence[Tensor]]) -> dict[str, Tensor]:
    return {
        "quant_oracle_energy": torch.cat(list(accumulator["quant_oracle_energy"]), dim=0),
        "quant_total_energy": torch.cat(list(accumulator["quant_total_energy"]), dim=0),
    }


def _ffn_hidden_from_fc1_projection(layer: nn.Module, projection: Tensor) -> Tensor:
    if str(getattr(layer, "activation_name", "")) == "glu":
        hidden = projection
    else:
        activation_fn = getattr(layer, "activation_fn", None)
        if not callable(activation_fn):
            raise RuntimeError("FFN layer is missing callable activation_fn for hidden activation reconstruction")
        hidden = activation_fn(projection)
    dropout2 = getattr(layer, "dropout2", None)
    if callable(dropout2):
        hidden = dropout2(hidden)
    return hidden


def _collect_quant_eval(
    model: nn.Module,
    loader: DataLoader,
    cfg: Any,
    quant_module: Any,
    oer_oracle: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
) -> dict[str, Any]:
    device = _device(cfg)
    model.eval()
    embeddings: list[Tensor] = []
    labels: list[Tensor] = []
    sample_ids: list[str] = []
    oer_energy_chunks: dict[str, dict[str, list[Tensor]]] = {
        feature_kind: {"quant_oracle_energy": [], "quant_total_energy": []}
        for feature_kind in OER_FEATURE_KINDS
    }
    projection_distortion_chunks: dict[str, list[Tensor]] = {
        "l2_rel_error": [],
        "cosine_drift": [],
        "mse": [],
        "mae": [],
        "error_p95": [],
        "error_p99": [],
        "channel_error_mean": [],
        "token_error_mean": [],
        "near_zero_rate": [],
        "error_histogram": [],
    }
    hidden_distortion_chunks: dict[str, list[Tensor]] = {
        "l2_rel_error": [],
        "cosine_drift": [],
        "mse": [],
        "mae": [],
        "error_p95": [],
        "error_p99": [],
        "channel_error_mean": [],
        "token_error_mean": [],
        "near_zero_rate": [],
        "error_histogram": [],
    }
    clipping_rates: list[Tensor] = []
    clipping_rate_by_sample: list[Tensor] = []
    batch_start = 0
    layer = model.encoder.layers[int(layer_idx)]
    hidden_state: dict[str, Any] = {"tensor": None, "capture_count": 0}

    def hidden_pre_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del _module
        hidden_state["tensor"] = _first_tensor_from_hook_inputs(
            args,
            kwargs,
            f"{_target_name(layer_idx, module_name)}.quant_ffn_hidden_activation",
        ).detach()
        hidden_state["capture_count"] = int(hidden_state.get("capture_count", 0)) + 1

    hidden_handle = layer.fc2.register_forward_pre_hook(hidden_pre_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch_to_device(batch, device)
                hidden_state["tensor"] = None
                hidden_state["capture_count"] = 0
                with _autocast_context(cfg, device):
                    features, feature_padding_mask = _forward_features(model, batch, cfg)
                embeddings.append(_pool_embedding(features.float(), feature_padding_mask).cpu())
                batch_label = batch["label_id"].detach().cpu().long()
                labels.append(batch_label)
                batch_size = int(batch_label.numel())
                sample_ids.extend(str(item) for item in batch["sample_id"])

                get_activation_tensors = getattr(quant_module, "get_oer_activation_tensors", None)
                if not callable(get_activation_tensors):
                    raise RuntimeError(f"{quant_module.target_name}: missing OER activation tensor API")
                fp32_input, quant_input = get_activation_tensors()
                if fp32_input is None or quant_input is None:
                    raise RuntimeError(f"{quant_module.target_name}: missing OER input activation after fake quant eval")

                get_projection_tensors = getattr(quant_module, "get_oer_projection_tensors", None)
                if not callable(get_projection_tensors):
                    raise RuntimeError(f"{quant_module.target_name}: missing OER projection tensor API")
                fp32_projection, quant_projection = get_projection_tensors()
                if fp32_projection is None or quant_projection is None:
                    raise RuntimeError(f"{quant_module.target_name}: missing projection tensors after fake quant eval")

                input_energy = _quant_oer_energy_chunk(
                    quant_input,
                    _oer_feature_oracle(oer_oracle, INPUT_OER_FEATURE_KIND),
                    batch_start,
                    batch_size,
                    f"{quant_module.target_name}.quant_input_activation",
                )
                _append_oer_energy_chunk(oer_energy_chunks[INPUT_OER_FEATURE_KIND], input_energy)

                projection_energy = _quant_oer_energy_chunk(
                    quant_projection,
                    _oer_feature_oracle(oer_oracle, PROJECTION_OUTPUT_OER_FEATURE_KIND),
                    batch_start,
                    batch_size,
                    f"{quant_module.target_name}.quant_projection_output",
                )
                _append_oer_energy_chunk(oer_energy_chunks[PROJECTION_OUTPUT_OER_FEATURE_KIND], projection_energy)

                fp32_projection_btc = _activation_to_btc(fp32_projection, batch_size, f"{quant_module.target_name}.fp32_projection_output")
                quant_projection_btc = _activation_to_btc(quant_projection, batch_size, f"{quant_module.target_name}.quant_projection_output")
                _append_metric_chunks(
                    projection_distortion_chunks,
                    compute_tensor_distortion_metrics(fp32_projection_btc, quant_projection_btc),
                )

                if module_name == "fc2":
                    fp32_hidden = fp32_input
                    quant_hidden = quant_input
                else:
                    if int(hidden_state.get("capture_count", 0)) != 1 or not isinstance(hidden_state.get("tensor"), Tensor):
                        raise RuntimeError(
                            f"{quant_module.target_name}: expected one quant FFN hidden activation capture, "
                            f"got {hidden_state.get('capture_count', 0)}"
                        )
                    fp32_hidden = _ffn_hidden_from_fc1_projection(layer, fp32_projection)
                    quant_hidden = hidden_state["tensor"]

                hidden_energy = _quant_oer_energy_chunk(
                    quant_hidden,
                    _oer_feature_oracle(oer_oracle, FFN_HIDDEN_ACTIVATION_FEATURE_KIND),
                    batch_start,
                    batch_size,
                    f"{quant_module.target_name}.quant_ffn_hidden_activation",
                )
                _append_oer_energy_chunk(oer_energy_chunks[FFN_HIDDEN_ACTIVATION_FEATURE_KIND], hidden_energy)

                fp32_hidden_btc = _activation_to_btc(fp32_hidden, batch_size, f"{quant_module.target_name}.fp32_ffn_hidden_activation")
                quant_hidden_btc = _activation_to_btc(quant_hidden, batch_size, f"{quant_module.target_name}.quant_ffn_hidden_activation")
                _append_metric_chunks(
                    hidden_distortion_chunks,
                    compute_tensor_distortion_metrics(fp32_hidden_btc, quant_hidden_btc),
                )

                batch_start += batch_size
                clipping_rate = getattr(quant_module, "last_activation_clipping_rate", None)
                clipping_by_sample = getattr(quant_module, "last_activation_clipping_rate_by_sample", None)
                if isinstance(clipping_by_sample, Tensor) and int(clipping_by_sample.numel()) == batch_size:
                    clipping_rate_by_sample.append(clipping_by_sample.detach().float().cpu().reshape(-1))
                elif isinstance(clipping_rate, Tensor):
                    scalar_clipping = clipping_rate.detach().float().cpu().reshape(())
                    clipping_rates.append(scalar_clipping)
                    clipping_rate_by_sample.append(torch.full((batch_size,), float(scalar_clipping.item()), dtype=torch.float32))
                _clear_quant_module_oer_cache(quant_module)
    finally:
        hidden_handle.remove()

    if not embeddings:
        raise RuntimeError("eval_loader produced no batches")
    expected_samples = int(torch.as_tensor(_oer_feature_oracle(oer_oracle, PROJECTION_OUTPUT_OER_FEATURE_KIND)["fp32_oracle_energy"]).numel())
    if int(batch_start) != expected_samples:
        raise RuntimeError(f"quant eval sample count {batch_start} does not match FP32 oracle {expected_samples}")
    for feature_kind in OER_FEATURE_KINDS:
        feature_expected = int(torch.as_tensor(_oer_feature_oracle(oer_oracle, feature_kind)["fp32_oracle_energy"]).numel())
        if feature_expected != int(batch_start):
            raise RuntimeError(f"{feature_kind} FP32 oracle sample count {feature_expected} does not match quant eval {batch_start}")
    clipping = torch.cat(clipping_rate_by_sample, dim=0).mean() if clipping_rate_by_sample else (torch.stack(clipping_rates).mean() if clipping_rates else torch.tensor(float("nan")))
    oer_energy_by_feature = {
        feature_kind: _finalize_oer_energy_chunks(oer_energy_chunks[feature_kind])
        for feature_kind in OER_FEATURE_KINDS
    }
    return {
        "sample_ids": sample_ids,
        "labels": torch.cat(labels, dim=0),
        "embeddings": torch.cat(embeddings, dim=0),
        "oer_energy": oer_energy_by_feature[PROJECTION_OUTPUT_OER_FEATURE_KIND],
        "oer_energy_by_feature": oer_energy_by_feature,
        "oer_feature_layout": OER_FEATURE_LAYOUT,
        "target_projection_distortion": _finalize_metric_chunks(projection_distortion_chunks),
        "ffn_hidden_activation_distortion": _finalize_metric_chunks(hidden_distortion_chunks),
        "clipping_rate": clipping,
        "clipping_rate_by_sample": torch.cat(clipping_rate_by_sample, dim=0) if clipping_rate_by_sample else torch.full((batch_start,), float("nan"), dtype=torch.float32),
    }

def _assert_same_eval_order(fp32_cache: Mapping[str, Any], quant_cache: Mapping[str, Any]) -> None:
    if list(fp32_cache["sample_ids"]) != list(quant_cache["sample_ids"]):
        raise RuntimeError("quant eval sample_id order differs from FP32 baseline; cannot compute aligned metrics")
    if not torch.equal(fp32_cache["labels"].cpu(), quant_cache["labels"].cpu()):
        raise RuntimeError("quant eval label_id values differ from FP32 baseline; cannot compute aligned metrics")


def _assert_same_oracle_order(fp32_cache: Mapping[str, Any], oer_oracle: Mapping[str, Any]) -> None:
    if list(fp32_cache["sample_ids"]) != list(oer_oracle.get("sample_id_order", [])):
        raise RuntimeError("FP32 OER oracle sample_id order does not match FP32 embedding baseline")
    oracle_labels = oer_oracle.get("label_id")
    if not isinstance(oracle_labels, Tensor) or not torch.equal(fp32_cache["labels"].cpu(), oracle_labels.cpu()):
        raise RuntimeError("FP32 OER oracle label_id order does not match FP32 embedding baseline")

def _quant_mode_label(cfg: Any) -> str:
    return str(_quant_cfg(cfg).get("DEFAULT_MODE", "w4a4")).upper()


def _granularity_label(cfg: Any) -> str:
    quant = _quant_cfg(cfg)
    weight_cfg = quant.get("WEIGHT", {})
    activation_cfg = quant.get("ACTIVATION", {})
    weight_granularity = weight_cfg.get("GRANULARITY", "per_channel") if isinstance(weight_cfg, Mapping) else "per_channel"
    activation_granularity = activation_cfg.get("GRANULARITY", "per_tensor") if isinstance(activation_cfg, Mapping) else "per_tensor"
    return f"W:{weight_granularity};A:{activation_granularity}"


def _embedding_dir(cfg: Any) -> Path:
    return _result_dir(cfg) / "embeddings"


def _baseline_embedding_path(cfg: Any) -> Path:
    return _embedding_dir(cfg) / BASELINE_EMBEDDING_FILENAME


def _target_quant_embedding_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _embedding_dir(cfg) / f"layer_{int(layer_idx):02d}_{module_name}_quant_embeddings.pt"


def _target_fp32_oer_oracle_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _embedding_dir(cfg) / "oer_oracle" / f"layer_{int(layer_idx):02d}_{module_name}_fp32_oer_oracle.pt"


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        return torch.load(path, map_location="cpu")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON payload must be a mapping: {path}")
    return dict(payload)


def _assert_reference_identity_equal(name: str, expected: Any, actual: Any, *, bundle_path: Path) -> None:
    expected_value = None if expected is None else str(expected)
    actual_value = None if actual is None else str(actual)
    if expected_value != actual_value:
        raise RuntimeError(
            f"FP32 reference bundle identity mismatch for {name}: "
            f"expected={expected_value!r}, actual={actual_value!r}, bundle={bundle_path}. "
            "Set FORCE_REBUILD_FP32_REFERENCE=True or rebuild with python -m src.verify.build_fp32_reference_cache --force."
        )


def _expected_reference_identity(cfg: Any) -> dict[str, str | None]:
    return {
        "checkpoint_hash": _sha256_optional_file(_checkpoint_path_for_reference(cfg)),
        "eval_csv_hash": _sha256_optional_file(_split_csv(cfg, "eval")),
        "label_mapping_hash": _sha256_optional_file(_label_mapping_path_for_reference(cfg)),
        "audio_config_hash": _audio_config_hash(cfg),
    }


def _require_reference_positions(positions: Any, *, context: str) -> list[str]:
    position_set = {str(item).strip().lower() for item in (positions or [])}
    required = set(REQUIRED_FP32_OER_ORACLE_POSITIONS)
    if not required.issubset(position_set):
        raise RuntimeError(
            f"{context}: FP32 OER oracle must include activation_positions={sorted(required)}, "
            f"got {sorted(position_set)}. Rebuild fp32_reference_bundle with --force."
        )
    return sorted(position_set)


def _validate_fp32_reference_bundle(
    cfg: Any,
    bundle: Mapping[str, Any],
    *,
    bundle_path: Path,
    core_path: Path,
    oer_oracle_path: Path,
    core_cache: Mapping[str, Any],
    oer_oracle: Mapping[str, Any],
) -> None:
    shared_identity = bundle.get("shared_identity", {})
    if not isinstance(shared_identity, Mapping):
        raise RuntimeError(f"FP32 reference bundle missing shared_identity: {bundle_path}")
    expected_identity = _expected_reference_identity(cfg)
    for key, expected_value in expected_identity.items():
        _assert_reference_identity_equal(key, expected_value, shared_identity.get(key), bundle_path=bundle_path)

    core_entry = bundle.get("core_cache", {})
    oer_entry = bundle.get("oer_oracle_cache", {})
    if not isinstance(core_entry, Mapping) or not isinstance(oer_entry, Mapping):
        raise RuntimeError(f"FP32 reference bundle missing core_cache/oer_oracle_cache sections: {bundle_path}")
    _assert_reference_identity_equal("core_cache.identity", _sha256_file(core_path), core_entry.get("identity"), bundle_path=bundle_path)
    _assert_reference_identity_equal("oer_oracle_cache.identity", _sha256_file(oer_oracle_path), oer_entry.get("identity"), bundle_path=bundle_path)

    expected_target_count = len(TARGET_LAYER_INDICES) * len(TARGET_MODULE_NAMES)
    target_count = int(oer_entry.get("target_count", -1))
    if target_count != expected_target_count:
        raise RuntimeError(
            f"FP32 reference bundle OER target_count mismatch: expected={expected_target_count}, "
            f"actual={target_count}, bundle={bundle_path}"
        )
    _require_reference_positions(oer_entry.get("activation_positions"), context="fp32_reference_bundle.oer_oracle_cache")
    if not bool(oer_entry.get("contains_fp32_oracle_energy", False)) or not bool(oer_entry.get("contains_fp32_total_energy", False)):
        raise RuntimeError(f"FP32 reference bundle OER oracle lacks FP32 energy fields: {bundle_path}")
    if not math.isclose(float(oer_entry.get("topk_ratio", "nan")), _oer_top_ratio(cfg), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"FP32 reference bundle OER topk_ratio mismatch: expected={_oer_top_ratio(cfg)}, "
            f"actual={oer_entry.get('topk_ratio')}, bundle={bundle_path}"
        )

    core_manifest = core_cache.get("manifest", {})
    oer_manifest = oer_oracle.get("manifest", {})
    if not isinstance(core_manifest, Mapping) or not isinstance(oer_manifest, Mapping):
        raise RuntimeError("FP32 reference core/oer cache must contain manifest mappings")
    _assert_reference_identity_equal("core.sample_order_hash", core_manifest.get("sample_order_hash"), core_entry.get("sample_order_hash"), bundle_path=bundle_path)
    for key in ("checkpoint_sha256", "eval_csv_sha256"):
        shared_key = "checkpoint_hash" if key == "checkpoint_sha256" else "eval_csv_hash"
        _assert_reference_identity_equal(f"core.{key}", shared_identity.get(shared_key), core_manifest.get(key), bundle_path=bundle_path)
        _assert_reference_identity_equal(f"oer.{key}", shared_identity.get(shared_key), oer_manifest.get(key), bundle_path=bundle_path)
    _assert_reference_identity_equal("core.label_mapping_sha256", shared_identity.get("label_mapping_hash"), core_manifest.get("label_mapping_sha256"), bundle_path=bundle_path)
    _require_reference_positions(oer_manifest.get("activation_positions"), context="oer_oracle_eval.manifest")
    if not bool(oer_manifest.get("contains_fp32_oracle_energy", False)) or not bool(oer_manifest.get("contains_fp32_total_energy", False)):
        raise RuntimeError(f"oer_oracle_eval.pt lacks FP32 energy fields: {oer_oracle_path}")
    manifest_target_count = len(oer_manifest.get("target_modules") or [])
    if manifest_target_count != expected_target_count:
        raise RuntimeError(
            f"oer_oracle_eval.pt target count mismatch: expected={expected_target_count}, actual={manifest_target_count}"
        )


def _fp32_cache_from_core_reference(core_cache: Mapping[str, Any], cfg: Any) -> dict[str, Any]:
    raw_sample_ids = core_cache.get("sample_id", core_cache.get("sample_ids"))
    raw_labels = core_cache.get("label_id", core_cache.get("labels"))
    raw_embeddings = core_cache.get("embedding", core_cache.get("embeddings"))
    raw_buckets = core_cache.get("bucket", core_cache.get("buckets"))
    if raw_sample_ids is None or raw_labels is None or raw_embeddings is None or raw_buckets is None:
        raise KeyError("core_eval.pt missing sample_id/label_id/embedding/bucket")
    sample_ids = [str(item) for item in raw_sample_ids]
    labels = torch.as_tensor(raw_labels).detach().cpu().long().reshape(-1)
    embeddings = torch.as_tensor(raw_embeddings).detach().cpu().float()
    bucket_result = read_canonical_bucket(
        bucket=raw_buckets,
        context="core_eval.bucket",
        allow_shot_group_fallback=False,
    )
    buckets = bucket_result.bucket
    if int(labels.numel()) != len(sample_ids):
        raise ValueError("core_eval sample_id and label_id count mismatch")
    if len(buckets) != len(sample_ids):
        raise ValueError("core_eval bucket count mismatch")
    if embeddings.ndim < 2 or int(embeddings.shape[0]) != int(labels.numel()):
        raise ValueError("core_eval embedding shape does not match labels")
    manifest = core_cache.get("manifest", {})
    if isinstance(manifest, Mapping):
        expected_hash = manifest.get("sample_order_hash")
        actual_hash = _sample_order_hash(sample_ids)
        if expected_hash is not None and str(expected_hash) != str(actual_hash):
            raise RuntimeError("core_eval sample_order_hash does not match sample_id order")

    analysis_group_labels = _analysis_group_class_labels(labels, buckets)
    sample_counts = _analysis_group_sample_counts(buckets)
    margin_fp32_by_group: dict[str, Tensor] = {}
    fisher_fp32_by_group: dict[str, Tensor] = {}
    effective_class_count_by_group: dict[str, int] = {}
    for analysis_group in ANALYSIS_GROUP_NAMES:
        group_labels = analysis_group_labels[analysis_group]
        margin_fp32_by_group[analysis_group] = _compute_group_margin_samples_or_empty(embeddings, labels, group_labels)
        fisher_fp32_by_group[analysis_group] = _compute_group_fisher_or_nan(embeddings, labels, group_labels)
        effective_class_count_by_group[analysis_group] = compute_effective_class_count(labels, group_labels)
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "embeddings": embeddings,
        "batch_sizes": [int(labels.numel())],
        "bucket": buckets,
        "analysis_group_labels": analysis_group_labels,
        "analysis_group_sample_counts": sample_counts,
        "effective_class_count_by_group": effective_class_count_by_group,
        "margin_fp32_by_group": margin_fp32_by_group,
        "fisher_fp32_by_group": fisher_fp32_by_group,
    }


def _row_value(values: Any, row_idx: int) -> Any:
    if isinstance(values, Tensor):
        return values[int(row_idx)]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return values[int(row_idx)]
    raise TypeError(f"row-wise OER oracle field must be a tensor or sequence, got {type(values)!r}")


def _oer_row_index_by_key(oer_oracle: Mapping[str, Any]) -> dict[tuple[str, int, str, str], int]:
    sample_ids = [str(item) for item in oer_oracle.get("sample_id", [])]
    layer_idx = torch.as_tensor(oer_oracle.get("layer_idx")).reshape(-1).long()
    module_names = [str(item) for item in oer_oracle.get("module_name", [])]
    positions = [str(item) for item in oer_oracle.get("activation_position", [])]
    row_count = len(sample_ids)
    if int(layer_idx.numel()) != row_count or len(module_names) != row_count or len(positions) != row_count:
        raise ValueError("oer_oracle_eval row fields have inconsistent lengths")
    index: dict[tuple[str, int, str, str], int] = {}
    for row_idx, sample_id in enumerate(sample_ids):
        key = (str(sample_id), int(layer_idx[row_idx].item()), module_names[row_idx], positions[row_idx])
        if key in index:
            raise ValueError(f"duplicate row-wise OER oracle key: {key}")
        index[key] = int(row_idx)
    return index


def _feature_source_for_target(layer_idx: int, module_name: str, feature_kind: str) -> tuple[int, str, str]:
    if feature_kind == INPUT_OER_FEATURE_KIND:
        return int(layer_idx), str(module_name), "input"
    if feature_kind == PROJECTION_OUTPUT_OER_FEATURE_KIND:
        return int(layer_idx), str(module_name), "output"
    if feature_kind == FFN_HIDDEN_ACTIVATION_FEATURE_KIND:
        return int(layer_idx), "fc2", "input"
    raise KeyError(f"unknown OER feature kind: {feature_kind}")


def _rowwise_oer_feature_oracle(
    rowwise_oracle: Mapping[str, Any],
    row_index: Mapping[tuple[str, int, str, str], int],
    fp32_cache: Mapping[str, Any],
    *,
    target_layer_idx: int,
    target_module_name: str,
    feature_kind: str,
) -> dict[str, Any]:
    source_layer_idx, source_module_name, source_position = _feature_source_for_target(
        int(target_layer_idx),
        str(target_module_name),
        str(feature_kind),
    )
    selected_rows: list[int] = []
    for sample_id in fp32_cache["sample_ids"]:
        key = (str(sample_id), int(source_layer_idx), str(source_module_name), str(source_position))
        if key not in row_index:
            raise KeyError(f"oer_oracle_eval.pt missing row for {key}")
        selected_rows.append(int(row_index[key]))

    topk_rows = [torch.as_tensor(_row_value(rowwise_oracle["topk_scalar_indices"], row_idx)).detach().cpu().long().reshape(-1) for row_idx in selected_rows]
    topk_lengths = {int(row.numel()) for row in topk_rows}
    topk_indices: Tensor | list[Tensor]
    if len(topk_lengths) == 1:
        topk_indices = torch.stack(topk_rows, dim=0)
    else:
        topk_indices = [row.contiguous() for row in topk_rows]

    activation_shapes = [tuple(int(dim) for dim in _row_value(rowwise_oracle["activation_shape"], row_idx)) for row_idx in selected_rows]
    nonbatch_shapes = {shape[1:] for shape in activation_shapes if len(shape) == 3}
    if len(nonbatch_shapes) != 1:
        raise ValueError(
            f"L{int(target_layer_idx)}.{target_module_name}:{feature_kind} activation non-batch shape mismatch: {activation_shapes[:8]}"
        )
    nonbatch_shape = tuple(next(iter(nonbatch_shapes)))
    sample_count = len(selected_rows)
    fp32_oracle_energy = torch.as_tensor(
        [float(torch.as_tensor(_row_value(rowwise_oracle["fp32_oracle_energy"], row_idx)).reshape(()).item()) for row_idx in selected_rows],
        dtype=torch.float32,
    )
    fp32_total_energy = torch.as_tensor(
        [float(torch.as_tensor(_row_value(rowwise_oracle["fp32_total_energy"], row_idx)).reshape(()).item()) for row_idx in selected_rows],
        dtype=torch.float32,
    )
    topk_ratio_values = torch.as_tensor([float(torch.as_tensor(_row_value(rowwise_oracle["topk_ratio"], row_idx)).reshape(()).item()) for row_idx in selected_rows], dtype=torch.float32)
    if int(topk_ratio_values.numel()) > 0:
        first_topk_ratio = float(topk_ratio_values[0].item())
        if not torch.allclose(topk_ratio_values, torch.full_like(topk_ratio_values, first_topk_ratio), rtol=0.0, atol=1e-12):
            raise ValueError("row-wise OER oracle topk_ratio differs across selected samples")
    else:
        first_topk_ratio = float("nan")

    return {
        "layout": OER_FEATURE_LAYOUT,
        "contains_full_activation": False,
        "source_cache": "oer_oracle_eval.pt",
        "layer_idx": int(target_layer_idx),
        "module_name": str(target_module_name),
        "target_name": _target_name(int(target_layer_idx), target_module_name),
        "source_layer_idx": int(source_layer_idx),
        "source_module_name": str(source_module_name),
        "activation_position": str(source_position),
        "activation_kind": str(feature_kind),
        "feature_kind": str(feature_kind),
        "activation_shape": (int(sample_count), *nonbatch_shape),
        "canonical_layout": "BTC",
        "canonical_shape": (int(sample_count), *nonbatch_shape),
        "topk_ratio": float(first_topk_ratio),
        "topk_indices": topk_indices,
        "flat_index": topk_indices,
        "batch_dim": 0,
        "token_dim": 1,
        "channel_dim": 2,
        "fp32_oracle_energy": fp32_oracle_energy,
        "fp32_total_energy": fp32_total_energy,
        "sample_id_order": list(fp32_cache["sample_ids"]),
        "label_id": torch.as_tensor(fp32_cache["labels"]).detach().cpu().long().reshape(-1),
    }


def _materialize_rowwise_oer_layer_if_needed(rowwise_oracle: Mapping[str, Any], layer_idx: int) -> Mapping[str, Any]:
    if not is_sharded_fp32_oer_oracle(rowwise_oracle):
        return rowwise_oracle
    oracle_path = rowwise_oracle.get("__oracle_path__")
    if oracle_path is None:
        raise KeyError("sharded rowwise OER oracle missing __oracle_path__")
    layer_shards = list(iter_fp32_oer_oracle_shards(rowwise_oracle, oracle_path, layer_idx=int(layer_idx)))
    if not layer_shards:
        raise KeyError(f"sharded OER oracle has no shards for layer {int(layer_idx)}")
    result = dict(rowwise_oracle)
    sample_ids: list[str] = []
    layer_rows: list[Tensor] = []
    module_names: list[str] = []
    positions: list[str] = []
    topk_scalar_indices: list[Tensor] = []
    oer_topk_token_indices_aux: list[Tensor] = []
    fp32_oracle_energy: list[Tensor] = []
    fp32_total_energy: list[Tensor] = []
    topk_ratio: list[Tensor] = []
    activation_shape: list[Any] = []
    original_layout: list[Any] = []
    original_shape: list[Any] = []
    canonical_layout: list[Any] = []
    canonical_shape: list[Any] = []
    layout_inference_source: list[Any] = []
    padding_mask_available: list[Any] = []
    padding_mask_aligned: list[Any] = []
    mask_status: list[Any] = []
    valid_token_count: list[Tensor] = []
    for shard in layer_shards:
        shard_sample_ids = [str(item) for item in shard.get("sample_id", [])]
        row_count = len(shard_sample_ids)
        sample_ids.extend(shard_sample_ids)
        layer_rows.append(torch.as_tensor(shard["layer_idx"], dtype=torch.long).reshape(-1))
        module_names.extend(str(item) for item in shard["module_name"])
        positions.extend(str(item) for item in shard["activation_position"])
        topk_scalar_indices.extend(torch.as_tensor(item).detach().cpu().long().reshape(-1) for item in shard["topk_scalar_indices"])
        aux_token_values = shard.get("oer_topk_token_indices_aux", shard.get("topk_token_indices"))
        if aux_token_values is None:
            raise KeyError("OER oracle shard missing oer_topk_token_indices_aux")
        oer_topk_token_indices_aux.extend(torch.as_tensor(item).detach().cpu().long().reshape(-1) for item in aux_token_values)
        fp32_oracle_energy.append(torch.as_tensor(shard["fp32_oracle_energy"], dtype=torch.float32).reshape(-1))
        fp32_total_energy.append(torch.as_tensor(shard["fp32_total_energy"], dtype=torch.float32).reshape(-1))
        topk_ratio.append(torch.as_tensor(shard["topk_ratio"], dtype=torch.float32).reshape(-1))
        activation_shape.extend(shard["activation_shape"])
        original_layout.extend(shard.get("original_layout", [None] * row_count))
        original_shape.extend(shard.get("original_shape", [None] * row_count))
        canonical_layout.extend(shard.get("canonical_layout", [None] * row_count))
        canonical_shape.extend(shard.get("canonical_shape", [None] * row_count))
        layout_inference_source.extend(shard.get("layout_inference_source", [None] * row_count))
        padding_mask_available.extend(shard.get("padding_mask_available", [None] * row_count))
        padding_mask_aligned.extend(shard.get("padding_mask_aligned", [None] * row_count))
        mask_status.extend(shard.get("mask_status", [None] * row_count))
        valid_token_count.append(torch.as_tensor(shard["valid_token_count"], dtype=torch.long).reshape(-1))
    result.update(
        {
            "sample_id": sample_ids,
            "layer_idx": torch.cat(layer_rows, dim=0),
            "module_name": module_names,
            "activation_position": positions,
            "topk_scalar_indices": topk_scalar_indices,
            "oer_topk_token_indices_aux": oer_topk_token_indices_aux,
            "fp32_oracle_energy": torch.cat(fp32_oracle_energy, dim=0),
            "fp32_total_energy": torch.cat(fp32_total_energy, dim=0),
            "topk_ratio": torch.cat(topk_ratio, dim=0),
            "activation_shape": activation_shape,
            "original_layout": original_layout,
            "original_shape": original_shape,
            "canonical_layout": canonical_layout,
            "canonical_shape": canonical_shape,
            "layout_inference_source": layout_inference_source,
            "padding_mask_available": padding_mask_available,
            "padding_mask_aligned": padding_mask_aligned,
            "mask_status": mask_status,
            "valid_token_count": torch.cat(valid_token_count, dim=0),
        }
    )
    return result

def _target_oer_oracle_from_rowwise_reference(
    rowwise_oracle: Mapping[str, Any],
    fp32_cache: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
) -> dict[str, Any]:
    rowwise_oracle = _materialize_rowwise_oer_layer_if_needed(rowwise_oracle, int(layer_idx))
    row_index = _oer_row_index_by_key(rowwise_oracle)
    feature_oracles = {
        feature_kind: _rowwise_oer_feature_oracle(
            rowwise_oracle,
            row_index,
            fp32_cache,
            target_layer_idx=int(layer_idx),
            target_module_name=str(module_name),
            feature_kind=feature_kind,
        )
        for feature_kind in OER_FEATURE_KINDS
    }
    primary = dict(feature_oracles[PROJECTION_OUTPUT_OER_FEATURE_KIND])
    primary["sample_id_order"] = list(fp32_cache["sample_ids"])
    primary["label_id"] = torch.as_tensor(fp32_cache["labels"]).detach().cpu().long().reshape(-1)
    primary["feature_oracles"] = feature_oracles
    primary["feature_kinds"] = list(OER_FEATURE_KINDS)
    primary["primary_feature_kind"] = PROJECTION_OUTPUT_OER_FEATURE_KIND
    primary["source_cache"] = "oer_oracle_eval.pt"
    return primary


def _load_fp32_reference_bundle_if_valid(cfg: Any) -> dict[str, Any] | None:
    bundle_path = _fp32_reference_bundle_path(cfg)
    if not bundle_path.exists():
        return None
    bundle = _load_json_mapping(bundle_path)
    cache_root = bundle_path.parent
    core_entry = bundle.get("core_cache", {})
    oer_entry = bundle.get("oer_oracle_cache", {})
    if not isinstance(core_entry, Mapping) or not isinstance(oer_entry, Mapping):
        raise RuntimeError(f"FP32 reference bundle missing cache sections: {bundle_path}")
    core_path = _resolve_reference_artifact_path(
        cache_root,
        core_entry.get("path"),
        default_filename=FP32_REFERENCE_CORE_FILENAME,
        field_name="core_cache.path",
    )
    oer_oracle_path = _resolve_reference_artifact_path(
        cache_root,
        oer_entry.get("path"),
        default_filename=FP32_REFERENCE_OER_ORACLE_FILENAME,
        field_name="oer_oracle_cache.path",
    )
    core_cache = load_fp32_core_cache(core_path, materialize=True, include_padding_mask=False, include_logits=False)
    oer_oracle = load_fp32_oer_oracle(oer_oracle_path, materialize=False)
    oer_oracle["__oracle_path__"] = oer_oracle_path
    if not isinstance(core_cache, Mapping) or not isinstance(oer_oracle, Mapping):
        raise TypeError("FP32 reference core/oer cache payloads must be mappings")
    _validate_fp32_reference_bundle(
        cfg,
        bundle,
        bundle_path=bundle_path,
        core_path=core_path,
        oer_oracle_path=oer_oracle_path,
        core_cache=core_cache,
        oer_oracle=oer_oracle,
    )
    fp32_cache = _fp32_cache_from_core_reference(core_cache, cfg)
    sample_hash = _sample_order_hash(fp32_cache["sample_ids"])
    expected_hash = bundle.get("core_cache", {}).get("sample_order_hash") if isinstance(bundle.get("core_cache", {}), Mapping) else None
    _assert_reference_identity_equal("sample_order_hash", expected_hash, sample_hash, bundle_path=bundle_path)
    print(f"[reference] loaded FP32 reference bundle: {bundle_path}")
    return {
        "bundle": bundle,
        "bundle_path": bundle_path,
        "core_cache_path": core_path,
        "oer_oracle_path": oer_oracle_path,
        "fp32_cache": fp32_cache,
        "rowwise_oer_oracle": oer_oracle,
    }

def _load_fp32_embeddings_if_valid(cfg: Any, protocol_version: str, experiment_signature: str) -> dict[str, Any] | None:
    path = _baseline_embedding_path(cfg)
    if not path.exists():
        return None
    try:
        payload = _torch_load_cpu(path)
        if not isinstance(payload, Mapping):
            raise TypeError("baseline cache must be a mapping")
        if not _cache_protocol_matches(payload, protocol_version, experiment_signature):
            raise ValueError("baseline protocol/signature mismatch")
        raw_sample_ids = payload.get("sample_id", payload.get("sample_ids"))
        raw_labels = payload.get("label_id", payload.get("labels"))
        raw_embeddings = payload.get("embedding", payload.get("embeddings"))
        raw_buckets = payload.get("bucket", payload.get("buckets"))
        if raw_sample_ids is None or raw_labels is None or raw_embeddings is None or raw_buckets is None:
            raise KeyError("baseline cache missing sample_id/label_id/embedding/bucket")
        sample_ids = [str(item) for item in raw_sample_ids]
        bucket_result = read_canonical_bucket(
            bucket=raw_buckets,
            context="fp32_baseline_cache.bucket",
            allow_shot_group_fallback=False,
        )
        buckets = bucket_result.bucket
        labels = torch.as_tensor(raw_labels).detach().cpu().long().reshape(-1)
        embeddings = torch.as_tensor(raw_embeddings).detach().cpu().float()
        if int(labels.numel()) != len(sample_ids):
            raise ValueError("baseline sample_id and label_id count mismatch")
        if len(buckets) != len(sample_ids):
            raise ValueError("baseline bucket count mismatch")
        if embeddings.ndim < 2 or int(embeddings.shape[0]) != int(labels.numel()):
            raise ValueError("baseline embedding shape does not match labels")

        analysis_group_labels = _analysis_group_class_labels(labels, buckets)
        sample_counts = _analysis_group_sample_counts(buckets)
        margin_fp32_by_group: dict[str, Tensor] = {}
        fisher_fp32_by_group: dict[str, Tensor] = {}
        effective_class_count_by_group: dict[str, int] = {}
        for analysis_group in ANALYSIS_GROUP_NAMES:
            group_labels = analysis_group_labels[analysis_group]
            margin_fp32_by_group[analysis_group] = _compute_group_margin_samples_or_empty(embeddings, labels, group_labels)
            fisher_fp32_by_group[analysis_group] = _compute_group_fisher_or_nan(embeddings, labels, group_labels)
            effective_class_count_by_group[analysis_group] = compute_effective_class_count(labels, group_labels)
        cache: dict[str, Any] = {
            "sample_ids": sample_ids,
            "labels": labels,
            "embeddings": embeddings,
            "batch_sizes": [int(labels.numel())],
            "bucket": buckets,
            "analysis_group_labels": analysis_group_labels,
            "analysis_group_sample_counts": sample_counts,
            "effective_class_count_by_group": effective_class_count_by_group,
            "margin_fp32_by_group": margin_fp32_by_group,
            "fisher_fp32_by_group": fisher_fp32_by_group,
        }
        print(f"[resume] loaded FP32 baseline: {path}")
        return cache
    except Exception as exc:  # pragma: no cover - defensive resume path.
        print(f"[resume] ignore invalid FP32 baseline cache {path}: {exc}")
        return None


def _load_target_fp32_oer_oracle_if_valid(
    cfg: Any,
    fp32_cache: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
    protocol_version: str,
    experiment_signature: str,
) -> dict[str, Any] | None:
    path = _target_fp32_oer_oracle_path(cfg, int(layer_idx), module_name)
    if not path.exists():
        return None
    try:
        payload = _torch_load_cpu(path)
        if not isinstance(payload, Mapping):
            raise TypeError("OER oracle cache must be a mapping")
        if not _cache_protocol_matches(payload, protocol_version, experiment_signature):
            raise ValueError("oracle protocol/signature mismatch")
        oracle = dict(payload)
        if int(oracle.get("layer_idx", int(layer_idx))) != int(layer_idx):
            raise ValueError("oracle layer_idx mismatch")
        if str(oracle.get("module_name", module_name)) != str(module_name):
            raise ValueError("oracle module_name mismatch")
        if str(oracle.get("activation_kind", "")) != OER_ACTIVATION_KIND:
            raise ValueError("oracle activation_kind mismatch")
        cached_topk_ratio = float(oracle.get("topk_ratio", "nan"))
        if not math.isclose(cached_topk_ratio, _oer_top_ratio(cfg), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("oracle topk_ratio mismatch")
        sample_order = oracle.get("sample_id_order", oracle.get("sample_id"))
        if sample_order is None:
            raise KeyError("oracle missing sample_id_order")
        oracle["sample_id_order"] = [str(item) for item in sample_order]
        oracle.pop("sample_id", None)
        if "label_id" not in oracle:
            raise KeyError("oracle missing label_id")
        oracle["label_id"] = torch.as_tensor(oracle["label_id"]).detach().cpu().long().reshape(-1)

        feature_oracles_raw = oracle.get("feature_oracles")
        if not isinstance(feature_oracles_raw, Mapping):
            raise KeyError("oracle missing feature_oracles")
        normalized_feature_oracles: dict[str, dict[str, Any]] = {}
        tensor_keys = (
            "topk_indices",
            "flat_index",
            "sample_index",
            "token_idx",
            "channel_idx",
            "fp32_oracle_energy",
            "fp32_total_energy",
        )
        expected_samples = len(fp32_cache["sample_ids"])
        for feature_kind in OER_FEATURE_KINDS:
            if feature_kind not in feature_oracles_raw:
                raise KeyError(f"oracle missing feature_oracles[{feature_kind!r}]")
            feature_oracle = dict(feature_oracles_raw[feature_kind])
            if str(feature_oracle.get("feature_kind", feature_oracle.get("activation_kind", ""))) != feature_kind:
                raise ValueError(f"oracle feature_kind mismatch for {feature_kind}")
            if int(feature_oracle.get("layer_idx", int(layer_idx))) != int(layer_idx):
                raise ValueError(f"oracle {feature_kind} layer_idx mismatch")
            if str(feature_oracle.get("module_name", module_name)) != str(module_name):
                raise ValueError(f"oracle {feature_kind} module_name mismatch")
            feature_topk_ratio = float(feature_oracle.get("topk_ratio", "nan"))
            if not math.isclose(feature_topk_ratio, _oer_top_ratio(cfg), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"oracle {feature_kind} topk_ratio mismatch")
            for key in tensor_keys:
                if key in feature_oracle:
                    feature_oracle[key] = torch.as_tensor(feature_oracle[key]).detach().cpu()
            for key in ("topk_indices", "fp32_oracle_energy", "fp32_total_energy"):
                if key not in feature_oracle:
                    raise KeyError(f"oracle {feature_kind} missing {key}")
            if int(torch.as_tensor(feature_oracle["fp32_oracle_energy"]).numel()) != int(expected_samples):
                raise ValueError(f"oracle {feature_kind} sample count mismatch")
            feature_oracle["sample_id_order"] = list(oracle["sample_id_order"])
            feature_oracle["label_id"] = oracle["label_id"]
            normalized_feature_oracles[feature_kind] = feature_oracle
        oracle["feature_oracles"] = normalized_feature_oracles
        oracle.update(normalized_feature_oracles[PROJECTION_OUTPUT_OER_FEATURE_KIND])
        oracle["feature_oracles"] = normalized_feature_oracles
        oracle["feature_kinds"] = list(OER_FEATURE_KINDS)
        oracle["primary_feature_kind"] = PROJECTION_OUTPUT_OER_FEATURE_KIND
        _assert_same_oracle_order(fp32_cache, oracle)
        print(f"[resume] loaded FP32 OER oracle: {path}")
        return oracle
    except Exception as exc:  # pragma: no cover - defensive resume path.
        print(f"[resume] ignore invalid FP32 OER oracle {path}: {exc}")
        return None


def _read_existing_csv_rows(csv_path: Path, columns: Sequence[str], *, label: str) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"existing {label} CSV has no header: {csv_path}")
        rows = []
        for row in reader:
            normalized = {key: row.get(key, "") for key in columns}
            if any(str(value).strip() for value in normalized.values()):
                rows.append(normalized)
        if rows:
            print(f"[resume] loaded {len(rows)} existing {label} CSV rows: {csv_path}")
        return rows


def _read_existing_rows(csv_path: Path) -> list[dict[str, Any]]:
    return _read_existing_csv_rows(csv_path, HEATMAP_CSV_COLUMNS, label="heatmap detail")


def _read_existing_projection_detail_rows(csv_path: Path) -> list[dict[str, Any]]:
    return _read_existing_csv_rows(csv_path, PROJECTION_DETAIL_CSV_COLUMNS, label="projection detail")


def _target_completed(rows: Sequence[Mapping[str, Any]], layer_idx: int, module_name: str, protocol_version: str, experiment_signature: str) -> bool:
    groups: set[str] = set()
    for row in rows:
        try:
            row_layer = int(float(str(row.get("layer_idx", "")).strip()))
        except ValueError:
            continue
        row_module = str(row.get("module_name", "")).strip()
        if row_layer != int(layer_idx) or row_module != str(module_name):
            continue
        if not _row_protocol_matches(row, protocol_version, experiment_signature):
            continue
        try:
            analysis_group = normalize_analysis_group(row.get("analysis_group", ""))
        except Exception:
            continue
        if analysis_group in COMPLETED_GROUP_NAMES:
            groups.add(analysis_group)
    return COMPLETED_GROUP_NAMES.issubset(groups)


def _drop_target_rows(rows: Sequence[Mapping[str, Any]], layer_idx: int, module_name: str) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        try:
            row_layer = int(float(str(row.get("layer_idx", "")).strip()))
        except ValueError:
            kept.append(dict(row))
            continue
        row_module = str(row.get("module_name", "")).strip()
        if row_layer == int(layer_idx) and row_module == str(module_name):
            continue
        kept.append(dict(row))
    return kept

def _save_quant_embeddings(cfg: Any, layer_idx: int, module_name: str, quant_cache: Mapping[str, Any]) -> Path:
    path = _target_quant_embedding_path(cfg, int(layer_idx), module_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer_idx": int(layer_idx),
            "module_name": module_name,
            "sample_id": list(quant_cache["sample_ids"]),
            "label_id": quant_cache["labels"].cpu(),
            "embedding": quant_cache["embeddings"].cpu(),
            "bucket": list(quant_cache.get("bucket", [])),
        },
        path,
    )
    return path


def run_single_intervention(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    calibration_loader: DataLoader,
    eval_loader: DataLoader,
    fp32_cache: Mapping[str, Any],
    oer_oracle: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    """Run clean reload -> target quantization -> calibration -> fake quant eval."""

    device = _device(cfg)
    model = restore_clean_model_from_checkpoint(cfg).to(device)
    model.float()
    quant_cfg = _quant_cfg(cfg)
    quant_module = replace_ffn_module_with_quant(model, int(layer_idx), module_name, quant_cfg)
    quant_mode = str(quant_cfg.get("DEFAULT_MODE", "w4a4"))
    controller = make_single_ffn_quant_controller(
        model,
        quant_module,
        layer_idx=int(layer_idx),
        module_name=module_name,
        quant_config=quant_cfg,
        quant_mode=quant_mode,
    )
    quant_state_path = _target_quant_state_path(cfg, int(layer_idx), module_name)
    try:
        _run_two_pass_calibration(controller, model, calibration_loader, cfg)
        quant_state_path = _save_target_quant_state(cfg, int(layer_idx), module_name, quant_module, quant_mode)
        _clear_quant_module_oer_cache(quant_module)
        _release_memory(device)
        quant_cache = _collect_quant_eval(model, eval_loader, cfg, quant_module, oer_oracle, int(layer_idx), module_name)
    finally:
        controller.remove_padding_mask_hooks()

    _assert_same_eval_order(fp32_cache, quant_cache)
    quant_cache["bucket"] = list(fp32_cache["bucket"])
    _save_quant_embeddings(cfg, int(layer_idx), module_name, quant_cache)

    labels = fp32_cache["labels"]
    buckets = list(fp32_cache["bucket"])
    analysis_group_labels: Mapping[str, Tensor] = fp32_cache["analysis_group_labels"]
    sample_counts: Mapping[str, int] = fp32_cache["analysis_group_sample_counts"]
    effective_class_counts: Mapping[str, int] = fp32_cache["effective_class_count_by_group"]
    margin_fp32_by_group: Mapping[str, Tensor] = fp32_cache["margin_fp32_by_group"]
    fisher_fp32_by_group: Mapping[str, Tensor] = fp32_cache["fisher_fp32_by_group"]

    rows: list[dict[str, Any]] = []
    projection_detail_rows: list[dict[str, Any]] = []
    for analysis_group in ANALYSIS_GROUP_NAMES:
        group_labels = analysis_group_labels[analysis_group]
        group_mask = _sample_bucket_mask(buckets, analysis_group)
        margin_quant = _compute_group_margin_samples_or_empty(quant_cache["embeddings"], labels, group_labels)
        fisher_quant = _compute_group_fisher_or_nan(quant_cache["embeddings"], labels, group_labels)
        input_oer_metrics = compute_group_oer_metrics_from_energy_cache(
            _oer_feature_oracle(oer_oracle, INPUT_OER_FEATURE_KIND),
            quant_cache["oer_energy_by_feature"][INPUT_OER_FEATURE_KIND],
            group_mask,
        )
        projection_output_oer_metrics = compute_group_oer_metrics_from_energy_cache(
            _oer_feature_oracle(oer_oracle, PROJECTION_OUTPUT_OER_FEATURE_KIND),
            quant_cache["oer_energy_by_feature"][PROJECTION_OUTPUT_OER_FEATURE_KIND],
            group_mask,
        )
        hidden_oer_metrics = compute_group_oer_metrics_from_energy_cache(
            _oer_feature_oracle(oer_oracle, FFN_HIDDEN_ACTIVATION_FEATURE_KIND),
            quant_cache["oer_energy_by_feature"][FFN_HIDDEN_ACTIVATION_FEATURE_KIND],
            group_mask,
        )
        projection_clipping_rate = compute_group_scalar_mean(quant_cache["clipping_rate_by_sample"], group_mask)
        projection_detail_rows.append(
            compute_group_projection_detail_row_summary(
                {
                    "layer_idx": int(layer_idx),
                    "module_name": module_name,
                    "quant_mode": _quant_mode_label(cfg),
                    "granularity": _granularity_label(cfg),
                    "analysis_group": analysis_group,
                    "group_protocol_version": GROUPING_PROTOCOL_VERSION,
                    "sample_count": int(sample_counts.get(analysis_group, 0)),
                    "group_mask": group_mask,
                    "projection_distortion": quant_cache["target_projection_distortion"],
                    "projection_oer_metrics": projection_output_oer_metrics,
                    "projection_clipping_rate": projection_clipping_rate,
                }
            )
        )
        rows.append(
            compute_group_heatmap_row_summary(
                {
                    "layer_idx": int(layer_idx),
                    "module_name": module_name,
                    "quant_mode": _quant_mode_label(cfg),
                    "granularity": _granularity_label(cfg),
                    "analysis_group": analysis_group,
                    "group_protocol_version": GROUPING_PROTOCOL_VERSION,
                    "sample_count": int(sample_counts.get(analysis_group, 0)),
                    "effective_class_count": int(effective_class_counts.get(analysis_group, 0)),
                    "margin_fp32": margin_fp32_by_group[analysis_group],
                    "margin_quant": margin_quant,
                    "fisher_fp32": fisher_fp32_by_group[analysis_group],
                    "fisher_quant": fisher_quant,
                    "teacher_center_margin_quant": compute_teacher_center_margin_quant(
                        fp32_cache["embeddings"],
                        quant_cache["embeddings"],
                        labels,
                        group_labels,
                    ),
                    "teacher_anchor_center_shift": compute_teacher_anchor_center_shift(
                        fp32_cache["embeddings"],
                        quant_cache["embeddings"],
                        labels,
                        group_labels,
                    ),
                    "teacher_anchor_embedding_drift": compute_teacher_anchor_embedding_drift(
                        fp32_cache["embeddings"],
                        quant_cache["embeddings"],
                        labels,
                        group_mask,
                    ),
                    "oer_metrics": projection_output_oer_metrics,
                    "input_oer_metrics": input_oer_metrics,
                    "projection_output_oer_metrics": projection_output_oer_metrics,
                    "ffn_hidden_activation_oer_metrics": hidden_oer_metrics,
                    "target_projection_distortion": compute_group_tensor_distortion_metrics_from_cache(
                        quant_cache["target_projection_distortion"],
                        group_mask,
                    ),
                    "ffn_hidden_activation_distortion_metrics": compute_group_tensor_distortion_metrics_from_cache(
                        quant_cache["ffn_hidden_activation_distortion"],
                        group_mask,
                    ),
                    "clipping_rate": projection_clipping_rate,
                    "embedding_distortion": compute_group_embedding_distortion(
                        fp32_cache["embeddings"],
                        quant_cache["embeddings"],
                        group_mask,
                    ),
                }
            )
        )
    del model, quant_module, controller, quant_cache
    _release_memory(device)
    return rows, projection_detail_rows, quant_state_path

def _to_csv_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _write_csv_rows(csv_path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_csv_value(row.get(key, "")) for key in columns})
    os.replace(tmp_path, csv_path)


def _write_rows(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv_rows(csv_path, rows, HEATMAP_CSV_COLUMNS)


def _write_projection_detail_rows(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv_rows(csv_path, rows, PROJECTION_DETAIL_CSV_COLUMNS)


def _write_summary_rows(csv_path: Path, detail_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = build_heatmap_metric_summary_rows(detail_rows) if detail_rows else []
    _write_csv_rows(csv_path, summary_rows, HEATMAP_SUMMARY_CSV_COLUMNS)
    return summary_rows


def _save_fp32_embeddings(
    cfg: Any,
    fp32_cache: Mapping[str, Any],
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> Path:
    path = _baseline_embedding_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": protocol_version,
            "experiment_signature": experiment_signature,
            "group_protocol_version": GROUPING_PROTOCOL_VERSION,
            "experiment_signature_payload": _jsonable(signature_payload),
            "sample_id": list(fp32_cache["sample_ids"]),
            "label_id": fp32_cache["labels"].cpu(),
            "embedding": fp32_cache["embeddings"].cpu(),
            "bucket": list(fp32_cache["bucket"]),
            "analysis_group_labels": {
                group: labels.cpu()
                for group, labels in fp32_cache["analysis_group_labels"].items()
            },
            "analysis_group_sample_counts": dict(fp32_cache["analysis_group_sample_counts"]),
            "effective_class_count_by_group": dict(fp32_cache["effective_class_count_by_group"]),
        },
        path,
    )
    return path

def _save_target_fp32_oer_oracle(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    oer_oracle: Mapping[str, Any],
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> Path:
    path = _target_fp32_oer_oracle_path(cfg, int(layer_idx), module_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(oer_oracle)
    payload["protocol_version"] = protocol_version
    payload["experiment_signature"] = experiment_signature
    payload["group_protocol_version"] = GROUPING_PROTOCOL_VERSION
    payload["experiment_signature_payload"] = _jsonable(signature_payload)
    torch.save(payload, path)
    return path

def _save_fp32_oer_oracle_index(
    cfg: Any,
    oracle_paths: Mapping[tuple[int, str], Path],
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> Path:
    path = _embedding_dir(cfg) / BASELINE_OER_ORACLE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": protocol_version,
            "experiment_signature": experiment_signature,
            "group_protocol_version": GROUPING_PROTOCOL_VERSION,
            "experiment_signature_payload": _jsonable(signature_payload),
            "layout": OER_FEATURE_LAYOUT,
            "oer_feature_layout": OER_FEATURE_LAYOUT,
            "activation_kind": OER_ACTIVATION_KIND,
            "primary_feature_kind": PROJECTION_OUTPUT_OER_FEATURE_KIND,
            "feature_kinds": list(OER_FEATURE_KINDS),
            "topk_ratio": _oer_top_ratio(cfg),
            "contains_full_activation": False,
            "amp_enabled": False,
            "autocast_enabled": False,
            "reference_dtype": "float32",
            "oracle_paths": {
                f"{int(layer_idx)}:{module_name}": str(oracle_path)
                for (layer_idx, module_name), oracle_path in oracle_paths.items()
            },
        },
        path,
    )
    return path
def main(cfg: Any) -> dict[str, Any]:
    """Run FFN Layer Heatmap with per-target resumable checkpoints."""

    seed_everything = _cfg_get(cfg, "seed_everything", None)
    if callable(seed_everything):
        seed_everything(_seed(cfg))

    protocol_version = PROTOCOL_VERSION
    signature_payload = _experiment_signature_payload(cfg)
    experiment_signature = _experiment_signature(signature_payload)

    output_dir = _result_dir(cfg)
    csv_path = output_dir / CSV_FILENAME
    summary_csv_path = output_dir / SUMMARY_CSV_FILENAME
    projection_detail_csv_path = output_dir / PROJECTION_DETAIL_CSV_FILENAME
    plot_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_experiment_manifest(output_dir, protocol_version, experiment_signature, signature_payload)

    target_keys = [(int(layer_idx), module_name) for layer_idx in TARGET_LAYER_INDICES for module_name in TARGET_MODULE_NAMES]
    resume_enabled = _resume_enabled(cfg)
    existing_rows = _read_existing_rows(csv_path) if resume_enabled else []
    rows: list[dict[str, Any]] = (
        _filter_rows_for_current_protocol(existing_rows, protocol_version, experiment_signature)
        if resume_enabled
        else []
    )
    existing_projection_detail_rows = _read_existing_projection_detail_rows(projection_detail_csv_path) if resume_enabled else []
    projection_detail_rows: list[dict[str, Any]] = (
        _filter_rows_for_current_protocol(existing_projection_detail_rows, protocol_version, experiment_signature)
        if resume_enabled
        else []
    )
    oracle_paths: dict[tuple[int, str], Path] = {}
    quant_state_paths: dict[tuple[int, str], Path] = {}
    fp32_reference_state = _load_fp32_reference_bundle_if_valid(cfg)
    rowwise_fp32_oer_oracle = None if fp32_reference_state is None else fp32_reference_state["rowwise_oer_oracle"]
    fp32_embedding_path: Path = _baseline_embedding_path(cfg) if fp32_reference_state is None else fp32_reference_state["core_cache_path"]
    _write_rows(csv_path, rows)
    _write_summary_rows(summary_csv_path, rows)
    _write_projection_detail_rows(projection_detail_csv_path, projection_detail_rows)

    completed_before = sum(
        1
        for layer_idx, module_name in target_keys
        if _target_completed(rows, layer_idx, module_name, protocol_version, experiment_signature)
    )
    if resume_enabled and completed_before == len(target_keys) and rows:
        for layer_idx, module_name in target_keys:
            if fp32_reference_state is not None:
                oracle_paths[(layer_idx, module_name)] = fp32_reference_state["oer_oracle_path"]
            else:
                oracle_path = _target_fp32_oer_oracle_path(cfg, layer_idx, module_name)
                if oracle_path.exists():
                    oracle_paths[(layer_idx, module_name)] = oracle_path
            quant_state_path = _target_quant_state_path(cfg, layer_idx, module_name)
            if quant_state_path.exists():
                quant_state_paths[(layer_idx, module_name)] = quant_state_path
        fp32_oer_oracle_path = _save_fp32_oer_oracle_index(
            cfg,
            oracle_paths,
            protocol_version,
            experiment_signature,
            signature_payload,
        )
        import pandas as pd

        detail_df = pd.read_csv(csv_path)
        summary_df = pd.read_csv(summary_csv_path)
        plot_paths = plot_all_heatmaps(detail_df, summary_df, plot_dir)
        return {
            "result_dir": output_dir,
            "csv_path": csv_path,
            "summary_csv_path": summary_csv_path,
            "projection_detail_csv_path": projection_detail_csv_path,
            "manifest_path": manifest_path,
            "plot_paths": plot_paths,
            "fp32_embedding_path": fp32_embedding_path,
            "fp32_oer_oracle_path": fp32_oer_oracle_path,
            "fp32_oer_oracle_paths": oracle_paths,
            "quant_state_paths": quant_state_paths,
            "protocol_version": protocol_version,
            "experiment_signature": experiment_signature,
            "num_experiments": len(target_keys),
            "num_completed_targets": completed_before,
            "num_rows": len(rows),
            "num_projection_detail_rows": len(projection_detail_rows),
            "resume_enabled": resume_enabled,
        }

    calibration_loader = _build_audio_loader(cfg, name="calibration", mode="calibration")
    eval_loader = _build_audio_loader(cfg, name="eval", mode="eval")

    if fp32_reference_state is not None:
        fp32_cache = fp32_reference_state["fp32_cache"]
        fp32_embedding_path = fp32_reference_state["core_cache_path"]
    else:
        fp32_cache = (
            _load_fp32_embeddings_if_valid(cfg, protocol_version, experiment_signature)
            if resume_enabled
            else None
        )
        if fp32_cache is None:
            baseline_model = restore_clean_model_from_checkpoint(cfg)
            fp32_cache = run_fp32_baseline(cfg, baseline_model, eval_loader)
            fp32_embedding_path = _save_fp32_embeddings(
                cfg,
                fp32_cache,
                protocol_version,
                experiment_signature,
                signature_payload,
            )
            del baseline_model
            _release_memory(_device(cfg))
        else:
            fp32_embedding_path = _baseline_embedding_path(cfg)
    for layer_idx, module_name in target_keys:
        target_key = (int(layer_idx), module_name)
        if resume_enabled and _target_completed(rows, int(layer_idx), module_name, protocol_version, experiment_signature):
            if fp32_reference_state is not None:
                oracle_paths[target_key] = fp32_reference_state["oer_oracle_path"]
            else:
                oracle_path = _target_fp32_oer_oracle_path(cfg, int(layer_idx), module_name)
                if oracle_path.exists():
                    oracle_paths[target_key] = oracle_path
            quant_state_path = _target_quant_state_path(cfg, int(layer_idx), module_name)
            if quant_state_path.exists():
                quant_state_paths[target_key] = quant_state_path
            print(f"[resume] skip completed target: layer_{int(layer_idx):02d}_{module_name}")
            continue

        if resume_enabled:
            cleaned_rows = _drop_target_rows(rows, int(layer_idx), module_name)
            if len(cleaned_rows) != len(rows):
                rows = cleaned_rows
                _write_rows(csv_path, rows)
                _write_summary_rows(summary_csv_path, rows)
            cleaned_projection_detail_rows = _drop_target_rows(projection_detail_rows, int(layer_idx), module_name)
            if len(cleaned_projection_detail_rows) != len(projection_detail_rows):
                projection_detail_rows = cleaned_projection_detail_rows
                _write_projection_detail_rows(projection_detail_csv_path, projection_detail_rows)

        if rowwise_fp32_oer_oracle is not None:
            oer_oracle = _target_oer_oracle_from_rowwise_reference(
                rowwise_fp32_oer_oracle,
                fp32_cache,
                int(layer_idx),
                module_name,
            )
            _assert_same_oracle_order(fp32_cache, oer_oracle)
            oracle_paths[target_key] = fp32_reference_state["oer_oracle_path"]
        else:
            oer_oracle = (
                _load_target_fp32_oer_oracle_if_valid(
                    cfg,
                    fp32_cache,
                    int(layer_idx),
                    module_name,
                    protocol_version,
                    experiment_signature,
                )
                if resume_enabled
                else None
            )
            if oer_oracle is None:
                oer_oracle = _collect_target_fp32_oer_oracle(cfg, eval_loader, int(layer_idx), module_name)
                _assert_same_oracle_order(fp32_cache, oer_oracle)
                oracle_paths[target_key] = _save_target_fp32_oer_oracle(
                    cfg,
                    int(layer_idx),
                    module_name,
                    oer_oracle,
                    protocol_version,
                    experiment_signature,
                    signature_payload,
                )
            else:
                oracle_paths[target_key] = _target_fp32_oer_oracle_path(cfg, int(layer_idx), module_name)
        group_rows, group_projection_detail_rows, quant_state_path = run_single_intervention(
            cfg,
            int(layer_idx),
            module_name,
            calibration_loader,
            eval_loader,
            fp32_cache,
            oer_oracle,
        )
        quant_state_paths[target_key] = quant_state_path
        rows.extend(_attach_protocol_fields(group_rows, protocol_version, experiment_signature))
        projection_detail_rows.extend(_attach_protocol_fields(group_projection_detail_rows, protocol_version, experiment_signature))
        _write_rows(csv_path, rows)
        _write_summary_rows(summary_csv_path, rows)
        _write_projection_detail_rows(projection_detail_csv_path, projection_detail_rows)
        print(f"[resume] completed target: layer_{int(layer_idx):02d}_{module_name}")
        del oer_oracle
        _release_memory(_device(cfg))

    fp32_oer_oracle_path = _save_fp32_oer_oracle_index(
        cfg,
        oracle_paths,
        protocol_version,
        experiment_signature,
        signature_payload,
    )

    import pandas as pd

    detail_df = pd.read_csv(csv_path)
    summary_df = pd.read_csv(summary_csv_path)
    plot_paths = plot_all_heatmaps(detail_df, summary_df, plot_dir)
    completed_after = sum(
        1
        for layer_idx, module_name in target_keys
        if _target_completed(rows, layer_idx, module_name, protocol_version, experiment_signature)
    )
    return {
        "result_dir": output_dir,
        "csv_path": csv_path,
        "summary_csv_path": summary_csv_path,
        "projection_detail_csv_path": projection_detail_csv_path,
        "manifest_path": manifest_path,
        "plot_paths": plot_paths,
        "fp32_embedding_path": fp32_embedding_path,
        "fp32_oer_oracle_path": fp32_oer_oracle_path,
        "fp32_oer_oracle_paths": oracle_paths,
        "quant_state_paths": quant_state_paths,
        "protocol_version": protocol_version,
        "experiment_signature": experiment_signature,
        "num_experiments": len(target_keys),
        "num_completed_targets": completed_after,
        "num_rows": len(rows),
        "num_projection_detail_rows": len(projection_detail_rows),
        "resume_enabled": resume_enabled,
    }
if __name__ == "__main__":
    import config as project_config

    summary = main(project_config)
    print(f"FFN 层热图实验完成：{summary['csv_path']}")




















