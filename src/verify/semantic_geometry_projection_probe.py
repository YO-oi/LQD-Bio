from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

try:
    from src.utils.semantic_geometry_cache import (
        DEFAULT_EMBEDDING_DIM,
        SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION,
        SEMANTIC_GEOMETRY_EXPERIMENT_PROTOCOL_VERSION,
        atomic_json_write,
        atomic_torch_save,
        build_experiment_signature,
        build_semantic_embedding_identity,
        done_marker_path,
        done_marker_valid,
        file_identity,
        load_embedding_cache,
        load_fp32_embedding_cache_from_core_eval,
        qparams_identity_manifest,
        sample_order_hash,
        stable_json_hash,
        write_done_marker,
    )
    from src.metrics.semantic_geometry_metrics import (
        CLASS_LEVEL_COLUMNS,
        COMPARISON_SUMMARY_COLUMNS,
        GLOBAL_LEVEL_COLUMNS,
        GROUP_LEVEL_COLUMNS,
        SAMPLE_LEVEL_COLUMNS,
        SUMMARY_V2_LONG_COLUMNS,
        build_fp32_semantic_geometry_reference,
        build_semantic_geometry_comparison_summary,
        build_semantic_geometry_summary_v2_long,
        compute_semantic_geometry_metrics,
    )
    from src.plots.semantic_geometry_plotter import plot_all_semantic_geometry_figures
    from src.utils.group_utils import ANALYSIS_GROUPS, normalize_analysis_group
    from src.utils.semantic_geometry_mode_runner import (
        DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES,
        ModeRunResult,
        build_calibration_loader,
        build_eval_loader,
        load_eval_metadata,
        load_or_calibrate_qparams,
        normalize_activation_granularity,
        normalize_mode_label,
        run_embedding_mode,
        target_modules_from_config,
    )
except ImportError:  # pragma: no cover - 兼容直接以脚本方式运行。
    from ..utils.semantic_geometry_cache import (
        DEFAULT_EMBEDDING_DIM,
        SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION,
        SEMANTIC_GEOMETRY_EXPERIMENT_PROTOCOL_VERSION,
        atomic_json_write,
        atomic_torch_save,
        build_experiment_signature,
        build_semantic_embedding_identity,
        done_marker_path,
        done_marker_valid,
        file_identity,
        load_embedding_cache,
        load_fp32_embedding_cache_from_core_eval,
        qparams_identity_manifest,
        sample_order_hash,
        stable_json_hash,
        write_done_marker,
    )
    from ..metrics.semantic_geometry_metrics import (
        CLASS_LEVEL_COLUMNS,
        COMPARISON_SUMMARY_COLUMNS,
        GLOBAL_LEVEL_COLUMNS,
        GROUP_LEVEL_COLUMNS,
        SAMPLE_LEVEL_COLUMNS,
        SUMMARY_V2_LONG_COLUMNS,
        build_fp32_semantic_geometry_reference,
        build_semantic_geometry_comparison_summary,
        build_semantic_geometry_summary_v2_long,
        compute_semantic_geometry_metrics,
    )
    from ..plots.semantic_geometry_plotter import plot_all_semantic_geometry_figures
    from ..utils.group_utils import ANALYSIS_GROUPS, normalize_analysis_group
    from ..utils.semantic_geometry_mode_runner import (
        DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES,
        ModeRunResult,
        build_calibration_loader,
        build_eval_loader,
        load_eval_metadata,
        load_or_calibrate_qparams,
        normalize_activation_granularity,
        normalize_mode_label,
        run_embedding_mode,
        target_modules_from_config,
    )


PROBE_CFG_NAME = "SEMANTIC_GEOMETRY_PROJECTION"
DEFAULT_QUANT_MODES = ("A4-only", "W4A4", "W4-only")
DEFAULT_ACT_QUANT_GRANULARITIES = ("per_tensor",)
OPTIONAL_ACT_QUANT_GRANULARITIES = ("per_channel",)
DEFAULT_ACTIVATION_GRANULARITIES = DEFAULT_ACT_QUANT_GRANULARITIES
DEFAULT_RUN_SPECS = (("A4-only", "per_tensor"), ("W4A4", "per_tensor"), ("W4-only", "per_tensor"))
OPTIONAL_PER_CHANNEL_RUN_SPECS = (("A4-only", "per_channel"), ("W4A4", "per_channel"))
DEFAULT_SCOPES = ("full_model",)
SINGLE_TARGET_SCOPE = "single_target"
FULL_MODEL_SCOPE = "full_model"
FP32_LAYER_IDX = -1
FP32_MODULE_NAME = "fp32"
FP32_REFERENCE_CORE_FILENAME = "core_eval.pt"
CORE_CACHE_TYPE = "fp32_core_cache"
CORE_STORAGE_FORMAT = "sample_sharded_v1"
CORE_LAYOUT_PROTOCOL_VERSION = 1
CORE_CHECKPOINT_HASH_TYPE = "sha256"
FULL_MODEL_LAYER_IDX = -2
FULL_MODEL_MODULE_NAME = "all_ffn"
METRIC_SHARD_DIRNAME = "metric_shards"
SAMPLE_METRICS_FILENAME = "semantic_sample_metrics.csv"
CLASS_METRICS_FILENAME = "semantic_class_metrics.csv"
GROUP_METRICS_FILENAME = "semantic_group_metrics.csv"
SUMMARY_METRICS_FILENAME = "semantic_summary_metrics.csv"
SUMMARY_V2_LONG_FILENAME = "semantic_geometry_summary_v2_long.csv"
COMPARISON_SUMMARY_FILENAME = "semantic_geometry_comparison_summary.csv"
SUMMARY_JSON_FILENAME = "semantic_geometry_projection_summary.json"
PROJECTION_METRICS_FILENAME = "semantic_embedding_projection_2d.csv"
CLASS_CENTER_ARTIFACT_FILENAME = "semantic_class_centers.pt"
SAMPLE_MARGIN_VECTOR_FILENAME = "semantic_sample_margin_vectors.csv"
DISTORTION_TOPK_FILENAME = "semantic_distortion_topk.csv"
KNOWN_QUANT_MODES = ("A4-only", "W4A4", "W4-only")
PROJECTION_METRICS_COLUMNS = [
    "scope", "target", "layer_idx", "module_name", "target_name", "embedding_cache_path",
    "sample_id", "label", "label_id", "bucket", "shot_group", "tail_included",
    "quant_mode", "granularity", "projection_x", "projection_y",
]
SAMPLE_MARGIN_VECTOR_COLUMNS = [
    "scope", "target", "layer_idx", "module_name", "target_name", "embedding_cache_path", "qparams_path",
    "sample_id", "label", "label_id", "bucket", "shot_group", "tail_included",
    "quant_mode", "granularity", "embedding_distortion",
    "fp32_embedding_centroid_margin", "quant_embedding_centroid_margin",
    "embedding_centroid_margin_retention", "embedding_centroid_margin_drop",
    "fp32_margin", "quant_margin", "margin_retention", "margin_drop",
]
DISTORTION_TOPK_COLUMNS = [
    "scope", "target", "layer_idx", "module_name", "target_name", "embedding_cache_path", "qparams_path",
    "distortion_rank", "sample_id", "label", "label_id", "bucket", "shot_group", "tail_included",
    "quant_mode", "granularity", "embedding_distortion",
    "fp32_embedding_centroid_margin", "quant_embedding_centroid_margin",
    "embedding_centroid_margin_retention", "embedding_centroid_margin_drop",
    "fp32_margin", "quant_margin", "margin_retention", "margin_drop",
]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def _probe_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, PROBE_CFG_NAME, {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    probe = _probe_cfg(cfg)
    if key in probe:
        return probe[key]
    return _cfg_get(cfg, key, default)


def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
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
    raise AttributeError("无法确定数据集名称：请配置 ACTIVE_DATASET 或 CURRENT_DATASET['ROOT']")


def _output_dir(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "semantic_geometry"


def _embedding_dir(cfg: Any) -> Path:
    return _output_dir(cfg) / "embeddings"


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
    for key in ("CORE_CACHE_PATH", "CORE_EVAL_PATH", "FP32_CORE_CACHE_PATH", "FP32_CORE_EVAL_PATH"):
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
    reference_cfg = _fp32_reference_cfg(cfg)
    for key in ("CORE_CACHE_PATH", "CORE_EVAL_PATH", "FP32_CORE_CACHE_PATH", "FP32_CORE_EVAL_PATH"):
        value = _nested_get(reference_cfg, key, None)
        if value is not None:
            return Path(value)
    return _fp32_reference_cache_root(cfg) / FP32_REFERENCE_CORE_FILENAME


def _core_eval_manifest(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"core_eval.pt must be a mapping: {path}")
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"core_eval.pt manifest must be a mapping: {path}")
    return manifest


def _validate_core_eval_manifest(cfg: Any, *, manifest: Mapping[str, Any], path: Path) -> None:
    required_pairs = {
        "cache_type": CORE_CACHE_TYPE,
        "stage": "quant_injection_pre_fp32_reference",
        "storage_format": CORE_STORAGE_FORMAT,
        "checkpoint_hash_type": CORE_CHECKPOINT_HASH_TYPE,
        "layout_protocol_version": CORE_LAYOUT_PROTOCOL_VERSION,
        "embedding_pooling": "mask_aware_mean",
    }
    for key, expected in required_pairs.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"core_eval.pt manifest[{key}] expected {expected!r}, got {manifest.get(key)!r}: {path}")
    checkpoint_sha256 = manifest.get("checkpoint_sha256")
    checkpoint_hash = manifest.get("checkpoint_hash")
    if not checkpoint_sha256 or not checkpoint_hash:
        raise RuntimeError(f"core_eval.pt must record checkpoint_sha256 and checkpoint_hash: {path}")
    if str(checkpoint_sha256) != str(checkpoint_hash):
        raise RuntimeError(f"core_eval.pt checkpoint_hash must equal checkpoint_sha256: {path}")
    if manifest.get("sample_order_hash") is None:
        raise RuntimeError(f"core_eval.pt missing sample_order_hash: {path}")
    expected_dim = int(_expected_dim(cfg))
    if int(manifest.get("embedding_dim", -1)) != expected_dim:
        raise RuntimeError(f"core_eval.pt embedding_dim expected {expected_dim}, got {manifest.get('embedding_dim')!r}: {path}")


def _figures_dir(cfg: Any) -> Path:
    return _output_dir(cfg) / "figures"


def _target_name(layer_idx: int, module_name: str) -> str:
    if int(layer_idx) == FP32_LAYER_IDX:
        return "fp32"
    if int(layer_idx) == FULL_MODEL_LAYER_IDX:
        return "full_model_quantization"
    return f"encoder.layers.{int(layer_idx)}.{module_name}"

def _mode_slug(mode: str) -> str:
    return normalize_mode_label(mode).lower().replace("-", "_")


def _granularity_slug(granularity: str) -> str:
    return normalize_activation_granularity(granularity).lower()


def _embedding_path(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    granularity: str,
    scope: str = SINGLE_TARGET_SCOPE,
) -> Path:
    if normalize_mode_label(quant_mode) == "FP32":
        return _embedding_dir(cfg) / "fp32_embedding.pt"
    normalized_scope = _normalize_scope(scope)
    if normalized_scope == FULL_MODEL_SCOPE:
        return _embedding_dir(cfg) / FULL_MODEL_SCOPE / f"full_model_{_mode_slug(quant_mode)}_{_granularity_slug(granularity)}_embedding.pt"
    return (
        _embedding_dir(cfg)
        / SINGLE_TARGET_SCOPE
        / f"layer_{int(layer_idx):02d}_{module_name}_{_mode_slug(quant_mode)}_{_granularity_slug(granularity)}_embedding.pt"
    )

def _cache_expected_granularity(quant_mode: str, granularity: str) -> str:
    if normalize_mode_label(quant_mode) == "FP32":
        return "fp32"
    return normalize_activation_granularity(granularity)


def _assert_cache_identity(cache: Mapping[str, Any], *, quant_mode: str, granularity: str, context: str) -> None:
    expected_mode = normalize_mode_label(quant_mode)
    actual_mode = normalize_mode_label(str(cache.get("quant_mode", "")))
    if actual_mode != expected_mode:
        raise ValueError(f"{context}: embedding cache quant_mode mismatch: expected={expected_mode}, actual={actual_mode}")
    expected_granularity = _cache_expected_granularity(expected_mode, granularity)
    actual_granularity = str(cache.get("granularity", ""))
    if actual_granularity != expected_granularity:
        raise ValueError(
            f"{context}: embedding cache granularity mismatch: "
            f"expected={expected_granularity}, actual={actual_granularity}"
        )


def _assert_cache_matches_metadata(
    cfg: Any,
    cache: Mapping[str, Any],
    sample_metadata: Mapping[str, Mapping[str, Any]],
    *,
    context: str,
) -> None:
    if not bool(_setting_get(cfg, "CHECK_CACHE_SAMPLE_ORDER", True)):
        return
    expected_ids = [str(item) for item in sample_metadata.keys()]
    if not expected_ids:
        return
    cache_ids = [str(item) for item in cache.get("sample_id", [])]
    if expected_ids != cache_ids:
        first_bad = next((idx for idx, (left, right) in enumerate(zip(expected_ids, cache_ids)) if left != right), None)
        raise RuntimeError(
            f"{context}: cached sample_id order does not match eval metadata; first_bad={first_bad}. "
            "Set FORCE_REBUILD_EMBEDDINGS=True to rebuild stale embeddings."
        )


def _maybe_load_embedding_cache(
    cfg: Any,
    *,
    path: Path,
    quant_mode: str,
    granularity: str,
    layer_idx: int,
    module_name: str,
    sample_metadata: Mapping[str, Mapping[str, Any]],
    qparams: Mapping[str, Any] | None = None,
    qparams_by_target: Mapping[Any, Mapping[str, Any]] | None = None,
    target_modules: Sequence[tuple[int, str]] | None = None,
    scope: str = SINGLE_TARGET_SCOPE,
) -> ModeRunResult | None:
    if _force_rebuild(cfg, "FORCE_REBUILD_EMBEDDINGS") or not path.exists():
        return None
    cache = load_embedding_cache(path, expected_dim=_expected_dim(cfg), require_label_id=True)
    context = f"{_target_name(layer_idx, module_name)} {normalize_mode_label(quant_mode)} {granularity}"
    _assert_cache_identity(cache, quant_mode=quant_mode, granularity=granularity, context=context)
    _assert_cache_matches_metadata(cfg, cache, sample_metadata, context=context)
    identity_target_modules = (
        [(int(layer_idx), str(module_name))] if target_modules is None else [(int(item_layer), str(item_module)) for item_layer, item_module in target_modules]
    )
    if normalize_mode_label(quant_mode) == "FP32":
        qparams_identity_info = qparams_identity_manifest(None, source_hint="fp32_no_qparams")
    elif qparams_by_target is not None:
        qparams_identity_info = _qparams_identity_for_target_map(qparams_by_target, identity_target_modules)
    else:
        qparams_identity_info = qparams_identity_manifest(qparams, source_hint=_target_name(layer_idx, module_name))
    expected_sample_ids = [str(item) for item in sample_metadata.keys()] or [str(item) for item in cache.get("sample_id", [])]
    _assert_embedding_cache_identity(
        cfg,
        cache,
        scope=scope,
        quant_mode=quant_mode,
        granularity=granularity,
        target_modules=identity_target_modules,
        sample_ids=expected_sample_ids,
        qparams_identity_info=qparams_identity_info,
        context=context,
    )
    runtime_summary: dict[str, Any] = {
        "quant_mode": normalize_mode_label(quant_mode),
        "granularity": _cache_expected_granularity(quant_mode, granularity),
        "num_samples": len(cache["sample_id"]),
        "cache_hit": True,
        "embedding_cache_path": str(path),
    }
    if int(layer_idx) != FP32_LAYER_IDX:
        runtime_summary.update({
            "layer_idx": int(layer_idx),
            "module_name": str(module_name),
            "target_name": _target_name(layer_idx, module_name),
            "qparams_path": None if qparams is None else qparams.get("qparams_path"),
        })
    return ModeRunResult(
        embedding_cache=cache,
        qparams=None if qparams is None else dict(qparams),
        cache_path=path,
        runtime_summary=runtime_summary,
    )


def _embedding_source(
    cache: Mapping[str, Any],
    *,
    layer_idx: int,
    module_name: str,
    embedding_cache_path: Path | None,
    scope: str = SINGLE_TARGET_SCOPE,
) -> dict[str, Any]:
    target_name = _target_name(layer_idx, module_name)
    return {
        "cache": cache,
        "scope": _normalize_scope(scope) if int(layer_idx) != FP32_LAYER_IDX else "fp32",
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "target_name": target_name,
        "target": target_name,
        "embedding_cache_path": None if embedding_cache_path is None else str(embedding_cache_path),
    }


def _cache_qparams_path(cache: Mapping[str, Any]) -> Any:
    manifest = cache.get("manifest", {})
    if isinstance(manifest, Mapping):
        for key in ("qparams_path", "qparams_cache_path"):
            value = manifest.get(key)
            if value is not None:
                return value
    return cache.get("qparams_path")


def _normalize_scope(scope: str) -> str:
    normalized = str(scope).strip().lower().replace("-", "_")
    aliases = {
        "single": SINGLE_TARGET_SCOPE,
        "single_target": SINGLE_TARGET_SCOPE,
        "target": SINGLE_TARGET_SCOPE,
        "full": FULL_MODEL_SCOPE,
        "full_model": FULL_MODEL_SCOPE,
        "full_model_quantization": FULL_MODEL_SCOPE,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported semantic geometry scope: {scope!r}")
    return aliases[normalized]


def _configured_quant_modes(cfg: Any) -> list[str]:
    raw = _setting_get(cfg, "QUANT_MODES", DEFAULT_QUANT_MODES)
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    modes = [normalize_mode_label(str(item)) for item in raw]
    modes = [mode for mode in modes if mode != "FP32"]
    if not modes:
        raise ValueError("SEMANTIC_GEOMETRY_PROJECTION.QUANT_MODES cannot contain only FP32")
    return list(dict.fromkeys(modes))


def _configured_activation_granularities(cfg: Any) -> list[str]:
    raw = _setting_get(cfg, "ACTIVATION_GRANULARITIES", DEFAULT_ACTIVATION_GRANULARITIES)
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    granularities = [normalize_activation_granularity(str(item)) for item in raw]
    if not granularities:
        raise ValueError("ACTIVATION_GRANULARITIES cannot be empty")
    return list(dict.fromkeys(granularities))


def _configured_run_specs(cfg: Any) -> list[tuple[str, str]]:
    raw_specs = _setting_get(cfg, "RUN_SPECS", None)
    specs: list[tuple[str, str]] = []
    if raw_specs is not None:
        if isinstance(raw_specs, Mapping):
            raw_specs = [raw_specs]
        elif isinstance(raw_specs, (list, tuple)) and len(raw_specs) == 2 and isinstance(raw_specs[0], (str, bytes)):
            raw_specs = [raw_specs]
        for item in raw_specs:
            if isinstance(item, Mapping):
                mode = item.get("quant_mode", item.get("mode"))
                granularity = item.get("granularity", item.get("activation_granularity", "per_tensor"))
            else:
                mode, granularity = item
            specs.append((normalize_mode_label(str(mode)), normalize_activation_granularity(str(granularity))))
    else:
        modes = _configured_quant_modes(cfg)
        granularities = _configured_activation_granularities(cfg)
        for mode in modes:
            if mode == "W4-only":
                specs.append((mode, "per_tensor"))
                continue
            for granularity in granularities:
                specs.append((mode, granularity))
        if bool(_setting_get(cfg, "RUN_ACT_PER_CHANNEL_ABLATION", False)):
            specs.extend((normalize_mode_label(mode), normalize_activation_granularity(granularity)) for mode, granularity in OPTIONAL_PER_CHANNEL_RUN_SPECS)
    deduped = list(dict.fromkeys(specs))
    if not deduped:
        raise ValueError("semantic geometry RUN_SPECS cannot be empty")
    spec_set = set(deduped)
    for mode, granularity in deduped:
        if mode == "W4-only" and granularity != "per_tensor":
            raise ValueError("W4-only is a weight-only control and must use per_tensor granularity in this probe")
        if granularity == "per_channel":
            if mode not in {"A4-only", "W4A4"}:
                raise ValueError("activation per_channel ablation is only valid for A4-only and W4A4")
            if (mode, "per_tensor") not in spec_set:
                raise ValueError(f"{mode} per_channel must be paired with {mode} per_tensor")
    return deduped


def _target_modules_explicitly_configured(cfg: Any) -> bool:
    return _setting_get(cfg, "TARGET_MODULES", None) is not None


def _configured_scopes(cfg: Any) -> list[str]:
    raw = _setting_get(cfg, "SCOPES", None)
    if raw is None:
        raw = (SINGLE_TARGET_SCOPE,) if _target_modules_explicitly_configured(cfg) else DEFAULT_SCOPES
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    scopes = [_normalize_scope(str(item)) for item in raw]
    scopes = list(dict.fromkeys(scopes))
    if not scopes:
        raise ValueError("semantic geometry SCOPES cannot be empty")
    return scopes


def _validate_runtime_plan(
    *,
    scopes: Sequence[str],
    target_modules: Sequence[tuple[int, str]],
    custom_quant_runner: Any,
) -> None:
    _ = (scopes, target_modules, custom_quant_runner)


def _expected_dim(cfg: Any) -> int | None:
    value = _setting_get(cfg, "EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
    return None if value is None else int(value)


def _require_logits(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "REQUIRE_LOGITS_FOR_GEOMETRY", _setting_get(cfg, "REQUIRE_LOGITS", False)))


def _compute_logits_if_available(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "COMPUTE_LOGITS_IF_AVAILABLE", True))


def _require_logits_for_classification(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "REQUIRE_LOGITS_FOR_CLASSIFICATION", True))


def _include_fp32_baseline_metrics(cfg: Any) -> bool:
    """是否把 FP32 自身也写入 metrics 表。

    默认关闭：FP32 主要作为 reference/cache。若开启，会产生 distortion=0、retention=1、drop=0
    这类 sanity baseline 行，不能把它当成量化损伤结论。
    """

    return bool(_setting_get(cfg, "INCLUDE_FP32_BASELINE_METRICS", False))


def _force_rebuild(cfg: Any, key: str) -> bool:
    return bool(_setting_get(cfg, key, _setting_get(cfg, "FORCE_REBUILD", False)))


def _group_names(cfg: Any, fp32_cache: Mapping[str, Any]) -> list[str]:
    configured = _setting_get(cfg, "GROUP_NAMES", None)
    if configured is not None:
        return [normalize_analysis_group(item) for item in configured]
    return list(ANALYSIS_GROUPS)


def _qparams_path_from_config(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    activation_granularity: str,
) -> Path | None:
    paths = _setting_get(cfg, "QPARAMS_PATHS", None)
    if not isinstance(paths, Mapping):
        return None
    keys = (
        f"{int(layer_idx)}:{module_name}:{activation_granularity}",
        f"layer_{int(layer_idx):02d}_{module_name}_{activation_granularity}",
        f"{int(layer_idx)}_{module_name}_{activation_granularity}",
        f"{_target_name(layer_idx, module_name)}:{activation_granularity}",
    )
    for key in keys:
        if key in paths and paths[key] is not None:
            return Path(paths[key])
    return None


def _custom_quant_runner(cfg: Any) -> Any:
    runner = _setting_get(cfg, "CUSTOM_QUANT_RUNNER", None)
    if runner is None:
        runner = _cfg_get(cfg, "semantic_geometry_custom_quant_runner", None)
    if runner is not None and not callable(runner):
        raise TypeError("CUSTOM_QUANT_RUNNER 必须是 callable")
    return runner


def _assert_same_sample_order(reference_cache: Mapping[str, Any], current_cache: Mapping[str, Any], context: str) -> None:
    ref_ids = [str(item) for item in reference_cache["sample_id"]]
    cur_ids = [str(item) for item in current_cache["sample_id"]]
    if ref_ids != cur_ids:
        first_bad = next((idx for idx, (left, right) in enumerate(zip(ref_ids, cur_ids)) if left != right), None)
        raise RuntimeError(f"{context}: sample_id 顺序必须与 FP32 reference 完全一致，first_bad={first_bad}")
    if reference_cache.get("label_id") is not None and current_cache.get("label_id") is not None:
        if not torch.equal(reference_cache["label_id"].detach().cpu().long(), current_cache["label_id"].detach().cpu().long()):
            raise RuntimeError(f"{context}: label_id 必须与 FP32 reference 完全一致")


def _to_csv_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if int(tensor.numel()) == 1:
            return tensor.reshape(()).item()
        return json.dumps(tensor.reshape(-1).tolist(), ensure_ascii=False)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], preferred_columns: Sequence[str]) -> Path:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    all_columns: list[str] = []
    for column in preferred_columns:
        if column not in all_columns:
            all_columns.append(column)
    for row in rows:
        for column in row.keys():
            if column not in all_columns:
                all_columns.append(column)
    frame = pd.DataFrame([{key: _to_csv_value(row.get(key)) for key in all_columns} for row in rows], columns=all_columns)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(tmp_path, index=False, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def _target_columns() -> list[str]:
    return ["scope", "target", "layer_idx", "module_name", "target_name", "embedding_cache_path", "qparams_path"]


def _enrich_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    layer_idx: int,
    module_name: str,
    embedding_cache_path: Path | None,
    qparams_path: str | Path | None,
    sample_label_lookup: Mapping[str, Any],
    scope: str = SINGLE_TARGET_SCOPE,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["scope"] = _normalize_scope(scope) if int(layer_idx) != FP32_LAYER_IDX else "fp32"
        item["layer_idx"] = int(layer_idx)
        item["module_name"] = str(module_name)
        item["target_name"] = _target_name(layer_idx, module_name)
        item["target"] = item["target_name"]
        item["embedding_cache_path"] = None if embedding_cache_path is None else str(embedding_cache_path)
        item["qparams_path"] = None if qparams_path is None else str(qparams_path)
        sample_id = item.get("sample_id")
        if sample_id is not None and str(sample_id) in sample_label_lookup:
            item.setdefault("primary_label", sample_label_lookup[str(sample_id)])
        enriched.append(item)
    return enriched


def _build_fp32_reference(
    cfg: Any,
    fp32_cache: Mapping[str, Any],
) -> dict[str, Any]:
    if fp32_cache.get("label_id") is None:
        raise KeyError("FP32 embedding cache 必须包含 label_id，不能只靠行号或临时 label 映射")
    return build_fp32_semantic_geometry_reference(
        sample_id=fp32_cache["sample_id"],
        label_id=fp32_cache["label_id"],
        bucket=fp32_cache["bucket"],
        shot_group=fp32_cache.get("shot_group"),
        fp32_embeddings=fp32_cache["embedding"],
        fp32_logits=fp32_cache.get("logits"),
        group_names=_group_names(cfg, fp32_cache),
        normalize_fisher_by_dof=bool(_setting_get(cfg, "NORMALIZE_FISHER_BY_DOF", False)),
        eps=float(_setting_get(cfg, "EPS", 1e-12)),
    )


def _compute_and_enrich_metrics(
    cfg: Any,
    *,
    fp32_reference: Mapping[str, Any],
    cache: Mapping[str, Any],
    layer_idx: int,
    module_name: str,
    embedding_cache_path: Path | None,
    qparams_path: str | Path | None,
    sample_label_lookup: Mapping[str, Any],
    scope: str = SINGLE_TARGET_SCOPE,
) -> dict[str, list[dict[str, Any]]]:
    tables = compute_semantic_geometry_metrics(
        fp32_reference=fp32_reference,
        quant_embeddings=cache["embedding"],
        quant_logits=cache.get("logits"),
        sample_id=cache["sample_id"],
        quant_mode=str(cache["quant_mode"]),
        granularity=str(cache["granularity"]),
        label_id=cache.get("label_id"),
        bucket=cache.get("bucket"),
        shot_group=cache.get("shot_group"),
        group_names=fp32_reference.get("group_names"),
        few_group=str(_setting_get(cfg, "FEW_GROUP", "few")),
        many_group=str(_setting_get(cfg, "MANY_GROUP", "many")),
        require_logits=_require_logits(cfg),
        eps=float(_setting_get(cfg, "EPS", 1e-12)),
    )
    return {
        name: _enrich_rows(
            rows,
            layer_idx=layer_idx,
            module_name=module_name,
            embedding_cache_path=embedding_cache_path,
            qparams_path=qparams_path,
            sample_label_lookup=sample_label_lookup,
            scope=scope,
        )
        for name, rows in tables.items()
    }


def _append_tables(destination: dict[str, list[dict[str, Any]]], source: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for key in ("sample_level", "class_level", "group_level", "global_level"):
        destination.setdefault(key, []).extend([dict(row) for row in source.get(key, [])])


def _qparams_lookup_key(layer_idx: int, module_name: str) -> tuple[int, str]:
    return (int(layer_idx), str(module_name))


def _metric_shard_dir(cfg: Any) -> Path:
    return _output_dir(cfg) / METRIC_SHARD_DIRNAME


def _metric_shard_paths(cfg: Any, shard_key: str) -> dict[str, Path]:
    root = _metric_shard_dir(cfg)
    return {
        "sample_level": root / f"{shard_key}.sample.csv",
        "class_level": root / f"{shard_key}.class.csv",
        "group_level": root / f"{shard_key}.group.csv",
        "global_level": root / f"{shard_key}.global.csv",
    }


def _metric_shard_columns(level: str) -> list[str]:
    columns_by_level = {
        "sample_level": [*_target_columns(), *SAMPLE_LEVEL_COLUMNS],
        "class_level": [*_target_columns(), *CLASS_LEVEL_COLUMNS],
        "group_level": [*_target_columns(), *GROUP_LEVEL_COLUMNS],
        "global_level": [*_target_columns(), *GLOBAL_LEVEL_COLUMNS],
    }
    return columns_by_level[level]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(path)
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return [dict(row) for row in rows]


def _load_metric_shard(paths: Mapping[str, Path]) -> dict[str, list[dict[str, Any]]]:
    return {level: _read_csv_rows(path) for level, path in paths.items()}


def _write_metric_shard(paths: Mapping[str, Path], tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for level, path in paths.items():
        _write_csv(path, tables.get(level, []), _metric_shard_columns(level))


def _split_csv_path_from_config(cfg: Any, name: str) -> Path | None:
    key = f"{name.upper()}_CSV"
    value = _setting_get(cfg, key, None)
    if value is not None:
        return Path(value)
    dataset_value = _nested_get(_dataset_cfg(cfg), key, None)
    if dataset_value is not None:
        return Path(dataset_value)
    return None


def _checkpoint_identity(cfg: Any) -> dict[str, Any] | None:
    for key in ("CHECKPOINT_PATH", "CKPT_PATH", "MODEL_CHECKPOINT", "MODEL_CHECKPOINT_PATH", "BEATS_CHECKPOINT"):
        value = _setting_get(cfg, key, _cfg_get(cfg, key, None))
        if value is not None:
            return file_identity(value)
    return None


def _qparams_path_map(qparams_by_target: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, qparams in qparams_by_target.items():
        if isinstance(key, tuple) and len(key) == 2:
            target = _target_name(int(key[0]), str(key[1]))
        else:
            target = str(key)
        result[target] = qparams.get("qparams_path") if isinstance(qparams, Mapping) else None
    return result


def _target_qparams_keys(layer_idx: int, module_name: str) -> tuple[Any, ...]:
    return (
        (int(layer_idx), str(module_name)),
        f"{int(layer_idx)}:{module_name}",
        f"layer_{int(layer_idx):02d}_{module_name}",
        _target_name(int(layer_idx), str(module_name)),
    )


def _qparams_for_target_from_map(
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    *,
    layer_idx: int,
    module_name: str,
) -> Mapping[str, Any]:
    for key in _target_qparams_keys(layer_idx, module_name):
        value = qparams_by_target.get(key)
        if isinstance(value, Mapping):
            return value
    known = ", ".join(str(key) for key in list(qparams_by_target.keys())[:8])
    raise KeyError(f"missing qparams for {_target_name(layer_idx, module_name)}; known keys: {known}")


def _qparams_identity_for_target_map(
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    target_modules: Sequence[tuple[int, str]],
) -> dict[str, Any]:
    detail: dict[str, dict[str, Any]] = {}
    identity_map: dict[str, Any] = {}
    source_map: dict[str, Any] = {}
    type_map: dict[str, Any] = {}
    for layer_idx, module_name in target_modules:
        target = _target_name(int(layer_idx), str(module_name))
        qparams = _qparams_for_target_from_map(qparams_by_target, layer_idx=int(layer_idx), module_name=str(module_name))
        item = qparams_identity_manifest(qparams, source_hint=target)
        detail[target] = item
        identity_map[target] = item.get("qparams_identity")
        source_map[target] = item.get("qparams_source")
        type_map[target] = item.get("qparams_identity_type")
    return {
        "qparams_identity_type": "target_map",
        "qparams_identity": identity_map,
        "qparams_source": source_map,
        "qparams_identity_detail": detail,
        "qparams_identity_type_by_target": type_map,
    }


def _target_modules_identity(target_modules: Sequence[tuple[int, str]]) -> list[list[Any]]:
    return [[int(layer_idx), str(module_name)] for layer_idx, module_name in target_modules]


def _embedding_identity_payload(
    cfg: Any,
    *,
    scope: str,
    quant_mode: str,
    granularity: str,
    target_modules: Sequence[tuple[int, str]],
    sample_ids: Sequence[Any],
    qparams_identity_info: Mapping[str, Any],
) -> dict[str, Any]:
    return build_semantic_embedding_identity(
        scope=_normalize_scope(scope) if scope in (SINGLE_TARGET_SCOPE, FULL_MODEL_SCOPE) else str(scope),
        quant_mode=normalize_mode_label(quant_mode),
        granularity=_cache_expected_granularity(quant_mode, granularity),
        target_modules=_target_modules_identity(target_modules),
        sample_ids=sample_ids,
        eval_csv=_split_csv_path_from_config(cfg, "eval"),
        calibration_csv=None if normalize_mode_label(quant_mode) == "FP32" else _split_csv_path_from_config(cfg, "calibration"),
        checkpoint=_checkpoint_identity(cfg),
        quant_config=_cfg_get(cfg, "QUANT", {}),
        compute_logits_if_available=_compute_logits_if_available(cfg),
        embedding_dim=_expected_dim(cfg),
        qparams_identity_type=str(qparams_identity_info.get("qparams_identity_type")),
        qparams_identity=qparams_identity_info.get("qparams_identity"),
        qparams_source=qparams_identity_info.get("qparams_source"),
    )


def _assert_embedding_cache_identity(
    cfg: Any,
    cache: Mapping[str, Any],
    *,
    scope: str,
    quant_mode: str,
    granularity: str,
    target_modules: Sequence[tuple[int, str]],
    sample_ids: Sequence[Any],
    qparams_identity_info: Mapping[str, Any],
    context: str,
) -> None:
    manifest = cache.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError(
            f"{context}: embedding cache lacks a mapping manifest. "
            "Set FORCE_REBUILD_EMBEDDINGS=True to rebuild stale embeddings."
        )
    if manifest.get("embedding_cache_identity_version") != SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION:
        raise RuntimeError(
            f"{context}: embedding cache lacks semantic geometry identity v1. "
            "Set FORCE_REBUILD_EMBEDDINGS=True to rebuild stale embeddings."
        )
    cached_identity = manifest.get("embedding_identity")
    if not isinstance(cached_identity, Mapping):
        raise RuntimeError(
            f"{context}: embedding cache manifest lacks embedding_identity. "
            "Set FORCE_REBUILD_EMBEDDINGS=True to rebuild stale embeddings."
        )
    for required_key in ("qparams_identity_type", "qparams_identity", "qparams_source"):
        if required_key not in manifest:
            raise RuntimeError(
                f"{context}: embedding cache manifest lacks {required_key}. "
                "Set FORCE_REBUILD_EMBEDDINGS=True to rebuild stale embeddings."
            )
    expected_identity = _embedding_identity_payload(
        cfg,
        scope=scope,
        quant_mode=quant_mode,
        granularity=granularity,
        target_modules=target_modules,
        sample_ids=sample_ids,
        qparams_identity_info=qparams_identity_info,
    )
    cached = dict(cached_identity)
    if cached != expected_identity:
        changed = sorted(set(cached.keys()) | set(expected_identity.keys()))
        mismatched = [key for key in changed if cached.get(key) != expected_identity.get(key)]
        preview = ", ".join(mismatched[:8])
        raise RuntimeError(
            f"{context}: embedding cache identity mismatch ({preview}). "
            "Set FORCE_REBUILD_EMBEDDINGS=True to rebuild stale embeddings."
        )


def _run_signature_payload(
    cfg: Any,
    *,
    scope: str,
    quant_mode: str,
    activation_granularity: str,
    target_modules: Sequence[tuple[int, str]],
    sample_ids: Sequence[Any],
    qparams_paths: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_scope = _normalize_scope(scope)
    return {
        "protocol_version": SEMANTIC_GEOMETRY_EXPERIMENT_PROTOCOL_VERSION,
        "scope": normalized_scope,
        "quant_mode": normalize_mode_label(quant_mode),
        "activation_granularity": normalize_activation_granularity(activation_granularity),
        "target_modules": [[int(layer_idx), str(module_name)] for layer_idx, module_name in target_modules],
        "sample_order_hash": sample_order_hash(sample_ids),
        "fp32_core_eval_cache": file_identity(_fp32_core_eval_path(cfg)),
        "eval_csv": file_identity(_split_csv_path_from_config(cfg, "eval")),
        "calibration_csv": file_identity(_split_csv_path_from_config(cfg, "calibration")),
        "checkpoint": _checkpoint_identity(cfg),
        "quant_config_hash": stable_json_hash(_cfg_get(cfg, "QUANT", {})),
        "amp": bool(_setting_get(cfg, "AMP", _cfg_get(cfg, "AMP", False))),
        "compute_logits_if_available": _compute_logits_if_available(cfg),
        "embedding_dim": _expected_dim(cfg),
        "margin_definition": "embedding_centroid_margin",
        "logits_margin_role": "classification_auxiliary_sanity_metric",
        "qparams_paths": {str(key): file_identity(value) for key, value in qparams_paths.items()},
    }


def _shard_key(
    *,
    scope: str,
    quant_mode: str,
    activation_granularity: str,
    layer_idx: int,
    module_name: str,
) -> str:
    normalized_scope = _normalize_scope(scope)
    if normalized_scope == FULL_MODEL_SCOPE:
        target_part = "full_model"
    else:
        target_part = f"L{int(layer_idx):02d}__{module_name}"
    return "__".join([
        "eval",
        normalized_scope,
        _mode_slug(quant_mode),
        _granularity_slug(activation_granularity),
        target_part,
        "shard0000",
    ])


def _completed_shard_tables(
    cfg: Any,
    *,
    shard_key: str,
    signature: str,
    embedding_path: Path,
) -> dict[str, list[dict[str, Any]]] | None:
    paths = _metric_shard_paths(cfg, shard_key)
    required_paths = [embedding_path, *paths.values()]
    marker_path = done_marker_path(_output_dir(cfg), shard_key)
    if _force_rebuild(cfg, "FORCE_REBUILD_SEMANTIC_SHARDS") or _force_rebuild(cfg, "FORCE_REBUILD_EMBEDDINGS"):
        return None
    if not done_marker_valid(marker_path, experiment_signature=signature, required_paths=required_paths):
        return None
    return _load_metric_shard(paths)


def _mark_shard_done(
    cfg: Any,
    *,
    shard_key: str,
    signature: str,
    signature_payload: Mapping[str, Any],
    embedding_path: Path,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    paths = _metric_shard_paths(cfg, shard_key)
    _write_metric_shard(paths, tables)
    write_done_marker(
        done_marker_path(_output_dir(cfg), shard_key),
        experiment_signature=signature,
        payload={
            "shard_key": shard_key,
            "signature_payload": dict(signature_payload),
            "num_sample_rows": len(tables.get("sample_level", [])),
            "num_class_rows": len(tables.get("class_level", [])),
            "num_group_rows": len(tables.get("group_level", [])),
            "num_global_rows": len(tables.get("global_level", [])),
        },
        required_paths=[embedding_path, *paths.values()],
    )

def _plot_figures(
    cfg: Any,
    *,
    summary_v2_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    if not bool(_setting_get(cfg, "PLOT_FIGURES", True)):
        return {}
    save_dir = _figures_dir(cfg)
    save_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    summary_frame = pd.DataFrame(summary_v2_rows)
    comparison_frame = pd.DataFrame(comparison_rows)
    projection_frame = pd.DataFrame(projection_rows) if projection_rows else None
    outputs = plot_all_semantic_geometry_figures(
        summary_v2_long_csv=summary_frame,
        comparison_summary_csv=comparison_frame,
        projection_metrics_csv=projection_frame,
        save_dir=save_dir,
    )
    return {key: str(value) for key, value in outputs.items()}


def _save_all_metric_tables(
    cfg: Any,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Path]:
    output_dir = _output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sample_level": output_dir / SAMPLE_METRICS_FILENAME,
        "class_level": output_dir / CLASS_METRICS_FILENAME,
        "group_level": output_dir / GROUP_METRICS_FILENAME,
        "global_level": output_dir / SUMMARY_METRICS_FILENAME,
        "summary_v2_long": output_dir / SUMMARY_V2_LONG_FILENAME,
        "comparison_summary": output_dir / COMPARISON_SUMMARY_FILENAME,
    }
    if _projection_enabled(cfg):
        paths["projection_2d"] = output_dir / PROJECTION_METRICS_FILENAME
    _write_csv(paths["sample_level"], tables.get("sample_level", []), [*_target_columns(), *SAMPLE_LEVEL_COLUMNS])
    _write_csv(paths["class_level"], tables.get("class_level", []), [*_target_columns(), *CLASS_LEVEL_COLUMNS])
    _write_csv(paths["group_level"], tables.get("group_level", []), [*_target_columns(), *GROUP_LEVEL_COLUMNS])
    _write_csv(paths["global_level"], tables.get("global_level", []), [*_target_columns(), *GLOBAL_LEVEL_COLUMNS])
    _write_csv(paths["summary_v2_long"], tables.get("summary_v2_long", []), SUMMARY_V2_LONG_COLUMNS)
    _write_csv(paths["comparison_summary"], tables.get("comparison_summary", []), COMPARISON_SUMMARY_COLUMNS)
    if "projection_2d" in paths:
        _write_csv(paths["projection_2d"], tables.get("projection_2d", []), PROJECTION_METRICS_COLUMNS)
    return paths


def _projection_enabled(cfg: Any) -> bool:
    return bool(_setting_get(
        cfg,
        "SAVE_SEMANTIC_EMBEDDING_PROJECTION_2D",
        _setting_get(cfg, "SAVE_EMBEDDING_PROJECTION_2D", False),
    ))


def _save_class_centers_enabled(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "SAVE_CLASS_CENTERS", True))


def _save_sample_margin_vectors_enabled(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "SAVE_SAMPLE_MARGIN_VECTORS", True))


def _save_distortion_topk_enabled(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "SAVE_DISTORTION_TOPK", False))


def _fit_fp32_projection(fp32_cache: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = fp32_cache["embedding"].detach().cpu().float()
    if embedding.ndim != 2:
        raise ValueError(f"FP32 projection embedding must be [N,D], got {tuple(embedding.shape)}")
    _, dim = embedding.shape
    mean = embedding.mean(dim=0, keepdim=True)
    centered = embedding - mean
    if int(dim) <= 0:
        raise ValueError("embedding dimension must be positive")
    basis_cols = min(2, int(dim))
    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[:basis_cols].T.contiguous()
    except RuntimeError:
        basis = torch.eye(int(dim), dtype=embedding.dtype)[:, :basis_cols].contiguous()
    if int(basis.shape[1]) < 2:
        basis = torch.cat([basis, basis.new_zeros((int(dim), 2 - int(basis.shape[1])))], dim=1)
    return mean, basis


def _label_id_value(cache: Mapping[str, Any], index: int) -> Any:
    value = cache.get("label_id")
    if value is None:
        return None
    tensor = value.detach().cpu().reshape(-1) if isinstance(value, torch.Tensor) else torch.as_tensor(value).reshape(-1)
    return int(tensor[int(index)].item())


def _build_projection_rows(
    cfg: Any,
    *,
    fp32_cache: Mapping[str, Any],
    embedding_sources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not _projection_enabled(cfg):
        return []
    mean, basis = _fit_fp32_projection(fp32_cache)
    rows: list[dict[str, Any]] = []
    for source in embedding_sources:
        cache = source["cache"]
        embedding = cache["embedding"].detach().cpu().float()
        if int(embedding.shape[1]) != int(mean.shape[1]):
            raise ValueError(f"projection dim mismatch for {source.get('target_name')}: {tuple(embedding.shape)} vs {tuple(mean.shape)}")
        coords = (embedding - mean) @ basis
        sample_ids = [str(item) for item in cache["sample_id"]]
        labels = list(cache.get("label", [None] * len(sample_ids)))
        buckets = [str(item) for item in cache.get("bucket", [""] * len(sample_ids))]
        shot_groups = [str(item) for item in cache.get("shot_group", buckets)]
        tail_flags = list(cache.get("tail_included", [False] * len(sample_ids)))
        for idx, sample_id in enumerate(sample_ids):
            rows.append({
                "scope": source.get("scope"),
                "target": source.get("target"),
                "layer_idx": source.get("layer_idx"),
                "module_name": source.get("module_name"),
                "target_name": source.get("target_name"),
                "embedding_cache_path": source.get("embedding_cache_path"),
                "sample_id": sample_id,
                "label": labels[idx],
                "label_id": _label_id_value(cache, idx),
                "bucket": buckets[idx],
                "shot_group": shot_groups[idx],
                "tail_included": bool(tail_flags[idx]),
                "quant_mode": str(cache.get("quant_mode", "")),
                "granularity": str(cache.get("granularity", "")),
                "projection_x": float(coords[idx, 0].item()),
                "projection_y": float(coords[idx, 1].item()),
            })
    return rows


def _finite_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _class_center_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    cache = source["cache"]
    embedding = cache["embedding"].detach().cpu().float()
    label_id = cache.get("label_id")
    if label_id is None:
        raise KeyError(f"{source.get('target_name')} {cache.get('quant_mode')} cache lacks label_id for class center export")
    labels = label_id.detach().cpu().long().reshape(-1) if isinstance(label_id, torch.Tensor) else torch.as_tensor(label_id, dtype=torch.long).reshape(-1)
    if int(labels.numel()) != int(embedding.shape[0]):
        raise ValueError("label_id count must match embedding count for class center export")
    class_ids = torch.unique(labels, sorted=True)
    centers: list[torch.Tensor] = []
    counts: list[int] = []
    class_labels: list[Any] = []
    class_buckets: list[str] = []
    cache_labels = list(cache.get("label", [None] * int(labels.numel())))
    buckets = [str(item) for item in cache.get("bucket", [""] * int(labels.numel()))]
    for class_id in class_ids:
        mask = labels == class_id
        indices = mask.nonzero(as_tuple=False).reshape(-1)
        first_idx = int(indices[0].item())
        centers.append(embedding[mask].mean(dim=0))
        counts.append(int(mask.sum().item()))
        class_labels.append(cache_labels[first_idx])
        class_buckets.append(buckets[first_idx])
    return {
        "scope": source.get("scope"),
        "target": source.get("target"),
        "layer_idx": source.get("layer_idx"),
        "module_name": source.get("module_name"),
        "target_name": source.get("target_name"),
        "quant_mode": str(cache.get("quant_mode", "")),
        "granularity": str(cache.get("granularity", "")),
        "embedding_cache_path": source.get("embedding_cache_path"),
        "class_ids": class_ids,
        "class_labels": class_labels,
        "class_buckets": class_buckets,
        "counts": torch.as_tensor(counts, dtype=torch.long),
        "centers": torch.stack(centers, dim=0),
    }


def _sample_margin_vector_rows(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in tables.get("sample_level", []):
        rows.append({column: row.get(column) for column in SAMPLE_MARGIN_VECTOR_COLUMNS})
    return rows


def _distortion_topk_rows(cfg: Any, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    topk = int(_setting_get(cfg, "DISTORTION_TOPK", 50))
    if topk <= 0:
        return []
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    key_columns = ("target", "layer_idx", "module_name", "target_name", "quant_mode", "granularity")
    for row in tables.get("sample_level", []):
        distortion = _finite_float(row.get("embedding_distortion"))
        if not math.isfinite(distortion):
            continue
        key = tuple(row.get(column, "") for column in key_columns)
        groups.setdefault(key, []).append(row)
    rows: list[dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda item: tuple(str(part) for part in item)):
        sorted_rows = sorted(groups[key], key=lambda row: _finite_float(row.get("embedding_distortion")), reverse=True)
        for rank, row in enumerate(sorted_rows[:topk], start=1):
            item = {column: row.get(column) for column in DISTORTION_TOPK_COLUMNS}
            item["distortion_rank"] = rank
            rows.append(item)
    return rows


def _save_explanation_outputs(
    cfg: Any,
    *,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    embedding_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    output_dir = _output_dir(cfg)
    paths: dict[str, Path] = {}
    if _save_class_centers_enabled(cfg):
        paths["class_centers"] = output_dir / CLASS_CENTER_ARTIFACT_FILENAME
        class_center_payload = {
            "format_version": 1,
            "definition": "class center = mean final semantic embedding by label_id for each target/mode cache",
            "cache_module_path": "src/utils/semantic_geometry_cache.py",
            "class_centers": {},
        }
        for source in embedding_sources:
            cache = source["cache"]
            key = "|".join([
                str(source.get("target_name")),
                str(cache.get("quant_mode", "")),
                str(cache.get("granularity", "")),
            ])
            class_center_payload["class_centers"][key] = _class_center_payload(source)
        paths["class_centers"].parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(class_center_payload, paths["class_centers"])
    if _save_sample_margin_vectors_enabled(cfg):
        paths["sample_margin_vectors"] = output_dir / SAMPLE_MARGIN_VECTOR_FILENAME
        _write_csv(paths["sample_margin_vectors"], _sample_margin_vector_rows(tables), SAMPLE_MARGIN_VECTOR_COLUMNS)
    if _save_distortion_topk_enabled(cfg):
        paths["distortion_topk"] = output_dir / DISTORTION_TOPK_FILENAME
        _write_csv(paths["distortion_topk"], _distortion_topk_rows(cfg, tables), DISTORTION_TOPK_COLUMNS)
    return paths


def _save_summary_json(cfg: Any, summary: Mapping[str, Any]) -> Path:
    path = _output_dir(cfg) / SUMMARY_JSON_FILENAME
    return atomic_json_write(path, summary)

def _run_fp32(
    cfg: Any,
    *,
    eval_loader: DataLoader,
    sample_metadata: Mapping[str, Mapping[str, Any]],
) -> ModeRunResult:
    _ = eval_loader
    path = _fp32_core_eval_path(cfg)
    manifest = _core_eval_manifest(path)
    _validate_core_eval_manifest(cfg, manifest=manifest, path=path)
    cache = load_fp32_embedding_cache_from_core_eval(
        path,
        expected_dim=_expected_dim(cfg),
        require_core_cache_type=True,
    )
    cache["source_core_eval_manifest"] = dict(manifest)
    cache["source_core_eval_path"] = path
    _assert_cache_matches_metadata(cfg, cache, sample_metadata, context="core_eval.pt FP32 reference")
    return ModeRunResult(
        embedding_cache=cache,
        qparams=None,
        cache_path=path,
        runtime_summary={
            "quant_mode": "FP32",
            "granularity": "fp32",
            "scope": "fp32",
            "device": "core_eval_cache",
            "num_samples": len(cache["sample_id"]),
            "cache_hit": True,
            "fp32_reference_source": "core_eval.pt",
            "core_eval_path": str(path),
            "core_eval_cache_type": manifest.get("cache_type"),
            "sample_order_hash": manifest.get("sample_order_hash"),
            "checkpoint_hash": manifest.get("checkpoint_hash"),
            "checkpoint_hash_type": manifest.get("checkpoint_hash_type"),
            "layout_protocol_version": manifest.get("layout_protocol_version"),
            "compute_logits_if_available": _compute_logits_if_available(cfg),
            "logits_saved": "logits" in cache,
        },
    )


def _run_quant_mode(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    activation_granularity: str,
    qparams: Mapping[str, Any],
    eval_loader: DataLoader,
    calibration_loader: DataLoader,
    sample_metadata: Mapping[str, Mapping[str, Any]],
    custom_quant_runner: Any,
    scope: str = SINGLE_TARGET_SCOPE,
) -> ModeRunResult:
    path = _embedding_path(
        cfg,
        layer_idx=int(layer_idx),
        module_name=module_name,
        quant_mode=quant_mode,
        granularity=activation_granularity,
        scope=scope,
    )
    cached = _maybe_load_embedding_cache(
        cfg,
        path=path,
        quant_mode=quant_mode,
        granularity=activation_granularity,
        layer_idx=int(layer_idx),
        module_name=module_name,
        sample_metadata=sample_metadata,
        qparams=qparams,
        target_modules=[(int(layer_idx), str(module_name))],
        scope=scope,
    )
    if cached is not None:

        cached.runtime_summary.setdefault("scope", _normalize_scope(scope))
        return cached
    result = run_embedding_mode(
        cfg,
        quant_mode=quant_mode,
        layer_idx=int(layer_idx),
        module_name=module_name,
        activation_granularity=activation_granularity,
        qparams=qparams,
        calibration_loader=calibration_loader,
        eval_loader=eval_loader,
        sample_metadata=sample_metadata,
        save=True,
        save_path=path,
        expected_dim=_expected_dim(cfg),
        force_recalibrate=False,
        custom_quant_runner=custom_quant_runner,
        scope=scope,
    )
    result.runtime_summary["cache_hit"] = False
    result.runtime_summary["scope"] = _normalize_scope(scope)
    return result


def _run_full_model_quant_mode(
    cfg: Any,
    *,
    quant_mode: str,
    activation_granularity: str,
    target_modules: Sequence[tuple[int, str]],
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    eval_loader: DataLoader,
    calibration_loader: DataLoader,
    sample_metadata: Mapping[str, Mapping[str, Any]],
    custom_quant_runner: Any,
) -> ModeRunResult:
    path = _embedding_path(
        cfg,
        layer_idx=FULL_MODEL_LAYER_IDX,
        module_name=FULL_MODEL_MODULE_NAME,
        quant_mode=quant_mode,
        granularity=activation_granularity,
        scope=FULL_MODEL_SCOPE,
    )
    cached = _maybe_load_embedding_cache(
        cfg,
        path=path,
        quant_mode=quant_mode,
        granularity=activation_granularity,
        layer_idx=FULL_MODEL_LAYER_IDX,
        module_name=FULL_MODEL_MODULE_NAME,
        sample_metadata=sample_metadata,
        qparams_by_target=qparams_by_target,
        target_modules=[(int(layer_idx), str(module_name)) for layer_idx, module_name in target_modules],
        scope=FULL_MODEL_SCOPE,
    )
    if cached is not None:
        cached.runtime_summary.setdefault("scope", FULL_MODEL_SCOPE)
        cached.runtime_summary.setdefault("num_targets", len(target_modules))
        return cached
    result = run_embedding_mode(
        cfg,
        quant_mode=quant_mode,
        activation_granularity=activation_granularity,
        qparams_by_target=qparams_by_target,
        target_modules=[(int(layer_idx), str(module_name)) for layer_idx, module_name in target_modules],
        scope=FULL_MODEL_SCOPE,
        calibration_loader=calibration_loader,
        eval_loader=eval_loader,
        sample_metadata=sample_metadata,
        save=True,
        save_path=path,
        expected_dim=_expected_dim(cfg),
        custom_quant_runner=custom_quant_runner,
    )
    result.runtime_summary["cache_hit"] = False
    result.runtime_summary["scope"] = FULL_MODEL_SCOPE
    return result

def run_semantic_geometry_projection_probe(cfg: Any) -> dict[str, Any]:
    """Schedule final semantic-space geometry projection with resumable shards."""

    output_dir = _output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    _embedding_dir(cfg).mkdir(parents=True, exist_ok=True)
    (_embedding_dir(cfg) / SINGLE_TARGET_SCOPE).mkdir(parents=True, exist_ok=True)
    (_embedding_dir(cfg) / FULL_MODEL_SCOPE).mkdir(parents=True, exist_ok=True)
    _metric_shard_dir(cfg).mkdir(parents=True, exist_ok=True)

    custom_runner = _custom_quant_runner(cfg)
    run_specs = _configured_run_specs(cfg)
    scopes = _configured_scopes(cfg)
    target_modules = target_modules_from_config(cfg)
    _validate_runtime_plan(scopes=scopes, target_modules=target_modules, custom_quant_runner=custom_runner)

    eval_loader = build_eval_loader(cfg)
    calibration_loader = build_calibration_loader(cfg)
    sample_metadata = load_eval_metadata(cfg)

    fp32_result = _run_fp32(cfg, eval_loader=eval_loader, sample_metadata=sample_metadata)
    fp32_cache = fp32_result.embedding_cache
    fp32_reference = _build_fp32_reference(cfg, fp32_cache)
    sample_label_lookup = {str(sid): label for sid, label in zip(fp32_cache["sample_id"], fp32_cache["label"])}
    fp32_sample_ids = [str(item) for item in fp32_cache["sample_id"]]

    tables: dict[str, list[dict[str, Any]]] = {
        "sample_level": [],
        "class_level": [],
        "group_level": [],
        "global_level": [],
        "summary_v2_long": [],
        "comparison_summary": [],
    }
    if _include_fp32_baseline_metrics(cfg):
        fp32_tables = _compute_and_enrich_metrics(
            cfg,
            fp32_reference=fp32_reference,
            cache=fp32_cache,
            layer_idx=FP32_LAYER_IDX,
            module_name=FP32_MODULE_NAME,
            embedding_cache_path=fp32_result.cache_path,
            qparams_path=None,
            sample_label_lookup=sample_label_lookup,
            scope="fp32",
        )
        for level_rows in fp32_tables.values():
            for row in level_rows:
                row["is_fp32_baseline_metric"] = True
        _append_tables(tables, fp32_tables)

    runtime_runs: list[dict[str, Any]] = [dict(fp32_result.runtime_summary)]
    embedding_sources: list[dict[str, Any]] = [
        _embedding_source(
            fp32_cache,
            layer_idx=FP32_LAYER_IDX,
            module_name=FP32_MODULE_NAME,
            embedding_cache_path=fp32_result.cache_path,
            scope="fp32",
        )
    ]
    completed_shard_count = 0
    computed_shard_count = 0

    for scope in scopes:
        if scope == SINGLE_TARGET_SCOPE:
            for layer_idx, module_name in target_modules:
                for quant_mode, activation_granularity in run_specs:
                    if activation_granularity not in DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES and custom_runner is None:
                        raise NotImplementedError(
                            f"{_target_name(layer_idx, module_name)} {activation_granularity}: "
                            "default runtime only supports activation per_tensor; configure CUSTOM_QUANT_RUNNER for per_channel"
                        )
                    qparams_path = _qparams_path_from_config(
                        cfg,
                        layer_idx=int(layer_idx),
                        module_name=module_name,
                        activation_granularity=activation_granularity,
                    )
                    qparams = load_or_calibrate_qparams(
                        cfg,
                        layer_idx=int(layer_idx),
                        module_name=module_name,
                        calibration_loader=calibration_loader,
                        activation_granularity=activation_granularity,
                        force_recalibrate=_force_rebuild(cfg, "FORCE_RECALIBRATE_QPARAMS"),
                        qparams_path=qparams_path,
                    )
                    target_list = [(int(layer_idx), str(module_name))]
                    embedding_path = _embedding_path(
                        cfg,
                        layer_idx=int(layer_idx),
                        module_name=module_name,
                        quant_mode=quant_mode,
                        granularity=activation_granularity,
                        scope=scope,
                    )
                    shard_key = _shard_key(
                        scope=scope,
                        quant_mode=quant_mode,
                        activation_granularity=activation_granularity,
                        layer_idx=int(layer_idx),
                        module_name=module_name,
                    )
                    qparams_paths = {_target_name(layer_idx, module_name): qparams.get("qparams_path")}
                    signature_payload = _run_signature_payload(
                        cfg,
                        scope=scope,
                        quant_mode=quant_mode,
                        activation_granularity=activation_granularity,
                        target_modules=target_list,
                        sample_ids=fp32_sample_ids,
                        qparams_paths=qparams_paths,
                    )
                    signature = build_experiment_signature(signature_payload)
                    done_tables = _completed_shard_tables(cfg, shard_key=shard_key, signature=signature, embedding_path=embedding_path)
                    if done_tables is not None:
                        result = _maybe_load_embedding_cache(
                            cfg,
                            path=embedding_path,
                            quant_mode=quant_mode,
                            granularity=activation_granularity,
                            layer_idx=int(layer_idx),
                            module_name=module_name,
                            sample_metadata=sample_metadata,
                            qparams=qparams,
                            target_modules=target_list,
                            scope=scope,
                        )
                        if result is None:
                            raise RuntimeError(f"DONE marker is valid but embedding cache could not be loaded: {embedding_path}")
                        _assert_same_sample_order(fp32_cache, result.embedding_cache, f"{_target_name(layer_idx, module_name)} {quant_mode}")
                        _append_tables(tables, done_tables)
                        embedding_sources.append(_embedding_source(result.embedding_cache, layer_idx=int(layer_idx), module_name=module_name, embedding_cache_path=result.cache_path, scope=scope))
                        runtime_runs.append({**dict(result.runtime_summary), "scope": scope, "shard_key": shard_key, "resumed_from_done": True})
                        completed_shard_count += 1
                        continue

                    result = _run_quant_mode(
                        cfg,
                        layer_idx=int(layer_idx),
                        module_name=module_name,
                        quant_mode=quant_mode,
                        activation_granularity=activation_granularity,
                        qparams=qparams,
                        eval_loader=eval_loader,
                        calibration_loader=calibration_loader,
                        sample_metadata=sample_metadata,
                        custom_quant_runner=custom_runner,
                        scope=scope,
                    )
                    _assert_same_sample_order(fp32_cache, result.embedding_cache, f"{_target_name(layer_idx, module_name)} {quant_mode}")
                    resolved_qparams_path = qparams.get("qparams_path") if qparams is not None else _cache_qparams_path(result.embedding_cache)
                    quant_tables = _compute_and_enrich_metrics(
                        cfg,
                        fp32_reference=fp32_reference,
                        cache=result.embedding_cache,
                        layer_idx=int(layer_idx),
                        module_name=module_name,
                        embedding_cache_path=result.cache_path,
                        qparams_path=resolved_qparams_path,
                        sample_label_lookup=sample_label_lookup,
                        scope=scope,
                    )
                    _mark_shard_done(
                        cfg,
                        shard_key=shard_key,
                        signature=signature,
                        signature_payload=signature_payload,
                        embedding_path=embedding_path,
                        tables=quant_tables,
                    )
                    _append_tables(tables, quant_tables)
                    embedding_sources.append(_embedding_source(result.embedding_cache, layer_idx=int(layer_idx), module_name=module_name, embedding_cache_path=result.cache_path, scope=scope))
                    runtime_runs.append({**dict(result.runtime_summary), "scope": scope, "shard_key": shard_key, "resumed_from_done": False})
                    computed_shard_count += 1
        elif scope == FULL_MODEL_SCOPE:
            for quant_mode, activation_granularity in run_specs:
                if activation_granularity not in DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES and custom_runner is None:
                    raise NotImplementedError(
                        f"full_model {activation_granularity}: default runtime only supports activation per_tensor; "
                        "configure CUSTOM_QUANT_RUNNER for per_channel"
                    )
                qparams_by_target: dict[Any, Mapping[str, Any]] = {}
                for layer_idx, module_name in target_modules:
                    qparams_path = _qparams_path_from_config(
                        cfg,
                        layer_idx=int(layer_idx),
                        module_name=module_name,
                        activation_granularity=activation_granularity,
                    )
                    qparams_by_target[_qparams_lookup_key(int(layer_idx), module_name)] = load_or_calibrate_qparams(
                        cfg,
                        layer_idx=int(layer_idx),
                        module_name=module_name,
                        calibration_loader=calibration_loader,
                        activation_granularity=activation_granularity,
                        force_recalibrate=_force_rebuild(cfg, "FORCE_RECALIBRATE_QPARAMS"),
                        qparams_path=qparams_path,
                    )
                embedding_path = _embedding_path(
                    cfg,
                    layer_idx=FULL_MODEL_LAYER_IDX,
                    module_name=FULL_MODEL_MODULE_NAME,
                    quant_mode=quant_mode,
                    granularity=activation_granularity,
                    scope=scope,
                )
                shard_key = _shard_key(
                    scope=scope,
                    quant_mode=quant_mode,
                    activation_granularity=activation_granularity,
                    layer_idx=FULL_MODEL_LAYER_IDX,
                    module_name=FULL_MODEL_MODULE_NAME,
                )
                signature_payload = _run_signature_payload(
                    cfg,
                    scope=scope,
                    quant_mode=quant_mode,
                    activation_granularity=activation_granularity,
                    target_modules=[(int(layer_idx), str(module_name)) for layer_idx, module_name in target_modules],
                    sample_ids=fp32_sample_ids,
                    qparams_paths=_qparams_path_map(qparams_by_target),
                )
                signature = build_experiment_signature(signature_payload)
                done_tables = _completed_shard_tables(cfg, shard_key=shard_key, signature=signature, embedding_path=embedding_path)
                if done_tables is not None:
                    result = _maybe_load_embedding_cache(
                        cfg,
                        path=embedding_path,
                        quant_mode=quant_mode,
                        granularity=activation_granularity,
                        layer_idx=FULL_MODEL_LAYER_IDX,
                        module_name=FULL_MODEL_MODULE_NAME,
                        sample_metadata=sample_metadata,
                        qparams_by_target=qparams_by_target,
                        target_modules=[(int(layer_idx), str(module_name)) for layer_idx, module_name in target_modules],
                        scope=scope,
                    )
                    if result is None:
                        raise RuntimeError(f"DONE marker is valid but full-model embedding cache could not be loaded: {embedding_path}")
                    _assert_same_sample_order(fp32_cache, result.embedding_cache, f"full_model {quant_mode}")
                    _append_tables(tables, done_tables)
                    embedding_sources.append(_embedding_source(result.embedding_cache, layer_idx=FULL_MODEL_LAYER_IDX, module_name=FULL_MODEL_MODULE_NAME, embedding_cache_path=result.cache_path, scope=scope))
                    runtime_runs.append({**dict(result.runtime_summary), "scope": scope, "shard_key": shard_key, "resumed_from_done": True})
                    completed_shard_count += 1
                    continue

                result = _run_full_model_quant_mode(
                    cfg,
                    quant_mode=quant_mode,
                    activation_granularity=activation_granularity,
                    target_modules=[(int(layer_idx), str(module_name)) for layer_idx, module_name in target_modules],
                    qparams_by_target=qparams_by_target,
                    eval_loader=eval_loader,
                    calibration_loader=calibration_loader,
                    sample_metadata=sample_metadata,
                    custom_quant_runner=custom_runner,
                )
                _assert_same_sample_order(fp32_cache, result.embedding_cache, f"full_model {quant_mode}")
                quant_tables = _compute_and_enrich_metrics(
                    cfg,
                    fp32_reference=fp32_reference,
                    cache=result.embedding_cache,
                    layer_idx=FULL_MODEL_LAYER_IDX,
                    module_name=FULL_MODEL_MODULE_NAME,
                    embedding_cache_path=result.cache_path,
                    qparams_path=json.dumps(_qparams_path_map(qparams_by_target), ensure_ascii=False),
                    sample_label_lookup=sample_label_lookup,
                    scope=scope,
                )
                _mark_shard_done(
                    cfg,
                    shard_key=shard_key,
                    signature=signature,
                    signature_payload=signature_payload,
                    embedding_path=embedding_path,
                    tables=quant_tables,
                )
                _append_tables(tables, quant_tables)
                embedding_sources.append(_embedding_source(result.embedding_cache, layer_idx=FULL_MODEL_LAYER_IDX, module_name=FULL_MODEL_MODULE_NAME, embedding_cache_path=result.cache_path, scope=scope))
                runtime_runs.append({**dict(result.runtime_summary), "scope": scope, "shard_key": shard_key, "resumed_from_done": False})
                computed_shard_count += 1

    tables["projection_2d"] = _build_projection_rows(
        cfg,
        fp32_cache=fp32_cache,
        embedding_sources=embedding_sources,
    )
    tables["summary_v2_long"] = build_semantic_geometry_summary_v2_long(tables)
    tables["comparison_summary"] = build_semantic_geometry_comparison_summary(tables["summary_v2_long"])

    metric_paths = _save_all_metric_tables(cfg, tables)
    explanation_paths = _save_explanation_outputs(cfg, tables=tables, embedding_sources=embedding_sources)
    metric_paths.update(explanation_paths)
    figure_paths: dict[str, str] = {}
    plot_error: str | None = None
    try:
        figure_paths = _plot_figures(
            cfg,
            summary_v2_rows=tables["summary_v2_long"],
            comparison_rows=tables["comparison_summary"],
            projection_rows=tables.get("projection_2d", []),
        )
    except Exception as exc:
        if bool(_setting_get(cfg, "STRICT_PLOTS", False)):
            raise
        plot_error = str(exc)

    quant_modes = list(dict.fromkeys(mode for mode, _ in run_specs))
    granularities = list(dict.fromkeys(granularity for _, granularity in run_specs))
    fp32_core_manifest = fp32_cache.get("source_core_eval_manifest", {})
    if not isinstance(fp32_core_manifest, Mapping):
        fp32_core_manifest = {}
    summary = {
        "output_dir": str(output_dir),
        "embedding_dir": str(_embedding_dir(cfg)),
        "metric_shard_dir": str(_metric_shard_dir(cfg)),
        "figures_dir": str(_figures_dir(cfg)),
        "metric_paths": {key: str(value) for key, value in metric_paths.items()},
        "figure_paths": figure_paths,
        "plot_error": plot_error,
        "num_samples": len(fp32_cache["sample_id"]),
        "num_targets": len(target_modules),
        "target_modules": [[int(layer_idx), module_name] for layer_idx, module_name in target_modules],
        "scopes": scopes,
        "run_specs": [[mode, granularity] for mode, granularity in run_specs],
        "quant_modes": quant_modes,
        "observed_quant_modes": list(quant_modes),
        "known_quant_modes": list(KNOWN_QUANT_MODES),
        "missing_quant_modes": sorted(set(KNOWN_QUANT_MODES) - set(quant_modes)),
        "activation_granularities": granularities,
        "act_quant_granularities": granularities,
        "optional_act_quant_granularities": list(OPTIONAL_ACT_QUANT_GRANULARITIES),
        "run_act_per_channel_ablation": any(granularity == "per_channel" for _, granularity in run_specs),
        "activation_granularity_protocol": {
            "main_specs": [[mode, granularity] for mode, granularity in DEFAULT_RUN_SPECS],
            "optional_per_channel_specs": [[mode, granularity] for mode, granularity in OPTIONAL_PER_CHANNEL_RUN_SPECS],
            "configured": list(granularities),
            "default_runtime_supported": list(DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES),
            "per_channel_protocol_supported": True,
            "per_channel_default_runtime_supported": "per_channel" in DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES,
            "custom_quant_runner_enabled": custom_runner is not None,
            "per_channel_requires_custom_quant_runner": True,
            "per_channel_requires_paired_per_tensor": True,
            "w4_only_per_channel_disallowed": True,
            "quant_modes_are_separate_from_activation_granularity": True,
        },
        "scope_protocol": {
            "single_target": "local damage to final embedding causal mapping",
            "full_model": "real quantized-model final embedding endpoint",
            "default_scopes": list(DEFAULT_SCOPES),
        },
        "runtime_runs": runtime_runs,
        "shard_resume": {
            "protocol_version": SEMANTIC_GEOMETRY_EXPERIMENT_PROTOCOL_VERSION,
            "completed_shard_count": completed_shard_count,
            "computed_shard_count": computed_shard_count,
            "force_rebuild_semantic_shards": _force_rebuild(cfg, "FORCE_REBUILD_SEMANTIC_SHARDS"),
            "done_marker_dir": str(done_marker_path(_output_dir(cfg), "example").parent),
            "write_pattern": "metric shard csv + embedding cache + DONE marker, all atomic where possible",
        },
        "embedding_cache_reuse": {
            "force_rebuild_embeddings": _force_rebuild(cfg, "FORCE_REBUILD_EMBEDDINGS"),
            "cache_hit_count": sum(1 for item in runtime_runs if bool(item.get("cache_hit"))),
            "cache_miss_count": sum(1 for item in runtime_runs if not bool(item.get("cache_hit"))),
            "check_cache_sample_order": bool(_setting_get(cfg, "CHECK_CACHE_SAMPLE_ORDER", True)),
        },
        "fp32_embedding_cache": str(fp32_result.cache_path) if fp32_result.cache_path is not None else None,
        "fp32_reference_source": "core_eval.pt",
        "fp32_core_eval_cache": {
            "path": str(fp32_result.cache_path) if fp32_result.cache_path is not None else None,
            "identity": file_identity(fp32_result.cache_path) if fp32_result.cache_path is not None else None,
            "cache_type": fp32_core_manifest.get("cache_type"),
            "storage_format": fp32_core_manifest.get("storage_format"),
            "sample_order_hash": fp32_core_manifest.get("sample_order_hash"),
            "checkpoint_sha256": fp32_core_manifest.get("checkpoint_sha256"),
            "checkpoint_hash": fp32_core_manifest.get("checkpoint_hash"),
            "checkpoint_hash_type": fp32_core_manifest.get("checkpoint_hash_type"),
            "layout_protocol_version": fp32_core_manifest.get("layout_protocol_version"),
            "embedding_dim": fp32_core_manifest.get("embedding_dim"),
            "embedding_pooling": fp32_core_manifest.get("embedding_pooling"),
            "num_samples": fp32_core_manifest.get("num_samples"),
        },
        "cache_module_path": "src/utils/semantic_geometry_cache.py",
        "main_metric_protocol": {
            "main_margin_metric": "embedding_centroid_margin",
            "definition": "min_{c != y} ||z_i - mu_c||_2 - ||z_i - mu_y||_2",
            "class_centroids": "computed from the same fixed eval set full-class embeddings",
            "group_usage": "many/medium/few/tail are only used after sample-level metrics for summary slicing",
            "logits_margin_role": "classification auxiliary / sanity metric only",
            "logits_margin_definition": "logit_y - max_{c != y} logit_c",
            "logits_unavailable_blocks_geometry": False,
        },
        "group_protocol_version": 2,
        "base_buckets": ["many", "medium", "few"],
        "analysis_groups": ["many", "medium", "few", "tail"],
        "canonical_group_field": "bucket",
        "legacy_group_alias": "shot_group",
        "tail_definition": "bucket in {'medium', 'few'}",
        "main_comparison_group": "tail",
        "supplementary_extreme_group": "few",
        "summary_format": "v2_long",
        "comparison_summary_format": "v2_comparison",
        "summary_v2_long_filename": SUMMARY_V2_LONG_FILENAME,
        "comparison_summary_filename": COMPARISON_SUMMARY_FILENAME,
        "projection_metrics_filename": PROJECTION_METRICS_FILENAME,
        "class_center_artifact_filename": CLASS_CENTER_ARTIFACT_FILENAME,
        "sample_margin_vector_filename": SAMPLE_MARGIN_VECTOR_FILENAME,
        "distortion_topk_filename": DISTORTION_TOPK_FILENAME,
        "output_controls": {
            "save_class_centers": _save_class_centers_enabled(cfg),
            "save_sample_margin_vectors": _save_sample_margin_vectors_enabled(cfg),
            "save_semantic_embedding_projection_2d": _projection_enabled(cfg),
            "save_distortion_topk": _save_distortion_topk_enabled(cfg),
        },
        "logits_protocol": {
            "compute_logits_if_available": _compute_logits_if_available(cfg),
            "require_logits_for_geometry": _require_logits(cfg),
            "require_logits_for_classification": _require_logits_for_classification(cfg),
            "definition": "final embedding geometry does not depend on logits; logits margin is auxiliary only",
        },
        "fisher_normalization": "dof_normalized_scatter_ratio" if bool(fp32_reference.get("normalize_fisher_by_dof")) else "raw_scatter_ratio",
        "include_fp32_baseline_metrics": _include_fp32_baseline_metrics(cfg),
        "require_logits": _require_logits(cfg),
    }
    summary_path = _save_summary_json(cfg, summary)
    summary["summary_path"] = str(summary_path)
    return summary

def main(cfg: Any) -> dict[str, Any]:
    seed_everything = _cfg_get(cfg, "seed_everything", None)
    if callable(seed_everything):
        seed_everything(_seed(cfg))
    return run_semantic_geometry_projection_probe(cfg)


def _path_from_summary(summary: Mapping[str, Any], key: str) -> str | None:
    value = summary.get(key)
    if value is None:
        return None
    return str(value)


def _output_file_from_summary(summary: Mapping[str, Any], filename_key: str) -> str | None:
    output_dir = summary.get("output_dir")
    filename = summary.get(filename_key)
    if output_dir is None or filename is None:
        return None
    return str(Path(str(output_dir)) / str(filename))


def _print_saved_outputs(summary: Mapping[str, Any]) -> None:
    output_paths = [
        ("output_dir", _path_from_summary(summary, "output_dir")),
        ("summary_json", _path_from_summary(summary, "summary_path")),
        ("summary_v2_long", _output_file_from_summary(summary, "summary_v2_long_filename")),
        ("comparison_summary", _output_file_from_summary(summary, "comparison_summary_filename")),
        ("embedding_dir", _path_from_summary(summary, "embedding_dir")),
        ("metric_shard_dir", _path_from_summary(summary, "metric_shard_dir")),
        ("figures_dir", _path_from_summary(summary, "figures_dir")),
    ]
    print("[semantic_geometry_projection_probe] saved outputs:")
    for label, path in output_paths:
        if path:
            print(f"  {label}: {path}")


__all__ = [
    "CLASS_METRICS_FILENAME",
    "COMPARISON_SUMMARY_FILENAME",
    "GROUP_METRICS_FILENAME",
    "PROBE_CFG_NAME",
    "SAMPLE_METRICS_FILENAME",
    "PROJECTION_METRICS_FILENAME",
    "CLASS_CENTER_ARTIFACT_FILENAME",
    "SAMPLE_MARGIN_VECTOR_FILENAME",
    "DISTORTION_TOPK_FILENAME",
    "SUMMARY_JSON_FILENAME",
    "SUMMARY_METRICS_FILENAME",
    "SUMMARY_V2_LONG_FILENAME",
    "DEFAULT_ACT_QUANT_GRANULARITIES",
    "OPTIONAL_ACT_QUANT_GRANULARITIES",
    "main",
    "run_semantic_geometry_projection_probe",
]


if __name__ == "__main__":  # pragma: no cover - 手动运行入口。
    import config as project_config

    _print_saved_outputs(main(project_config))
