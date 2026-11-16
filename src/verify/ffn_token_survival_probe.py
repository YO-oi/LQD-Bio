from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from src.analysis.token_survival_summary import (
        COMPARISON_SUMMARY_OUTPUT_FILENAME,
        DETAIL_COMPARISON_SUMMARY_OUTPUT_FILENAME,
        DETAIL_QUANT_MODE_COMPARISON_OUTPUT_FILENAME,
        DETAIL_SUMMARY_OUTPUT_FILENAME,
        DIAGNOSTIC_COMPARISON_SUMMARY_OUTPUT_FILENAME,
        DIAGNOSTIC_QUANT_MODE_COMPARISON_OUTPUT_FILENAME,
        INTERVENTION_LEVEL_COMPARISON_OUTPUT_FILENAME,
        INTERVENTION_LEVEL_SUMMARY_OUTPUT_FILENAME,
        KNOWN_QUANT_MODES,
        QUANT_MODE_COMPARISON_OUTPUT_FILENAME,
        SUMMARY_OUTPUT_FILENAME,
        build_summary_table_from_shards,
    )
    from src.analysis.build_fp32_token_trace_oracle import ORACLE_PROTOCOL_VERSION, ORACLE_STORAGE_FORMAT, load_fp32_token_trace_oracle
    from src.export.export_fp32_core_cache import forward_core_outputs, load_fp32_core_cache
    from src.metrics.token_survival_metrics import SAMPLE_OUTPUT_COLUMNS, compute_token_survival_rows
    from src.plots.token_survival_plotter import plot_all_token_survival_figures
    from src.quant.ffn_quant_injector import get_ffn_module, make_single_ffn_quant_controller, replace_ffn_module_with_quant
    from src.utils.ffn_fp32_reference_cache import _batch_labels, _batch_sample_ids, _metadata_for_samples
    from src.utils.group_utils import group_protocol_metadata, read_canonical_bucket, tail_included
    from src.utils.token_trace_hook import TokenTraceHookManager, TokenTraceRecord, resolve_full_layer_indices
    from src.utils.layout_utils import canonicalize_padding_mask_bt, validate_valid_token_policy
    from src.verify.ffn_quant_mode_ablation import (
        _assert_quant_module_has_fixed_qparams,
        _autocast_context,
        _mode_label,
        _mode_uses_activation,
        _mode_uses_weight,
        _quant_cfg,
        _require_qparams_for_quant_mode,
        _restore_clean_model,
        _target_name,
        calibrate_target_module,
    )
except ImportError:  # pragma: no cover - script execution fallback.
    from ..data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from ..analysis.token_survival_summary import (
        COMPARISON_SUMMARY_OUTPUT_FILENAME,
        DETAIL_COMPARISON_SUMMARY_OUTPUT_FILENAME,
        DETAIL_QUANT_MODE_COMPARISON_OUTPUT_FILENAME,
        DETAIL_SUMMARY_OUTPUT_FILENAME,
        DIAGNOSTIC_COMPARISON_SUMMARY_OUTPUT_FILENAME,
        DIAGNOSTIC_QUANT_MODE_COMPARISON_OUTPUT_FILENAME,
        INTERVENTION_LEVEL_COMPARISON_OUTPUT_FILENAME,
        INTERVENTION_LEVEL_SUMMARY_OUTPUT_FILENAME,
        KNOWN_QUANT_MODES,
        QUANT_MODE_COMPARISON_OUTPUT_FILENAME,
        SUMMARY_OUTPUT_FILENAME,
        build_summary_table_from_shards,
    )
    from ..analysis.build_fp32_token_trace_oracle import ORACLE_PROTOCOL_VERSION, ORACLE_STORAGE_FORMAT, load_fp32_token_trace_oracle
    from ..export.export_fp32_core_cache import forward_core_outputs, load_fp32_core_cache
    from ..metrics.token_survival_metrics import SAMPLE_OUTPUT_COLUMNS, compute_token_survival_rows
    from ..plots.token_survival_plotter import plot_all_token_survival_figures
    from ..quant.ffn_quant_injector import get_ffn_module, make_single_ffn_quant_controller, replace_ffn_module_with_quant
    from ..utils.ffn_fp32_reference_cache import _batch_labels, _batch_sample_ids, _metadata_for_samples
    from ..utils.group_utils import group_protocol_metadata, read_canonical_bucket, tail_included
    from ..utils.token_trace_hook import TokenTraceHookManager, TokenTraceRecord, resolve_full_layer_indices
    from ..utils.layout_utils import canonicalize_padding_mask_bt, validate_valid_token_policy
    from .ffn_quant_mode_ablation import (
        _assert_quant_module_has_fixed_qparams,
        _autocast_context,
        _mode_label,
        _mode_uses_activation,
        _mode_uses_weight,
        _quant_cfg,
        _require_qparams_for_quant_mode,
        _restore_clean_model,
        _target_name,
        calibrate_target_module,
    )


CSV_FILENAME = "token_survival_metrics.csv"
SUMMARY_FILENAME = SUMMARY_OUTPUT_FILENAME
DETAIL_SUMMARY_FILENAME = DETAIL_SUMMARY_OUTPUT_FILENAME
COMPARISON_SUMMARY_FILENAME = COMPARISON_SUMMARY_OUTPUT_FILENAME
DETAIL_COMPARISON_SUMMARY_FILENAME = DETAIL_COMPARISON_SUMMARY_OUTPUT_FILENAME
DIAGNOSTIC_COMPARISON_SUMMARY_FILENAME = DIAGNOSTIC_COMPARISON_SUMMARY_OUTPUT_FILENAME
QUANT_MODE_COMPARISON_FILENAME = QUANT_MODE_COMPARISON_OUTPUT_FILENAME
DETAIL_QUANT_MODE_COMPARISON_FILENAME = DETAIL_QUANT_MODE_COMPARISON_OUTPUT_FILENAME
DIAGNOSTIC_QUANT_MODE_COMPARISON_FILENAME = DIAGNOSTIC_QUANT_MODE_COMPARISON_OUTPUT_FILENAME
INTERVENTION_LEVEL_SUMMARY_FILENAME = INTERVENTION_LEVEL_SUMMARY_OUTPUT_FILENAME
INTERVENTION_LEVEL_COMPARISON_FILENAME = INTERVENTION_LEVEL_COMPARISON_OUTPUT_FILENAME
MANIFEST_FILENAME = "token_survival_manifest.json"
QPARAMS_DIRNAME = "qparams"
QPARAMS_MANIFEST_SUFFIX = ".manifest.json"
SHARDS_DIRNAME = "shards"
DONE_DIRNAME = "done"
TOPK_COMPACT_DIRNAME = "topk_compact"
PARTIAL_SUMMARY_DIRNAME = "partial_summary"
PROTOCOL_VERSION = 2
SHARD_METRIC_SEMANTIC_VERSION = "token_survival_post_quant_dequant_v1"
SHARD_MAIN_OER_COLUMN = "token_oer_retention"
LEGACY_OER_ALIAS_COLUMN = "oer_retention"
SHARD_COMMON_REQUIRED_METRIC_COLUMNS = (
    SHARD_MAIN_OER_COLUMN,
    "mean_norm_retention",
    LEGACY_OER_ALIAS_COLUMN,
)
SHARD_ACTIVATION_REQUIRED_METRIC_COLUMNS = ("high_energy_scalar_exact_zero_rate",)
SHARD_REQUIRED_METRIC_COLUMNS = tuple(
    dict.fromkeys([*SHARD_COMMON_REQUIRED_METRIC_COLUMNS, *SHARD_ACTIVATION_REQUIRED_METRIC_COLUMNS])
)
QUANT_MODES = ("A4-only", "W4A4")
OPTIONAL_CONTROL_QUANT_MODES = ("W4-only",)
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
EXTRA_CSV_COLUMNS = [
    "intervention_layer_idx",
    "intervention_module_name",
    "intervention_target",
    "trace_target_name",
    "trace_scope",
    "trace_layer_idx",
    "topk_unit",
    "topk_source",
    "topk_oracle_path",
    "topk_oracle_cache_type",
    "topk_ratio",
    "topk_k_policy",
    "oracle_topk_k",
    "oracle_valid_token_count",
    "oracle_protocol_version",
    "oracle_valid_unit",
    "oracle_padding_excluded",
    "oracle_topk_unit",
    "oracle_topk_source",
    "oracle_storage_format",
    "oracle_shard_path",
    "oracle_row_key",
    "token_energy_definition",
    "topk_selected_on",
    "tracked_token_ids_fixed",
    "aux_topk_source",
    "metric_role",
    LEGACY_OER_ALIAS_COLUMN,
    "oer_retention_deprecated_alias_of",
    "shard_key",
    "sample_shard_id",
    "protocol_version",
    "experiment_signature",
    "qparams_hash",
    "qparams_path",
    "actual_integer_mapping_available",
    "integer_mapping_is_direct_quantlinear_forward",
    "direct_quantlinear_forward_grid",
    "metric_valid_for_paper",
    "not_applicable_reason",
    "integer_grid_source",
    "integer_mapping_qparams_source",
    "int4_activation_source",
    "topk_token_clip_rate_deprecated_alias_of",
    "topk_token_near_zero_rate_deprecated_alias_of",
    "high_energy_scalar_clip_rate_deprecated_alias_of",
    "high_energy_scalar_near_zero_rate_deprecated_alias_of",
    "topk_clip_rate_deprecated_alias_of",
    "topk_near_zero_rate_deprecated_alias_of",
    "activation_qparams_enabled",
    "weight_qparams_enabled",
    "activation_scale_shape",
    "activation_scale_min",
    "activation_scale_max",
    "activation_scale_mean",
    "activation_scale_std",
    "activation_scale_finite_check",
    "activation_zero_point_shape",
    "activation_zero_point_mean",
    "activation_zero_point_finite_check",
    "activation_qmin",
    "activation_qmax",
    "activation_granularity",
    "activation_qparams_method",
    "weight_scale_shape",
    "weight_scale_min",
    "weight_scale_max",
    "weight_scale_mean",
    "weight_scale_std",
    "weight_scale_finite_check",
    "weight_zero_point_shape",
    "weight_zero_point_mean",
    "weight_zero_point_finite_check",
    "weight_qmin",
    "weight_qmax",
    "weight_granularity",
    "weight_qparams_method",
    "w4a4_minus_a4_topk_survival_rate_delta",
    "w4a4_minus_a4_token_survival_drop_delta",
]
CSV_COLUMNS = list(dict.fromkeys([*SAMPLE_OUTPUT_COLUMNS, *EXTRA_CSV_COLUMNS]))


def _row_with_metric_aliases(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    if SHARD_MAIN_OER_COLUMN in result:
        result.setdefault(LEGACY_OER_ALIAS_COLUMN, result.get(SHARD_MAIN_OER_COLUMN, ""))
        result.setdefault("oer_retention_deprecated_alias_of", SHARD_MAIN_OER_COLUMN)
    return result


def _required_metric_columns_for_quant_mode(quant_mode: str) -> tuple[str, ...]:
    columns = list(SHARD_COMMON_REQUIRED_METRIC_COLUMNS)
    if _mode_uses_activation(quant_mode):
        columns.extend(SHARD_ACTIVATION_REQUIRED_METRIC_COLUMNS)
    return tuple(dict.fromkeys(columns))


@dataclass(frozen=True)
class TraceJob:
    scope: str
    layer_idx: int
    module_name: str
    tensor_position: str
    public_tensor_position: str
    layer_indices: tuple[int, ...]
    target_modules: tuple[tuple[int, str], ...]
    positions: tuple[str, ...]
    shard_key: str
    sample_shard_id: int = 0


@dataclass(frozen=True)
class CoreEvalReference:
    path: Path
    manifest: Mapping[str, Any]
    sample_id_to_index: Mapping[str, int]
    embedding: Tensor
    padding_mask: Tensor
    valid_token_count: Tensor
    sample_order_hash: str | None
    checkpoint_hash: str | None

    def _indices(self, sample_ids: Sequence[str]) -> Tensor:
        missing = [str(item) for item in sample_ids if str(item) not in self.sample_id_to_index]
        if missing:
            raise KeyError(f"core_eval.pt missing sample_id values: {missing[:8]}")
        return torch.tensor([int(self.sample_id_to_index[str(item)]) for item in sample_ids], dtype=torch.long)

    def batch_embedding(self, sample_ids: Sequence[str]) -> Tensor:
        return self.embedding.index_select(0, self._indices(sample_ids)).detach().cpu().float().contiguous()

    def batch_padding_mask(self, sample_ids: Sequence[str]) -> Tensor:
        return self.padding_mask.index_select(0, self._indices(sample_ids)).detach().cpu().bool().contiguous()

    def batch_valid_token_count(self, sample_ids: Sequence[str]) -> Tensor:
        return self.valid_token_count.index_select(0, self._indices(sample_ids)).detach().cpu().long().contiguous()


@dataclass(frozen=True)
class TokenTraceOracleReference:
    path: Path
    manifest: Mapping[str, Any]
    row_index: Mapping[str, int]
    topk_token_indices: Sequence[Any]
    topk_token_energy: Sequence[Any]
    topk_k: Tensor
    valid_token_count: Tensor
    module_name: Sequence[str]
    position: Sequence[str]
    sample_order_hash: str | None
    checkpoint_hash: str | None

    def rows_for_batch(self, *, layer_idx: int, module_name: str, position: str, sample_ids: Sequence[str]) -> dict[str, Any]:
        ids: list[Tensor] = []
        energies: list[Tensor] = []
        k_values: list[int] = []
        valid_counts: list[int] = []
        for sample_id in sample_ids:
            key = f"{sample_id}|L{int(layer_idx)}|{str(module_name)}|{str(position)}"
            if key not in self.row_index:
                raise KeyError(f"token_trace_oracle_eval.pt missing row {key}")
            row_idx = int(self.row_index[key])
            row_ids = torch.as_tensor(self.topk_token_indices[row_idx], dtype=torch.long).detach().cpu().reshape(-1).contiguous()
            row_energy = torch.as_tensor(self.topk_token_energy[row_idx], dtype=torch.float32).detach().cpu().reshape(-1).contiguous()
            row_k = int(self.topk_k[row_idx].item())
            if row_k != int(row_ids.numel()) or row_k != int(row_energy.numel()):
                raise ValueError(f"token oracle topk_k mismatch for {key}: k={row_k}, ids={int(row_ids.numel())}, energy={int(row_energy.numel())}")
            ids.append(row_ids)
            energies.append(row_energy)
            k_values.append(row_k)
            valid_counts.append(int(self.valid_token_count[row_idx].item()))
        return {
            "fp32_topk_token_ids": ids,
            "topk_token_energy": energies,
            "topk_k": k_values,
            "valid_token_count": valid_counts,
            "module_name": str(module_name),
            "position": str(position),
        }

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def _probe_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FFN_TOKEN_SURVIVAL_PROBE", {})
    return value if isinstance(value, Mapping) else {}


def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    probe = _probe_cfg(cfg)
    if key in probe:
        return probe[key]
    return _cfg_get(cfg, key, default)


def _portable_path_text(value: Path | str) -> str:
    text = str(value).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def _portable_path(value: Path | str) -> Path:
    return Path(_portable_path_text(value))


def _looks_like_windows_absolute(text: str) -> bool:
    return len(text) >= 3 and text[0].isalpha() and text[1] == ":" and text[2] == "/"


def _ancestor_named(path: Path, name: str) -> Path | None:
    for item in (path, *path.parents):
        if item.name == name:
            return item
    return None


def _local_data_root_from_cfg(cfg: Any) -> Path | None:
    candidates = [
        _nested_get(_dataset_cfg(cfg), "ROOT", None),
        _cfg_get(cfg, "BASE_DIR", None),
        Path.cwd(),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = _portable_path(candidate)
        if path.name == "local_data":
            return path
        located = _ancestor_named(path, "local_data")
        if located is not None:
            return located
        possible = path / "local_data"
        if possible.exists():
            return possible
    return None


def _resolve_uploaded_artifact_path(anchor_path: Path | str, value: Path | str, *, cfg: Any | None = None) -> Path:
    text = _portable_path_text(value)
    path = Path(text)
    if path.is_absolute():
        return path
    anchor = _portable_path(anchor_path)
    if text.startswith("local_data/") or _looks_like_windows_absolute(text):
        local_root = _local_data_root_from_cfg(cfg) if cfg is not None else _ancestor_named(anchor, "local_data")
        marker = "local_data/"
        if local_root is not None and marker in text:
            return local_root / text.split(marker, 1)[1]
    parent = anchor.parent
    marker = f"/{parent.name}/"
    if _looks_like_windows_absolute(text) and marker in text:
        return parent / text.split(marker, 1)[1]
    return parent / path


def _cfg_path_value(cfg: Any, value: Path | str) -> Path:
    text = _portable_path_text(value)
    path = Path(text)
    if path.is_absolute():
        return path
    if text.startswith("local_data/") or _looks_like_windows_absolute(text):
        local_root = _local_data_root_from_cfg(cfg)
        marker = "local_data/"
        if local_root is not None and marker in text:
            return local_root / text.split(marker, 1)[1]
    return path


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
        return _portable_path(dataset_root).name
    current_result_dir = _cfg_get(cfg, "CURRENT_RESULT_DIR", None)
    if current_result_dir is not None:
        return _portable_path(current_result_dir).name
    raise AttributeError("missing dataset name; configure ACTIVE_DATASET or CURRENT_DATASET['ROOT']")


def _result_dir(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return _cfg_path_value(cfg, configured)
    return _portable_path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "token_survival"


def _reference_cache_dir(cfg: Any) -> Path:
    configured = _setting_get(cfg, "FP32_REFERENCE_CACHE_DIR", _setting_get(cfg, "REFERENCE_CACHE_DIR", None))
    if configured is not None:
        return _cfg_path_value(cfg, configured)
    return _portable_path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "cache" / "fp32_reference"


def _core_eval_cache_path(cfg: Any) -> Path:
    configured = _setting_get(cfg, "CORE_EVAL_PATH", _setting_get(cfg, "FP32_CORE_EVAL_PATH", None))
    return _cfg_path_value(cfg, configured) if configured is not None else _reference_cache_dir(cfg) / "core_eval.pt"


def _token_trace_oracle_eval_path(cfg: Any) -> Path:
    configured = _setting_get(cfg, "TOKEN_TRACE_ORACLE_PATH", _setting_get(cfg, "FP32_TOKEN_TRACE_ORACLE_PATH", None))
    return _cfg_path_value(cfg, configured) if configured is not None else _reference_cache_dir(cfg) / "token_trace_oracle_eval.pt"


def _manifest_checkpoint_hash(manifest: Mapping[str, Any]) -> str | None:
    value = manifest.get("checkpoint_hash") or manifest.get("checkpoint_sha256")
    return None if value is None else str(value)


def _load_core_eval_reference(cfg: Any) -> CoreEvalReference:
    path = _core_eval_cache_path(cfg)
    cache = load_fp32_core_cache(path, materialize=True, include_padding_mask=True, include_logits=False)
    manifest = cache.get("manifest", {})
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"core_eval.pt manifest must be a mapping: {path}")
    sample_ids = [str(item) for item in cache.get("sample_id", [])]
    if not sample_ids:
        raise RuntimeError(f"core_eval.pt has no sample_id order: {path}")
    sample_id_to_index = cache.get("sample_id_to_index")
    if not isinstance(sample_id_to_index, Mapping):
        sample_id_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    embedding = torch.as_tensor(cache["embedding"], dtype=torch.float32).detach().cpu().contiguous()
    padding_mask = torch.as_tensor(cache["padding_mask"]).detach().cpu().bool().contiguous()
    valid_token_count = torch.as_tensor(cache["valid_token_count"], dtype=torch.long).detach().cpu().reshape(-1).contiguous()
    if embedding.ndim != 2 or int(embedding.shape[0]) != len(sample_ids):
        raise RuntimeError(f"core_eval.pt embedding must be [N,D], got {tuple(embedding.shape)} for {len(sample_ids)} samples")
    expected_embedding_dim = int(_setting_get(cfg, "EXPECTED_EMBEDDING_DIM", 768))
    if int(embedding.shape[1]) != expected_embedding_dim:
        raise RuntimeError(f"core_eval.pt embedding dim must be {expected_embedding_dim}, got {int(embedding.shape[1])}")
    if padding_mask.ndim != 2 or int(padding_mask.shape[0]) != len(sample_ids):
        raise RuntimeError(f"core_eval.pt padding_mask must be [N,T], got {tuple(padding_mask.shape)}")
    if int(valid_token_count.shape[0]) != len(sample_ids):
        raise RuntimeError("core_eval.pt valid_token_count length must equal sample count")
    derived_valid = (~padding_mask).sum(dim=1).long()
    if not torch.equal(derived_valid, valid_token_count):
        raise RuntimeError("core_eval.pt valid_token_count does not match padding_mask")
    return CoreEvalReference(
        path=path,
        manifest=manifest,
        sample_id_to_index={str(key): int(value) for key, value in sample_id_to_index.items()},
        embedding=embedding,
        padding_mask=padding_mask,
        valid_token_count=valid_token_count,
        sample_order_hash=None if manifest.get("sample_order_hash") is None else str(manifest.get("sample_order_hash")),
        checkpoint_hash=_manifest_checkpoint_hash(manifest),
    )


def _validate_token_trace_oracle_manifest(
    cfg: Any,
    *,
    oracle_header: Mapping[str, Any],
    manifest: Mapping[str, Any],
    core_reference: CoreEvalReference,
    layer_indices: Sequence[int],
    required_positions: Sequence[str],
    required_input_targets: Sequence[tuple[int, str]],
    oracle_path: Path,
) -> None:
    if manifest.get("cache_type") != "fp32_token_trace_oracle":
        raise RuntimeError(f"Token Survival formal oracle must be fp32_token_trace_oracle, got {manifest.get('cache_type')!r}: {oracle_path}")
    if manifest.get("oracle_protocol_version") != ORACLE_PROTOCOL_VERSION:
        raise RuntimeError(
            "Token Survival formal probe requires token_trace_oracle.v2; "
            f"got {manifest.get('oracle_protocol_version')!r}. Rebuild token_trace_oracle_eval.pt with force_rebuild=True."
        )
    if manifest.get("storage_format") != ORACLE_STORAGE_FORMAT:
        raise RuntimeError(
            "Token Survival formal probe requires multi_position_sharded_v2 oracle storage; "
            f"got {manifest.get('storage_format')!r}. Rebuild token_trace_oracle_eval.pt with force_rebuild=True."
        )
    if manifest.get("topk_unit") != "token":
        raise RuntimeError(f"Token Survival formal oracle requires topk_unit='token', got {manifest.get('topk_unit')!r}: {oracle_path}")
    if manifest.get("topk_source") != "fp32_token_energy":
        raise RuntimeError(f"Token Survival formal oracle requires topk_source='fp32_token_energy', got {manifest.get('topk_source')!r}: {oracle_path}")
    available_positions = [str(item) for item in manifest.get("available_positions", [])]
    missing_positions = sorted(set(str(item) for item in required_positions) - set(available_positions))
    if missing_positions:
        raise RuntimeError(f"token_trace_oracle_eval.pt missing requested positions {missing_positions}; rebuild v2 oracle: {oracle_path}")
    declared_shards = oracle_header.get("shards", manifest.get("shards", {}))
    if not isinstance(declared_shards, Mapping):
        raise RuntimeError(f"token_trace_oracle_eval.pt v2 manifest requires shards mapping: {oracle_path}")
    for position in available_positions:
        if position not in declared_shards:
            raise RuntimeError(f"token_trace_oracle_eval.pt claims position {position!r} but has no shard path")
        shard_path = _resolve_uploaded_artifact_path(oracle_path, declared_shards[position], cfg=cfg)
        if not shard_path.exists() or not shard_path.is_file():
            raise RuntimeError(f"token_trace_oracle_eval.pt claims position {position!r} but shard is missing: {shard_path}")
    expected_ratio = float(_token_topk_ratio(cfg))
    actual_ratio = float(manifest.get("topk_ratio", float("nan")))
    if not math.isfinite(actual_ratio) or abs(actual_ratio - expected_ratio) > 1e-12:
        raise RuntimeError(f"Token oracle topk_ratio mismatch: expected {expected_ratio}, got {actual_ratio}: {oracle_path}")
    required_pairs = {
        "topk_k_policy": "ceil(topk_ratio * valid_unit_count)",
        "valid_unit": "valid_token",
        "valid_unit_count_field": "valid_token_count",
    }
    for key, expected in required_pairs.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Token oracle manifest[{key}] expected {expected!r}, got {manifest.get(key)!r}: {oracle_path}")
    if manifest.get("padding_excluded") is not True:
        raise RuntimeError(f"Token oracle requires padding_excluded=true: {oracle_path}")
    if manifest.get("layout_protocol_version") is None:
        raise RuntimeError(f"Token oracle missing layout_protocol_version: {oracle_path}")
    if manifest.get("checkpoint_hash") is None or manifest.get("checkpoint_sha256") is None:
        raise RuntimeError(f"Token oracle formal cache must record checkpoint_hash and checkpoint_sha256: {oracle_path}")
    if core_reference.sample_order_hash is not None and manifest.get("sample_order_hash") != core_reference.sample_order_hash:
        raise RuntimeError("token_trace_oracle_eval.pt sample_order_hash does not match core_eval.pt")
    token_hash = _manifest_checkpoint_hash(manifest)
    if core_reference.checkpoint_hash is not None and token_hash is not None and token_hash != core_reference.checkpoint_hash:
        raise RuntimeError("token_trace_oracle_eval.pt checkpoint_hash does not match core_eval.pt")
    available_layers = {int(item) for item in manifest.get("layer_indices", [])}
    missing_layers = sorted(set(int(item) for item in layer_indices) - available_layers)
    if "block_output" in set(str(item) for item in required_positions) and missing_layers:
        raise RuntimeError(f"token_trace_oracle_eval.pt missing block_output layers: {missing_layers}")
    if "input_activation" in set(str(item) for item in required_positions):
        coverage = manifest.get("position_coverage", {})
        input_coverage = coverage.get("input_activation", {}) if isinstance(coverage, Mapping) else {}
        declared_targets = input_coverage.get("target_modules", []) if isinstance(input_coverage, Mapping) else []
        declared = {
            (int(item.get("layer_idx")), str(item.get("module_name")))
            for item in declared_targets
            if isinstance(item, Mapping) and item.get("layer_idx") is not None and item.get("module_name") is not None
        }
        missing_targets = sorted(set((int(layer), str(module)) for layer, module in required_input_targets) - declared)
        if missing_targets:
            raise RuntimeError(f"token_trace_oracle_eval.pt missing input_activation targets: {missing_targets[:8]}")


def _load_token_trace_oracle_reference(
    cfg: Any,
    *,
    core_reference: CoreEvalReference,
    layer_indices: Sequence[int],
    required_positions: Sequence[str],
    required_input_targets: Sequence[tuple[int, str]],
) -> TokenTraceOracleReference:
    path = _token_trace_oracle_eval_path(cfg)
    oracle_header = load_fp32_token_trace_oracle(path, materialize=False)
    manifest = oracle_header.get("manifest", {})
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"token_trace_oracle_eval.pt manifest must be a mapping: {path}")
    _validate_token_trace_oracle_manifest(
        cfg,
        oracle_header=oracle_header,
        manifest=manifest,
        core_reference=core_reference,
        layer_indices=layer_indices,
        required_positions=required_positions,
        required_input_targets=required_input_targets,
        oracle_path=path,
    )
    oracle = load_fp32_token_trace_oracle(path, materialize=True, positions=required_positions)
    required_fields = ("row_index", "topk_token_indices", "topk_token_energy", "topk_k", "valid_token_count", "module_name", "position")
    for field in required_fields:
        if field not in oracle:
            raise RuntimeError(f"token_trace_oracle_eval.pt missing materialized field {field}: {path}")
    return TokenTraceOracleReference(
        path=path,
        manifest=manifest,
        row_index={str(key): int(value) for key, value in dict(oracle["row_index"]).items()},
        topk_token_indices=list(oracle["topk_token_indices"]),
        topk_token_energy=list(oracle["topk_token_energy"]),
        topk_k=torch.as_tensor(oracle["topk_k"], dtype=torch.long).detach().cpu().reshape(-1).contiguous(),
        valid_token_count=torch.as_tensor(oracle["valid_token_count"], dtype=torch.long).detach().cpu().reshape(-1).contiguous(),
        module_name=[str(item) for item in oracle["module_name"]],
        position=[str(item) for item in oracle["position"]],
        sample_order_hash=None if manifest.get("sample_order_hash") is None else str(manifest.get("sample_order_hash")),
        checkpoint_hash=_manifest_checkpoint_hash(manifest),
    )


def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", _cfg_get(cfg, "DEVICE", None))
    if configured is None:
        raise AttributeError("ffn_token_survival_probe requires config.py DEVICE")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py DEVICE={configured!r}, but CUDA is not available")
    return device


def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


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
    key = f"{name.upper()}_CSV"
    value = _setting_get(cfg, key, None)
    if value is not None:
        return _cfg_path_value(cfg, value)
    dataset_value = _nested_get(_dataset_cfg(cfg), key, None)
    if dataset_value is not None:
        return _cfg_path_value(cfg, dataset_value)
    raise AttributeError(f"missing {key}; configure it in cfg.CURRENT_DATASET or cfg.FFN_TOKEN_SURVIVAL_PROBE")


def _batch_size(cfg: Any, name: str) -> int:
    key = f"{name.upper()}_BATCH_SIZE"
    return int(_setting_get(cfg, key, _setting_get(cfg, "BATCH_SIZE", 8)))


def _num_workers(cfg: Any) -> int:
    return int(_setting_get(cfg, "NUM_WORKERS", 0))


def _persistent_workers(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "PERSISTENT_WORKERS", False)) and _num_workers(cfg) > 0


def _prefetch_factor(cfg: Any) -> int | None:
    if _num_workers(cfg) <= 0:
        return None
    configured = _setting_get(cfg, "PREFETCH_FACTOR", None)
    return None if configured is None else int(configured)


def _pin_memory(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "PIN_MEMORY", _device(cfg).type == "cuda"))


def _build_audio_loader(cfg: Any, *, name: str, mode: str) -> DataLoader:
    override = _cfg_get(cfg, f"build_{name}_loader", None)
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
        require_label_id=True,
        include_audio_stats=False,
    )
    if str(mode).lower() == "train":
        raise ValueError("Token Survival must not use train random crop; use deterministic eval/calibration mode")
    loader_kwargs: dict[str, Any] = {
        "batch_size": _batch_size(cfg, name),
        "shuffle": False,
        "num_workers": _num_workers(cfg),
        "pin_memory": _pin_memory(cfg),
        "collate_fn": partial(audio_input_collate_fn, include_audio_stats=False),
    }
    if _num_workers(cfg) > 0:
        loader_kwargs["persistent_workers"] = _persistent_workers(cfg)
        prefetch_factor = _prefetch_factor(cfg)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
    return moved


def _quant_modes(cfg: Any) -> tuple[str, ...]:
    configured = _setting_get(cfg, "QUANT_MODES", None)
    if configured is None:
        modes = [_mode_label(str(item)) for item in QUANT_MODES]
    else:
        modes = [_mode_label(str(item)) for item in configured]

    optional_configured = _setting_get(cfg, "OPTIONAL_CONTROL_QUANT_MODES", ())
    if optional_configured is None:
        optional_configured = ()
    if bool(_setting_get(cfg, "INCLUDE_W4_ONLY_CONTROL", False)):
        optional_configured = tuple(optional_configured) + OPTIONAL_CONTROL_QUANT_MODES
    for item in optional_configured:
        label = _mode_label(str(item))
        if label not in modes:
            modes.append(label)

    required = set(QUANT_MODES)
    missing = sorted(required - set(modes))
    if missing:
        raise ValueError(f"Token Survival formal experiment requires A4-only and W4A4; missing {missing}, got {tuple(modes)}")
    allowed = set(QUANT_MODES) | set(OPTIONAL_CONTROL_QUANT_MODES)
    unsupported = sorted(set(modes) - allowed)
    if unsupported:
        raise ValueError(f"Token Survival unsupported quant modes: {unsupported}; allowed={sorted(allowed)}")
    return tuple(dict.fromkeys(modes))


def _configured_num_transformer_layers(cfg: Any) -> int | None:
    model_cfg = _cfg_get(cfg, "MODEL", {})
    candidates = (
        _setting_get(cfg, "NUM_TRANSFORMER_LAYERS", None),
        _setting_get(cfg, "MODEL_NUM_TRANSFORMER_LAYERS", None),
        _nested_get(model_cfg, "NUM_TRANSFORMER_LAYERS", None),
        _nested_get(model_cfg, "num_transformer_layers", None),
        _nested_get(model_cfg, "NUM_LAYERS", None),
        _nested_get(model_cfg, "num_layers", None),
    )
    for value in candidates:
        if value is not None:
            return int(value)
    return None


def _resolve_probe_layer_indices(cfg: Any, model: nn.Module) -> tuple[int, ...]:
    layer_indices = resolve_full_layer_indices(model)
    configured = _configured_num_transformer_layers(cfg)
    if configured is not None and int(configured) != len(layer_indices):
        raise ValueError(
            f"configured num_transformer_layers={configured} does not match model encoder.layers={len(layer_indices)}"
        )
    return layer_indices


def _slug(value: Any) -> str:
    text = str(value).strip().replace(" ", "_")
    allowed = []
    for char in text:
        allowed.append(char if char.isalnum() or char in {"_", "-", "."} else "_")
    return "".join(allowed).strip("_") or "x"


def _normalise_target_module_list(configured: Sequence[Any]) -> tuple[tuple[int, str], ...]:
    result: list[tuple[int, str]] = []
    for item in configured:
        if isinstance(item, Mapping):
            layer_idx = item["layer_idx"]
            module_name = item["module_name"]
        else:
            layer_idx, module_name = item
        module_name = str(module_name)
        if module_name not in {"fc1", "fc2"}:
            raise ValueError(f"module_name must be fc1/fc2, got {module_name!r}")
        result.append((int(layer_idx), module_name))
    if len(set(result)) != len(result):
        raise ValueError(f"target module list contains duplicate entries: {result}")
    return tuple(result)


def _input_activation_trace_targets(
    cfg: Any,
    *,
    available_targets: Sequence[tuple[int, str]],
    intervention_layer_idx: int,
    intervention_module_name: str,
) -> tuple[tuple[int, str], ...]:
    configured = _setting_get(cfg, "INPUT_ACTIVATION_TARGET_MODULES", None)
    if configured is None:
        configured = _setting_get(cfg, "SENSITIVE_TARGET_MODULES", None)
    if configured is None:
        return ((int(intervention_layer_idx), str(intervention_module_name)),)
    selected = _normalise_target_module_list(configured)
    available = set((int(layer_idx), str(module_name)) for layer_idx, module_name in available_targets)
    missing = [item for item in selected if item not in available]
    if missing:
        raise ValueError(f"INPUT_ACTIVATION_TARGET_MODULES contains targets not present in probe target modules: {missing}")
    return selected


def _trace_jobs_for_intervention(
    cfg: Any,
    *,
    layer_indices: Sequence[int],
    available_targets: Sequence[tuple[int, str]],
    intervention_layer_idx: int,
    intervention_module_name: str,
) -> tuple[TraceJob, ...]:
    jobs: list[TraceJob] = []
    if bool(_setting_get(cfg, "RUN_BLOCK_OUTPUT_TRACE", True)):
        for layer_idx in layer_indices:
            layer_idx = int(layer_idx)
            jobs.append(
                TraceJob(
                    scope="block_output_flow",
                    layer_idx=layer_idx,
                    module_name="block",
                    tensor_position="block_output",
                    public_tensor_position="block_output",
                    layer_indices=(layer_idx,),
                    target_modules=(),
                    positions=("block_output",),
                    shard_key=(
                        f"eval__{{quant_mode}}__intervene_L{int(intervention_layer_idx):02d}_{intervention_module_name}__"
                        f"trace_L{layer_idx:02d}_block_output__shard0000"
                    ),
                )
            )
    if bool(_setting_get(cfg, "RUN_INPUT_ACTIVATION_TRACE", True)):
        selected = _input_activation_trace_targets(
            cfg,
            available_targets=available_targets,
            intervention_layer_idx=int(intervention_layer_idx),
            intervention_module_name=str(intervention_module_name),
        )
        for layer_idx, module_name in selected:
            jobs.append(
                TraceJob(
                    scope="input_activation_mechanism",
                    layer_idx=int(layer_idx),
                    module_name=str(module_name),
                    tensor_position="input",
                    public_tensor_position="input_activation",
                    layer_indices=(int(layer_idx),),
                    target_modules=((int(layer_idx), str(module_name)),),
                    positions=("input",),
                    shard_key=(
                        f"eval__{{quant_mode}}__intervene_L{int(intervention_layer_idx):02d}_{intervention_module_name}__"
                        f"trace_L{int(layer_idx):02d}_{module_name}_input_activation__shard0000"
                    ),
                )
            )
    if bool(_setting_get(cfg, "RUN_OUTPUT_PROJECTION_TRACE", False)):
        selected = _input_activation_trace_targets(
            cfg,
            available_targets=available_targets,
            intervention_layer_idx=int(intervention_layer_idx),
            intervention_module_name=str(intervention_module_name),
        )
        for layer_idx, module_name in selected:
            jobs.append(
                TraceJob(
                    scope="linear_output_projection_detail",
                    layer_idx=int(layer_idx),
                    module_name=str(module_name),
                    tensor_position="output",
                    public_tensor_position="linear_output_projection",
                    layer_indices=(int(layer_idx),),
                    target_modules=((int(layer_idx), str(module_name)),),
                    positions=("output",),
                    shard_key=(
                        f"eval__{{quant_mode}}__intervene_L{int(intervention_layer_idx):02d}_{intervention_module_name}__"
                        f"trace_L{int(layer_idx):02d}_{module_name}_linear_output_projection__shard0000"
                    ),
                )
            )
    if not jobs:
        raise ValueError("Token Survival trace plan is empty; enable block_output or input_activation trace")
    return tuple(jobs)


PendingTraceJob = tuple[TraceJob, str, str, dict[str, Path]]


def _block_trace_group_size(cfg: Any) -> int:
    value = int(_setting_get(cfg, "BLOCK_TRACE_GROUP_SIZE", 1))
    return max(1, value)


def _is_groupable_block_trace_job(trace_job: TraceJob) -> bool:
    return (
        str(trace_job.public_tensor_position) == "block_output"
        and str(trace_job.tensor_position) == "block_output"
        and tuple(trace_job.positions) == ("block_output",)
        and len(trace_job.target_modules) == 0
        and len(trace_job.layer_indices) == 1
    )


def _pending_trace_groups(cfg: Any, pending: Sequence[PendingTraceJob]) -> tuple[tuple[PendingTraceJob, ...], ...]:
    group_size = _block_trace_group_size(cfg)
    groups: list[tuple[PendingTraceJob, ...]] = []
    block_buffer: list[PendingTraceJob] = []

    def flush_block_buffer() -> None:
        nonlocal block_buffer
        if not block_buffer:
            return
        for start in range(0, len(block_buffer), group_size):
            groups.append(tuple(block_buffer[start : start + group_size]))
        block_buffer = []

    for item in pending:
        trace_job = item[0]
        if group_size > 1 and _is_groupable_block_trace_job(trace_job):
            block_buffer.append(item)
            continue
        flush_block_buffer()
        groups.append((item,))
    flush_block_buffer()
    return tuple(groups)


def _merged_trace_job_for_group(entries: Sequence[PendingTraceJob]) -> TraceJob:
    if len(entries) == 1:
        return entries[0][0]
    jobs = [entry[0] for entry in entries]
    if not all(_is_groupable_block_trace_job(job) for job in jobs):
        raise ValueError("Only block_output trace jobs can be merged")
    layer_indices = tuple(int(job.layer_idx) for job in jobs)
    return TraceJob(
        scope="block_output_flow",
        layer_idx=int(layer_indices[0]),
        module_name="block",
        tensor_position="block_output",
        public_tensor_position="block_output",
        layer_indices=layer_indices,
        target_modules=(),
        positions=("block_output",),
        shard_key="__merged_block_output_group__",
    )


def _records_for_trace_job(records: Sequence[TokenTraceRecord], trace_job: TraceJob) -> list[TokenTraceRecord]:
    expected = (
        int(trace_job.layer_idx),
        str(trace_job.module_name),
        str(trace_job.tensor_position),
    )
    selected = [record for record in records if _trace_key(record) == expected]
    if len(selected) != 1:
        observed = sorted(_trace_key(record) for record in records)
        raise RuntimeError(f"expected exactly one token trace record for {expected}, got {len(selected)} from {observed}")
    return selected


def _target_name_short(layer_idx: int, module_name: str) -> str:
    return f"L{int(layer_idx)}_{module_name}"


def _all_ffn_target_modules(model: nn.Module) -> list[tuple[int, str]]:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise AttributeError("model.encoder.layers is required to iterate BEATs Transformer FFN modules")
    targets: list[tuple[int, str]] = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        for module_name in ("fc1", "fc2"):
            if not hasattr(layer, module_name):
                raise AttributeError(f"encoder.layers.{layer_idx}.{module_name} does not exist")
            targets.append((int(layer_idx), module_name))
    if not targets:
        raise ValueError("no FFN fc1/fc2 target modules were found")
    return targets


def _target_modules_for_probe(cfg: Any, model: nn.Module) -> list[tuple[int, str]]:
    configured = _setting_get(cfg, "TARGET_MODULES", None)
    if configured is None:
        return _all_ffn_target_modules(model)
    result: list[tuple[int, str]] = []
    for item in configured:
        layer_idx, module_name = item
        module_name = str(module_name)
        if module_name not in {"fc1", "fc2"}:
            raise ValueError(f"module_name must be fc1/fc2, got {module_name!r}")
        get_ffn_module(model, int(layer_idx), module_name)
        result.append((int(layer_idx), module_name))
    if not result:
        raise ValueError("TARGET_MODULES cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"TARGET_MODULES contains duplicate entries: {result}")
    return result


def _trace_key(record: TokenTraceRecord) -> tuple[int, str, str]:
    return (int(record.layer_idx), str(record.module_name), str(record.tensor_position))


def _record_map(records: Sequence[TokenTraceRecord]) -> dict[tuple[int, str, str], TokenTraceRecord]:
    result: dict[tuple[int, str, str], TokenTraceRecord] = {}
    for record in records:
        key = _trace_key(record)
        if key in result:
            raise RuntimeError(f"duplicate token trace record: {key}")
        result[key] = record
    return result


def _normalise_feature_padding_mask(
    features: Tensor,
    feature_padding_mask: Tensor | None,
    cfg: Any,
    *,
    batch_size: int,
) -> Tensor:
    if features.ndim != 3:
        raise ValueError(f"Token Survival requires model token features [B,T,D], got shape={tuple(features.shape)}")
    if int(features.shape[0]) != int(batch_size):
        raise ValueError(f"model output first dimension must equal batch_size={batch_size}, got shape={tuple(features.shape)}")
    if feature_padding_mask is None:
        if _valid_token_policy(cfg) == "explicit_no_padding":
            return torch.zeros((batch_size, int(features.shape[1])), dtype=torch.bool)
        raise RuntimeError("model forward did not return a token-level padding_mask; waveform-level mask cannot substitute for Token Survival")
    expected = (batch_size, int(features.shape[1]))
    return canonicalize_padding_mask_bt(
        feature_padding_mask,
        batch_size=int(expected[0]),
        token_count=int(expected[1]),
        context="token_survival.feature_padding_mask",
    )


def _group_metadata_for_batch(cfg: Any, sample_ids: Sequence[str], labels: Tensor, batch: Mapping[str, Any]) -> dict[str, list[Any]]:
    del batch
    metadata = _metadata_for_samples(sample_ids, labels, cfg)
    bucket_values = metadata.get("bucket") if isinstance(metadata, Mapping) else None
    shot_values = metadata.get("shot_group") if isinstance(metadata, Mapping) else None
    bucket_result = read_canonical_bucket(
        bucket=bucket_values,
        shot_group=shot_values,
        context="token_survival batch metadata",
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in bucket_result.bucket]
    return {
        "bucket": buckets,
        "shot_group": list(buckets),
        "tail_included": [bool(item) for item in tail_included(buckets)],
    }


def _forward_token_mask_only(
    model: nn.Module,
    batch: Mapping[str, Any],
    cfg: Any,
    *,
    sample_ids: Sequence[str],
) -> Tensor:
    device = _device(cfg)
    with torch.no_grad(), _autocast_context(cfg, device):
        _logits, _embedding, token_padding_mask, _valid_token_count = forward_core_outputs(model, batch, cfg)
    return token_padding_mask.detach().cpu().bool().contiguous()


def _forward_with_token_trace(
    model: nn.Module,
    batch: Mapping[str, Any],
    cfg: Any,
    *,
    trace_job: TraceJob,
    sample_ids: Sequence[str],
    token_padding_mask: Tensor,
) -> tuple[Tensor, Tensor, list[TokenTraceRecord]]:
    device = _device(cfg)
    manager = TokenTraceHookManager(
        model,
        layer_indices=trace_job.layer_indices,
        target_modules=trace_job.target_modules,
        positions=trace_job.positions,
        require_padding_mask=True,
    )
    manager.register_hooks()
    try:
        manager.set_batch_context(sample_id=list(sample_ids), padding_mask=token_padding_mask)
        with torch.no_grad(), _autocast_context(cfg, device):
            _logits, embedding, returned_mask, _valid_token_count = forward_core_outputs(model, batch, cfg)
            embedding = embedding.detach().cpu().float().contiguous()
        returned_mask = returned_mask.detach().cpu().bool().contiguous()
        if tuple(returned_mask.shape) != tuple(token_padding_mask.shape) or not torch.equal(returned_mask, token_padding_mask.cpu()):
            raise RuntimeError("token padding_mask changed during trace forward within the same batch")
        records = manager.pop_cache()
        if not records:
            raise RuntimeError("TokenTraceHookManager did not capture any token trace")
        return embedding, returned_mask, records
    finally:
        manager.clear_hooks()
        manager.clear_cache()
        manager.clear_batch_context()


def _activation_mapping_from_quant_input(quant_input: Tensor, activation_qparams: Mapping[str, Any]) -> dict[str, Any]:
    scale = torch.as_tensor(activation_qparams["scale"], dtype=torch.float32).detach().cpu()
    zero_point = torch.as_tensor(activation_qparams.get("zero_point", 0.0), dtype=torch.float32).detach().cpu()
    qmin = int(activation_qparams["qmin"])
    qmax = int(activation_qparams["qmax"])
    if int(scale.numel()) != 1:
        raise ValueError(f"Token Survival currently accepts only per-tensor activation scale, got shape={tuple(scale.shape)}")
    if not torch.isfinite(scale).all().item() or float(scale.reshape(-1)[0].item()) <= 0.0:
        raise ValueError("activation scale must be positive and finite")
    if not torch.isfinite(zero_point).all().item():
        raise ValueError("activation zero_point must be finite")
    x = quant_input.detach().cpu().float().contiguous()
    q_preclip = torch.round(x / scale.reshape(1, 1, 1) + zero_point.reshape(1, 1, 1))
    q_int = q_preclip.clamp(qmin, qmax)
    return {
        "x": x,
        "q_preclip": q_preclip,
        "q_int": q_int,
        "exact_zero_mask": q_int == zero_point.reshape(1, 1, 1),
        "near_zero_pm1_mask": (q_int - zero_point.reshape(1, 1, 1)).abs() <= 1.0,
        "boundary_hit_mask": (q_int <= qmin) | (q_int >= qmax),
        "strict_overflow_mask": (q_preclip < qmin) | (q_preclip > qmax),
        "near_zero_mask": q_int == zero_point.reshape(1, 1, 1),
        "clip_mask": (q_preclip < qmin) | (q_preclip > qmax),
        "scale": scale,
        "zero_point": zero_point,
        "qmin": qmin,
        "qmax": qmax,
        "mapping_source": "calibration_qparams_reconstructed_from_quant_input",
        "integer_grid_source": "reconstructed_from_activation_and_qparams",
        "is_direct_quantlinear_forward_grid": False,
        "qparams_source": "calibration_qparams",
        "near_zero_mask_deprecated_alias_of": "exact_zero_mask",
        "clip_mask_deprecated_alias_of": "strict_overflow_mask",
    }


def _tensor_stats(prefix: str, value: Any | None) -> dict[str, Any]:
    if value is None:
        return {
            f"{prefix}_shape": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
            f"{prefix}_mean": "",
            f"{prefix}_std": "",
            f"{prefix}_finite_check": False,
        }
    raw = torch.as_tensor(value).detach().cpu()
    tensor = raw.float().reshape(-1)
    if int(tensor.numel()) <= 0:
        return {
            f"{prefix}_shape": tuple(int(item) for item in raw.shape),
            f"{prefix}_min": "",
            f"{prefix}_max": "",
            f"{prefix}_mean": "",
            f"{prefix}_std": "",
            f"{prefix}_finite_check": False,
        }
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    if int(finite_values.numel()) <= 0:
        min_value = max_value = mean_value = std_value = float("nan")
    else:
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        mean_value = float(finite_values.mean().item())
        std_value = float(finite_values.std(unbiased=False).item())
    return {
        f"{prefix}_shape": tuple(int(item) for item in raw.shape),
        f"{prefix}_min": min_value,
        f"{prefix}_max": max_value,
        f"{prefix}_mean": mean_value,
        f"{prefix}_std": std_value,
        f"{prefix}_finite_check": bool(finite.all().item()),
    }


def _qparams_summary(qparams: Mapping[str, Any], quant_mode: str) -> dict[str, Any]:
    activation_qparams = qparams.get("activation") if _mode_uses_activation(quant_mode) else None
    weight_qparams = qparams.get("weight") if _mode_uses_weight(quant_mode) else None
    row: dict[str, Any] = {
        "activation_qparams_enabled": bool(activation_qparams is not None),
        "weight_qparams_enabled": bool(weight_qparams is not None),
        "activation_qmin": "" if activation_qparams is None else activation_qparams.get("qmin", ""),
        "activation_qmax": "" if activation_qparams is None else activation_qparams.get("qmax", ""),
        "activation_granularity": "" if activation_qparams is None else activation_qparams.get("granularity", ""),
        "activation_qparams_method": "" if activation_qparams is None else activation_qparams.get("method", ""),
        "activation_zero_point_shape": "",
        "activation_zero_point_mean": "",
        "activation_zero_point_finite_check": False,
        "weight_qmin": "" if weight_qparams is None else weight_qparams.get("qmin", ""),
        "weight_qmax": "" if weight_qparams is None else weight_qparams.get("qmax", ""),
        "weight_granularity": "" if weight_qparams is None else weight_qparams.get("granularity", ""),
        "weight_qparams_method": "" if weight_qparams is None else weight_qparams.get("method", ""),
        "weight_zero_point_shape": "",
        "weight_zero_point_mean": "",
        "weight_zero_point_finite_check": False,
    }
    row.update(_tensor_stats("activation_scale", None if activation_qparams is None else activation_qparams.get("scale")))
    row.update(_tensor_stats("weight_scale", None if weight_qparams is None else weight_qparams.get("scale")))
    for prefix, source in (("activation_zero_point", activation_qparams), ("weight_zero_point", weight_qparams)):
        stats = _tensor_stats(prefix, None if source is None else source.get("zero_point"))
        row[f"{prefix}_shape"] = stats[f"{prefix}_shape"]
        row[f"{prefix}_mean"] = stats[f"{prefix}_mean"]
        row[f"{prefix}_finite_check"] = stats[f"{prefix}_finite_check"]
    return row


def _valid_token_policy(cfg: Any) -> str:
    explicit = _setting_get(cfg, "VALID_TOKEN_POLICY", None)
    if explicit is not None:
        return validate_valid_token_policy(explicit)
    if bool(_setting_get(cfg, "ALLOW_UNMASKED_TOKEN_TRACE", False)):
        return "explicit_no_padding"
    return "require_mask"


def _capture_quantlinear_integer_grid(cfg: Any) -> bool:
    _ = cfg
    return True


def _allow_reconstructed_grid_diagnostic(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "ALLOW_RECONSTRUCTED_GRID_DIAGNOSTIC", False))


def _integer_grid_capture_options(cfg: Any) -> dict[str, Any]:
    return {
        "scope": str(_setting_get(cfg, "INTEGER_GRID_CAPTURE_SCOPE", "selected_targets_only")),
        "save_device": str(_setting_get(cfg, "INTEGER_GRID_SAVE_DEVICE", "cpu")),
        "detach": bool(_setting_get(cfg, "INTEGER_GRID_DETACH", True)),
        "canonical_layout": str(_setting_get(cfg, "INTEGER_GRID_CANONICAL_LAYOUT", "BTC")),
    }


def _set_quantlinear_integer_grid_capture(module: nn.Module, cfg: Any, *, enabled: bool) -> None:
    if not hasattr(module, "set_integer_grid_capture"):
        if enabled:
            if _allow_reconstructed_grid_diagnostic(cfg):
                return
            raise TypeError("target quant module does not support QuantLinear integer-grid diagnostics")
        return
    module.set_integer_grid_capture(bool(enabled), **_integer_grid_capture_options(cfg))


def _get_quantlinear_integer_grid(module: nn.Module) -> dict[str, Any] | None:
    if not hasattr(module, "get_integer_grid_diagnostics"):
        return None
    grid = module.get_integer_grid_diagnostics()
    return None if grid is None else dict(grid)


def _get_quantlinear_quant_activation_for_oer(module: nn.Module | None) -> Tensor | None:
    if module is None or not hasattr(module, "get_oer_activation_tensors"):
        return None
    tensors = module.get_oer_activation_tensors()
    if not isinstance(tensors, tuple) or len(tensors) != 2:
        raise TypeError("QuantLinear.get_oer_activation_tensors() must return (fp32_activation, quant_activation)")
    quant_activation = tensors[1]
    if quant_activation is None:
        return None
    return quant_activation.detach().cpu().float().contiguous()


def _dequant_activation_from_integer_mapping(
    integer_mapping: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int, int],
    context: str,
) -> Tensor:
    missing = [key for key in ("q_int", "scale") if key not in integer_mapping]
    if missing:
        raise KeyError(f"{context}: integer_mapping is missing fields needed for post-quant dequant activation: {missing}")
    q_int = torch.as_tensor(integer_mapping["q_int"]).detach().cpu().float()
    if tuple(q_int.shape) != tuple(expected_shape):
        raise RuntimeError(f"{context}: q_int shape {tuple(q_int.shape)} does not match expected activation shape {expected_shape}")
    scale = torch.as_tensor(integer_mapping["scale"], dtype=torch.float32, device=q_int.device).detach()
    zero_point = torch.as_tensor(integer_mapping.get("zero_point", 0.0), dtype=torch.float32, device=q_int.device).detach()
    try:
        dequant = (q_int - zero_point) * scale
    except RuntimeError as exc:
        raise RuntimeError(f"{context}: q_int/scale/zero_point cannot broadcast for post-quant dequant activation") from exc
    if tuple(dequant.shape) != tuple(expected_shape):
        raise RuntimeError(f"{context}: dequantized activation shape {tuple(dequant.shape)} does not match expected shape {expected_shape}")
    return dequant.detach().cpu().float().contiguous()


def _usable_quant_activation_cache(cache: Tensor | None, *, expected_shape: tuple[int, int, int]) -> Tensor | None:
    if cache is None:
        return None
    tensor = cache.detach().cpu().float().contiguous()
    if tuple(tensor.shape) != tuple(expected_shape):
        return None
    return tensor


def _post_quant_metric_tensor_for_input_activation(
    *,
    trace_job: TraceJob,
    has_mapping: bool,
    fp32_record: TokenTraceRecord,
    integer_mapping: Mapping[str, Any] | None,
    direct_quant_activation_for_oer: Tensor | None,
) -> Tensor | None:
    if str(trace_job.public_tensor_position) != "input_activation":
        return None
    if not has_mapping:
        return None
    if integer_mapping is None:
        raise RuntimeError("input_activation OER/norm requires integer_mapping for post-quant dequant activation")

    expected_shape = tuple(int(item) for item in fp32_record.tensor.shape)
    context = f"{_target_name(fp32_record.layer_idx, fp32_record.module_name)} input_activation"
    dequant_from_grid = _dequant_activation_from_integer_mapping(
        integer_mapping,
        expected_shape=expected_shape,
        context=context,
    )
    cached = _usable_quant_activation_cache(direct_quant_activation_for_oer, expected_shape=expected_shape)
    if cached is None:
        return dequant_from_grid

    if not torch.allclose(cached, dequant_from_grid, atol=1e-5, rtol=1e-5):
        max_abs_diff = torch.max(torch.abs(cached - dequant_from_grid)).detach().cpu().item()
        raise RuntimeError(
            f"{context}: last_quant_activation_for_oer disagrees with q_int/scale/zero_point dequantization "
            f"(max_abs_diff={float(max_abs_diff):.8g})"
        )
    return cached


def _token_topk_ratio(cfg: Any) -> float:
    return float(_setting_get(
        cfg,
        "TOKEN_TOPK_RATIO",
        _nested_get(_cfg_get(cfg, "METRICS", {}), "SCALAR_OER_TOP_RATIO", 0.001),
    ))


def _oracle_shard_path_for_position(token_oracle_reference: TokenTraceOracleReference | None, position: str) -> str:
    if token_oracle_reference is None:
        return ""
    shards = token_oracle_reference.manifest.get("shards", {})
    if not isinstance(shards, Mapping):
        return ""
    value = shards.get(str(position), "")
    if not value:
        return ""
    path = _resolve_uploaded_artifact_path(token_oracle_reference.path, value)
    return str(path)


def _oracle_row_key(sample_id: Any, *, layer_idx: int, module_name: str, position: str) -> str:
    return f"{sample_id}|L{int(layer_idx)}|{str(module_name)}|{str(position)}"


def _sample_delta_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sample_id"),
        row.get("intervention_target"),
        row.get("layer_idx"),
        row.get("module_name"),
        row.get("tensor_position"),
        row.get("granularity"),
        row.get("integer_mapping_source"),
        row.get("actual_integer_mapping_available"),
    )


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _attach_quant_mode_sample_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_sample_delta_key(row), {})[str(row.get("quant_mode"))] = row
    for row in rows:
        row.setdefault("w4a4_minus_a4_topk_survival_rate_delta", float("nan"))
        row.setdefault("w4a4_minus_a4_token_survival_drop_delta", float("nan"))
    for modes in grouped.values():
        if "W4A4" not in modes or "A4-only" not in modes:
            continue
        w4a4 = modes["W4A4"]
        a4 = modes["A4-only"]
        survival_delta = _finite_float(w4a4.get("topk_survival_rate")) - _finite_float(a4.get("topk_survival_rate"))
        drop_delta = _finite_float(w4a4.get("token_survival_drop")) - _finite_float(a4.get("token_survival_drop"))
        for target in (w4a4, a4):
            target["w4a4_minus_a4_topk_survival_rate_delta"] = survival_delta
            target["w4a4_minus_a4_token_survival_drop_delta"] = drop_delta
    return rows


def _trace_has_actual_activation_mapping(
    record: TokenTraceRecord,
    *,
    intervention_layer_idx: int,
    intervention_module_name: str,
    quant_mode: str,
) -> bool:
    return (
        _mode_uses_activation(quant_mode)
        and int(record.layer_idx) == int(intervention_layer_idx)
        and str(record.module_name) == str(intervention_module_name)
        and str(record.tensor_position) == "input"
    )


def _rows_for_batch(
    *,
    cfg: Any,
    fp32_records: Sequence[TokenTraceRecord],
    quant_records: Sequence[TokenTraceRecord],
    fp32_embedding: Tensor,
    quant_embedding: Tensor,
    sample_ids: Sequence[str],
    labels: Tensor,
    bucket: Sequence[str],
    shot_group: Sequence[str],
    quant_mode: str,
    intervention_layer_idx: int,
    intervention_module_name: str,
    activation_qparams: Mapping[str, Any] | None,
    qparams_path: Path,
    qparams_summary: Mapping[str, Any],
    trace_job: TraceJob,
    shard_key: str,
    experiment_signature: str,
    qparams_hash: str,
    direct_integer_mapping: Mapping[str, Any] | None = None,
    direct_quant_activation_for_oer: Tensor | None = None,
    token_oracle_reference: TokenTraceOracleReference | None = None,
) -> list[dict[str, Any]]:
    fp32_by_key = _record_map(fp32_records)
    quant_by_key = _record_map(quant_records)
    if set(fp32_by_key) != set(quant_by_key):
        missing = sorted(set(fp32_by_key) ^ set(quant_by_key))
        raise RuntimeError(f"FP32 and INT4 token trace keys differ: {missing}")

    rows: list[dict[str, Any]] = []
    for key, fp32_record in sorted(fp32_by_key.items()):
        quant_record = quant_by_key[key]
        if list(fp32_record.sample_id) != list(sample_ids) or list(quant_record.sample_id) != list(sample_ids):
            raise RuntimeError(f"{key}: trace sample_id does not match current batch")
        if tuple(fp32_record.padding_mask.shape) != tuple(quant_record.padding_mask.shape) or not torch.equal(
            fp32_record.padding_mask, quant_record.padding_mask
        ):
            raise RuntimeError(f"{key}: FP32 and INT4 token padding_mask differ")

        has_mapping = _trace_has_actual_activation_mapping(
            quant_record,
            intervention_layer_idx=intervention_layer_idx,
            intervention_module_name=intervention_module_name,
            quant_mode=quant_mode,
        )
        integer_mapping = None
        if has_mapping:
            if direct_integer_mapping is not None:
                integer_mapping = dict(direct_integer_mapping)
            else:
                if activation_qparams is None:
                    raise RuntimeError(f"{_target_name(intervention_layer_idx, intervention_module_name)} {quant_mode}: missing activation qparams")
                integer_mapping = _activation_mapping_from_quant_input(quant_record.tensor, activation_qparams)
            is_direct_grid = bool(integer_mapping.get("is_direct_quantlinear_forward_grid", False))
            mapping_source = str(integer_mapping.get("mapping_source", integer_mapping.get("integer_grid_source", "")))
            if str(trace_job.public_tensor_position) == "input_activation" and (mapping_source != "quantlinear_forward" or not is_direct_grid):
                if not _allow_reconstructed_grid_diagnostic(cfg):
                    raise RuntimeError(
                        "formal input_activation integer-grid metrics require direct QuantLinear forward grid; "
                        "the probe should capture this automatically from QuantLinear.forward. "
                        "Set ALLOW_RECONSTRUCTED_GRID_DIAGNOSTIC=true only for diagnostic-only reconstructed-grid output."
                    )

        oracle_batch = None
        if str(trace_job.public_tensor_position) in {"block_output", "input_activation"}:
            if token_oracle_reference is None:
                raise RuntimeError(f"{trace_job.public_tensor_position} Token Survival formal metrics require token_trace_oracle_eval.pt")
            oracle_batch = token_oracle_reference.rows_for_batch(
                layer_idx=int(fp32_record.layer_idx),
                module_name=str(fp32_record.module_name),
                position=str(trace_job.public_tensor_position),
                sample_ids=sample_ids,
            )
            observed_valid = (~fp32_record.padding_mask.detach().cpu().bool()).sum(dim=1).long().reshape(-1)
            oracle_valid = torch.tensor(oracle_batch["valid_token_count"], dtype=torch.long).reshape(-1)
            if not torch.equal(observed_valid, oracle_valid):
                raise RuntimeError(
                    f"token oracle valid_token_count mismatch for {trace_job.public_tensor_position} "
                    f"layer {int(fp32_record.layer_idx)} {fp32_record.module_name}: {observed_valid.tolist()} vs {oracle_valid.tolist()}"
                )

        oracle_manifest = token_oracle_reference.manifest if token_oracle_reference is not None else {}
        oracle_position = str(trace_job.public_tensor_position)
        oracle_shard_path = "" if oracle_batch is None else _oracle_shard_path_for_position(token_oracle_reference, oracle_position)
        post_quant_metric_tensor = _post_quant_metric_tensor_for_input_activation(
            trace_job=trace_job,
            has_mapping=bool(has_mapping),
            fp32_record=fp32_record,
            integer_mapping=integer_mapping,
            direct_quant_activation_for_oer=direct_quant_activation_for_oer,
        )
        int4_metric_tensor = quant_record.tensor if post_quant_metric_tensor is None else post_quant_metric_tensor
        metric_rows = compute_token_survival_rows(
            fp32_record.tensor,
            int4_metric_tensor,
            fp32_record.padding_mask,
            sample_id=sample_ids,
            label=labels,
            bucket=bucket,
            shot_group=shot_group,
            layer_idx=int(fp32_record.layer_idx),
            module_name=str(fp32_record.module_name),
            tensor_position=str(trace_job.public_tensor_position),
            quant_mode=_mode_label(quant_mode),
            granularity="per_tensor",
            integer_mapping=integer_mapping,
            require_integer_mapping=bool(has_mapping),
            fp32_embedding=fp32_embedding,
            int4_embedding=quant_embedding,
            topk=_setting_get(cfg, "TOKEN_TOPK", None),
            topk_ratio=_token_topk_ratio(cfg),
            fp32_topk_token_ids=None if oracle_batch is None else oracle_batch["fp32_topk_token_ids"],
            fp32_topk_unit=None if oracle_batch is None else "token",
            fp32_topk_source=None if oracle_batch is None else "fp32_token_energy",
            survival_min_norm_retention=float(_setting_get(cfg, "SURVIVAL_MIN_NORM_RETENTION", 0.5)),
        )
        for row_idx, row in enumerate(metric_rows):
            row.update(
                {
                    "intervention_layer_idx": int(intervention_layer_idx),
                    "intervention_module_name": str(intervention_module_name),
                    "intervention_target": _target_name_short(intervention_layer_idx, intervention_module_name),
                    "trace_target_name": str(fp32_record.target_name),
                    "trace_scope": str(trace_job.scope),
                    "trace_layer_idx": int(trace_job.layer_idx),
                    "tensor_position": str(trace_job.public_tensor_position),
                    "topk_unit": "token",
                    "topk_source": "fp32_token_energy",
                    "topk_oracle_path": "" if token_oracle_reference is None or oracle_batch is None else str(token_oracle_reference.path),
                    "topk_oracle_cache_type": "" if token_oracle_reference is None or oracle_batch is None else str(token_oracle_reference.manifest.get("cache_type", "")),
                    "topk_ratio": _token_topk_ratio(cfg),
                    "topk_k_policy": "runtime_topk" if token_oracle_reference is None or oracle_batch is None else str(token_oracle_reference.manifest.get("topk_k_policy", "")),
                    "oracle_topk_k": "" if oracle_batch is None else int(oracle_batch["topk_k"][row_idx]),
                    "oracle_valid_token_count": "" if oracle_batch is None else int(oracle_batch["valid_token_count"][row_idx]),
                    "oracle_protocol_version": "" if oracle_batch is None else oracle_manifest.get("oracle_protocol_version", ""),
                    "oracle_valid_unit": "" if oracle_batch is None else oracle_manifest.get("valid_unit", ""),
                    "oracle_padding_excluded": "" if oracle_batch is None else bool(oracle_manifest.get("padding_excluded", False)),
                    "oracle_topk_unit": "" if oracle_batch is None else oracle_manifest.get("topk_unit", ""),
                    "oracle_topk_source": "" if oracle_batch is None else oracle_manifest.get("topk_source", ""),
                    "oracle_storage_format": "" if oracle_batch is None else oracle_manifest.get("storage_format", ""),
                    "oracle_shard_path": oracle_shard_path,
                    "oracle_row_key": "" if oracle_batch is None else _oracle_row_key(
                        sample_ids[row_idx],
                        layer_idx=int(fp32_record.layer_idx),
                        module_name=str(fp32_record.module_name),
                        position=oracle_position,
                    ),
                    "token_energy_definition": "l2_norm",
                    "topk_selected_on": "fp32",
                    "tracked_token_ids_fixed": True,
                    "aux_topk_source": "",
                    "metric_role": "optional_control" if _mode_label(quant_mode) == "W4-only" else "main",
                    "metric_valid_for_paper": bool(
                        not has_mapping
                        or (
                            integer_mapping is not None
                            and str(integer_mapping.get("mapping_source", integer_mapping.get("integer_grid_source", ""))) == "quantlinear_forward"
                            and bool(integer_mapping.get("is_direct_quantlinear_forward_grid", False))
                        )
                    ),
                    "not_applicable_reason": "" if not has_mapping or (integer_mapping is not None and bool(integer_mapping.get("is_direct_quantlinear_forward_grid", False))) else "reconstructed_grid_not_valid_for_paper",
                    "shard_key": str(shard_key),
                    "sample_shard_id": int(trace_job.sample_shard_id),
                    "protocol_version": PROTOCOL_VERSION,
                    "experiment_signature": str(experiment_signature),
                    "qparams_hash": str(qparams_hash),
                    "activation_shape": (1, int(fp32_record.canonical_shape[1]), int(fp32_record.canonical_shape[2])),
                    "original_layout": str(fp32_record.original_layout),
                    "original_shape": tuple(int(item) for item in fp32_record.original_shape),
                    "canonical_layout": str(fp32_record.canonical_layout),
                    "canonical_shape": (1, int(fp32_record.canonical_shape[1]), int(fp32_record.canonical_shape[2])),
                    "layout_inference_source": str(fp32_record.layout_inference_source),
                    "padding_mask_available": bool(fp32_record.padding_mask_available),
                    "padding_mask_aligned": bool(fp32_record.padding_mask_aligned),
                    "mask_status": str(fp32_record.mask_status),
                    "valid_token_policy": _valid_token_policy(cfg),
                    "qparams_path": str(qparams_path),
                    "actual_integer_mapping_available": bool(has_mapping),
                    "integer_mapping_is_direct_quantlinear_forward": bool(integer_mapping.get("is_direct_quantlinear_forward_grid", False)) if has_mapping and integer_mapping is not None else "",
                    "direct_quantlinear_forward_grid": bool(integer_mapping.get("is_direct_quantlinear_forward_grid", False)) if has_mapping and integer_mapping is not None else "",
                    "integer_grid_source": integer_mapping.get("integer_grid_source", integer_mapping.get("mapping_source", "")) if has_mapping and integer_mapping is not None else "",
                    "integer_mapping_qparams_source": integer_mapping.get("qparams_source", "quantlinear_forward" if integer_mapping.get("is_direct_quantlinear_forward_grid", False) else "calibration_qparams") if has_mapping and integer_mapping is not None else "",
                    "int4_activation_source": "post_quant_dequant" if str(trace_job.public_tensor_position) == "input_activation" and post_quant_metric_tensor is not None else "",
                    "topk_token_clip_rate_deprecated_alias_of": "topk_token_strict_overflow_rate",
                    "topk_token_near_zero_rate_deprecated_alias_of": "topk_token_exact_zero_rate",
                    "high_energy_scalar_clip_rate_deprecated_alias_of": "high_energy_scalar_strict_overflow_rate",
                    "high_energy_scalar_near_zero_rate_deprecated_alias_of": "high_energy_scalar_exact_zero_rate",
                    "topk_clip_rate_deprecated_alias_of": "high_energy_scalar_strict_overflow_rate",
                    "topk_near_zero_rate_deprecated_alias_of": "high_energy_scalar_exact_zero_rate",
                }
            )
            row.update(dict(qparams_summary))
            rows.append(row)
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        flat = value.detach().cpu().reshape(-1)
        if int(flat.numel()) == 1:
            return flat[0].item()
        return flat.tolist()
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {str(key): _json_safe(item) for key, item in vars(value).items() if not str(key).startswith("_")}
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    _write_json_atomic(path, manifest)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_with_aliases = _row_with_metric_aliases(row)
            writer.writerow({key: _csv_value(row_with_aliases.get(key, "")) for key in CSV_COLUMNS})
    os.replace(tmp_path, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(tmp_path, path)


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _file_sha256(path: Path) -> str:
    resolved = _portable_path(path)
    stat = resolved.stat()
    key = (str(resolved.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


def _file_identity(path: Path | None, *, hash_file: bool) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "sha256": "", "size": "", "mtime_ns": ""}
    resolved = _portable_path(path)
    exists = resolved.exists()
    identity = {
        "path": str(resolved),
        "exists": bool(exists),
        "sha256": "",
        "size": "",
        "mtime_ns": "",
    }
    if exists and resolved.is_file():
        stat = resolved.stat()
        identity["size"] = int(stat.st_size)
        identity["mtime_ns"] = int(stat.st_mtime_ns)
        if hash_file:
            identity["sha256"] = _file_sha256(resolved)
    return identity


def _cfg_path(cfg: Any, *keys: str) -> Path | None:
    for key in keys:
        value = _setting_get(cfg, key, None)
        if value is None:
            value = _cfg_get(cfg, key, None)
        if value is not None:
            return _cfg_path_value(cfg, value)
    return None


def _hash_artifact_files(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "HASH_ARTIFACT_FILES", True))


def _checkpoint_identity(cfg: Any) -> dict[str, Any]:
    path = _cfg_path(
        cfg,
        "CHECKPOINT_PATH",
        "CKPT_PATH",
        "MODEL_CHECKPOINT",
        "PRETRAINED_CHECKPOINT_PATH",
        "WEIGHTS_PATH",
    )
    return _file_identity(path, hash_file=_hash_artifact_files(cfg))


def _quant_config_identity(cfg: Any) -> Any:
    quant_cfg = _quant_cfg(cfg)
    if hasattr(quant_cfg, "__dict__"):
        return {key: value for key, value in vars(quant_cfg).items() if not str(key).startswith("_")}
    return str(quant_cfg)


def _qparams_path(result_dir: Path, layer_idx: int, module_name: str) -> Path:
    return result_dir / QPARAMS_DIRNAME / f"layer_{int(layer_idx):02d}_{module_name}_qparams.pt"


def _qparams_manifest_path(qparams_path: Path) -> Path:
    return qparams_path.with_name(f"{qparams_path.name}{QPARAMS_MANIFEST_SUFFIX}")


def _qparams_signature(cfg: Any, layer_idx: int, module_name: str) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "token_survival_qparams",
        "checkpoint": _checkpoint_identity(cfg),
        "calibration_csv": _file_identity(_split_csv(cfg, "calibration"), hash_file=_hash_artifact_files(cfg)),
        "quant_config": _quant_config_identity(cfg),
        "target": {"layer_idx": int(layer_idx), "module_name": str(module_name)},
    }
    return _stable_hash(payload)


def _save_qparams(
    result_dir: Path,
    layer_idx: int,
    module_name: str,
    qparams: Mapping[str, Any],
    *,
    cfg: Any,
) -> Path:
    path = _qparams_path(result_dir, int(layer_idx), module_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(dict(qparams), tmp_path)
    os.replace(tmp_path, path)
    _write_json_atomic(
        _qparams_manifest_path(path),
        {
            "status": "completed",
            "protocol_version": PROTOCOL_VERSION,
            "qparams_signature": _qparams_signature(cfg, int(layer_idx), module_name),
            "qparams_hash": _file_sha256(path),
            "layer_idx": int(layer_idx),
            "module_name": str(module_name),
        },
    )
    return path


def _load_qparams_if_valid(result_dir: Path, layer_idx: int, module_name: str, *, cfg: Any) -> tuple[dict[str, Any], Path] | None:
    path = _qparams_path(result_dir, int(layer_idx), module_name)
    manifest_path = _qparams_manifest_path(path)
    if not path.exists() or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("status") != "completed":
        return None
    if int(manifest.get("protocol_version", -1)) != PROTOCOL_VERSION:
        return None
    if str(manifest.get("qparams_signature", "")) != _qparams_signature(cfg, int(layer_idx), module_name):
        return None
    if str(manifest.get("qparams_hash", "")) != _file_sha256(path):
        return None
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise TypeError(f"qparams cache must be a mapping, got {type(loaded)!r}: {path}")
    return dict(loaded), path


def _load_or_calibrate_qparams(
    cfg: Any,
    *,
    result_dir: Path,
    layer_idx: int,
    module_name: str,
    calibration_loader: DataLoader,
) -> tuple[dict[str, Any], Path]:
    cached = _load_qparams_if_valid(result_dir, int(layer_idx), module_name, cfg=cfg)
    if cached is not None:
        return cached
    qparams = calibrate_target_module(cfg, int(layer_idx), module_name, calibration_loader)
    qparams_path = _save_qparams(result_dir, int(layer_idx), module_name, qparams, cfg=cfg)
    return dict(qparams), qparams_path


def _shard_key(quant_mode: str, trace_job: TraceJob) -> str:
    return trace_job.shard_key.format(quant_mode=_slug(_mode_label(quant_mode)))


def _shard_paths(result_dir: Path, shard_key: str) -> dict[str, Path]:
    return {
        "sample_metrics": result_dir / SHARDS_DIRNAME / f"{shard_key}.csv",
        "topk_compact": result_dir / TOPK_COMPACT_DIRNAME / f"{shard_key}.jsonl",
        "partial_summary": result_dir / PARTIAL_SUMMARY_DIRNAME / f"{shard_key}.json",
        "done": result_dir / DONE_DIRNAME / f"{shard_key}.done.json",
    }


def _sample_order_hash_from_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return _stable_hash([str(row.get("sample_id", "")) for row in rows])


def _shard_signature(
    cfg: Any,
    *,
    quant_mode: str,
    trace_job: TraceJob,
    intervention_layer_idx: int,
    intervention_module_name: str,
    qparams_hash: str,
) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "metric_semantic_version": SHARD_METRIC_SEMANTIC_VERSION,
        "experiment": "ffn_token_survival_probe",
        "checkpoint": _checkpoint_identity(cfg),
        "eval_csv": _file_identity(_split_csv(cfg, "eval"), hash_file=_hash_artifact_files(cfg)),
        "calibration_csv": _file_identity(_split_csv(cfg, "calibration"), hash_file=_hash_artifact_files(cfg)),
        "quant_config": _quant_config_identity(cfg),
        "quant_mode": _mode_label(quant_mode),
        "intervention": {"layer_idx": int(intervention_layer_idx), "module_name": str(intervention_module_name)},
        "trace_job": {
            "scope": trace_job.scope,
            "layer_idx": int(trace_job.layer_idx),
            "module_name": trace_job.module_name,
            "tensor_position": trace_job.public_tensor_position,
            "sample_shard_id": int(trace_job.sample_shard_id),
        },
        "topk_source": "fp32_token_energy",
        "token_energy_definition": "l2_norm",
        "topk_selected_on": "fp32",
        "tracked_token_ids_fixed": True,
        "topk": _setting_get(cfg, "TOKEN_TOPK", None),
        "topk_ratio": _token_topk_ratio(cfg),
        "valid_token_policy": _valid_token_policy(cfg),
        "main_oer_metric": SHARD_MAIN_OER_COLUMN,
        "required_metric_columns": list(_required_metric_columns_for_quant_mode(quant_mode)),
        "input_activation_metric_tensor": "post_quant_dequant_activation",
        "amp_enabled": bool(_setting_get(cfg, "AMP", _setting_get(cfg, "USE_AMP", False))),
        "qparams_hash": str(qparams_hash),
    }
    return _stable_hash(payload)


def _finite_csv_metric(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if text == "":
        return False
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def _completed_shard_metrics_valid(sample_metrics_path: Path, *, expected_rows: int | None, quant_mode: str) -> bool:
    required_metric_columns = _required_metric_columns_for_quant_mode(quant_mode)
    try:
        with sample_metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            required_columns = {"sample_id", *required_metric_columns}
            if required_columns - fieldnames:
                return False
            row_count = 0
            valid_counts = {column: 0 for column in required_metric_columns}
            for row in reader:
                row_count += 1
                for column in required_metric_columns:
                    if _finite_csv_metric(row.get(column)):
                        valid_counts[column] += 1
    except OSError:
        return False
    if row_count <= 0:
        return False
    if expected_rows is not None and int(expected_rows) >= 0 and int(expected_rows) != int(row_count):
        return False
    return all(count > 0 for count in valid_counts.values())


def _shard_completed(paths: Mapping[str, Path], *, signature: str) -> bool:
    done_path = paths["done"]
    if not done_path.exists() or not paths["sample_metrics"].exists() or not paths["topk_compact"].exists():
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        done.get("status") == "completed"
        and int(done.get("protocol_version", -1)) == PROTOCOL_VERSION
        and str(done.get("metric_semantic_version", "")) == SHARD_METRIC_SEMANTIC_VERSION
        and str(done.get("experiment_signature", "")) == str(signature)
    ):
        return False
    try:
        expected_rows = int(done.get("num_rows", -1))
    except (TypeError, ValueError):
        return False
    return _completed_shard_metrics_valid(
        paths["sample_metrics"],
        expected_rows=expected_rows,
        quant_mode=str(done.get("quant_mode", "")),
    )


def _topk_compact_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "sample_id",
        "label",
        "bucket",
        "tail_included",
        "quant_mode",
        "intervention_layer_idx",
        "intervention_module_name",
        "layer_idx",
        "module_name",
        "tensor_position",
        "topk",
        "fp32_topk_token_ids",
        "int4_topk_token_ids",
        "topk_norm_retention_values",
        "topk_survival_rate",
        "token_survival_drop",
        LEGACY_OER_ALIAS_COLUMN,
        "token_oer_retention",
        "oer_retention_deprecated_alias_of",
        "topk_token_near_zero_pm1_rate",
        "topk_token_strict_overflow_rate",
        "topk_token_boundary_hit_rate",
        "shard_key",
        "sample_shard_id",
    ]
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        row_with_aliases = _row_with_metric_aliases(row)
        compact_rows.append({key: row_with_aliases.get(key, "") for key in keys})
    return compact_rows


class _SampleOrderHashBuilder:
    def __init__(self) -> None:
        self._hasher = hashlib.sha256()
        self._hasher.update(b"[")
        self._count = 0

    def update(self, sample_id: Any) -> None:
        if self._count:
            self._hasher.update(b",")
        self._hasher.update(json.dumps(str(sample_id), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        self._count += 1

    def hexdigest(self) -> str:
        clone = self._hasher.copy()
        clone.update(b"]")
        return clone.hexdigest()


class _ShardArtifactStreamWriter:
    def __init__(
        self,
        *,
        paths: Mapping[str, Path],
        signature: str,
        shard_key: str,
        trace_job: TraceJob,
        quant_mode: str,
        qparams_hash: str,
    ) -> None:
        self.paths = dict(paths)
        self.signature = str(signature)
        self.shard_key = str(shard_key)
        self.trace_job = trace_job
        self.quant_mode = str(quant_mode)
        self.qparams_hash = str(qparams_hash)
        self.sample_tmp = self.paths["sample_metrics"].with_name(f"{self.paths['sample_metrics'].name}.tmp")
        self.topk_tmp = self.paths["topk_compact"].with_name(f"{self.paths['topk_compact'].name}.tmp")
        self.sample_handle: Any | None = None
        self.topk_handle: Any | None = None
        self.csv_writer: csv.DictWriter | None = None
        self.sample_hash = _SampleOrderHashBuilder()
        self.num_rows = 0

    def __enter__(self) -> "_ShardArtifactStreamWriter":
        self.paths["sample_metrics"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["topk_compact"].parent.mkdir(parents=True, exist_ok=True)
        self.sample_handle = self.sample_tmp.open("w", encoding="utf-8-sig", newline="")
        self.topk_handle = self.topk_tmp.open("w", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.sample_handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        self.csv_writer.writeheader()
        return self

    def append(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.csv_writer is None or self.topk_handle is None:
            raise RuntimeError("shard artifact writer is not open")
        for row in rows:
            row_with_aliases = _row_with_metric_aliases(row)
            self.csv_writer.writerow({key: _csv_value(row_with_aliases.get(key, "")) for key in CSV_COLUMNS})
            compact_row = _topk_compact_rows((row_with_aliases,))[0]
            self.topk_handle.write(json.dumps(_json_safe(compact_row), ensure_ascii=False, sort_keys=True))
            self.topk_handle.write("\n")
            self.sample_hash.update(row.get("sample_id", ""))
            self.num_rows += 1

    def close_success(self) -> None:
        if self.sample_handle is not None:
            self.sample_handle.close()
            self.sample_handle = None
        if self.topk_handle is not None:
            self.topk_handle.close()
            self.topk_handle = None
        os.replace(self.sample_tmp, self.paths["sample_metrics"])
        os.replace(self.topk_tmp, self.paths["topk_compact"])
        sample_order_hash = self.sample_hash.hexdigest()
        _write_json_atomic(
            self.paths["partial_summary"],
            {
                "status": "completed",
                "protocol_version": PROTOCOL_VERSION,
                "metric_semantic_version": SHARD_METRIC_SEMANTIC_VERSION,
                "main_oer_metric": SHARD_MAIN_OER_COLUMN,
                "required_metric_columns": list(_required_metric_columns_for_quant_mode(self.quant_mode)),
                "experiment_signature": self.signature,
                "shard_key": self.shard_key,
                "num_rows": int(self.num_rows),
                "sample_order_hash": sample_order_hash,
                "quant_mode": _mode_label(self.quant_mode),
                "trace_scope": self.trace_job.scope,
                "trace_layer_idx": int(self.trace_job.layer_idx),
                "trace_position": self.trace_job.public_tensor_position,
                "qparams_hash": self.qparams_hash,
            },
        )
        _write_json_atomic(
            self.paths["done"],
            {
                "status": "completed",
                "protocol_version": PROTOCOL_VERSION,
                "metric_semantic_version": SHARD_METRIC_SEMANTIC_VERSION,
                "main_oer_metric": SHARD_MAIN_OER_COLUMN,
                "required_metric_columns": list(_required_metric_columns_for_quant_mode(self.quant_mode)),
                "experiment_signature": self.signature,
                "shard_key": self.shard_key,
                "num_rows": int(self.num_rows),
                "sample_order_hash": sample_order_hash,
                "sample_shard_id": int(self.trace_job.sample_shard_id),
                "quant_mode": _mode_label(self.quant_mode),
                "trace_scope": self.trace_job.scope,
                "trace_layer_idx": int(self.trace_job.layer_idx),
                "trace_position": self.trace_job.public_tensor_position,
                "qparams_hash": self.qparams_hash,
            },
        )

    def close_abort(self) -> None:
        if self.sample_handle is not None:
            self.sample_handle.close()
            self.sample_handle = None
        if self.topk_handle is not None:
            self.topk_handle.close()
            self.topk_handle = None
        for path in (self.sample_tmp, self.topk_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.close_success()
        else:
            self.close_abort()
        return False


def _write_shard_artifacts(
    *,
    paths: Mapping[str, Path],
    rows: Sequence[Mapping[str, Any]],
    signature: str,
    shard_key: str,
    trace_job: TraceJob,
    quant_mode: str,
    qparams_hash: str,
) -> None:
    _write_rows(paths["sample_metrics"], rows)
    _write_jsonl_atomic(paths["topk_compact"], _topk_compact_rows(rows))
    sample_order_hash = _sample_order_hash_from_rows(rows)
    _write_json_atomic(
        paths["partial_summary"],
        {
            "status": "completed",
            "protocol_version": PROTOCOL_VERSION,
            "metric_semantic_version": SHARD_METRIC_SEMANTIC_VERSION,
            "main_oer_metric": SHARD_MAIN_OER_COLUMN,
            "required_metric_columns": list(_required_metric_columns_for_quant_mode(quant_mode)),
            "experiment_signature": signature,
            "shard_key": shard_key,
            "num_rows": int(len(rows)),
            "sample_order_hash": sample_order_hash,
            "quant_mode": _mode_label(quant_mode),
            "trace_scope": trace_job.scope,
            "trace_layer_idx": int(trace_job.layer_idx),
            "trace_position": trace_job.public_tensor_position,
            "qparams_hash": qparams_hash,
        },
    )
    _write_json_atomic(
        paths["done"],
        {
            "status": "completed",
            "protocol_version": PROTOCOL_VERSION,
            "metric_semantic_version": SHARD_METRIC_SEMANTIC_VERSION,
            "main_oer_metric": SHARD_MAIN_OER_COLUMN,
            "required_metric_columns": list(_required_metric_columns_for_quant_mode(quant_mode)),
            "experiment_signature": signature,
            "shard_key": shard_key,
            "num_rows": int(len(rows)),
            "sample_order_hash": sample_order_hash,
            "sample_shard_id": int(trace_job.sample_shard_id),
            "quant_mode": _mode_label(quant_mode),
            "trace_scope": trace_job.scope,
            "trace_layer_idx": int(trace_job.layer_idx),
            "trace_position": trace_job.public_tensor_position,
            "qparams_hash": qparams_hash,
        },
    )


def _read_shard_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing token survival shard csv: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
    return rows


def _streaming_summary_enabled(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "STREAMING_SUMMARY", False))


def _write_combined_sample_csv(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "WRITE_COMBINED_SAMPLE_CSV", not _streaming_summary_enabled(cfg)))


def _run_case_plots(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "RUN_CASE_PLOTS", not _streaming_summary_enabled(cfg)))


def _export_diagnostic_comparison(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "EXPORT_DIAGNOSTIC_COMPARISON", False))


def _build_manifest(
    *,
    cfg: Any,
    result_dir: Path,
    metrics_csv: Path,
    summary_csv: Path,
    comparison_summary_csv: Path,
    quant_mode_comparison_csv: Path,
    manifest_path: Path,
    plot_paths: Mapping[str, Any],
    sample_table: Any,
    summary_df: Any,
    comparison_df: Any,
    quant_mode_comparison_df: Any,
    target_modules: Sequence[tuple[int, str]],
    num_transformer_layers: int,
    shard_csv_paths: Sequence[Path],
    num_rows: int,
    core_reference: CoreEvalReference,
    token_oracle_reference: TokenTraceOracleReference,
    combined_sample_csv_written: bool,
) -> dict[str, Any]:
    observed_quant_modes = summary_df.attrs.get("observed_quant_modes")
    if not observed_quant_modes and isinstance(sample_table, pd.DataFrame) and "quant_mode" in sample_table.columns:
        observed_quant_modes = sorted(str(item) for item in sample_table["quant_mode"].dropna().unique().tolist())
    observed_quant_modes = observed_quant_modes or []
    available_comparisons = quant_mode_comparison_df.attrs.get("available_quant_mode_comparisons") or sorted(str(item) for item in quant_mode_comparison_df.get("comparison_name", pd.Series(dtype=str)).dropna().unique().tolist())
    missing_quant_modes = quant_mode_comparison_df.attrs.get("missing_quant_modes") or sorted(set(KNOWN_QUANT_MODES) - set(observed_quant_modes))
    label_mapping_warnings = summary_df.attrs.get("label_mapping_warnings", [])
    extra_tables = summary_df.attrs.get("extra_tables", {})
    if not isinstance(extra_tables, Mapping):
        extra_tables = {}
    detail_summary_df = extra_tables.get("detail_summary", pd.DataFrame())
    detail_comparison_df = extra_tables.get("detail_comparison", pd.DataFrame())
    diagnostic_comparison_df = extra_tables.get("diagnostic_comparison", pd.DataFrame())
    detail_quant_mode_comparison_df = extra_tables.get("detail_quant_mode_comparison", pd.DataFrame())
    diagnostic_quant_mode_comparison_df = extra_tables.get("diagnostic_quant_mode_comparison", pd.DataFrame())
    intervention_level_summary_df = extra_tables.get("intervention_level_summary", pd.DataFrame())
    intervention_level_comparison_df = extra_tables.get("intervention_level_comparison", pd.DataFrame())
    diagnostic_exported = bool(_export_diagnostic_comparison(cfg))
    protocol = group_protocol_metadata(
        tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
    )
    outputs: dict[str, Any] = {
        "metrics_csv": metrics_csv,
        "summary_v2_long_csv": summary_csv,
        "detail_summary_v2_long_csv": result_dir / DETAIL_SUMMARY_FILENAME,
        "comparison_summary_csv": comparison_summary_csv,
        "detail_comparison_summary_csv": result_dir / DETAIL_COMPARISON_SUMMARY_FILENAME,
        "quant_mode_comparison_csv": quant_mode_comparison_csv,
        "detail_quant_mode_comparison_csv": result_dir / DETAIL_QUANT_MODE_COMPARISON_FILENAME,
        "intervention_level_summary_csv": result_dir / INTERVENTION_LEVEL_SUMMARY_FILENAME,
        "intervention_level_comparison_csv": result_dir / INTERVENTION_LEVEL_COMPARISON_FILENAME,
        "manifest": manifest_path,
        "plots": plot_paths,
        "shard_csvs": [str(path) for path in shard_csv_paths],
    }
    if diagnostic_exported:
        outputs["diagnostic_comparison_summary_csv"] = result_dir / DIAGNOSTIC_COMPARISON_SUMMARY_FILENAME
        outputs["diagnostic_quant_mode_comparison_csv"] = result_dir / DIAGNOSTIC_QUANT_MODE_COMPARISON_FILENAME
    return {
        "experiment": "ffn_token_survival_probe",
        "result_dir": result_dir,
        "protocol_version": PROTOCOL_VERSION,
        "metric_semantic_version": SHARD_METRIC_SEMANTIC_VERSION,
        "main_oer_metric": SHARD_MAIN_OER_COLUMN,
        "required_metric_columns": list(SHARD_REQUIRED_METRIC_COLUMNS),
        "input_activation_metric_tensor": "post_quant_dequant_activation",
        "num_transformer_layers": int(num_transformer_layers),
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
        "comparison_summary_policy": {
            "formal": "metric_role == formal_main and metric_valid == true and metric_valid_for_paper == true",
            "detail": "metric_role == detail and metric_valid == true and gap_enabled == true and comparable_across_groups == true",
            "diagnostic": "metric_role == diagnostic; exported only when EXPORT_DIAGNOSTIC_COMPARISON=true",
        },
        "intervention_level_auxiliary_policy": {
            "embedding_distortion": "deduplicated from block_output trace rows into intervention-level auxiliary summary",
            "dedup_source": "min_trace_layer_idx_after_consistency_check",
            "consistency_tolerance": {"rtol": 1e-6, "atol": 1e-8},
        },
        "streaming_summary_enabled": bool(_streaming_summary_enabled(cfg)),
        "summary_read_chunksize": _setting_get(cfg, "SUMMARY_READ_CHUNKSIZE", None),
        "combined_sample_csv_written": bool(combined_sample_csv_written),
        "block_trace_group_size": int(_block_trace_group_size(cfg)),
        "topk_unit": "token",
        "topk_source": "fp32_token_energy",
        "token_energy_definition": "l2_norm",
        "topk_selected_on": "fp32",
        "tracked_token_ids_fixed": True,
        "aux_topk_source": "",
        "oer_auxiliary_diagnostic_enabled": bool(_setting_get(cfg, "RUN_OER_AUX_DIAGNOSTIC", False)),
        "oer_auxiliary_policy": "disabled_by_default; scalar oracle is rejected by formal main and may only write isolated auxiliary diagnostics",
        "token_topk_unit": "token",
        "token_topk_source": "fp32_token_energy",
        "fp32_topk_fixed": True,
        "core_eval_cache": {
            "path": str(core_reference.path),
            "sample_order_hash": core_reference.sample_order_hash,
            "checkpoint_hash": core_reference.checkpoint_hash,
            "embedding_shape": tuple(int(item) for item in core_reference.embedding.shape),
            "padding_mask_shape": tuple(int(item) for item in core_reference.padding_mask.shape),
        },
        "token_trace_oracle_cache": {
            "path": str(token_oracle_reference.path),
            "oracle_protocol_version": token_oracle_reference.manifest.get("oracle_protocol_version"),
            "cache_type": token_oracle_reference.manifest.get("cache_type"),
            "storage_format": token_oracle_reference.manifest.get("storage_format"),
            "available_positions": token_oracle_reference.manifest.get("available_positions"),
            "topk_unit": token_oracle_reference.manifest.get("topk_unit"),
            "topk_source": token_oracle_reference.manifest.get("topk_source"),
            "topk_ratio": token_oracle_reference.manifest.get("topk_ratio"),
            "topk_k_policy": token_oracle_reference.manifest.get("topk_k_policy"),
            "valid_unit": token_oracle_reference.manifest.get("valid_unit"),
            "padding_excluded": token_oracle_reference.manifest.get("padding_excluded"),
            "layout_protocol_version": token_oracle_reference.manifest.get("layout_protocol_version"),
            "sample_order_hash": token_oracle_reference.sample_order_hash,
            "checkpoint_hash": token_oracle_reference.checkpoint_hash,
        },
        "valid_token_policy": _valid_token_policy(cfg),
        "allow_unmasked_token_trace_deprecated_alias": bool(_setting_get(cfg, "ALLOW_UNMASKED_TOKEN_TRACE", False)),
        "token_topk_ratio": _token_topk_ratio(cfg),
        "token_topk_ratio_default_source": "FFN_TOKEN_SURVIVAL_PROBE.TOKEN_TOPK_RATIO overrides METRICS.SCALAR_OER_TOP_RATIO",
        "token_metric_aggregation": {
            "aggregation_unit": "token_sample_summary",
            "aggregation_fn": "mean",
            "weighting": "sample_equal",
            "description": "Token metrics are summarized inside each sample first, then group values are sample-equal macro means.",
            "sample_unit_reserved_for": "final embedding/logit/sample scalar metrics",
        },
        "integer_mapping_protocol": {
            "capture_quantlinear_integer_grid": _capture_quantlinear_integer_grid(cfg),
            "capture_quantlinear_integer_grid_policy": "automatic_for_formal_input_activation",
            "capture_scope": str(_setting_get(cfg, "INTEGER_GRID_CAPTURE_SCOPE", "selected_targets_only")),
            "save_device": str(_setting_get(cfg, "INTEGER_GRID_SAVE_DEVICE", "cpu")),
            "detach": bool(_setting_get(cfg, "INTEGER_GRID_DETACH", True)),
            "canonical_layout": str(_setting_get(cfg, "INTEGER_GRID_CANONICAL_LAYOUT", "BTC")),
            "direct_quantlinear_forward_grid": bool(_capture_quantlinear_integer_grid(cfg)),
            "allow_reconstructed_grid_diagnostic": bool(_allow_reconstructed_grid_diagnostic(cfg)),
            "integer_grid_source": "quantlinear_forward",
            "fallback_source": "reconstructed_from_activation_and_qparams",
            "mapping_source_field": "integer_grid_source",
            "definitions": {
                "q_preclip": "round(x / scale + zero_point) before clamp",
                "q_int": "clamp(q_preclip, qmin, qmax)",
                "boundary_hit_mask": "q_int <= qmin or q_int >= qmax",
                "strict_overflow_mask": "q_preclip < qmin or q_preclip > qmax",
                "exact_zero_mask": "q_int == zero_point",
                "near_zero_pm1_mask": "abs(q_int - zero_point) <= 1",
            },
            "drift_risk": "Only quantlinear_forward is a true forward integer grid; reconstructed_from_activation_and_qparams is a diagnostic reconstruction.",
        },
        "sample_csv_extra_outputs": {
            "topk_norm_retention_values": "JSON list of per-FP32-topk-token norm retention values",
            "topk_token_exact_zero_rate": "token-level exact zero: q_int == zero_point inside FP32 top-k tokens",
            "topk_token_near_zero_pm1_rate": "token-level pm1 near-zero: abs(q_int - zero_point) <= 1 inside FP32 top-k tokens",
            "topk_token_boundary_hit_rate": "token-level boundary hit: q_int <= qmin or q_int >= qmax inside FP32 top-k tokens",
            "topk_token_strict_overflow_rate": "token-level strict overflow: q_preclip < qmin or q_preclip > qmax inside FP32 top-k tokens",
            "high_energy_scalar_exact_zero_rate": "scalar exact-zero rate inside FP32 top-k tokens",
            "high_energy_scalar_near_zero_pm1_rate": "scalar pm1 near-zero rate inside FP32 top-k tokens",
            "high_energy_scalar_boundary_hit_rate": "scalar boundary-hit rate inside FP32 top-k tokens",
            "high_energy_scalar_strict_overflow_rate": "scalar strict-overflow rate inside FP32 top-k tokens",
            "deprecated_aliases": {
                "topk_token_clip_rate": "topk_token_strict_overflow_rate",
                "topk_token_near_zero_rate": "topk_token_exact_zero_rate",
                "high_energy_scalar_clip_rate": "high_energy_scalar_strict_overflow_rate",
                "high_energy_scalar_near_zero_rate": "high_energy_scalar_exact_zero_rate",
                "topk_clip_rate": "high_energy_scalar_strict_overflow_rate",
                "topk_near_zero_rate": "high_energy_scalar_exact_zero_rate"
            },
            "qparams_summary_columns": "activation/weight scale and zero_point summary fields are repeated on sample rows",
            "w4a4_minus_a4_token_survival_drop_delta": "sample-level W4A4 token_survival_drop minus A4-only token_survival_drop when both modes are present",
        },
        "observed_quant_modes": list(observed_quant_modes),
        "available_quant_mode_comparisons": list(available_comparisons),
        "missing_quant_modes": list(missing_quant_modes),
        "target_modules": [
            {"layer_idx": int(layer_idx), "module_name": str(module_name), "target": _target_name_short(int(layer_idx), str(module_name))}
            for layer_idx, module_name in target_modules
        ],
        "counts": {
            "num_sample_rows": int(num_rows),
            "num_summary_rows": int(len(summary_df)),
            "num_detail_summary_rows": int(len(detail_summary_df)),
            "num_comparison_rows": int(len(comparison_df)),
            "num_detail_comparison_rows": int(len(detail_comparison_df)),
            "num_diagnostic_comparison_rows": int(len(diagnostic_comparison_df)) if diagnostic_exported else 0,
            "num_quant_mode_comparison_rows": int(len(quant_mode_comparison_df)),
            "num_detail_quant_mode_comparison_rows": int(len(detail_quant_mode_comparison_df)),
            "num_diagnostic_quant_mode_comparison_rows": int(len(diagnostic_quant_mode_comparison_df)) if diagnostic_exported else 0,
            "num_intervention_level_summary_rows": int(len(intervention_level_summary_df)),
            "num_intervention_level_comparison_rows": int(len(intervention_level_comparison_df)),
            "num_shard_csvs": int(len(shard_csv_paths)),
        },
        "label_mapping_warnings": list(label_mapping_warnings),
        "grouping_protocol": protocol,
        "outputs": outputs,
    }




def _run_streamed_token_eval(
    cfg: Any,
    *,
    fp32_model: nn.Module,
    quant_model: nn.Module,
    eval_loader: DataLoader,
    trace_job: TraceJob,
    quant_mode: str,
    intervention_layer_idx: int,
    intervention_module_name: str,
    activation_qparams: Mapping[str, Any] | None,
    qparams_path: Path,
    qparams_summary: Mapping[str, Any],
    shard_key: str,
    experiment_signature: str,
    qparams_hash: str,
    core_reference: CoreEvalReference,
    token_oracle_reference: TokenTraceOracleReference | None,
    quant_module: nn.Module | None = None,
) -> list[dict[str, Any]]:
    device = _device(cfg)
    fp32_model.eval()
    quant_model.eval()
    rows: list[dict[str, Any]] = []

    for batch in eval_loader:
        sample_ids = _batch_sample_ids(batch)
        labels = _batch_labels(batch)
        group_metadata = _group_metadata_for_batch(cfg, sample_ids, labels, batch)
        buckets = group_metadata["bucket"]
        shot_group = group_metadata["shot_group"]
        if len(buckets) != len(sample_ids):
            raise RuntimeError(f"bucket count must equal batch size, got {len(buckets)} vs {len(sample_ids)}")
        fp32_embedding = core_reference.batch_embedding(sample_ids)
        token_padding_mask = core_reference.batch_padding_mask(sample_ids)
        moved = _move_batch_to_device(batch, device)

        fp32_live_embedding, _, fp32_records = _forward_with_token_trace(
            fp32_model,
            moved,
            cfg,
            trace_job=trace_job,
            sample_ids=sample_ids,
            token_padding_mask=token_padding_mask,
        )
        if tuple(fp32_live_embedding.shape) != tuple(fp32_embedding.shape):
            raise RuntimeError(f"FP32 live embedding shape differs from core_eval.pt: {tuple(fp32_live_embedding.shape)} vs {tuple(fp32_embedding.shape)}")
        if quant_module is not None and hasattr(quant_module, "clear_integer_grid_diagnostics"):
            quant_module.clear_integer_grid_diagnostics()
        quant_embedding, quant_mask, quant_records = _forward_with_token_trace(
            quant_model,
            moved,
            cfg,
            trace_job=trace_job,
            sample_ids=sample_ids,
            token_padding_mask=token_padding_mask,
        )
        direct_integer_mapping = _get_quantlinear_integer_grid(quant_module) if quant_module is not None else None
        direct_quant_activation_for_oer = _get_quantlinear_quant_activation_for_oer(quant_module)
        if not torch.equal(quant_mask, token_padding_mask):
            raise RuntimeError("FP32 and INT4 token padding_mask differ")

        rows.extend(
            _rows_for_batch(
                cfg=cfg,
                fp32_records=fp32_records,
                quant_records=quant_records,
                fp32_embedding=fp32_embedding,
                quant_embedding=quant_embedding,
                sample_ids=sample_ids,
                labels=labels,
                bucket=buckets,
                shot_group=shot_group,
                quant_mode=quant_mode,
                intervention_layer_idx=intervention_layer_idx,
                intervention_module_name=intervention_module_name,
                activation_qparams=activation_qparams,
                qparams_path=qparams_path,
                qparams_summary=qparams_summary,
                trace_job=trace_job,
                shard_key=shard_key,
                experiment_signature=experiment_signature,
                qparams_hash=qparams_hash,
                direct_integer_mapping=direct_integer_mapping,
                direct_quant_activation_for_oer=direct_quant_activation_for_oer,
                token_oracle_reference=token_oracle_reference,
            )
        )

        if quant_module is not None and hasattr(quant_module, "clear_integer_grid_diagnostics"):
            quant_module.clear_integer_grid_diagnostics()
        del fp32_records, quant_records, fp32_live_embedding, fp32_embedding, quant_embedding, moved
    return rows


def _run_streamed_token_eval_group(
    cfg: Any,
    *,
    fp32_model: nn.Module,
    quant_model: nn.Module,
    eval_loader: DataLoader,
    trace_entries: Sequence[PendingTraceJob],
    quant_mode: str,
    intervention_layer_idx: int,
    intervention_module_name: str,
    activation_qparams: Mapping[str, Any] | None,
    qparams_path: Path,
    qparams_summary: Mapping[str, Any],
    qparams_hash: str,
    core_reference: CoreEvalReference,
    token_oracle_reference: TokenTraceOracleReference | None,
    artifact_writers: Mapping[str, _ShardArtifactStreamWriter],
    quant_module: nn.Module | None = None,
) -> None:
    if not trace_entries:
        return
    device = _device(cfg)
    fp32_model.eval()
    quant_model.eval()
    group_trace_job = _merged_trace_job_for_group(trace_entries)

    for batch in eval_loader:
        sample_ids = _batch_sample_ids(batch)
        labels = _batch_labels(batch)
        group_metadata = _group_metadata_for_batch(cfg, sample_ids, labels, batch)
        buckets = group_metadata["bucket"]
        shot_group = group_metadata["shot_group"]
        if len(buckets) != len(sample_ids):
            raise RuntimeError(f"bucket count must equal batch size, got {len(buckets)} vs {len(sample_ids)}")
        fp32_embedding = core_reference.batch_embedding(sample_ids)
        token_padding_mask = core_reference.batch_padding_mask(sample_ids)
        moved = _move_batch_to_device(batch, device)

        fp32_live_embedding, _, fp32_records = _forward_with_token_trace(
            fp32_model,
            moved,
            cfg,
            trace_job=group_trace_job,
            sample_ids=sample_ids,
            token_padding_mask=token_padding_mask,
        )
        if tuple(fp32_live_embedding.shape) != tuple(fp32_embedding.shape):
            raise RuntimeError(f"FP32 live embedding shape differs from core_eval.pt: {tuple(fp32_live_embedding.shape)} vs {tuple(fp32_embedding.shape)}")
        if quant_module is not None and hasattr(quant_module, "clear_integer_grid_diagnostics"):
            quant_module.clear_integer_grid_diagnostics()
        quant_embedding, quant_mask, quant_records = _forward_with_token_trace(
            quant_model,
            moved,
            cfg,
            trace_job=group_trace_job,
            sample_ids=sample_ids,
            token_padding_mask=token_padding_mask,
        )
        direct_integer_mapping = _get_quantlinear_integer_grid(quant_module) if len(trace_entries) == 1 and quant_module is not None else None
        direct_quant_activation_for_oer = (
            _get_quantlinear_quant_activation_for_oer(quant_module)
            if len(trace_entries) == 1
            else None
        )
        if not torch.equal(quant_mask, token_padding_mask):
            raise RuntimeError("FP32 and INT4 token padding_mask differ")

        for trace_job, shard_key, signature, _paths in trace_entries:
            rows = _rows_for_batch(
                cfg=cfg,
                fp32_records=_records_for_trace_job(fp32_records, trace_job),
                quant_records=_records_for_trace_job(quant_records, trace_job),
                fp32_embedding=fp32_embedding,
                quant_embedding=quant_embedding,
                sample_ids=sample_ids,
                labels=labels,
                bucket=buckets,
                shot_group=shot_group,
                quant_mode=quant_mode,
                intervention_layer_idx=intervention_layer_idx,
                intervention_module_name=intervention_module_name,
                activation_qparams=activation_qparams,
                qparams_path=qparams_path,
                qparams_summary=qparams_summary,
                trace_job=trace_job,
                shard_key=shard_key,
                experiment_signature=signature,
                qparams_hash=qparams_hash,
                direct_integer_mapping=direct_integer_mapping if len(trace_entries) == 1 else None,
                direct_quant_activation_for_oer=direct_quant_activation_for_oer if len(trace_entries) == 1 else None,
                token_oracle_reference=token_oracle_reference,
            )
            artifact_writers[shard_key].append(rows)
            del rows

        if quant_module is not None and hasattr(quant_module, "clear_integer_grid_diagnostics"):
            quant_module.clear_integer_grid_diagnostics()
        del fp32_records, quant_records, fp32_live_embedding, fp32_embedding, quant_embedding, moved
    return


def run_single_intervention(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    calibration_loader: DataLoader,
    eval_loader: DataLoader,
    target_modules: Sequence[tuple[int, str]],
    layer_indices: Sequence[int],
    result_dir: Path,
    core_reference: CoreEvalReference,
    token_oracle_reference: TokenTraceOracleReference,
) -> list[Path]:
    qparams, qparams_path = _load_or_calibrate_qparams(
        cfg,
        result_dir=result_dir,
        layer_idx=int(layer_idx),
        module_name=module_name,
        calibration_loader=calibration_loader,
    )
    qparams_hash = _file_sha256(qparams_path)
    trace_jobs = _trace_jobs_for_intervention(
        cfg,
        layer_indices=layer_indices,
        available_targets=target_modules,
        intervention_layer_idx=int(layer_idx),
        intervention_module_name=str(module_name),
    )
    shard_csv_paths: list[Path] = []

    for quant_mode in _quant_modes(cfg):
        weight_qparams = qparams.get("weight") if _mode_uses_weight(quant_mode) else None
        activation_qparams = qparams.get("activation") if _mode_uses_activation(quant_mode) else None
        _require_qparams_for_quant_mode(
            quant_mode=quant_mode,
            weight_qparams=weight_qparams,
            activation_qparams=activation_qparams,
            target_name=_target_name(layer_idx, module_name),
        )
        qparams_summary = _qparams_summary(qparams, quant_mode)
        pending: list[PendingTraceJob] = []
        for trace_job in trace_jobs:
            shard_key = _shard_key(quant_mode, trace_job)
            signature = _shard_signature(
                cfg,
                quant_mode=quant_mode,
                trace_job=trace_job,
                intervention_layer_idx=int(layer_idx),
                intervention_module_name=module_name,
                qparams_hash=qparams_hash,
            )
            paths = _shard_paths(result_dir, shard_key)
            if _shard_completed(paths, signature=signature):
                shard_csv_paths.append(paths["sample_metrics"])
                continue
            pending.append((trace_job, shard_key, signature, paths))
        if not pending:
            continue

        device = _device(cfg)
        fp32_model = _restore_clean_model(cfg).to(device)
        quant_model = _restore_clean_model(cfg).to(device)
        quant_module = replace_ffn_module_with_quant(
            quant_model,
            int(layer_idx),
            module_name,
            _quant_cfg(cfg),
            quant_mode=quant_mode,
            weight_qparams=weight_qparams,
            activation_qparams=activation_qparams,
            require_calibrated_qparams=True,
            enable_fake_quant=False,
        )
        controller = make_single_ffn_quant_controller(
            quant_model,
            quant_module,
            layer_idx=int(layer_idx),
            module_name=module_name,
            quant_config=_quant_cfg(cfg),
            quant_mode=quant_mode,
        )
        try:
            controller.qparams_frozen = True
            controller.prepare_quant_forward()
            _assert_quant_module_has_fixed_qparams(quant_module, quant_mode, _target_name(layer_idx, module_name))
            for trace_entries in _pending_trace_groups(cfg, pending):
                trace_job = trace_entries[0][0]
                capture_grid = bool(
                    len(trace_entries) == 1
                    and _mode_uses_activation(quant_mode)
                    and trace_job.tensor_position == "input"
                    and int(trace_job.layer_idx) == int(layer_idx)
                    and str(trace_job.module_name) == str(module_name)
                )
                _set_quantlinear_integer_grid_capture(quant_module, cfg, enabled=capture_grid)
                with ExitStack() as stack:
                    artifact_writers: dict[str, _ShardArtifactStreamWriter] = {}
                    for trace_job, shard_key, signature, paths in trace_entries:
                        artifact_writers[shard_key] = stack.enter_context(
                            _ShardArtifactStreamWriter(
                                paths=paths,
                                signature=signature,
                                shard_key=shard_key,
                                trace_job=trace_job,
                                quant_mode=quant_mode,
                                qparams_hash=qparams_hash,
                            )
                        )
                    _run_streamed_token_eval_group(
                        cfg,
                        fp32_model=fp32_model,
                        quant_model=quant_model,
                        eval_loader=eval_loader,
                        trace_entries=trace_entries,
                        quant_mode=quant_mode,
                        intervention_layer_idx=int(layer_idx),
                        intervention_module_name=module_name,
                        activation_qparams=activation_qparams,
                        qparams_path=qparams_path,
                        qparams_summary=qparams_summary,
                        qparams_hash=qparams_hash,
                        core_reference=core_reference,
                        token_oracle_reference=token_oracle_reference,
                        artifact_writers=artifact_writers,
                        quant_module=quant_module,
                    )
                for _trace_job, _completed_shard_key, _signature, paths in trace_entries:
                    shard_csv_paths.append(paths["sample_metrics"])
        finally:
            _set_quantlinear_integer_grid_capture(quant_module, cfg, enabled=False)
            controller.remove_padding_mask_hooks()
            del fp32_model
            del quant_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return shard_csv_paths



def run_token_survival_probe(cfg: Any) -> dict[str, Any]:
    result_dir = _result_dir(cfg)
    result_dir.mkdir(parents=True, exist_ok=True)
    calibration_loader = _build_audio_loader(cfg, name="calibration", mode="calibration")
    eval_loader = _build_audio_loader(cfg, name="eval", mode="eval")

    probe_model = _restore_clean_model(cfg)
    layer_indices = _resolve_probe_layer_indices(cfg, probe_model)
    target_modules = tuple(_target_modules_for_probe(cfg, probe_model))
    num_transformer_layers = len(layer_indices)
    del probe_model
    if not target_modules:
        raise ValueError("Token Survival requires at least one FFN fc1/fc2 target module")

    core_reference = _load_core_eval_reference(cfg)
    required_positions: list[str] = []
    if bool(_setting_get(cfg, "RUN_BLOCK_OUTPUT_TRACE", True)):
        required_positions.append("block_output")
    if bool(_setting_get(cfg, "RUN_INPUT_ACTIVATION_TRACE", True)):
        required_positions.append("input_activation")
    if not required_positions:
        raise ValueError("Token Survival trace plan is empty; no oracle positions requested")
    token_oracle_reference = _load_token_trace_oracle_reference(
        cfg,
        core_reference=core_reference,
        layer_indices=layer_indices,
        required_positions=required_positions,
        required_input_targets=target_modules if "input_activation" in required_positions else (),
    )

    shard_csv_paths: list[Path] = []
    for layer_idx, module_name in target_modules:
        shard_csv_paths.extend(
            run_single_intervention(
                cfg,
                layer_idx=int(layer_idx),
                module_name=module_name,
                calibration_loader=calibration_loader,
                eval_loader=eval_loader,
                target_modules=target_modules,
                layer_indices=layer_indices,
                result_dir=result_dir,
                core_reference=core_reference,
                token_oracle_reference=token_oracle_reference,
            )
        )

    shard_csv_paths = list(dict.fromkeys(shard_csv_paths))
    metrics_csv = result_dir / CSV_FILENAME
    raw_sample_df: pd.DataFrame | None = None
    combined_sample_csv_written = False
    if not _streaming_summary_enabled(cfg):
        raise ValueError(
            "FFN token survival production finalization requires "
            "FFN_TOKEN_SURVIVAL_PROBE.STREAMING_SUMMARY=True; "
            "non-streaming all_rows/DataFrame summary finalization is disabled."
        )
    sample_df, summary_df, comparison_df, quant_mode_comparison_df = build_summary_table_from_shards(
        shard_csv_paths,
        cfg=cfg,
        output_dir=result_dir,
        save=True,
    )
    num_rows = int(summary_df.attrs.get("num_sample_rows", sample_df.attrs.get("num_sample_rows", 0)))
    if _write_combined_sample_csv(cfg):
        all_rows = _attach_quant_mode_sample_deltas(_read_shard_rows(shard_csv_paths))
        _write_rows(metrics_csv, all_rows)
        combined_sample_csv_written = True
        if _run_case_plots(cfg):
            raw_sample_df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
        del all_rows
    plot_paths = plot_all_token_survival_figures(
        summary_df,
        result_dir,
        raw_sample_df=raw_sample_df,
        comparison_df=comparison_df,
        quant_mode_comparison_df=quant_mode_comparison_df,
    )
    summary_csv = result_dir / SUMMARY_FILENAME
    detail_summary_csv = result_dir / DETAIL_SUMMARY_FILENAME
    comparison_summary_csv = result_dir / COMPARISON_SUMMARY_FILENAME
    detail_comparison_summary_csv = result_dir / DETAIL_COMPARISON_SUMMARY_FILENAME
    diagnostic_comparison_summary_csv = result_dir / DIAGNOSTIC_COMPARISON_SUMMARY_FILENAME
    quant_mode_comparison_csv = result_dir / QUANT_MODE_COMPARISON_FILENAME
    detail_quant_mode_comparison_csv = result_dir / DETAIL_QUANT_MODE_COMPARISON_FILENAME
    diagnostic_quant_mode_comparison_csv = result_dir / DIAGNOSTIC_QUANT_MODE_COMPARISON_FILENAME
    intervention_level_summary_csv = result_dir / INTERVENTION_LEVEL_SUMMARY_FILENAME
    intervention_level_comparison_csv = result_dir / INTERVENTION_LEVEL_COMPARISON_FILENAME
    manifest_path = result_dir / MANIFEST_FILENAME
    extra_tables = summary_df.attrs.get("extra_tables", {})
    if not isinstance(extra_tables, Mapping):
        extra_tables = {}
    manifest = _build_manifest(
        cfg=cfg,
        result_dir=result_dir,
        metrics_csv=metrics_csv,
        summary_csv=summary_csv,
        comparison_summary_csv=comparison_summary_csv,
        quant_mode_comparison_csv=quant_mode_comparison_csv,
        manifest_path=manifest_path,
        plot_paths=plot_paths,
        sample_table=sample_df,
        summary_df=summary_df,
        comparison_df=comparison_df,
        quant_mode_comparison_df=quant_mode_comparison_df,
        target_modules=target_modules,
        num_transformer_layers=num_transformer_layers,
        shard_csv_paths=shard_csv_paths,
        num_rows=int(num_rows),
        core_reference=core_reference,
        token_oracle_reference=token_oracle_reference,
        combined_sample_csv_written=combined_sample_csv_written,
    )
    _write_manifest(manifest_path, manifest)
    return {
        "metrics_csv": metrics_csv,
        "summary_csv": summary_csv,
        "detail_summary_csv": detail_summary_csv,
        "comparison_summary_csv": comparison_summary_csv,
        "detail_comparison_summary_csv": detail_comparison_summary_csv,
        "diagnostic_comparison_summary_csv": diagnostic_comparison_summary_csv,
        "quant_mode_comparison_csv": quant_mode_comparison_csv,
        "detail_quant_mode_comparison_csv": detail_quant_mode_comparison_csv,
        "diagnostic_quant_mode_comparison_csv": diagnostic_quant_mode_comparison_csv,
        "intervention_level_summary_csv": intervention_level_summary_csv,
        "intervention_level_comparison_csv": intervention_level_comparison_csv,
        "manifest_path": manifest_path,
        "summary": summary_df,
        "detail_summary": extra_tables.get("detail_summary", pd.DataFrame()),
        "comparison_summary": comparison_df,
        "detail_comparison_summary": extra_tables.get("detail_comparison", pd.DataFrame()),
        "diagnostic_comparison_summary": extra_tables.get("diagnostic_comparison", pd.DataFrame()),
        "quant_mode_comparison": quant_mode_comparison_df,
        "detail_quant_mode_comparison": extra_tables.get("detail_quant_mode_comparison", pd.DataFrame()),
        "diagnostic_quant_mode_comparison": extra_tables.get("diagnostic_quant_mode_comparison", pd.DataFrame()),
        "intervention_level_summary": extra_tables.get("intervention_level_summary", pd.DataFrame()),
        "intervention_level_comparison": extra_tables.get("intervention_level_comparison", pd.DataFrame()),
        "plot_paths": plot_paths,
        "num_rows": int(num_rows),
        "combined_sample_csv_written": bool(combined_sample_csv_written),
        "num_summary_rows": int(len(summary_df)),
        "num_detail_summary_rows": int(len(extra_tables.get("detail_summary", pd.DataFrame()))),
        "num_comparison_rows": int(len(comparison_df)),
        "num_quant_mode_comparison_rows": int(len(quant_mode_comparison_df)),
        "target_modules": target_modules,
        "num_transformer_layers": num_transformer_layers,
        "shard_csv_paths": shard_csv_paths,
    }


def main(cfg: Any) -> dict[str, Any]:
    seed_everything = _cfg_get(cfg, "seed_everything", None)
    if callable(seed_everything):
        seed_everything(_seed(cfg))
    return run_token_survival_probe(cfg)


if __name__ == "__main__":
    import config as project_config

    summary = main(project_config)
    primary_output = summary["metrics_csv"] if summary.get("combined_sample_csv_written") else summary["summary_csv"]
    print(f"FFN Token Survival Probe complete: {primary_output}")


__all__ = [
    "CSV_FILENAME",
    "QUANT_MODES",
    "run_single_intervention",
    "run_token_survival_probe",
    "main",
]


