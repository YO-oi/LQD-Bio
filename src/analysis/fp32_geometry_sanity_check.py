from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor

try:
    from src.export.export_fp32_core_cache import (
        is_sharded_fp32_core_cache,
        iter_fp32_core_cache_shards,
        load_fp32_core_cache,
        CACHE_FILENAME as CORE_CACHE_FILENAME,
        EXPECTED_EMBEDDING_DIM,
        _cfg_get,
        _dataset_cfg,
        _default_cfg,
        _nested_get,
        _restore_fp32_teacher,
    )
    from src.metrics.common_metrics import fisher_ratio
    from src.analysis.build_fp32_oer_oracle import is_sharded_fp32_oer_oracle, iter_fp32_oer_oracle_shards
    from src.metrics.geometry_metrics import class_trace_variance, compute_group_fisher, compute_group_margin
    from src.utils.token_trace_hook import FFN_MODULE_NAMES, FULL_LAYER_INDICES, get_ffn_module, resolve_full_layer_indices
    from src.utils.group_utils import (
        ANALYSIS_GROUPS,
        BASE_BUCKETS,
        GroupValidationError,
        analysis_group_mask,
        tail_included,
        validate_bucket_values,
    )
except ImportError:  # pragma: no cover - package-relative fallback.
    from ..export.export_fp32_core_cache import (
        is_sharded_fp32_core_cache,
        iter_fp32_core_cache_shards,
        load_fp32_core_cache,
        CACHE_FILENAME as CORE_CACHE_FILENAME,
        EXPECTED_EMBEDDING_DIM,
        _cfg_get,
        _dataset_cfg,
        _default_cfg,
        _nested_get,
        _restore_fp32_teacher,
    )
    from ..metrics.common_metrics import fisher_ratio
    from .build_fp32_oer_oracle import is_sharded_fp32_oer_oracle, iter_fp32_oer_oracle_shards
    from ..metrics.geometry_metrics import class_trace_variance, compute_group_fisher, compute_group_margin
    from ..utils.token_trace_hook import FFN_MODULE_NAMES, FULL_LAYER_INDICES, get_ffn_module, resolve_full_layer_indices
    from ..utils.group_utils import (
        ANALYSIS_GROUPS,
        BASE_BUCKETS,
        GroupValidationError,
        analysis_group_mask,
        tail_included,
        validate_bucket_values,
    )


REPORT_VERSION = 1
REPORT_DIRNAME = "fp32_geometry_sanity"
REPORT_FILENAME = "fp32_geometry_sanity_report.json"
SHOT_GROUPS = BASE_BUCKETS

DEFAULT_THRESHOLDS = {
    "MIN_EFFECTIVE_CLASSES": 2,
    "MIN_GEOMETRY_SAMPLES": 2,
    "MIN_FEW_SAMPLES": 5,
    "MIN_TAIL_SAMPLES": 10,
    "MIN_MARGIN_POSITIVE_RATIO": 0.5,
    "MIN_FISHER_RATIO": 1e-8,
    "TRACE_VARIANCE_LOW_EPS": 1e-12,
    "TRACE_VARIANCE_HIGH_P95": 1e6,
    "MAX_SHOT_GROUP_IMBALANCE_RATIO": 100.0,
}

EXPECTED_CORE_MANIFEST = {
    "cache_type": "fp32_core_cache",
    "stage": "quant_injection_pre_fp32_reference",
    "embedding_pooling": "mask_aware_mean",
}

DEFAULT_BLOCKING_WARNING_CODES = (
    "fisher_compute_failed",
    "fisher_not_finite",
    "fisher_negative",
    "trace_variance_all_nan",
    "margin_all_invalid",
    "tail_group_too_small",
)


class FP32GeometrySanityError(RuntimeError):
    """FP32 reference cache ???????????????????????????????"""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


def _sanity_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_GEOMETRY_SANITY", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    sanity_cfg = _sanity_cfg(cfg)
    if key in sanity_cfg:
        return sanity_cfg[key]
    return _cfg_get(cfg, key, default)


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
    raise AttributeError("????????????????????ACTIVE_DATASET ??CURRENT_DATASET['ROOT']")


def _core_cache_path(cfg: Any, core_cache_path: str | Path | None = None) -> Path:
    if core_cache_path is not None:
        return Path(core_cache_path)
    configured = _setting_get(cfg, "CORE_CACHE_PATH", None)
    if configured is not None:
        return Path(configured)
    fp32_core_cfg = _cfg_get(cfg, "FP32_CORE_CACHE", {})
    if isinstance(fp32_core_cfg, Mapping):
        for key in ("CACHE_PATH", "CORE_CACHE_PATH"):
            value = fp32_core_cfg.get(key)
            if value is not None:
                return Path(value)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "fp32_core_cache" / CORE_CACHE_FILENAME


def _report_path(cfg: Any, report_path: str | Path | None = None) -> Path:
    if report_path is not None:
        return Path(report_path)
    configured = _setting_get(cfg, "REPORT_PATH", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / REPORT_DIRNAME / REPORT_FILENAME


def _eval_csv_path(cfg: Any) -> Path | None:
    configured = _setting_get(cfg, "EVAL_CSV", None)
    if configured is not None:
        return Path(configured)
    dataset_value = _nested_get(_dataset_cfg(cfg), "EVAL_CSV", None)
    if dataset_value is not None:
        return Path(dataset_value)
    return None


def _thresholds(cfg: Any) -> dict[str, float | int]:
    values: dict[str, float | int] = dict(DEFAULT_THRESHOLDS)
    configured = _setting_get(cfg, "THRESHOLDS", {})
    if isinstance(configured, Mapping):
        for key, value in configured.items():
            if key in values:
                default = values[key]
                values[key] = int(value) if isinstance(default, int) and not isinstance(default, bool) else float(value)
    return values


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    severity: str = "error",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "details": dict(details or {}),
        }
    )


def _check_metric_interface_compatibility() -> tuple[dict[str, str], list[dict[str, Any]]]:
    """???????????????????????????????????????????"""

    issues: list[dict[str, Any]] = []
    requirements = {
        "class_trace_variance": (
            class_trace_variance,
            {"embeddings", "label_id", "normalization", "reduction", "return_details"},
        ),
        "compute_group_margin": (
            compute_group_margin,
            {"embeddings", "label_id", "group", "reduction"},
        ),
    }
    signatures: dict[str, str] = {}
    for name, (fn, required_params) in requirements.items():
        signature = inspect.signature(fn)
        signatures[name] = str(signature)
        available = set(signature.parameters)
        missing = sorted(required_params - available)
        if missing:
            _add_issue(
                issues,
                code="metric_interface_incompatible",
                message="validation issue",
                details={"function": name, "signature": str(signature), "missing_params": missing},
            )
    return signatures, issues


def _validate_core_cache_manifest(cache: Mapping[str, Any], sample_ids: Sequence[str], hard_errors: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = cache.get("manifest")
    if not isinstance(manifest, Mapping):
        _add_issue(
            hard_errors,
            code="missing_core_cache_manifest",
            message="validation issue",
        )
        return {"present": False}

    summary = {"present": True}
    for key, expected in EXPECTED_CORE_MANIFEST.items():
        actual = manifest.get(key)
        summary[key] = actual
        if actual != expected:
            _add_issue(
                hard_errors,
                code="core_cache_manifest_mismatch",
                message="validation issue",
                details={"key": key, "expected": expected, "actual": actual},
            )
    actual_order_hash = _sample_order_hash(sample_ids)
    summary["sample_order_hash"] = actual_order_hash
    cache_order_hash = cache.get("sample_order_hash")
    if cache_order_hash is not None:
        summary["top_level_sample_order_hash"] = str(cache_order_hash)
        if str(cache_order_hash) != actual_order_hash:
            _add_issue(
                hard_errors,
                code="core_cache_sample_order_hash_mismatch",
                message="validation issue",
                details={"top_level_sample_order_hash": str(cache_order_hash), "actual_sample_order_hash": actual_order_hash},
            )
    else:
        summary["top_level_sample_order_hash"] = None
    manifest_order_hash = manifest.get("sample_order_hash")
    if manifest_order_hash is not None:
        summary["manifest_sample_order_hash"] = str(manifest_order_hash)
        if str(manifest_order_hash) != actual_order_hash:
            _add_issue(
                hard_errors,
                code="core_cache_manifest_sample_order_hash_mismatch",
                message="validation issue",
                details={"manifest_sample_order_hash": str(manifest_order_hash), "actual_sample_order_hash": actual_order_hash},
            )
    else:
        summary["manifest_sample_order_hash"] = None
    for key in ("eval_csv", "checkpoint_path", "num_samples", "num_classes", "embedding_dim", "sample_order_hash_algorithm"):
        if key in manifest:
            summary[key] = manifest.get(key)
    return summary


def _blocking_warning_codes(cfg: Any) -> set[str]:
    configured = _setting_get(cfg, "BLOCKING_WARNING_CODES", DEFAULT_BLOCKING_WARNING_CODES)
    if configured is None:
        return set()
    if isinstance(configured, str):
        text = configured.strip().lower()
        if text in {"", "none", "off", "false"}:
            return set()
        if text in {"all", "*"}:
            return {"*"}
        return {item.strip() for item in configured.split(",") if item.strip()}
    return {str(item) for item in configured}


def _select_blocking_warnings(warnings: Sequence[Mapping[str, Any]], blocking_codes: set[str]) -> list[dict[str, Any]]:
    if not blocking_codes:
        return []
    if "*" in blocking_codes:
        return [dict(item) for item in warnings]
    return [dict(item) for item in warnings if str(item.get("code")) in blocking_codes]


def _oer_oracle_path(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OER_ORACLE_PATH", None)
    if configured is not None:
        return Path(configured)
    oracle_cfg = _cfg_get(cfg, "FP32_OER_ORACLE", {})
    if isinstance(oracle_cfg, Mapping):
        for key in ("CACHE_PATH", "ORACLE_PATH"):
            value = oracle_cfg.get(key)
            if value is not None:
                return Path(value)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "fp32_oer_oracle" / "fp32_oer_oracle.pt"


def _normalise_oer_target_manifest(value: Any) -> list[tuple[int, str]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        iterable = value.values()
    else:
        iterable = value
    result: list[tuple[int, str]] = []
    for item in iterable:
        if isinstance(item, Mapping):
            if "layer_idx" not in item or "module_name" not in item:
                raise KeyError("target_modules mapping must include layer_idx and module_name")
            layer_idx = item["layer_idx"]
            module_name = item["module_name"]
        else:
            try:
                layer_idx, module_name = item
            except (TypeError, ValueError) as exc:
                raise TypeError(f"target_modules item must be (layer_idx, module_name), got {item!r}") from exc
        module_name = str(module_name)
        if module_name not in FFN_MODULE_NAMES:
            raise ValueError(f"module_name must be one of {FFN_MODULE_NAMES!r}, got {module_name!r}")
        result.append((int(layer_idx), module_name))
    return result


def _expected_oer_targets_from_manifest(
    manifest: Mapping[str, Any],
    hard_errors: list[dict[str, Any]],
) -> tuple[set[tuple[int, str]], str]:
    """Return the required full-FFN oracle target set and audit manifest coverage."""

    raw_targets = manifest.get("target_modules")
    manifest_targets: list[tuple[int, str]] | None = None
    if raw_targets is not None:
        try:
            manifest_targets = _normalise_oer_target_manifest(raw_targets)
        except Exception as exc:  # noqa: BLE001 - manifest audit should report the exact invalid payload.
            _add_issue(
                hard_errors,
                code="oer_oracle_target_manifest_invalid",
                message="validation issue",
                details={"error": str(exc), "target_modules_type": repr(type(raw_targets))},
            )
            manifest_targets = []
        duplicates = sorted({target for target in manifest_targets if manifest_targets.count(target) > 1})
        if duplicates:
            _add_issue(
                hard_errors,
                code="oer_oracle_target_manifest_duplicates",
                message="validation issue",
                details={"duplicates": duplicates[:24]},
            )

    num_layers = manifest.get("num_layers")
    if num_layers is not None:
        try:
            count = int(num_layers)
        except (TypeError, ValueError) as exc:
            _add_issue(
                hard_errors,
                code="oer_oracle_num_layers_invalid",
                message="validation issue",
                details={"num_layers": num_layers, "error": str(exc)},
            )
            count = 0
        if count <= 0:
            _add_issue(
                hard_errors,
                code="oer_oracle_num_layers_invalid",
                message="validation issue",
                details={"num_layers": num_layers},
            )
            return set(), "oracle_manifest.num_layers_invalid"
        expected = {(layer_idx, module_name) for layer_idx in range(count) for module_name in FFN_MODULE_NAMES}
        expected_source = "oracle_manifest.num_layers_full_ffn"
    elif FULL_LAYER_INDICES:
        expected = {(int(layer_idx), module_name) for layer_idx in FULL_LAYER_INDICES for module_name in FFN_MODULE_NAMES}
        expected_source = "token_trace_hook.FULL_LAYER_INDICES_full_ffn"
    else:
        _add_issue(
            hard_errors,
            code="oer_oracle_expected_targets_unresolved",
            message="validation issue",
        )
        return set(), "unresolved"

    if manifest_targets is not None:
        manifest_target_set = set(manifest_targets)
        if manifest_target_set != expected:
            _add_issue(
                hard_errors,
                code="oer_oracle_manifest_target_not_full_ffn",
                message="validation issue",
                details={
                    "expected_target_source": expected_source,
                    "manifest_target_count": len(manifest_target_set),
                    "expected_target_count": len(expected),
                    "missing_targets": sorted(expected - manifest_target_set)[:24],
                    "extra_targets": sorted(manifest_target_set - expected)[:24],
                },
            )
    return expected, expected_source
def _check_sharded_oer_oracle_cache(
    *,
    oracle: Mapping[str, Any],
    oracle_path: Path,
    manifest: Mapping[str, Any],
    expected_sample_ids: Sequence[str],
    summary: dict[str, Any],
    hard_errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row_count = 0
    oracle_sample_set: set[str] = set()
    actual_targets: set[tuple[int, str]] = set()
    positions: set[str] = set()
    keys: set[tuple[str, int, str, str]] = set()
    duplicates: list[str] = []
    required_fields = (
        "sample_id",
        "layer_idx",
        "module_name",
        "activation_position",
        "topk_scalar_indices",
        "oer_topk_token_indices_aux",
        "fp32_oracle_energy",
        "fp32_total_energy",
        "topk_ratio",
        "activation_shape",
        "valid_token_count",
    )
    for shard in iter_fp32_oer_oracle_shards(oracle, oracle_path):
        missing_fields = [field for field in required_fields if field not in shard]
        shard_manifest = shard.get("manifest", {})
        if missing_fields:
            _add_issue(
                hard_errors,
                code="oer_oracle_missing_fields",
                message="validation issue",
                details={"missing_fields": missing_fields, "shard_manifest": dict(shard_manifest) if isinstance(shard_manifest, Mapping) else {}},
            )
            continue
        sample_ids = _normalise_sample_ids(shard.get("sample_id"))
        shard_rows = len(sample_ids)
        layer_idx = torch.as_tensor(shard["layer_idx"]).reshape(-1).long()
        module_names = [str(item) for item in shard["module_name"]]
        shard_positions = [str(item) for item in shard["activation_position"]]
        field_lengths = {
            "layer_idx": int(layer_idx.numel()),
            "module_name": len(module_names),
            "activation_position": len(shard_positions),
            "topk_scalar_indices": len(shard["topk_scalar_indices"]),
            "oer_topk_token_indices_aux": len(shard["oer_topk_token_indices_aux"]),
            "fp32_oracle_energy": int(torch.as_tensor(shard["fp32_oracle_energy"]).numel()),
            "fp32_total_energy": int(torch.as_tensor(shard["fp32_total_energy"]).numel()),
            "activation_shape": len(shard["activation_shape"]),
            "valid_token_count": int(torch.as_tensor(shard["valid_token_count"]).numel()),
        }
        bad_lengths = {key: value for key, value in field_lengths.items() if value != shard_rows}
        if bad_lengths:
            _add_issue(
                hard_errors,
                code="oer_oracle_field_length_mismatch",
                message="validation issue",
                details={"row_count": shard_rows, "bad_lengths": bad_lengths},
            )
            continue
        oracle_sample_set.update(sample_ids)
        row_count += shard_rows
        for idx, sample_id in enumerate(sample_ids):
            layer_value = int(layer_idx[idx].item())
            module_name = module_names[idx]
            position = shard_positions[idx]
            actual_targets.add((layer_value, module_name))
            positions.add(position)
            key = (sample_id, layer_value, module_name, position)
            if key in keys:
                duplicates.append(f"{key[0]}|L{key[1]}|{key[2]}|{key[3]}")
            keys.add(key)
        for energy_field in ("fp32_oracle_energy", "fp32_total_energy"):
            energy = torch.as_tensor(shard[energy_field]).detach().float().reshape(-1)
            if not torch.isfinite(energy).all().item() or bool((energy < 0).any().item()):
                _add_issue(
                    hard_errors,
                    code="oer_oracle_energy_invalid",
                    message="validation issue",
                    details={"field": energy_field, "shard_manifest": dict(shard_manifest) if isinstance(shard_manifest, Mapping) else {}},
                )

    summary.update(
        {
            "status": "checked",
            "storage_format": str(manifest.get("storage_format")),
            "row_count": int(row_count),
            "unique_samples": len(oracle_sample_set),
            "activation_positions": sorted(positions),
            "num_shards": len(oracle.get("shards", []) or []),
        }
    )
    expected_sample_set = set(str(item) for item in expected_sample_ids)
    if expected_sample_set and oracle_sample_set != expected_sample_set:
        _add_issue(
            hard_errors,
            code="oer_oracle_sample_id_mismatch",
            message="validation issue",
            details={
                "missing_in_oracle": sorted(expected_sample_set - oracle_sample_set)[:16],
                "extra_in_oracle": sorted(oracle_sample_set - expected_sample_set)[:16],
            },
        )
    expected_targets, expected_target_source = _expected_oer_targets_from_manifest(manifest, hard_errors)
    summary["actual_target_count"] = int(len(actual_targets))
    summary["expected_target_count"] = int(len(expected_targets))
    summary["expected_target_source"] = expected_target_source
    if expected_targets and actual_targets != expected_targets:
        _add_issue(
            hard_errors,
            code="oer_oracle_target_incomplete",
            message="validation issue",
            details={
                "expected_target_source": expected_target_source,
                "missing_targets": sorted(expected_targets - actual_targets)[:24],
                "extra_targets": sorted(actual_targets - expected_targets)[:24],
            },
        )
    if duplicates:
        _add_issue(
            hard_errors,
            code="oer_oracle_duplicate_rows",
            message="validation issue",
            details={"duplicates": duplicates[:16]},
        )
    required_positions = {"input", "output"}
    manifest_position_set = {str(item) for item in manifest.get("activation_positions", [])}
    if not required_positions.issubset(positions):
        _add_issue(
            hard_errors,
            code="oer_oracle_activation_positions_incomplete",
            message="validation issue",
            details={"required_positions": sorted(required_positions), "actual_positions": sorted(positions)},
        )
    if not required_positions.issubset(manifest_position_set):
        _add_issue(
            hard_errors,
            code="oer_oracle_manifest_activation_positions_incomplete",
            message="validation issue",
            details={"required_positions": sorted(required_positions), "manifest_positions": sorted(manifest_position_set)},
        )
    expected_positions = list(manifest.get("activation_positions", sorted(positions) or ["input"]))
    expected_rows = len(expected_sample_set or oracle_sample_set) * len(expected_targets) * len(expected_positions)
    summary["expected_row_count"] = int(expected_rows)
    if row_count != expected_rows:
        _add_issue(
            hard_errors,
            code="oer_oracle_row_count_mismatch",
            message="validation issue",
            details={"row_count": row_count, "expected_row_count": expected_rows},
        )
    return summary, hard_errors

def _check_oer_oracle_cache(
    cfg: Any,
    *,
    expected_sample_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """???????? FP32 OER oracle cache ????????????????????????????????"""

    hard_errors: list[dict[str, Any]] = []
    path = _oer_oracle_path(cfg)
    required = bool(_setting_get(cfg, "CHECK_OER_ORACLE_CACHE", False))
    summary: dict[str, Any] = {
        "checked": False,
        "required": required,
        "path": str(path.expanduser().resolve()),
        "status": "not_found_skipped",
    }
    if not path.exists():
        if required:
            _add_issue(
                hard_errors,
                code="missing_oer_oracle_cache",
                message="validation issue",
                details={"path": str(path)},
            )
            summary["status"] = "missing"
        return summary, hard_errors

    oracle = torch.load(path, map_location="cpu")
    if not isinstance(oracle, Mapping):
        _add_issue(
            hard_errors,
            code="invalid_oer_oracle_cache_type",
            message="validation issue",
            details={"type": repr(type(oracle))},
        )
        summary["checked"] = True
        summary["status"] = "invalid"
        return summary, hard_errors

    summary["checked"] = True
    summary["status"] = "checked"
    manifest = oracle.get("manifest")
    if not isinstance(manifest, Mapping):
        _add_issue(hard_errors, code="missing_oer_oracle_manifest", message="OER oracle cache ??? manifest")
        manifest = {}
    expected_manifest = {
        "cache_type": "fp32_oer_oracle",
        "oracle_source": "fp32_real_activation",
        "stage": "quant_injection_pre_fp32_oer_oracle",
    }
    for key, expected in expected_manifest.items():
        actual = manifest.get(key)
        summary[key] = actual
        if actual != expected:
            _add_issue(
                hard_errors,
                code="oer_oracle_manifest_mismatch",
                message="validation issue",
                details={"key": key, "expected": expected, "actual": actual},
            )

    if is_sharded_fp32_oer_oracle(oracle):
        return _check_sharded_oer_oracle_cache(
            oracle=oracle,
            oracle_path=path,
            manifest=manifest,
            expected_sample_ids=expected_sample_ids,
            summary=summary,
            hard_errors=hard_errors,
        )
    required_fields = (
        "sample_id",
        "layer_idx",
        "module_name",
        "activation_position",
        "topk_scalar_indices",
        "oer_topk_token_indices_aux",
        "fp32_oracle_energy",
        "fp32_total_energy",
        "topk_ratio",
        "activation_shape",
        "valid_token_count",
    )
    missing_fields = [field for field in required_fields if field not in oracle]
    if missing_fields:
        _add_issue(
            hard_errors,
            code="oer_oracle_missing_fields",
            message="validation issue",
            details={"missing_fields": missing_fields},
        )
        summary["status"] = "invalid"
        return summary, hard_errors

    sample_ids = _normalise_sample_ids(oracle.get("sample_id"))
    row_count = len(sample_ids)
    layer_idx = torch.as_tensor(oracle["layer_idx"]).reshape(-1).long()
    module_names = [str(item) for item in oracle["module_name"]]
    positions = [str(item) for item in oracle["activation_position"]]
    summary.update(
        {
            "row_count": row_count,
            "unique_samples": len(set(sample_ids)),
            "activation_positions": sorted(set(positions)),
        }
    )
    field_lengths = {
        "layer_idx": int(layer_idx.numel()),
        "module_name": len(module_names),
        "activation_position": len(positions),
        "topk_scalar_indices": len(oracle["topk_scalar_indices"]),
        "oer_topk_token_indices_aux": len(oracle["oer_topk_token_indices_aux"]),
        "fp32_oracle_energy": int(torch.as_tensor(oracle["fp32_oracle_energy"]).numel()),
        "fp32_total_energy": int(torch.as_tensor(oracle["fp32_total_energy"]).numel()),
        "activation_shape": len(oracle["activation_shape"]),
        "valid_token_count": int(torch.as_tensor(oracle["valid_token_count"]).numel()),
    }
    bad_lengths = {key: value for key, value in field_lengths.items() if value != row_count}
    if bad_lengths:
        _add_issue(
            hard_errors,
            code="oer_oracle_field_length_mismatch",
            message="validation issue",
            details={"row_count": row_count, "bad_lengths": bad_lengths},
        )
        summary["status"] = "invalid"
        return summary, hard_errors

    expected_sample_set = set(str(item) for item in expected_sample_ids)
    oracle_sample_set = set(sample_ids)
    if expected_sample_set and oracle_sample_set != expected_sample_set:
        _add_issue(
            hard_errors,
            code="oer_oracle_sample_id_mismatch",
            message="validation issue",
            details={
                "missing_in_oracle": sorted(expected_sample_set - oracle_sample_set)[:16],
                "extra_in_oracle": sorted(oracle_sample_set - expected_sample_set)[:16],
            },
        )

    actual_targets = {(int(layer_idx[idx].item()), module_names[idx]) for idx in range(row_count)}
    expected_targets, expected_target_source = _expected_oer_targets_from_manifest(manifest, hard_errors)
    summary["actual_target_count"] = int(len(actual_targets))
    summary["expected_target_count"] = int(len(expected_targets))
    summary["expected_target_source"] = expected_target_source
    if expected_targets and actual_targets != expected_targets:
        _add_issue(
            hard_errors,
            code="oer_oracle_target_incomplete",
            message="validation issue",
            details={
                "expected_target_source": expected_target_source,
                "missing_targets": sorted(expected_targets - actual_targets)[:24],
                "extra_targets": sorted(actual_targets - expected_targets)[:24],
            },
        )

    keys: set[tuple[str, int, str, str]] = set()
    duplicates: list[str] = []
    for idx, sample_id in enumerate(sample_ids):
        key = (sample_id, int(layer_idx[idx].item()), module_names[idx], positions[idx])
        if key in keys:
            duplicates.append(f"{key[0]}|L{key[1]}|{key[2]}|{key[3]}")
        keys.add(key)
    if duplicates:
        _add_issue(
            hard_errors,
            code="oer_oracle_duplicate_rows",
            message="validation issue",
            details={"duplicates": duplicates[:16]},
        )

    required_positions = {"input", "output"}
    actual_position_set = {str(item) for item in positions}
    manifest_position_set = {str(item) for item in manifest.get("activation_positions", [])}
    if not required_positions.issubset(actual_position_set):
        _add_issue(
            hard_errors,
            code="oer_oracle_activation_positions_incomplete",
            message="validation issue",
            details={"required_positions": sorted(required_positions), "actual_positions": sorted(actual_position_set)},
        )
    if not required_positions.issubset(manifest_position_set):
        _add_issue(
            hard_errors,
            code="oer_oracle_manifest_activation_positions_incomplete",
            message="validation issue",
            details={"required_positions": sorted(required_positions), "manifest_positions": sorted(manifest_position_set)},
        )
    for energy_field in ("fp32_oracle_energy", "fp32_total_energy"):
        energy = torch.as_tensor(oracle[energy_field]).detach().float().reshape(-1)
        if not torch.isfinite(energy).all().item() or bool((energy < 0).any().item()):
            _add_issue(
                hard_errors,
                code="oer_oracle_energy_invalid",
                message="validation issue",
                details={"field": energy_field},
            )
    expected_positions = list(manifest.get("activation_positions", sorted(set(positions)) or ["input"]))
    expected_rows = len(expected_sample_set or oracle_sample_set) * len(expected_targets) * len(expected_positions)
    summary["expected_row_count"] = expected_rows
    if row_count != expected_rows:
        _add_issue(
            hard_errors,
            code="oer_oracle_row_count_mismatch",
            message="validation issue",
            details={"row_count": row_count, "expected_row_count": expected_rows},
        )
    return summary, hard_errors


def _as_tensor(cache: Mapping[str, Any], key: str, *, dtype: torch.dtype | None = None) -> Tensor | None:
    if key not in cache:
        return None
    value = cache[key]
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _normalise_sample_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Tensor):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    """??sample_id ????????? hash???????????????????????????"""

    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_core_cache(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cache = load_fp32_core_cache(path, materialize=False)
    cache["__cache_path__"] = path
    return cache

def _load_eval_frame(path: Path | None) -> Any | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"eval.csv ????? {path}")
    import pandas as pd

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"eval.csv ???: {path}")
    return frame


def _label_mapping_num_classes(label_mapping: Any) -> int | None:
    if not isinstance(label_mapping, Mapping) or not label_mapping:
        return None
    ids: list[int] = []
    for value in label_mapping.values():
        if isinstance(value, bool):
            return None
        ids.append(int(value))
    if len(set(ids)) != len(ids):
        return None
    if sorted(ids) != list(range(len(ids))):
        return None
    return len(ids)


def _predictor_out_features(model: nn.Module) -> int | None:
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return None
    if isinstance(predictor, nn.Linear):
        return int(predictor.out_features)
    if isinstance(predictor, nn.Sequential):
        for module in reversed(list(predictor.modules())):
            if isinstance(module, nn.Linear):
                return int(module.out_features)
    for module in predictor.modules() if isinstance(predictor, nn.Module) else []:
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    return None


def _check_oer_targets(model: nn.Module, hard_errors: list[dict[str, Any]]) -> int:
    missing: list[str] = []
    wrong_type: list[str] = []
    layer_indices = resolve_full_layer_indices(model)
    for layer_idx in layer_indices:
        for module_name in FFN_MODULE_NAMES:
            target = f"encoder.layers.{int(layer_idx)}.{module_name}"
            try:
                module = get_ffn_module(model, int(layer_idx), module_name)
            except Exception as exc:  # noqa: BLE001 - report ?????????????                missing.append(f"{target}: {exc}")
                continue
            if not isinstance(module, nn.Linear):
                wrong_type.append(f"{target}: {type(module)!r}")
    if missing:
        _add_issue(
            hard_errors,
            code="oer_target_missing",
            message="validation issue",
            details={"missing": missing[:24]},
        )
    if wrong_type:
        _add_issue(
            hard_errors,
            code="oer_target_wrong_type",
            message="validation issue",
            details={"wrong_type": wrong_type[:24]},
        )
    return len(layer_indices) * len(FFN_MODULE_NAMES)


def _check_eval_csv(
    *,
    frame: Any | None,
    sample_ids: Sequence[str],
    label_id: Tensor,
    hard_errors: list[dict[str, Any]],
) -> None:
    if frame is None:
        return
    required = {"sample_id", "label_id"}
    missing_columns = sorted(required - set(str(column) for column in frame.columns))
    if missing_columns:
        _add_issue(
            hard_errors,
            code="eval_csv_missing_required_columns",
            message="validation issue",
            details={"missing_columns": missing_columns},
        )
        return
    eval_sample_ids = frame["sample_id"].astype(str).tolist()
    duplicates = sorted({item for item in eval_sample_ids if eval_sample_ids.count(item) > 1})
    if duplicates:
        _add_issue(
            hard_errors,
            code="eval_csv_duplicate_sample_id",
            message="validation issue",
            details={"duplicates": duplicates[:16]},
        )
    cache_set = set(sample_ids)
    eval_set = set(eval_sample_ids)
    if cache_set != eval_set:
        _add_issue(
            hard_errors,
            code="eval_csv_core_cache_sample_mismatch",
            message="validation issue",
            details={
                "missing_in_cache": sorted(eval_set - cache_set)[:16],
                "missing_in_eval_csv": sorted(cache_set - eval_set)[:16],
            },
        )
    elif list(sample_ids) != eval_sample_ids:
        first_mismatches = []
        for idx, (cache_id, eval_id) in enumerate(zip(sample_ids, eval_sample_ids)):
            if str(cache_id) != str(eval_id):
                first_mismatches.append({"index": int(idx), "core_cache": str(cache_id), "eval_csv": str(eval_id)})
            if len(first_mismatches) >= 16:
                break
        _add_issue(
            hard_errors,
            code="eval_csv_core_cache_order_mismatch",
            message="validation issue",
            details={
                "core_cache_sample_order_hash": _sample_order_hash(sample_ids),
                "eval_csv_sample_order_hash": _sample_order_hash(eval_sample_ids),
                "first_mismatches": first_mismatches,
            },
        )
    label_map = {str(row["sample_id"]): int(row["label_id"]) for _, row in frame.iterrows()}
    mismatch: list[str] = []
    for idx, sample_id in enumerate(sample_ids):
        if sample_id in label_map and int(label_id[idx].item()) != int(label_map[sample_id]):
            mismatch.append(f"{sample_id}: cache={int(label_id[idx].item())}, eval_csv={int(label_map[sample_id])}")
    if mismatch:
        _add_issue(
            hard_errors,
            code="eval_csv_label_mismatch",
            message="validation issue",
            details={"mismatch": mismatch[:16]},
        )


def _validate_sharded_core_cache_structure(
    cache: Mapping[str, Any],
    *,
    expected_embedding_dim: int,
    eval_frame: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hard_errors: list[dict[str, Any]] = []
    cache_path = Path(cache.get("__cache_path__", ""))
    sample_ids = _normalise_sample_ids(cache.get("sample_id"))
    if not sample_ids:
        _add_issue(hard_errors, code="missing_sample_id", message="core cache ??? sample_id")
    duplicates = sorted({item for item in sample_ids if sample_ids.count(item) > 1})
    if duplicates:
        _add_issue(
            hard_errors,
            code="duplicate_sample_id",
            message="validation issue",
            details={"duplicates": duplicates[:16]},
        )

    label_id = _as_tensor(cache, "label_id", dtype=torch.long)
    valid_token_count = _as_tensor(cache, "valid_token_count", dtype=torch.long)
    if label_id is None:
        _add_issue(hard_errors, code="missing_label_id", message="sharded core cache manifest ??? label_id")
    if valid_token_count is None:
        _add_issue(hard_errors, code="missing_valid_token_count", message="sharded core cache manifest ??? valid_token_count")
    if hard_errors:
        return {}, hard_errors
    assert label_id is not None and valid_token_count is not None
    label_id = label_id.reshape(-1).long()
    valid_token_count = valid_token_count.reshape(-1).long()

    num_samples = len(sample_ids)
    if int(label_id.numel()) != num_samples:
        _add_issue(
            hard_errors,
            code="label_id_sample_count_mismatch",
            message="validation issue",
            details={"label_id_count": int(label_id.numel()), "num_sample_id": num_samples},
        )
    if int(valid_token_count.numel()) != num_samples:
        _add_issue(
            hard_errors,
            code="valid_token_count_shape_mismatch",
            message="validation issue",
            details={"valid_token_count": int(valid_token_count.numel()), "num_sample_id": num_samples},
        )

    manifest_summary = _validate_core_cache_manifest(cache, sample_ids, hard_errors)
    label_mapping = cache.get("label_mapping")
    num_classes = _label_mapping_num_classes(label_mapping)
    if num_classes is None:
        _add_issue(
            hard_errors,
            code="invalid_label_mapping",
            message="validation issue",
        )
    elif int(label_id.numel()) == num_samples:
        invalid = (label_id < 0) | (label_id >= int(num_classes))
        if bool(invalid.any().item()):
            bad_indices = torch.nonzero(invalid, as_tuple=False).flatten()[:16].tolist()
            _add_issue(
                hard_errors,
                code="label_id_out_of_range",
                message="validation issue",
                details={
                    "num_classes": int(num_classes),
                    "bad_samples": [sample_ids[int(idx)] for idx in bad_indices],
                    "bad_values": [int(label_id[int(idx)].item()) for idx in bad_indices],
                },
            )

    effective_classes = int(torch.unique(label_id).numel()) if int(label_id.numel()) > 0 else 0
    if effective_classes < 2:
        _add_issue(
            hard_errors,
            code="insufficient_effective_classes",
            message="validation issue",
            details={"effective_classes": effective_classes},
        )

    canonical_buckets: list[str] | None = None
    shot_values: list[Any] | None = None
    tail_included_matches_bucket: bool | None = None
    raw_bucket = cache.get("bucket")
    raw_shot_group = cache.get("shot_group")
    if not isinstance(raw_bucket, Sequence) or isinstance(raw_bucket, (str, bytes)):
        _add_issue(
            hard_errors,
            code="missing_core_cache_bucket",
            message="validation issue",
        )
    else:
        bucket_values = list(raw_bucket)
        if isinstance(raw_shot_group, Sequence) and not isinstance(raw_shot_group, (str, bytes)):
            shot_values = list(raw_shot_group)
        try:
            canonical_buckets = validate_bucket_values(
                bucket_values,
                shot_group=shot_values,
                context="core_cache",
            )
        except GroupValidationError as exc:
            _add_issue(
                hard_errors,
                code="invalid_core_cache_bucket",
                message="validation issue",
                details={"error": str(exc)},
            )
            canonical_buckets = None
        if canonical_buckets is not None and len(canonical_buckets) != num_samples:
            _add_issue(
                hard_errors,
                code="bucket_sample_count_mismatch",
                message="validation issue",
                details={"bucket_count": len(canonical_buckets), "num_sample_id": num_samples},
            )
            canonical_buckets = None

    if canonical_buckets is not None:
        class_bucket: dict[int, str] = {}
        conflicts: list[dict[str, Any]] = []
        for idx, bucket in enumerate(canonical_buckets):
            class_id = int(label_id[idx].item())
            previous = class_bucket.get(class_id)
            if previous is not None and previous != bucket:
                conflicts.append({"label_id": class_id, "first_bucket": previous, "bucket": bucket})
            class_bucket.setdefault(class_id, bucket)
        if conflicts:
            _add_issue(
                hard_errors,
                code="class_bucket_conflict",
                message="validation issue",
                details={"conflicts": conflicts[:16]},
            )
        raw_tail = cache.get("tail_included")
        if raw_tail is not None:
            tail_values = _bool_list(raw_tail) if isinstance(raw_tail, (Tensor, Sequence)) and not isinstance(raw_tail, (str, bytes)) else []
            expected_tail = _bool_list(tail_included(canonical_buckets))
            tail_included_matches_bucket = len(tail_values) == len(expected_tail) and tail_values == expected_tail
            if not tail_included_matches_bucket:
                _add_issue(
                    hard_errors,
                    code="tail_included_mismatch",
                    message="validation issue",
                    details={"tail_count": len(tail_values), "expected_count": len(expected_tail)},
                )

    embedding_chunks: list[Tensor] = []
    total_shard_samples = 0
    logits_dim: int | None = None
    token_count: int | None = None
    if cache_path.exists():
        for shard in iter_fp32_core_cache_shards(cache, cache_path):
            shard_sample_ids = _normalise_sample_ids(shard.get("sample_id"))
            shard_manifest = shard.get("manifest", {})
            shard_start = int(shard_manifest.get("sample_start", total_shard_samples)) if isinstance(shard_manifest, Mapping) else total_shard_samples
            shard_end = shard_start + len(shard_sample_ids)
            expected_ids = sample_ids[shard_start:shard_end]
            if shard_sample_ids != expected_ids:
                _add_issue(
                    hard_errors,
                    code="core_shard_sample_order_mismatch",
                    message="validation issue",
                    details={"sample_start": shard_start, "sample_end": shard_end},
                )
            embedding = _as_tensor(shard, "embedding", dtype=torch.float32)
            logits = _as_tensor(shard, "logits", dtype=torch.float32)
            shard_label = _as_tensor(shard, "label_id", dtype=torch.long)
            padding_mask = _as_tensor(shard, "padding_mask")
            shard_valid = _as_tensor(shard, "valid_token_count", dtype=torch.long)
            missing = [
                name
                for name, value in (
                    ("embedding", embedding),
                    ("logits", logits),
                    ("label_id", shard_label),
                    ("padding_mask", padding_mask),
                    ("valid_token_count", shard_valid),
                )
                if value is None
            ]
            if missing:
                _add_issue(
                    hard_errors,
                    code="core_shard_missing_fields",
                    message="validation issue",
                    details={"sample_start": shard_start, "missing_fields": missing},
                )
                continue
            assert embedding is not None and logits is not None and shard_label is not None and padding_mask is not None and shard_valid is not None
            if embedding.ndim != 2 or int(embedding.shape[1]) != int(expected_embedding_dim):
                _add_issue(
                    hard_errors,
                    code="invalid_embedding_shape",
                    message="validation issue",
                    details={"shape": list(embedding.shape), "expected_dim": int(expected_embedding_dim), "sample_start": shard_start},
                )
            else:
                embedding_chunks.append(embedding.cpu().contiguous())
            if logits.ndim != 2:
                _add_issue(
                    hard_errors,
                    code="invalid_logits_shape",
                    message="validation issue",
                    details={"shape": list(logits.shape), "sample_start": shard_start},
                )
            else:
                logits_dim = int(logits.shape[1]) if logits_dim is None else logits_dim
                if int(logits.shape[1]) != int(logits_dim):
                    _add_issue(
                        hard_errors,
                        code="core_shard_logits_dim_mismatch",
                        message="validation issue",
                        details={"expected": logits_dim, "actual": int(logits.shape[1]), "sample_start": shard_start},
                    )
                if num_classes is not None and int(logits.shape[1]) != int(num_classes):
                    _add_issue(
                        hard_errors,
                        code="logits_label_mapping_dim_mismatch",
                        message="validation issue",
                        details={"num_classes": int(num_classes), "logits_dim": int(logits.shape[1]), "sample_start": shard_start},
                    )
            for name, tensor in (("embedding", embedding), ("logits", logits), ("label_id", shard_label)):
                if int(tensor.shape[0]) != len(shard_sample_ids):
                    _add_issue(
                        hard_errors,
                        code=f"{name}_sample_count_mismatch",
                        message="validation issue",
                        details={"shape": list(tensor.shape), "num_sample_id": len(shard_sample_ids), "sample_start": shard_start},
                    )
            if not torch.isfinite(embedding).all().item():
                _add_issue(hard_errors, code="embedding_non_finite", message="embedding ??? NaN/Inf", details={"sample_start": shard_start})
            if not torch.isfinite(logits).all().item():
                _add_issue(hard_errors, code="logits_non_finite", message="logits ??? NaN/Inf", details={"sample_start": shard_start})
            shard_label = shard_label.reshape(-1).long()
            if shard_end <= int(label_id.numel()) and not torch.equal(shard_label.cpu(), label_id[shard_start:shard_end].cpu()):
                _add_issue(
                    hard_errors,
                    code="core_shard_label_id_mismatch",
                    message="validation issue",
                    details={"sample_start": shard_start, "sample_end": shard_end},
                )
            if padding_mask.ndim != 2:
                _add_issue(
                    hard_errors,
                    code="invalid_padding_mask_shape",
                    message="validation issue",
                    details={"shape": list(padding_mask.shape), "sample_start": shard_start},
                )
            else:
                token_count = int(padding_mask.shape[1]) if token_count is None else token_count
                if int(padding_mask.shape[0]) != len(shard_sample_ids):
                    _add_issue(
                        hard_errors,
                        code="padding_mask_sample_count_mismatch",
                        message="validation issue",
                        details={"shape": list(padding_mask.shape), "num_sample_id": len(shard_sample_ids), "sample_start": shard_start},
                    )
                if padding_mask.dtype is not torch.bool:
                    unique_values = torch.unique(padding_mask)
                    if not torch.all((unique_values == 0) | (unique_values == 1)).item():
                        _add_issue(
                            hard_errors,
                            code="invalid_padding_mask_values",
                            message="validation issue",
                            details={"unique_values": unique_values[:8].tolist(), "sample_start": shard_start},
                        )
                    padding_bool = padding_mask.to(dtype=torch.bool)
                else:
                    padding_bool = padding_mask
                computed_valid = (~padding_bool).sum(dim=1).long()
                shard_valid = shard_valid.reshape(-1).long()
                if int(shard_valid.numel()) != len(shard_sample_ids):
                    _add_issue(
                        hard_errors,
                        code="valid_token_count_shape_mismatch",
                        message="validation issue",
                        details={"shape": list(shard_valid.shape), "num_sample_id": len(shard_sample_ids), "sample_start": shard_start},
                    )
                elif not torch.equal(computed_valid.cpu(), shard_valid.cpu()):
                    mismatch = torch.nonzero(computed_valid.cpu() != shard_valid.cpu(), as_tuple=False).flatten()
                    _add_issue(
                        hard_errors,
                        code="padding_mask_valid_count_mismatch",
                        message="validation issue",
                        details={"mismatch_samples": [shard_sample_ids[int(idx)] for idx in mismatch[:16].tolist()]},
                    )
                if shard_end <= int(valid_token_count.numel()) and not torch.equal(shard_valid.cpu(), valid_token_count[shard_start:shard_end].cpu()):
                    _add_issue(
                        hard_errors,
                        code="core_shard_valid_token_count_mismatch",
                        message="validation issue",
                        details={"sample_start": shard_start, "sample_end": shard_end},
                    )
                if bool((computed_valid <= 0).any().item()):
                    bad = torch.nonzero(computed_valid <= 0, as_tuple=False).flatten()[:16].tolist()
                    _add_issue(
                        hard_errors,
                        code="no_valid_token",
                        message="validation issue",
                        details={"bad_samples": [shard_sample_ids[int(idx)] for idx in bad]},
                    )
            total_shard_samples += len(shard_sample_ids)
    else:
        _add_issue(hard_errors, code="missing_core_cache", message="validation issue", details={"path": str(cache_path)})

    if total_shard_samples != num_samples:
        _add_issue(
            hard_errors,
            code="core_shard_sample_count_mismatch",
            message="validation issue",
            details={"total_shard_samples": total_shard_samples, "num_samples": num_samples},
        )
    if int(label_id.numel()) == num_samples:
        _check_eval_csv(frame=eval_frame, sample_ids=sample_ids, label_id=label_id.reshape(-1), hard_errors=hard_errors)

    embedding = torch.cat(embedding_chunks, dim=0) if embedding_chunks else torch.empty((0, expected_embedding_dim), dtype=torch.float32)
    bucket_counts = _group_counts_from_buckets(canonical_buckets or []) if canonical_buckets is not None else {}
    analysis_group_counts = _analysis_group_counts_from_buckets(canonical_buckets or []) if canonical_buckets is not None else {}
    class_count_by_analysis_group = (
        _class_counts_by_analysis_group(label_id.reshape(-1), canonical_buckets)
        if canonical_buckets is not None
        else {}
    )
    group_protocol = _bucket_protocol_summary(
        bucket_present=isinstance(raw_bucket, Sequence) and not isinstance(raw_bucket, (str, bytes)),
        shot_group_present=shot_values is not None,
        canonical_buckets=canonical_buckets,
        tail_included_present=cache.get("tail_included") is not None,
        tail_included_matches_bucket=tail_included_matches_bucket,
    )
    summary = {
        "num_samples": num_samples,
        "num_classes_in_label_mapping": None if num_classes is None else int(num_classes),
        "effective_classes": effective_classes,
        "bucket_counts": bucket_counts,
        "analysis_group_counts": analysis_group_counts,
        "sample_count_by_analysis_group": analysis_group_counts,
        "class_count_by_analysis_group": class_count_by_analysis_group,
        "group_protocol": group_protocol,
        "tail_definition": "tail = medium + few",
        "storage_format": "sample_sharded_v1",
        "num_shards": len(cache.get("shards", []) or []),
        "embedding_shape": list(embedding.shape),
        "logits_shape": [num_samples, logits_dim] if logits_dim is not None else None,
        "padding_mask_shape": [num_samples, token_count] if token_count is not None else None,
        "sample_id_preview": sample_ids[:5],
        "sample_order_hash": _sample_order_hash(sample_ids),
        "eval_csv_sample_order_hash": None
        if eval_frame is None or "sample_id" not in eval_frame.columns
        else _sample_order_hash(eval_frame["sample_id"].astype(str).tolist()),
        "sample_order_hash_algorithm": "sha256-json-list",
        "manifest": manifest_summary,
    }
    tensors = {
        "sample_ids": sample_ids,
        "embedding": embedding,
        "label_id": label_id.reshape(-1),
        "valid_token_count": valid_token_count.reshape(-1),
        "num_classes": num_classes,
        "bucket": canonical_buckets or [],
    }
    return {"summary": summary, "tensors": tensors}, hard_errors

def _validate_core_cache_structure(
    cache: Mapping[str, Any],
    *,
    expected_embedding_dim: int,
    eval_frame: Any | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if is_sharded_fp32_core_cache(cache):
        return _validate_sharded_core_cache_structure(
            cache,
            expected_embedding_dim=expected_embedding_dim,
            eval_frame=eval_frame,
        )

    hard_errors: list[dict[str, Any]] = []
    sample_ids = _normalise_sample_ids(cache.get("sample_id"))
    if not sample_ids:
        _add_issue(hard_errors, code="missing_sample_id", message="core cache ??? sample_id")
    duplicates = sorted({item for item in sample_ids if sample_ids.count(item) > 1})
    if duplicates:
        _add_issue(
            hard_errors,
            code="duplicate_sample_id",
            message="validation issue",
            details={"duplicates": duplicates[:16]},
        )

    embedding = _as_tensor(cache, "embedding", dtype=torch.float32)
    logits = _as_tensor(cache, "logits", dtype=torch.float32)
    label_id = _as_tensor(cache, "label_id", dtype=torch.long)
    padding_mask = _as_tensor(cache, "padding_mask")
    valid_token_count = _as_tensor(cache, "valid_token_count", dtype=torch.long)

    if embedding is None:
        _add_issue(hard_errors, code="missing_embedding", message="core cache ??? embedding")
    if logits is None:
        _add_issue(hard_errors, code="missing_logits", message="core cache ??? logits")
    if label_id is None:
        _add_issue(hard_errors, code="missing_label_id", message="core cache ??? label_id")
    if padding_mask is None:
        _add_issue(hard_errors, code="missing_padding_mask", message="core cache ??? padding_mask")
    if valid_token_count is None:
        _add_issue(hard_errors, code="missing_valid_token_count", message="core cache ??? valid_token_count")

    if hard_errors:
        return {}, hard_errors

    assert embedding is not None and logits is not None and label_id is not None
    assert padding_mask is not None and valid_token_count is not None
    manifest_summary = _validate_core_cache_manifest(cache, sample_ids, hard_errors)

    num_samples = len(sample_ids)
    if embedding.ndim != 2 or int(embedding.shape[1]) != int(expected_embedding_dim):
        _add_issue(
            hard_errors,
            code="invalid_embedding_shape",
            message="validation issue",
            details={"shape": list(embedding.shape), "expected_dim": int(expected_embedding_dim)},
        )
    if logits.ndim != 2:
        _add_issue(
            hard_errors,
            code="invalid_logits_shape",
            message="validation issue",
            details={"shape": list(logits.shape)},
        )
    if label_id.ndim != 1:
        label_id = label_id.reshape(-1)
    for name, tensor in (("embedding", embedding), ("logits", logits), ("label_id", label_id)):
        if int(tensor.shape[0]) != num_samples:
            _add_issue(
                hard_errors,
                code=f"{name}_sample_count_mismatch",
                message="validation issue",
                details={"shape": list(tensor.shape), "num_sample_id": num_samples},
            )
    if not torch.isfinite(embedding).all().item():
        _add_issue(hard_errors, code="embedding_non_finite", message="embedding ??? NaN/Inf")
    if not torch.isfinite(logits).all().item():
        _add_issue(hard_errors, code="logits_non_finite", message="logits ??? NaN/Inf")

    label_mapping = cache.get("label_mapping")
    num_classes = _label_mapping_num_classes(label_mapping)
    if num_classes is None:
        _add_issue(
            hard_errors,
            code="invalid_label_mapping",
            message="validation issue",
        )
    elif logits.ndim == 2 and int(logits.shape[1]) != int(num_classes):
        _add_issue(
            hard_errors,
            code="logits_label_mapping_dim_mismatch",
            message="validation issue",
            details={"num_classes": int(num_classes), "logits_dim": int(logits.shape[1])},
        )
    if num_classes is not None:
        invalid = (label_id < 0) | (label_id >= int(num_classes))
        if bool(invalid.any().item()):
            bad_indices = torch.nonzero(invalid, as_tuple=False).flatten()[:16].tolist()
            _add_issue(
                hard_errors,
                code="label_id_out_of_range",
                message="validation issue",
                details={
                    "num_classes": int(num_classes),
                    "bad_samples": [sample_ids[int(idx)] for idx in bad_indices],
                    "bad_values": [int(label_id[int(idx)].item()) for idx in bad_indices],
                },
            )

    effective_classes = int(torch.unique(label_id).numel())
    if effective_classes < 2:
        _add_issue(
            hard_errors,
            code="insufficient_effective_classes",
            message="validation issue",
            details={"effective_classes": effective_classes},
        )

    canonical_buckets: list[str] | None = None
    shot_values: list[Any] | None = None
    tail_included_matches_bucket: bool | None = None
    raw_bucket = cache.get("bucket")
    raw_shot_group = cache.get("shot_group")
    if not isinstance(raw_bucket, Sequence) or isinstance(raw_bucket, (str, bytes)):
        _add_issue(
            hard_errors,
            code="missing_core_cache_bucket",
            message="validation issue",
        )
    else:
        bucket_values = list(raw_bucket)
        if isinstance(raw_shot_group, Sequence) and not isinstance(raw_shot_group, (str, bytes)):
            shot_values = list(raw_shot_group)
        try:
            canonical_buckets = validate_bucket_values(
                bucket_values,
                shot_group=shot_values,
                context="core_cache",
            )
        except GroupValidationError as exc:
            _add_issue(
                hard_errors,
                code="invalid_core_cache_bucket",
                message="validation issue",
                details={"error": str(exc)},
            )
            canonical_buckets = None
        if canonical_buckets is not None and len(canonical_buckets) != num_samples:
            _add_issue(
                hard_errors,
                code="bucket_sample_count_mismatch",
                message="validation issue",
                details={"bucket_count": len(canonical_buckets), "num_sample_id": num_samples},
            )
            canonical_buckets = None

    if canonical_buckets is not None:
        class_bucket: dict[int, str] = {}
        conflicts: list[dict[str, Any]] = []
        flat_label_id = label_id.reshape(-1)
        for idx, bucket in enumerate(canonical_buckets):
            class_id = int(flat_label_id[idx].item())
            previous = class_bucket.get(class_id)
            if previous is not None and previous != bucket:
                conflicts.append({"label_id": class_id, "first_bucket": previous, "bucket": bucket})
            class_bucket.setdefault(class_id, bucket)
        if conflicts:
            _add_issue(
                hard_errors,
                code="class_bucket_conflict",
                message="validation issue",
                details={"conflicts": conflicts[:16]},
            )

        raw_tail = cache.get("tail_included")
        if raw_tail is not None:
            if isinstance(raw_tail, Tensor):
                tail_values = [bool(item) for item in raw_tail.detach().cpu().reshape(-1).tolist()]
            elif isinstance(raw_tail, Sequence) and not isinstance(raw_tail, (str, bytes)):
                tail_values = [bool(item) for item in raw_tail]
            else:
                tail_values = []
            expected_tail = _bool_list(tail_included(canonical_buckets))
            tail_included_matches_bucket = len(tail_values) == len(expected_tail) and tail_values == expected_tail
            if not tail_included_matches_bucket:
                _add_issue(
                    hard_errors,
                    code="tail_included_mismatch",
                    message="validation issue",
                    details={"tail_count": len(tail_values), "expected_count": len(expected_tail)},
                )

    if padding_mask.ndim != 2:
        _add_issue(
            hard_errors,
            code="invalid_padding_mask_shape",
            message="validation issue",
            details={"shape": list(padding_mask.shape)},
        )
    elif int(padding_mask.shape[0]) != num_samples:
        _add_issue(
            hard_errors,
            code="padding_mask_sample_count_mismatch",
            message="validation issue",
            details={"shape": list(padding_mask.shape), "num_sample_id": num_samples},
        )
    else:
        if padding_mask.dtype is not torch.bool:
            unique_values = torch.unique(padding_mask)
            if not torch.all((unique_values == 0) | (unique_values == 1)).item():
                _add_issue(
                    hard_errors,
                    code="invalid_padding_mask_values",
                    message="validation issue",
                    details={"unique_values": unique_values[:8].tolist()},
                )
            padding_bool = padding_mask.to(dtype=torch.bool)
        else:
            padding_bool = padding_mask
        computed_valid = (~padding_bool).sum(dim=1).long()
        if int(valid_token_count.numel()) != num_samples:
            _add_issue(
                hard_errors,
                code="valid_token_count_shape_mismatch",
                message="validation issue",
                details={"shape": list(valid_token_count.shape), "num_sample_id": num_samples},
            )
        elif not torch.equal(computed_valid.cpu(), valid_token_count.reshape(-1).cpu().long()):
            mismatch = torch.nonzero(computed_valid.cpu() != valid_token_count.reshape(-1).cpu().long(), as_tuple=False).flatten()
            _add_issue(
                hard_errors,
                code="padding_mask_valid_count_mismatch",
                message="validation issue",
                details={"mismatch_samples": [sample_ids[int(idx)] for idx in mismatch[:16].tolist()]},
            )
        if bool((computed_valid <= 0).any().item()):
            bad = torch.nonzero(computed_valid <= 0, as_tuple=False).flatten()[:16].tolist()
            _add_issue(
                hard_errors,
                code="no_valid_token",
                message="validation issue",
                details={"bad_samples": [sample_ids[int(idx)] for idx in bad]},
            )

    if label_id is not None:
        _check_eval_csv(frame=eval_frame, sample_ids=sample_ids, label_id=label_id.reshape(-1), hard_errors=hard_errors)

    bucket_counts = _group_counts_from_buckets(canonical_buckets or []) if canonical_buckets is not None else {}
    analysis_group_counts = _analysis_group_counts_from_buckets(canonical_buckets or []) if canonical_buckets is not None else {}
    class_count_by_analysis_group = (
        _class_counts_by_analysis_group(label_id.reshape(-1), canonical_buckets)
        if canonical_buckets is not None
        else {}
    )
    group_protocol = _bucket_protocol_summary(
        bucket_present=isinstance(raw_bucket, Sequence) and not isinstance(raw_bucket, (str, bytes)),
        shot_group_present=shot_values is not None,
        canonical_buckets=canonical_buckets,
        tail_included_present=raw_tail is not None if canonical_buckets is not None else cache.get("tail_included") is not None,
        tail_included_matches_bucket=tail_included_matches_bucket,
    )

    summary = {
        "num_samples": num_samples,
        "num_classes_in_label_mapping": None if num_classes is None else int(num_classes),
        "effective_classes": effective_classes,
        "bucket_counts": bucket_counts,
        "analysis_group_counts": analysis_group_counts,
        "sample_count_by_analysis_group": analysis_group_counts,
        "class_count_by_analysis_group": class_count_by_analysis_group,
        "group_protocol": group_protocol,
        "tail_definition": "tail = medium + few",
        "embedding_shape": None if embedding is None else list(embedding.shape),
        "logits_shape": None if logits is None else list(logits.shape),
        "padding_mask_shape": None if padding_mask is None else list(padding_mask.shape),
        "sample_id_preview": sample_ids[:5],
        "sample_order_hash": _sample_order_hash(sample_ids),
        "eval_csv_sample_order_hash": None
        if eval_frame is None or "sample_id" not in eval_frame.columns
        else _sample_order_hash(eval_frame["sample_id"].astype(str).tolist()),
        "sample_order_hash_algorithm": "sha256-json-list",
        "manifest": manifest_summary,
    }
    tensors = {
        "sample_ids": sample_ids,
        "embedding": embedding,
        "logits": logits,
        "label_id": label_id.reshape(-1),
        "padding_mask": padding_mask,
        "valid_token_count": valid_token_count.reshape(-1),
        "num_classes": num_classes,
        "bucket": canonical_buckets or [],
    }
    return {"summary": summary, "tensors": tensors}, hard_errors


def _bool_list(values: Any) -> list[bool]:
    if isinstance(values, Tensor):
        return [bool(item) for item in values.detach().cpu().reshape(-1).tolist()]
    if hasattr(values, "tolist") and not isinstance(values, list):
        try:
            return [bool(item) for item in values.tolist()]
        except TypeError:
            pass
    return [bool(item) for item in list(values)]


def _group_counts_from_buckets(buckets: Sequence[str]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in BASE_BUCKETS}
    for group in BASE_BUCKETS:
        mask = _bool_list(analysis_group_mask(buckets, group))
        counts[str(group)] = int(sum(1 for item in mask if item))
    return dict(counts)


def _analysis_group_counts_from_buckets(buckets: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for group in ANALYSIS_GROUPS:
        mask = _bool_list(analysis_group_mask(buckets, group))
        result[str(group)] = int(sum(1 for item in mask if item))
    return result


def _class_ids_for_analysis_group(label_id: Tensor, buckets: Sequence[str], analysis_group: str) -> Tensor:
    labels = label_id.reshape(-1).cpu().long()
    if len(buckets) != int(labels.numel()):
        return torch.empty(0, dtype=torch.long)
    mask = torch.as_tensor(_bool_list(analysis_group_mask(buckets, analysis_group)), dtype=torch.bool)
    if not bool(mask.any().item()):
        return torch.empty(0, dtype=torch.long)
    return torch.unique(labels[mask], sorted=True)


def _class_counts_by_analysis_group(label_id: Tensor, buckets: Sequence[str]) -> dict[str, int]:
    return {
        str(group): int(_class_ids_for_analysis_group(label_id, buckets, str(group)).numel())
        for group in ANALYSIS_GROUPS
    }


def _bucket_protocol_summary(
    *,
    bucket_present: bool,
    shot_group_present: bool,
    canonical_buckets: Sequence[str] | None,
    tail_included_present: bool,
    tail_included_matches_bucket: bool | None,
) -> dict[str, Any]:
    bucket_valid = canonical_buckets is not None
    return {
        "bucket_present": bool(bucket_present),
        "bucket_valid": bool(bucket_valid),
        "allowed_bucket_values": list(BASE_BUCKETS),
        "allowed_analysis_group_values": list(ANALYSIS_GROUPS),
        "sample_level_tail_forbidden": True,
        "tail_absent_from_bucket": bool(bucket_valid),
        "shot_group_present": bool(shot_group_present),
        "shot_group_equals_bucket": (bool(bucket_valid) if shot_group_present else None),
        "tail_included_present": bool(tail_included_present),
        "tail_included_matches_bucket": tail_included_matches_bucket,
        "tail_definition": "tail = medium + few",
    }

def _float_or_nan(value: Any) -> float:
    try:
        number = float(value.item() if isinstance(value, Tensor) else value)
    except (TypeError, ValueError):
        return float("nan")
    if number != number or number in {float("inf"), float("-inf")}:
        return float("nan")
    return number


def _fisher_global_anchored_summary(embedding: Tensor, label_id: Tensor, buckets: Sequence[str]) -> dict[str, Any]:
    fisher_by_group: dict[str, float] = {}
    effective_counts: dict[str, int] = {}
    class_ids_by_group: dict[str, list[int]] = {}

    for group in ANALYSIS_GROUPS:
        class_ids = _class_ids_for_analysis_group(label_id, buckets, str(group))
        effective_counts[str(group)] = int(class_ids.numel())
        class_ids_by_group[str(group)] = [int(item) for item in class_ids.tolist()]
        if int(class_ids.numel()) < 2:
            fisher_by_group[str(group)] = float("nan")
            continue
        fisher_by_group[str(group)] = _float_or_nan(
            compute_group_fisher(
                embedding,
                label_id,
                class_ids,
                normalize_by_dof=False,
            )
        )

    if int(torch.unique(label_id.reshape(-1)).numel()) < 2:
        global_value = float("nan")
    else:
        global_value = _float_or_nan(
            compute_group_fisher(
                embedding,
                label_id,
                None,
                normalize_by_dof=False,
            )
        )

    return {
        "fisher_global_anchored": global_value,
        "fisher_by_analysis_group": fisher_by_group,
        "fisher_effective_class_count_by_analysis_group": effective_counts,
        "fisher_class_ids_by_analysis_group": class_ids_by_group,
    }

def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    if result == float("inf") or result == float("-inf"):
        return None
    return result


def _quality_checks(
    cache: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float | int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    embedding: Tensor = tensors["embedding"]
    label_id: Tensor = tensors["label_id"]
    sample_ids: list[str] = list(tensors["sample_ids"])

    buckets: list[str] = list(tensors.get("bucket", []))
    fisher_details = _fisher_global_anchored_summary(embedding, label_id, buckets)
    fisher_value = fisher_details["fisher_global_anchored"]
    fisher_check_value = _safe_float(fisher_value)
    if fisher_check_value is None:
        _add_issue(warnings, code="fisher_not_finite", message="validation issue", severity="warning")
    elif fisher_check_value < 0:
        _add_issue(
            warnings,
            code="fisher_negative",
            message="validation issue",
            details={"fisher": fisher_check_value},
            severity="warning",
        )
    elif fisher_check_value < float(thresholds["MIN_FISHER_RATIO"]):
        _add_issue(
            warnings,
            code="fisher_too_small",
            message="validation issue",
            details={"fisher": fisher_check_value, "threshold": float(thresholds["MIN_FISHER_RATIO"])},
            severity="warning",
        )

    trace_values, trace_details = class_trace_variance(
        embedding,
        label_id,
        normalization="dof",
        reduction="none",
        return_details=True,
    )
    assert isinstance(trace_values, Tensor)
    finite_trace = trace_values[torch.isfinite(trace_values)]
    trace_summary: dict[str, Any] = {
        "finite_class_count": int(finite_trace.numel()),
        "nan_class_count": int((~torch.isfinite(trace_values)).sum().item()),
    }
    if int(finite_trace.numel()) > 0:
        trace_summary.update(
            {
                "mean": float(finite_trace.mean().item()),
                "median": float(torch.quantile(finite_trace, 0.5).item()),
                "p95": float(torch.quantile(finite_trace, 0.95).item()),
                "min": float(finite_trace.min().item()),
                "max": float(finite_trace.max().item()),
            }
        )
        if float(finite_trace.min().item()) <= float(thresholds["TRACE_VARIANCE_LOW_EPS"]):
            _add_issue(
                warnings,
                code="trace_variance_too_small",
                message="validation issue",
                details={"min": float(finite_trace.min().item())},
                severity="warning",
            )
        if float(torch.quantile(finite_trace, 0.95).item()) >= float(thresholds["TRACE_VARIANCE_HIGH_P95"]):
            _add_issue(
                warnings,
                code="trace_variance_too_large",
                message="validation issue",
                details={"p95": float(torch.quantile(finite_trace, 0.95).item())},
                severity="warning",
            )
    else:
        _add_issue(
            warnings,
            code="trace_variance_all_nan",
            message="validation issue",
            severity="warning",
        )

    margins = compute_group_margin(embedding, label_id, None, reduction="none")
    assert isinstance(margins, Tensor)
    margin_total_count = int(margins.numel())
    finite_margins = margins[torch.isfinite(margins)]
    margin_valid_count = int(finite_margins.numel())
    margin_valid_rate = float(margin_valid_count / margin_total_count) if margin_total_count > 0 else None
    if margin_valid_count == 0:
        positive_ratio = None
        _add_issue(
            warnings,
            code="margin_all_invalid",
            message="validation issue",
            severity="warning",
        )
    else:
        positive_ratio = float((finite_margins > 0).float().mean().item())
        if positive_ratio < float(thresholds["MIN_MARGIN_POSITIVE_RATIO"]):
            _add_issue(
                warnings,
                code="low_positive_margin_ratio",
                message="validation issue",
                details={
                    "positive_ratio": positive_ratio,
                    "threshold": float(thresholds["MIN_MARGIN_POSITIVE_RATIO"]),
                },
                severity="warning",
            )

    group_counts = _group_counts_from_buckets(buckets)
    analysis_group_counts = _analysis_group_counts_from_buckets(buckets)
    class_count_by_analysis_group = _class_counts_by_analysis_group(label_id, buckets)
    few_count = int(group_counts.get("few", 0))
    few_class_count = int(class_count_by_analysis_group.get("few", 0))
    if few_count < int(thresholds["MIN_FEW_SAMPLES"]):
        _add_issue(
            warnings,
            code="few_samples_too_few",
            message="validation issue",
            details={
                "sample_count": few_count,
                "class_count": few_class_count,
                "threshold": int(thresholds["MIN_FEW_SAMPLES"]),
                "strict_blocking": False,
            },
            severity="warning",
        )

    tail_count = int(analysis_group_counts.get("tail", 0))
    tail_class_count = int(class_count_by_analysis_group.get("tail", 0))
    min_tail_samples = int(thresholds["MIN_TAIL_SAMPLES"])
    min_effective_classes = int(thresholds["MIN_EFFECTIVE_CLASSES"])
    if tail_count < min_tail_samples or tail_class_count < min_effective_classes:
        _add_issue(
            warnings,
            code="tail_group_too_small",
            message="validation issue",
            details={
                "sample_count": tail_count,
                "class_count": tail_class_count,
                "min_tail_samples": min_tail_samples,
                "min_effective_classes": min_effective_classes,
                "strict_blocking": True,
            },
            severity="warning",
        )

    known_group_counts = {key: int(group_counts.get(key, 0)) for key in SHOT_GROUPS}
    nonzero_counts = [value for value in known_group_counts.values() if value > 0]
    if len(nonzero_counts) >= 2:
        imbalance_ratio = float(max(nonzero_counts) / max(1, min(nonzero_counts)))
        if imbalance_ratio > float(thresholds["MAX_SHOT_GROUP_IMBALANCE_RATIO"]):
            _add_issue(
                warnings,
                code="shot_group_imbalance",
                message="validation issue",
                details={"group_counts": known_group_counts, "imbalance_ratio": imbalance_ratio},
                severity="warning",
            )
    else:
        imbalance_ratio = None
        _add_issue(
            warnings,
            code="shot_group_missing",
            message="validation issue",
            details={"group_counts": group_counts},
            severity="warning",
        )
    class_ids = trace_details["class_ids"]
    counts = trace_details["counts"]
    assert isinstance(class_ids, Tensor) and isinstance(counts, Tensor)
    low_count_mask = counts < int(thresholds["MIN_GEOMETRY_SAMPLES"])
    low_count_classes = [
        {"label_id": int(class_ids[idx].item()), "count": int(counts[idx].item())}
        for idx in torch.nonzero(low_count_mask, as_tuple=False).flatten().tolist()
    ]
    if low_count_classes:
        _add_issue(
            warnings,
            code="class_sample_count_too_low",
            message="validation issue",
            details={
                "threshold": int(thresholds["MIN_GEOMETRY_SAMPLES"]),
                "classes": low_count_classes[:32],
                "num_low_count_classes": len(low_count_classes),
            },
            severity="warning",
        )

    geometry_summary = {
        "fisher_ratio": fisher_value,
        "fisher_global_anchored": fisher_value,
        "fisher_by_analysis_group": fisher_details["fisher_by_analysis_group"],
        "fisher_effective_class_count_by_analysis_group": fisher_details["fisher_effective_class_count_by_analysis_group"],
        "fisher_class_ids_by_analysis_group": fisher_details["fisher_class_ids_by_analysis_group"],
        "class_trace_variance": trace_summary,
        "margin": {
            "positive_ratio": positive_ratio,
            "finite_count": margin_valid_count,
            "valid_count": margin_valid_count,
            "total_count": margin_total_count,
            "valid_rate": margin_valid_rate,
            "margin_valid_count": margin_valid_count,
            "margin_total_count": margin_total_count,
            "margin_valid_rate": margin_valid_rate,
            "mean": None if margin_valid_count == 0 else float(finite_margins.mean().item()),
            "median": None if margin_valid_count == 0 else float(torch.quantile(finite_margins, 0.5).item()),
        },
        "bucket_counts": group_counts,
        "analysis_group_counts": analysis_group_counts,
        "sample_count_by_analysis_group": analysis_group_counts,
        "class_count_by_analysis_group": class_count_by_analysis_group,
        "shot_group_counts": group_counts,
        "known_shot_group_counts": known_group_counts,
        "shot_group_imbalance_ratio": imbalance_ratio,
        "num_low_count_classes": len(low_count_classes),
    }
    return geometry_summary, warnings


def _model_structure_checks(
    cfg: Any,
    *,
    cache_summary: Mapping[str, Any],
    model: nn.Module | None,
    enabled: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hard_errors: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "checked": bool(enabled),
        "predictor_out_features": None,
        "oer_target_count": None,
    }
    if not enabled:
        return summary, hard_errors

    if model is None:
        model = _restore_fp32_teacher(cfg)
    model.eval()
    predictor_dim = _predictor_out_features(model)
    summary["predictor_out_features"] = predictor_dim
    expected_classes = cache_summary.get("num_classes_in_label_mapping")
    if predictor_dim is None:
        _add_issue(
            hard_errors,
            code="missing_checkpoint_predictor",
            message="validation issue",
        )
    elif expected_classes is not None and int(predictor_dim) != int(expected_classes):
        _add_issue(
            hard_errors,
            code="checkpoint_head_label_mapping_mismatch",
            message="validation issue",
            details={"predictor_out_features": int(predictor_dim), "num_classes": int(expected_classes)},
        )
    summary["oer_target_count"] = _check_oer_targets(model, hard_errors)
    return summary, hard_errors


def save_report(report: Mapping[str, Any], report_path: str | Path) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(report), handle, ensure_ascii=False, indent=2)
    return path


def run_fp32_geometry_sanity_check(
    cfg: Any | None = None,
    *,
    core_cache_path: str | Path | None = None,
    report_path: str | Path | None = None,
    model: nn.Module | None = None,
    strict: bool | None = None,
    check_model_structure: bool | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """??? FP32 core cache?????Quant Injection ?????? sanity check??
    A ??????????????????? ?????????????????? report??trict=True
    ???B ??warning ???????????    """

    cfg = _default_cfg() if cfg is None else cfg
    strict = bool(_setting_get(cfg, "STRICT", False) if strict is None else strict)
    check_model_structure = bool(
        _setting_get(cfg, "CHECK_MODEL_STRUCTURE", True)
        if check_model_structure is None
        else check_model_structure
    )
    thresholds = _thresholds(cfg)
    blocking_warning_codes = _blocking_warning_codes(cfg)
    metric_interface_summary, interface_errors = _check_metric_interface_compatibility()
    path = _core_cache_path(cfg, core_cache_path)
    report_file = _report_path(cfg, report_path)
    cache = _load_core_cache(path)

    eval_frame = _load_eval_frame(_eval_csv_path(cfg))
    expected_dim = int(_setting_get(cfg, "EXPECTED_EMBEDDING_DIM", EXPECTED_EMBEDDING_DIM))

    structure_payload, hard_errors = _validate_core_cache_structure(
        cache,
        expected_embedding_dim=expected_dim,
        eval_frame=eval_frame,
    )
    hard_errors = interface_errors + hard_errors
    cache_summary = structure_payload.get("summary", {})

    model_summary, model_errors = _model_structure_checks(
        cfg,
        cache_summary=cache_summary,
        model=model,
        enabled=check_model_structure,
    )
    hard_errors.extend(model_errors)

    tensor_payload = structure_payload.get("tensors", {})
    expected_oer_sample_ids = tensor_payload.get("sample_ids", _normalise_sample_ids(cache.get("sample_id")))
    oer_oracle_summary, oer_oracle_errors = _check_oer_oracle_cache(
        cfg,
        expected_sample_ids=expected_oer_sample_ids,
    )
    hard_errors.extend(oer_oracle_errors)

    geometry_summary: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    if not hard_errors:
        geometry_summary, warnings = _quality_checks(
            cache,
            structure_payload["tensors"],
            thresholds=thresholds,
        )

    blocking_warnings = _select_blocking_warnings(warnings, blocking_warning_codes)
    status = "pass"
    if hard_errors:
        status = "hard_error"
    elif warnings:
        status = "warning"
    if strict and blocking_warnings and not hard_errors:
        status = "strict_failed"

    report = {
        "report_version": REPORT_VERSION,
        "status": status,
        "strict": bool(strict),
        "blocking_warning_codes": sorted(blocking_warning_codes),
        "core_cache_path": str(path.expanduser().resolve()),
        "report_path": str(report_file.expanduser().resolve()),
        "thresholds": thresholds,
        "summary": {
            "cache": cache_summary,
            "metric_interfaces": metric_interface_summary,
            "model_structure": model_summary,
            "oer_oracle_cache": oer_oracle_summary,
            "geometry": geometry_summary,
        },
        "hard_error_count": len(hard_errors),
        "warning_count": len(warnings),
        "blocking_warning_count": len(blocking_warnings),
        "hard_errors": hard_errors,
        "warnings": warnings,
        "blocking_warnings": blocking_warnings,
    }
    if save:
        save_report(report, report_file)

    if hard_errors:
        raise FP32GeometrySanityError("FP32 core cache sanity failed", report)
    if strict and blocking_warnings:
        raise FP32GeometrySanityError("FP32 geometry sanity strict mode failed", report)
    return report


def main(cfg: Any | None = None) -> dict[str, Any]:
    return run_fp32_geometry_sanity_check(cfg)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "FP32GeometrySanityError",
    "REPORT_DIRNAME",
    "REPORT_FILENAME",
    "REPORT_VERSION",
    "run_fp32_geometry_sanity_check",
    "save_report",
    "main",
]
