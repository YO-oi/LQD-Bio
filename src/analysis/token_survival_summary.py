from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from src.utils.group_utils import (
        ANALYSIS_GROUPS,
        GROUP_PROTOCOL_NAME,
        GROUP_PROTOCOL_VERSION,
        MetricSpec,
        aggregate_metric_for_analysis_group_from_raw_records,
        analysis_group_mask,
        build_metric_spec_registry,
        compute_tail_and_few_gaps,
        define_metric_spec,
        metric_spec_metadata,
        read_canonical_bucket,
        tail_included,
    )
except ImportError:  # pragma: no cover
    from ..utils.group_utils import (
        ANALYSIS_GROUPS,
        GROUP_PROTOCOL_NAME,
        GROUP_PROTOCOL_VERSION,
        MetricSpec,
        aggregate_metric_for_analysis_group_from_raw_records,
        analysis_group_mask,
        build_metric_spec_registry,
        compute_tail_and_few_gaps,
        define_metric_spec,
        metric_spec_metadata,
        read_canonical_bucket,
        tail_included,
    )

SUMMARY_OUTPUT_FILENAME = "token_survival_summary_v2_long.csv"
DETAIL_SUMMARY_OUTPUT_FILENAME = "token_survival_detail_summary_v2_long.csv"
COMPARISON_SUMMARY_OUTPUT_FILENAME = "token_survival_comparison_summary.csv"
DETAIL_COMPARISON_SUMMARY_OUTPUT_FILENAME = "token_survival_detail_comparison_summary.csv"
DIAGNOSTIC_COMPARISON_SUMMARY_OUTPUT_FILENAME = "token_survival_diagnostic_comparison_summary.csv"
QUANT_MODE_COMPARISON_OUTPUT_FILENAME = "token_survival_quant_mode_comparison.csv"
DETAIL_QUANT_MODE_COMPARISON_OUTPUT_FILENAME = "token_survival_detail_quant_mode_comparison.csv"
DIAGNOSTIC_QUANT_MODE_COMPARISON_OUTPUT_FILENAME = "token_survival_diagnostic_quant_mode_comparison.csv"
INTERVENTION_LEVEL_SUMMARY_OUTPUT_FILENAME = "token_survival_intervention_level_summary.csv"
INTERVENTION_LEVEL_COMPARISON_OUTPUT_FILENAME = "token_survival_intervention_level_comparison.csv"
DEFAULT_SUMMARY_READ_CHUNKSIZE = 50000

ANALYSIS_GROUP_ORDER = tuple(ANALYSIS_GROUPS)
SUMMARY_FACET_COLUMNS = [
    "intervention_target",
    "intervention_layer_idx",
    "intervention_module_name",
    "trace_target_name",
    "trace_layer_idx",
    "trace_module_name",
    "target",
    "layer_idx",
    "module_name",
    "tensor_position",
    "quant_mode",
    "granularity",
    "integer_mapping_source",
    "actual_integer_mapping_available",
    "integer_mapping_is_direct_quantlinear_forward",
]
TRACE_FACET_COLUMNS = ["trace_layer_idx", "trace_module_name", "tensor_position"]
INTERVENTION_LEVEL_FACET_COLUMNS = [
    "intervention_target",
    "intervention_layer_idx",
    "intervention_module_name",
    "quant_mode",
    "granularity",
]
INTERVENTION_EMBEDDING_CONSISTENCY_KEY_COLUMNS = [
    "sample_id",
    "intervention_target",
    "intervention_layer_idx",
    "intervention_module_name",
    "quant_mode",
    "granularity",
    "qparams_hash",
]
QUANT_MODE_PAIR_ORDER = (
    ("W4A4", "A4-only"),
    ("W4A4", "W4-only"),
    ("A4-only", "W4-only"),
)
KNOWN_QUANT_MODES = ("A4-only", "W4A4", "W4-only")

_QUANT_MODE_ALIASES = {
    "a4": "A4-only",
    "a4-only": "A4-only",
    "a4_only": "A4-only",
    "activation": "A4-only",
    "activation_only": "A4-only",
    "activation-only": "A4-only",
    "w4a4": "W4A4",
    "w4-a4": "W4A4",
    "w4_a4": "W4A4",
    "w4": "W4-only",
    "w4-only": "W4-only",
    "w4_only": "W4-only",
    "weight": "W4-only",
    "weight_only": "W4-only",
    "weight-only": "W4-only",
}


@dataclass(frozen=True)
class TokenMetricSpec:
    source_column: str
    spec: MetricSpec
    comparable_across_groups: bool
    gap_enabled: bool
    weighting: str = "sample_equal"


_DEF = define_metric_spec


def _token_summary_spec(
    metric_name: str,
    *,
    source_column: str | None = None,
    metric_direction: str,
    comparable_across_groups: bool = True,
    gap_enabled: bool = True,
    metric_role: str = "main",
) -> TokenMetricSpec:
    return TokenMetricSpec(
        source_column or metric_name,
        _DEF(
            metric_name,
            metric_direction=metric_direction,
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
            metric_role=metric_role,
        ),
        comparable_across_groups,
        gap_enabled,
    )


def _sample_scalar_spec(
    metric_name: str,
    *,
    source_column: str | None = None,
    metric_direction: str,
    comparable_across_groups: bool = True,
    gap_enabled: bool = True,
    aggregation_fn: str = "nanmean",
) -> TokenMetricSpec:
    return TokenMetricSpec(
        source_column or metric_name,
        _DEF(
            metric_name,
            metric_direction=metric_direction,
            aggregation_unit="sample",
            aggregation_fn=aggregation_fn,
        ),
        comparable_across_groups,
        gap_enabled,
    )


TOKEN_METRIC_SPECS: dict[str, TokenMetricSpec] = {
    "mean_norm_retention": _token_summary_spec("mean_norm_retention", metric_direction="higher_is_better", metric_role="formal_main"),
    "topk_survival_rate": _token_summary_spec("topk_survival_rate", metric_direction="higher_is_better", metric_role="detail"),
    "token_survival_drop": _token_summary_spec("token_survival_drop", metric_direction="lower_is_better", metric_role="detail"),
    "topk_k": _token_summary_spec("topk_k", metric_direction="diagnostic", comparable_across_groups=False, gap_enabled=False, metric_role="detail"),
    "valid_token_count": _token_summary_spec("valid_token_count", metric_direction="diagnostic", comparable_across_groups=False, gap_enabled=False, metric_role="detail"),
    "survival_min_norm_retention": _token_summary_spec("survival_min_norm_retention", metric_direction="diagnostic", comparable_across_groups=False, gap_enabled=False, metric_role="detail"),
    "topk_overlap": _token_summary_spec("topk_overlap", metric_direction="higher_is_better", comparable_across_groups=False, gap_enabled=False, metric_role="formal_main"),
    "token_oer_retention": _token_summary_spec("token_oer_retention", metric_direction="higher_is_better", metric_role="formal_main"),
    "token_oer_loss": _token_summary_spec("token_oer_loss", metric_direction="lower_is_better", metric_role="formal_main"),
    "token_distortion": _token_summary_spec("token_distortion", source_column="mean_token_distortion", metric_direction="lower_is_better", metric_role="formal_main"),
    "topk_token_exact_zero_rate": _token_summary_spec("topk_token_exact_zero_rate", metric_direction="lower_is_better", metric_role="formal_main"),
    "topk_token_near_zero_pm1_rate": _token_summary_spec("topk_token_near_zero_pm1_rate", metric_direction="lower_is_better", metric_role="formal_main"),
    "topk_token_boundary_hit_rate": _token_summary_spec("topk_token_boundary_hit_rate", metric_direction="lower_is_better", metric_role="formal_main"),
    "topk_token_strict_overflow_rate": _token_summary_spec("topk_token_strict_overflow_rate", metric_direction="lower_is_better", metric_role="formal_main"),
    "high_energy_scalar_exact_zero_rate": _token_summary_spec("high_energy_scalar_exact_zero_rate", metric_direction="lower_is_better", metric_role="formal_main"),
    "high_energy_scalar_near_zero_pm1_rate": _token_summary_spec("high_energy_scalar_near_zero_pm1_rate", metric_direction="lower_is_better", metric_role="formal_main"),
    "high_energy_scalar_boundary_hit_rate": _token_summary_spec("high_energy_scalar_boundary_hit_rate", metric_direction="lower_is_better", metric_role="formal_main"),
    "high_energy_scalar_strict_overflow_rate": _token_summary_spec("high_energy_scalar_strict_overflow_rate", metric_direction="lower_is_better", metric_role="formal_main"),
    "embedding_distortion": _sample_scalar_spec("embedding_distortion", metric_direction="lower_is_better"),
}
TOKEN_METRIC_REGISTRY = build_metric_spec_registry(tuple(item.spec for item in TOKEN_METRIC_SPECS.values()))
METRIC_COLUMNS = [item.source_column for item in TOKEN_METRIC_SPECS.values()]

BLOCK_OUTPUT_FORMAL_MAIN_METRICS = {
    "mean_norm_retention",
    "token_distortion",
    "token_oer_retention",
    "token_oer_loss",
    "topk_overlap",
}
BLOCK_OUTPUT_DETAIL_METRICS = {
    "topk_survival_rate",
    "token_survival_drop",
    "topk_k",
    "valid_token_count",
    "survival_min_norm_retention",
}
INTEGER_GRID_METRICS = {
    "topk_token_exact_zero_rate",
    "topk_token_near_zero_pm1_rate",
    "topk_token_boundary_hit_rate",
    "topk_token_strict_overflow_rate",
    "high_energy_scalar_exact_zero_rate",
    "high_energy_scalar_near_zero_pm1_rate",
    "high_energy_scalar_boundary_hit_rate",
    "high_energy_scalar_strict_overflow_rate",
}
INPUT_ACTIVATION_TRIVIAL_METRICS = {
    "mean_norm_retention",
    "topk_survival_rate",
    "token_survival_drop",
    "topk_overlap",
    "token_oer_retention",
    "token_oer_loss",
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _probe_cfg(cfg: Any | None) -> Mapping[str, Any]:
    if cfg is None:
        return {}
    value = _cfg_get(cfg, "FFN_TOKEN_SURVIVAL_PROBE", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any | None, key: str, default: Any = None) -> Any:
    probe = _probe_cfg(cfg)
    if key in probe:
        return probe[key]
    if cfg is None:
        return default
    return _cfg_get(cfg, key, default)


def _dataset_name(cfg: Any) -> str:
    dataset_cfg = _cfg_get(cfg, "CURRENT_DATASET", {})
    if isinstance(dataset_cfg, Mapping):
        for key in ("NAME", "name", "DATASET_NAME", "dataset_name"):
            value = dataset_cfg.get(key)
            if value:
                return str(value)
        root = dataset_cfg.get("ROOT")
        if root is not None:
            return Path(root).name
    active = _cfg_get(cfg, "ACTIVE_DATASET", None)
    if active:
        return str(active)
    result_dir = _cfg_get(cfg, "CURRENT_RESULT_DIR", None)
    if result_dir is not None:
        return Path(result_dir).name
    raise AttributeError("Cannot resolve dataset name from cfg")


def _resolve_output_dir(cfg: Any | None = None, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if cfg is None:
        import config as project_config
        cfg = project_config
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "token_survival"


def _read_sample_table(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    return data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_quant_mode(value: Any) -> str:
    text = str(value).strip()
    return _QUANT_MODE_ALIASES.get(text.lower().replace(" ", "_"), text)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "", "nan", "<na>", "none"}:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        return bool(int(value))
    raise ValueError(f"actual_integer_mapping_available must be bool-like, got {value!r}")


def _boolish(value: Any) -> bool:
    try:
        return _coerce_bool(value)
    except ValueError:
        return False


def _first_present(frame: pd.DataFrame, candidates: Sequence[str], default: Any = np.nan) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _target_column(frame: pd.DataFrame) -> pd.Series:
    if "intervention_target" in frame.columns:
        target = frame["intervention_target"].astype(str).str.strip()
        if not target.str.lower().isin({"", "nan", "none", "<na>"}).all():
            return target
    if {"intervention_layer_idx", "intervention_module_name"}.issubset(frame.columns):
        layer = _to_numeric(frame["intervention_layer_idx"]).astype("Int64")
        module = frame["intervention_module_name"].astype(str)
        return "L" + layer.astype(str) + "_" + module
    layer = _to_numeric(frame["layer_idx"]).astype("Int64")
    return "L" + layer.astype(str) + "_" + frame["module_name"].astype(str)


def _validate_label_mapping(frame: pd.DataFrame) -> tuple[str | None, list[str]]:
    candidates = [name for name in ("label_id", "label", "primary_label") if name in frame.columns]
    warnings: list[str] = []
    for left, right in (("label_id", "label"), ("label_id", "primary_label"), ("label", "primary_label")):
        if left not in frame.columns or right not in frame.columns:
            continue
        pair = frame[[left, right]].dropna().astype(str)
        if pair.empty:
            continue
        if (pair.groupby(left)[right].nunique() > 1).any() or (pair.groupby(right)[left].nunique() > 1).any():
            raise ValueError(f"Token Survival label mapping is inconsistent between {left} and {right}")
    for name in ("label_id", "label", "primary_label"):
        if name in frame.columns:
            return name, warnings
    warnings.append("group_class_count unavailable: missing label_id/label/primary_label")
    return None, warnings


def _ensure_required_columns(frame: pd.DataFrame) -> None:
    required = ["sample_id", "bucket", "quant_mode", "layer_idx", "module_name", "tensor_position", "topk_survival_rate"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise KeyError(f"Token Survival sample table is missing required columns: {', '.join(missing)}")


def normalise_sample_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    work = sample_df.copy()
    _ensure_required_columns(work)
    shot_values = work["shot_group"] if "shot_group" in work.columns else None
    bucket_result = read_canonical_bucket(bucket=work["bucket"], shot_group=shot_values, context="token_survival sample table", allow_shot_group_fallback=False)
    buckets = [str(item) for item in bucket_result.bucket]
    work["bucket"] = buckets
    work["shot_group"] = buckets
    expected_tail = [bool(item) for item in tail_included(buckets)]
    if "tail_included" in work.columns:
        observed = [_coerce_bool(item) for item in work["tail_included"].tolist()]
        if observed != expected_tail:
            raise ValueError("Token Survival tail_included must match bucket-derived tail membership")
    work["tail_included"] = expected_tail
    work["target"] = _target_column(work)
    work["quant_mode"] = work["quant_mode"].map(_normalize_quant_mode)
    work["granularity"] = _first_present(work, ("granularity",), "per_tensor").astype(str)
    work["integer_mapping_source"] = _first_present(work, ("integer_mapping_source",), "unknown").astype(str)
    work["actual_integer_mapping_available"] = _first_present(work, ("actual_integer_mapping_available",), False).map(_coerce_bool)
    if "token_survival_drop" not in work.columns:
        work["token_survival_drop"] = 1.0 - _to_numeric(work["topk_survival_rate"])
    if "token_oer_loss" not in work.columns and "token_oer_retention" in work.columns:
        work["token_oer_loss"] = 1.0 - _to_numeric(work["token_oer_retention"])
    # Deprecated sample aliases are accepted only as compatibility inputs and
    # mapped according to Token Survival's historical local semantics.
    if "topk_token_strict_overflow_rate" not in work.columns and "topk_token_clip_rate" in work.columns:
        work["topk_token_strict_overflow_rate"] = _to_numeric(work["topk_token_clip_rate"])
    if "high_energy_scalar_strict_overflow_rate" not in work.columns and "high_energy_scalar_clip_rate" in work.columns:
        work["high_energy_scalar_strict_overflow_rate"] = _to_numeric(work["high_energy_scalar_clip_rate"])
    elif "high_energy_scalar_strict_overflow_rate" not in work.columns and "topk_clip_rate" in work.columns:
        work["high_energy_scalar_strict_overflow_rate"] = _to_numeric(work["topk_clip_rate"])
    if "topk_token_exact_zero_rate" not in work.columns and "topk_token_near_zero_rate" in work.columns:
        work["topk_token_exact_zero_rate"] = _to_numeric(work["topk_token_near_zero_rate"])
    if "high_energy_scalar_exact_zero_rate" not in work.columns and "high_energy_scalar_near_zero_rate" in work.columns:
        work["high_energy_scalar_exact_zero_rate"] = _to_numeric(work["high_energy_scalar_near_zero_rate"])
    elif "high_energy_scalar_exact_zero_rate" not in work.columns and "topk_near_zero_rate" in work.columns:
        work["high_energy_scalar_exact_zero_rate"] = _to_numeric(work["topk_near_zero_rate"])
    for column in {spec.source_column for spec in TOKEN_METRIC_SPECS.values()} | {"layer_idx", "num_tokens", "topk", "hidden_dim"}:
        if column in work.columns:
            work[column] = _to_numeric(work[column])
    class_field, warnings = _validate_label_mapping(work)
    work.attrs["class_field"] = class_field
    work.attrs["label_mapping_warnings"] = warnings
    if "topk_k" not in work.columns and "topk" in work.columns:
        work["topk_k"] = work["topk"]
    if "valid_token_count" not in work.columns and "num_tokens" in work.columns:
        work["valid_token_count"] = work["num_tokens"]
    if "survival_min_norm_retention" not in work.columns:
        work["survival_min_norm_retention"] = np.nan
    if "intervention_target" not in work.columns:
        work["intervention_target"] = _target_column(work)
    if "intervention_layer_idx" not in work.columns:
        if "target" in work.columns:
            extracted = work["target"].astype(str).str.extract(r"^L(?P<layer>\d+)_", expand=True)["layer"]
            work["intervention_layer_idx"] = pd.to_numeric(extracted, errors="coerce")
        else:
            work["intervention_layer_idx"] = np.nan
    if "intervention_module_name" not in work.columns:
        if "target" in work.columns:
            extracted = work["target"].astype(str).str.extract(r"^L\d+_(?P<module>.+)$", expand=True)["module"]
            work["intervention_module_name"] = extracted.where(extracted.notna(), "")
        else:
            work["intervention_module_name"] = ""
    if "trace_layer_idx" not in work.columns:
        work["trace_layer_idx"] = work["layer_idx"]
    if "trace_module_name" not in work.columns:
        work["trace_module_name"] = work["module_name"]
    if "trace_target_name" not in work.columns:
        work["trace_target_name"] = (
            "Layer" + _to_numeric(work["trace_layer_idx"]).astype("Int64").astype(str) + "." + work["trace_module_name"].astype(str) + "." + work["tensor_position"].astype(str)
        )
    if "integer_mapping_is_direct_quantlinear_forward" not in work.columns:
        if "direct_quantlinear_forward_grid" in work.columns:
            work["integer_mapping_is_direct_quantlinear_forward"] = work["direct_quantlinear_forward_grid"].map(_boolish)
        else:
            work["integer_mapping_is_direct_quantlinear_forward"] = False
    else:
        work["integer_mapping_is_direct_quantlinear_forward"] = work["integer_mapping_is_direct_quantlinear_forward"].map(_boolish)
    return work


def _selected(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return frame.loc[analysis_group_mask(frame["bucket"], group)]


def _group_value(frame: pd.DataFrame, metric_name: str, group: str) -> float:
    item = TOKEN_METRIC_SPECS[metric_name]
    return aggregate_metric_for_analysis_group_from_raw_records(
        frame[item.source_column],
        frame["bucket"],
        group,
        spec=item.spec,
        record_unit=item.spec.aggregation_unit,
    )


def _class_count(frame: pd.DataFrame, class_field: str | None) -> float:
    if class_field is None or class_field not in frame.columns:
        return float("nan")
    if len(frame) == 0:
        return 0.0
    values = frame[class_field].dropna().astype(str)
    return float(values.nunique())


def _sample_count(frame: pd.DataFrame) -> int:
    return int(frame["sample_id"].dropna().astype(str).nunique()) if "sample_id" in frame.columns else int(len(frame))


def _is_direct_quantlinear_forward_grid(facet: Mapping[str, Any]) -> bool:
    return (
        str(facet.get("integer_mapping_source", "")).strip() == "quantlinear_forward"
        and _boolish(facet.get("integer_mapping_is_direct_quantlinear_forward", False))
    )


def _metric_context_metadata(metric_name: str, facet: Mapping[str, Any]) -> dict[str, Any]:
    position = str(facet.get("tensor_position", "")).strip()
    result = {
        "include_in_summary": True,
        "summary_scope": "trace_level",
        "metric_family": "token_flow" if position == "block_output" else "input_integer_grid",
        "trace_invariant": False,
        "do_not_average_over_trace_layer": False,
        "metric_role": TOKEN_METRIC_SPECS[metric_name].spec.metric_role,
        "metric_valid": True,
        "metric_valid_for_paper": True,
        "not_applicable_reason": "",
    }
    if position == "block_output":
        if metric_name in BLOCK_OUTPUT_FORMAL_MAIN_METRICS:
            result["metric_role"] = "formal_main"
            return result
        if metric_name in BLOCK_OUTPUT_DETAIL_METRICS:
            result["metric_role"] = "detail"
            return result
        if metric_name in INTEGER_GRID_METRICS:
            result.update(
                {
                    "metric_family": "input_integer_grid",
                    "metric_role": "not_applicable",
                    "metric_valid": False,
                    "metric_valid_for_paper": False,
                    "not_applicable_reason": "block_output_has_no_activation_integer_grid",
                }
            )
            return result
        result["include_in_summary"] = False
        return result

    if position == "input_activation":
        if metric_name in INTEGER_GRID_METRICS:
            if _is_direct_quantlinear_forward_grid(facet):
                result["metric_role"] = "formal_main"
                return result
            source = str(facet.get("integer_mapping_source", "")).strip()
            if source in {"calibration_qparams_reconstructed_from_quant_input", "reconstructed_from_activation_and_qparams"}:
                result.update(
                    {
                        "metric_role": "diagnostic",
                        "metric_valid": False,
                        "metric_valid_for_paper": False,
                        "not_applicable_reason": "reconstructed_grid_not_valid_for_paper",
                    }
                )
                return result
            result["include_in_summary"] = False
            return result
        if metric_name in INPUT_ACTIVATION_TRIVIAL_METRICS:
            result["include_in_summary"] = False
            return result
        result["include_in_summary"] = False
        return result

    result["include_in_summary"] = False
    return result


def _base_metric_row(facet: Mapping[str, Any], metric_name: str, context: Mapping[str, Any]) -> dict[str, Any]:
    item = TOKEN_METRIC_SPECS[metric_name]
    meta = metric_spec_metadata(item.spec)
    meta["metric_role"] = str(context["metric_role"])
    return {
        "group_protocol_version": GROUP_PROTOCOL_VERSION,
        "group_protocol_name": GROUP_PROTOCOL_NAME,
        **dict(facet),
        **meta,
        "summary_scope": str(context.get("summary_scope", "trace_level")),
        "metric_family": str(context.get("metric_family", "")),
        "trace_invariant": bool(context.get("trace_invariant", False)),
        "do_not_average_over_trace_layer": bool(context.get("do_not_average_over_trace_layer", False)),
        "metric_valid": bool(context["metric_valid"]),
        "metric_valid_for_paper": bool(context["metric_valid_for_paper"]),
        "not_applicable_reason": str(context.get("not_applicable_reason", "")),
        "source_column": item.source_column,
        "weighting": item.weighting,
        "comparable_across_groups": bool(item.comparable_across_groups),
        "gap_enabled": bool(item.gap_enabled),
    }


def _include_in_formal_trace_summary(context: Mapping[str, Any]) -> bool:
    return (
        bool(context.get("include_in_summary", False))
        and str(context.get("metric_role", "")) == "formal_main"
        and bool(context.get("metric_valid", False))
        and bool(context.get("metric_valid_for_paper", False))
    )


def _include_in_detail_trace_summary(context: Mapping[str, Any]) -> bool:
    return (
        bool(context.get("include_in_summary", False))
        and str(context.get("metric_role", "")) == "detail"
        and bool(context.get("metric_valid", False))
    )


def _build_summary_v2_long(
    sample_df: pd.DataFrame,
    *,
    include_predicate: Any,
) -> pd.DataFrame:
    work = normalise_sample_table(sample_df)
    class_field = work.attrs.get("class_field")
    warnings = ";".join(work.attrs.get("label_mapping_warnings", []))
    rows: list[dict[str, Any]] = []
    for facet_key, facet in work.groupby(SUMMARY_FACET_COLUMNS, dropna=False, sort=True):
        facet_dict = dict(zip(SUMMARY_FACET_COLUMNS, facet_key))
        for metric_name, item in TOKEN_METRIC_SPECS.items():
            if item.source_column not in facet.columns:
                continue
            context = _metric_context_metadata(metric_name, facet_dict)
            if not include_predicate(context):
                continue
            base = _base_metric_row(facet_dict, metric_name, context)
            for group in ANALYSIS_GROUP_ORDER:
                selected = _selected(facet, group)
                value = float("nan") if not bool(context["metric_valid"]) else _group_value(facet, metric_name, group)
                rows.append({
                    "target": facet_dict["target"],
                    "bucket": "",
                    "shot_group": "",
                    "analysis_group": group,
                    "value": value,
                    "group_sample_count": _sample_count(selected),
                    "group_class_count": _class_count(selected, class_field),
                    "class_count_source": class_field or "",
                    "class_count_warning": warnings,
                    **{key: value for key, value in base.items() if key != "target"},
                })
    return pd.DataFrame(rows)


def build_summary_v2_long(sample_df: pd.DataFrame) -> pd.DataFrame:
    return _build_summary_v2_long(sample_df, include_predicate=_include_in_formal_trace_summary)


def build_detail_summary_v2_long(sample_df: pd.DataFrame) -> pd.DataFrame:
    return _build_summary_v2_long(sample_df, include_predicate=_include_in_detail_trace_summary)


def build_comparison_summary(
    sample_df: pd.DataFrame,
    *,
    metric_role: str = "formal_main",
    require_gap_comparable: bool = False,
    require_valid_for_paper: bool = True,
) -> pd.DataFrame:
    if str(metric_role) == "detail":
        summary = build_detail_summary_v2_long(sample_df)
    else:
        summary = build_summary_v2_long(sample_df)
    return _build_comparison_summary_from_summary(
        summary,
        metric_role=metric_role,
        require_gap_comparable=require_gap_comparable,
        require_valid_for_paper=require_valid_for_paper,
    )


def build_quant_mode_comparison(
    summary_v2_long: pd.DataFrame,
    *,
    metric_role: str = "formal_main",
    require_gap_comparable: bool = False,
    require_valid_for_paper: bool = True,
) -> pd.DataFrame:
    if summary_v2_long.empty:
        return pd.DataFrame()
    summary_v2_long = _filtered_summary_for_comparison(
        summary_v2_long,
        metric_role=metric_role,
        require_gap_comparable=require_gap_comparable,
        require_valid_for_paper=require_valid_for_paper,
    )
    if summary_v2_long.empty:
        return pd.DataFrame()
    id_cols = [column for column in SUMMARY_FACET_COLUMNS if column != "quant_mode"] + ["metric_name", "analysis_group"]
    rows: list[dict[str, Any]] = []
    for key_values, group in summary_v2_long.groupby(id_cols, dropna=False, sort=True):
        if len(id_cols) == 1 and not isinstance(key_values, tuple):
            key_values = (key_values,)
        key = dict(zip(id_cols, key_values))
        by_mode = {str(row["quant_mode"]): row for _, row in group.iterrows()}
        for compare_mode, base_mode in QUANT_MODE_PAIR_ORDER:
            if compare_mode not in by_mode or base_mode not in by_mode:
                continue
            compare = by_mode[compare_mode]
            base = by_mode[base_mode]
            rows.append({
                **key,
                "comparison_name": f"{compare_mode}_minus_{base_mode}",
                "base_quant_mode": base_mode,
                "compare_quant_mode": compare_mode,
                "base_value": float(base["value"]),
                "compare_value": float(compare["value"]),
                "delta_value": float(compare["value"]) - float(base["value"]),
                "metric_direction": compare["metric_direction"],
                "aggregation_unit": compare["aggregation_unit"],
                "aggregation_fn": compare["aggregation_fn"],
                "metric_role": compare.get("metric_role", ""),
                "metric_valid": bool(_boolish(compare.get("metric_valid", True))),
                "metric_valid_for_paper": bool(_boolish(compare.get("metric_valid_for_paper", True))),
                "metric_family": compare.get("metric_family", ""),
                "summary_scope": compare.get("summary_scope", ""),
                "trace_invariant": bool(_boolish(compare.get("trace_invariant", False))),
                "do_not_average_over_trace_layer": bool(_boolish(compare.get("do_not_average_over_trace_layer", False))),
                "source_column": compare.get("source_column", ""),
                "weighting": compare.get("weighting", "sample_equal"),
                "gap_enabled": bool(_boolish(compare.get("gap_enabled", True))),
                "comparable_across_groups": bool(_boolish(compare.get("comparable_across_groups", True))),
                "group_protocol_version": GROUP_PROTOCOL_VERSION,
                "group_protocol_name": GROUP_PROTOCOL_NAME,
            })
    return pd.DataFrame(rows)


@dataclass
class _StreamingMetricState:
    facet: dict[str, Any]
    metric_name: str
    context: dict[str, Any]
    analysis_group: str
    value_sum: float = 0.0
    value_count: int = 0
    sample_count: int = 0
    class_values: set[str] = field(default_factory=set)


def _stream_key_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stream_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _analysis_groups_for_bucket(bucket: Any) -> tuple[str, ...]:
    text = str(bucket)
    groups = [text] if text in {"many", "medium", "few"} else []
    if text in {"medium", "few"}:
        groups.append("tail")
    return tuple(group for group in ANALYSIS_GROUP_ORDER if group in groups)


def _row_class_value(row: pd.Series, class_field: str | None) -> str | None:
    if class_field is None or class_field not in row.index:
        return None
    value = row[class_field]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _normalise_summary_chunksize(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    chunksize = int(value)
    return None if chunksize <= 0 else chunksize


def _summary_read_chunksize(cfg: Any | None) -> int | None:
    return _normalise_summary_chunksize(_setting_get(cfg, "SUMMARY_READ_CHUNKSIZE", DEFAULT_SUMMARY_READ_CHUNKSIZE))


def _iter_sample_table_chunks(path: str | Path, *, chunksize: int | None) -> Iterable[pd.DataFrame]:
    if chunksize is None:
        return (pd.read_csv(path),)
    return pd.read_csv(path, chunksize=chunksize)


def _export_diagnostic_comparison(cfg: Any | None) -> bool:
    return bool(_setting_get(cfg, "EXPORT_DIAGNOSTIC_COMPARISON", False))


def _filtered_summary_for_comparison(
    summary_v2_long: pd.DataFrame,
    *,
    metric_role: str,
    require_gap_comparable: bool,
    require_valid_for_paper: bool,
) -> pd.DataFrame:
    if summary_v2_long.empty:
        return pd.DataFrame()
    work = summary_v2_long.copy()
    if "metric_role" in work.columns:
        work = work[work["metric_role"].astype(str).eq(str(metric_role))].copy()
    else:
        work = work.iloc[0:0].copy()
    if "metric_valid" in work.columns:
        work = work[work["metric_valid"].map(_boolish)].copy()
    if require_valid_for_paper and "metric_valid_for_paper" in work.columns:
        work = work[work["metric_valid_for_paper"].map(_boolish)].copy()
    if require_gap_comparable:
        if "gap_enabled" in work.columns:
            work = work[work["gap_enabled"].map(_boolish)].copy()
        else:
            work = work.iloc[0:0].copy()
        if "comparable_across_groups" in work.columns:
            work = work[work["comparable_across_groups"].map(_boolish)].copy()
        else:
            work = work.iloc[0:0].copy()
    return work


def _build_comparison_summary_from_summary(
    summary_v2_long: pd.DataFrame,
    *,
    metric_role: str = "formal_main",
    require_gap_comparable: bool = False,
    require_valid_for_paper: bool = True,
) -> pd.DataFrame:
    if summary_v2_long.empty:
        return pd.DataFrame()
    work = _filtered_summary_for_comparison(
        summary_v2_long,
        metric_role=metric_role,
        require_gap_comparable=require_gap_comparable,
        require_valid_for_paper=require_valid_for_paper,
    )
    if work.empty:
        return pd.DataFrame()
    id_cols = SUMMARY_FACET_COLUMNS + ["metric_name"]
    rows: list[dict[str, Any]] = []
    for key_values, group in work.groupby(id_cols, dropna=False, sort=True):
        if len(id_cols) == 1 and not isinstance(key_values, tuple):
            key_values = (key_values,)
        facet_with_metric = dict(zip(id_cols, key_values))
        metric_name = str(facet_with_metric.pop("metric_name"))
        item = TOKEN_METRIC_SPECS[metric_name]
        by_group = {str(row["analysis_group"]): row for _, row in group.iterrows()}
        values = {name: float(by_group[name]["value"]) if name in by_group else float("nan") for name in ANALYSIS_GROUP_ORDER}
        if item.gap_enabled:
            gaps = compute_tail_and_few_gaps(values, metric_direction=item.spec.metric_direction)
        else:
            gaps = {"tail_vs_many_gap": float("nan"), "few_vs_many_gap": float("nan")}
        first = next(iter(by_group.values()))
        context = _metric_context_metadata(metric_name, facet_with_metric)
        row = {
            **_base_metric_row(facet_with_metric, metric_name, context),
            "many_value": values.get("many", float("nan")),
            "medium_value": values.get("medium", float("nan")),
            "few_value": values.get("few", float("nan")),
            "tail_value": values.get("tail", float("nan")),
            "tail_vs_many_gap": gaps["tail_vs_many_gap"],
            "few_vs_many_gap": gaps["few_vs_many_gap"],
        }
        for analysis_group in ANALYSIS_GROUP_ORDER:
            source = by_group.get(analysis_group)
            row[f"{analysis_group}_sample_count"] = 0 if source is None else int(source.get("group_sample_count", 0))
            row[f"{analysis_group}_class_count"] = float("nan") if source is None else float(source.get("group_class_count", float("nan")))
        if "class_count_source" in first:
            row["class_count_source"] = first.get("class_count_source", "")
        if "class_count_warning" in first:
            row["class_count_warning"] = first.get("class_count_warning", "")
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class _EmbeddingSampleState:
    key: tuple[Any, ...]
    sample_id: str
    intervention_target: Any
    intervention_layer_idx: Any
    intervention_module_name: Any
    quant_mode: str
    granularity: str
    qparams_hash: str
    value_min: float
    value_max: float
    canonical_value: float
    canonical_trace_layer_idx: int
    bucket: str
    class_value: str | None
    experiment_signature: str
    qparams_path: str
    trace_layers: set[int] = field(default_factory=set)
    row_count: int = 0

    def observe(self, row: pd.Series, *, value: float, trace_layer_idx: int, class_value: str | None) -> None:
        self.value_min = min(self.value_min, value)
        self.value_max = max(self.value_max, value)
        self.trace_layers.add(int(trace_layer_idx))
        self.row_count += 1
        if int(trace_layer_idx) < int(self.canonical_trace_layer_idx):
            self.canonical_value = float(value)
            self.canonical_trace_layer_idx = int(trace_layer_idx)
            self.bucket = str(row["bucket"])
            self.class_value = class_value
            self.experiment_signature = str(row.get("experiment_signature", ""))
            self.qparams_path = str(row.get("qparams_path", ""))

    def check_consistent(self) -> None:
        if not np.allclose([self.value_min], [self.value_max], rtol=1e-6, atol=1e-8, equal_nan=False):
            raise ValueError(
                "embedding_distortion must be trace-invariant within "
                "sample_id/intervention_target/intervention_layer/module/quant_mode/granularity/qparams_hash; "
                f"key={self.key}, min={self.value_min}, max={self.value_max}"
            )


@dataclass
class _InterventionSummaryState:
    facet: dict[str, Any]
    analysis_group: str
    value_sum: float = 0.0
    value_count: int = 0
    sample_count: int = 0
    class_values: set[str] = field(default_factory=set)
    experiment_signatures: set[str] = field(default_factory=set)
    qparams_hashes: set[str] = field(default_factory=set)
    qparams_paths: set[str] = field(default_factory=set)
    canonical_trace_layer_min: int | None = None
    canonical_trace_layer_max: int | None = None
    duplicate_count_min: int | None = None
    duplicate_count_max: int | None = None
    duplicate_count_sum: int = 0

    def update(self, sample: _EmbeddingSampleState) -> None:
        self.sample_count += 1
        if sample.class_value is not None:
            self.class_values.add(str(sample.class_value))
        if math.isfinite(sample.canonical_value):
            self.value_sum += float(sample.canonical_value)
            self.value_count += 1
        if sample.experiment_signature:
            self.experiment_signatures.add(str(sample.experiment_signature))
        if sample.qparams_hash:
            self.qparams_hashes.add(str(sample.qparams_hash))
        if sample.qparams_path:
            self.qparams_paths.add(str(sample.qparams_path))
        layer = int(sample.canonical_trace_layer_idx)
        self.canonical_trace_layer_min = layer if self.canonical_trace_layer_min is None else min(self.canonical_trace_layer_min, layer)
        self.canonical_trace_layer_max = layer if self.canonical_trace_layer_max is None else max(self.canonical_trace_layer_max, layer)
        duplicate_count = int(sample.row_count)
        self.duplicate_count_min = duplicate_count if self.duplicate_count_min is None else min(self.duplicate_count_min, duplicate_count)
        self.duplicate_count_max = duplicate_count if self.duplicate_count_max is None else max(self.duplicate_count_max, duplicate_count)
        self.duplicate_count_sum += duplicate_count


def _joined_unique(values: set[str]) -> str:
    return ";".join(sorted(str(item) for item in values if str(item) != ""))


def _trace_layer_value(row: pd.Series) -> int:
    value = row["trace_layer_idx"] if "trace_layer_idx" in row.index else row["layer_idx"]
    numeric = _stream_float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"missing trace_layer_idx for embedding_distortion row: {row.to_dict()}")
    return int(numeric)


def _embedding_sample_key(row: pd.Series) -> tuple[Any, ...]:
    return tuple(_stream_key_value(row.get(column, "")) for column in INTERVENTION_EMBEDDING_CONSISTENCY_KEY_COLUMNS)


def _new_embedding_sample_state(row: pd.Series, *, key: tuple[Any, ...], value: float, trace_layer_idx: int, class_value: str | None) -> _EmbeddingSampleState:
    return _EmbeddingSampleState(
        key=key,
        sample_id=str(row["sample_id"]),
        intervention_target=_stream_key_value(row["intervention_target"]),
        intervention_layer_idx=_stream_key_value(row["intervention_layer_idx"]),
        intervention_module_name=_stream_key_value(row["intervention_module_name"]),
        quant_mode=str(row["quant_mode"]),
        granularity=str(row["granularity"]),
        qparams_hash=str(row.get("qparams_hash", "")),
        value_min=float(value),
        value_max=float(value),
        canonical_value=float(value),
        canonical_trace_layer_idx=int(trace_layer_idx),
        bucket=str(row["bucket"]),
        class_value=class_value,
        experiment_signature=str(row.get("experiment_signature", "")),
        qparams_path=str(row.get("qparams_path", "")),
        trace_layers={int(trace_layer_idx)},
        row_count=1,
    )


def _observe_embedding_rows(sample_states: dict[tuple[Any, ...], _EmbeddingSampleState], frame: pd.DataFrame) -> None:
    class_field = frame.attrs.get("class_field")
    if "embedding_distortion" not in frame.columns:
        return
    work = frame[frame["tensor_position"].astype(str).eq("block_output")].copy()
    if work.empty:
        return
    for _, row in work.iterrows():
        value = _stream_float(row["embedding_distortion"])
        if not math.isfinite(value):
            raise ValueError(f"embedding_distortion must be finite for intervention-level auxiliary summary: {row.get('sample_id', '')}")
        trace_layer_idx = _trace_layer_value(row)
        key = _embedding_sample_key(row)
        class_value = _row_class_value(row, class_field)
        state = sample_states.get(key)
        if state is None:
            sample_states[key] = _new_embedding_sample_state(
                row,
                key=key,
                value=value,
                trace_layer_idx=trace_layer_idx,
                class_value=class_value,
            )
        else:
            state.observe(row, value=value, trace_layer_idx=trace_layer_idx, class_value=class_value)


def _intervention_summary_base(metric_name: str = "embedding_distortion") -> dict[str, Any]:
    item = TOKEN_METRIC_SPECS[metric_name]
    meta = metric_spec_metadata(item.spec)
    meta["metric_role"] = "formal_auxiliary"
    return {
        "group_protocol_version": GROUP_PROTOCOL_VERSION,
        "group_protocol_name": GROUP_PROTOCOL_NAME,
        **meta,
        "summary_scope": "intervention_level",
        "metric_family": "final_embedding_aux",
        "trace_invariant": True,
        "do_not_average_over_trace_layer": True,
        "metric_valid": True,
        "metric_valid_for_paper": True,
        "not_applicable_reason": "",
        "source_column": item.source_column,
        "weighting": item.weighting,
        "comparable_across_groups": bool(item.comparable_across_groups),
        "gap_enabled": bool(item.gap_enabled),
    }


def _update_intervention_summary_states(
    states: dict[tuple[Any, ...], _InterventionSummaryState],
    sample_states: Mapping[tuple[Any, ...], _EmbeddingSampleState],
) -> None:
    for sample in sample_states.values():
        sample.check_consistent()
        facet = {
            "intervention_target": sample.intervention_target,
            "intervention_layer_idx": sample.intervention_layer_idx,
            "intervention_module_name": sample.intervention_module_name,
            "quant_mode": sample.quant_mode,
            "granularity": sample.granularity,
        }
        for analysis_group in _analysis_groups_for_bucket(sample.bucket):
            key = tuple(facet[column] for column in INTERVENTION_LEVEL_FACET_COLUMNS) + (analysis_group,)
            state = states.get(key)
            if state is None:
                state = _InterventionSummaryState(facet=dict(facet), analysis_group=analysis_group)
                states[key] = state
            state.update(sample)


def _intervention_summary_frame(
    states: Mapping[tuple[Any, ...], _InterventionSummaryState],
    *,
    class_field: str | None,
    label_mapping_warnings: Sequence[str],
) -> pd.DataFrame:
    warnings = ";".join(sorted(str(item) for item in label_mapping_warnings if str(item)))
    rows: list[dict[str, Any]] = []
    base = _intervention_summary_base()
    for key in sorted(states, key=lambda item: tuple(str(part) for part in item)):
        state = states[key]
        value = float("nan") if state.value_count <= 0 else float(state.value_sum / state.value_count)
        duplicate_min = state.duplicate_count_min
        duplicate_max = state.duplicate_count_max
        canonical_min = state.canonical_trace_layer_min
        canonical_max = state.canonical_trace_layer_max
        rows.append({
            **base,
            **state.facet,
            "metric_name": "embedding_distortion",
            "analysis_group": state.analysis_group,
            "value": value,
            "group_sample_count": int(state.sample_count),
            "group_class_count": float(len(state.class_values)) if class_field is not None else float("nan"),
            "class_count_source": class_field or "",
            "class_count_warning": warnings,
            "experiment_signature": _joined_unique(state.experiment_signatures),
            "qparams_hash": _joined_unique(state.qparams_hashes),
            "qparams_path": _joined_unique(state.qparams_paths),
            "canonical_trace_layer_idx": canonical_min if canonical_min == canonical_max else "",
            "canonical_trace_layer_idx_min": "" if canonical_min is None else int(canonical_min),
            "canonical_trace_layer_idx_max": "" if canonical_max is None else int(canonical_max),
            "dedup_source": "min_trace_layer_idx_after_consistency_check",
            "trace_invariant_check_passed": True,
            "num_duplicate_trace_rows_checked": duplicate_min if duplicate_min == duplicate_max else "",
            "num_duplicate_trace_rows_checked_min": "" if duplicate_min is None else int(duplicate_min),
            "num_duplicate_trace_rows_checked_max": "" if duplicate_max is None else int(duplicate_max),
            "num_duplicate_trace_rows_checked_mean": (
                float("nan") if state.sample_count <= 0 else float(state.duplicate_count_sum / state.sample_count)
            ),
        })
    return pd.DataFrame(rows)


def _filter_formal_auxiliary_intervention(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = ("metric_role", "metric_valid", "metric_valid_for_paper")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return frame.iloc[0:0].copy()
    mask = (
        frame["metric_role"].astype(str).eq("formal_auxiliary")
        & frame["metric_valid"].map(_boolish)
        & frame["metric_valid_for_paper"].map(_boolish)
    )
    return frame[mask].copy()


def _shard_intervention_group_key(path: str | Path) -> tuple[Any, ...] | None:
    try:
        head = pd.read_csv(path, nrows=1)
    except pd.errors.EmptyDataError:
        return None
    if head.empty or "embedding_distortion" not in head.columns:
        return None
    head = normalise_sample_table(head)
    row = head.iloc[0]
    if str(row["tensor_position"]) != "block_output":
        return None
    columns = [*INTERVENTION_LEVEL_FACET_COLUMNS, "qparams_hash"]
    return tuple(_stream_key_value(row.get(column, "")) for column in columns)


def build_intervention_level_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    work = normalise_sample_table(sample_df)
    label_mapping_warnings = list(work.attrs.get("label_mapping_warnings", []))
    class_field = work.attrs.get("class_field")
    states: dict[tuple[Any, ...], _InterventionSummaryState] = {}
    group_columns = [*INTERVENTION_LEVEL_FACET_COLUMNS, "qparams_hash"]
    block_rows = work[work["tensor_position"].astype(str).eq("block_output")].copy()
    if not block_rows.empty and "embedding_distortion" in block_rows.columns:
        for _group_key, group in block_rows.groupby(group_columns, dropna=False, sort=True):
            sample_states: dict[tuple[Any, ...], _EmbeddingSampleState] = {}
            group.attrs["class_field"] = class_field
            _observe_embedding_rows(sample_states, group)
            _update_intervention_summary_states(states, sample_states)
    result = _filter_formal_auxiliary_intervention(
        _intervention_summary_frame(states, class_field=class_field, label_mapping_warnings=label_mapping_warnings)
    )
    result.attrs["label_mapping_warnings"] = label_mapping_warnings
    return result


def build_intervention_level_summary_from_shards(
    shard_csv_paths: Sequence[str | Path],
    *,
    chunksize: int | None = DEFAULT_SUMMARY_READ_CHUNKSIZE,
) -> pd.DataFrame:
    grouped_paths: dict[tuple[Any, ...], list[Path]] = {}
    for path in shard_csv_paths:
        group_key = _shard_intervention_group_key(path)
        if group_key is not None:
            grouped_paths.setdefault(group_key, []).append(Path(path))

    states: dict[tuple[Any, ...], _InterventionSummaryState] = {}
    label_mapping_warnings: set[str] = set()
    class_field: str | None = None
    read_chunksize = _normalise_summary_chunksize(chunksize)
    for _group_key, paths in sorted(grouped_paths.items(), key=lambda item: tuple(str(part) for part in item[0])):
        sample_states: dict[tuple[Any, ...], _EmbeddingSampleState] = {}
        for path in sorted(paths, key=lambda item: str(item)):
            for chunk in _iter_sample_table_chunks(path, chunksize=read_chunksize):
                frame = normalise_sample_table(chunk)
                for warning in frame.attrs.get("label_mapping_warnings", []):
                    if warning:
                        label_mapping_warnings.add(str(warning))
                if class_field is None and frame.attrs.get("class_field"):
                    class_field = str(frame.attrs["class_field"])
                _observe_embedding_rows(sample_states, frame)
        _update_intervention_summary_states(states, sample_states)
    result = _filter_formal_auxiliary_intervention(
        _intervention_summary_frame(states, class_field=class_field, label_mapping_warnings=sorted(label_mapping_warnings))
    )
    result.attrs["label_mapping_warnings"] = sorted(label_mapping_warnings)
    return result


def build_intervention_level_comparison_summary(intervention_summary: pd.DataFrame) -> pd.DataFrame:
    intervention_summary = _filter_formal_auxiliary_intervention(intervention_summary)
    if intervention_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    id_cols = INTERVENTION_LEVEL_FACET_COLUMNS + ["metric_name"]
    for key_values, group in intervention_summary.groupby(id_cols, dropna=False, sort=True):
        if len(id_cols) == 1 and not isinstance(key_values, tuple):
            key_values = (key_values,)
        key = dict(zip(id_cols, key_values))
        by_group = {str(row["analysis_group"]): row for _, row in group.iterrows()}
        values = {name: float(by_group[name]["value"]) if name in by_group else float("nan") for name in ANALYSIS_GROUP_ORDER}
        first = next(iter(by_group.values()))
        gaps = compute_tail_and_few_gaps(values, metric_direction=str(first.get("metric_direction", "lower_is_better")))
        row = {
            **{name: first.get(name, "") for name in first.index if name not in {"analysis_group", "value", "group_sample_count", "group_class_count"}},
            **key,
            "many_value": values.get("many", float("nan")),
            "medium_value": values.get("medium", float("nan")),
            "few_value": values.get("few", float("nan")),
            "tail_value": values.get("tail", float("nan")),
            "tail_vs_many_gap": gaps["tail_vs_many_gap"],
            "few_vs_many_gap": gaps["few_vs_many_gap"],
        }
        for analysis_group in ANALYSIS_GROUP_ORDER:
            source = by_group.get(analysis_group)
            row[f"{analysis_group}_sample_count"] = 0 if source is None else int(source.get("group_sample_count", 0))
            row[f"{analysis_group}_class_count"] = float("nan") if source is None else float(source.get("group_class_count", float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison_output_tables(
    summary: pd.DataFrame,
    *,
    cfg: Any | None,
    detail_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    tables = {
        "comparison": _build_comparison_summary_from_summary(
            summary,
            metric_role="formal_main",
            require_gap_comparable=False,
            require_valid_for_paper=True,
        ),
        "detail_comparison": _build_comparison_summary_from_summary(
            detail_summary,
            metric_role="detail",
            require_gap_comparable=True,
            require_valid_for_paper=False,
        ),
        "quant_mode_comparison": build_quant_mode_comparison(
            summary,
            metric_role="formal_main",
            require_gap_comparable=False,
            require_valid_for_paper=True,
        ),
        "detail_quant_mode_comparison": build_quant_mode_comparison(
            detail_summary,
            metric_role="detail",
            require_gap_comparable=True,
            require_valid_for_paper=False,
        ),
    }
    if _export_diagnostic_comparison(cfg):
        tables["diagnostic_comparison"] = _build_comparison_summary_from_summary(
            summary,
            metric_role="diagnostic",
            require_gap_comparable=True,
            require_valid_for_paper=False,
        )
        tables["diagnostic_quant_mode_comparison"] = build_quant_mode_comparison(
            summary,
            metric_role="diagnostic",
            require_gap_comparable=True,
            require_valid_for_paper=False,
        )
    else:
        tables["diagnostic_comparison"] = pd.DataFrame()
        tables["diagnostic_quant_mode_comparison"] = pd.DataFrame()
    return tables


def _set_common_attrs(
    frames: Sequence[pd.DataFrame],
    *,
    observed: Sequence[str],
    available: Sequence[str],
    missing: Sequence[str],
    label_mapping_warnings: Sequence[str],
    num_sample_rows: int | None = None,
    extra_tables: Mapping[str, pd.DataFrame] | None = None,
) -> None:
    for frame in frames:
        frame.attrs["observed_quant_modes"] = list(observed)
        frame.attrs["available_quant_mode_comparisons"] = list(available)
        frame.attrs["missing_quant_modes"] = list(missing)
        frame.attrs["label_mapping_warnings"] = list(label_mapping_warnings)
        if num_sample_rows is not None:
            frame.attrs["num_sample_rows"] = int(num_sample_rows)
        if extra_tables is not None:
            frame.attrs["extra_tables"] = dict(extra_tables)


def _save_extra_summary_outputs(
    *,
    target_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    export_diagnostic: bool,
) -> None:
    tables["detail_summary"].to_csv(target_dir / DETAIL_SUMMARY_OUTPUT_FILENAME, index=False)
    tables["detail_comparison"].to_csv(target_dir / DETAIL_COMPARISON_SUMMARY_OUTPUT_FILENAME, index=False)
    tables["detail_quant_mode_comparison"].to_csv(target_dir / DETAIL_QUANT_MODE_COMPARISON_OUTPUT_FILENAME, index=False)
    tables["intervention_level_summary"].to_csv(target_dir / INTERVENTION_LEVEL_SUMMARY_OUTPUT_FILENAME, index=False)
    tables["intervention_level_comparison"].to_csv(target_dir / INTERVENTION_LEVEL_COMPARISON_OUTPUT_FILENAME, index=False)
    if export_diagnostic:
        tables["diagnostic_comparison"].to_csv(target_dir / DIAGNOSTIC_COMPARISON_SUMMARY_OUTPUT_FILENAME, index=False)
        tables["diagnostic_quant_mode_comparison"].to_csv(target_dir / DIAGNOSTIC_QUANT_MODE_COMPARISON_OUTPUT_FILENAME, index=False)


def _build_summary_v2_long_from_shards(
    shard_csv_paths: Sequence[str | Path],
    *,
    chunksize: int | None = DEFAULT_SUMMARY_READ_CHUNKSIZE,
    include_predicate: Any,
) -> pd.DataFrame:
    states: dict[tuple[Any, ...], _StreamingMetricState] = {}
    observed_modes: set[str] = set()
    label_mapping_warnings: set[str] = set()
    class_field: str | None = None
    total_rows = 0
    read_chunksize = _normalise_summary_chunksize(chunksize)

    for shard_path in shard_csv_paths:
        for shard_chunk in _iter_sample_table_chunks(shard_path, chunksize=read_chunksize):
            shard = normalise_sample_table(shard_chunk)
            total_rows += int(len(shard))
            observed_modes.update(str(item) for item in shard["quant_mode"].dropna().unique().tolist())
            for warning in shard.attrs.get("label_mapping_warnings", []):
                if warning:
                    label_mapping_warnings.add(str(warning))
            if class_field is None and shard.attrs.get("class_field"):
                class_field = str(shard.attrs["class_field"])
            for _, row in shard.iterrows():
                facet = {column: _stream_key_value(row[column]) for column in SUMMARY_FACET_COLUMNS}
                row_groups = _analysis_groups_for_bucket(row["bucket"])
                if not row_groups:
                    continue
                row_class = _row_class_value(row, class_field)
                for metric_name, item in TOKEN_METRIC_SPECS.items():
                    if item.source_column not in shard.columns:
                        continue
                    context = _metric_context_metadata(metric_name, facet)
                    if not include_predicate(context):
                        continue
                    for analysis_group in row_groups:
                        key = tuple(facet[column] for column in SUMMARY_FACET_COLUMNS) + (metric_name, analysis_group)
                        state = states.get(key)
                        if state is None:
                            state = _StreamingMetricState(
                                facet=dict(facet),
                                metric_name=metric_name,
                                context=dict(context),
                                analysis_group=analysis_group,
                            )
                            states[key] = state
                        state.sample_count += 1
                        if row_class is not None:
                            state.class_values.add(row_class)
                        if bool(context["metric_valid"]):
                            metric_value = _stream_float(row[item.source_column])
                            if math.isfinite(metric_value):
                                state.value_sum += metric_value
                                state.value_count += 1

    warning_text = ";".join(sorted(label_mapping_warnings))
    rows: list[dict[str, Any]] = []
    for key in sorted(states, key=lambda item: tuple(str(part) for part in item)):
        state = states[key]
        base = _base_metric_row(state.facet, state.metric_name, state.context)
        value = float("nan")
        if bool(state.context["metric_valid"]) and state.value_count > 0:
            value = float(state.value_sum / state.value_count)
        rows.append({
            "target": state.facet["target"],
            "bucket": "",
            "shot_group": "",
            "analysis_group": state.analysis_group,
            "value": value,
            "group_sample_count": int(state.sample_count),
            "group_class_count": float(len(state.class_values)) if class_field is not None else float("nan"),
            "class_count_source": class_field or "",
            "class_count_warning": warning_text,
            **{name: item for name, item in base.items() if name != "target"},
        })

    summary = pd.DataFrame(rows)
    observed = sorted(observed_modes)
    missing = sorted(set(KNOWN_QUANT_MODES) - set(observed))
    summary.attrs["observed_quant_modes"] = observed
    summary.attrs["missing_quant_modes"] = missing
    summary.attrs["label_mapping_warnings"] = sorted(label_mapping_warnings)
    summary.attrs["num_sample_rows"] = int(total_rows)
    return summary


def build_summary_v2_long_from_shards(
    shard_csv_paths: Sequence[str | Path],
    *,
    chunksize: int | None = DEFAULT_SUMMARY_READ_CHUNKSIZE,
) -> pd.DataFrame:
    return _build_summary_v2_long_from_shards(
        shard_csv_paths,
        chunksize=chunksize,
        include_predicate=_include_in_formal_trace_summary,
    )


def build_detail_summary_v2_long_from_shards(
    shard_csv_paths: Sequence[str | Path],
    *,
    chunksize: int | None = DEFAULT_SUMMARY_READ_CHUNKSIZE,
) -> pd.DataFrame:
    return _build_summary_v2_long_from_shards(
        shard_csv_paths,
        chunksize=chunksize,
        include_predicate=_include_in_detail_trace_summary,
    )


def build_summary_table_from_shards(
    shard_csv_paths: Sequence[str | Path],
    *,
    cfg: Any | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = build_summary_v2_long_from_shards(shard_csv_paths, chunksize=_summary_read_chunksize(cfg))
    detail_summary = build_detail_summary_v2_long_from_shards(shard_csv_paths, chunksize=_summary_read_chunksize(cfg))
    comparison_tables = _comparison_output_tables(summary, cfg=cfg, detail_summary=detail_summary)
    comparison = comparison_tables["comparison"]
    quant_mode_comparison = comparison_tables["quant_mode_comparison"]
    intervention_summary = build_intervention_level_summary_from_shards(
        shard_csv_paths,
        chunksize=_summary_read_chunksize(cfg),
    )
    intervention_comparison = build_intervention_level_comparison_summary(intervention_summary)
    extra_tables = {
        "detail_summary": detail_summary,
        "detail_comparison": comparison_tables["detail_comparison"],
        "diagnostic_comparison": comparison_tables["diagnostic_comparison"],
        "detail_quant_mode_comparison": comparison_tables["detail_quant_mode_comparison"],
        "diagnostic_quant_mode_comparison": comparison_tables["diagnostic_quant_mode_comparison"],
        "intervention_level_summary": intervention_summary,
        "intervention_level_comparison": intervention_comparison,
    }
    observed = summary.attrs.get("observed_quant_modes", [])
    available = sorted(str(item) for item in quant_mode_comparison.get("comparison_name", pd.Series(dtype=str)).dropna().unique().tolist())
    missing = sorted(set(KNOWN_QUANT_MODES) - set(observed))
    sample_table = pd.DataFrame()
    all_frames = [sample_table, summary, comparison, quant_mode_comparison, *extra_tables.values()]
    _set_common_attrs(
        all_frames,
        observed=observed,
        available=available,
        missing=missing,
        label_mapping_warnings=summary.attrs.get("label_mapping_warnings", []),
        num_sample_rows=int(summary.attrs.get("num_sample_rows", 0)),
    )
    for frame in (sample_table, summary, comparison, quant_mode_comparison):
        frame.attrs["extra_tables"] = dict(extra_tables)
    if save:
        target_dir = _resolve_output_dir(cfg=cfg, output_dir=output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(target_dir / SUMMARY_OUTPUT_FILENAME, index=False)
        comparison.to_csv(target_dir / COMPARISON_SUMMARY_OUTPUT_FILENAME, index=False)
        quant_mode_comparison.to_csv(target_dir / QUANT_MODE_COMPARISON_OUTPUT_FILENAME, index=False)
        _save_extra_summary_outputs(
            target_dir=target_dir,
            tables=extra_tables,
            export_diagnostic=_export_diagnostic_comparison(cfg),
        )
    return sample_table, summary, comparison, quant_mode_comparison


def build_summary_table(
    sample_csv: str | Path | pd.DataFrame,
    *,
    cfg: Any | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_table = normalise_sample_table(_read_sample_table(sample_csv))
    summary = build_summary_v2_long(sample_table)
    detail_summary = build_detail_summary_v2_long(sample_table)
    comparison_tables = _comparison_output_tables(summary, cfg=cfg, detail_summary=detail_summary)
    comparison = comparison_tables["comparison"]
    quant_mode_comparison = comparison_tables["quant_mode_comparison"]
    intervention_summary = build_intervention_level_summary(sample_table)
    intervention_comparison = build_intervention_level_comparison_summary(intervention_summary)
    extra_tables = {
        "detail_summary": detail_summary,
        "detail_comparison": comparison_tables["detail_comparison"],
        "diagnostic_comparison": comparison_tables["diagnostic_comparison"],
        "detail_quant_mode_comparison": comparison_tables["detail_quant_mode_comparison"],
        "diagnostic_quant_mode_comparison": comparison_tables["diagnostic_quant_mode_comparison"],
        "intervention_level_summary": intervention_summary,
        "intervention_level_comparison": intervention_comparison,
    }
    observed = sorted(str(item) for item in sample_table["quant_mode"].dropna().unique().tolist())
    available = sorted(str(item) for item in quant_mode_comparison.get("comparison_name", pd.Series(dtype=str)).dropna().unique().tolist())
    missing = sorted(set(KNOWN_QUANT_MODES) - set(observed))
    _set_common_attrs(
        [sample_table, summary, comparison, quant_mode_comparison, *extra_tables.values()],
        observed=observed,
        available=available,
        missing=missing,
        label_mapping_warnings=sample_table.attrs.get("label_mapping_warnings", []),
        num_sample_rows=int(len(sample_table)),
    )
    for frame in (sample_table, summary, comparison, quant_mode_comparison):
        frame.attrs["extra_tables"] = dict(extra_tables)
    if save:
        target_dir = _resolve_output_dir(cfg=cfg, output_dir=output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(target_dir / SUMMARY_OUTPUT_FILENAME, index=False)
        comparison.to_csv(target_dir / COMPARISON_SUMMARY_OUTPUT_FILENAME, index=False)
        quant_mode_comparison.to_csv(target_dir / QUANT_MODE_COMPARISON_OUTPUT_FILENAME, index=False)
        _save_extra_summary_outputs(
            target_dir=target_dir,
            tables=extra_tables,
            export_diagnostic=_export_diagnostic_comparison(cfg),
        )
    return sample_table, summary, comparison, quant_mode_comparison


__all__ = [
    "ANALYSIS_GROUP_ORDER",
    "COMPARISON_SUMMARY_OUTPUT_FILENAME",
    "DETAIL_SUMMARY_OUTPUT_FILENAME",
    "DETAIL_COMPARISON_SUMMARY_OUTPUT_FILENAME",
    "DETAIL_QUANT_MODE_COMPARISON_OUTPUT_FILENAME",
    "DIAGNOSTIC_COMPARISON_SUMMARY_OUTPUT_FILENAME",
    "DIAGNOSTIC_QUANT_MODE_COMPARISON_OUTPUT_FILENAME",
    "INTERVENTION_LEVEL_COMPARISON_OUTPUT_FILENAME",
    "INTERVENTION_LEVEL_SUMMARY_OUTPUT_FILENAME",
    "INTERVENTION_LEVEL_FACET_COLUMNS",
    "KNOWN_QUANT_MODES",
    "METRIC_COLUMNS",
    "QUANT_MODE_COMPARISON_OUTPUT_FILENAME",
    "SUMMARY_FACET_COLUMNS",
    "SUMMARY_OUTPUT_FILENAME",
    "TOKEN_METRIC_REGISTRY",
    "TOKEN_METRIC_SPECS",
    "build_comparison_summary",
    "build_detail_summary_v2_long",
    "build_detail_summary_v2_long_from_shards",
    "build_intervention_level_comparison_summary",
    "build_intervention_level_summary",
    "build_intervention_level_summary_from_shards",
    "build_quant_mode_comparison",
    "build_summary_table",
    "build_summary_table_from_shards",
    "build_summary_v2_long",
    "build_summary_v2_long_from_shards",
    "normalise_sample_table",
]
