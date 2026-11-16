from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

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
        group_protocol_metadata,
        metric_spec_metadata,
        normalize_analysis_group,
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
        group_protocol_metadata,
        metric_spec_metadata,
        normalize_analysis_group,
        read_canonical_bucket,
        tail_included,
    )


SAMPLE_OUTPUT_FILENAME = "quant_danger_zone_metrics.csv"
SUMMARY_OUTPUT_FILENAME = "quant_danger_zone_summary.csv"
COMPARISON_SUMMARY_OUTPUT_FILENAME = "quant_danger_zone_comparison_summary.csv"
CORRELATION_OUTPUT_FILENAME = "quant_danger_zone_correlation_v2.csv"
DEFAULT_QUANT_MODE = "A4-only"

SAMPLE_COLUMNS = [
    "target", "sample_id", "label", "bucket", "shot_group", "tail_included",
    "layer_idx", "module_name", "tensor_position", "granularity", "quant_mode", "scale_type",
    "activation_shape", "original_layout", "original_shape", "canonical_layout", "canonical_shape",
    "layout_inference_source", "padding_mask_available", "padding_mask_aligned", "mask_status",
    "valid_token_policy", "metric_status", "key_token_source", "aux_token_source",
    "integer_grid_source", "is_direct_quantlinear_forward_grid",
    "num_tokens", "hidden_dim", "valid_token_count", "total_token_count", "padding_token_ratio",
    "valid_token_mask_applied", "qmin", "qmax", "scale_mean", "scale_median", "scale_p95",
    "upper_clip_rate", "lower_clip_rate", "total_clip_rate", "boundary_hit_rate", "strict_overflow_rate",
    "saturation_boundary_energy_share", "overflow_energy_proxy", "exact_zero_rate", "near_zero_pm1_rate", "near_zero_rate",
    "topk_token_exact_zero_rate", "topk_token_near_zero_pm1_rate", "topk_token_boundary_hit_rate", "topk_token_strict_overflow_rate",
    "high_energy_scalar_exact_zero_rate", "high_energy_scalar_near_zero_pm1_rate", "high_energy_scalar_boundary_hit_rate", "high_energy_scalar_strict_overflow_rate",
    "topk_token_clip_rate", "high_energy_scalar_clip_rate", "topk_token_near_zero_rate",
    "near_zero_swallow_rate",
    "high_energy_near_zero_rate", "oer_scalar_clip_rate", "key_token_near_zero_swallow_rate",
    "saturation_clip_energy_share", "near_zero_swallow_energy_share", "near_zero_swallow_loss_share",
    "key_token_clip_energy_share", "key_token_near_zero_energy_share", "key_token_near_zero_loss_share",
    "oer_retention", "oer_loss_total", "oer_loss", "max_abs_over_p99", "p999_over_p99",
    "projection_rel_l2_error", "projection_cosine_drift", "projection_mse", "projection_oer_shift",
    "projection_oer_retention", "projection_oer_loss", "projection_off_oracle_energy_ratio", "output_energy_distortion",
    "near_zero_rate_deprecated_alias_of", "total_clip_rate_deprecated_alias_of",
    "topk_token_clip_rate_deprecated_alias_of", "topk_token_near_zero_rate_deprecated_alias_of",
    "high_energy_scalar_clip_rate_deprecated_alias_of",
]


FACET_COLUMNS = ["target", "tensor_position", "granularity", "quant_mode"]
GROUP_COLUMNS = [*FACET_COLUMNS, "analysis_group"]
SAMPLE_KEY_COLUMNS = ["target", "sample_id", "bucket", "shot_group", "tensor_position", "granularity", "quant_mode"]

AGGREGATE_VALUE_COLUMNS = [
    "num_tokens", "hidden_dim", "valid_token_count", "total_token_count", "padding_token_ratio",
    "qmin", "qmax", "scale_mean", "scale_median", "scale_p95",
    "boundary_hit_rate", "strict_overflow_rate",
    "saturation_boundary_energy_share", "overflow_energy_proxy", "exact_zero_rate", "near_zero_pm1_rate",
    "topk_token_exact_zero_rate", "topk_token_near_zero_pm1_rate", "topk_token_boundary_hit_rate", "topk_token_strict_overflow_rate",
    "high_energy_scalar_exact_zero_rate", "high_energy_scalar_near_zero_pm1_rate", "high_energy_scalar_boundary_hit_rate", "high_energy_scalar_strict_overflow_rate",
    "near_zero_swallow_rate",
    "high_energy_near_zero_rate", "oer_scalar_clip_rate", "key_token_near_zero_swallow_rate",
    "saturation_clip_energy_share", "near_zero_swallow_energy_share", "near_zero_swallow_loss_share",
    "key_token_clip_energy_share", "key_token_near_zero_energy_share", "key_token_near_zero_loss_share",
    "oer_retention", "oer_loss_total", "max_abs_over_p99", "p999_over_p99",
    "projection_rel_l2_error", "projection_cosine_drift", "projection_mse", "projection_oer_shift",
    "projection_oer_retention", "projection_oer_loss", "projection_off_oracle_energy_ratio", "output_energy_distortion",
]
NUMERIC_SAMPLE_COLUMNS = ["layer_idx", *AGGREGATE_VALUE_COLUMNS, "oer_loss"]

DANGER_COMPONENT_METRICS = {
    "boundary_hit_rate": "boundary_hit_rate",
    "strict_overflow_rate": "strict_overflow_rate",
    "overflow_energy_proxy": "overflow_energy_proxy",
    "exact_zero_rate": "exact_zero_rate",
    "near_zero_pm1_rate": "near_zero_pm1_rate",
    "high_energy_near_zero_rate": "high_energy_near_zero_rate",
    "topk_token_boundary_hit_rate": "topk_token_boundary_hit_rate",
    "topk_token_strict_overflow_rate": "topk_token_strict_overflow_rate",
    "high_energy_scalar_boundary_hit_rate": "high_energy_scalar_boundary_hit_rate",
    "high_energy_scalar_strict_overflow_rate": "high_energy_scalar_strict_overflow_rate",
    "near_zero_swallow_loss_share": "near_zero_swallow_loss_share",
    "key_token_near_zero_loss_share": "key_token_near_zero_loss_share",
    "oer_loss_total": "oer_loss_total",
}
CORRELATION_COMPONENT_METRICS = [
    "boundary_hit_rate", "strict_overflow_rate", "overflow_energy_proxy",
    "exact_zero_rate", "near_zero_pm1_rate", "high_energy_near_zero_rate",
    "topk_token_boundary_hit_rate", "topk_token_strict_overflow_rate",
    "high_energy_scalar_boundary_hit_rate", "high_energy_scalar_strict_overflow_rate",
    "near_zero_swallow_loss_share", "key_token_near_zero_loss_share",
    "key_token_clip_energy_share",
    "projection_rel_l2_error", "projection_cosine_drift", "projection_mse", "projection_oer_shift",
    "projection_oer_loss", "output_energy_distortion",
]

LOWER_IS_BETTER_METRICS = {
    "boundary_hit_rate", "strict_overflow_rate",
    "saturation_boundary_energy_share", "overflow_energy_proxy", "exact_zero_rate", "near_zero_pm1_rate",
    "topk_token_exact_zero_rate", "topk_token_near_zero_pm1_rate", "topk_token_boundary_hit_rate", "topk_token_strict_overflow_rate",
    "high_energy_scalar_exact_zero_rate", "high_energy_scalar_near_zero_pm1_rate", "high_energy_scalar_boundary_hit_rate", "high_energy_scalar_strict_overflow_rate",
    "near_zero_swallow_rate",
    "high_energy_near_zero_rate", "key_token_near_zero_swallow_rate",
    "saturation_clip_energy_share", "near_zero_swallow_energy_share", "near_zero_swallow_loss_share",
    "key_token_clip_energy_share", "key_token_near_zero_energy_share", "key_token_near_zero_loss_share",
    "oer_loss_total", "max_abs_over_p99", "p999_over_p99",
    "projection_rel_l2_error", "projection_cosine_drift", "projection_mse", "projection_oer_shift",
    "projection_oer_loss", "output_energy_distortion",
}
HIGHER_IS_BETTER_METRICS = {"oer_retention"}
DIAGNOSTIC_METRICS = {
    "num_tokens", "hidden_dim", "valid_token_count", "total_token_count", "padding_token_ratio",
    "qmin", "qmax", "scale_mean", "scale_median", "scale_p95", "oer_scalar_clip_rate",
    "projection_oer_retention", "projection_off_oracle_energy_ratio",
}
GAP_ENABLED_METRICS = set(LOWER_IS_BETTER_METRICS) | HIGHER_IS_BETTER_METRICS
TOKEN_SAMPLE_SUMMARY_METRICS = {"topk_token_exact_zero_rate", "topk_token_near_zero_pm1_rate", "topk_token_boundary_hit_rate", "topk_token_strict_overflow_rate", "high_energy_scalar_exact_zero_rate", "high_energy_scalar_near_zero_pm1_rate", "high_energy_scalar_boundary_hit_rate", "high_energy_scalar_strict_overflow_rate"}


def _metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER_METRICS:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER_METRICS:
        return "lower_is_better"
    return "diagnostic"


def _metric_role(metric: str) -> str:
    return "detail" if metric in DIAGNOSTIC_METRICS else "main"


def _metric_spec(metric: str) -> MetricSpec:
    aggregation_unit = "token_sample_summary" if metric in TOKEN_SAMPLE_SUMMARY_METRICS else "sample"
    aggregation_fn = "mean" if metric in TOKEN_SAMPLE_SUMMARY_METRICS else "nanmean_valid_only"
    return define_metric_spec(
        metric,
        metric_direction=_metric_direction(metric),
        aggregation_unit=aggregation_unit,
        aggregation_fn=aggregation_fn,
        metric_role=_metric_role(metric),
    )


DANGER_ZONE_METRIC_SPECS = build_metric_spec_registry(tuple(_metric_spec(name) for name in AGGREGATE_VALUE_COLUMNS))
CORRELATION_METRIC_SPEC = define_metric_spec(
    "component_oer_loss_correlation",
    metric_direction="diagnostic",
    aggregation_unit="group_recompute",
    aggregation_fn="custom",
    metric_role="detail",
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _dataset_name(cfg: Any) -> str:
    dataset_cfg = _cfg_get(cfg, "CURRENT_DATASET", {})
    if isinstance(dataset_cfg, Mapping):
        for key in ("NAME", "name", "DATASET_NAME", "dataset_name"):
            if dataset_cfg.get(key):
                return str(dataset_cfg[key])
        if dataset_cfg.get("ROOT") is not None:
            return Path(dataset_cfg["ROOT"]).name
    active = _cfg_get(cfg, "ACTIVE_DATASET", None)
    if active:
        return str(active)
    result_dir = _cfg_get(cfg, "CURRENT_RESULT_DIR", None)
    if result_dir is not None:
        return Path(result_dir).name
    raise AttributeError("Cannot determine dataset name")


def _resolve_output_dir(cfg: Any | None = None, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if cfg is None:
        import config as project_config
        cfg = project_config
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "quant_danger_zone"


def _read_main_csv(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    return data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _series(df: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _first_present(df: pd.DataFrame, names: Sequence[str], default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _parse_bool(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{context}: value must be bool-like, got {value!r}")


def _target_from_parts(df: pd.DataFrame) -> pd.Series:
    if "target" in df.columns and not df["target"].isna().all():
        return df["target"].astype(str)
    layer = _to_numeric(df["layer_idx"])
    module = df["module_name"].astype(str)
    return "L" + layer.astype(int).astype(str) + "_" + module


def _normalise_sample_level_table(main_df: pd.DataFrame) -> pd.DataFrame:
    df = main_df.copy()
    required = ["sample_id", "layer_idx", "module_name", "bucket", "tensor_position", "granularity"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(f"danger-zone sample CSV is missing required columns: {', '.join(missing)}")

    bucket_result = read_canonical_bucket(
        bucket=df["bucket"],
        shot_group=df["shot_group"] if "shot_group" in df.columns else None,
        train_label_n=df["train_label_n"] if "train_label_n" in df.columns else None,
        context="quant_danger_zone sample table",
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in bucket_result.bucket]
    computed_tail = [bool(item) for item in tail_included(buckets)]
    if "tail_included" in df.columns:
        observed = [_parse_bool(value, context=f"quant_danger_zone sample row {idx}") for idx, value in enumerate(df["tail_included"].tolist())]
        if observed != computed_tail:
            raise ValueError("quant_danger_zone sample table: tail_included must match bucket-derived tail membership")

    sample = pd.DataFrame(index=df.index)
    sample["target"] = _target_from_parts(df)
    sample["sample_id"] = df["sample_id"].astype(str)
    sample["label"] = _first_present(df, ("label", "label_id", "label_name"))
    sample["bucket"] = buckets
    sample["shot_group"] = buckets
    sample["tail_included"] = computed_tail
    sample["layer_idx"] = _to_numeric(df["layer_idx"])
    sample["module_name"] = df["module_name"].astype(str)
    sample["tensor_position"] = df["tensor_position"].astype(str)
    sample["granularity"] = df["granularity"].astype(str)
    sample["quant_mode"] = _series(df, "quant_mode", DEFAULT_QUANT_MODE).astype(str)
    sample["scale_type"] = _series(df, "scale_type", np.nan)
    sample["activation_shape"] = _series(df, "activation_shape", "")
    sample["original_layout"] = _series(df, "original_layout", "")
    sample["original_shape"] = _series(df, "original_shape", "")
    sample["canonical_layout"] = _series(df, "canonical_layout", "BTC")
    sample["canonical_shape"] = _series(df, "canonical_shape", sample["activation_shape"])
    sample["layout_inference_source"] = _series(df, "layout_inference_source", "")
    sample["padding_mask_available"] = _series(df, "padding_mask_available", np.nan)
    sample["padding_mask_aligned"] = _series(df, "padding_mask_aligned", np.nan)
    sample["mask_status"] = _series(df, "mask_status", "")
    sample["valid_token_policy"] = _series(df, "valid_token_policy", "require_mask")
    sample["metric_status"] = _series(df, "metric_status", "ok")
    sample["key_token_source"] = _series(df, "key_token_source", "")
    sample["aux_token_source"] = _series(df, "aux_token_source", "")
    sample["integer_grid_source"] = _series(df, "integer_grid_source", "")
    sample["is_direct_quantlinear_forward_grid"] = _series(df, "is_direct_quantlinear_forward_grid", np.nan)
    sample["valid_token_mask_applied"] = _series(df, "valid_token_mask_applied", np.nan)
    sample["near_zero_rate_deprecated_alias_of"] = _series(df, "near_zero_rate_deprecated_alias_of", "near_zero_pm1_rate")
    sample["total_clip_rate_deprecated_alias_of"] = _series(df, "total_clip_rate_deprecated_alias_of", "boundary_hit_rate")
    sample["topk_token_clip_rate_deprecated_alias_of"] = _series(df, "topk_token_clip_rate_deprecated_alias_of", "topk_token_boundary_hit_rate")
    sample["topk_token_near_zero_rate_deprecated_alias_of"] = _series(df, "topk_token_near_zero_rate_deprecated_alias_of", "topk_token_near_zero_pm1_rate")
    sample["high_energy_scalar_clip_rate_deprecated_alias_of"] = _series(df, "high_energy_scalar_clip_rate_deprecated_alias_of", "high_energy_scalar_boundary_hit_rate")

    aliases = {
        "num_tokens": ("num_tokens",), "hidden_dim": ("hidden_dim",),
        "valid_token_count": ("valid_token_count",), "total_token_count": ("total_token_count",),
        "padding_token_ratio": ("padding_token_ratio",),
        "qmin": ("qmin",), "qmax": ("qmax",),
        "scale_mean": ("scale_mean",), "scale_median": ("scale_median",), "scale_p95": ("scale_p95",),
        "upper_clip_rate": ("upper_clip_rate",), "lower_clip_rate": ("lower_clip_rate",),
        "total_clip_rate": ("total_clip_rate", "clip_rate", "clipping_rate"),
        "boundary_hit_rate": ("boundary_hit_rate", "saturation_boundary_hit_rate", "total_clip_rate", "clip_rate"),
        "strict_overflow_rate": ("strict_overflow_rate", "overflow_rate", "saturation_strict_overflow_rate"),
        "saturation_boundary_hit_rate": ("saturation_boundary_hit_rate", "boundary_hit_rate", "saturation_clip_rate", "total_clip_rate", "clip_rate"),
        "saturation_strict_overflow_rate": ("saturation_strict_overflow_rate", "saturation_overflow_rate", "strict_overflow_rate"),
        "saturation_boundary_energy_share": ("saturation_boundary_energy_share", "saturation_clip_energy_share"),
        "overflow_energy_proxy": ("overflow_energy_proxy",),
        "exact_zero_rate": ("exact_zero_rate",),
        "near_zero_pm1_rate": ("near_zero_pm1_rate", "near_zero_rate"),
        "near_zero_rate": ("near_zero_rate", "near_zero_pm1_rate"),
        "topk_token_exact_zero_rate": ("topk_token_exact_zero_rate",),
        "topk_token_near_zero_pm1_rate": ("topk_token_near_zero_pm1_rate", "topk_token_near_zero_rate"),
        "topk_token_boundary_hit_rate": ("topk_token_boundary_hit_rate", "topk_token_clip_rate"),
        "topk_token_strict_overflow_rate": ("topk_token_strict_overflow_rate",),
        "high_energy_scalar_exact_zero_rate": ("high_energy_scalar_exact_zero_rate",),
        "high_energy_scalar_near_zero_pm1_rate": ("high_energy_scalar_near_zero_pm1_rate",),
        "high_energy_scalar_boundary_hit_rate": ("high_energy_scalar_boundary_hit_rate", "high_energy_scalar_clip_rate"),
        "high_energy_scalar_strict_overflow_rate": ("high_energy_scalar_strict_overflow_rate",),
        "topk_token_clip_rate": ("topk_token_clip_rate", "topk_token_boundary_hit_rate"),
        "high_energy_scalar_clip_rate": ("high_energy_scalar_clip_rate", "high_energy_scalar_boundary_hit_rate"),
        "topk_token_near_zero_rate": ("topk_token_near_zero_rate", "topk_token_near_zero_pm1_rate"),
        "saturation_clip_rate": ("saturation_clip_rate", "saturation_boundary_hit_rate", "total_clip_rate"),
        "saturation_overflow_rate": ("saturation_overflow_rate", "saturation_strict_overflow_rate", "strict_overflow_rate"),
        "near_zero_swallow_rate": ("near_zero_swallow_rate",),
        "high_energy_near_zero_rate": ("high_energy_near_zero_rate", "key_token_near_zero_swallow_rate"),
        "oer_scalar_clip_rate": ("oer_scalar_clip_rate",),
        "key_token_near_zero_swallow_rate": ("key_token_near_zero_swallow_rate", "high_energy_near_zero_rate"),
        "saturation_clip_energy_share": ("saturation_clip_energy_share", "saturation_boundary_energy_share"),
        "near_zero_swallow_energy_share": ("near_zero_swallow_energy_share",),
        "near_zero_swallow_loss_share": ("near_zero_swallow_loss_share",),
        "key_token_clip_energy_share": ("key_token_clip_energy_share",),
        "key_token_near_zero_energy_share": ("key_token_near_zero_energy_share",),
        "key_token_near_zero_loss_share": ("key_token_near_zero_loss_share",),
        "oer_retention": ("oer_retention",),
        "oer_loss_total": ("oer_loss_total", "oer_loss", "delta_oer", "oer_retention_loss"),
        "oer_loss": ("oer_loss", "oer_loss_total", "delta_oer", "oer_retention_loss"),
        "max_abs_over_p99": ("max_abs_over_p99",), "p999_over_p99": ("p999_over_p99",),
        "projection_rel_l2_error": ("projection_rel_l2_error",),
        "projection_cosine_drift": ("projection_cosine_drift",),
        "projection_mse": ("projection_mse",),
        "projection_oer_shift": ("projection_oer_shift", "projection_oer_loss"),
        "projection_oer_retention": ("projection_oer_retention",),
        "projection_oer_loss": ("projection_oer_loss", "projection_oer_shift"),
        "projection_off_oracle_energy_ratio": ("projection_off_oracle_energy_ratio",),
        "output_energy_distortion": ("output_energy_distortion",),
    }
    for out, names in aliases.items():
        sample[out] = _first_present(df, names)
    if sample["oer_loss_total"].isna().all():
        sample["oer_loss_total"] = 1.0 - _to_numeric(sample["oer_retention"])
    if sample["oer_loss"].isna().all():
        sample["oer_loss"] = sample["oer_loss_total"]
    if sample["near_zero_swallow_rate"].isna().all():
        sample["near_zero_swallow_rate"] = sample["high_energy_near_zero_rate"]
    sample["scale_type"] = sample["scale_type"].where(sample["scale_type"].notna(), sample["granularity"])
    for name in NUMERIC_SAMPLE_COLUMNS:
        sample[name] = _to_numeric(sample[name])
    _validate_standard_sample_table(sample)
    return sample[SAMPLE_COLUMNS]


def _validate_standard_sample_table(sample_df: pd.DataFrame) -> None:
    missing = [name for name in SAMPLE_COLUMNS if name not in sample_df.columns]
    if missing:
        raise KeyError(f"danger-zone standard sample table is missing columns: {', '.join(missing)}")
    empty_keys = []
    invalid_tokens = {"", "nan", "<na>", "none"}
    for name in SAMPLE_KEY_COLUMNS:
        values = sample_df[name].astype(str).str.strip().str.lower()
        if sample_df[name].isna().any() or values.isin(invalid_tokens).any():
            empty_keys.append(name)
    if empty_keys:
        raise ValueError(f"danger-zone sample table key columns contain missing values: {', '.join(empty_keys)}")
    read_canonical_bucket(bucket=sample_df["bucket"], shot_group=sample_df["shot_group"], context="quant_danger_zone standard sample table")
    expected_tail = [bool(item) for item in tail_included(sample_df["bucket"])]
    observed_tail = [bool(item) for item in sample_df["tail_included"].tolist()]
    if observed_tail != expected_tail:
        raise ValueError("danger-zone sample table tail_included must match bucket-derived tail membership")

    valid_policies = {"require_mask", "explicit_no_padding"}
    policies = sample_df["valid_token_policy"].fillna("").astype(str).str.strip()
    invalid_policy = sorted(set(policies) - valid_policies)
    if invalid_policy:
        raise ValueError(f"danger-zone valid_token_policy contains invalid values: {invalid_policy}")
    padding_available = sample_df["padding_mask_available"].map(
        lambda item: _parse_bool(item, context="danger-zone padding_mask_available")
    )
    mask_applied = sample_df["valid_token_mask_applied"].map(
        lambda item: _parse_bool(item, context="danger-zone valid_token_mask_applied")
    )
    require_mask_rows = policies == "require_mask"
    invalid_mask_rows = require_mask_rows & (~padding_available | ~mask_applied)
    if bool(invalid_mask_rows.any()):
        bad = sample_df.loc[invalid_mask_rows, ["target", "sample_id", "tensor_position", "granularity"]].head(8).to_dict("records")
        raise ValueError(
            "danger-zone require_mask rows must have padding_mask_available=True and "
            f"valid_token_mask_applied=True; examples={bad}"
        )


def _selected(frame: pd.DataFrame, analysis_group: str) -> pd.DataFrame:
    group = normalize_analysis_group(analysis_group)
    return frame.loc[analysis_group_mask(frame["bucket"], group)]


def _valid_counts(values: pd.Series) -> dict[str, Any]:
    numeric = _to_numeric(values).to_numpy(dtype=float)
    total = int(len(numeric))
    valid = int(np.isfinite(numeric).sum())
    return {"valid_count": valid, "total_count": total, "valid_rate": float(valid / total) if total else float("nan"), "invalid_count": int(total - valid)}


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    try:
        num, den = float(numerator), float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(num) and math.isfinite(den)) or abs(den) <= 0.0:
        return float("nan")
    return float(num / den)


def _group_value(frame: pd.DataFrame, metric: str, analysis_group: str) -> float:
    return aggregate_metric_for_analysis_group_from_raw_records(
        frame[metric], frame["bucket"], analysis_group,
        spec=DANGER_ZONE_METRIC_SPECS[metric], record_unit=DANGER_ZONE_METRIC_SPECS[metric].aggregation_unit,
    )


def _base_row(facet: Mapping[str, Any], metric: str) -> dict[str, Any]:
    spec = DANGER_ZONE_METRIC_SPECS[metric]
    return {
        "group_protocol_version": GROUP_PROTOCOL_VERSION,
        "group_protocol_name": GROUP_PROTOCOL_NAME,
        **dict(facet),
        **metric_spec_metadata(spec),
        "comparable_across_groups": bool(metric in GAP_ENABLED_METRICS),
        "gap_enabled": bool(metric in GAP_ENABLED_METRICS),
    }


def build_group_summary_v2(sample_df: pd.DataFrame) -> pd.DataFrame:
    _validate_standard_sample_table(sample_df)
    rows: list[dict[str, Any]] = []
    for facet_key, frame in sample_df.groupby(FACET_COLUMNS, dropna=False, sort=True):
        facet = dict(zip(FACET_COLUMNS, facet_key))
        for metric in AGGREGATE_VALUE_COLUMNS:
            base = _base_row(facet, metric)
            for group in ANALYSIS_GROUPS:
                part = _selected(frame, group)
                rows.append({
                    "row_type": "group_summary_v2", **base,
                    "analysis_group": group, "shot_group": "",
                    "metric_value": _group_value(frame, metric, group),
                    "sample_count": int(part["sample_id"].astype(str).nunique()) if len(part) else 0,
                    "num_rows": int(len(part)), **_valid_counts(part[metric] if len(part) else pd.Series([], dtype=float)),
                })
    return pd.DataFrame(rows)


def build_comparison_summary_v2(sample_df: pd.DataFrame) -> pd.DataFrame:
    _validate_standard_sample_table(sample_df)
    rows: list[dict[str, Any]] = []
    for facet_key, frame in sample_df.groupby(FACET_COLUMNS, dropna=False, sort=True):
        facet = dict(zip(FACET_COLUMNS, facet_key))
        for metric in AGGREGATE_VALUE_COLUMNS:
            spec = DANGER_ZONE_METRIC_SPECS[metric]
            values = {group: _group_value(frame, metric, group) for group in ANALYSIS_GROUPS}
            if metric in GAP_ENABLED_METRICS:
                gaps = compute_tail_and_few_gaps(values, metric_direction=spec.metric_direction)
                tail_ratio = _safe_ratio(values.get("tail"), values.get("many"))
                few_ratio = _safe_ratio(values.get("few"), values.get("many"))
            else:
                gaps = {"tail_vs_many_gap": float("nan"), "few_vs_many_gap": float("nan")}
                tail_ratio = few_ratio = float("nan")
            rows.append({
                "row_type": "comparison_summary_v2", **_base_row(facet, metric),
                "many_value": values.get("many", float("nan")),
                "medium_value": values.get("medium", float("nan")),
                "few_value": values.get("few", float("nan")),
                "tail_value": values.get("tail", float("nan")),
                "tail_vs_many_gap": gaps["tail_vs_many_gap"],
                "few_vs_many_gap": gaps["few_vs_many_gap"],
                "tail_to_many_ratio": tail_ratio,
                "few_to_many_ratio": few_ratio,
                "many_sample_count": int(_selected(frame, "many")["sample_id"].astype(str).nunique()),
                "medium_sample_count": int(_selected(frame, "medium")["sample_id"].astype(str).nunique()),
                "few_sample_count": int(_selected(frame, "few")["sample_id"].astype(str).nunique()),
                "tail_sample_count": int(_selected(frame, "tail")["sample_id"].astype(str).nunique()),
            })
    return pd.DataFrame(rows)


def _corr_pair(frame: pd.DataFrame, left: str, right: str) -> dict[str, Any]:
    values = frame[[left, right]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    valid_count = int(len(values))
    if valid_count < 2 or values[left].nunique(dropna=True) < 2 or values[right].nunique(dropna=True) < 2:
        return {"pearson": float("nan"), "spearman": float("nan"), "valid_count": valid_count}
    return {"pearson": float(values[left].corr(values[right], method="pearson")), "spearman": float(values[left].corr(values[right], method="spearman")), "valid_count": valid_count}


def build_correlation_v2(sample_df: pd.DataFrame) -> pd.DataFrame:
    _validate_standard_sample_table(sample_df)
    rows: list[dict[str, Any]] = []
    corr_meta = metric_spec_metadata(CORRELATION_METRIC_SPEC)
    for facet_key, frame in sample_df.groupby(FACET_COLUMNS, dropna=False, sort=True):
        facet = dict(zip(FACET_COLUMNS, facet_key))
        for group in ANALYSIS_GROUPS:
            part = _selected(frame, group)
            for component in CORRELATION_COMPONENT_METRICS:
                corr = _corr_pair(part, component, "oer_loss_total") if len(part) else {"pearson": float("nan"), "spearman": float("nan"), "valid_count": 0}
                rows.append({
                    "row_type": "correlation_v2", "group_protocol_version": GROUP_PROTOCOL_VERSION,
                    "group_protocol_name": GROUP_PROTOCOL_NAME, **facet,
                    "analysis_group": group, "shot_group": "", **corr_meta,
                    "component_metric": component, "reference_metric": "oer_loss_total",
                    "correlation_method_pearson": "pearson", "correlation_method_spearman": "spearman",
                    "pearson": corr["pearson"], "spearman": corr["spearman"], "metric_value": corr["pearson"],
                    "valid_count": int(corr["valid_count"]), "total_count": int(len(part)),
                })
    return pd.DataFrame(rows)


def aggregate_by_target_and_group(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample = sample_df.copy() if set(SAMPLE_COLUMNS).issubset(sample_df.columns) else _normalise_sample_level_table(sample_df)
    return build_group_summary_v2(sample)


def compute_tail_danger_amplification(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample = sample_df.copy() if set(SAMPLE_COLUMNS).issubset(sample_df.columns) else _normalise_sample_level_table(sample_df)
    out = build_comparison_summary_v2(sample)
    return out[out["metric_name"].isin(DANGER_COMPONENT_METRICS.values())].reset_index(drop=True)


def compute_few_clip_amplification(sample_df: pd.DataFrame, *, few_label: str = "few", many_label: str = "many") -> pd.DataFrame:
    sample = sample_df.copy() if set(SAMPLE_COLUMNS).issubset(sample_df.columns) else _normalise_sample_level_table(sample_df)
    out = build_comparison_summary_v2(sample)
    return out[out["metric_name"].eq("boundary_hit_rate")].reset_index(drop=True)


def compute_few_near_zero_amplification(sample_df: pd.DataFrame, *, few_label: str = "few", many_label: str = "many") -> pd.DataFrame:
    sample = sample_df.copy() if set(SAMPLE_COLUMNS).issubset(sample_df.columns) else _normalise_sample_level_table(sample_df)
    out = build_comparison_summary_v2(sample)
    return out[out["metric_name"].eq("near_zero_pm1_rate")].reset_index(drop=True)


def compute_danger_oer_correlations(sample_df: pd.DataFrame) -> pd.DataFrame:
    sample = sample_df.copy() if set(SAMPLE_COLUMNS).issubset(sample_df.columns) else _normalise_sample_level_table(sample_df)
    return build_correlation_v2(sample)


def compute_clip_oer_correlation(sample_df: pd.DataFrame) -> pd.DataFrame:
    out = compute_danger_oer_correlations(sample_df)
    return out[out["component_metric"].isin(["boundary_hit_rate", "strict_overflow_rate"])].reset_index(drop=True)


def compute_near_zero_oer_correlation(sample_df: pd.DataFrame) -> pd.DataFrame:
    out = compute_danger_oer_correlations(sample_df)
    return out[out["component_metric"].isin(["near_zero_pm1_rate", "high_energy_near_zero_rate"])].reset_index(drop=True)


def build_summary_table(
    main_csv: str | Path | pd.DataFrame,
    *,
    cfg: Any | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_table = _normalise_sample_level_table(_read_main_csv(main_csv))
    summary = build_group_summary_v2(sample_table)
    comparison = build_comparison_summary_v2(sample_table)
    correlation = build_correlation_v2(sample_table)
    if save:
        target_dir = _resolve_output_dir(cfg=cfg, output_dir=output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        sample_table.to_csv(target_dir / SAMPLE_OUTPUT_FILENAME, index=False)
        summary.to_csv(target_dir / SUMMARY_OUTPUT_FILENAME, index=False)
        comparison.to_csv(target_dir / COMPARISON_SUMMARY_OUTPUT_FILENAME, index=False)
        correlation.to_csv(target_dir / CORRELATION_OUTPUT_FILENAME, index=False)
    return sample_table, summary, comparison, correlation


__all__ = [
    "AGGREGATE_VALUE_COLUMNS", "COMPARISON_SUMMARY_OUTPUT_FILENAME", "CORRELATION_COMPONENT_METRICS",
    "CORRELATION_OUTPUT_FILENAME", "DANGER_COMPONENT_METRICS", "DANGER_ZONE_METRIC_SPECS",
    "DEFAULT_QUANT_MODE", "FACET_COLUMNS", "GROUP_COLUMNS", "SAMPLE_COLUMNS", "SAMPLE_OUTPUT_FILENAME",
    "SUMMARY_OUTPUT_FILENAME", "aggregate_by_target_and_group", "build_comparison_summary_v2",
    "build_correlation_v2", "build_group_summary_v2", "build_summary_table", "compute_clip_oer_correlation",
    "compute_danger_oer_correlations", "compute_few_clip_amplification", "compute_few_near_zero_amplification",
    "compute_near_zero_oer_correlation", "compute_tail_danger_amplification", "group_protocol_metadata",
]
