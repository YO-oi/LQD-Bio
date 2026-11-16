from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from src.metrics.metric_config import DEFAULT_MARGIN_MIN_ABS
except ImportError:  # pragma: no cover - package-relative fallback
    from ..metrics.metric_config import DEFAULT_MARGIN_MIN_ABS


BASE_BUCKETS = ("many", "medium", "few")
TAIL_BUCKETS = ("medium", "few")
ANALYSIS_GROUPS = ("many", "medium", "few", "tail")
GROUP_PROTOCOL_VERSION = 2
GROUP_PROTOCOL_NAME = "bucket_analysis_group.v2"

METRIC_DIRECTIONS = ("higher_is_better", "lower_is_better", "diagnostic")
METRIC_ROLES = ("main", "detail", "formal_main", "formal_auxiliary", "diagnostic", "not_applicable")
AGGREGATION_UNITS = (
    "sample",
    "class",
    "token_sample_summary",
    "raw_token",
    "group_recompute",
    "global_derived",
    "histogram_bin",
)
RAW_RECORD_AGGREGATION_UNITS = (
    "sample",
    "class",
    "token_sample_summary",
    "raw_token",
)
AGGREGATION_FNS = (
    "mean",
    "nanmean",
    "median",
    "weighted_mean_by_sample",
    "macro_mean_by_class",
    "nanmean_valid_only",
    "sum",
    "ratio",
    "sum_then_normalize",
    "density_normalization",
    "scalar_weighted_mean",
    "token_weighted_mean",
    "fisher_global_anchored_recompute",
    "damage_gap_from_group_values",
    "group_gap",
    "group_max",
    "custom",
)

TRAIN_LABEL_MANY_MIN = 100
TRAIN_LABEL_MEDIUM_MIN = 20
DEFAULT_MARGIN_EPS = DEFAULT_MARGIN_MIN_ABS


class GroupValidationError(ValueError):
    """Raised when bucket or analysis-group metadata violates the protocol."""


@dataclass(frozen=True)
class MetricSpec:
    metric_name: str
    metric_direction: str
    aggregation_unit: str
    aggregation_fn: str
    metric_role: str = "main"

    def __post_init__(self) -> None:
        if self.metric_direction not in METRIC_DIRECTIONS:
            raise GroupValidationError(
                f"{self.metric_name}: metric_direction={self.metric_direction!r} is invalid; "
                f"expected one of {METRIC_DIRECTIONS}"
            )
        if self.aggregation_unit not in AGGREGATION_UNITS:
            raise GroupValidationError(
                f"{self.metric_name}: aggregation_unit={self.aggregation_unit!r} is invalid; "
                f"expected one of {AGGREGATION_UNITS}"
            )
        if not str(self.aggregation_fn).strip():
            raise GroupValidationError(f"{self.metric_name}: aggregation_fn must be explicit")
        if self.aggregation_fn not in AGGREGATION_FNS:
            raise GroupValidationError(
                f"{self.metric_name}: aggregation_fn={self.aggregation_fn!r} is not registered; "
                f"expected one of {AGGREGATION_FNS}"
            )
        if self.metric_role not in METRIC_ROLES:
            raise GroupValidationError(
                f"{self.metric_name}: metric_role={self.metric_role!r} is invalid; "
                f"expected one of {METRIC_ROLES}"
            )


def _metric_key(metric_name: Any) -> str:
    key = str(metric_name).strip().lower()
    if not key:
        raise GroupValidationError("metric_name must be non-empty")
    return key


def define_metric_spec(
    metric_name: str,
    *,
    metric_direction: str,
    aggregation_unit: str,
    aggregation_fn: str,
    metric_role: str = "main",
) -> MetricSpec:
    if str(aggregation_fn) not in AGGREGATION_FNS:
        raise GroupValidationError(
            f"{metric_name}: aggregation_fn={aggregation_fn!r} is not registered; "
            "use an explicit supported function or add it to AGGREGATION_FNS"
        )
    return MetricSpec(
        metric_name=str(metric_name),
        metric_direction=str(metric_direction),
        aggregation_unit=str(aggregation_unit),
        aggregation_fn=str(aggregation_fn),
        metric_role=str(metric_role),
    )


def build_metric_spec_registry(specs: Sequence[MetricSpec]) -> dict[str, MetricSpec]:
    registry: dict[str, MetricSpec] = {}
    duplicates: list[str] = []
    for spec in specs:
        key = _metric_key(spec.metric_name)
        if key in registry:
            duplicates.append(spec.metric_name)
        registry[key] = spec
    if duplicates:
        raise GroupValidationError(f"Duplicate MetricSpec entries: {duplicates}")
    return registry


DEFAULT_METRIC_SPECS = build_metric_spec_registry(
    (
        define_metric_spec(
            "embedding_distortion",
            metric_direction="lower_is_better",
            aggregation_unit="sample",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "oer_retention",
            metric_direction="higher_is_better",
            aggregation_unit="sample",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "oer_loss",
            metric_direction="lower_is_better",
            aggregation_unit="sample",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "clipping_rate",
            metric_direction="lower_is_better",
            aggregation_unit="sample",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "near_zero_rate",
            metric_direction="lower_is_better",
            aggregation_unit="sample",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "topk_survival_rate",
            metric_direction="higher_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "topk_overlap",
            metric_direction="higher_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "mean_norm_retention",
            metric_direction="higher_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "mean_token_distortion",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "token_distortion",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "token_oer_retention",
            metric_direction="higher_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "token_oer_loss",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "topk_token_exact_zero_rate",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "topk_token_near_zero_pm1_rate",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "topk_token_boundary_hit_rate",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "topk_token_strict_overflow_rate",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "high_energy_scalar_exact_zero_rate",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "high_energy_scalar_near_zero_pm1_rate",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "high_energy_scalar_boundary_hit_rate",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "high_energy_scalar_strict_overflow_rate",
            metric_direction="lower_is_better",
            aggregation_unit="token_sample_summary",
            aggregation_fn="mean",
        ),
        define_metric_spec(
            "margin_retention",
            metric_direction="higher_is_better",
            aggregation_unit="sample",
            aggregation_fn="nanmean_valid_only",
        ),
        define_metric_spec(
            "margin_fp32",
            metric_direction="diagnostic",
            aggregation_unit="sample",
            aggregation_fn="nanmean_valid_only",
            metric_role="detail",
        ),
        define_metric_spec(
            "margin_quant",
            metric_direction="diagnostic",
            aggregation_unit="sample",
            aggregation_fn="nanmean_valid_only",
            metric_role="detail",
        ),
        define_metric_spec(
            "delta_margin",
            metric_direction="lower_is_better",
            aggregation_unit="sample",
            aggregation_fn="nanmean_valid_only",
            metric_role="detail",
        ),
        define_metric_spec(
            "margin_valid_rate",
            metric_direction="diagnostic",
            aggregation_unit="sample",
            aggregation_fn="ratio",
            metric_role="detail",
        ),
        define_metric_spec(
            "effective_class_count",
            metric_direction="diagnostic",
            aggregation_unit="class",
            aggregation_fn="sum",
            metric_role="detail",
        ),
        define_metric_spec(
            "class_trace_variance",
            metric_direction="diagnostic",
            aggregation_unit="class",
            aggregation_fn="macro_mean_by_class",
        ),
        define_metric_spec(
            "class_center_shift",
            metric_direction="lower_is_better",
            aggregation_unit="class",
            aggregation_fn="macro_mean_by_class",
        ),
        define_metric_spec(
            "per_class_recall",
            metric_direction="higher_is_better",
            aggregation_unit="class",
            aggregation_fn="macro_mean_by_class",
        ),
        define_metric_spec(
            "per_class_f1",
            metric_direction="higher_is_better",
            aggregation_unit="class",
            aggregation_fn="macro_mean_by_class",
        ),
        define_metric_spec(
            "variance_inflation",
            metric_direction="diagnostic",
            aggregation_unit="class",
            aggregation_fn="macro_mean_by_class",
        ),
        define_metric_spec(
            "variance_inflation_deviation",
            metric_direction="lower_is_better",
            aggregation_unit="class",
            aggregation_fn="macro_mean_by_class",
        ),
        define_metric_spec(
            "fisher",
            metric_direction="higher_is_better",
            aggregation_unit="group_recompute",
            aggregation_fn="fisher_global_anchored_recompute",
        ),
        define_metric_spec(
            "fisher_fp32",
            metric_direction="diagnostic",
            aggregation_unit="group_recompute",
            aggregation_fn="fisher_global_anchored_recompute",
            metric_role="detail",
        ),
        define_metric_spec(
            "fisher_quant",
            metric_direction="diagnostic",
            aggregation_unit="group_recompute",
            aggregation_fn="fisher_global_anchored_recompute",
            metric_role="detail",
        ),
        define_metric_spec(
            "fisher_retention",
            metric_direction="higher_is_better",
            aggregation_unit="group_recompute",
            aggregation_fn="fisher_global_anchored_recompute",
        ),
        define_metric_spec(
            "delta_fisher",
            metric_direction="lower_is_better",
            aggregation_unit="global_derived",
            aggregation_fn="custom",
            metric_role="detail",
        ),
        define_metric_spec(
            "tail_vs_many_gap",
            metric_direction="diagnostic",
            aggregation_unit="global_derived",
            aggregation_fn="damage_gap_from_group_values",
        ),
        define_metric_spec(
            "few_vs_many_gap",
            metric_direction="diagnostic",
            aggregation_unit="global_derived",
            aggregation_fn="damage_gap_from_group_values",
        ),
        define_metric_spec(
            "collapse_gap",
            metric_direction="diagnostic",
            aggregation_unit="global_derived",
            aggregation_fn="group_gap",
        ),
        define_metric_spec(
            "histogram_density",
            metric_direction="diagnostic",
            aggregation_unit="histogram_bin",
            aggregation_fn="sum_then_normalize",
        ),
    )
)


def require_metric_spec(
    metric_name: str,
    *,
    registry: Mapping[str, MetricSpec] | None = None,
) -> MetricSpec:
    specs = DEFAULT_METRIC_SPECS if registry is None else registry
    key = _metric_key(metric_name)
    if key not in specs:
        raise GroupValidationError(
            f"MetricSpec for {metric_name!r} is required before group aggregation; "
            "refusing to default to mean over numeric columns"
        )
    return specs[key]


def require_metric_specs(
    metric_names: Sequence[str],
    *,
    registry: Mapping[str, MetricSpec] | None = None,
) -> dict[str, MetricSpec]:
    return {str(name): require_metric_spec(str(name), registry=registry) for name in metric_names}


def metric_spec_metadata(spec: MetricSpec) -> dict[str, str]:
    return {
        "metric_name": spec.metric_name,
        "metric_role": spec.metric_role,
        "metric_direction": spec.metric_direction,
        "aggregation_unit": spec.aggregation_unit,
        "aggregation_fn": spec.aggregation_fn,
    }



def group_protocol_metadata(
    *,
    tail_rule: str = "tail is recomputed from raw records with bucket in {medium, few}",
    canonical_group_field: str = "bucket",
    legacy_group_alias: str = "shot_group",
    main_comparison_group: str = "tail",
    supplementary_extreme_group: str = "few",
) -> dict[str, Any]:
    """Return the canonical v2 bucket/analysis-group protocol metadata."""

    return {
        "group_protocol_version": GROUP_PROTOCOL_VERSION,
        "group_protocol_name": GROUP_PROTOCOL_NAME,
        "base_buckets": list(BASE_BUCKETS),
        "sample_bucket_values": list(BASE_BUCKETS),
        "canonical_group_field": str(canonical_group_field),
        "legacy_group_alias": str(legacy_group_alias),
        "shot_group_rule": "optional legacy column; when present it must equal bucket row-by-row",
        "tail_included_rule": "optional bool column validated against bucket in {medium, few}",
        "analysis_groups": list(ANALYSIS_GROUPS),
        "aggregate_analysis_groups": list(ANALYSIS_GROUPS),
        "tail_definition": "tail = medium + few",
        "tail_buckets": list(TAIL_BUCKETS),
        "tail_is_bucket": False,
        "sample_bucket_tail_forbidden": True,
        "main_comparison_group": str(main_comparison_group),
        "supplementary_extreme_group": str(supplementary_extreme_group),
        "tail_rule": str(tail_rule),
    }


@dataclass(frozen=True)
class BucketValidationResult:
    bucket: list[str]
    warnings: tuple[str, ...] = ()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "<na>"}


def _as_list(values: Any, *, name: str) -> list[Any]:
    if values is None:
        raise GroupValidationError(f"{name} is missing")
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not a single string")
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        return values.detach().cpu().reshape(-1).tolist()
    if hasattr(values, "to_numpy"):
        return values.to_numpy().reshape(-1).tolist()
    if hasattr(values, "reshape") and hasattr(values, "tolist") and not isinstance(values, list):
        try:
            return values.reshape(-1).tolist()
        except TypeError:
            pass
    return list(values)


def _same_length(left: Sequence[Any], right: Sequence[Any], *, left_name: str, right_name: str) -> None:
    if len(left) != len(right):
        raise GroupValidationError(
            f"{left_name} length={len(left)} must match {right_name} length={len(right)}"
        )


def normalize_bucket(value: Any, *, field_name: str = "bucket") -> str:
    if _is_missing(value):
        raise GroupValidationError(f"{field_name} is missing")
    bucket = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if bucket == "tail":
        raise GroupValidationError(f"{field_name} must not contain derived group 'tail'")
    if bucket not in BASE_BUCKETS:
        raise GroupValidationError(f"{field_name}={value!r} is invalid; expected one of {BASE_BUCKETS}")
    return bucket


def normalize_analysis_group(value: Any) -> str:
    if _is_missing(value):
        raise GroupValidationError("analysis_group is missing")
    group = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if group not in ANALYSIS_GROUPS:
        raise GroupValidationError(f"analysis_group={value!r} is invalid; expected one of {ANALYSIS_GROUPS}")
    return group


def bucket_from_train_label_n(train_label_n: Any) -> str:
    count = parse_train_label_n(train_label_n)
    if count >= TRAIN_LABEL_MANY_MIN:
        return "many"
    if count >= TRAIN_LABEL_MEDIUM_MIN:
        return "medium"
    return "few"


def parse_train_label_n(value: Any) -> int:
    if _is_missing(value) or isinstance(value, bool):
        raise GroupValidationError(f"train_label_n={value!r} is invalid")
    if isinstance(value, int):
        parsed = int(value)
    elif isinstance(value, float):
        if not (math.isfinite(value) and value.is_integer()):
            raise GroupValidationError(f"train_label_n={value!r} is invalid")
        parsed = int(value)
    else:
        text = str(value).strip()
        try:
            numeric = float(text)
        except ValueError as exc:
            raise GroupValidationError(f"train_label_n={value!r} is invalid") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise GroupValidationError(f"train_label_n={value!r} is invalid")
        parsed = int(numeric)
    if parsed < 0:
        raise GroupValidationError(f"train_label_n={value!r} must be non-negative")
    return parsed


def validate_bucket_values(
    bucket: Sequence[Any],
    *,
    shot_group: Sequence[Any] | None = None,
    train_label_n: Sequence[Any] | None = None,
    context: str = "bucket",
) -> list[str]:
    buckets = [normalize_bucket(item, field_name=f"{context}.bucket") for item in _as_list(bucket, name=f"{context}.bucket")]

    if shot_group is not None:
        shot_groups = [
            normalize_bucket(item, field_name=f"{context}.shot_group")
            for item in _as_list(shot_group, name=f"{context}.shot_group")
        ]
        _same_length(buckets, shot_groups, left_name=f"{context}.bucket", right_name=f"{context}.shot_group")
        mismatches = [
            idx for idx, (left, right) in enumerate(zip(buckets, shot_groups)) if str(left) != str(right)
        ]
        if mismatches:
            preview = ", ".join(str(idx) for idx in mismatches[:8])
            raise GroupValidationError(f"{context}: shot_group must equal bucket row-by-row; mismatch indices={preview}")

    if train_label_n is not None:
        counts = _as_list(train_label_n, name=f"{context}.train_label_n")
        _same_length(buckets, counts, left_name=f"{context}.bucket", right_name=f"{context}.train_label_n")
        mismatches = []
        for idx, (observed, count) in enumerate(zip(buckets, counts)):
            expected = bucket_from_train_label_n(count)
            if observed != expected:
                mismatches.append((idx, observed, expected, parse_train_label_n(count)))
        if mismatches:
            examples = ", ".join(
                f"idx={idx}: bucket={observed}, expected={expected}, train_label_n={count}"
                for idx, observed, expected, count in mismatches[:8]
            )
            raise GroupValidationError(f"{context}: train_label_n is inconsistent with bucket thresholds; {examples}")

    return buckets


def read_canonical_bucket(
    *,
    bucket: Sequence[Any] | None = None,
    shot_group: Sequence[Any] | None = None,
    train_label_n: Sequence[Any] | None = None,
    context: str = "bucket",
    allow_shot_group_fallback: bool = False,
) -> BucketValidationResult:
    warnings: list[str] = []
    if bucket is None:
        if shot_group is None or not allow_shot_group_fallback:
            raise GroupValidationError(f"{context}: bucket is required")
        warnings.append(f"{context}: bucket missing; used legacy shot_group fallback as canonical bucket")
        canonical = validate_bucket_values(shot_group, train_label_n=train_label_n, context=context)
        return BucketValidationResult(bucket=canonical, warnings=tuple(warnings))

    canonical = validate_bucket_values(
        bucket,
        shot_group=shot_group,
        train_label_n=train_label_n,
        context=context,
    )
    return BucketValidationResult(bucket=canonical, warnings=tuple(warnings))


def _mask_like(mask: list[bool], reference: Any) -> Any:
    module = type(reference).__module__.split(".", 1)[0]
    if module == "pandas" and hasattr(reference, "index"):
        import pandas as pd

        return pd.Series(mask, index=reference.index, dtype=bool)
    if module == "numpy":
        import numpy as np

        return np.asarray(mask, dtype=bool)
    if module == "torch":
        import torch

        return torch.tensor(mask, dtype=torch.bool)
    return mask


def analysis_group_mask(bucket: Sequence[Any], analysis_group: Any) -> Any:
    group = normalize_analysis_group(analysis_group)
    buckets = validate_bucket_values(bucket)
    if group == "tail":
        mask = [item in TAIL_BUCKETS for item in buckets]
    else:
        mask = [item == group for item in buckets]
    return _mask_like(mask, bucket)


def tail_included(bucket: Sequence[Any]) -> Any:
    return analysis_group_mask(bucket, "tail")


def add_tail_included_column(
    frame: Any,
    *,
    bucket_col: str = "bucket",
    output_col: str = "tail_included",
) -> Any:
    if bucket_col not in frame.columns:
        raise GroupValidationError(f"frame is missing required bucket column {bucket_col!r}")
    work = frame.copy()
    work[output_col] = tail_included(work[bucket_col])
    return work


def canonicalize_bucket_frame(
    frame: Any,
    *,
    bucket_col: str = "bucket",
    shot_group_col: str = "shot_group",
    train_label_n_col: str = "train_label_n",
    add_tail_included: bool = False,
    context: str = "frame",
) -> tuple[Any, tuple[str, ...]]:
    work = frame.copy()
    bucket_values = work[bucket_col] if bucket_col in work.columns else None
    shot_values = work[shot_group_col] if shot_group_col in work.columns else None
    train_values = work[train_label_n_col] if train_label_n_col in work.columns else None
    result = read_canonical_bucket(
        bucket=bucket_values,
        shot_group=shot_values,
        train_label_n=train_values,
        context=context,
    )
    work[bucket_col] = result.bucket
    if shot_group_col in work.columns:
        work[shot_group_col] = [normalize_bucket(item, field_name=f"{context}.shot_group") for item in work[shot_group_col]]
    if add_tail_included:
        work["tail_included"] = [item in TAIL_BUCKETS for item in result.bucket]
    return work, result.warnings


def _as_float_list(values: Any, *, name: str) -> list[float]:
    result: list[float] = []
    for value in _as_list(values, name=name):
        if _is_missing(value):
            result.append(float("nan"))
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError) as exc:
            raise GroupValidationError(f"{name} contains non-numeric value {value!r}") from exc
    return result


def _as_float_scalar(value: Any, *, name: str) -> float:
    if _is_missing(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise GroupValidationError(f"{name} contains non-numeric value {value!r}") from exc

def _finite(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _aggregate_numeric_values(
    values: Sequence[float],
    *,
    aggregation_fn: str,
    weights: Sequence[Any] | None = None,
) -> float:
    fn = str(aggregation_fn)
    finite_values = _finite(values)
    if fn in {"mean", "nanmean", "macro_mean_by_class", "nanmean_valid_only"}:
        if not finite_values:
            return float("nan")
        return float(sum(finite_values) / len(finite_values))
    if fn == "median":
        if not finite_values:
            return float("nan")
        ordered = sorted(finite_values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return float((ordered[mid - 1] + ordered[mid]) / 2.0)
    if fn == "sum":
        if not finite_values:
            return float("nan")
        return float(sum(finite_values))
    if fn in {"weighted_mean_by_sample", "scalar_weighted_mean", "token_weighted_mean"}:
        if weights is None:
            raise GroupValidationError(f"aggregation_fn={fn!r} requires explicit weights")
        weight_values = _as_float_list(weights, name="weights")
        if len(weight_values) != len(values):
            raise GroupValidationError(f"weights length={len(weight_values)} must match values length={len(values)}")
        numerator = 0.0
        denominator = 0.0
        for value, weight in zip(values, weight_values):
            if not (math.isfinite(float(value)) and math.isfinite(float(weight))):
                continue
            if float(weight) < 0:
                raise GroupValidationError("weights must be non-negative")
            numerator += float(value) * float(weight)
            denominator += float(weight)
        return float(numerator / denominator) if denominator > 0.0 else float("nan")
    raise GroupValidationError(
        f"aggregation_fn={fn!r} cannot be applied by raw-record numeric aggregation; "
        "use a dedicated recompute or derived-metric helper"
    )


def aggregate_metric_for_analysis_group_from_raw_records(
    values: Sequence[Any],
    bucket: Sequence[Any],
    analysis_group: str,
    *,
    metric_name: str | None = None,
    spec: MetricSpec | None = None,
    record_unit: str,
    weights: Sequence[Any] | None = None,
    registry: Mapping[str, MetricSpec] | None = None,
) -> float:
    """Aggregate one analysis group from raw records, including tail.

    This is the hard entry point for tail = medium + few. It requires a raw
    record unit matching MetricSpec.aggregation_unit and refuses derived or
    recompute-only metrics, so callers cannot silently compute tail from
    many/medium/few summary rows.
    """

    resolved_spec = spec if spec is not None else require_metric_spec(str(metric_name), registry=registry)
    unit = str(record_unit)
    if unit != resolved_spec.aggregation_unit:
        raise GroupValidationError(
            f"{resolved_spec.metric_name}: record_unit={unit!r} must match "
            f"MetricSpec.aggregation_unit={resolved_spec.aggregation_unit!r}"
        )
    if resolved_spec.aggregation_unit not in RAW_RECORD_AGGREGATION_UNITS:
        raise GroupValidationError(
            f"{resolved_spec.metric_name}: aggregation_unit={resolved_spec.aggregation_unit!r} "
            "cannot be aggregated from raw value records by this helper"
        )

    numeric_values = _as_float_list(values, name=f"{resolved_spec.metric_name}.values")
    buckets = validate_bucket_values(bucket, context=f"{resolved_spec.metric_name}.raw_records")
    if len(numeric_values) != len(buckets):
        raise GroupValidationError(
            f"{resolved_spec.metric_name}: values length={len(numeric_values)} must match bucket length={len(buckets)}"
        )
    mask = _as_list(analysis_group_mask(buckets, analysis_group), name="analysis_group_mask")
    selected = [value for value, keep in zip(numeric_values, mask) if bool(keep)]
    selected_weights = None
    if weights is not None:
        weight_values = _as_list(weights, name="weights")
        if len(weight_values) != len(buckets):
            raise GroupValidationError(f"weights length={len(weight_values)} must match bucket length={len(buckets)}")
        selected_weights = [weight for weight, keep in zip(weight_values, mask) if bool(keep)]
    return _aggregate_numeric_values(
        selected,
        aggregation_fn=resolved_spec.aggregation_fn,
        weights=selected_weights,
    )


def aggregate_metric_by_analysis_group_from_raw_records(
    values: Sequence[Any],
    bucket: Sequence[Any],
    *,
    metric_name: str | None = None,
    spec: MetricSpec | None = None,
    record_unit: str,
    analysis_groups: Sequence[str] = ANALYSIS_GROUPS,
    weights: Sequence[Any] | None = None,
    registry: Mapping[str, MetricSpec] | None = None,
) -> dict[str, float]:
    resolved_spec = spec if spec is not None else require_metric_spec(str(metric_name), registry=registry)
    return {
        normalize_analysis_group(group): aggregate_metric_for_analysis_group_from_raw_records(
            values,
            bucket,
            str(group),
            spec=resolved_spec,
            record_unit=record_unit,
            weights=weights,
            registry=registry,
        )
        for group in analysis_groups
    }


def tail_value_from_raw_records(
    values: Sequence[Any],
    bucket: Sequence[Any],
    *,
    metric_name: str | None = None,
    spec: MetricSpec | None = None,
    record_unit: str,
    weights: Sequence[Any] | None = None,
    registry: Mapping[str, MetricSpec] | None = None,
) -> float:
    return aggregate_metric_for_analysis_group_from_raw_records(
        values,
        bucket,
        "tail",
        metric_name=metric_name,
        spec=spec,
        record_unit=record_unit,
        weights=weights,
        registry=registry,
    )


def _common_margin_value_retention():
    try:
        from src.metrics.common_metrics import margin_value_retention
    except ImportError:  # pragma: no cover - package-relative fallback
        from ..metrics.common_metrics import margin_value_retention
    return margin_value_retention


def _resolve_margin_min_abs(*, margin_min_abs: float = DEFAULT_MARGIN_MIN_ABS, eps: float | None = None) -> float:
    threshold = float(margin_min_abs)
    if eps is not None and float(eps) != threshold:
        raise GroupValidationError(
            "margin_retention uses fixed margin_min_abs; "
            f"got eps={float(eps)} but margin_min_abs={threshold}"
        )
    return threshold


def _tensor_scalar_to_int(value: Any) -> int:
    if hasattr(value, "detach"):
        return int(value.detach().cpu().item())
    return int(value)


def margin_retention_invalid_reason(
    margin_fp32: Any,
    margin_quant: Any,
    *,
    margin_min_abs: float = DEFAULT_MARGIN_MIN_ABS,
    eps: float | None = None,
) -> str | None:
    """Return why margin_retention is invalid, or None when it may be computed."""

    fp32 = _as_float_scalar(margin_fp32, name="margin_fp32")
    quant = _as_float_scalar(margin_quant, name="margin_quant")
    threshold = _resolve_margin_min_abs(margin_min_abs=margin_min_abs, eps=eps)
    if math.isnan(fp32):
        return "margin_fp32 is NaN"
    if math.isinf(fp32):
        return "margin_fp32 is Inf"
    if fp32 <= threshold:
        return "margin_fp32 <= margin_min_abs"
    if math.isnan(quant):
        return "margin_quant is NaN"
    if math.isinf(quant):
        return "margin_quant is Inf"
    return None


def compute_margin_retention_value(
    margin_fp32: Any,
    margin_quant: Any,
    *,
    margin_min_abs: float = DEFAULT_MARGIN_MIN_ABS,
    eps: float | None = None,
) -> float:
    """Compute retention through common_metrics.margin_value_retention."""

    threshold = _resolve_margin_min_abs(margin_min_abs=margin_min_abs, eps=eps)
    retention = _common_margin_value_retention()(
        [margin_fp32],
        [margin_quant],
        reduction="none",
        min_abs_margin=threshold,
    )
    values = retention.detach().cpu().reshape(-1).tolist()
    return float(values[0]) if values else float("nan")


def _format_invalid_reason_counts(counts: Mapping[str, int]) -> str:
    parts = [f"{reason}:{int(count)}" for reason, count in sorted(counts.items()) if int(count) > 0]
    return ";".join(parts) if parts else "none"


def summarize_margin_retention(
    margin_fp32: Sequence[Any],
    margin_quant: Sequence[Any],
    *,
    bucket: Sequence[Any] | None = None,
    analysis_group: str | None = None,
    margin_min_abs: float = DEFAULT_MARGIN_MIN_ABS,
    eps: float | None = None,
) -> dict[str, Any]:
    """Summarize margin_retention with the required valid-rate metadata.

    If bucket and analysis_group are provided, records are first filtered by the
    canonical analysis-group mask, so tail is recomputed from medium + few raw
    records rather than from pre-aggregated group values.
    """

    fp32_values = _as_float_list(margin_fp32, name="margin_fp32")
    quant_values = _as_float_list(margin_quant, name="margin_quant")
    _same_length(fp32_values, quant_values, left_name="margin_fp32", right_name="margin_quant")
    threshold = _resolve_margin_min_abs(margin_min_abs=margin_min_abs, eps=eps)

    if (bucket is None) != (analysis_group is None):
        raise GroupValidationError("bucket and analysis_group must be provided together for group margin_retention")
    if bucket is not None and analysis_group is not None:
        buckets = validate_bucket_values(bucket, context="margin_retention")
        _same_length(fp32_values, buckets, left_name="margin_fp32", right_name="bucket")
        mask = _as_list(analysis_group_mask(buckets, analysis_group), name="analysis_group_mask")
        fp32_values = [value for value, keep in zip(fp32_values, mask) if bool(keep)]
        quant_values = [value for value, keep in zip(quant_values, mask) if bool(keep)]

    retention_tensor, diagnostics = _common_margin_value_retention()(
        fp32_values,
        quant_values,
        reduction="none",
        min_abs_margin=threshold,
        return_diagnostics=True,
    )
    retention_values = [float(value) for value in retention_tensor.detach().cpu().reshape(-1).tolist()]
    invalid_counts: dict[str, int] = {}
    for fp32, quant, retention in zip(fp32_values, quant_values, retention_values):
        if not math.isfinite(retention):
            reason = margin_retention_invalid_reason(fp32, quant, margin_min_abs=threshold)
            if reason is None:
                reason = "margin_retention is not finite"
            invalid_counts[reason] = int(invalid_counts.get(reason, 0)) + 1

    total_count = len(fp32_values)
    valid_count = _tensor_scalar_to_int(diagnostics["valid_count"])
    valid_rate = float(valid_count / total_count) if total_count > 0 else float("nan")
    return {
        "margin_retention": _aggregate_numeric_values(
            retention_values,
            aggregation_fn="nanmean_valid_only",
        ),
        "margin_retention_values": retention_values,
        "margin_valid_count": int(valid_count),
        "margin_total_count": int(total_count),
        "margin_valid_rate": valid_rate,
        "margin_invalid_reason": _format_invalid_reason_counts(invalid_counts),
        "margin_invalid_reason_counts": dict(sorted(invalid_counts.items())),
    }


def compute_damage_gap(
    *,
    baseline_value: Any,
    comparison_value: Any,
    metric_direction: str,
    diagnostic_gap_direction: str | None = None,
) -> float:
    direction = str(metric_direction)
    if direction == "diagnostic":
        if diagnostic_gap_direction is None:
            return float("nan")
        direction = str(diagnostic_gap_direction)
    if direction not in {"higher_is_better", "lower_is_better"}:
        raise GroupValidationError(f"metric_direction={metric_direction!r} is invalid for gap computation")

    baseline = float(baseline_value)
    comparison = float(comparison_value)
    if math.isnan(baseline) or math.isnan(comparison):
        return float("nan")
    if direction == "higher_is_better":
        return baseline - comparison
    return comparison - baseline


def compute_group_gap(
    group_values: Mapping[str, Any],
    *,
    comparison_group: str,
    baseline_group: str = "many",
    metric_direction: str,
    diagnostic_gap_direction: str | None = None,
) -> float:
    baseline = normalize_analysis_group(baseline_group)
    comparison = normalize_analysis_group(comparison_group)
    return compute_damage_gap(
        baseline_value=group_values.get(baseline, float("nan")),
        comparison_value=group_values.get(comparison, float("nan")),
        metric_direction=metric_direction,
        diagnostic_gap_direction=diagnostic_gap_direction,
    )


def compute_tail_and_few_gaps(
    group_values: Mapping[str, Any],
    *,
    metric_direction: str,
    diagnostic_gap_direction: str | None = None,
) -> dict[str, float]:
    return {
        "tail_vs_many_gap": compute_group_gap(
            group_values,
            comparison_group="tail",
            metric_direction=metric_direction,
            diagnostic_gap_direction=diagnostic_gap_direction,
        ),
        "few_vs_many_gap": compute_group_gap(
            group_values,
            comparison_group="few",
            metric_direction=metric_direction,
            diagnostic_gap_direction=diagnostic_gap_direction,
        ),
    }


def _metric_summary_float(value: Any, *, name: str, default: float = float("nan"), strict: bool = False) -> float:
    if _is_missing(value):
        return float(default)
    try:
        return _as_float_scalar(value, name=name)
    except GroupValidationError:
        if strict:
            raise
        return float("nan")


def build_metric_summary_rows_from_group_detail(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    id_columns: Sequence[str],
    group_column: str = "analysis_group",
    per_group_columns: Sequence[str] = (),
    integer_per_group_columns: Sequence[str] = (),
    required_analysis_groups: Sequence[str] = ANALYSIS_GROUPS,
    metric_value_columns: Mapping[str, str] | None = None,
    registry: Mapping[str, MetricSpec] | None = None,
    diagnostic_gap_direction: str | None = None,
    min_group_protocol_version: int = GROUP_PROTOCOL_VERSION,
    context: str = "metric summary",
) -> list[dict[str, Any]]:
    """Build standard metric summary rows from protocol-v2 group detail rows."""

    id_keys = tuple(str(column) for column in id_columns)
    if "group_protocol_version" not in id_keys:
        raise GroupValidationError(f"{context}: id_columns must include group_protocol_version")

    required_groups = tuple(normalize_analysis_group(group) for group in required_analysis_groups)
    missing_standard = [group for group in ANALYSIS_GROUPS if group not in required_groups]
    if missing_standard:
        raise GroupValidationError(f"{context}: required_analysis_groups must include {ANALYSIS_GROUPS}")

    int_columns = {str(column) for column in integer_per_group_columns}
    group_columns = tuple(str(column) for column in per_group_columns)
    value_columns = {str(key): str(value) for key, value in (metric_value_columns or {}).items()}

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(column, "") for column in id_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, part in grouped.items():
        rows_by_group: dict[str, Mapping[str, Any]] = {}
        for row in part:
            group = normalize_analysis_group(row.get(group_column, ""))
            if group in rows_by_group:
                raise GroupValidationError(f"{context}: duplicate detail row for key={key!r}, analysis_group={group!r}")
            rows_by_group[group] = row

        missing = [group for group in required_groups if group not in rows_by_group]
        if missing:
            raise GroupValidationError(f"{context}: key={key!r} missing analysis_group rows: {missing}")

        tail_row = rows_by_group["tail"]
        group_protocol_version = _metric_summary_float(
            tail_row.get("group_protocol_version", float("nan")),
            name=f"{context}.group_protocol_version",
            strict=True,
        )
        if not math.isfinite(group_protocol_version) or int(group_protocol_version) < int(min_group_protocol_version):
            raise GroupValidationError(
                f"{context}: requires group_protocol_version >= {int(min_group_protocol_version)}; "
                f"got {tail_row.get('group_protocol_version')!r}"
            )

        common = {
            column: (int(group_protocol_version) if column == "group_protocol_version" else tail_row.get(column, ""))
            for column in id_keys
        }

        per_group_values: dict[str, dict[str, Any]] = {}
        for group in required_groups:
            group_row = rows_by_group[group]
            values: dict[str, Any] = {}
            for column in group_columns:
                default = 0.0 if column in int_columns else float("nan")
                number = _metric_summary_float(
                    group_row.get(column, default),
                    name=f"{context}.{group}.{column}",
                    default=default,
                    strict=column in int_columns,
                )
                if column in int_columns:
                    if not math.isfinite(number):
                        raise GroupValidationError(f"{context}: {group}.{column} must be finite")
                    values[column] = int(number)
                else:
                    values[column] = number
            per_group_values[group] = values

        for metric_name in metric_names:
            spec = require_metric_spec(metric_name, registry=registry)
            metadata = metric_spec_metadata(spec)
            metric_column = value_columns.get(str(metric_name), str(metric_name))
            group_values = {
                group: _metric_summary_float(
                    rows_by_group[group].get(metric_column, float("nan")),
                    name=f"{context}.{group}.{metric_column}",
                )
                for group in required_groups
            }
            gaps = compute_tail_and_few_gaps(
                group_values,
                metric_direction=spec.metric_direction,
                diagnostic_gap_direction=diagnostic_gap_direction,
            )
            row = {
                **common,
                **metadata,
                "many_value": group_values["many"],
                "medium_value": group_values["medium"],
                "few_value": group_values["few"],
                "tail_value": group_values["tail"],
                "tail_vs_many_gap": gaps["tail_vs_many_gap"],
                "few_vs_many_gap": gaps["few_vs_many_gap"],
            }
            for group in required_groups:
                for column, value in per_group_values[group].items():
                    row[f"{group}_{column}"] = value
            summary_rows.append(row)
    return summary_rows


__all__ = [
    "AGGREGATION_FNS",
    "RAW_RECORD_AGGREGATION_UNITS",
    "AGGREGATION_UNITS",
    "ANALYSIS_GROUPS",
    "BASE_BUCKETS",
    "DEFAULT_METRIC_SPECS",
    "DEFAULT_MARGIN_MIN_ABS",
    "DEFAULT_MARGIN_EPS",
    "GROUP_PROTOCOL_NAME",
    "GROUP_PROTOCOL_VERSION",
    "METRIC_DIRECTIONS",
    "METRIC_ROLES",
    "TAIL_BUCKETS",
    "BucketValidationResult",
    "GroupValidationError",
    "MetricSpec",
    "add_tail_included_column",
    "aggregate_metric_by_analysis_group_from_raw_records",
    "aggregate_metric_for_analysis_group_from_raw_records",
    "analysis_group_mask",
    "build_metric_spec_registry",
    "build_metric_summary_rows_from_group_detail",
    "bucket_from_train_label_n",
    "canonicalize_bucket_frame",
    "compute_damage_gap",
    "compute_group_gap",
    "compute_margin_retention_value",
    "compute_tail_and_few_gaps",
    "define_metric_spec",
    "group_protocol_metadata",
    "margin_retention_invalid_reason",
    "metric_spec_metadata",
    "normalize_analysis_group",
    "normalize_bucket",
    "parse_train_label_n",
    "read_canonical_bucket",
    "require_metric_spec",
    "require_metric_specs",
    "summarize_margin_retention",
    "tail_included",
    "tail_value_from_raw_records",
    "validate_bucket_values",
]





