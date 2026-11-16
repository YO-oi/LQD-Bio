from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

try:
    from .common_metrics import (
        DEFAULT_EPS,
        DEFAULT_FISHER_NORMALIZE_BY_DOF,
        DEFAULT_MARGIN_MIN_ABS,
        embedding_distortion as common_embedding_distortion,
        margin_value_retention,
        metric_drop,
    )
    from .geometry_metrics import (
        DEFAULT_VARIANCE_NORMALIZATION,
        VarianceNormalization,
        class_center_cosine_margin,
        class_trace_variance,
        compute_group_fisher,
        compute_group_margin,
    )
    from ..utils.group_utils import (
        ANALYSIS_GROUPS,
        GROUP_PROTOCOL_NAME,
        GROUP_PROTOCOL_VERSION,
        MetricSpec,
        analysis_group_mask,
        build_metric_spec_registry,
        compute_group_gap,
        compute_tail_and_few_gaps,
        define_metric_spec,
        metric_spec_metadata,
        read_canonical_bucket,
        tail_included,
    )
except ImportError:  # pragma: no cover - direct module debugging fallback.
    from src.metrics.common_metrics import (
        DEFAULT_EPS,
        DEFAULT_FISHER_NORMALIZE_BY_DOF,
        DEFAULT_MARGIN_MIN_ABS,
        embedding_distortion as common_embedding_distortion,
        margin_value_retention,
        metric_drop,
    )
    from src.metrics.geometry_metrics import (
        DEFAULT_VARIANCE_NORMALIZATION,
        VarianceNormalization,
        class_center_cosine_margin,
        class_trace_variance,
        compute_group_fisher,
        compute_group_margin,
    )
    from src.utils.group_utils import (
        ANALYSIS_GROUPS,
        GROUP_PROTOCOL_NAME,
        GROUP_PROTOCOL_VERSION,
        MetricSpec,
        analysis_group_mask,
        build_metric_spec_registry,
        compute_group_gap,
        compute_tail_and_few_gaps,
        define_metric_spec,
        metric_spec_metadata,
        read_canonical_bucket,
        tail_included,
    )


DEFAULT_ANALYSIS_GROUPS = tuple(ANALYSIS_GROUPS)
DEFAULT_SHOT_GROUPS = DEFAULT_ANALYSIS_GROUPS
REFERENCE_CACHE_TYPE = "fp32_semantic_geometry_reference"

SAMPLE_LEVEL_COLUMNS = [
    "sample_id", "label", "label_id", "bucket", "shot_group", "tail_included", "quant_mode", "granularity",
    "embedding_distortion", "margin_definition",
    "fp32_embedding_centroid_margin", "quant_embedding_centroid_margin",
    "embedding_centroid_margin_retention", "embedding_centroid_margin_drop",
    "fp32_margin", "quant_margin", "margin_retention", "margin_drop",
    "fp32_logits_margin", "quant_logits_margin", "logits_margin_retention", "logits_margin_drop",
    "logits_margin_available", "fp32_pred", "quant_pred", "correct_fp32", "correct_quant",
]
CLASS_LEVEL_COLUMNS = [
    "label", "label_id", "bucket", "shot_group", "tail_included", "num_samples", "quant_mode", "granularity",
    "fp32_variance", "quant_variance", "variance_inflation", "variance_inflation_deviation",
    "fp32_embedding_centroid_margin", "quant_embedding_centroid_margin",
    "embedding_centroid_margin_retention", "embedding_centroid_margin_drop",
    "fp32_margin", "quant_margin", "margin_retention", "margin_drop",
    "margin_valid_ratio",
    "fp32_center_cosine_margin", "quant_center_cosine_margin",
    "center_cosine_margin_retention", "center_cosine_margin_drop",
]
GROUP_LEVEL_COLUMNS = [
    "bucket", "analysis_group", "shot_group", "quant_mode", "granularity", "num_samples", "num_classes",
    "fisher_fp32", "fisher_quant", "fisher_retention",
    "embedding_centroid_margin_retention", "embedding_centroid_margin_damage", "margin_retention",
    "center_cosine_margin_retention", "variance_inflation", "variance_inflation_deviation", "embedding_distortion_mean",
    "fisher_damage", "margin_damage", "group_protocol_version", "fisher_normalization",
    "margin_valid_count", "margin_total_count", "margin_valid_rate", "margin_invalid_count",
    "margin_nonpositive_count", "margin_near_threshold_count", "margin_nan_count", "margin_inf_count",
    "margin_invalid_reason",
]
GLOBAL_LEVEL_COLUMNS = [
    "quant_mode", "granularity", "num_samples", "num_classes", "fisher_fp32",
    "fisher_quant", "global_fisher_retention",
    "many_embedding_centroid_margin_retention", "tail_embedding_centroid_margin_retention",
    "few_embedding_centroid_margin_retention", "many_margin_retention",
    "tail_margin_retention", "few_margin_retention", "tail_collapse_gap", "few_collapse_gap", "collapse_gap",
    "many_variance_inflation", "tail_variance_inflation", "few_variance_inflation",
    "many_variance_inflation_deviation", "tail_variance_inflation_deviation", "few_variance_inflation_deviation",
    "tail_vs_many_variance_inflation_deviation_gap", "few_vs_many_variance_inflation_deviation_gap",
    "embedding_distortion_mean",
]

@dataclass(frozen=True)
class SemanticGeometryMetricSpec:
    source_column: str
    spec: MetricSpec
    comparable_across_groups: bool
    gap_enabled: bool


_DEF = define_metric_spec
SEMANTIC_GEOMETRY_METRIC_SPECS: dict[str, SemanticGeometryMetricSpec] = {
    "embedding_distortion": SemanticGeometryMetricSpec(
        "embedding_distortion_mean",
        _DEF("embedding_distortion", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean"),
        True,
        True,
    ),
    "fisher_retention": SemanticGeometryMetricSpec(
        "fisher_retention",
        _DEF("fisher_retention", metric_direction="higher_is_better", aggregation_unit="group_recompute", aggregation_fn="fisher_global_anchored_recompute"),
        True,
        True,
    ),
    "embedding_centroid_margin_retention": SemanticGeometryMetricSpec(
        "embedding_centroid_margin_retention",
        _DEF("embedding_centroid_margin_retention", metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        True,
        True,
    ),
    "embedding_centroid_margin_damage": SemanticGeometryMetricSpec(
        "embedding_centroid_margin_damage",
        _DEF("embedding_centroid_margin_damage", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="custom", metric_role="detail"),
        True,
        True,
    ),
    "margin_retention": SemanticGeometryMetricSpec(
        "margin_retention",
        _DEF("margin_retention", metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only", metric_role="detail"),
        True,
        False,
    ),
    "variance_inflation": SemanticGeometryMetricSpec(
        "variance_inflation",
        _DEF("variance_inflation", metric_direction="diagnostic", aggregation_unit="class", aggregation_fn="macro_mean_by_class", metric_role="detail"),
        False,
        False,
    ),
    "variance_inflation_deviation": SemanticGeometryMetricSpec(
        "variance_inflation_deviation",
        _DEF("variance_inflation_deviation", metric_direction="lower_is_better", aggregation_unit="class", aggregation_fn="macro_mean_by_class"),
        True,
        True,
    ),
    "center_cosine_margin_retention": SemanticGeometryMetricSpec(
        "center_cosine_margin_retention",
        _DEF("center_cosine_margin_retention", metric_direction="higher_is_better", aggregation_unit="class", aggregation_fn="nanmean_valid_only"),
        True,
        True,
    ),
    "fisher_damage": SemanticGeometryMetricSpec(
        "fisher_damage",
        _DEF("fisher_damage", metric_direction="lower_is_better", aggregation_unit="group_recompute", aggregation_fn="custom", metric_role="detail"),
        True,
        True,
    ),
    "margin_damage": SemanticGeometryMetricSpec(
        "margin_damage",
        _DEF("margin_damage", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="custom", metric_role="detail"),
        True,
        True,
    ),
}
SEMANTIC_GEOMETRY_METRIC_REGISTRY = build_metric_spec_registry(
    tuple(item.spec for item in SEMANTIC_GEOMETRY_METRIC_SPECS.values())
)

COMPARISON_GROUP_COUNT_COLUMNS = [
    f"{group}_sample_count"
    for group in DEFAULT_ANALYSIS_GROUPS
] + [
    f"{group}_effective_class_count"
    for group in DEFAULT_ANALYSIS_GROUPS
]
COMPARISON_MARGIN_DIAGNOSTIC_FIELDS = (
    "margin_valid_count",
    "margin_total_count",
    "margin_valid_rate",
    "margin_invalid_count",
    "margin_nonpositive_count",
    "margin_near_threshold_count",
    "margin_nan_count",
    "margin_inf_count",
    "margin_invalid_reason",
)
COMPARISON_MARGIN_DIAGNOSTIC_COLUMNS = [
    f"{group}_{field}"
    for group in DEFAULT_ANALYSIS_GROUPS
    for field in COMPARISON_MARGIN_DIAGNOSTIC_FIELDS
]

SUMMARY_V2_LONG_COLUMNS = [
    "scope", "target", "layer_idx", "module_name", "target_name", "quant_mode", "granularity",
    "metric_name", "analysis_group", "value", "metric_direction", "aggregation_unit", "aggregation_fn",
    "group_sample_count", "group_class_count", "group_protocol_version", "group_protocol_name",
    "fisher_normalization", "bucket", "shot_group", "source_column", "metric_role",
    "comparable_across_groups", "gap_enabled",
    "margin_valid_count", "margin_total_count", "margin_valid_rate", "margin_invalid_count",
    "margin_nonpositive_count", "margin_near_threshold_count", "margin_nan_count", "margin_inf_count",
    "margin_invalid_reason",
]

COMPARISON_SUMMARY_COLUMNS = [
    "scope", "target", "layer_idx", "module_name", "target_name", "quant_mode", "granularity", "metric_name",
    "many_value", "medium_value", "few_value", "tail_value", "tail_vs_many_gap", "few_vs_many_gap",
    "metric_direction", "aggregation_unit", "aggregation_fn", "metric_role", "source_column",
    "gap_enabled", "comparable_across_groups", "group_protocol_version", "group_protocol_name",
    "fisher_normalization", *COMPARISON_GROUP_COUNT_COLUMNS, *COMPARISON_MARGIN_DIAGNOSTIC_COLUMNS,
]


def _float_tensor(value: Any, name: str) -> Tensor:
    tensor = value.detach().float() if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if int(tensor.numel()) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def _embedding(value: Any, name: str) -> Tensor:
    tensor = _float_tensor(value, name)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be an embedding matrix [N, D], got shape={tuple(tensor.shape)}")
    return tensor


def _labels(value: Any, *, n: int, device: torch.device) -> Tensor:
    tensor = value.detach().long().reshape(-1) if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.long).reshape(-1)
    if int(tensor.numel()) != int(n):
        raise ValueError(f"label_id length must match N, got {int(tensor.numel())} vs {n}")
    if int(tensor.numel()) == 0 or int(tensor.min().item()) < 0:
        raise ValueError("label_id must be non-empty and non-negative")
    return tensor.to(device=device)


def _logits(value: Any | None, *, n: int, name: str, required: bool, device: torch.device) -> Tensor | None:
    if value is None:
        if required:
            raise ValueError(f"{name} is required when pred/correct fields are requested")
        return None
    tensor = _float_tensor(value, name).to(device=device)
    if tensor.ndim != 2 or int(tensor.shape[0]) != int(n):
        raise ValueError(f"{name} must be [N, num_classes], got shape={tuple(tensor.shape)}")
    return tensor


def _classification_logits_margin(logits: Tensor | None, labels: Tensor) -> Tensor | None:
    if logits is None:
        return None
    if logits.ndim != 2 or int(logits.shape[1]) <= 1:
        return None
    labels = labels.to(device=logits.device, dtype=torch.long).reshape(-1)
    if int(labels.numel()) != int(logits.shape[0]):
        return None
    if bool((labels < 0).any().item()) or bool((labels >= int(logits.shape[1])).any().item()):
        return None
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    rival_logits = logits.clone()
    rival_logits.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - rival_logits.max(dim=1).values

def _list(value: Sequence[Any] | Tensor, *, n: int, name: str) -> list[Any]:
    if isinstance(value, Tensor):
        items = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of length N, not a scalar string")
    else:
        items = list(value)
    if len(items) != int(n):
        raise ValueError(f"{name} length must match N, got {len(items)} vs {n}")
    return items


def _sample_ids(value: Sequence[Any] | Tensor, *, n: int, context: str) -> list[str]:
    ids = [str(item) for item in _list(value, n=n, name=f"{context}.sample_id")]
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        duplicated: list[str] = []
        for item in ids:
            if item in seen:
                duplicated.append(item)
            seen.add(item)
        preview = ", ".join(list(dict.fromkeys(duplicated))[:5])
        raise ValueError(f"{context} sample_id values must be unique; duplicates: {preview}")
    return ids


def _check_ids(expected: Sequence[str], current: Sequence[str], context: str) -> None:
    if len(expected) != len(current):
        raise ValueError(f"{context} sample_id length mismatch: {len(expected)} vs {len(current)}")
    bad = [idx for idx, (left, right) in enumerate(zip(expected, current)) if str(left) != str(right)]
    if bad:
        preview = ", ".join(f"idx={idx}: {expected[idx]!r}!={current[idx]!r}" for idx in bad[:5])
        raise ValueError(f"{context} sample_id order mismatch: {preview}")


def _analysis_mask(bucket: Sequence[Any], analysis_group: str, device: torch.device) -> Tensor:
    mask = analysis_group_mask(list(bucket), analysis_group)
    return torch.tensor([bool(item) for item in mask], dtype=torch.bool, device=device)


def _finite_summary(values: Any) -> dict[str, float]:
    x = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    x = x[torch.isfinite(x)]
    if int(x.numel()) == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "median": nan, "p95": nan, "num_valid": 0.0}
    std = x.std(unbiased=False) if int(x.numel()) > 1 else x.new_tensor(0.0)
    return {
        "mean": float(x.mean().item()),
        "std": float(std.item()),
        "median": float(torch.median(x).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
        "num_valid": float(x.numel()),
    }


def _add_summary(row: dict[str, Any], prefix: str, values: Any) -> None:
    summary = _finite_summary(values)
    for key, value in summary.items():
        row[f"{prefix}_{key}"] = value


def _scalar(value: Any) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if int(tensor.numel()) != 1:
        raise ValueError(f"only scalar tensors can be converted to float, got numel={int(tensor.numel())}")
    return float(tensor[0].item())


def _safe_retention(quant_value: Any, fp32_value: Any, eps: float) -> Tensor:
    quant = torch.as_tensor(quant_value, dtype=torch.float32).reshape(-1)
    fp32 = torch.as_tensor(fp32_value, dtype=torch.float32, device=quant.device).reshape(-1)
    valid = torch.isfinite(quant) & torch.isfinite(fp32) & (fp32.abs() > float(eps))
    value = quant / torch.clamp(fp32.abs(), min=float(eps))
    value = torch.where(valid, value, torch.full_like(value, float("nan")))
    return value.reshape(()) if int(value.numel()) == 1 else value


def _safe_inflation(quant_variance: Tensor, fp32_variance: Tensor, eps: float) -> Tensor:
    valid = torch.isfinite(quant_variance) & torch.isfinite(fp32_variance) & (fp32_variance > float(eps))
    value = quant_variance / torch.clamp(fp32_variance, min=float(eps))
    return torch.where(valid, value, torch.full_like(value, float("nan")))


def _variance_inflation_deviation(values: Any) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    deviation = torch.abs(tensor - 1.0)
    return torch.where(torch.isfinite(tensor), deviation, torch.full_like(deviation, float("nan")))


def _finite_or_nan(value: Any) -> float:
    numeric = float(value)
    return numeric if torch.isfinite(torch.tensor(numeric, dtype=torch.float32)).item() else float("nan")


def _one_minus(value: Any) -> float:
    numeric = _finite_or_nan(value)
    return float(1.0 - numeric) if torch.isfinite(torch.tensor(numeric, dtype=torch.float32)).item() else float("nan")


def _fisher_normalization_name(normalize_by_dof: bool) -> str:
    return "dof_normalized_scatter_ratio" if bool(normalize_by_dof) else "raw_scatter_ratio"


def _margin_count_summary(
    fp32_margin: Tensor,
    mask: Tensor,
    *,
    threshold: float,
    valid_mask: Tensor | None = None,
) -> dict[str, Any]:
    values = fp32_margin[mask].reshape(-1)
    total = int(values.numel())
    if total == 0:
        return {
            "margin_valid_count": 0,
            "margin_total_count": 0,
            "margin_valid_rate": float("nan"),
            "margin_invalid_count": 0,
            "margin_nonpositive_count": 0,
            "margin_near_threshold_count": 0,
            "margin_nan_count": 0,
            "margin_inf_count": 0,
            "margin_invalid_reason": "see_count_fields",
        }
    finite = torch.isfinite(values)
    valid = (valid_mask[mask].reshape(-1).bool() if valid_mask is not None else finite & (values > float(threshold)))
    nonpositive = finite & (values <= 0)
    near_threshold = finite & (values > 0) & (values <= float(threshold))
    nan_mask = torch.isnan(values)
    inf_mask = torch.isinf(values)
    valid_count = int(valid.sum().item())
    invalid_count = int(total - valid_count)
    return {
        "margin_valid_count": valid_count,
        "margin_total_count": total,
        "margin_valid_rate": float(valid_count / total),
        "margin_invalid_count": invalid_count,
        "margin_nonpositive_count": int(nonpositive.sum().item()),
        "margin_near_threshold_count": int(near_threshold.sum().item()),
        "margin_nan_count": int(nan_mask.sum().item()),
        "margin_inf_count": int(inf_mask.sum().item()),
        "margin_invalid_reason": "see_count_fields",
    }


def _class_buckets(labels: Tensor, bucket: Sequence[str], class_ids: Tensor) -> list[str]:
    label_list = [int(item) for item in labels.detach().cpu().tolist()]
    result: list[str] = []
    for class_id in [int(item) for item in class_ids.detach().cpu().tolist()]:
        values = [bucket[idx] for idx, label in enumerate(label_list) if label == class_id]
        unique = list(dict.fromkeys(values))
        if len(unique) > 1:
            raise ValueError(f"label_id={class_id} appears in multiple buckets: {unique}")
        if not unique:
            raise ValueError(f"label_id={class_id} has no bucket")
        result.append(unique[0])
    return result


def _class_mean(values: Tensor, labels: Tensor, class_ids: Tensor) -> Tensor:
    output: list[Tensor] = []
    for class_id in class_ids:
        part = values[labels == class_id]
        output.append(torch.as_tensor(_finite_summary(part)["mean"], dtype=values.dtype, device=values.device))
    return torch.stack(output, dim=0)

def build_fp32_semantic_geometry_reference(
    *,
    sample_id: Sequence[Any] | Tensor,
    label_id: Sequence[Any] | Tensor,
    bucket: Sequence[Any] | Tensor,
    shot_group: Sequence[Any] | Tensor | None = None,
    fp32_embeddings: Any,
    fp32_logits: Any | None = None,
    group_names: Sequence[str] = DEFAULT_SHOT_GROUPS,
    variance_normalization: VarianceNormalization = DEFAULT_VARIANCE_NORMALIZATION,
    normalize_fisher_by_dof: bool = DEFAULT_FISHER_NORMALIZE_BY_DOF,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Build the FP32 semantic-geometry reference."""


    fp32 = _embedding(fp32_embeddings, "fp32_embeddings")
    n = int(fp32.shape[0])
    ids = _sample_ids(sample_id, n=n, context="fp32_reference")
    labels = _labels(label_id, n=n, device=fp32.device)
    bucket_result = read_canonical_bucket(
        bucket=_list(bucket, n=n, name="bucket"),
        shot_group=None if shot_group is None else _list(shot_group, n=n, name="shot_group"),
        context="fp32_semantic_geometry_reference",
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in bucket_result.bucket]
    shot_groups = list(buckets)
    sample_tail_flags = [bool(item) for item in tail_included(buckets)]
    logits = _logits(fp32_logits, n=n, name="fp32_logits", required=False, device=fp32.device)
    fp32_logits_margin = _classification_logits_margin(logits, labels)

    fp32_variance, variance_details = class_trace_variance(
        fp32,
        labels,
        normalization=variance_normalization,
        reduction="none",
        return_details=True,
    )
    fp32_center_cosine_margin, cosine_details = class_center_cosine_margin(
        fp32,
        labels,
        reduction="none",
        return_details=True,
        eps=eps,
    )
    fp32_sample_margin = compute_group_margin(fp32, labels, None, reduction="none")
    assert isinstance(fp32_variance, Tensor)
    assert isinstance(fp32_center_cosine_margin, Tensor)
    assert isinstance(fp32_sample_margin, Tensor)

    class_ids = variance_details["class_ids"]
    counts = variance_details["counts"]
    if not torch.equal(class_ids, cosine_details["class_ids"]):
        raise RuntimeError("FP32 variance and center cosine margin class_id order differ")
    fp32_class_margin = _class_mean(fp32_sample_margin, labels, class_ids)
    class_buckets = _class_buckets(labels, buckets, class_ids)

    resolved_groups = [str(item) for item in group_names]
    group_fisher: dict[str, Tensor] = {}
    for group_name in resolved_groups:
        group_fisher[group_name] = compute_group_fisher(
            fp32,
            labels,
            _analysis_mask(buckets, group_name, fp32.device),
            normalize_by_dof=normalize_fisher_by_dof,
            eps=eps,
        )
    global_fisher = compute_group_fisher(
        fp32,
        labels,
        None,
        normalize_by_dof=normalize_fisher_by_dof,
        eps=eps,
    )

    return {
        "cache_type": REFERENCE_CACHE_TYPE,
        "sample_id": ids,
        "label_id": labels.detach().cpu(),
        "label": [int(item) for item in labels.detach().cpu().tolist()],
        "bucket": buckets,
        "shot_group": shot_groups,
        "tail_included": sample_tail_flags,
        "fp32_embeddings": fp32.detach().cpu(),
        "fp32_logits": None if logits is None else logits.detach().cpu(),
        "fp32_logits_margin": None if fp32_logits_margin is None else fp32_logits_margin.detach().cpu(),
        "class_ids": class_ids.detach().cpu(),
        "class_label": [int(item) for item in class_ids.detach().cpu().tolist()],
        "class_bucket": class_buckets,
        "class_shot_group": list(class_buckets),
        "class_tail_included": [bool(item) for item in tail_included(class_buckets)],
        "class_num_samples": counts.detach().cpu(),
        "fp32_sample_margin": fp32_sample_margin.detach().cpu(),
        "fp32_class_margin": fp32_class_margin.detach().cpu(),
        "fp32_class_variance": fp32_variance.detach().cpu(),
        "fp32_class_center_cosine_margin": fp32_center_cosine_margin.detach().cpu(),
        "fp32_global_fisher": global_fisher.detach().cpu(),
        "fp32_group_fisher": {key: value.detach().cpu() for key, value in group_fisher.items()},
        "group_names": resolved_groups,
        "variance_normalization": str(variance_normalization),
        "normalize_fisher_by_dof": bool(normalize_fisher_by_dof),
        "margin_definition": "sample-level margin = nearest non-label class center L2 distance - own class center L2 distance",
        "center_cosine_margin_definition": "class-level angular margin = 1 - max_other_center_cosine",
        "fisher_definition": "global_anchored_group_fisher_between_class_center_scatter_over_within_class_trace_scatter",
    }


def _load_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    if reference.get("cache_type") != REFERENCE_CACHE_TYPE:
        raise ValueError(f"FP32 reference cache_type must be {REFERENCE_CACHE_TYPE!r}")
    fp32 = _embedding(reference["fp32_embeddings"], "reference.fp32_embeddings")
    n = int(fp32.shape[0])
    labels = _labels(reference["label_id"], n=n, device=fp32.device)
    ids = [str(item) for item in list(reference["sample_id"])]
    if len(ids) != n:
        raise ValueError("reference.sample_id length must match embedding count")
    bucket_result = read_canonical_bucket(
        bucket=_list(reference["bucket"], n=n, name="reference.bucket"),
        shot_group=_list(reference["shot_group"], n=n, name="reference.shot_group") if "shot_group" in reference else None,
        context="semantic_geometry_reference",
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in bucket_result.bucket]
    shot_groups = list(buckets)
    sample_tail_flags = [bool(item) for item in tail_included(buckets)]
    logits = _logits(reference.get("fp32_logits"), n=n, name="reference.fp32_logits", required=False, device=fp32.device)
    fp32_logits_margin = _classification_logits_margin(logits, labels)
    class_ids = torch.as_tensor(reference["class_ids"], dtype=torch.long, device=fp32.device).reshape(-1)
    return {
        "fp32_embeddings": fp32,
        "label_id": labels,
        "sample_id": ids,
        "label": list(reference.get("label", [int(item) for item in labels.detach().cpu().tolist()])),
        "bucket": buckets,
        "shot_group": shot_groups,
        "tail_included": sample_tail_flags,
        "fp32_logits": logits,
        "fp32_logits_margin": fp32_logits_margin,
        "class_ids": class_ids,
        "class_label": list(reference.get("class_label", reference["class_ids"])),
        "class_bucket": [str(item) for item in list(reference["class_bucket"])],
        "class_shot_group": [str(item) for item in list(reference["class_shot_group"])],
        "class_tail_included": [bool(item) for item in list(reference.get("class_tail_included", tail_included(reference["class_bucket"])))],
        "class_num_samples": torch.as_tensor(reference["class_num_samples"], dtype=torch.long, device=fp32.device).reshape(-1),
        "fp32_sample_margin": torch.as_tensor(reference["fp32_sample_margin"], dtype=torch.float32, device=fp32.device).reshape(-1),
        "fp32_class_margin": torch.as_tensor(reference["fp32_class_margin"], dtype=torch.float32, device=fp32.device).reshape(-1),
        "fp32_class_variance": torch.as_tensor(reference["fp32_class_variance"], dtype=torch.float32, device=fp32.device).reshape(-1),
        "fp32_class_center_cosine_margin": torch.as_tensor(
            reference["fp32_class_center_cosine_margin"],
            dtype=torch.float32,
            device=fp32.device,
        ).reshape(-1),
        "fp32_global_fisher": torch.as_tensor(reference["fp32_global_fisher"], dtype=torch.float32, device=fp32.device),
        "fp32_group_fisher": {
            str(key): torch.as_tensor(value, dtype=torch.float32, device=fp32.device)
            for key, value in dict(reference["fp32_group_fisher"]).items()
        },
        "group_names": [str(item) for item in reference.get("group_names", DEFAULT_SHOT_GROUPS)],
        "variance_normalization": str(reference.get("variance_normalization", DEFAULT_VARIANCE_NORMALIZATION)),
        "normalize_fisher_by_dof": bool(reference.get("normalize_fisher_by_dof", DEFAULT_FISHER_NORMALIZE_BY_DOF)),
    }

def _sample_rows(
    ref: Mapping[str, Any],
    quant: Tensor,
    qlogits: Tensor | None,
    mode: str,
    granularity: str,
    require_logits: bool,
    eps: float,
    sample_margin_metrics: Mapping[str, Tensor] | None = None,
) -> tuple[list[dict[str, Any]], Tensor]:
    fp32 = ref["fp32_embeddings"]
    labels = ref["label_id"]
    flogits = ref["fp32_logits"]
    if require_logits and (flogits is None or qlogits is None):
        raise ValueError("fp32_logits and quant_logits are both required for auxiliary pred/correct fields")
    distortion = common_embedding_distortion(fp32, quant, dim=-1, reduction="none", eps=eps)
    assert isinstance(distortion, Tensor)
    fpred = None if flogits is None else flogits.argmax(dim=1)
    qpred = None if qlogits is None else qlogits.argmax(dim=1)
    fp32_logits_margin = ref.get("fp32_logits_margin")
    if fp32_logits_margin is not None:
        fp32_logits_margin = torch.as_tensor(fp32_logits_margin, dtype=torch.float32, device=fp32.device).reshape(-1)
    qlogits_margin = _classification_logits_margin(qlogits, labels)
    logits_margin_available = fp32_logits_margin is not None and qlogits_margin is not None
    logits_margin_retention = None
    logits_margin_drop = None
    if logits_margin_available:
        logits_margin_retention = _safe_retention(qlogits_margin, fp32_logits_margin, eps)
        logits_margin_drop = metric_drop(fp32_logits_margin, qlogits_margin, reduction="none")
    rows: list[dict[str, Any]] = []
    fp32_margin = None if sample_margin_metrics is None else sample_margin_metrics.get("fp32_margin")
    quant_margin = None if sample_margin_metrics is None else sample_margin_metrics.get("quant_margin")
    margin_retention = None if sample_margin_metrics is None else sample_margin_metrics.get("margin_retention")
    margin_drop = None if sample_margin_metrics is None else sample_margin_metrics.get("margin_drop")
    margin_definition = "embedding_centroid_margin = nearest non-label class center L2 distance - own class center L2 distance"
    for idx, sid in enumerate(ref["sample_id"]):
        label_id = int(labels[idx].item())
        fp32_margin_value = None if fp32_margin is None else float(fp32_margin[idx].item())
        quant_margin_value = None if quant_margin is None else float(quant_margin[idx].item())
        margin_retention_value = None if margin_retention is None else float(margin_retention[idx].item())
        margin_drop_value = None if margin_drop is None else float(margin_drop[idx].item())
        rows.append({
            "sample_id": str(sid),
            "label": ref["label"][idx],
            "label_id": label_id,
            "bucket": ref["bucket"][idx],
            "shot_group": ref["shot_group"][idx],
            "tail_included": bool(ref["tail_included"][idx]),
            "quant_mode": str(mode),
            "granularity": str(granularity),
            "embedding_distortion": float(distortion[idx].item()),
            "margin_definition": margin_definition,
            "fp32_embedding_centroid_margin": fp32_margin_value,
            "quant_embedding_centroid_margin": quant_margin_value,
            "embedding_centroid_margin_retention": margin_retention_value,
            "embedding_centroid_margin_drop": margin_drop_value,
            "fp32_margin": fp32_margin_value,
            "quant_margin": quant_margin_value,
            "margin_retention": margin_retention_value,
            "margin_drop": margin_drop_value,
            "fp32_logits_margin": None if fp32_logits_margin is None else float(fp32_logits_margin[idx].item()),
            "quant_logits_margin": None if qlogits_margin is None else float(qlogits_margin[idx].item()),
            "logits_margin_retention": None if logits_margin_retention is None else float(logits_margin_retention[idx].item()),
            "logits_margin_drop": None if logits_margin_drop is None else float(logits_margin_drop[idx].item()),
            "logits_margin_available": bool(logits_margin_available),
            "fp32_pred": None if fpred is None else int(fpred[idx].item()),
            "quant_pred": None if qpred is None else int(qpred[idx].item()),
            "correct_fp32": None if fpred is None else bool(int(fpred[idx].item()) == label_id),
            "correct_quant": None if qpred is None else bool(int(qpred[idx].item()) == label_id),
        })
    return rows, distortion.detach()

def _quant_geometry(
    ref: Mapping[str, Any],
    quant: Tensor,
    variance_normalization: VarianceNormalization,
    eps: float,
) -> dict[str, Tensor]:
    labels = ref["label_id"]
    class_ids = ref["class_ids"]
    qvar, qvar_details = class_trace_variance(
        quant,
        labels,
        normalization=variance_normalization,
        reduction="none",
        return_details=True,
    )
    qcenter_cosine, qcosine_details = class_center_cosine_margin(
        quant,
        labels,
        reduction="none",
        return_details=True,
        eps=eps,
    )
    qsample_margin = compute_group_margin(quant, labels, None, reduction="none")
    assert isinstance(qvar, Tensor)
    assert isinstance(qcenter_cosine, Tensor)
    assert isinstance(qsample_margin, Tensor)
    if not torch.equal(class_ids, qvar_details["class_ids"]):
        raise ValueError("quant variance class_id order differs from FP32 reference")
    if not torch.equal(class_ids, qcosine_details["class_ids"]):
        raise ValueError("quant center cosine margin class_id order differs from FP32 reference")
    qclass_margin = _class_mean(qsample_margin, labels, class_ids)
    return {
        "quant_class_variance": qvar.detach(),
        "quant_sample_margin": qsample_margin.detach(),
        "quant_class_margin": qclass_margin.detach(),
        "quant_class_center_cosine_margin": qcenter_cosine.detach(),
    }


def _class_rows(
    ref: Mapping[str, Any],
    quant_metrics: Mapping[str, Tensor],
    mode: str,
    granularity: str,
    eps: float,
) -> tuple[list[dict[str, Any]], dict[str, Tensor]]:
    labels = ref["label_id"]
    class_ids = ref["class_ids"]
    fp32_var = ref["fp32_class_variance"]
    qvar = quant_metrics["quant_class_variance"]
    vinflation = _safe_inflation(qvar, fp32_var, eps)

    fp32_sample_margin = ref["fp32_sample_margin"]
    quant_sample_margin = quant_metrics["quant_sample_margin"]
    margin_threshold = float(DEFAULT_MARGIN_MIN_ABS)
    sample_margin_retention, sample_margin_diag = margin_value_retention(
        fp32_sample_margin,
        quant_sample_margin,
        reduction="none",
        min_abs_margin=margin_threshold,
        return_diagnostics=True,
        eps=eps,
    )
    assert isinstance(sample_margin_retention, Tensor)
    finite_valid_margin = sample_margin_diag["valid_margin_mask"].to(device=fp32_sample_margin.device)
    sample_margin_retention = torch.where(
        finite_valid_margin,
        sample_margin_retention,
        torch.full_like(sample_margin_retention, float("nan")),
    )
    sample_margin_drop = metric_drop(fp32_sample_margin, quant_sample_margin, reduction="none")

    fp32_cosine = ref["fp32_class_center_cosine_margin"]
    quant_cosine = quant_metrics["quant_class_center_cosine_margin"]
    cosine_retention = _safe_retention(quant_cosine, fp32_cosine, eps)
    cosine_drop = metric_drop(fp32_cosine, quant_cosine, reduction="none")

    rows: list[dict[str, Any]] = []
    class_margin_retention: list[Tensor] = []
    class_margin_drop: list[Tensor] = []
    for idx, cid_tensor in enumerate(class_ids):
        mask = labels == cid_tensor
        fp32_margin_mean = _finite_summary(fp32_sample_margin[mask])["mean"]
        quant_margin_mean = _finite_summary(quant_sample_margin[mask])["mean"]
        retention_mean = _finite_summary(sample_margin_retention[mask])["mean"]
        drop_mean = _finite_summary(sample_margin_drop[mask])["mean"]
        class_margin_retention.append(torch.as_tensor(retention_mean, dtype=torch.float32, device=labels.device))
        class_margin_drop.append(torch.as_tensor(drop_mean, dtype=torch.float32, device=labels.device))
        rows.append({
            "label": ref["class_label"][idx],
            "label_id": int(cid_tensor.detach().cpu().item()),
            "bucket": ref["class_bucket"][idx],
            "shot_group": ref["class_shot_group"][idx],
            "tail_included": bool(ref["class_tail_included"][idx]),
            "num_samples": int(ref["class_num_samples"][idx].item()),
            "quant_mode": str(mode),
            "granularity": str(granularity),
            "fp32_variance": float(fp32_var[idx].item()),
            "quant_variance": float(qvar[idx].item()),
            "variance_inflation": float(vinflation[idx].item()),
            "variance_inflation_deviation": float(_variance_inflation_deviation(vinflation[idx]).item()),
            "fp32_embedding_centroid_margin": float(fp32_margin_mean),
            "quant_embedding_centroid_margin": float(quant_margin_mean),
            "embedding_centroid_margin_retention": float(retention_mean),
            "embedding_centroid_margin_drop": float(drop_mean),
            "fp32_margin": float(fp32_margin_mean),
            "quant_margin": float(quant_margin_mean),
            "margin_retention": float(retention_mean),
            "margin_drop": float(drop_mean),
            "margin_valid_ratio": _finite_summary(sample_margin_diag["valid_margin_mask"][mask].float())["mean"],
            "fp32_center_cosine_margin": float(fp32_cosine[idx].item()),
            "quant_center_cosine_margin": float(quant_cosine[idx].item()),
            "center_cosine_margin_retention": float(cosine_retention[idx].item()),
            "center_cosine_margin_drop": float(cosine_drop[idx].item()),
        })
    return rows, {
        "variance_inflation": vinflation.detach(),
        "variance_inflation_deviation": _variance_inflation_deviation(vinflation).detach(),
        "sample_margin_retention": sample_margin_retention.detach(),
        "sample_margin_drop": sample_margin_drop.detach(),
        "class_margin_retention": torch.stack(class_margin_retention),
        "class_margin_drop": torch.stack(class_margin_drop),
        "center_cosine_margin_retention": cosine_retention.detach(),
        "center_cosine_margin_drop": cosine_drop.detach(),
        "fp32_sample_margin": fp32_sample_margin.detach(),
        "margin_valid_mask": finite_valid_margin.detach(),
        "margin_invalid_mask": sample_margin_diag["invalid_margin_mask"].detach(),
        "margin_nonpositive_mask": sample_margin_diag["nonpositive_margin_mask"].detach(),
        "margin_near_threshold_mask": sample_margin_diag["near_threshold_margin_mask"].detach(),
        "margin_threshold": torch.as_tensor(margin_threshold, dtype=torch.float32),
    }


def _group_rows(
    ref: Mapping[str, Any],
    quant: Tensor,
    mode: str,
    granularity: str,
    distortion: Tensor,
    class_metrics: Mapping[str, Tensor],
    group_names: Sequence[str],
    eps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = ref["label_id"]
    class_buckets = ref["class_bucket"]
    fisher_normalization = _fisher_normalization_name(bool(ref["normalize_fisher_by_dof"]))
    for group_name in [str(item) for item in group_names]:
        sample_mask = _analysis_mask(ref["bucket"], group_name, quant.device)
        class_mask = _analysis_mask(class_buckets, group_name, quant.device)
        fp32_fisher = ref["fp32_group_fisher"].get(group_name, torch.tensor(float("nan"), device=quant.device))
        if bool(sample_mask.any().item()):
            quant_fisher = compute_group_fisher(
                quant,
                labels,
                sample_mask,
                normalize_by_dof=bool(ref["normalize_fisher_by_dof"]),
                eps=eps,
            )
        else:
            quant_fisher = torch.tensor(float("nan"), device=quant.device)
        fisher_retention = _safe_retention(quant_fisher, fp32_fisher, eps)
        fisher_retention_value = _scalar(fisher_retention)
        margin_values = class_metrics["sample_margin_retention"][sample_mask]
        variance_values = class_metrics["variance_inflation"][class_mask]
        variance_deviation_values = class_metrics["variance_inflation_deviation"][class_mask]
        center_cosine_values = class_metrics["center_cosine_margin_retention"][class_mask]
        distortion_values = distortion[sample_mask]
        margin_mean = _finite_summary(margin_values)["mean"]
        margin_counts = _margin_count_summary(
            class_metrics["fp32_sample_margin"],
            sample_mask,
            threshold=float(class_metrics["margin_threshold"].item()),
            valid_mask=class_metrics["margin_valid_mask"],
        )
        row: dict[str, Any] = {
            "bucket": "",
            "analysis_group": group_name,
            "shot_group": "",
            "quant_mode": str(mode),
            "granularity": str(granularity),
            "num_samples": int(sample_mask.sum().item()),
            "num_classes": int(class_mask.sum().item()),
            "fisher_fp32": _scalar(fp32_fisher),
            "fisher_quant": _scalar(quant_fisher),
            "fisher_retention": fisher_retention_value,
            "fisher_damage": _one_minus(fisher_retention_value),
            "embedding_centroid_margin_retention": margin_mean,
            "embedding_centroid_margin_damage": _one_minus(margin_mean),
            "margin_retention": margin_mean,
            "margin_damage": _one_minus(margin_mean),
            "center_cosine_margin_retention": _finite_summary(center_cosine_values)["mean"],
            "variance_inflation": _finite_summary(variance_values)["mean"],
            "variance_inflation_deviation": _finite_summary(variance_deviation_values)["mean"],
            "embedding_distortion_mean": _finite_summary(distortion_values)["mean"],
            "group_protocol_version": GROUP_PROTOCOL_VERSION,
            "group_protocol_name": GROUP_PROTOCOL_NAME,
            "fisher_normalization": fisher_normalization,
            **margin_counts,
        }
        _add_summary(row, "margin_retention", margin_values)
        _add_summary(row, "margin_drop", class_metrics["sample_margin_drop"][sample_mask])
        _add_summary(row, "center_cosine_margin_retention", center_cosine_values)
        _add_summary(row, "variance_inflation", variance_values)
        _add_summary(row, "variance_inflation_deviation", variance_deviation_values)
        _add_summary(row, "embedding_distortion", distortion_values)
        rows.append(row)
    return rows

def _group_value(rows: Sequence[Mapping[str, Any]], group_name: str, metric: str) -> float:
    for row in rows:
        if str(row.get("analysis_group", row.get("shot_group"))) == str(group_name):
            return float(row.get(metric, float("nan")))
    return float("nan")


def _global_rows(
    ref: Mapping[str, Any],
    quant: Tensor,
    mode: str,
    granularity: str,
    distortion: Tensor,
    groups: Sequence[Mapping[str, Any]],
    few_group: str,
    many_group: str,
    eps: float,
) -> list[dict[str, Any]]:
    fisher_quant = compute_group_fisher(
        quant,
        ref["label_id"],
        None,
        normalize_by_dof=bool(ref["normalize_fisher_by_dof"]),
        eps=eps,
    )
    fisher_retention = _safe_retention(fisher_quant, ref["fp32_global_fisher"], eps)
    margin_by_group = {str(group): _group_value(groups, str(group), "embedding_centroid_margin_retention") for group in DEFAULT_ANALYSIS_GROUPS}
    variance_by_group = {str(group): _group_value(groups, str(group), "variance_inflation") for group in DEFAULT_ANALYSIS_GROUPS}
    variance_deviation_by_group = {
        str(group): _group_value(groups, str(group), "variance_inflation_deviation")
        for group in DEFAULT_ANALYSIS_GROUPS
    }
    tail_gap = compute_group_gap(margin_by_group, comparison_group="tail", baseline_group=many_group, metric_direction="higher_is_better")
    few_gap = compute_group_gap(margin_by_group, comparison_group=few_group, baseline_group=many_group, metric_direction="higher_is_better")
    tail_var_gap = compute_group_gap(
        variance_deviation_by_group,
        comparison_group="tail",
        baseline_group=many_group,
        metric_direction="lower_is_better",
    )
    few_var_gap = compute_group_gap(
        variance_deviation_by_group,
        comparison_group=few_group,
        baseline_group=many_group,
        metric_direction="lower_is_better",
    )
    many_var = variance_by_group.get(many_group, float("nan"))
    tail_var = variance_by_group.get("tail", float("nan"))
    few_var = variance_by_group.get(few_group, float("nan"))
    many_var_deviation = variance_deviation_by_group.get(many_group, float("nan"))
    tail_var_deviation = variance_deviation_by_group.get("tail", float("nan"))
    few_var_deviation = variance_deviation_by_group.get(few_group, float("nan"))
    row = {
        "quant_mode": str(mode),
        "granularity": str(granularity),
        "num_samples": int(quant.shape[0]),
        "num_classes": int(ref["class_ids"].numel()),
        "fisher_fp32": _scalar(ref["fp32_global_fisher"]),
        "fisher_quant": _scalar(fisher_quant),
        "global_fisher_retention": _scalar(fisher_retention),
        "many_embedding_centroid_margin_retention": margin_by_group.get(many_group, float("nan")),
        "tail_embedding_centroid_margin_retention": margin_by_group.get("tail", float("nan")),
        "few_embedding_centroid_margin_retention": margin_by_group.get(few_group, float("nan")),
        "many_margin_retention": margin_by_group.get(many_group, float("nan")),
        "tail_margin_retention": margin_by_group.get("tail", float("nan")),
        "few_margin_retention": margin_by_group.get(few_group, float("nan")),
        "tail_collapse_gap": float(tail_gap),
        "few_collapse_gap": float(few_gap),
        "collapse_gap": float(tail_gap),
        "many_variance_inflation": many_var,
        "tail_variance_inflation": tail_var,
        "few_variance_inflation": few_var,
        "many_variance_inflation_deviation": many_var_deviation,
        "tail_variance_inflation_deviation": tail_var_deviation,
        "few_variance_inflation_deviation": few_var_deviation,
        "tail_vs_many_variance_inflation_deviation_gap": float(tail_var_gap),
        "few_vs_many_variance_inflation_deviation_gap": float(few_var_gap),
        "embedding_distortion_mean": _finite_summary(distortion)["mean"],
        "fisher_normalization": _fisher_normalization_name(bool(ref["normalize_fisher_by_dof"])),
    }
    _add_summary(row, "embedding_distortion", distortion)
    return [row]


SUMMARY_FACET_COLUMNS = ["scope", "target", "layer_idx", "module_name", "target_name", "quant_mode", "granularity"]


def _rows_from_tables(data: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]], level: str) -> list[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        return [dict(row) for row in data.get(level, [])]
    return [dict(row) for row in data]


def _target_from_row(row: Mapping[str, Any]) -> str:
    value = row.get("target", row.get("target_name"))
    if value is not None and str(value).strip().lower() not in {"", "nan", "none", "<na>"}:
        return str(value)
    layer = row.get("layer_idx", "")
    module = row.get("module_name", "")
    if str(layer).strip() and str(module).strip():
        return f"encoder.layers.{layer}.{module}"
    return "unknown"


def _facet_from_group_row(row: Mapping[str, Any]) -> dict[str, Any]:
    target = _target_from_row(row)
    return {
        "scope": row.get("scope", ""),
        "target": target,
        "layer_idx": row.get("layer_idx", ""),
        "module_name": row.get("module_name", ""),
        "target_name": row.get("target_name", target),
        "quant_mode": row.get("quant_mode", ""),
        "granularity": row.get("granularity", ""),
    }


def _metric_row_base(group_row: Mapping[str, Any], metric_name: str) -> dict[str, Any]:
    item = SEMANTIC_GEOMETRY_METRIC_SPECS[metric_name]
    meta = metric_spec_metadata(item.spec)
    return {
        **_facet_from_group_row(group_row),
        **meta,
        "source_column": item.source_column,
        "comparable_across_groups": bool(item.comparable_across_groups),
        "gap_enabled": bool(item.gap_enabled),
        "group_protocol_version": GROUP_PROTOCOL_VERSION,
        "group_protocol_name": GROUP_PROTOCOL_NAME,
        "fisher_normalization": group_row.get("fisher_normalization", ""),
    }


def _metric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _count_value(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _comparison_diagnostic_fields(by_group: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for group in DEFAULT_ANALYSIS_GROUPS:
        row = by_group.get(group, {})
        fields[f"{group}_sample_count"] = _count_value(row.get("group_sample_count", 0))
        fields[f"{group}_effective_class_count"] = _count_value(row.get("group_class_count", 0))
        for name in COMPARISON_MARGIN_DIAGNOSTIC_FIELDS:
            fields[f"{group}_{name}"] = row.get(name, "")
    return fields


def build_semantic_geometry_summary_v2_long(
    tables_or_group_rows: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build protocol-v2 semantic geometry group summary rows.

    One row is target x quant_mode x metric_name x analysis_group. The input is
    group-level detail produced by this module; no cross-target averaging is done.
    """

    rows: list[dict[str, Any]] = []
    for group_row in _rows_from_tables(tables_or_group_rows, "group_level"):
        analysis_group = str(group_row.get("analysis_group", ""))
        if analysis_group not in DEFAULT_ANALYSIS_GROUPS:
            continue
        for metric_name, item in SEMANTIC_GEOMETRY_METRIC_SPECS.items():
            if item.source_column not in group_row:
                continue
            base = _metric_row_base(group_row, metric_name)
            rows.append({
                **base,
                "analysis_group": analysis_group,
                "value": _metric_value(group_row.get(item.source_column)),
                "group_sample_count": int(group_row.get("num_samples", 0) or 0),
                "group_class_count": int(group_row.get("num_classes", 0) or 0),
                "bucket": "",
                "shot_group": "",
                "margin_valid_count": group_row.get("margin_valid_count", ""),
                "margin_total_count": group_row.get("margin_total_count", ""),
                "margin_valid_rate": group_row.get("margin_valid_rate", ""),
                "margin_invalid_count": group_row.get("margin_invalid_count", ""),
                "margin_nonpositive_count": group_row.get("margin_nonpositive_count", ""),
                "margin_near_threshold_count": group_row.get("margin_near_threshold_count", ""),
                "margin_nan_count": group_row.get("margin_nan_count", ""),
                "margin_inf_count": group_row.get("margin_inf_count", ""),
                "margin_invalid_reason": group_row.get("margin_invalid_reason", "see_count_fields"),
            })
    return rows


def build_semantic_geometry_comparison_summary(
    summary_v2_long_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one row per target x quant_mode x metric_name comparison summary."""

    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    id_columns = tuple(SUMMARY_FACET_COLUMNS + ["metric_name"])
    for row in summary_v2_long_rows:
        group = str(row.get("analysis_group", ""))
        if group not in DEFAULT_ANALYSIS_GROUPS:
            continue
        key = tuple(row.get(column, "") for column in id_columns)
        if group in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate semantic geometry v2 row for key={key!r}, analysis_group={group!r}")
        grouped[key][group] = row

    rows: list[dict[str, Any]] = []
    for key, by_group in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        metric_name = str(key[-1])
        if metric_name not in SEMANTIC_GEOMETRY_METRIC_SPECS:
            continue
        item = SEMANTIC_GEOMETRY_METRIC_SPECS[metric_name]
        values = {group: _metric_value(by_group.get(group, {}).get("value", float("nan"))) for group in DEFAULT_ANALYSIS_GROUPS}
        if bool(item.gap_enabled) and bool(item.comparable_across_groups):
            gaps = compute_tail_and_few_gaps(values, metric_direction=item.spec.metric_direction)
        else:
            gaps = {"tail_vs_many_gap": float("nan"), "few_vs_many_gap": float("nan")}
        exemplar = next(iter(by_group.values()))
        meta = metric_spec_metadata(item.spec)
        diagnostic_fields = _comparison_diagnostic_fields(by_group)
        rows.append({
            **dict(zip(id_columns, key)),
            **meta,
            "source_column": item.source_column,
            "many_value": values.get("many", float("nan")),
            "medium_value": values.get("medium", float("nan")),
            "few_value": values.get("few", float("nan")),
            "tail_value": values.get("tail", float("nan")),
            "tail_vs_many_gap": gaps["tail_vs_many_gap"],
            "few_vs_many_gap": gaps["few_vs_many_gap"],
            "gap_enabled": bool(item.gap_enabled),
            "comparable_across_groups": bool(item.comparable_across_groups),
            "group_protocol_version": GROUP_PROTOCOL_VERSION,
            "group_protocol_name": GROUP_PROTOCOL_NAME,
            "fisher_normalization": exemplar.get("fisher_normalization", ""),
            **diagnostic_fields,
        })
    return rows


def compute_semantic_geometry_tables(
    *,
    fp32_reference: Mapping[str, Any],
    quant_embeddings: Any,
    quant_logits: Any | None,
    sample_id: Sequence[Any] | Tensor,
    quant_mode: str,
    granularity: str,
    label_id: Sequence[Any] | Tensor | None = None,
    bucket: Sequence[Any] | Tensor | None = None,
    shot_group: Sequence[Any] | Tensor | None = None,
    group_names: Sequence[str] | None = None,
    few_group: str = "few",
    many_group: str = "many",
    require_logits: bool = True,
    eps: float = DEFAULT_EPS,
) -> dict[str, list[dict[str, Any]]]:
    """Compute sample-, class-, group-, and global-level semantic geometry metrics."""


    ref = _load_reference(fp32_reference)
    fp32 = ref["fp32_embeddings"]
    n = int(fp32.shape[0])
    _check_ids(ref["sample_id"], _sample_ids(sample_id, n=n, context="semantic_geometry"), "semantic_geometry")
    if label_id is not None:
        if not torch.equal(ref["label_id"], _labels(label_id, n=n, device=fp32.device)):
            raise ValueError("current label_id differs from FP32 reference")
    if bucket is not None or shot_group is not None:
        bucket_result = read_canonical_bucket(
            bucket=None if bucket is None else _list(bucket, n=n, name="bucket"),
            shot_group=None if shot_group is None else _list(shot_group, n=n, name="shot_group"),
            context="semantic_geometry current cache",
            allow_shot_group_fallback=False,
        )
        current_bucket = [str(item) for item in bucket_result.bucket]
        if current_bucket != ref["bucket"]:
            raise ValueError("current bucket differs from FP32 reference")

    quant = _embedding(quant_embeddings, "quant_embeddings").to(device=fp32.device)
    if tuple(quant.shape) != tuple(fp32.shape):
        raise ValueError(f"quant_embeddings shape must match FP32 embeddings, got {tuple(quant.shape)} vs {tuple(fp32.shape)}")
    qlogits = _logits(quant_logits, n=n, name="quant_logits", required=require_logits, device=fp32.device)
    if ref["fp32_logits"] is not None and qlogits is not None and int(ref["fp32_logits"].shape[1]) != int(qlogits.shape[1]):
        raise ValueError("quant_logits final dimension must match FP32 logits")

    quant_geometry = _quant_geometry(
        ref,
        quant,
        ref["variance_normalization"],  # type: ignore[arg-type]
        eps,
    )
    class_rows, class_metrics = _class_rows(ref, quant_geometry, quant_mode, granularity, eps)
    sample_rows, distortion = _sample_rows(
        ref,
        quant,
        qlogits,
        quant_mode,
        granularity,
        require_logits,
        eps,
        sample_margin_metrics={
            "fp32_margin": class_metrics["fp32_sample_margin"],
            "quant_margin": quant_geometry["quant_sample_margin"],
            "margin_retention": class_metrics["sample_margin_retention"],
            "margin_drop": class_metrics["sample_margin_drop"],
        },
    )
    resolved_groups = [str(item) for item in (group_names if group_names is not None else ref["group_names"])]
    group_rows = _group_rows(ref, quant, quant_mode, granularity, distortion, class_metrics, resolved_groups, eps)
    global_rows = _global_rows(ref, quant, quant_mode, granularity, distortion, group_rows, str(few_group), str(many_group), eps)
    return {
        "sample_level": sample_rows,
        "class_level": class_rows,
        "group_level": group_rows,
        "global_level": global_rows,
    }


def compute_semantic_geometry_metrics(**kwargs: Any) -> dict[str, list[dict[str, Any]]]:
    """Backward-compatible entry point for compute_semantic_geometry_tables."""

    return compute_semantic_geometry_tables(**kwargs)


__all__ = [
    "CLASS_LEVEL_COLUMNS",
    "COMPARISON_SUMMARY_COLUMNS",
    "DEFAULT_SHOT_GROUPS",
    "GLOBAL_LEVEL_COLUMNS",
    "GROUP_LEVEL_COLUMNS",
    "REFERENCE_CACHE_TYPE",
    "SAMPLE_LEVEL_COLUMNS",
    "SEMANTIC_GEOMETRY_METRIC_REGISTRY",
    "SEMANTIC_GEOMETRY_METRIC_SPECS",
    "SUMMARY_FACET_COLUMNS",
    "SUMMARY_V2_LONG_COLUMNS",
    "build_fp32_semantic_geometry_reference",
    "build_semantic_geometry_comparison_summary",
    "build_semantic_geometry_summary_v2_long",
    "compute_semantic_geometry_metrics",
    "compute_semantic_geometry_tables",
]
