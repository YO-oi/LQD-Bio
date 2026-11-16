from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

try:
    from .common_metrics import (
        DEFAULT_EPS,
        DEFAULT_MARGIN_MIN_ABS,
        embedding_distortion as common_embedding_distortion,
        margin_value_retention as common_margin_value_retention,
        metric_drop,
        scalar_oer_energy_metrics,
    )
    from .geometry_metrics import compute_group_fisher as compute_global_anchored_group_fisher
    from src.utils.group_utils import build_metric_summary_rows_from_group_detail, compute_group_gap, normalize_analysis_group, require_metric_spec
except ImportError:  # pragma: no cover
    from src.metrics.common_metrics import (
        DEFAULT_EPS,
        DEFAULT_MARGIN_MIN_ABS,
        embedding_distortion as common_embedding_distortion,
        margin_value_retention as common_margin_value_retention,
        metric_drop,
        scalar_oer_energy_metrics,
    )
    from src.metrics.geometry_metrics import compute_group_fisher as compute_global_anchored_group_fisher
    from src.utils.group_utils import build_metric_summary_rows_from_group_detail, compute_group_gap, normalize_analysis_group, require_metric_spec


HEATMAP_MAIN_METRICS = (
    "margin_retention",
    "fisher_retention",
    "oer_retention",
    "oer_loss",
    "clipping_rate",
    "embedding_distortion",
)

HEATMAP_CSV_COLUMNS = [
    "layer_idx",
    "module_name",
    "quant_mode",
    "granularity",
    "analysis_group",
    "protocol_version",
    "experiment_signature",
    "group_protocol_version",
    "sample_count",
    "effective_class_count",
    "margin_fp32",
    "margin_quant",
    "delta_margin",
    "margin_retention",
    "margin_invalid_ratio",
    "margin_valid_count",
    "margin_total_count",
    "margin_valid_rate",
    "margin_invalid_reason",
    "fisher_fp32",
    "fisher_quant",
    "delta_fisher",
    "fisher_retention",
    "teacher_center_margin_quant",
    "teacher_center_margin_retention",
    "teacher_anchor_center_shift",
    "teacher_anchor_embedding_drift",
    "oer_retention",
    "oer_loss",
    "off_oracle_energy_ratio",
    "input_oer_retention",
    "input_oer_loss",
    "input_off_oracle_energy_ratio",
    "projection_output_oer",
    "projection_output_oer_retention",
    "projection_output_oer_loss",
    "projection_output_off_oracle_energy_ratio",
    "output_energy_retention",
    "ffn_hidden_activation_oer",
    "ffn_hidden_activation_oer_retention",
    "ffn_hidden_activation_oer_loss",
    "ffn_hidden_activation_off_oracle_energy_ratio",
    "clipping_rate",
    "embedding_distortion",
    "target_projection_l2_rel_error",
    "target_projection_cosine_drift",
    "target_projection_mse",
    "target_projection_mae",
    "ffn_hidden_activation_distortion",
    "ffn_hidden_activation_l2_rel_error",
    "ffn_hidden_activation_cosine_drift",
    "ffn_hidden_activation_mse",
    "ffn_hidden_activation_mae",
]

HEATMAP_SUMMARY_CSV_COLUMNS = [
    "layer_idx",
    "module_name",
    "quant_mode",
    "granularity",
    "protocol_version",
    "experiment_signature",
    "group_protocol_version",
    "metric_name",
    "metric_role",
    "metric_direction",
    "aggregation_unit",
    "aggregation_fn",
    "many_value",
    "medium_value",
    "few_value",
    "tail_value",
    "tail_vs_many_gap",
    "few_vs_many_gap",
    "many_sample_count",
    "medium_sample_count",
    "few_sample_count",
    "tail_sample_count",
    "many_effective_class_count",
    "medium_effective_class_count",
    "few_effective_class_count",
    "tail_effective_class_count",
    "many_margin_valid_count",
    "medium_margin_valid_count",
    "few_margin_valid_count",
    "tail_margin_valid_count",
    "many_margin_total_count",
    "medium_margin_total_count",
    "few_margin_total_count",
    "tail_margin_total_count",
    "many_margin_valid_rate",
    "medium_margin_valid_rate",
    "few_margin_valid_rate",
    "tail_margin_valid_rate",
]

SUMMARY_ANALYSIS_GROUPS = ("many", "medium", "few", "tail")
HEATMAP_SUMMARY_ID_COLUMNS = (
    "protocol_version",
    "experiment_signature",
    "group_protocol_version",
    "layer_idx",
    "module_name",
    "quant_mode",
    "granularity",
)
HEATMAP_SUMMARY_GROUP_COLUMNS = (
    "sample_count",
    "effective_class_count",
    "margin_valid_count",
    "margin_total_count",
    "margin_valid_rate",
)
HEATMAP_SUMMARY_INTEGER_GROUP_COLUMNS = (
    "sample_count",
    "effective_class_count",
    "margin_valid_count",
    "margin_total_count",
)
PROJECTION_ERROR_HISTOGRAM_EDGES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf"))

PROJECTION_DETAIL_CSV_COLUMNS = [
    "layer_idx",
    "module_name",
    "quant_mode",
    "granularity",
    "analysis_group",
    "protocol_version",
    "experiment_signature",
    "group_protocol_version",
    "sample_count",
    "projection_rel_l2_error",
    "projection_cosine_drift",
    "projection_mse",
    "projection_mae",
    "projection_error_p95",
    "projection_error_p99",
    "projection_channel_error_mean",
    "projection_token_error_mean",
    "projection_oer_retention",
    "projection_oer_loss",
    "projection_off_oracle_energy_ratio",
    "projection_clipping_rate",
    "projection_near_zero_rate",
    "projection_error_histogram",
    "projection_error_histogram_edges",
]


def _as_float_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _as_long_tensor(value: Any, *, device: torch.device | None = None) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach().long()
    else:
        result = torch.as_tensor(value, dtype=torch.long)
    if device is not None:
        result = result.to(device=device)
    return result


def _flatten_embeddings(embeddings: Any, labels: Any) -> tuple[Tensor, Tensor]:
    x = _as_float_tensor(embeddings)
    if x.ndim < 2:
        raise ValueError("embeddings must contain sample and feature dimensions")
    samples = x.reshape(-1, int(x.shape[-1]))
    label_tensor = _as_long_tensor(labels, device=samples.device).reshape(-1)
    if int(label_tensor.numel()) != int(samples.shape[0]):
        raise ValueError(
            "labels must align with embedding samples, "
            f"got labels={int(label_tensor.numel())}, samples={int(samples.shape[0])}"
        )
    return samples, label_tensor


def _group_mask(labels: Tensor, group: Any) -> Tensor:
    if group is None:
        return torch.ones_like(labels, dtype=torch.bool)
    if isinstance(group, (set, frozenset)):
        group = list(group)
    group_tensor = torch.as_tensor(group, device=labels.device)
    if group_tensor.dtype == torch.bool and int(group_tensor.numel()) == int(labels.numel()):
        return group_tensor.reshape(labels.shape)
    group_tensor = group_tensor.reshape(-1).long()
    if int(group_tensor.numel()) == 0:
        return torch.zeros_like(labels, dtype=torch.bool)
    return (labels[:, None] == group_tensor[None, :]).any(dim=1)


def _nanmean(values: Tensor) -> Tensor:
    values = values.float()
    finite = torch.isfinite(values)
    if not finite.any().item():
        return values.new_tensor(float("nan"))
    return values[finite].mean()


def _safe_positive_retention(numerator: Any, denominator: Any, *, eps: float = DEFAULT_EPS) -> Tensor:
    num = _as_float_tensor(numerator)
    den = _as_float_tensor(denominator).to(device=num.device)
    valid = torch.isfinite(num) & torch.isfinite(den) & (den.abs() > float(eps))
    value = num / torch.clamp(den, min=float(eps))
    value = torch.where(valid, value, torch.full_like(value, float("nan")))
    return _nanmean(value.reshape(-1))


def _margin_retention_with_diagnostics(
    fp32_value: Any,
    quant_value: Any,
) -> tuple[Tensor, dict[str, Any]]:
    fp32 = _as_float_tensor(fp32_value)
    quant = _as_float_tensor(quant_value).to(device=fp32.device)
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"fp32_margin and quant_margin shapes must match, got {tuple(fp32.shape)} vs {tuple(quant.shape)}")

    retention, common_diag = common_margin_value_retention(
        fp32,
        quant,
        reduction="mean",
        min_abs_margin=DEFAULT_MARGIN_MIN_ABS,
        return_diagnostics=True,
    )
    valid = common_diag["valid_margin_mask"].to(device=fp32.device)
    invalid = ~valid
    total_count = int(common_diag["total_count"].detach().cpu().item())
    valid_count = int(common_diag["valid_count"].detach().cpu().item())
    invalid_count = int(total_count - valid_count)
    diagnostics = {
        "valid_mask": valid,
        "invalid_mask": invalid,
        "num_valid": fp32.new_tensor(float(valid_count)),
        "num_invalid": fp32.new_tensor(float(invalid_count)),
        "invalid_ratio": fp32.new_tensor(float(invalid_count / total_count) if total_count > 0 else float("nan")),
        "margin_valid_count": fp32.new_tensor(float(valid_count)),
        "margin_total_count": fp32.new_tensor(float(total_count)),
        "margin_valid_rate": common_diag["valid_rate"].to(device=fp32.device),
        "margin_invalid_reason": "see_count_fields",
        "margin_invalid_reason_counts": {
            "margin_fp32 <= margin_min_abs": int(common_diag["num_nonpositive"].detach().cpu().item())
            + int(common_diag["num_near_threshold"].detach().cpu().item()),
            "other_invalid": int(invalid_count)
            - int(common_diag["num_nonpositive"].detach().cpu().item())
            - int(common_diag["num_near_threshold"].detach().cpu().item()),
        },
    }
    return retention, diagnostics


def _extract_feature_tensor(cache: Any) -> Any:
    if isinstance(cache, Mapping):
        for key in ("activation", "feature", "features", "tensor", "value"):
            if key in cache:
                return cache[key]
        if len(cache) == 1:
            return next(iter(cache.values()))
    return cache


def _metric(metrics: Mapping[str, Any], name: str, *, default: Any = None) -> Any:
    return metrics[name] if name in metrics else default


def _require_metric(metrics: Mapping[str, Any], name: str) -> Any:
    if name not in metrics:
        raise KeyError(f"group heatmap row requires field: {name}")
    return metrics[name]


def _float_or_nan(value: Any) -> float:
    try:
        tensor = _as_float_tensor(value).reshape(-1)
        if int(tensor.numel()) == 0:
            return float("nan")
        return float(tensor[0].item())
    except Exception:
        try:
            return float(value)
        except Exception:
            return float("nan")


def compute_effective_class_count(labels: Any, group: Any) -> int:
    label_tensor = _as_long_tensor(labels).reshape(-1)
    mask = _group_mask(label_tensor, group)
    if not mask.any().item():
        return 0
    return int(torch.unique(label_tensor[mask], sorted=True).numel())


def compute_group_fisher(embeddings: Any, labels: Any, group: Any) -> Tensor:
    """Global-anchored group Fisher used by the layer-sensitivity experiment."""

    _samples, label_tensor = _flatten_embeddings(embeddings, labels)
    if compute_effective_class_count(label_tensor, group) < 2:
        return label_tensor.new_tensor(float("nan"), dtype=torch.float32)
    return compute_global_anchored_group_fisher(
        embeddings,
        labels,
        group,
        feature_dim=-1,
        normalize_by_dof=False,
        eps=DEFAULT_EPS,
    )


def compute_margin_retention(
    fp32_margin: Any,
    quant_margin: Any,
    *,
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    retention, diagnostics = _margin_retention_with_diagnostics(fp32_margin, quant_margin)
    if return_diagnostics:
        return retention, diagnostics
    return retention


def compute_collapse_gap(
    many_margin_fp32: Any,
    many_margin_quant: Any,
    few_margin_fp32: Any,
    few_margin_quant: Any,
) -> Tensor:
    """Legacy few-vs-many margin gap using the shared group gap direction helper."""

    spec = require_metric_spec("margin_retention")
    group_values = {
        "many": _float_or_nan(compute_margin_retention(many_margin_fp32, many_margin_quant)),
        "few": _float_or_nan(compute_margin_retention(few_margin_fp32, few_margin_quant)),
    }
    gap = compute_group_gap(
        group_values,
        comparison_group="few",
        metric_direction=spec.metric_direction,
    )
    return torch.tensor(float(gap), dtype=torch.float32)


def compute_embedding_distortion(fp32_embeddings: Any, quant_embeddings: Any, *, eps: float = DEFAULT_EPS) -> Tensor:
    fp32 = _as_float_tensor(fp32_embeddings)
    quant = _as_float_tensor(quant_embeddings).to(device=fp32.device)
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"fp32_embeddings and quant_embeddings shapes must match, got {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    if fp32.ndim < 2:
        raise ValueError("embeddings must contain sample and feature dimensions")

    distortion = common_embedding_distortion(fp32, quant, dim=-1, reduction="none", eps=eps)
    denominator = torch.linalg.vector_norm(fp32, ord=2, dim=-1)
    valid = denominator > float(eps)
    distortion = torch.where(valid, distortion, torch.full_like(distortion, float("nan")))
    return _nanmean(distortion.reshape(-1))


def compute_group_embedding_distortion(fp32_embeddings: Any, quant_embeddings: Any, mask: Any) -> Tensor:
    selected = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    if not selected.any().item():
        return torch.tensor(float("nan"))
    fp32 = _as_float_tensor(fp32_embeddings)
    quant = _as_float_tensor(quant_embeddings).to(device=fp32.device)
    if int(selected.numel()) != int(fp32.shape[0]):
        raise ValueError(f"embedding distortion mask length={int(selected.numel())} must match samples={int(fp32.shape[0])}")
    return compute_embedding_distortion(fp32[selected], quant[selected])



def _flatten_feature_by_sample(fp32_feature: Any, quant_feature: Any) -> tuple[Tensor, Tensor]:
    fp32 = _as_float_tensor(fp32_feature)
    quant = _as_float_tensor(quant_feature).to(device=fp32.device)
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"fp32_feature and quant_feature shapes must match, got {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    if fp32.ndim < 2:
        raise ValueError("feature tensors must contain a sample dimension and at least one feature dimension")
    return fp32.reshape(int(fp32.shape[0]), -1), quant.reshape(int(quant.shape[0]), -1)


def compute_tensor_distortion_metrics(fp32_feature: Any, quant_feature: Any, *, eps: float = DEFAULT_EPS) -> dict[str, Tensor]:
    """Per-sample distortion between aligned FP32 and quantized projection tensors."""

    fp32, quant = _flatten_feature_by_sample(fp32_feature, quant_feature)
    delta = quant - fp32
    fp32_norm = torch.linalg.vector_norm(fp32, ord=2, dim=1)
    quant_norm = torch.linalg.vector_norm(quant, ord=2, dim=1)
    l2_rel = torch.linalg.vector_norm(delta, ord=2, dim=1) / torch.clamp(fp32_norm, min=float(eps))
    l2_rel = torch.where(fp32_norm > float(eps), l2_rel, torch.full_like(l2_rel, float("nan")))

    cosine_denominator = fp32_norm * quant_norm
    cosine = (fp32 * quant).sum(dim=1) / torch.clamp(cosine_denominator, min=float(eps))
    cosine_drift = 1.0 - cosine
    cosine_drift = torch.where(
        cosine_denominator > float(eps),
        cosine_drift,
        torch.full_like(cosine_drift, float("nan")),
    )
    abs_delta = delta.abs()
    flat_abs_delta = abs_delta.reshape(int(abs_delta.shape[0]), -1)
    if int(flat_abs_delta.shape[1]) > 0:
        error_p95 = torch.quantile(flat_abs_delta, 0.95, dim=1)
        error_p99 = torch.quantile(flat_abs_delta, 0.99, dim=1)
    else:
        error_p95 = fp32.new_full((int(fp32.shape[0]),), float("nan"))
        error_p99 = fp32.new_full((int(fp32.shape[0]),), float("nan"))

    if abs_delta.ndim >= 3:
        channel_reduce_dims = tuple(range(1, abs_delta.ndim - 1))
        channel_error = abs_delta.mean(dim=channel_reduce_dims) if channel_reduce_dims else abs_delta
        channel_error_mean = channel_error.mean(dim=1)
        token_reduce_dims = tuple(range(2, abs_delta.ndim))
        token_error = abs_delta.mean(dim=token_reduce_dims) if token_reduce_dims else abs_delta
        token_error_mean = token_error.mean(dim=1)
    else:
        mean_abs = flat_abs_delta.mean(dim=1)
        channel_error_mean = mean_abs
        token_error_mean = mean_abs

    near_zero_rate = (quant.abs().reshape(int(quant.shape[0]), -1) <= float(eps)).float().mean(dim=1)
    histogram_edges = torch.as_tensor(PROJECTION_ERROR_HISTOGRAM_EDGES, device=l2_rel.device, dtype=l2_rel.dtype)
    finite_l2_rel = torch.where(torch.isfinite(l2_rel), l2_rel, torch.full_like(l2_rel, float("inf")))
    bin_indices = torch.bucketize(finite_l2_rel, histogram_edges[1:-1], right=False)
    error_histogram = torch.zeros(
        (int(l2_rel.shape[0]), len(PROJECTION_ERROR_HISTOGRAM_EDGES) - 1),
        device=l2_rel.device,
        dtype=torch.float32,
    )
    error_histogram.scatter_add_(1, bin_indices[:, None], torch.ones_like(bin_indices[:, None], dtype=torch.float32))

    return {
        "l2_rel_error": l2_rel.detach().cpu(),
        "cosine_drift": cosine_drift.detach().cpu(),
        "mse": delta.square().mean(dim=1).detach().cpu(),
        "mae": delta.abs().mean(dim=1).detach().cpu(),
        "error_p95": error_p95.detach().cpu(),
        "error_p99": error_p99.detach().cpu(),
        "channel_error_mean": channel_error_mean.detach().cpu(),
        "token_error_mean": token_error_mean.detach().cpu(),
        "near_zero_rate": near_zero_rate.detach().cpu(),
        "error_histogram": error_histogram.detach().cpu(),
    }


def compute_group_tensor_distortion_metrics_from_cache(cache: Mapping[str, Any], mask: Any) -> dict[str, Tensor]:
    return {
        "l2_rel_error": compute_group_scalar_mean(cache["l2_rel_error"], mask),
        "cosine_drift": compute_group_scalar_mean(cache["cosine_drift"], mask),
        "mse": compute_group_scalar_mean(cache["mse"], mask),
        "mae": compute_group_scalar_mean(cache["mae"], mask),
    }


def _group_histogram_sum(values: Any, mask: Any) -> list[float]:
    histogram = _as_float_tensor(values)
    if histogram.ndim != 2:
        raise ValueError(f"histogram cache must be [sample,bins], got {tuple(histogram.shape)}")
    selected = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    if int(selected.numel()) != int(histogram.shape[0]):
        raise ValueError(f"histogram mask length={int(selected.numel())} must match samples={int(histogram.shape[0])}")
    if not selected.any().item():
        return [float("nan") for _ in range(int(histogram.shape[1]))]
    return [float(item) for item in histogram[selected].sum(dim=0).detach().cpu().reshape(-1).tolist()]


def compute_group_projection_detail_row_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    projection_cache = _require_metric(metrics, "projection_distortion")
    projection_oer = _require_metric(metrics, "projection_oer_metrics")
    group_mask = _require_metric(metrics, "group_mask")
    reference = _as_float_tensor(projection_cache["l2_rel_error"])
    return {
        "layer_idx": _require_metric(metrics, "layer_idx"),
        "module_name": _require_metric(metrics, "module_name"),
        "quant_mode": _require_metric(metrics, "quant_mode"),
        "granularity": _require_metric(metrics, "granularity"),
        "analysis_group": normalize_analysis_group(_require_metric(metrics, "analysis_group")),
        "group_protocol_version": int(_metric(metrics, "group_protocol_version", default=2)),
        "sample_count": int(_metric(metrics, "sample_count", default=0)),
        "projection_rel_l2_error": compute_group_scalar_mean(projection_cache["l2_rel_error"], group_mask),
        "projection_cosine_drift": compute_group_scalar_mean(projection_cache["cosine_drift"], group_mask),
        "projection_mse": compute_group_scalar_mean(projection_cache["mse"], group_mask),
        "projection_mae": compute_group_scalar_mean(projection_cache["mae"], group_mask),
        "projection_error_p95": compute_group_scalar_mean(projection_cache["error_p95"], group_mask),
        "projection_error_p99": compute_group_scalar_mean(projection_cache["error_p99"], group_mask),
        "projection_channel_error_mean": compute_group_scalar_mean(projection_cache["channel_error_mean"], group_mask),
        "projection_token_error_mean": compute_group_scalar_mean(projection_cache["token_error_mean"], group_mask),
        "projection_oer_retention": _as_float_tensor(projection_oer["oer_retention"]).reshape(()),
        "projection_oer_loss": _as_float_tensor(projection_oer["oer_loss"]).reshape(()),
        "projection_off_oracle_energy_ratio": _as_float_tensor(projection_oer["off_oracle_energy_ratio"]).reshape(()),
        "projection_clipping_rate": _as_float_tensor(_metric(metrics, "projection_clipping_rate", default=reference.new_tensor(float("nan")))).reshape(()),
        "projection_near_zero_rate": compute_group_scalar_mean(projection_cache["near_zero_rate"], group_mask),
        "projection_error_histogram": _group_histogram_sum(projection_cache["error_histogram"], group_mask),
        "projection_error_histogram_edges": list(PROJECTION_ERROR_HISTOGRAM_EDGES),
    }


def _class_centers_from_samples(samples: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    class_ids = torch.unique(labels, sorted=True)
    if int(class_ids.numel()) == 0:
        raise ValueError("labels contain no classes")
    centers = []
    for class_id in class_ids:
        class_samples = samples[labels == class_id]
        if int(class_samples.shape[0]) <= 0:
            continue
        centers.append(class_samples.mean(dim=0))
    return torch.stack(centers, dim=0), class_ids


def _label_center_indices(labels: Tensor, class_ids: Tensor) -> Tensor:
    sorted_class_ids = class_ids.to(device=labels.device)
    indices = torch.searchsorted(sorted_class_ids, labels)
    if (indices >= int(sorted_class_ids.numel())).any().item() or not torch.equal(sorted_class_ids[indices], labels):
        raise RuntimeError("class center index cannot align with label_id")
    return indices


def compute_teacher_center_margin_quant(fp32_embeddings: Any, quant_embeddings: Any, labels: Any, group: Any) -> Tensor:
    """Quant sample margin measured against fixed FP32 teacher class centers."""

    fp32_samples, label_tensor = _flatten_embeddings(fp32_embeddings, labels)
    quant_samples, quant_labels = _flatten_embeddings(quant_embeddings, labels)
    if tuple(fp32_samples.shape) != tuple(quant_samples.shape):
        raise ValueError(
            "fp32_embeddings and quant_embeddings must align for teacher-center margin, "
            f"got {tuple(fp32_samples.shape)} vs {tuple(quant_samples.shape)}"
        )
    if not torch.equal(label_tensor, quant_labels):
        raise ValueError("label tensors differ between FP32 and quant embeddings")

    selected = _group_mask(label_tensor, group)
    if not selected.any().item():
        return fp32_samples.new_empty((0,))
    centers, class_ids = _class_centers_from_samples(fp32_samples, label_tensor)
    if int(class_ids.numel()) < 2:
        return fp32_samples.new_full((int(selected.sum().item()),), float("nan"))

    center_indices = _label_center_indices(label_tensor, class_ids)
    distances = torch.cdist(quant_samples, centers.to(device=quant_samples.device), p=2.0)
    own_distance = distances.gather(dim=1, index=center_indices[:, None]).squeeze(1)
    rival_distances = distances.clone()
    rival_distances.scatter_(dim=1, index=center_indices[:, None], value=float("inf"))
    nearest_rival_distance = rival_distances.min(dim=1).values
    margins = nearest_rival_distance - own_distance
    return margins[selected].detach().cpu()


def compute_teacher_anchor_center_shift(fp32_embeddings: Any, quant_embeddings: Any, labels: Any, group: Any) -> Tensor:
    """Mean class-center drift ||mu_quant,c - mu_fp32,c|| for classes in ``group``."""

    fp32_samples, label_tensor = _flatten_embeddings(fp32_embeddings, labels)
    quant_samples, quant_labels = _flatten_embeddings(quant_embeddings, labels)
    if tuple(fp32_samples.shape) != tuple(quant_samples.shape):
        raise ValueError(
            "fp32_embeddings and quant_embeddings must align for teacher anchor center shift, "
            f"got {tuple(fp32_samples.shape)} vs {tuple(quant_samples.shape)}"
        )
    if not torch.equal(label_tensor, quant_labels):
        raise ValueError("label tensors differ between FP32 and quant embeddings")

    fp32_centers, fp32_class_ids = _class_centers_from_samples(fp32_samples, label_tensor)
    quant_centers, quant_class_ids = _class_centers_from_samples(quant_samples, label_tensor)
    if not torch.equal(fp32_class_ids, quant_class_ids):
        raise ValueError("FP32 and quant class centers are not aligned")
    selected_classes = _group_mask(fp32_class_ids, group)
    if not selected_classes.any().item():
        return fp32_samples.new_tensor(float("nan"))
    shift = torch.linalg.vector_norm(quant_centers - fp32_centers, ord=2, dim=1)
    return _nanmean(shift[selected_classes])


def compute_teacher_anchor_embedding_drift(fp32_embeddings: Any, quant_embeddings: Any, labels: Any, mask: Any, *, eps: float = DEFAULT_EPS) -> Tensor:
    """Sample drift ||z_q - z_fp32|| normalized by FP32 distance to the own teacher center."""

    fp32_samples, label_tensor = _flatten_embeddings(fp32_embeddings, labels)
    quant_samples, quant_labels = _flatten_embeddings(quant_embeddings, labels)
    if tuple(fp32_samples.shape) != tuple(quant_samples.shape):
        raise ValueError(
            "fp32_embeddings and quant_embeddings must align for teacher anchor embedding drift, "
            f"got {tuple(fp32_samples.shape)} vs {tuple(quant_samples.shape)}"
        )
    if not torch.equal(label_tensor, quant_labels):
        raise ValueError("label tensors differ between FP32 and quant embeddings")
    selected = torch.as_tensor(mask, dtype=torch.bool, device=fp32_samples.device).reshape(-1)
    if int(selected.numel()) != int(fp32_samples.shape[0]):
        raise ValueError(f"embedding drift mask length={int(selected.numel())} must match samples={int(fp32_samples.shape[0])}")
    if not selected.any().item():
        return fp32_samples.new_tensor(float("nan"))

    centers, class_ids = _class_centers_from_samples(fp32_samples, label_tensor)
    center_indices = _label_center_indices(label_tensor, class_ids)
    own_centers = centers.to(device=fp32_samples.device)[center_indices]
    denominator = torch.linalg.vector_norm(fp32_samples - own_centers, ord=2, dim=1)
    drift = torch.linalg.vector_norm(quant_samples - fp32_samples, ord=2, dim=1) / torch.clamp(denominator, min=float(eps))
    drift = torch.where(denominator > float(eps), drift, torch.full_like(drift, float("nan")))
    return _nanmean(drift[selected])


def compute_group_scalar_mean(values: Any, mask: Any) -> Tensor:
    tensor = _as_float_tensor(values).reshape(-1)
    selected = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    if int(tensor.numel()) == 1 and int(selected.numel()) != 1:
        return tensor.reshape(())
    if int(tensor.numel()) != int(selected.numel()):
        raise ValueError(f"scalar metric length={int(tensor.numel())} must match mask length={int(selected.numel())}")
    if not selected.any().item():
        return tensor.new_tensor(float("nan"))
    return _nanmean(tensor[selected])


def _as_oracle_metadata(oracle_indices: Any, *, batch_size: int) -> Mapping[str, Any]:
    if not isinstance(oracle_indices, Mapping):
        raise TypeError(
            "oracle_indices must be metadata with sample_id/layer_idx/module_name/token_idx/channel_idx, "
            "not a bare flat index tensor"
        )
    required = ("sample_id", "layer_idx", "module_name", "token_idx", "channel_idx")
    missing = [key for key in required if key not in oracle_indices or oracle_indices[key] is None]
    if missing:
        raise KeyError(f"oracle metadata is missing fields: {missing}")
    if "topk_indices" not in oracle_indices and "flat_index" not in oracle_indices:
        raise KeyError("oracle metadata must contain topk_indices or flat_index")

    sample_id = oracle_indices["sample_id"]
    if isinstance(sample_id, Sequence) and not isinstance(sample_id, (str, bytes)):
        if len(sample_id) != int(batch_size):
            raise ValueError(f"oracle sample_id length must equal batch size, got {len(sample_id)} vs {batch_size}")
    else:
        raise TypeError("oracle metadata sample_id must be a batch-aligned sequence")
    return oracle_indices


def _oracle_topk_indices(oracle_metadata: Mapping[str, Any], *, device: torch.device, num_scalars: int, batch_size: int) -> Tensor:
    raw_indices = oracle_metadata.get("topk_indices", oracle_metadata.get("flat_index"))
    indices = _as_long_tensor(raw_indices, device=device).reshape(batch_size, -1)
    if int(indices.numel()) == 0:
        raise ValueError("oracle topk_indices must not be empty")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(num_scalars):
        raise ValueError("oracle topk_indices contains out-of-range flat index")

    token_idx = _as_long_tensor(oracle_metadata["token_idx"], device=device).reshape(batch_size, -1)
    channel_idx = _as_long_tensor(oracle_metadata["channel_idx"], device=device).reshape(batch_size, -1)
    if tuple(token_idx.shape) != tuple(indices.shape) or tuple(channel_idx.shape) != tuple(indices.shape):
        raise ValueError("oracle token_idx/channel_idx shape must match topk_indices")
    return indices


def compute_oer_retention(fp32_feature_cache: Any, quant_feature_cache: Any, oracle_indices: Any) -> dict[str, Tensor]:
    fp32 = _as_float_tensor(_extract_feature_tensor(fp32_feature_cache))
    quant = _as_float_tensor(_extract_feature_tensor(quant_feature_cache)).to(device=fp32.device)
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"fp32_feature and quant_feature shapes must match, got {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    if fp32.ndim < 2:
        raise ValueError("OER feature must contain batch and scalar dimensions")

    batch_size = int(fp32.shape[0])
    metadata = _as_oracle_metadata(oracle_indices, batch_size=batch_size)
    indices = _oracle_topk_indices(
        metadata,
        device=fp32.device,
        num_scalars=int(fp32.reshape(batch_size, -1).shape[1]),
        batch_size=batch_size,
    )
    return scalar_oer_energy_metrics(
        fp32,
        quant,
        topk_indices=indices,
        batch_dim=0,
        reduction="mean",
        eps=DEFAULT_EPS,
    )


def compute_oer_retention_from_energy(
    fp32_oracle_energy: Any,
    quant_oracle_energy: Any,
    quant_total_energy: Any,
    *,
    mask: Any | None = None,
    eps: float = DEFAULT_EPS,
) -> dict[str, Tensor]:
    """Compute OER metrics from per-sample energy caches."""

    fp32_energy = _as_float_tensor(fp32_oracle_energy).reshape(-1)
    quant_oracle = _as_float_tensor(quant_oracle_energy).to(device=fp32_energy.device).reshape(-1)
    quant_total = _as_float_tensor(quant_total_energy).to(device=fp32_energy.device).reshape(-1)
    if tuple(fp32_energy.shape) != tuple(quant_oracle.shape) or tuple(fp32_energy.shape) != tuple(quant_total.shape):
        raise ValueError(
            "OER energy tensors must have the same per-sample shape, "
            f"got fp32={tuple(fp32_energy.shape)}, quant_oracle={tuple(quant_oracle.shape)}, "
            f"quant_total={tuple(quant_total.shape)}"
        )
    if mask is not None:
        selected = torch.as_tensor(mask, device=fp32_energy.device, dtype=torch.bool).reshape(-1)
        if int(selected.numel()) != int(fp32_energy.numel()):
            raise ValueError(f"OER mask length={int(selected.numel())} must match samples={int(fp32_energy.numel())}")
        fp32_energy = fp32_energy[selected]
        quant_oracle = quant_oracle[selected]
        quant_total = quant_total[selected]
    if int(fp32_energy.numel()) == 0:
        value = torch.tensor(float("nan"))
        return {
            "oer_retention": value,
            "oer_loss": value,
            "off_oracle_energy_ratio": value,
        }

    retention = quant_oracle / torch.clamp(fp32_energy, min=float(eps))
    retention = torch.where(
        fp32_energy > float(eps),
        retention,
        torch.full_like(retention, float("nan")),
    )
    off_oracle = (quant_total - quant_oracle) / torch.clamp(quant_total, min=float(eps))
    off_oracle = torch.where(
        quant_total > float(eps),
        off_oracle,
        torch.full_like(off_oracle, float("nan")),
    )
    oer_retention = _nanmean(retention)
    return {
        "oer_retention": oer_retention,
        "oer_loss": 1.0 - oer_retention,
        "off_oracle_energy_ratio": _nanmean(off_oracle),
    }


def compute_oer_retention_from_energy_cache(
    oracle_cache: Mapping[str, Any],
    quant_energy_cache: Mapping[str, Any],
    *,
    mask: Any | None = None,
) -> dict[str, Tensor]:
    return compute_oer_retention_from_energy(
        oracle_cache["fp32_oracle_energy"],
        quant_energy_cache["quant_oracle_energy"],
        quant_energy_cache["quant_total_energy"],
        mask=mask,
    )


def compute_group_oer_metrics_from_energy_cache(
    oracle_cache: Mapping[str, Any],
    quant_energy_cache: Mapping[str, Any],
    mask: Any,
) -> dict[str, Tensor]:
    selected = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    if not selected.any().item():
        value = torch.tensor(float("nan"))
        return {
            "oer_retention": value,
            "oer_loss": value,
            "off_oracle_energy_ratio": value,
        }
    return compute_oer_retention_from_energy_cache(oracle_cache, quant_energy_cache, mask=selected)



def _optional_scalar(value: Any, reference: Tensor) -> Tensor:
    if value is None:
        return reference.new_tensor(float("nan"))
    return _as_float_tensor(value).to(device=reference.device).reshape(())


def _optional_mapping_scalar(metrics: Any, name: str, reference: Tensor) -> Tensor:
    if isinstance(metrics, Mapping) and name in metrics:
        return _optional_scalar(metrics[name], reference)
    return reference.new_tensor(float("nan"))


def compute_group_heatmap_row_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    margin_fp32 = _as_float_tensor(_require_metric(metrics, "margin_fp32"))
    margin_quant = _as_float_tensor(_require_metric(metrics, "margin_quant")).to(device=margin_fp32.device)
    fisher_fp32 = _as_float_tensor(_require_metric(metrics, "fisher_fp32"))
    fisher_quant = _as_float_tensor(_require_metric(metrics, "fisher_quant")).to(device=fisher_fp32.device)

    margin_retention, margin_diag = compute_margin_retention(
        margin_fp32,
        margin_quant,
        return_diagnostics=True,
    )
    delta_margin = metric_drop(margin_fp32, margin_quant, reduction="mean")
    delta_fisher = metric_drop(fisher_fp32, fisher_quant, reduction="mean")
    fisher_retention = _safe_positive_retention(fisher_quant, fisher_fp32)

    oer_metrics = _metric(metrics, "oer_metrics", default=None)
    if oer_metrics is None:
        oer_metrics = {
            "oer_retention": _require_metric(metrics, "oer_retention"),
            "oer_loss": _require_metric(metrics, "oer_loss"),
            "off_oracle_energy_ratio": _require_metric(metrics, "off_oracle_energy_ratio"),
        }
    input_oer_metrics = _metric(metrics, "input_oer_metrics", default=None)
    projection_output_oer_metrics = _metric(metrics, "projection_output_oer_metrics", default=oer_metrics)
    hidden_oer_metrics = _metric(metrics, "ffn_hidden_activation_oer_metrics", default=None)
    projection_distortion = _metric(metrics, "target_projection_distortion", default=None)
    hidden_distortion = _metric(metrics, "ffn_hidden_activation_distortion_metrics", default=None)

    teacher_center_margin_quant = _metric(metrics, "teacher_center_margin_quant", default=None)
    nan_scalar = margin_fp32.new_tensor(float("nan"))
    if teacher_center_margin_quant is None:
        teacher_center_margin_quant_tensor = margin_fp32.new_empty((0,))
        teacher_center_margin_retention = nan_scalar
    else:
        teacher_center_margin_quant_tensor = _as_float_tensor(teacher_center_margin_quant).to(device=margin_fp32.device)
        if int(margin_fp32.numel()) == 0 or int(teacher_center_margin_quant_tensor.numel()) == 0:
            teacher_center_margin_retention = nan_scalar
        elif tuple(margin_fp32.shape) != tuple(teacher_center_margin_quant_tensor.shape):
            raise ValueError(
                "margin_fp32 and teacher_center_margin_quant shapes must match, "
                f"got {tuple(margin_fp32.shape)} vs {tuple(teacher_center_margin_quant_tensor.shape)}"
            )
        else:
            teacher_center_margin_retention = compute_margin_retention(margin_fp32, teacher_center_margin_quant_tensor)

    projection_output_oer_retention = _optional_mapping_scalar(projection_output_oer_metrics, "oer_retention", margin_fp32)
    hidden_oer_retention = _optional_mapping_scalar(hidden_oer_metrics, "oer_retention", margin_fp32)
    projection_l2_rel_error = _optional_mapping_scalar(projection_distortion, "l2_rel_error", margin_fp32)
    hidden_l2_rel_error = _optional_mapping_scalar(hidden_distortion, "l2_rel_error", margin_fp32)

    analysis_group = normalize_analysis_group(_require_metric(metrics, "analysis_group"))
    row = {
        "layer_idx": _require_metric(metrics, "layer_idx"),
        "module_name": _require_metric(metrics, "module_name"),
        "quant_mode": _require_metric(metrics, "quant_mode"),
        "granularity": _require_metric(metrics, "granularity"),
        "analysis_group": analysis_group,
        "sample_count": int(_metric(metrics, "sample_count", default=0)),
        "effective_class_count": int(_metric(metrics, "effective_class_count", default=0)),
        "group_protocol_version": int(_metric(metrics, "group_protocol_version", default=2)),
        "margin_fp32": _nanmean(margin_fp32.reshape(-1)),
        "margin_quant": _nanmean(margin_quant.reshape(-1)),
        "delta_margin": delta_margin,
        "margin_retention": margin_retention,
        "margin_invalid_ratio": margin_diag["invalid_ratio"],
        "margin_valid_count": margin_diag["margin_valid_count"],
        "margin_total_count": margin_diag["margin_total_count"],
        "margin_valid_rate": margin_diag["margin_valid_rate"],
        "margin_invalid_reason": margin_diag["margin_invalid_reason"],
        "fisher_fp32": _nanmean(fisher_fp32.reshape(-1)),
        "fisher_quant": _nanmean(fisher_quant.reshape(-1)),
        "delta_fisher": delta_fisher,
        "fisher_retention": fisher_retention,
        "teacher_center_margin_quant": _nanmean(teacher_center_margin_quant_tensor.reshape(-1)) if int(teacher_center_margin_quant_tensor.numel()) > 0 else nan_scalar,
        "teacher_center_margin_retention": teacher_center_margin_retention,
        "teacher_anchor_center_shift": _optional_scalar(_metric(metrics, "teacher_anchor_center_shift", default=None), margin_fp32),
        "teacher_anchor_embedding_drift": _optional_scalar(_metric(metrics, "teacher_anchor_embedding_drift", default=None), margin_fp32),
        "oer_retention": _as_float_tensor(oer_metrics["oer_retention"]).reshape(()),
        "oer_loss": _as_float_tensor(oer_metrics["oer_loss"]).reshape(()),
        "off_oracle_energy_ratio": _as_float_tensor(oer_metrics["off_oracle_energy_ratio"]).reshape(()),
        "input_oer_retention": _optional_mapping_scalar(input_oer_metrics, "oer_retention", margin_fp32),
        "input_oer_loss": _optional_mapping_scalar(input_oer_metrics, "oer_loss", margin_fp32),
        "input_off_oracle_energy_ratio": _optional_mapping_scalar(input_oer_metrics, "off_oracle_energy_ratio", margin_fp32),
        "projection_output_oer": projection_output_oer_retention,
        "projection_output_oer_retention": projection_output_oer_retention,
        "projection_output_oer_loss": _optional_mapping_scalar(projection_output_oer_metrics, "oer_loss", margin_fp32),
        "projection_output_off_oracle_energy_ratio": _optional_mapping_scalar(projection_output_oer_metrics, "off_oracle_energy_ratio", margin_fp32),
        "output_energy_retention": projection_output_oer_retention,
        "ffn_hidden_activation_oer": hidden_oer_retention,
        "ffn_hidden_activation_oer_retention": hidden_oer_retention,
        "ffn_hidden_activation_oer_loss": _optional_mapping_scalar(hidden_oer_metrics, "oer_loss", margin_fp32),
        "ffn_hidden_activation_off_oracle_energy_ratio": _optional_mapping_scalar(hidden_oer_metrics, "off_oracle_energy_ratio", margin_fp32),
        "clipping_rate": _as_float_tensor(_require_metric(metrics, "clipping_rate")).reshape(()),
        "embedding_distortion": _as_float_tensor(_require_metric(metrics, "embedding_distortion")).reshape(()),
        "target_projection_l2_rel_error": projection_l2_rel_error,
        "target_projection_cosine_drift": _optional_mapping_scalar(projection_distortion, "cosine_drift", margin_fp32),
        "target_projection_mse": _optional_mapping_scalar(projection_distortion, "mse", margin_fp32),
        "target_projection_mae": _optional_mapping_scalar(projection_distortion, "mae", margin_fp32),
        "ffn_hidden_activation_distortion": hidden_l2_rel_error,
        "ffn_hidden_activation_l2_rel_error": hidden_l2_rel_error,
        "ffn_hidden_activation_cosine_drift": _optional_mapping_scalar(hidden_distortion, "cosine_drift", margin_fp32),
        "ffn_hidden_activation_mse": _optional_mapping_scalar(hidden_distortion, "mse", margin_fp32),
        "ffn_hidden_activation_mae": _optional_mapping_scalar(hidden_distortion, "mae", margin_fp32),
    }
    return row


def build_heatmap_metric_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return build_metric_summary_rows_from_group_detail(
        rows,
        metric_names=HEATMAP_MAIN_METRICS,
        id_columns=HEATMAP_SUMMARY_ID_COLUMNS,
        per_group_columns=HEATMAP_SUMMARY_GROUP_COLUMNS,
        integer_per_group_columns=HEATMAP_SUMMARY_INTEGER_GROUP_COLUMNS,
        required_analysis_groups=SUMMARY_ANALYSIS_GROUPS,
        context="heatmap summary",
    )


def compute_few_group_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return compute_group_heatmap_row_summary(metrics)


__all__ = [
    "HEATMAP_CSV_COLUMNS",
    "HEATMAP_MAIN_METRICS",
    "HEATMAP_SUMMARY_CSV_COLUMNS",
    "PROJECTION_DETAIL_CSV_COLUMNS",
    "PROJECTION_ERROR_HISTOGRAM_EDGES",
    "build_heatmap_metric_summary_rows",
    "compute_collapse_gap",
    "compute_effective_class_count",
    "compute_embedding_distortion",
    "compute_few_group_summary",
    "compute_group_embedding_distortion",
    "compute_group_fisher",
    "compute_group_heatmap_row_summary",
    "compute_group_oer_metrics_from_energy_cache",
    "compute_group_projection_detail_row_summary",
    "compute_group_scalar_mean",
    "compute_group_tensor_distortion_metrics_from_cache",
    "compute_margin_retention",
    "compute_teacher_anchor_center_shift",
    "compute_teacher_anchor_embedding_drift",
    "compute_teacher_center_margin_quant",
    "compute_tensor_distortion_metrics",
    "compute_oer_retention",
    "compute_oer_retention_from_energy",
    "compute_oer_retention_from_energy_cache",
]

