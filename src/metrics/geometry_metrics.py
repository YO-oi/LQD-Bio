from __future__ import annotations

from typing import Any, Literal, Sequence

import warnings

import torch
from torch import Tensor

try:
    from .metric_config import (
        DEFAULT_EPS,
        DEFAULT_FISHER_NORMALIZE_BY_DOF,
        DEFAULT_GEOMETRY_CENTER_SHIFT_MODE,
        DEFAULT_GEOMETRY_VARIANCE_NORMALIZATION,
    )
except ImportError:  # pragma: no cover - allow direct script-style debugging
    from src.metrics.metric_config import (
        DEFAULT_EPS,
        DEFAULT_FISHER_NORMALIZE_BY_DOF,
        DEFAULT_GEOMETRY_CENTER_SHIFT_MODE,
        DEFAULT_GEOMETRY_VARIANCE_NORMALIZATION,
    )

try:
    from src.utils.group_utils import analysis_group_mask, normalize_analysis_group
except ImportError:  # pragma: no cover - allow direct script-style debugging
    from ..utils.group_utils import analysis_group_mask, normalize_analysis_group


Reduction = Literal["none", "mean", "sum"]
VarianceNormalization = Literal["dof", "population", "scatter"]
CenterDistanceMode = Literal["absolute", "relative"]

if DEFAULT_GEOMETRY_VARIANCE_NORMALIZATION not in {"dof", "population", "scatter"}:
    raise ValueError(
        "METRICS.GEOMETRY_VARIANCE_NORMALIZATION must be one of dof/population/scatter, "
        f"got {DEFAULT_GEOMETRY_VARIANCE_NORMALIZATION!r}"
    )
if DEFAULT_GEOMETRY_CENTER_SHIFT_MODE not in {"absolute", "relative"}:
    raise ValueError(
        "METRICS.GEOMETRY_CENTER_SHIFT_MODE must be absolute or relative, "
        f"got {DEFAULT_GEOMETRY_CENTER_SHIFT_MODE!r}"
    )

DEFAULT_VARIANCE_NORMALIZATION: VarianceNormalization = DEFAULT_GEOMETRY_VARIANCE_NORMALIZATION  # type: ignore[assignment]
DEFAULT_CENTER_SHIFT_MODE: CenterDistanceMode = DEFAULT_GEOMETRY_CENTER_SHIFT_MODE  # type: ignore[assignment]


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


def _check_same_shape(a: Tensor, b: Tensor, *, name_a: str, name_b: str) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, "
            f"got {tuple(a.shape)} vs {tuple(b.shape)}"
        )


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("dim is invalid for a scalar tensor")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim={dim} is out of range for ndim={ndim}")
    return dim


def _nan_reduction(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values

    finite = torch.isfinite(values)
    cleaned = torch.where(finite, values, torch.zeros_like(values))
    denom = finite.float().sum()
    if reduction == "sum":
        return torch.where(
            denom > 0,
            cleaned.sum(),
            values.new_tensor(float("nan")),
        )
    if reduction == "mean":
        return torch.where(
            denom > 0,
            cleaned.sum() / torch.clamp(denom, min=1.0),
            values.new_tensor(float("nan")),
        )
    raise ValueError(f"Unsupported reduction={reduction!r}; use none/mean/sum")


def _flatten_embeddings(
    embeddings: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
) -> tuple[Tensor, Tensor]:
    x = _as_float_tensor(embeddings)
    if x.ndim < 2:
        raise ValueError("embeddings must have at least sample and feature dimensions")

    feature_dim = _normalize_dim(int(feature_dim), x.ndim)
    x_last = x.movedim(feature_dim, -1).contiguous()
    samples = x_last.reshape(-1, int(x_last.shape[-1]))
    if int(samples.shape[0]) <= 0:
        raise ValueError("embeddings must contain at least one sample")

    labels = _as_long_tensor(label_id, device=samples.device).reshape(-1)
    if int(labels.numel()) != int(samples.shape[0]):
        raise ValueError(
            f"label_id must provide one label per embedding sample, got {int(labels.numel())} "
            f"labels for {int(samples.shape[0])} samples"
        )
    return samples, labels


def _class_geometry(
    embeddings: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
) -> dict[str, Tensor]:
    samples, labels = _flatten_embeddings(embeddings, label_id, feature_dim=feature_dim)
    class_ids = torch.unique(labels, sorted=True)
    if int(class_ids.numel()) == 0:
        raise ValueError("label_id contains no classes")

    centers: list[Tensor] = []
    counts: list[int] = []
    within_scatter: list[Tensor] = []
    for class_id in class_ids:
        class_samples = samples[labels == class_id]
        count = int(class_samples.shape[0])
        if count <= 0:
            continue
        center = class_samples.mean(dim=0)
        centers.append(center)
        counts.append(count)
        within_scatter.append((class_samples - center).square().sum())

    return {
        "class_ids": class_ids,
        "counts": torch.tensor(counts, device=samples.device, dtype=torch.long),
        "centers": torch.stack(centers, dim=0),
        "within_scatter": torch.stack(within_scatter, dim=0),
    }


def _variance_from_scatter(
    within_scatter: Tensor,
    counts: Tensor,
    *,
    normalization: VarianceNormalization,
) -> tuple[Tensor, Tensor]:
    if normalization == "scatter":
        valid = counts > 0
        values = within_scatter
        return values, valid

    if normalization == "population":
        valid = counts > 0
        denominator = counts.float()
    elif normalization == "dof":
        valid = counts > 1
        denominator = (counts - 1).float()
    else:
        raise ValueError("normalization must be one of dof/population/scatter")

    values = within_scatter / torch.clamp(denominator, min=1.0)
    values = torch.where(valid, values, torch.full_like(values, float("nan")))
    return values, valid


def class_centers(
    embeddings: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    return_counts: bool = False,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    """Class centers ordered by sorted label_id."""

    stats = _class_geometry(embeddings, label_id, feature_dim=feature_dim)
    if not return_counts:
        return stats["centers"]
    return stats["centers"], stats["class_ids"], stats["counts"]




def class_center_cosine_margin(
    embeddings: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    reduction: Reduction = "none",
    return_details: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """按类别中心计算最近邻余弦间隔。

    对每个类别 c，先用完整 embedding 集合计算类别中心 mu_c，再找到余弦相似度
    最高的非本类中心。类别级 angular margin 定义为：
        margin_c = 1 - max_{j != c} cosine(mu_c, mu_j)

    这是 class-level center angular/cosine 指标，只描述类别中心之间的角度分离。
    它不是主线 sample-level 几何 margin，不能替代 compute_group_margin()。
    """

    stats = _class_geometry(embeddings, label_id, feature_dim=feature_dim)
    centers = stats["centers"]
    class_ids = stats["class_ids"]
    counts = stats["counts"]
    num_classes = int(class_ids.numel())
    if num_classes < 2:
        margins = torch.full((num_classes,), float("nan"), dtype=centers.dtype, device=centers.device)
        details = {
            "class_ids": class_ids,
            "counts": counts,
            "centers": centers,
            "nearest_rival_class_id": torch.full_like(class_ids, -1),
            "nearest_rival_cosine": margins.clone(),
            "valid_class_mask": torch.isfinite(margins),
        }
        result = _nan_reduction(margins, reduction)
        return (result, details) if return_details else result

    center_norm = torch.linalg.vector_norm(centers, ord=2, dim=1)
    valid_center = center_norm > float(eps)
    normalized = centers / torch.clamp(center_norm[:, None], min=float(eps))
    cosine = normalized @ normalized.T
    cosine = torch.where(
        valid_center[:, None] & valid_center[None, :],
        cosine,
        torch.full_like(cosine, float("nan")),
    )
    cosine.fill_diagonal_(float("-inf"))
    nearest = cosine.max(dim=1)
    nearest_cosine = nearest.values
    nearest_index = nearest.indices
    margins = 1.0 - nearest_cosine
    margins = torch.where(
        torch.isfinite(nearest_cosine) & valid_center,
        margins,
        torch.full_like(margins, float("nan")),
    )

    result = _nan_reduction(margins, reduction)
    if not return_details:
        return result
    return result, {
        "class_ids": class_ids,
        "counts": counts,
        "centers": centers,
        "nearest_rival_class_id": class_ids[nearest_index],
        "nearest_rival_cosine": nearest_cosine,
        "valid_class_mask": torch.isfinite(margins),
    }

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
        raise ValueError("group must not be empty")
    return (labels[:, None] == group_tensor[None, :]).any(dim=1)


def _is_analysis_group(value: Any) -> bool:
    try:
        normalize_analysis_group(value)
    except Exception:
        return False
    return True


def _bucket_analysis_group_mask(
    labels: Tensor,
    bucket: Sequence[Any],
    analysis_group: Any,
) -> tuple[Tensor, str]:
    group_name = normalize_analysis_group(analysis_group)
    mask = analysis_group_mask(bucket, group_name)
    if isinstance(mask, Tensor):
        mask_tensor = mask.detach().to(device=labels.device, dtype=torch.bool).reshape(-1)
    else:
        mask_tensor = torch.tensor([bool(item) for item in mask], device=labels.device, dtype=torch.bool).reshape(-1)
    if int(mask_tensor.numel()) != int(labels.numel()):
        raise ValueError(
            "bucket must provide one canonical bucket per embedding sample, "
            f"got {int(mask_tensor.numel())} buckets for {int(labels.numel())} samples"
        )
    return mask_tensor.reshape(labels.shape), group_name


def _resolve_group_selection_mask(
    labels: Tensor,
    group: Any,
    *,
    bucket: Sequence[Any] | None = None,
    analysis_group: Any | None = None,
) -> tuple[Tensor, str | None]:
    if analysis_group is not None:
        if bucket is None:
            raise ValueError("bucket is required when analysis_group is provided")
        return _bucket_analysis_group_mask(labels, bucket, analysis_group)
    if bucket is not None:
        if group is None:
            return torch.ones_like(labels, dtype=torch.bool), None
        return _bucket_analysis_group_mask(labels, bucket, group)
    if _is_analysis_group(group):
        group_name = normalize_analysis_group(group)
        raise ValueError(
            f"group={group_name!r} is a canonical analysis_group; pass bucket=... so tail can be "
            "resolved as medium+few under the bucket protocol"
        )
    return _group_mask(labels, group), None


def _selection_count_details(labels: Tensor, mask: Tensor, analysis_group_name: str | None) -> dict[str, Tensor | str]:
    selected_labels = labels[mask]
    selected_class_ids = torch.unique(selected_labels, sorted=True) if int(selected_labels.numel()) > 0 else labels.new_empty((0,))
    sample_count = int(mask.sum().item())
    effective_class_count = int(selected_class_ids.numel())
    details: dict[str, Tensor | str] = {
        "sample_count": torch.tensor(sample_count, device=labels.device, dtype=torch.long),
        "effective_class_count": torch.tensor(effective_class_count, device=labels.device, dtype=torch.long),
        "selected_class_ids": selected_class_ids,
    }
    if analysis_group_name is not None:
        details["analysis_group"] = analysis_group_name
    if analysis_group_name == "tail":
        details["tail_sample_count"] = torch.tensor(sample_count, device=labels.device, dtype=torch.long)
        details["tail_effective_class_count"] = torch.tensor(effective_class_count, device=labels.device, dtype=torch.long)
    return details


def compute_sample_margin(
    embeddings: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    reduction: Reduction = "none",
    return_details: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Sample-level embedding margin against the nearest non-own class center."""

    samples, labels = _flatten_embeddings(embeddings, label_id, feature_dim=feature_dim)
    centers, class_ids, counts = class_centers(samples, labels, feature_dim=-1, return_counts=True)
    assert isinstance(centers, Tensor)
    assert isinstance(class_ids, Tensor)
    assert isinstance(counts, Tensor)

    if int(class_ids.numel()) < 2:
        sample_margin = samples.new_full((int(samples.shape[0]),), float("nan"))
        result = _nan_reduction(sample_margin, reduction)
        if not return_details:
            return result
        return result, {
            "sample_label_id": labels,
            "class_ids": class_ids,
            "counts": counts,
            "centers": centers,
            "own_center_distance": sample_margin,
            "nearest_rival_distance": sample_margin,
            "nearest_rival_class_id": torch.full_like(labels, -1),
        }

    sorted_class_ids = class_ids.to(device=labels.device)
    class_index = torch.searchsorted(sorted_class_ids, labels)
    if (class_index >= int(sorted_class_ids.numel())).any().item() or not torch.equal(sorted_class_ids[class_index], labels):
        raise RuntimeError("class center index cannot align with label_id")

    distances = torch.cdist(samples, centers.to(device=samples.device), p=2.0)
    own_distance = distances.gather(dim=1, index=class_index[:, None]).squeeze(1)
    rival_distances = distances.clone()
    rival_distances.scatter_(dim=1, index=class_index[:, None], value=float("inf"))
    nearest = rival_distances.min(dim=1)
    nearest_rival_distance = nearest.values
    nearest_rival_class_id = sorted_class_ids[nearest.indices]
    sample_margin = nearest_rival_distance - own_distance
    result = _nan_reduction(sample_margin, reduction)
    if not return_details:
        return result
    return result, {
        "sample_label_id": labels,
        "class_ids": class_ids,
        "counts": counts,
        "centers": centers,
        "own_center_distance": own_distance,
        "nearest_rival_distance": nearest_rival_distance,
        "nearest_rival_class_id": nearest_rival_class_id,
    }


def compute_group_margin(
    embeddings: Any,
    label_id: Any,
    group: Any = None,
    *,
    feature_dim: int = -1,
    reduction: Reduction = "mean",
    bucket: Sequence[Any] | None = None,
    analysis_group: Any | None = None,
    return_details: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Group reduction of sample-level embedding margin.

    Legacy ``group`` accepts label ids or a sample bool mask. For canonical
    bucket protocol calls, pass ``bucket=...`` with either ``analysis_group=...``
    or ``group`` set to many/medium/few/tail; tail is then resolved as medium+few.
    """

    sample_margin, details = compute_sample_margin(
        embeddings,
        label_id,
        feature_dim=feature_dim,
        reduction="none",
        return_details=True,
    )
    assert isinstance(sample_margin, Tensor)
    labels = details["sample_label_id"]
    mask, group_name = _resolve_group_selection_mask(labels, group, bucket=bucket, analysis_group=analysis_group)
    selection_details = _selection_count_details(labels, mask, group_name)
    selection_details["selected_sample_mask"] = mask
    if not mask.any().item():
        result = sample_margin.new_tensor(float("nan"))
    else:
        result = _nan_reduction(sample_margin[mask], reduction)
    return (result, selection_details) if return_details else result


def compute_group_fisher(
    embeddings: Any,
    label_id: Any,
    group: Any = None,
    *,
    feature_dim: int = -1,
    bucket: Sequence[Any] | None = None,
    analysis_group: Any | None = None,
    return_details: bool = False,
    normalize_by_dof: bool = DEFAULT_FISHER_NORMALIZE_BY_DOF,
    eps: float = DEFAULT_EPS,
) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Compute a group Fisher discriminant ratio in the shared eval geometry.

    Legacy ``group`` accepts label ids or a sample bool mask. Bucket-protocol
    calls use ``bucket=...`` plus ``analysis_group=...``; for tail this selects
    medium+few samples before recomputing the group Fisher numerator/denominator.
    """

    samples, labels = _flatten_embeddings(embeddings, label_id, feature_dim=feature_dim)
    selected_mask, group_name = _resolve_group_selection_mask(labels, group, bucket=bucket, analysis_group=analysis_group)
    details = _selection_count_details(labels, selected_mask, group_name)
    details["selected_sample_mask"] = selected_mask

    def finish(value: Tensor) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
        return (value, details) if return_details else value

    if not selected_mask.any().item():
        return finish(samples.new_tensor(float("nan")))

    class_ids = torch.unique(labels, sorted=True)
    details["global_class_count"] = torch.tensor(int(class_ids.numel()), device=labels.device, dtype=torch.long)
    if int(class_ids.numel()) < 2:
        return finish(samples.new_tensor(float("nan")))

    global_mean = samples.mean(dim=0)
    selected_classes = torch.unique(labels[selected_mask], sorted=True)
    num_classes = int(selected_classes.numel())
    num_samples = int(selected_mask.sum().item())
    if num_classes < 2 or (normalize_by_dof and num_samples <= num_classes):
        return finish(samples.new_tensor(float("nan")))

    between = samples.new_tensor(0.0)
    within = samples.new_tensor(0.0)
    for class_id in selected_classes:
        full_class_samples = samples[labels == class_id]
        if int(full_class_samples.numel()) == 0:
            continue
        center = full_class_samples.mean(dim=0)
        group_class_samples = samples[(labels == class_id) & selected_mask]
        if int(group_class_samples.numel()) == 0:
            continue
        between = between + float(group_class_samples.shape[0]) * (center - global_mean).square().sum()
        within = within + (group_class_samples - center).square().sum()

    if normalize_by_dof:
        between = between / float(num_classes - 1)
        within = within / float(num_samples - num_classes)
    details["between_scatter"] = between
    details["within_scatter"] = within
    return finish(between / torch.clamp(within, min=float(eps)))


def class_trace_variance(
    embeddings: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    normalization: VarianceNormalization = DEFAULT_VARIANCE_NORMALIZATION,
    reduction: Reduction = "none",
    return_details: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Per-class within-class trace variance around each class center.

    normalization="dof" uses within_scatter / (n_c - 1) and marks singleton
    classes as NaN. This is the trace of the within-class covariance estimate,
    not per-dimension variance. "population" uses / n_c. "scatter" returns raw
    within-class total squared deviation.
    """

    stats = _class_geometry(embeddings, label_id, feature_dim=feature_dim)
    values, valid = _variance_from_scatter(
        stats["within_scatter"],
        stats["counts"],
        normalization=normalization,
    )
    result = _nan_reduction(values, reduction)
    if not return_details:
        return result

    details: dict[str, Tensor | str] = {
        "class_ids": stats["class_ids"],
        "counts": stats["counts"],
        "centers": stats["centers"],
        "within_scatter": stats["within_scatter"],
        "valid_class_mask": valid,
        "normalization": normalization,
    }
    return result, details


def class_variance(*args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Backward-compatible alias for class_trace_variance."""

    return class_trace_variance(*args, **kwargs)


def trace_variance_inflation(
    fp32_embedding: Any,
    quant_embedding: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    normalization: VarianceNormalization = DEFAULT_VARIANCE_NORMALIZATION,
    reduction: Reduction = "none",
    return_details: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Per-class trace variance inflation: quant trace variance / FP32 trace variance."""

    fp32 = _as_float_tensor(fp32_embedding)
    quant = _as_float_tensor(quant_embedding).to(device=fp32.device)
    _check_same_shape(fp32, quant, name_a="fp32_embedding", name_b="quant_embedding")

    fp32_var, fp32_details = class_trace_variance(
        fp32,
        label_id,
        feature_dim=feature_dim,
        normalization=normalization,
        reduction="none",
        return_details=True,
    )
    quant_var, quant_details = class_trace_variance(
        quant,
        label_id,
        feature_dim=feature_dim,
        normalization=normalization,
        reduction="none",
        return_details=True,
    )
    assert isinstance(fp32_var, Tensor)
    assert isinstance(quant_var, Tensor)

    valid = torch.isfinite(fp32_var) & torch.isfinite(quant_var) & (fp32_var.abs() >= float(eps))
    inflation = quant_var / torch.clamp(fp32_var, min=float(eps))
    inflation = torch.where(valid, inflation, torch.full_like(inflation, float("nan")))
    result = _nan_reduction(inflation, reduction)
    if not return_details:
        return result

    details: dict[str, Tensor | str] = {
        "class_ids": fp32_details["class_ids"],
        "counts": fp32_details["counts"],
        "fp32_class_trace_variance": fp32_var,
        "quant_class_trace_variance": quant_var,
        "valid_class_mask": valid,
        "normalization": normalization,
    }
    return result, details


def variance_inflation(*args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Backward-compatible alias for trace_variance_inflation."""

    return trace_variance_inflation(*args, **kwargs)


def trace_variance_inflation_deviation(
    fp32_embedding: Any,
    quant_embedding: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    normalization: VarianceNormalization = DEFAULT_VARIANCE_NORMALIZATION,
    reduction: Reduction = "none",
    return_details: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Absolute deviation of trace variance inflation from the no-change value 1."""

    inflation = trace_variance_inflation(
        fp32_embedding,
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        normalization=normalization,
        reduction="none",
        return_details=return_details,
        eps=eps,
    )
    if return_details:
        inflation_values, details = inflation
        assert isinstance(details, dict)
    else:
        inflation_values = inflation
        details = {}
    assert isinstance(inflation_values, Tensor)

    deviation_values = torch.where(
        torch.isfinite(inflation_values),
        (inflation_values - 1.0).abs(),
        torch.full_like(inflation_values, float("nan")),
    )
    result = _nan_reduction(deviation_values, reduction)
    if not return_details:
        return result

    details = dict(details)
    details["trace_variance_inflation"] = inflation_values
    details["trace_variance_inflation_deviation"] = deviation_values
    return result, details


def variance_inflation_deviation(*args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Backward-compatible alias for trace_variance_inflation_deviation."""

    return trace_variance_inflation_deviation(*args, **kwargs)


def class_center_shift(
    fp32_embedding: Any,
    quant_embedding: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    mode: CenterDistanceMode = DEFAULT_CENTER_SHIFT_MODE,
    reduction: Reduction = "none",
    return_details: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """Per-class center shift: ||mu_quant,c - mu_fp32,c||_2.

    This is same-class FP32-to-quant center drift, not inter-class center
    distance preservation.
    """

    fp32 = _as_float_tensor(fp32_embedding)
    quant = _as_float_tensor(quant_embedding).to(device=fp32.device)
    _check_same_shape(fp32, quant, name_a="fp32_embedding", name_b="quant_embedding")

    fp32_stats = _class_geometry(fp32, label_id, feature_dim=feature_dim)
    quant_stats = _class_geometry(quant, label_id, feature_dim=feature_dim)
    if not torch.equal(fp32_stats["class_ids"], quant_stats["class_ids"]):
        raise ValueError("FP32 and quant class ids are not aligned")

    fp32_centers = fp32_stats["centers"]
    quant_centers = quant_stats["centers"]
    distance = torch.linalg.vector_norm(quant_centers - fp32_centers, ord=2, dim=1)
    valid = torch.ones_like(distance, dtype=torch.bool)

    if mode == "relative":
        denominator = torch.linalg.vector_norm(fp32_centers, ord=2, dim=1)
        valid = denominator >= float(eps)
        distance = distance / torch.clamp(denominator, min=float(eps))
        distance = torch.where(valid, distance, torch.full_like(distance, float("nan")))
    elif mode != "absolute":
        raise ValueError("mode must be 'absolute' or 'relative'")

    result = _nan_reduction(distance, reduction)
    if not return_details:
        return result

    details: dict[str, Tensor | str] = {
        "class_ids": fp32_stats["class_ids"],
        "counts": fp32_stats["counts"],
        "fp32_centers": fp32_centers,
        "quant_centers": quant_centers,
        "valid_class_mask": valid,
        "mode": mode,
    }
    return result, details


def class_center_distance(*args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, dict[str, Tensor | str]]:
    """已弃用的兼容别名：实际计算 class_center_shift。

    它返回的是同一类别中心从 FP32 到 quant 的漂移 ||mu_quant,c - mu_fp32,c||，
    不是类间中心距离。需要类间中心距离时请使用 pairwise_class_center_distance()。
    """

    warnings.warn(
        "class_center_distance() 实际是 class_center_shift() 的兼容别名；"
        "类间中心距离请使用 pairwise_class_center_distance()。",
        DeprecationWarning,
        stacklevel=2,
    )
    return class_center_shift(*args, **kwargs)


def pairwise_class_center_distance(
    embeddings: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    p: float = 2.0,
    return_details: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Pairwise inter-class center distance matrix for one embedding set."""

    stats = _class_geometry(embeddings, label_id, feature_dim=feature_dim)
    distance = torch.cdist(stats["centers"], stats["centers"], p=float(p))
    if not return_details:
        return distance
    return distance, {
        "class_ids": stats["class_ids"],
        "counts": stats["counts"],
        "centers": stats["centers"],
    }


def pairwise_class_center_distance_retention(
    fp32_embedding: Any,
    quant_embedding: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    p: float = 2.0,
    return_details: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Inter-class center distance retention matrix: D_quant / D_fp32.

    Diagonal entries and near-zero FP32 distances are marked as NaN.
    """

    fp32_distance, fp32_details = pairwise_class_center_distance(
        fp32_embedding,
        label_id,
        feature_dim=feature_dim,
        p=p,
        return_details=True,
    )
    quant_distance, quant_details = pairwise_class_center_distance(
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        p=p,
        return_details=True,
    )
    assert isinstance(fp32_distance, Tensor)
    assert isinstance(quant_distance, Tensor)
    if not torch.equal(fp32_details["class_ids"], quant_details["class_ids"]):
        raise ValueError("FP32 and quant class ids are not aligned")

    valid = fp32_distance >= float(eps)
    retention = quant_distance / torch.clamp(fp32_distance, min=float(eps))
    retention = torch.where(valid, retention, torch.full_like(retention, float("nan")))
    retention.fill_diagonal_(float("nan"))

    if not return_details:
        return retention
    return retention, {
        "class_ids": fp32_details["class_ids"],
        "counts": fp32_details["counts"],
        "fp32_pairwise_class_center_distance": fp32_distance,
        "quant_pairwise_class_center_distance": quant_distance,
        "valid_pair_mask": valid,
    }


def _store_metric_result(summary: dict[str, Any], name: str, value: Any) -> None:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        summary[name] = value[0]
        summary[f"{name}_details"] = value[1]
        return
    summary[name] = value


def geometry_summary(
    fp32_embedding: Any,
    quant_embedding: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    variance_normalization: VarianceNormalization = DEFAULT_VARIANCE_NORMALIZATION,
    center_shift_mode: CenterDistanceMode = DEFAULT_CENTER_SHIFT_MODE,
    center_distance_mode: CenterDistanceMode | None = None,
    return_details: bool = False,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Compute geometry-only metrics for one aligned FP32/quant embedding pair."""

    if center_distance_mode is not None:
        center_shift_mode = center_distance_mode

    summary: dict[str, Any] = {}

    fp32_trace_variance = class_trace_variance(
        fp32_embedding,
        label_id,
        feature_dim=feature_dim,
        normalization=variance_normalization,
        reduction="none",
        return_details=return_details,
    )
    quant_trace_variance = class_trace_variance(
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        normalization=variance_normalization,
        reduction="none",
        return_details=return_details,
    )
    inflation = trace_variance_inflation(
        fp32_embedding,
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        normalization=variance_normalization,
        reduction="none",
        return_details=return_details,
        eps=eps,
    )
    inflation_deviation = trace_variance_inflation_deviation(
        fp32_embedding,
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        normalization=variance_normalization,
        reduction="none",
        return_details=return_details,
        eps=eps,
    )
    center_shift = class_center_shift(
        fp32_embedding,
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        mode=center_shift_mode,
        reduction="none",
        return_details=return_details,
        eps=eps,
    )
    fp32_pairwise_distance = pairwise_class_center_distance(
        fp32_embedding,
        label_id,
        feature_dim=feature_dim,
        return_details=return_details,
    )
    quant_pairwise_distance = pairwise_class_center_distance(
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        return_details=return_details,
    )
    pairwise_distance_retention = pairwise_class_center_distance_retention(
        fp32_embedding,
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        return_details=return_details,
        eps=eps,
    )

    _store_metric_result(summary, "fp32_class_trace_variance", fp32_trace_variance)
    _store_metric_result(summary, "quant_class_trace_variance", quant_trace_variance)
    _store_metric_result(summary, "trace_variance_inflation", inflation)
    _store_metric_result(summary, "trace_variance_inflation_deviation", inflation_deviation)
    _store_metric_result(summary, "class_center_shift", center_shift)
    _store_metric_result(summary, "fp32_pairwise_class_center_distance", fp32_pairwise_distance)
    _store_metric_result(summary, "quant_pairwise_class_center_distance", quant_pairwise_distance)
    _store_metric_result(
        summary,
        "pairwise_class_center_distance_retention",
        pairwise_distance_retention,
    )

    return summary


__all__ = [
    "CenterDistanceMode",
    "DEFAULT_CENTER_SHIFT_MODE",
    "DEFAULT_EPS",
    "DEFAULT_FISHER_NORMALIZE_BY_DOF",
    "DEFAULT_VARIANCE_NORMALIZATION",
    "VarianceNormalization",
    "class_center_shift",
    "class_center_cosine_margin",
    "class_centers",
    "class_trace_variance",
    "class_variance",
    "compute_sample_margin",
    "compute_group_margin",
    "compute_group_fisher",
    "geometry_summary",
    "pairwise_class_center_distance",
    "pairwise_class_center_distance_retention",
    "trace_variance_inflation",
    "trace_variance_inflation_deviation",
    "variance_inflation",
    "variance_inflation_deviation",
]
