from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import torch
from torch import Tensor

try:
    from .metric_config import (
        DEFAULT_EPS,
        DEFAULT_FISHER_NORMALIZE_BY_DOF,
        DEFAULT_MARGIN_MIN_ABS,
        DEFAULT_SCALAR_OER_TOP_RATIO,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    )
except ImportError:  # pragma: no cover - allow direct script-style debugging
    from src.metrics.metric_config import (
        DEFAULT_EPS,
        DEFAULT_FISHER_NORMALIZE_BY_DOF,
        DEFAULT_MARGIN_MIN_ABS,
        DEFAULT_SCALAR_OER_TOP_RATIO,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    )


try:
    from src.utils.group_utils import analysis_group_mask, compute_group_gap, compute_tail_and_few_gaps, normalize_analysis_group
except ImportError:  # pragma: no cover - allow package-relative debugging
    from ..utils.group_utils import analysis_group_mask, compute_group_gap, compute_tail_and_few_gaps, normalize_analysis_group

Reduction = Literal["none", "mean", "sum"]


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


def _as_bool_tensor(value: Any, *, device: torch.device | None = None) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach().bool()
    else:
        result = torch.as_tensor(value, dtype=torch.bool)
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


def _apply_reduction(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.float().mean()
    if reduction == "sum":
        return values.float().sum()
    raise ValueError(f"Unsupported reduction={reduction!r}; use none/mean/sum")


def _safe_positive_denominator(values: Tensor, eps: float) -> Tensor:
    return torch.clamp(values, min=float(eps))


def _finite_mean(values: Tensor) -> Tensor:
    values = values.float()
    finite = torch.isfinite(values)
    if not finite.any().item():
        return values.new_tensor(float("nan"))
    return values[finite].mean()


def _safe_signed_denominator(values: Tensor, eps: float) -> Tensor:
    sign = torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))
    magnitude = torch.clamp(values.abs(), min=float(eps))
    return sign * magnitude


def _flatten_with_batch_dim(x: Tensor, batch_dim: int | None) -> tuple[Tensor, tuple[int, ...]]:
    if batch_dim is None:
        return x.reshape(1, -1), ()

    if x.ndim == 0:
        raise ValueError("batch_dim cannot be used with a scalar tensor")
    batch_dim = _normalize_dim(int(batch_dim), x.ndim)
    moved = x.movedim(batch_dim, 0).contiguous()
    leading_shape = (int(moved.shape[0]),)
    return moved.reshape(moved.shape[0], -1), leading_shape


def embedding_distortion(
    fp32_embedding: Any,
    quant_embedding: Any,
    *,
    dim: int | None = -1,
    reduction: Reduction = "none",
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Relative embedding distortion: ||E_quant - E_fp32|| / ||E_fp32||."""

    fp32 = _as_float_tensor(fp32_embedding)
    quant = _as_float_tensor(quant_embedding).to(device=fp32.device)
    _check_same_shape(fp32, quant, name_a="fp32_embedding", name_b="quant_embedding")

    if dim is None:
        numerator = torch.linalg.vector_norm((quant - fp32).reshape(-1), ord=2)
        denominator = torch.linalg.vector_norm(fp32.reshape(-1), ord=2)
    else:
        dim = _normalize_dim(int(dim), fp32.ndim)
        numerator = torch.linalg.vector_norm(quant - fp32, ord=2, dim=dim)
        denominator = torch.linalg.vector_norm(fp32, ord=2, dim=dim)

    distortion = numerator / _safe_positive_denominator(denominator, eps)
    return _apply_reduction(distortion, reduction)


def metric_drop(
    fp32_value: Any,
    quant_value: Any,
    *,
    reduction: Reduction = "none",
    ignore_nan: bool = True,
) -> Tensor:
    """通用退化量：delta = fp32_value - quant_value。"""

    fp32 = _as_float_tensor(fp32_value)
    quant = _as_float_tensor(quant_value).to(device=fp32.device)
    _check_same_shape(fp32, quant, name_a="fp32_value", name_b="quant_value")
    drop = fp32 - quant
    if not ignore_nan:
        return _apply_reduction(drop, reduction)
    if reduction == "none":
        return drop
    if reduction == "mean":
        return _finite_mean(drop.reshape(-1))
    if reduction == "sum":
        return torch.nan_to_num(drop, nan=0.0).sum()
    raise ValueError(f"Unsupported reduction={reduction!r}; use none/mean/sum")



def _resolve_topk(num_scalars: int, *, topk: int | None, topk_ratio: float | None) -> int:
    if num_scalars <= 0:
        raise ValueError("Cannot compute OER on an empty activation tensor")
    if topk is None:
        ratio = DEFAULT_SCALAR_OER_TOP_RATIO if topk_ratio is None else float(topk_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {ratio}")
        topk = int(math.ceil(float(num_scalars) * ratio))
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    return min(int(topk), int(num_scalars))


def scalar_oer_topk_indices(
    fp32_activation: Any,
    *,
    topk: int | None = None,
    topk_ratio: float | None = DEFAULT_SCALAR_OER_TOP_RATIO,
    batch_dim: int | None = 0,
) -> Tensor:
    """FP32 activation energy top-k scalar indices.

    Indices are computed from FP32 activation only and are meant to be reused
    on quantized activation with identical flattened layout.
    """

    fp32 = _as_float_tensor(fp32_activation)
    flat, leading_shape = _flatten_with_batch_dim(fp32, batch_dim=batch_dim)
    k = _resolve_topk(flat.shape[1], topk=topk, topk_ratio=topk_ratio)
    indices = torch.topk(flat.square(), k=k, dim=1, largest=True, sorted=False).indices
    if batch_dim is None:
        return indices[0]
    return indices.reshape(*leading_shape, k)


def _unravel_flat_indices(indices: Tensor, shape: Sequence[int]) -> tuple[Tensor, ...]:
    if not shape:
        return (indices.new_zeros(indices.shape),)
    coords: list[Tensor] = []
    remainder = indices
    for size in reversed([int(item) for item in shape]):
        coords.append(remainder % size)
        remainder = torch.div(remainder, size, rounding_mode="floor")
    return tuple(reversed(coords))


def scalar_oer_topk_index_metadata(
    fp32_activation: Any,
    *,
    sample_id: Sequence[Any] | None = None,
    layer_idx: int | None = None,
    module_name: str | None = None,
    topk: int | None = None,
    topk_ratio: float | None = DEFAULT_SCALAR_OER_TOP_RATIO,
    topk_indices: Any | None = None,
    batch_dim: int = 0,
    token_dim: int = 1,
    channel_dim: int = -1,
) -> dict[str, Any]:
    """导出 scalar OER oracle index 的可追踪元数据。

    scalar_oer_retention 只需要 flat index；跨层 token survival / danger zone
    分析还必须保存 sample_id、layer_idx、module_name、token_idx、channel_idx。
    这里假定 activation 至少包含 batch/token/channel 三个维度。
    """

    fp32 = _as_float_tensor(fp32_activation)
    if fp32.ndim < 3:
        raise ValueError("scalar_oer_topk_index_metadata 需要至少 [batch, token, channel] 三个维度")
    batch_dim = _normalize_dim(int(batch_dim), fp32.ndim)
    token_dim = _normalize_dim(int(token_dim), fp32.ndim)
    channel_dim = _normalize_dim(int(channel_dim), fp32.ndim)
    if len({batch_dim, token_dim, channel_dim}) != 3:
        raise ValueError("batch_dim、token_dim、channel_dim 必须是三个不同维度")

    moved = fp32.movedim(batch_dim, 0).contiguous()
    batch_size = int(moved.shape[0])
    nonbatch_shape = tuple(int(item) for item in moved.shape[1:])
    remaining_dims = [dim for dim in range(fp32.ndim) if dim != batch_dim]
    token_pos = remaining_dims.index(token_dim)
    channel_pos = remaining_dims.index(channel_dim)

    if topk_indices is None:
        indices = scalar_oer_topk_indices(
            fp32,
            topk=topk,
            topk_ratio=topk_ratio,
            batch_dim=batch_dim,
        ).detach().long()
    else:
        indices = _as_long_tensor(topk_indices, device=fp32.device)
    indices = indices.reshape(batch_size, -1).cpu()
    coords = tuple(coord.cpu() for coord in _unravel_flat_indices(indices, nonbatch_shape))

    sample_index = torch.arange(batch_size, dtype=torch.long).reshape(batch_size, 1).expand_as(indices)
    if sample_id is not None:
        if len(sample_id) != batch_size:
            raise ValueError(f"sample_id 数量必须等于 batch size，当前 {len(sample_id)} vs {batch_size}")
        sample_id_out: Any = [[str(sample_id[row]) for _ in range(indices.shape[1])] for row in range(batch_size)]
    else:
        sample_id_out = None

    return {
        "topk_indices": indices,
        "flat_index": indices,
        "sample_index": sample_index,
        "sample_id": sample_id_out,
        "layer_idx": None if layer_idx is None else int(layer_idx),
        "module_name": None if module_name is None else str(module_name),
        "token_idx": coords[token_pos],
        "channel_idx": coords[channel_pos],
        "coordinate_indices": coords,
        "activation_shape": tuple(int(item) for item in fp32.shape),
        "batch_dim": int(batch_dim),
        "token_dim": int(token_dim),
        "channel_dim": int(channel_dim),
    }


def scalar_oer_retention(
    fp32_activation: Any,
    quant_activation: Any,
    *,
    topk: int | None = None,
    topk_ratio: float | None = DEFAULT_SCALAR_OER_TOP_RATIO,
    topk_indices: Any | None = None,
    batch_dim: int | None = 0,
    reduction: Reduction = "none",
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """FFN scalar-level OER retention.

    Formula:
        FP32 FFN input activation -> FP32 top-k scalar indices ->
        track the same indices in quant activation ->
        sum(quant[index]^2) / sum(fp32[index]^2)
    """

    fp32 = _as_float_tensor(fp32_activation)
    quant = _as_float_tensor(quant_activation).to(device=fp32.device)
    _check_same_shape(fp32, quant, name_a="fp32_activation", name_b="quant_activation")

    fp32_flat, leading_shape = _flatten_with_batch_dim(fp32, batch_dim=batch_dim)
    quant_flat, _ = _flatten_with_batch_dim(quant, batch_dim=batch_dim)

    if topk_indices is None:
        k = _resolve_topk(fp32_flat.shape[1], topk=topk, topk_ratio=topk_ratio)
        indices = torch.topk(fp32_flat.square(), k=k, dim=1, largest=True, sorted=False).indices
    else:
        indices = _as_long_tensor(topk_indices, device=fp32.device)
        if batch_dim is None:
            indices = indices.reshape(1, -1)
        else:
            indices = indices.reshape(fp32_flat.shape[0], -1)
        if indices.numel() == 0:
            raise ValueError("topk_indices must not be empty")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(fp32_flat.shape[1]):
            raise ValueError("topk_indices contains out-of-range indices")

    fp32_selected = torch.gather(fp32_flat, dim=1, index=indices)
    quant_selected = torch.gather(quant_flat, dim=1, index=indices)
    fp32_energy = fp32_selected.square().sum(dim=1)
    quant_energy = quant_selected.square().sum(dim=1)
    retention = quant_energy / _safe_positive_denominator(fp32_energy, eps)

    if batch_dim is None:
        retention = retention[0]
    else:
        retention = retention.reshape(*leading_shape)
    return _apply_reduction(retention, reduction)


def scalar_oer_retention_loss(
    fp32_activation: Any,
    quant_activation: Any,
    *,
    topk: int | None = None,
    topk_ratio: float | None = DEFAULT_SCALAR_OER_TOP_RATIO,
    topk_indices: Any | None = None,
    batch_dim: int | None = 0,
    reduction: Reduction = "none",
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """OER loss for heatmap/correlation: 1 - scalar_oer_retention."""

    retention = scalar_oer_retention(
        fp32_activation,
        quant_activation,
        topk=topk,
        topk_ratio=topk_ratio,
        topk_indices=topk_indices,
        batch_dim=batch_dim,
        reduction="none",
        eps=eps,
    )
    loss = 1.0 - retention
    return _apply_reduction(loss, reduction)


def scalar_oer_energy_metrics(
    fp32_activation: Any,
    quant_activation: Any,
    *,
    topk: int | None = None,
    topk_ratio: float | None = DEFAULT_SCALAR_OER_TOP_RATIO,
    topk_indices: Any | None = None,
    batch_dim: int | None = 0,
    reduction: Reduction = "mean",
    eps: float = DEFAULT_EPS,
) -> dict[str, Tensor]:
    """基于 FP32 oracle 标量位置计算 OER 相关指标。

    返回：
        oer_retention: sum(quant_oracle^2) / sum(fp32_oracle^2)
        oer_loss: 1 - oer_retention
        off_oracle_energy_ratio: quant 非 oracle 能量 / quant 总能量
    """

    fp32 = _as_float_tensor(fp32_activation)
    quant = _as_float_tensor(quant_activation).to(device=fp32.device)
    _check_same_shape(fp32, quant, name_a="fp32_activation", name_b="quant_activation")

    fp32_flat, leading_shape = _flatten_with_batch_dim(fp32, batch_dim=batch_dim)
    quant_flat, _ = _flatten_with_batch_dim(quant, batch_dim=batch_dim)
    batch_size = int(fp32_flat.shape[0])
    num_scalars = int(fp32_flat.shape[1])

    if topk_indices is None:
        k = _resolve_topk(num_scalars, topk=topk, topk_ratio=topk_ratio)
        indices = torch.topk(fp32_flat.square(), k=k, dim=1, largest=True, sorted=False).indices
    else:
        indices = _as_long_tensor(topk_indices, device=fp32.device)
        if batch_dim is None:
            indices = indices.reshape(1, -1)
        else:
            indices = indices.reshape(batch_size, -1)
        if int(indices.numel()) == 0:
            raise ValueError("topk_indices must not be empty")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= num_scalars:
            raise ValueError("topk_indices contains out-of-range indices")

    fp32_oracle = torch.gather(fp32_flat, dim=1, index=indices)
    quant_oracle = torch.gather(quant_flat, dim=1, index=indices)
    fp32_oracle_energy = fp32_oracle.square().sum(dim=1)
    quant_oracle_energy = quant_oracle.square().sum(dim=1)
    retention = quant_oracle_energy / torch.clamp(fp32_oracle_energy, min=float(eps))
    retention = torch.where(
        fp32_oracle_energy > float(eps),
        retention,
        torch.full_like(retention, float("nan")),
    )

    oracle_mask = torch.zeros_like(quant_flat, dtype=torch.bool)
    oracle_mask.scatter_(dim=1, index=indices, src=torch.ones_like(indices, dtype=torch.bool))
    quant_total_energy = quant_flat.square().sum(dim=1)
    quant_off_oracle_energy = quant_flat.masked_fill(oracle_mask, 0.0).square().sum(dim=1)
    off_oracle_ratio = quant_off_oracle_energy / torch.clamp(quant_total_energy, min=float(eps))
    off_oracle_ratio = torch.where(
        quant_total_energy > float(eps),
        off_oracle_ratio,
        torch.full_like(off_oracle_ratio, float("nan")),
    )

    if batch_dim is not None:
        retention = retention.reshape(*leading_shape)
        off_oracle_ratio = off_oracle_ratio.reshape(*leading_shape)
    else:
        retention = retention[0]
        off_oracle_ratio = off_oracle_ratio[0]

    if reduction == "none":
        oer_retention = retention
        off_oracle_energy_ratio = off_oracle_ratio
    elif reduction == "mean":
        oer_retention = _finite_mean(retention.reshape(-1))
        off_oracle_energy_ratio = _finite_mean(off_oracle_ratio.reshape(-1))
    elif reduction == "sum":
        oer_retention = torch.nan_to_num(retention, nan=0.0).sum()
        off_oracle_energy_ratio = torch.nan_to_num(off_oracle_ratio, nan=0.0).sum()
    else:
        raise ValueError(f"Unsupported reduction={reduction!r}; use none/mean/sum")

    return {
        "oer_retention": oer_retention,
        "oer_loss": 1.0 - oer_retention,
        "off_oracle_energy_ratio": off_oracle_energy_ratio,
    }



def margin_retention(
    fp32_logits: Any,
    quant_logits: Any,
    *,
    label_id: Any | None = None,
    class_dim: int = -1,
    reduction: Reduction = "none",
    min_abs_margin: float = DEFAULT_MARGIN_MIN_ABS,
    return_diagnostics: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Class-margin retention on a fixed FP32 class pair.

    If label_id is provided, margin = logit[label] - max_other_logit, where the
    "other" class is chosen from FP32 logits and reused for quant logits.
    Otherwise, the FP32 top-1 and top-2 classes define the margin pair.

    只有 FP32 margin 为正且大于等于 min_abs_margin 的样本才有“保持率”语义。
    FP32 负 margin 表示 baseline 已经更靠近竞争类；near-zero margin 表示边界不稳定。
    这些样本会在 reduction="none" 中变成 NaN，并从 mean/sum 中排除。
    """

    fp32 = _as_float_tensor(fp32_logits)
    quant = _as_float_tensor(quant_logits).to(device=fp32.device)
    _check_same_shape(fp32, quant, name_a="fp32_logits", name_b="quant_logits")
    if float(min_abs_margin) < 0.0:
        raise ValueError(f"min_abs_margin must be non-negative, got {min_abs_margin}")

    if fp32.ndim == 0:
        raise ValueError("logits must have a class dimension")
    class_dim = _normalize_dim(int(class_dim), fp32.ndim)
    fp32_last = fp32.movedim(class_dim, -1).contiguous()
    quant_last = quant.movedim(class_dim, -1).contiguous()
    leading_shape = tuple(fp32_last.shape[:-1])
    num_classes = int(fp32_last.shape[-1])
    if num_classes < 2:
        raise ValueError("margin_retention requires at least two classes")

    fp32_flat = fp32_last.reshape(-1, num_classes)
    quant_flat = quant_last.reshape(-1, num_classes)

    if label_id is None:
        pair = torch.topk(fp32_flat, k=2, dim=1, largest=True, sorted=True).indices
        positive_idx = pair[:, 0]
        negative_idx = pair[:, 1]
    else:
        positive_idx = _as_long_tensor(label_id, device=fp32.device).reshape(-1)
        if positive_idx.numel() == 1 and fp32_flat.shape[0] != 1:
            positive_idx = positive_idx.expand(fp32_flat.shape[0])
        if positive_idx.numel() != fp32_flat.shape[0]:
            raise ValueError(
                "label_id must be scalar or match the non-class logits shape, "
                f"got {positive_idx.numel()} labels for {fp32_flat.shape[0]} rows"
            )
        if int(positive_idx.min().item()) < 0 or int(positive_idx.max().item()) >= num_classes:
            raise ValueError("label_id contains out-of-range class ids")
        masked = fp32_flat.clone()
        masked.scatter_(1, positive_idx.unsqueeze(1), -torch.inf)
        negative_idx = masked.argmax(dim=1)

    row_idx = torch.arange(fp32_flat.shape[0], device=fp32.device)
    fp32_margin = fp32_flat[row_idx, positive_idx] - fp32_flat[row_idx, negative_idx]
    quant_margin = quant_flat[row_idx, positive_idx] - quant_flat[row_idx, negative_idx]
    finite_fp32_margin = torch.isfinite(fp32_margin)
    finite_quant_margin = torch.isfinite(quant_margin)
    valid_margin_mask = finite_fp32_margin & finite_quant_margin & (fp32_margin > float(min_abs_margin))
    near_zero_margin_mask = finite_fp32_margin & (fp32_margin > 0.0) & (fp32_margin <= float(min_abs_margin))
    negative_margin_mask = finite_fp32_margin & (fp32_margin < 0)

    candidate_retention = quant_margin / _safe_positive_denominator(fp32_margin, eps)
    retention_flat = torch.where(
        valid_margin_mask,
        candidate_retention,
        torch.full_like(candidate_retention, float("nan")),
    )

    if leading_shape:
        retention = retention_flat.reshape(*leading_shape)
        valid_mask = valid_margin_mask.reshape(*leading_shape)
        near_zero_mask = near_zero_margin_mask.reshape(*leading_shape)
        negative_mask = negative_margin_mask.reshape(*leading_shape)
        fp32_margin_out = fp32_margin.reshape(*leading_shape)
        quant_margin_out = quant_margin.reshape(*leading_shape)
        positive_idx_out = positive_idx.reshape(*leading_shape)
        negative_idx_out = negative_idx.reshape(*leading_shape)
    else:
        retention = retention_flat[0]
        valid_mask = valid_margin_mask[0]
        near_zero_mask = near_zero_margin_mask[0]
        negative_mask = negative_margin_mask[0]
        fp32_margin_out = fp32_margin[0]
        quant_margin_out = quant_margin[0]
        positive_idx_out = positive_idx[0]
        negative_idx_out = negative_idx[0]

    if reduction == "none":
        result = retention
    elif reduction == "mean":
        denominator = valid_margin_mask.float().sum()
        numerator = torch.nan_to_num(retention_flat, nan=0.0).sum()
        result = torch.where(
            denominator > 0,
            numerator / torch.clamp(denominator, min=1.0),
            retention_flat.new_tensor(float("nan")),
        )
    elif reduction == "sum":
        result = torch.nan_to_num(retention_flat, nan=0.0).sum()
    else:
        raise ValueError(f"Unsupported reduction={reduction!r}; use none/mean/sum")

    if not return_diagnostics:
        return result

    total_count = int(valid_margin_mask.numel())
    valid_count = valid_margin_mask.reshape(-1).sum()
    invalid_count = (~valid_margin_mask).reshape(-1).sum()
    total_count_tensor = fp32_margin.new_tensor(float(total_count))
    valid_rate = (
        valid_count.float() / torch.clamp(total_count_tensor, min=1.0)
        if total_count > 0
        else fp32_margin.new_tensor(float("nan"))
    )
    invalid_rate = (
        invalid_count.float() / torch.clamp(total_count_tensor, min=1.0)
        if total_count > 0
        else fp32_margin.new_tensor(float("nan"))
    )
    diagnostics: dict[str, Any] = {
        "valid_margin_mask": valid_mask,
        "near_zero_margin_mask": near_zero_mask,
        "negative_margin_mask": negative_mask,
        "fp32_margin": fp32_margin_out,
        "quant_margin": quant_margin_out,
        "positive_class": positive_idx_out,
        "negative_class": negative_idx_out,
        "min_abs_margin": float(min_abs_margin),
        "valid_count": valid_count,
        "total_count": total_count_tensor,
        "valid_rate": valid_rate,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_rate,
        "num_valid": valid_count,
        "num_near_zero": near_zero_margin_mask.sum(),
        "num_negative": negative_margin_mask.sum(),
        "num_invalid": invalid_count,
        "near_zero_rate": (
            near_zero_margin_mask.float().mean() if total_count > 0 else fp32_margin.new_tensor(float("nan"))
        ),
        "negative_margin_rate": (
            negative_margin_mask.float().mean() if total_count > 0 else fp32_margin.new_tensor(float("nan"))
        ),
    }
    return result, diagnostics


def margin_value_retention(
    fp32_margin: Any,
    quant_margin: Any,
    *,
    reduction: Reduction = "mean",
    min_abs_margin: float = DEFAULT_MARGIN_MIN_ABS,
    return_diagnostics: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """对已经计算好的几何 margin 值计算保持率。

    只把 FP32 margin 为正且大于 min_abs_margin 的样本视为有效样本；
    非正 margin 和 near-zero margin 没有保持率语义，会被标记为 NaN。
    """

    fp32 = _as_float_tensor(fp32_margin)
    quant = _as_float_tensor(quant_margin).to(device=fp32.device)
    _check_same_shape(fp32, quant, name_a="fp32_margin", name_b="quant_margin")
    if float(min_abs_margin) < 0.0:
        raise ValueError(f"min_abs_margin must be non-negative, got {min_abs_margin}")

    finite_fp32 = torch.isfinite(fp32)
    finite_quant = torch.isfinite(quant)
    valid = finite_fp32 & finite_quant & (fp32 > float(min_abs_margin))
    nonpositive = finite_fp32 & (fp32 <= 0)
    near_threshold = finite_fp32 & (fp32 > 0) & (fp32 <= float(min_abs_margin))
    retention = quant / _safe_positive_denominator(fp32, eps)
    retention = torch.where(valid, retention, torch.full_like(retention, float("nan")))

    if reduction == "none":
        result = retention
    elif reduction == "mean":
        result = _finite_mean(retention.reshape(-1))
    elif reduction == "sum":
        result = torch.nan_to_num(retention, nan=0.0).sum()
    else:
        raise ValueError(f"Unsupported reduction={reduction!r}; use none/mean/sum")

    if not return_diagnostics:
        return result

    invalid = ~valid
    total_count = int(valid.numel())
    valid_count = valid.reshape(-1).sum()
    invalid_count = invalid.reshape(-1).sum()
    total_count_tensor = fp32.new_tensor(float(total_count))
    valid_rate = (
        valid_count.float() / torch.clamp(total_count_tensor, min=1.0)
        if total_count > 0
        else fp32.new_tensor(float("nan"))
    )
    invalid_rate = (
        invalid_count.float() / torch.clamp(total_count_tensor, min=1.0)
        if total_count > 0
        else fp32.new_tensor(float("nan"))
    )
    diagnostics: dict[str, Any] = {
        "valid_margin_mask": valid,
        "invalid_margin_mask": invalid,
        "nonpositive_margin_mask": nonpositive,
        "near_threshold_margin_mask": near_threshold,
        "valid_count": valid_count,
        "total_count": total_count_tensor,
        "valid_rate": valid_rate,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_rate,
        "num_valid": valid_count,
        "num_invalid": invalid_count,
        "num_nonpositive": nonpositive.reshape(-1).sum(),
        "num_near_threshold": near_threshold.reshape(-1).sum(),
        "invalid_ratio": invalid_rate,
        "nonpositive_ratio": nonpositive.reshape(-1).float().mean() if nonpositive.numel() else fp32.new_tensor(float("nan")),
        "near_threshold_ratio": near_threshold.reshape(-1).float().mean() if near_threshold.numel() else fp32.new_tensor(float("nan")),
        "min_abs_margin": float(min_abs_margin),
    }
    return result, diagnostics


def fisher_ratio(
    embeddings: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    normalize_by_dof: bool = DEFAULT_FISHER_NORMALIZE_BY_DOF,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Fisher discriminant ratio.

    Default is the degree-of-freedom normalized covariance ratio:
        (between / (C - 1)) / (within / (N - C))

    Set normalize_by_dof=False to recover the raw scatter ratio:
        between / within
    """

    x = _as_float_tensor(embeddings)
    if x.ndim < 2:
        raise ValueError("embeddings must have at least sample and feature dimensions")
    feature_dim = _normalize_dim(int(feature_dim), x.ndim)
    x_last = x.movedim(feature_dim, -1).contiguous()
    feature_size = int(x_last.shape[-1])
    samples = x_last.reshape(-1, feature_size)
    labels = _as_long_tensor(label_id, device=samples.device).reshape(-1)
    if labels.numel() != samples.shape[0]:
        raise ValueError(
            f"label_id must provide one label per sample, got {labels.numel()} "
            f"labels for {samples.shape[0]} samples"
        )

    classes = torch.unique(labels, sorted=True)
    if classes.numel() < 2:
        raise ValueError("fisher_ratio requires at least two classes")
    num_samples = int(samples.shape[0])
    num_classes = int(classes.numel())
    if normalize_by_dof and num_samples <= num_classes:
        raise ValueError(
            "degree-of-freedom normalized fisher_ratio requires N > C, "
            f"got N={num_samples}, C={num_classes}"
        )

    global_mean = samples.mean(dim=0)
    between = samples.new_tensor(0.0)
    within = samples.new_tensor(0.0)

    for cls in classes:
        mask = labels == cls
        part = samples[mask]
        if part.numel() == 0:
            continue
        class_mean = part.mean(dim=0)
        between = between + float(part.shape[0]) * (class_mean - global_mean).square().sum()
        within = within + (part - class_mean).square().sum()

    if normalize_by_dof:
        between = between / float(num_classes - 1)
        within = within / float(num_samples - num_classes)

    return between / _safe_positive_denominator(within, eps)


def fisher_retention(
    fp32_embedding: Any,
    quant_embedding: Any,
    label_id: Any,
    *,
    feature_dim: int = -1,
    normalize_by_dof: bool = DEFAULT_FISHER_NORMALIZE_BY_DOF,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Fisher discriminant ratio retention: Fisher_quant / Fisher_fp32."""

    fp32_ratio = fisher_ratio(
        fp32_embedding,
        label_id,
        feature_dim=feature_dim,
        normalize_by_dof=normalize_by_dof,
        eps=eps,
    )
    quant_ratio = fisher_ratio(
        quant_embedding,
        label_id,
        feature_dim=feature_dim,
        normalize_by_dof=normalize_by_dof,
        eps=eps,
    )
    return quant_ratio / _safe_positive_denominator(fp32_ratio, eps)


def collapse_gap(
    metric_values: Any,
    bucket: Sequence[Any],
    *,
    many_label: str = "many",
    comparison_label: str = "tail",
    few_label: str | None = None,
    higher_is_better: bool = True,
) -> Tensor:
    """Primary group damage gap using tail-vs-many under protocol v2.

    ``few_label`` is kept as a legacy override for callers that explicitly ask
    for the old few-vs-many comparison. The default comparison is now tail,
    which means raw medium+few records are selected together by
    ``analysis_group_mask`` instead of averaging medium and few summaries.
    """

    values = _as_float_tensor(metric_values).reshape(-1)
    labels = [str(item) for item in bucket]
    if len(labels) != int(values.numel()):
        raise ValueError(f"bucket length={len(labels)} must match values={int(values.numel())}")

    baseline_group = normalize_analysis_group(many_label)
    selected_comparison = few_label if few_label is not None else comparison_label
    comparison_group = normalize_analysis_group(selected_comparison)
    group_values: dict[str, float] = {}
    for group in (baseline_group, comparison_group):
        mask = torch.as_tensor(analysis_group_mask(labels, group), dtype=torch.bool, device=values.device)
        if not mask.any().item():
            raise ValueError(f"bucket contains no analysis_group={group!r}")
        group_values[group] = float(_finite_mean(values[mask]).item())

    metric_direction = "higher_is_better" if higher_is_better else "lower_is_better"
    gap = compute_group_gap(
        group_values,
        baseline_group=baseline_group,
        comparison_group=comparison_group,
        metric_direction=metric_direction,
    )
    return values.new_tensor(float(gap))


def tail_and_few_damage_gaps(
    metric_values: Any,
    bucket: Sequence[Any],
    *,
    higher_is_better: bool = True,
) -> dict[str, Tensor]:
    """Return protocol-v2 group values plus tail/few damage gaps.

    Tail is recomputed from raw samples whose canonical bucket is medium or few;
    it is not the mean of medium_value and few_value.
    """

    values = _as_float_tensor(metric_values).reshape(-1)
    labels = [str(item) for item in bucket]
    if len(labels) != int(values.numel()):
        raise ValueError(f"bucket length={len(labels)} must match values={int(values.numel())}")

    group_values: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    for group in ("many", "medium", "few", "tail"):
        mask = torch.as_tensor(analysis_group_mask(labels, group), dtype=torch.bool, device=values.device)
        group_counts[group] = int(mask.sum().item())
        group_values[group] = float(_finite_mean(values[mask]).item()) if mask.any().item() else float("nan")

    metric_direction = "higher_is_better" if higher_is_better else "lower_is_better"
    gaps = compute_tail_and_few_gaps(group_values, metric_direction=metric_direction)
    result = {f"{group}_value": values.new_tensor(value) for group, value in group_values.items()}
    result.update({f"{group}_count": values.new_tensor(float(count)) for group, count in group_counts.items()})
    result.update({name: values.new_tensor(float(value)) for name, value in gaps.items()})
    return result

def _as_qparam_tensor(value: Any, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _reshape_channel_qparam(param: Tensor, values: Tensor, channel_axis: int | None) -> Tensor:
    if channel_axis is None or param.ndim != 1:
        return param
    channel_axis = _normalize_dim(int(channel_axis), values.ndim)
    if int(param.numel()) != int(values.shape[channel_axis]):
        raise ValueError(
            f"per-channel qparam length={int(param.numel())} does not match "
            f"values.shape[{channel_axis}]={int(values.shape[channel_axis])}"
        )
    shape = [1] * values.ndim
    shape[channel_axis] = int(param.numel())
    return param.reshape(shape)


def _validate_qparams(scale: Tensor, zero_point: Tensor) -> None:
    if not torch.isfinite(scale).all().item():
        raise ValueError("scale must be finite for clipping_rate")
    if not (scale > 0).all().item():
        raise ValueError("scale must be strictly positive for clipping_rate")
    if not torch.isfinite(zero_point).all().item():
        raise ValueError("zero_point must be finite for clipping_rate")


def _normalize_reduce_dims(reduce_dims: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if reduce_dims is None:
        return tuple(range(ndim))
    if isinstance(reduce_dims, int):
        dims = (reduce_dims,)
    else:
        dims = tuple(int(dim) for dim in reduce_dims)
    return tuple(sorted({_normalize_dim(dim, ndim) for dim in dims}))


def clipping_rate(
    values: Any,
    *,
    scale: Any,
    zero_point: Any = 0.0,
    qmin: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    valid_mask: Any | None = None,
    nonzero_only: bool = False,
    channel_axis: int | None = None,
    reduce_dims: int | Sequence[int] | None = None,
    reduction: Reduction = "none",
) -> Tensor:
    """Quant-domain clipping rate.

    A value is clipped if value / scale + zero_point is outside [qmin, qmax].
    Defaults follow the project symmetric signed INT4 effective codebook [-7, 7].
    valid_mask, when provided, must be True for entries included in the rate.
    """

    x = _as_float_tensor(values)
    if qmin >= qmax:
        raise ValueError(f"qmin must be smaller than qmax, got {qmin}, {qmax}")

    scale_t = _reshape_channel_qparam(_as_qparam_tensor(scale, x), x, channel_axis)
    zero_point_t = _reshape_channel_qparam(_as_qparam_tensor(zero_point, x), x, channel_axis)
    _validate_qparams(scale_t, zero_point_t)
    q = x / scale_t + zero_point_t
    clipped = (q < float(qmin)) | (q > float(qmax))

    included = torch.ones_like(clipped, dtype=torch.bool)
    if valid_mask is not None:
        mask = _as_bool_tensor(valid_mask, device=x.device)
        if tuple(mask.shape) != tuple(x.shape):
            raise ValueError(f"valid_mask shape {tuple(mask.shape)} must match values {tuple(x.shape)}")
        included = included & mask
    if nonzero_only:
        included = included & (x != 0)

    dims = _normalize_reduce_dims(reduce_dims, x.ndim)
    numerator = (clipped & included).float().sum(dim=dims)
    denominator = included.float().sum(dim=dims)
    rate = numerator / torch.clamp(denominator, min=1.0)
    rate = torch.where(denominator > 0, rate, torch.zeros_like(rate))
    return _apply_reduction(rate, reduction)


def activation_clipping_rate(
    activation: Any,
    *,
    scale: Any,
    zero_point: Any = 0.0,
    qmin: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    valid_mask: Any | None = None,
    effective_nonzero: bool = False,
    reduction: Reduction = "none",
) -> Tensor:
    """Activation clipping rate for fixed calibration qparams."""

    return clipping_rate(
        activation,
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        valid_mask=valid_mask,
        nonzero_only=effective_nonzero,
        channel_axis=None,
        reduce_dims=None,
        reduction=reduction,
    )


def _symmetric_per_channel_weight_scale(
    weight: Tensor,
    *,
    qmax: int,
    channel_axis: int,
    eps: float,
) -> Tensor:
    channel_axis = _normalize_dim(int(channel_axis), weight.ndim)
    reduce_dims = tuple(dim for dim in range(weight.ndim) if dim != channel_axis)
    max_abs = weight.abs().amax(dim=reduce_dims)
    return torch.clamp(max_abs / float(max(1, qmax)), min=float(eps))


def weight_clipping_rate(
    weight: Any,
    *,
    scale: Any | None = None,
    zero_point: Any = 0.0,
    qmin: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    channel_axis: int = 0,
    effective_nonzero: bool = False,
    allow_dynamic_scale: bool = False,
    reduction: Reduction = "none",
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Weight clipping rate; default channel axis is out_features.

    默认要求传入真实量化模块使用的 scale。若 scale=None，会报错；
    只有显式 allow_dynamic_scale=True 时才用当前 weight 自身范围临时构造 scale。
    """

    w = _as_float_tensor(weight)
    if scale is None:
        if not allow_dynamic_scale:
            raise ValueError(
                "weight_clipping_rate requires the actual quantizer scale. "
                "Pass scale explicitly, or set allow_dynamic_scale=True only for diagnostic self-range checks."
            )
        scale = _symmetric_per_channel_weight_scale(
            w,
            qmax=int(qmax),
            channel_axis=int(channel_axis),
            eps=float(eps),
        )
    reduce_dims = tuple(dim for dim in range(w.ndim) if dim != _normalize_dim(int(channel_axis), w.ndim))
    return clipping_rate(
        w,
        scale=scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        valid_mask=None,
        nonzero_only=effective_nonzero,
        channel_axis=channel_axis,
        reduce_dims=reduce_dims,
        reduction=reduction,
    )


def _normalize_percentile_value(percentile: float) -> float:
    q = float(percentile)
    if q > 1.0:
        q = q / 100.0
    if not (0.0 < q <= 1.0):
        raise ValueError(f"percentile must be in (0, 1] or (0, 100], got {percentile}")
    return q


def symmetric_percentile_scale(
    values: Any,
    *,
    percentile: float = 0.999,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    reduce_dims: Sequence[int] | None = None,
    keepdim: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """对称量化 percentile scale 的公共实现。

    scale = percentile(abs(values)) / qmax。reduce_dims=None 表示全张量
    per-tensor scale；传入 reduce_dims 时保留未 reduce 的维度，例如
    [B,T,C] 上 reduce_dims=(0,1) 得到 [C]。
    """

    x = _as_float_tensor(values)
    if int(x.numel()) <= 0:
        raise ValueError("values must not be empty")
    if not torch.isfinite(x).all().item():
        raise ValueError("values contains NaN or Inf")
    if int(qmax) <= 0:
        raise ValueError(f"qmax must be positive for symmetric scale, got {qmax}")
    q = _normalize_percentile_value(percentile)
    abs_x = x.abs()

    if reduce_dims is None:
        scale = torch.quantile(abs_x.reshape(-1), q) / float(qmax)
        if keepdim:
            scale = scale.reshape([1] * x.ndim)
        return torch.clamp(scale, min=float(eps))

    dims = tuple(sorted({_normalize_dim(int(dim), x.ndim) for dim in reduce_dims}))
    if not dims:
        return torch.clamp(abs_x / float(qmax), min=float(eps))
    keep_dims = tuple(dim for dim in range(x.ndim) if dim not in dims)
    if not keep_dims:
        scale = torch.quantile(abs_x.reshape(-1), q) / float(qmax)
        if keepdim:
            scale = scale.reshape([1] * x.ndim)
        return torch.clamp(scale, min=float(eps))

    permuted = abs_x.permute(*keep_dims, *dims).contiguous()
    keep_shape = tuple(int(abs_x.shape[dim]) for dim in keep_dims)
    flattened = permuted.reshape(*keep_shape, -1)
    scale = torch.quantile(flattened, q, dim=-1) / float(qmax)
    scale = torch.clamp(scale, min=float(eps))
    if not keepdim:
        return scale

    out_shape = [1] * x.ndim
    for size, dim in zip(scale.shape, keep_dims):
        out_shape[dim] = int(size)
    return scale.reshape(out_shape)


def boolean_mask_rate(
    mask: Any,
    *,
    valid_mask: Any | None = None,
    reduce_dims: Sequence[int] | None = None,
    per_sample: bool = False,
    reduction: Reduction = "mean",
) -> Tensor:
    """布尔 mask 比例的公共实现。

    per_sample=True 时默认保留第 0 维，把其余维度求比例；否则按
    reduce_dims 或全张量求比例，再应用 reduction。
    """

    mask_t = _as_bool_tensor(mask)
    included = torch.ones_like(mask_t, dtype=torch.bool)
    if valid_mask is not None:
        valid = _as_bool_tensor(valid_mask, device=mask_t.device)
        if tuple(valid.shape) != tuple(mask_t.shape):
            while valid.ndim < mask_t.ndim:
                valid = valid.unsqueeze(-1)
            if tuple(valid.shape) != tuple(mask_t.shape):
                valid = valid.expand_as(mask_t)
        included = included & valid

    if per_sample:
        if mask_t.ndim == 0:
            return (mask_t & included).float()
        flat_mask = (mask_t & included).float().reshape(mask_t.shape[0], -1)
        flat_included = included.float().reshape(included.shape[0], -1)
        denom = flat_included.sum(dim=1)
        rate = flat_mask.sum(dim=1) / torch.clamp(denom, min=1.0)
        return torch.where(denom > 0, rate, torch.full_like(rate, float("nan")))

    dims = tuple(range(mask_t.ndim)) if reduce_dims is None else _normalize_reduce_dims(reduce_dims, mask_t.ndim)
    numerator = (mask_t & included).float().sum(dim=dims)
    denominator = included.float().sum(dim=dims)
    rate = numerator / torch.clamp(denominator, min=1.0)
    rate = torch.where(denominator > 0, rate, torch.full_like(rate, float("nan")))
    return _apply_reduction(rate, reduction)


def token_energy_topk_indices(
    activation: Any,
    *,
    topk: int | None = None,
    topk_ratio: float | None = DEFAULT_SCALAR_OER_TOP_RATIO,
    batch_dim: int = 0,
    token_dim: int = 1,
    channel_dim: int = -1,
) -> Tensor:
    """按 token energy 选择每个样本的 top-k token index。"""

    x = _as_float_tensor(activation)
    if x.ndim < 3:
        raise ValueError("activation must contain batch/token/channel dimensions")
    batch_dim = _normalize_dim(int(batch_dim), x.ndim)
    token_dim = _normalize_dim(int(token_dim), x.ndim)
    channel_dim = _normalize_dim(int(channel_dim), x.ndim)
    if len({batch_dim, token_dim, channel_dim}) != 3:
        raise ValueError("batch_dim, token_dim, and channel_dim must be different")

    moved = x.movedim((batch_dim, token_dim, channel_dim), (0, 1, 2)).contiguous()
    token_energy = moved.square().sum(dim=tuple(range(2, moved.ndim)))
    k = _resolve_topk(int(token_energy.shape[1]), topk=topk, topk_ratio=topk_ratio)
    return torch.topk(token_energy, k=k, dim=1, largest=True, sorted=False).indices


def gather_token_values(
    values: Any,
    token_indices: Any,
    *,
    batch_dim: int = 0,
    token_dim: int = 1,
) -> Tensor:
    """按每个样本的 token index 取出 token 级张量切片。"""

    if isinstance(values, Tensor) and values.dtype == torch.bool:
        x = values.detach()
    else:
        x = _as_float_tensor(values)
    if x.ndim < 2:
        raise ValueError("values must contain batch and token dimensions")
    batch_dim = _normalize_dim(int(batch_dim), x.ndim)
    token_dim = _normalize_dim(int(token_dim), x.ndim)
    moved = x.movedim((batch_dim, token_dim), (0, 1)).contiguous()
    indices = _as_long_tensor(token_indices, device=moved.device).reshape(int(moved.shape[0]), -1)
    if int(indices.numel()) == 0:
        raise ValueError("token_indices must not be empty")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(moved.shape[1]):
        raise ValueError("token_indices contains out-of-range token ids")
    gather_shape = [int(indices.shape[0]), int(indices.shape[1]), *[int(item) for item in moved.shape[2:]]]
    gather_index = indices.reshape(indices.shape[0], indices.shape[1], *([1] * (moved.ndim - 2))).expand(gather_shape)
    return torch.gather(moved, dim=1, index=gather_index)


def group_mean_amplification(
    metric_values: Any,
    group_labels: Sequence[Any],
    *,
    numerator_label: str = "tail",
    denominator_label: str = "many",
    eps: float = DEFAULT_EPS,
) -> dict[str, Tensor]:
    """计算一个 analysis group 相对另一个 analysis group 的均值放大系数。"""

    values = _as_float_tensor(metric_values).reshape(-1)
    if len(group_labels) != int(values.numel()):
        raise ValueError(
            f"group_labels length must match metric_values, got {len(group_labels)} vs {int(values.numel())}"
        )
    labels = [str(item) for item in group_labels]
    numerator_group = normalize_analysis_group(numerator_label)
    denominator_group = normalize_analysis_group(denominator_label)
    numerator_mask = torch.as_tensor(analysis_group_mask(labels, numerator_group), dtype=torch.bool, device=values.device)
    denominator_mask = torch.as_tensor(analysis_group_mask(labels, denominator_group), dtype=torch.bool, device=values.device)
    numerator_mean = _finite_mean(values[numerator_mask])
    denominator_mean = _finite_mean(values[denominator_mask])
    ratio = numerator_mean / torch.clamp(denominator_mean, min=float(eps))
    if not torch.isfinite(denominator_mean) or float(denominator_mean.item()) <= float(eps):
        ratio = values.new_tensor(float("nan"))
    return {
        "numerator_mean": numerator_mean,
        "denominator_mean": denominator_mean,
        "difference": numerator_mean - denominator_mean,
        "ratio": ratio,
        "numerator_count": values.new_tensor(int(numerator_mask.sum().item())),
        "denominator_count": values.new_tensor(int(denominator_mask.sum().item())),
    }

def _rankdata(values: Tensor) -> Tensor:
    flat = values.float().reshape(-1)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(flat.numel(), device=flat.device, dtype=flat.dtype)
    return ranks


def _pearson_corr(x: Tensor, y: Tensor, eps: float) -> Tensor:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1).to(device=x.device)
    valid = torch.isfinite(x) & torch.isfinite(y)
    if int(valid.sum().item()) < 2:
        return x.new_tensor(float("nan"))
    x = x[valid]
    y = y[valid]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
    if float(denom.item()) <= float(eps):
        return x.new_tensor(float("nan"))
    return (x_centered * y_centered).sum() / denom


def metric_correlation(
    x_values: Any,
    y_values: Any,
    *,
    eps: float = DEFAULT_EPS,
) -> dict[str, Tensor]:
    """计算两个样本级指标之间的 Pearson / Spearman 相关性。"""

    x = _as_float_tensor(x_values).reshape(-1)
    y = _as_float_tensor(y_values).to(device=x.device).reshape(-1)
    _check_same_shape(x, y, name_a="x_values", name_b="y_values")
    valid = torch.isfinite(x) & torch.isfinite(y)
    if int(valid.sum().item()) < 2:
        nan = x.new_tensor(float("nan"))
        return {"pearson": nan, "spearman": nan, "num_valid": x.new_tensor(float(valid.sum().item()))}
    x_valid = x[valid]
    y_valid = y[valid]
    return {
        "pearson": _pearson_corr(x_valid, y_valid, eps),
        "spearman": _pearson_corr(_rankdata(x_valid), _rankdata(y_valid), eps),
        "num_valid": x.new_tensor(float(x_valid.numel())),
    }

__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_FISHER_NORMALIZE_BY_DOF",
    "DEFAULT_MARGIN_MIN_ABS",
    "DEFAULT_SCALAR_OER_TOP_RATIO",
    "DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN",
    "DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX",
    "activation_clipping_rate",
    "clipping_rate",
    "token_energy_topk_indices",
    "symmetric_percentile_scale",
    "metric_correlation",
    "group_mean_amplification",
    "gather_token_values",
    "boolean_mask_rate",
    "collapse_gap",
    "tail_and_few_damage_gaps",
    "embedding_distortion",
    "fisher_ratio",
    "fisher_retention",
    "margin_retention",
    "margin_value_retention",
    "metric_drop",
    "scalar_oer_energy_metrics",
    "scalar_oer_retention",
    "scalar_oer_retention_loss",
    "scalar_oer_topk_index_metadata",
    "scalar_oer_topk_indices",
    "weight_clipping_rate",
]
