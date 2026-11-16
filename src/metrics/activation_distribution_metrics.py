from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

try:
    from .common_metrics import (
        DEFAULT_EPS,
        activation_clipping_rate,
        scalar_oer_energy_metrics,
        scalar_oer_topk_indices,
    )
    from src.utils.group_utils import MetricSpec, define_metric_spec, metric_spec_metadata
    from src.utils.layout_utils import ensure_canonical_btc
except ImportError:  # pragma: no cover
    from src.metrics.common_metrics import (
        DEFAULT_EPS,
        activation_clipping_rate,
        scalar_oer_energy_metrics,
        scalar_oer_topk_indices,
    )
    from src.utils.group_utils import MetricSpec, define_metric_spec, metric_spec_metadata
    from src.utils.layout_utils import ensure_canonical_btc


def _build_activation_distribution_metric_specs() -> dict[str, MetricSpec]:
    specs = (
        define_metric_spec("max_abs", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="group_max"),
        define_metric_spec("p99", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("p999", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("mean", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("std", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("kurtosis", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("heavy_tail_ratio", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("max_abs_over_p99", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("p999_over_p99", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("sparsity", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("sparsity_ratio", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("near_zero_rate", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("near_zero_ratio", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("outlier_ratio", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("outlier_energy", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("outlier_energy_ratio", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("scale_mean", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("scale_median", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("scale_p95", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("scale_max", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("clipping_rate", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("oer_retention", metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_per_tensor_scale", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_per_channel_scale_mean", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_per_channel_scale_std", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_scale_ratio_mean", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_scale_ratio_std", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_clipping_rate_per_tensor", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_clipping_rate_per_channel", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_oer_retention_per_tensor", metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("adaptive_oer_retention_per_channel", metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_per_tensor_scale", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_per_channel_scale_mean", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_per_channel_scale_std", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_scale_ratio_mean", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_scale_ratio_std", metric_direction="diagnostic", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_clipping_rate_per_tensor", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_clipping_rate_per_channel", metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_oer_retention_per_tensor", metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("fixed_oer_retention_per_channel", metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only"),
        define_metric_spec("histogram_density", metric_direction="diagnostic", aggregation_unit="histogram_bin", aggregation_fn="sum_then_normalize"),
    )
    return {spec.metric_name: spec for spec in specs}


ACTIVATION_DISTRIBUTION_METRIC_SPECS = _build_activation_distribution_metric_specs()


def activation_distribution_metric_names() -> tuple[str, ...]:
    return tuple(ACTIVATION_DISTRIBUTION_METRIC_SPECS.keys())


def activation_distribution_metric_spec(metric_name: str) -> MetricSpec:
    key = str(metric_name)
    if key not in ACTIVATION_DISTRIBUTION_METRIC_SPECS:
        raise KeyError(f"Activation Distribution MetricSpec is missing for {metric_name!r}")
    return ACTIVATION_DISTRIBUTION_METRIC_SPECS[key]


def activation_distribution_metric_metadata(metric_name: str) -> dict[str, str]:
    return metric_spec_metadata(activation_distribution_metric_spec(metric_name))


def _as_float_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _validate_non_empty(x: Tensor, name: str) -> None:
    if int(x.numel()) <= 0:
        raise ValueError(f"{name} must not be empty")


def _validate_scale(scale: Tensor, *, name: str = "scale") -> None:
    if not torch.isfinite(scale).all().item():
        raise ValueError(f"{name} must be finite")
    if not (scale > 0).all().item():
        raise ValueError(f"{name} must be > 0")


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    value = numerator / torch.clamp(denominator, min=float(DEFAULT_EPS))
    return torch.where(denominator > float(DEFAULT_EPS), value, torch.full_like(value, float("nan")))


def _safe_quantile(values: Tensor, q: float) -> Tensor:
    values = values.reshape(-1)
    _validate_non_empty(values, "values")
    if not torch.isfinite(values).all().item():
        raise ValueError("values contains NaN or Inf; cannot compute quantile")
    return torch.quantile(values.float(), float(q))


def _normalize_quantile(value: float) -> float:
    q = float(value)
    if q > 1.0:
        q = q / 100.0
    if not (0.0 < q <= 1.0):
        raise ValueError(f"percentile must be in (0, 1] or (0, 100], got {value}")
    return q


def _resolve_topk_count(numel: int, topk_ratio: float) -> int:
    if numel <= 0:
        raise ValueError("cannot compute top-k on an empty activation")
    ratio = float(topk_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {ratio}")
    return min(int(math.ceil(float(numel) * ratio)), int(numel))


def _scale_denominator(qmax: int) -> float:
    if int(qmax) <= 0:
        raise ValueError(f"qmax must be positive for symmetric activation scale, got {qmax}")
    return float(qmax)


def _symmetric_scale_from_abs_percentile(abs_values: Tensor, percentile: float, qmax: int, eps: float) -> Tensor:
    scale = _safe_quantile(abs_values, _normalize_quantile(float(percentile))) / _scale_denominator(int(qmax))
    return torch.clamp(scale, min=float(eps))


def _reshape_channel_scale(scale: Tensor, x: Tensor, channel_dim: int) -> Tensor:
    channel_dim = int(channel_dim)
    if channel_dim < 0:
        channel_dim += x.ndim
    if channel_dim < 0 or channel_dim >= x.ndim:
        raise ValueError(f"channel_dim={channel_dim} out of range for x.ndim={x.ndim}")
    if int(scale.numel()) != int(x.shape[channel_dim]):
        raise ValueError(
            "per-channel scale length must match x.shape[channel_dim], "
            f"got {int(scale.numel())} vs {int(x.shape[channel_dim])}"
        )
    shape = [1] * x.ndim
    shape[channel_dim] = int(scale.numel())
    return scale.reshape(shape)


def _fake_quantize_dequantize(x: Tensor, scale: Tensor, *, qmin: int, qmax: int) -> Tensor:
    _validate_scale(scale)
    if int(qmin) >= int(qmax):
        raise ValueError(f"qmin must be smaller than qmax, got {qmin}, {qmax}")
    q = torch.round(x / scale).clamp(float(qmin), float(qmax))
    return q * scale


def compute_basic_distribution_stats(x: Any) -> dict[str, Tensor]:
    """Basic signed and absolute activation distribution statistics."""

    tensor = _as_float_tensor(x)
    _validate_non_empty(tensor, "x")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("x contains NaN or Inf; cannot compute activation distribution stats")

    flat = tensor.reshape(-1)
    abs_flat = flat.abs()
    mean = flat.mean()
    std = flat.std(unbiased=False)
    centered = flat - mean
    variance = centered.square().mean()
    if variance <= float(DEFAULT_EPS):
        kurtosis = flat.new_tensor(float("nan"))
    else:
        kurtosis = centered.pow(4).mean() / torch.clamp(variance.square(), min=float(DEFAULT_EPS))

    max_abs = abs_flat.max()
    p99 = _safe_quantile(abs_flat, 0.99)
    p999 = _safe_quantile(abs_flat, 0.999)
    max_abs_over_p99 = _safe_ratio(max_abs, p99)
    p999_over_p99 = _safe_ratio(p999, p99)
    return {
        "max_abs": max_abs,
        "p99": p99,
        "p999": p999,
        "mean": mean,
        "std": std,
        "kurtosis": kurtosis,
        "heavy_tail_ratio": max_abs_over_p99,
        "max_abs_over_p99": max_abs_over_p99,
        "p999_over_p99": p999_over_p99,
    }


def compute_sparsity_stats(x: Any, scale: Any) -> dict[str, Tensor]:
    """Exact-zero and quantization-zero-bin occupancy statistics."""

    tensor = _as_float_tensor(x)
    scale_t = _as_float_tensor(scale).to(device=tensor.device)
    _validate_non_empty(tensor, "x")
    _validate_scale(scale_t)
    near_zero = tensor.abs() <= (0.5 * scale_t)
    sparsity = (tensor == 0).float().mean()
    near_zero_ratio = near_zero.float().mean()
    return {
        "sparsity": sparsity,
        "sparsity_ratio": sparsity,
        "near_zero_rate": near_zero_ratio,
        "near_zero_ratio": near_zero_ratio,
    }


def compute_outlier_stats(x: Any, topk_ratio: float = 0.001) -> dict[str, Tensor]:
    """Energy top-k scalar concentration in the FP32 activation."""

    tensor = _as_float_tensor(x)
    _validate_non_empty(tensor, "x")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("x contains NaN or Inf; cannot compute activation outlier stats")
    flat = tensor.reshape(-1)
    k = _resolve_topk_count(int(flat.numel()), float(topk_ratio))
    topk_indices = scalar_oer_topk_indices(flat, topk=k, batch_dim=None).reshape(-1)
    topk_values = flat.gather(dim=0, index=topk_indices.to(device=flat.device))
    topk_energy = topk_values.square().sum()
    total_energy = flat.square().sum()
    energy_ratio = topk_energy / torch.clamp(total_energy, min=float(DEFAULT_EPS))
    energy_ratio = torch.where(
        total_energy > float(DEFAULT_EPS),
        energy_ratio,
        flat.new_tensor(float("nan")),
    )
    return {
        "outlier_ratio": flat.new_tensor(float(k) / float(flat.numel())),
        "outlier_energy": energy_ratio,
        "outlier_energy_ratio": energy_ratio,
        "topk_indices": topk_indices.cpu(),
        "topk_energy": topk_energy,
    }


def compute_a4_qparams_per_tensor(
    x: Any,
    percentile: float = 0.999,
    *,
    qmin: int = -8,
    qmax: int = 7,
) -> dict[str, Tensor]:
    """Per-tensor symmetric A4 activation qparams estimated from one tensor."""

    tensor = _as_float_tensor(x)
    _validate_non_empty(tensor, "x")
    if int(qmin) >= int(qmax):
        raise ValueError(f"qmin must be smaller than qmax, got {qmin}, {qmax}")
    scale = _symmetric_scale_from_abs_percentile(tensor.abs(), percentile, qmax=int(qmax), eps=DEFAULT_EPS)
    return {
        "scale": scale,
        "zero_point": torch.zeros_like(scale),
        "qmin": torch.tensor(int(qmin), device=scale.device, dtype=torch.int64),
        "qmax": torch.tensor(int(qmax), device=scale.device, dtype=torch.int64),
    }


def compute_a4_qparams_per_channel(
    x: Any,
    percentile: float = 0.999,
    channel_dim: int = -1,
    *,
    qmin: int = -8,
    qmax: int = 7,
) -> dict[str, Tensor]:
    """Per-channel symmetric A4 activation qparams estimated from one tensor."""

    tensor = _as_float_tensor(x)
    _validate_non_empty(tensor, "x")
    if int(qmin) >= int(qmax):
        raise ValueError(f"qmin must be smaller than qmax, got {qmin}, {qmax}")
    if tensor.ndim < 1:
        raise ValueError("x must have rank >= 1 for per-channel statistics")
    channel_dim = int(channel_dim)
    if channel_dim < 0:
        channel_dim += tensor.ndim
    if channel_dim < 0 or channel_dim >= tensor.ndim:
        raise ValueError(f"channel_dim={channel_dim} out of range for x.ndim={tensor.ndim}")

    moved = tensor.movedim(channel_dim, -1).contiguous()
    channel_values = moved.reshape(-1, int(moved.shape[-1])).abs()
    scale = torch.quantile(channel_values.float(), _normalize_quantile(float(percentile)), dim=0) / _scale_denominator(int(qmax))
    scale = torch.clamp(scale, min=float(DEFAULT_EPS))
    return {
        "scale": scale,
        "zero_point": torch.zeros_like(scale),
        "qmin": torch.full_like(scale, int(qmin), dtype=torch.int64),
        "qmax": torch.full_like(scale, int(qmax), dtype=torch.int64),
        "channel_dim": torch.tensor(channel_dim, dtype=torch.int64),
    }


def compute_clipping_rate(x: Any, scale: Any, qmin: int = -8, qmax: int = 7) -> Tensor:
    """Activation clipping rate under a fixed symmetric scale."""

    tensor = _as_float_tensor(x)
    scale_t = _as_float_tensor(scale).to(device=tensor.device)
    channel_axis = -1 if scale_t.ndim > 0 and int(scale_t.numel()) > 1 else None
    if channel_axis is None:
        return activation_clipping_rate(
            tensor,
            scale=scale_t,
            zero_point=0.0,
            qmin=int(qmin),
            qmax=int(qmax),
            reduction="mean",
        )
    return activation_clipping_rate(
        tensor,
        scale=_reshape_channel_scale(scale_t, tensor, channel_axis),
        zero_point=0.0,
        qmin=int(qmin),
        qmax=int(qmax),
        reduction="mean",
    )


def compute_oer_retention(
    x: Any,
    scale: Any,
    topk_indices: Any,
    qmin: int = -8,
    qmax: int = 7,
) -> Tensor:
    """OER retention on FP32 top-k indices after fake quantization with a fixed scale."""

    tensor = _as_float_tensor(x)
    scale_t = _as_float_tensor(scale).to(device=tensor.device)
    if scale_t.ndim > 0 and int(scale_t.numel()) > 1:
        scale_t = _reshape_channel_scale(scale_t, tensor, -1)
    quant = _fake_quantize_dequantize(tensor, scale_t, qmin=int(qmin), qmax=int(qmax))
    return scalar_oer_energy_metrics(
        tensor,
        quant,
        topk_indices=topk_indices,
        batch_dim=None,
        reduction="mean",
    )["oer_retention"]


def _nan_metrics(tensor: Tensor, names: tuple[str, ...]) -> dict[str, Tensor]:
    nan = tensor.new_tensor(float("nan"))
    return {name: nan for name in names}


def _scale_summary_metrics(
    tensor: Tensor,
    *,
    prefix: str,
    per_tensor_scale: Any | None,
    per_channel_scale: Any | None,
) -> dict[str, Tensor]:
    names = (
        f"{prefix}_per_tensor_scale",
        f"{prefix}_per_channel_scale_mean",
        f"{prefix}_per_channel_scale_std",
        f"{prefix}_scale_ratio_mean",
        f"{prefix}_scale_ratio_std",
    )
    if per_tensor_scale is None or per_channel_scale is None:
        return _nan_metrics(tensor, names)

    tensor_scale = _as_float_tensor(per_tensor_scale).to(device=tensor.device).reshape(-1)
    channel_scale = _as_float_tensor(per_channel_scale).to(device=tensor.device).reshape(-1)
    if int(tensor_scale.numel()) != 1:
        raise ValueError(f"{prefix}_per_tensor_scale must be scalar, got {tuple(tensor_scale.shape)}")
    _validate_scale(tensor_scale, name=f"{prefix}_per_tensor_scale")
    _validate_scale(channel_scale, name=f"{prefix}_per_channel_scale")
    scale_ratio = channel_scale / torch.clamp(tensor_scale.reshape(()), min=float(DEFAULT_EPS))
    return {
        f"{prefix}_per_tensor_scale": tensor_scale.reshape(()),
        f"{prefix}_per_channel_scale_mean": channel_scale.float().mean(),
        f"{prefix}_per_channel_scale_std": channel_scale.float().std(unbiased=False),
        f"{prefix}_scale_ratio_mean": scale_ratio.float().mean(),
        f"{prefix}_scale_ratio_std": scale_ratio.float().std(unbiased=False),
    }


def _generic_scale_metrics(tensor: Tensor, *, preferred_scale: Any | None) -> dict[str, Tensor]:
    names = ("scale_mean", "scale_median", "scale_p95", "scale_max")
    if preferred_scale is None:
        return _nan_metrics(tensor, names)
    scale = _as_float_tensor(preferred_scale).to(device=tensor.device).reshape(-1)
    _validate_scale(scale, name="scale")
    return {
        "scale_mean": scale.float().mean(),
        "scale_median": torch.quantile(scale.float(), 0.5),
        "scale_p95": torch.quantile(scale.float(), 0.95),
        "scale_max": scale.float().max(),
    }


def _quant_pressure_metrics(
    tensor: Tensor,
    *,
    prefix: str,
    per_tensor_scale: Any | None,
    per_channel_scale: Any | None,
    topk_indices: Any,
    qmin: int,
    qmax: int,
) -> dict[str, Tensor]:
    names = (
        f"{prefix}_clipping_rate_per_tensor",
        f"{prefix}_clipping_rate_per_channel",
        f"{prefix}_oer_retention_per_tensor",
        f"{prefix}_oer_retention_per_channel",
    )
    if per_tensor_scale is None or per_channel_scale is None:
        return _nan_metrics(tensor, names)

    return {
        f"{prefix}_clipping_rate_per_tensor": compute_clipping_rate(tensor, per_tensor_scale, qmin=qmin, qmax=qmax),
        f"{prefix}_clipping_rate_per_channel": compute_clipping_rate(tensor, per_channel_scale, qmin=qmin, qmax=qmax),
        f"{prefix}_oer_retention_per_tensor": compute_oer_retention(tensor, per_tensor_scale, topk_indices, qmin=qmin, qmax=qmax),
        f"{prefix}_oer_retention_per_channel": compute_oer_retention(tensor, per_channel_scale, topk_indices, qmin=qmin, qmax=qmax),
    }


def compute_activation_distribution_metrics(
    x: Any,
    *,
    percentile: float = 0.999,
    topk_ratio: float = 0.001,
    channel_dim: int = -1,
    qmin: int = -8,
    qmax: int = 7,
    fixed_per_tensor_scale: Any | None = None,
    fixed_per_channel_scale: Any | None = None,
) -> dict[str, Tensor]:
    """Summarize one activation tensor for sensitivity explanation.

    adaptive_* metrics use per-sample/per-tensor qparams estimated from this tensor.
    fixed_* metrics use qparams calibrated outside this function, typically from a
    calibration split. The fixed branch is the one to compare with deployment-like
    fixed activation quantization pressure.
    """

    tensor = ensure_canonical_btc(_as_float_tensor(x), context="activation_distribution_metrics.x")
    basic = compute_basic_distribution_stats(tensor)
    outlier = compute_outlier_stats(tensor, topk_ratio=topk_ratio)
    adaptive_per_tensor_qparams = compute_a4_qparams_per_tensor(tensor, percentile=percentile, qmin=qmin, qmax=qmax)
    adaptive_per_channel_qparams = compute_a4_qparams_per_channel(
        tensor,
        percentile=percentile,
        channel_dim=channel_dim,
        qmin=qmin,
        qmax=qmax,
    )
    adaptive_per_tensor_scale = adaptive_per_tensor_qparams["scale"]
    adaptive_per_channel_scale = adaptive_per_channel_qparams["scale"]

    sparsity = compute_sparsity_stats(tensor, adaptive_per_tensor_scale)
    topk_indices = outlier["topk_indices"]
    adaptive_pressure = _quant_pressure_metrics(
        tensor,
        prefix="adaptive",
        per_tensor_scale=adaptive_per_tensor_scale,
        per_channel_scale=adaptive_per_channel_scale,
        topk_indices=topk_indices,
        qmin=qmin,
        qmax=qmax,
    )
    fixed_pressure = _quant_pressure_metrics(
        tensor,
        prefix="fixed",
        per_tensor_scale=fixed_per_tensor_scale,
        per_channel_scale=fixed_per_channel_scale,
        topk_indices=topk_indices,
        qmin=qmin,
        qmax=qmax,
    )
    fixed_scale_available = fixed_per_tensor_scale is not None and fixed_per_channel_scale is not None
    preferred_scale = fixed_per_channel_scale if fixed_scale_available else adaptive_per_channel_scale
    preferred_clipping = (
        fixed_pressure["fixed_clipping_rate_per_tensor"]
        if fixed_scale_available
        else adaptive_pressure["adaptive_clipping_rate_per_tensor"]
    )
    preferred_oer = (
        fixed_pressure["fixed_oer_retention_per_tensor"]
        if fixed_scale_available
        else adaptive_pressure["adaptive_oer_retention_per_tensor"]
    )

    return {
        **basic,
        **sparsity,
        "outlier_ratio": outlier["outlier_ratio"],
        "outlier_energy": outlier["outlier_energy"],
        "outlier_energy_ratio": outlier["outlier_energy_ratio"],
        **_generic_scale_metrics(tensor, preferred_scale=preferred_scale),
        "clipping_rate": preferred_clipping,
        "oer_retention": preferred_oer,
        **_scale_summary_metrics(
            tensor,
            prefix="adaptive",
            per_tensor_scale=adaptive_per_tensor_scale,
            per_channel_scale=adaptive_per_channel_scale,
        ),
        **adaptive_pressure,
        **_scale_summary_metrics(
            tensor,
            prefix="fixed",
            per_tensor_scale=fixed_per_tensor_scale,
            per_channel_scale=fixed_per_channel_scale,
        ),
        **fixed_pressure,
    }


__all__ = [
    "ACTIVATION_DISTRIBUTION_METRIC_SPECS",
    "activation_distribution_metric_metadata",
    "activation_distribution_metric_names",
    "activation_distribution_metric_spec",
    "compute_a4_qparams_per_channel",
    "compute_a4_qparams_per_tensor",
    "compute_activation_distribution_metrics",
    "compute_basic_distribution_stats",
    "compute_clipping_rate",
    "compute_oer_retention",
    "compute_outlier_stats",
    "compute_sparsity_stats",
]

