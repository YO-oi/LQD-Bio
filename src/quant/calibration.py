from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .observer import TensorDistributionObserver


def _as_runtime_qparam_tensor(value: Tensor, *, dtype: torch.dtype = torch.float32) -> Tensor:
    value = value.detach().to(dtype=dtype)
    if int(value.numel()) == 1:
        return value.reshape(())
    return value


def calculate_activation_qparams_from_observer(
    observer: TensorDistributionObserver,
    *,
    symmetric: bool,
    retained_percentile: float,
    lower_percentile: float | None,
    upper_percentile: float | None,
    qmin: int,
    qmax: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """根据 activation observer 统计计算 scale / zero_point。

    observer 只提供 FP32 activation 分布；这里才把统计摘要转成 qparams。
    symmetric 模式使用 abs activation 的 retained percentile。
    asymmetric 模式使用显式 lower / upper percentile 边界。
    """

    if qmin >= qmax:
        raise ValueError(f"qmin must be smaller than qmax, got qmin={qmin}, qmax={qmax}")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")

    if symmetric:
        if qmax <= 0:
            raise ValueError("symmetric activation qparams require qmax > 0")
        threshold = observer.percentile_value(float(retained_percentile), absolute=True)
        scale = torch.clamp(threshold / float(qmax), min=float(eps))
        zero_point = torch.zeros_like(scale)
        return _as_runtime_qparam_tensor(scale), _as_runtime_qparam_tensor(zero_point)

    if lower_percentile is None or upper_percentile is None:
        raise ValueError("asymmetric activation qparams require lower_percentile and upper_percentile")
    x_min = observer.percentile_value(float(lower_percentile), absolute=False)
    x_max = observer.percentile_value(float(upper_percentile), absolute=False)
    x_max = torch.maximum(x_max, x_min + float(eps))

    scale = torch.clamp((x_max - x_min) / float(qmax - qmin), min=float(eps))
    zero_point = torch.round(float(qmin) - x_min / scale)
    zero_point = torch.clamp(zero_point, min=float(qmin), max=float(qmax))
    return _as_runtime_qparam_tensor(scale), _as_runtime_qparam_tensor(zero_point)


def calculate_symmetric_weight_qparams_from_tensor(
    weight: Tensor,
    *,
    qmax: int,
    eps: float,
    channel_axis: int = 0,
) -> tuple[Tensor, Tensor]:
    """根据 FP32 weight 计算 symmetric per-channel scale / zero_point。

    这里只读取 weight，不修改 weight.data，也不把权重写成 INT4 容器。
    signed symmetric INT4 主实验使用有效码本 [-7, 7]，因此 scale = abs_max / 7。
    """

    if qmax <= 0:
        raise ValueError("symmetric weight qparams require qmax > 0")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if weight.ndim == 0:
        raise ValueError("weight qparams require tensor rank >= 1")

    axis = int(channel_axis)
    if axis < 0:
        axis += weight.ndim
    if axis < 0 or axis >= weight.ndim:
        raise ValueError(f"channel_axis={channel_axis} is invalid for weight shape={tuple(weight.shape)}")

    reduce_dims = tuple(dim for dim in range(weight.ndim) if dim != axis)
    abs_max = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp(abs_max / float(qmax), min=float(eps))
    zero_point = torch.zeros_like(scale)
    return scale, zero_point


def freeze(
    target: TensorDistributionObserver | Tensor,
    *,
    qmin: int | None = None,
    qmax: int | None = None,
    symmetric: bool | None = None,
    retained_percentile: float | None = None,
    lower_percentile: float | None = None,
    upper_percentile: float | None = None,
    eps: float | None = None,
    channel_axis: int | None = None,
) -> dict[str, Any]:
    """冻结校准统计并返回 qparams。

    这是 calibration.py 的公共冻结入口：
    - 传入 observer 时，冻结 observer 当前统计并返回 scale / zero_point / qmin / qmax；
    - 传入 weight tensor 时，按 per-channel MinMax 直接计算 qparams；
    - 不做 fake quant，不修改 weight.data，不管理模型 forward 状态。
    """

    if isinstance(target, TensorDistributionObserver):
        qmin_i = int(target.config.qmin if qmin is None else qmin)
        qmax_i = int(target.config.qmax if qmax is None else qmax)
        symmetric_b = bool(target.config.symmetric if symmetric is None else symmetric)
        eps_f = float(target.config.eps if eps is None else eps)
        if qmin_i >= qmax_i:
            raise ValueError(f"qmin must be smaller than qmax, got qmin={qmin_i}, qmax={qmax_i}")
        if eps_f <= 0.0:
            raise ValueError(f"eps must be positive, got {eps_f}")

        if target.config.observer_type == "activation":
            if target.config.granularity not in {"per_tensor", "per_channel"}:
                raise ValueError("activation qparams 只支持 per_tensor / per_channel observer")
            percentile = float(
                retained_percentile
                if retained_percentile is not None
                else target.config.percentile_high
                if target.config.percentile_high is not None
                else 99.9
            )
            low = lower_percentile if lower_percentile is not None else target.config.percentile_low
            high = upper_percentile if upper_percentile is not None else target.config.percentile_high
            scale, zero_point = calculate_activation_qparams_from_observer(
                target,
                symmetric=symmetric_b,
                retained_percentile=percentile,
                lower_percentile=low,
                upper_percentile=high,
                qmin=qmin_i,
                qmax=qmax_i,
                eps=eps_f,
            )
            method = (
                f"activation_symmetric_{target.config.granularity}_abs_p{percentile:g}"
                if symmetric_b
                else f"activation_asymmetric_{target.config.granularity}_p{float(low):g}_p{float(high):g}"
            )
        else:
            if target.config.granularity != "per_channel":
                raise ValueError("weight qparams 必须使用 per_channel observer")
            if not target.has_data:
                raise RuntimeError("weight MinMax qparams 需要先 update FP32 weight")
            if symmetric_b:
                if qmax_i <= 0:
                    raise ValueError("symmetric weight qparams require qmax > 0")
                scale = torch.clamp(target.abs_max_value.detach() / float(qmax_i), min=eps_f)
                zero_point = torch.zeros_like(scale)
                method = "weight_symmetric_per_channel_minmax"
            else:
                x_min = target.min_value.detach()
                x_max = torch.maximum(target.max_value.detach(), x_min + eps_f)
                scale = torch.clamp((x_max - x_min) / float(qmax_i - qmin_i), min=eps_f)
                zero_point = torch.round(float(qmin_i) - x_min / scale)
                zero_point = torch.clamp(zero_point, min=float(qmin_i), max=float(qmax_i))
                method = "weight_asymmetric_per_channel_minmax"

        qparams = {
            "scale": scale,
            "zero_point": zero_point,
            "qmin": qmin_i,
            "qmax": qmax_i,
            "symmetric": symmetric_b,
            "granularity": target.config.granularity,
            "observer_type": target.config.observer_type,
            "method": method,
            "channel_axis": int(target.config.channel_axis),
        }
        return target.set_frozen_qparams(qparams)

    if not isinstance(target, Tensor):
        raise TypeError(f"freeze 只支持 TensorDistributionObserver 或 Tensor，当前为 {type(target)!r}")

    qmin_i = int(-7 if qmin is None else qmin)
    qmax_i = int(7 if qmax is None else qmax)
    symmetric_b = bool(True if symmetric is None else symmetric)
    eps_f = float(1e-8 if eps is None else eps)
    axis = int(0 if channel_axis is None else channel_axis)
    if not symmetric_b:
        raise ValueError("直接传入 weight tensor 的 freeze 当前只支持 symmetric per-channel MinMax")
    if qmin_i >= qmax_i:
        raise ValueError(f"qmin must be smaller than qmax, got qmin={qmin_i}, qmax={qmax_i}")

    scale, zero_point = calculate_symmetric_weight_qparams_from_tensor(
        target,
        qmax=qmax_i,
        eps=eps_f,
        channel_axis=axis,
    )
    return {
        "scale": scale,
        "zero_point": zero_point,
        "qmin": qmin_i,
        "qmax": qmax_i,
        "symmetric": symmetric_b,
        "granularity": "per_channel",
        "observer_type": "weight",
        "method": "weight_symmetric_per_channel_minmax",
        "channel_axis": axis,
    }


def freeze_weight_qparams_from_tensor(
    weight: Tensor,
    *,
    qmin: int | None = None,
    qmax: int | None = None,
    symmetric: bool = True,
    eps: float | None = None,
    channel_axis: int = 0,
) -> dict[str, Any]:
    """从 FP32 weight tensor 计算并返回 per-channel weight qparams。"""

    return freeze(
        weight,
        qmin=qmin,
        qmax=qmax,
        symmetric=symmetric,
        eps=eps,
        channel_axis=channel_axis,
    )

__all__ = [
    "calculate_activation_qparams_from_observer",
    "calculate_symmetric_weight_qparams_from_tensor",
    "freeze",
    "freeze_weight_qparams_from_tensor",
]
