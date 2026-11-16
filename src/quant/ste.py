from __future__ import annotations

import math
from typing import SupportsFloat

import torch
from torch import Tensor


class ClippedIdentitySTE(torch.autograd.Function):
    """带裁剪区间的恒等直通估计器。

    前向：
        x -> x / scale + zero_point -> clamp -> round -> dequant

    反向：
        round 视为恒等映射传递梯度；
        clamp 饱和区外的输入梯度置 0，区间内梯度原样传递。

    注意：
        这里不做全量 finite 检查。fake quant 是训练热路径，
        对激活张量调用 isfinite(...).all().item() 会触发 GPU 同步和全量扫描。
        需要排查数值问题时，在外层函数显式打开 debug_checks。
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        scale: Tensor,
        zero_point: Tensor,
        qmin: int,
        qmax: int,
    ) -> Tensor:
        if qmin >= qmax:
            raise ValueError(f"qmin must be smaller than qmax, got qmin={qmin}, qmax={qmax}")
        if not torch.is_floating_point(x):
            raise TypeError(f"x must be a floating point Tensor, got dtype={x.dtype}")
        if not torch.is_floating_point(scale):
            raise TypeError(f"scale must be a floating point Tensor, got dtype={scale.dtype}")
        if not torch.is_floating_point(zero_point):
            raise TypeError(f"zero_point must be a floating point Tensor, got dtype={zero_point.dtype}")

        qmin_f = float(qmin)
        qmax_f = float(qmax)

        # 进入量化整数域。scale/zero_point 允许按 PyTorch 广播规则工作。
        q = x / scale + zero_point

        # 只有需要 x 的梯度时才保存 mask。严格 clipped identity STE 的反向
        # 必须知道 clamp 前是否饱和；bool mask 是保持语义的最小原生张量状态。
        if ctx.needs_input_grad[0]:
            pass_through_mask = (q >= qmin_f) & (q <= qmax_f)
            ctx.save_for_backward(pass_through_mask)
        else:
            ctx.save_for_backward()

        # 前向必须使用离散量化结果：clamp、round、再反量化。
        q_clamped = torch.clamp(q, min=qmin_f, max=qmax_f)
        q_rounded = torch.round(q_clamped)
        y = (q_rounded - zero_point) * scale
        return y.to(dtype=x.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor | None, None, None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None

        (pass_through_mask,) = ctx.saved_tensors

        # 避免 pass_through_mask.to(float) 产生同尺寸浮点临时张量。
        # clone 是返回 grad_x 所需的新张量，masked_fill_ 在原地屏蔽饱和区梯度。
        grad_x = grad_output.clone()
        grad_x.masked_fill_(~pass_through_mask, 0)
        return grad_x, None, None, None, None


def _as_scale_tensor_like(value: Tensor | SupportsFloat, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)

    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"scale must be a finite positive scalar, got {value!r}")
    return torch.tensor(numeric, device=reference.device, dtype=reference.dtype)


def _as_zero_point_tensor_like(value: Tensor | SupportsFloat, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"zero_point must be a finite scalar, got {value!r}")
    return torch.tensor(numeric, device=reference.device, dtype=reference.dtype)


def _debug_validate_quant_inputs(x: Tensor, scale: Tensor, zero_point: Tensor) -> None:
    """显式调试检查。

    这个函数会扫描张量并在 GPU 上触发同步，只能用于定位数值问题，
    不应在 teacher/student/KD/fake quant 训练热路径默认开启。
    """

    if torch.any(scale <= 0).item():
        raise ValueError("scale contains non-positive values")
    if not torch.isfinite(x).all().item():
        raise ValueError("x contains NaN or Inf")
    if not torch.isfinite(scale).all().item():
        raise ValueError("scale contains NaN or Inf")
    if not torch.isfinite(zero_point).all().item():
        raise ValueError("zero_point contains NaN or Inf")


def quantize_dequantize_ste(
    x: Tensor,
    scale: Tensor | SupportsFloat,
    zero_point: Tensor | SupportsFloat = 0.0,
    *,
    qmin: int,
    qmax: int,
    debug_checks: bool = False,
) -> Tensor:
    """使用 clipped identity STE 的底层 fake-quant 算子。

    该函数只处理量化中的梯度直通近似，不读取配置文件，不决定 INT4
    策略，也不负责 observer/calibrator 统计。

    debug_checks=False 是训练热路径默认值，避免全量 finite 扫描和 GPU 同步。
    """

    scale_tensor = _as_scale_tensor_like(scale, x)
    zero_point_tensor = _as_zero_point_tensor_like(zero_point, x)
    if debug_checks:
        _debug_validate_quant_inputs(x, scale_tensor, zero_point_tensor)
    return ClippedIdentitySTE.apply(x, scale_tensor, zero_point_tensor, int(qmin), int(qmax))


def clipped_identity_ste(
    x: Tensor,
    scale: Tensor | SupportsFloat,
    zero_point: Tensor | SupportsFloat = 0.0,
    *,
    qmin: int,
    qmax: int,
    debug_checks: bool = False,
) -> Tensor:
    """quantize_dequantize_ste 的语义化别名。"""

    return quantize_dequantize_ste(
        x,
        scale,
        zero_point,
        qmin=int(qmin),
        qmax=int(qmax),
        debug_checks=bool(debug_checks),
    )


__all__ = [
    "ClippedIdentitySTE",
    "clipped_identity_ste",
    "quantize_dequantize_ste",
]
