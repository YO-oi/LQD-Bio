from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor

try:
    from .common_metrics import (
        boolean_mask_rate,
        gather_token_values,
        scalar_oer_energy_metrics,
        scalar_oer_topk_indices,
        symmetric_percentile_scale,
        token_energy_topk_indices,
    )
    from .metric_config import (
        DEFAULT_EPS,
        DEFAULT_SCALAR_OER_TOP_RATIO,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    )
    from src.utils.layout_utils import canonicalize_activation_btc, ensure_canonical_btc, require_valid_token_policy
except ImportError:  # pragma: no cover - 閸忕厧閻╁瓨甯撮懘姘拱鐠嬪啳鐦妴?
    from src.metrics.common_metrics import (
        boolean_mask_rate,
        gather_token_values,
        scalar_oer_energy_metrics,
        scalar_oer_topk_indices,
        symmetric_percentile_scale,
        token_energy_topk_indices,
    )
    from src.metrics.metric_config import (
        DEFAULT_EPS,
        DEFAULT_SCALAR_OER_TOP_RATIO,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    )
    from src.utils.layout_utils import canonicalize_activation_btc, ensure_canonical_btc, require_valid_token_policy


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


def _validate_activation(x: Tensor, name: str = "activation") -> None:
    if int(x.numel()) <= 0:
        raise ValueError(f"{name} must be finite")
    if not torch.isfinite(x).all().item():
        raise ValueError(f"{name} must be finite")


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("閺嶅洭鍣?tensor 濞屸剝婀侀崣鏁ょ紒鏉戝")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim={dim} 鐡掑﹦鏅敍瀹痙im={ndim}")
    return dim


def _validate_qrange(qmin: int, qmax: int) -> None:
    if int(qmin) >= int(qmax):
        raise ValueError(f"qmin 韫囧懘銆忕亸蹇庣艾 qmax閿涘苯缍嬮崜?qmin={qmin}, qmax={qmax}")
    if int(qmax) <= 0:
        raise ValueError(f"symmetric A4 qmax 韫囧懘銆忔稉鐑橀弫甯礉瑜版挸澧?qmax={qmax}")


def _validate_scale(scale: Tensor, *, name: str = "scale") -> None:
    if int(scale.numel()) <= 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(scale).all().item():
        raise ValueError(f"{name} must be finite")
    if not (scale > 0).all().item():
        raise ValueError(f"{name} must be positive")


def _normalize_valid_token_mask(valid_token_mask: Any | None, x_btc: Tensor, *, name: str = "valid_token_mask") -> Tensor | None:
    if valid_token_mask is None:
        return None
    valid = torch.as_tensor(valid_token_mask, dtype=torch.bool, device=x_btc.device)
    if valid.ndim == 1:
        valid = valid.reshape(1, -1)
    if valid.ndim != 2:
        raise ValueError(f"{name} must be [B,T] or [T], got shape={tuple(valid.shape)}")
    if int(valid.shape[0]) == 1 and int(x_btc.shape[0]) != 1:
        valid = valid.expand(int(x_btc.shape[0]), int(valid.shape[1]))
    expected = (int(x_btc.shape[0]), int(x_btc.shape[1]))
    if tuple(valid.shape) != expected:
        raise ValueError(f"{name} shape must align with activation [B,T], got {tuple(valid.shape)} vs {expected}")
    if not valid.any().item():
        raise ValueError(f"{name} contains no valid tokens")
    return valid.contiguous()


def _expand_valid_token_mask(valid_token_mask: Any | None, x_btc: Tensor) -> Tensor | None:
    valid = _normalize_valid_token_mask(valid_token_mask, x_btc)
    if valid is None:
        return None
    return valid.unsqueeze(-1).expand_as(x_btc)


def _broadcast_qparam(param: Any, x_btc: Tensor, *, channel_dim: int = -1, name: str = "qparam") -> Tensor:
    tensor = _as_float_tensor(param).to(device=x_btc.device)
    if tensor.ndim == 0 or int(tensor.numel()) == 1:
        return tensor.reshape(()).to(dtype=x_btc.dtype)

    channel_dim = _normalize_dim(int(channel_dim), x_btc.ndim)
    if int(tensor.numel()) != int(x_btc.shape[channel_dim]):
        raise ValueError(
            f"{name} 閼汇儰璐熼柅鎰扳偓姘朵壕瀵娀鍣洪敍宀勬毐鎼达箑绻€妞よ崵鐡戞禍?channel 缂佹潙銇囩亸蹇ョ礉"
            f"瑜版挸澧?{int(tensor.numel())} vs {int(x_btc.shape[channel_dim])}"
        )
    shape = [1] * x_btc.ndim
    shape[channel_dim] = int(tensor.numel())
    return tensor.reshape(shape).to(dtype=x_btc.dtype)



def normalize_activation_layout(
    x: Any,
    *,
    batch_dim: int = 0,
    token_dim: int = 1,
    channel_dim: int = -1,
    layout: str | None = None,
) -> Tensor:
    """Strict compatibility wrapper returning canonical [B,T,C].

    New protocol forbids rank-2 expansion and heuristic dimension guessing. If
    layout is omitted the tensor must already be canonical [B,T,C]. If layout is
    provided, only BTC/TBC are accepted by the shared layout adapter.
    """

    del batch_dim, token_dim, channel_dim
    tensor = _as_float_tensor(x)
    if layout is None:
        return ensure_canonical_btc(tensor, context="quant_danger_zone.activation")
    return canonicalize_activation_btc(
        tensor,
        raw_layout=str(layout),
        context="quant_danger_zone.activation",
    ).tensor


def compute_per_tensor_scale(
    x: Any,
    *,
    percentile: float = 0.999,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """鐠侊紕鐣?A4 per-tensor symmetric scale閵?

    scale = percentile(abs(x)) / qmax閵嗗倿绮拋?qmax 閺夈儴鍤滄い鍦窗缂佺喍绔撮張澶嬫櫏閻焦婀伴妴?
    """

    x_btc = normalize_activation_layout(x)
    _validate_qrange(DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN, int(qmax))
    return symmetric_percentile_scale(x_btc, percentile=percentile, qmax=qmax, reduce_dims=None, eps=eps)


def compute_per_channel_scale(
    x: Any,
    *,
    percentile: float = 0.999,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """鐠侊紕鐣?A4 per-channel symmetric scale閵?

    鏉堟挸鍙嗘导姘帥缂佺喍绔存稉?[B, T, C]閿涘瞼鍔ч崥搴㈤儴 [B, T] reduce閿涘奔绻氶悾娆愭付閸氬海娈?C 闁岸浜鹃妴?
    """

    x_btc = normalize_activation_layout(x)
    _validate_qrange(DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN, int(qmax))
    return symmetric_percentile_scale(x_btc, percentile=percentile, qmax=qmax, reduce_dims=(0, 1), eps=eps)


def compute_a4_integer_mapping(
    x: Any,
    *,
    scale: Any | None = None,
    zero_point: Any = 0.0,
    qmin: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    percentile: float = 0.999,
    channel_dim: int = -1,
    layout: str | None = None,
) -> dict[str, Tensor | int | str]:
    """鐠侊紕鐣婚獮鏈电箽鐎?A4 閺佸瓨鏆熺純鎴炵壐閺勭姴鐨犻妴?

    閺堚偓閸忔娊鏁惃鍕厬闂傛挳鍣洪弰绱?
        x_int_raw = round(x / scale + zero_point)

    閸楅亶娅撻崠铏圭埠鐠佲€崇唨娴?x_int_raw閿涘矁鈧奔绗夐弰褰ч惇瀣冀闁插繐瀵查崥搴ｆ畱 x_q閿?
    - x_int_raw <= qmin 鐠侀璐熺拹鐔烘鍗炴嫲 / 閹存焽閸楅亶娅撻崠?
    - x_int_raw >= qmax 鐠侀璐熷锝囨鍗炴嫲 / 閹存焽閸楅亶娅撻崠?
    - abs(x_int_raw - zero_point) <= 1 鐠侀璐熸潻鎴︽祩閺佸瓨鏆熼弽鐓庡祫闂勨晛灏?
    """

    qmin_i = int(qmin)
    qmax_i = int(qmax)
    _validate_qrange(qmin_i, qmax_i)
    x_btc = normalize_activation_layout(x, layout=layout)
    if scale is None:
        scale_t = compute_per_tensor_scale(x_btc, percentile=percentile, qmax=qmax_i)
        granularity = "per_tensor"
    else:
        scale_t = _as_float_tensor(scale).to(device=x_btc.device)
        granularity = "per_tensor" if int(scale_t.numel()) == 1 else "per_channel"
    _validate_scale(scale_t)

    scale_b = _broadcast_qparam(scale_t, x_btc, channel_dim=channel_dim, name="scale")
    zero_b = _broadcast_qparam(zero_point, x_btc, channel_dim=channel_dim, name="zero_point")
    x_int_raw = torch.round(x_btc / scale_b + zero_b)
    x_int_clamped = torch.clamp(x_int_raw, min=float(qmin_i), max=float(qmax_i))
    x_q = (x_int_clamped - zero_b) * scale_b

    low_clip_mask = x_int_raw <= float(qmin_i)
    high_clip_mask = x_int_raw >= float(qmax_i)
    boundary_hit_mask = low_clip_mask | high_clip_mask
    strict_overflow_mask = (x_int_raw < float(qmin_i)) | (x_int_raw > float(qmax_i))
    x_int_zero = torch.clamp(x_int_raw, min=float(qmin_i), max=float(qmax_i))
    exact_zero_mask = x_int_zero == zero_b
    near_zero_pm1_mask = (x_int_zero - zero_b).abs() <= 1.0

    return {
        "x": x_btc,
        "scale": scale_t.detach().clone(),
        "zero_point": _as_float_tensor(zero_point).detach().clone(),
        "x_int_raw": x_int_raw,
        "x_int_clamped": x_int_clamped,
        "x_q": x_q,
        "low_clip_mask": low_clip_mask,
        "high_clip_mask": high_clip_mask,
        "boundary_hit_mask": boundary_hit_mask,
        "strict_overflow_mask": strict_overflow_mask,
        "exact_zero_mask": exact_zero_mask,
        "near_zero_pm1_mask": near_zero_pm1_mask,
        "clip_mask": boundary_hit_mask,
        "near_zero_mask": near_zero_pm1_mask,
        "clip_mask_deprecated_alias_of": "boundary_hit_mask",
        "near_zero_mask_deprecated_alias_of": "near_zero_pm1_mask",
        "qmin": qmin_i,
        "qmax": qmax_i,
        "granularity": granularity,
    }


def compute_clip_rates(
    integer_mapping: Mapping[str, Any] | Tensor,
    *,
    qmin: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    valid_mask: Any | None = None,
    per_sample: bool = False,
) -> dict[str, Tensor]:
    """閸╄桨绨?x_int_raw 閻ㄥ嫯绔熼悾灞芥嚒娑撳芳缂佺喕閵?

    clip_rate 娴ｈ法鏁ゆい鍦窗閸楅亶娅撻崠鍝勭暰娑斿绱皒_int_raw <= qmin 閹?x_int_raw >= qmax閵?
    strict_overflow_rate 閸欑埠鐠侊紕婀″锝堢Ш閻ｅ矉绱皒_int_raw < qmin 閹?x_int_raw > qmax閵?
    """

    if isinstance(integer_mapping, Mapping):
        low = integer_mapping["low_clip_mask"]
        high = integer_mapping["high_clip_mask"]
        boundary = integer_mapping.get("boundary_hit_mask", integer_mapping.get("clip_mask"))
        strict = integer_mapping.get("strict_overflow_mask")
    else:
        raw = _as_float_tensor(integer_mapping)
        low = raw <= float(qmin)
        high = raw >= float(qmax)
        boundary = low | high
        strict = (raw < float(qmin)) | (raw > float(qmax))
    low_t = torch.as_tensor(low, dtype=torch.bool)
    high_t = torch.as_tensor(high, dtype=torch.bool, device=low_t.device)
    boundary_t = torch.as_tensor(boundary, dtype=torch.bool, device=low_t.device)
    strict_t = torch.as_tensor(strict, dtype=torch.bool, device=low_t.device)
    boundary_rate = boolean_mask_rate(boundary_t, valid_mask=valid_mask, per_sample=per_sample)
    strict_rate = boolean_mask_rate(strict_t, valid_mask=valid_mask, per_sample=per_sample)
    return {
        "boundary_hit_rate": boundary_rate,
        "clip_rate": boundary_rate,
        "clip_low_rate": boolean_mask_rate(low_t, valid_mask=valid_mask, per_sample=per_sample),
        "clip_high_rate": boolean_mask_rate(high_t, valid_mask=valid_mask, per_sample=per_sample),
        "strict_overflow_rate": strict_rate,
    }


def compute_near_zero_rate(
    integer_mapping: Mapping[str, Any] | Tensor,
    *,
    zero_point: Any = 0.0,
    threshold: int = 1,
    valid_mask: Any | None = None,
    per_sample: bool = False,
) -> Tensor:
    """Compute near-zero rate on the integer grid."""

    if isinstance(integer_mapping, Mapping):
        mask = torch.as_tensor(integer_mapping["near_zero_mask"], dtype=torch.bool)
    else:
        raw = _as_float_tensor(integer_mapping)
        zero = _broadcast_qparam(zero_point, raw, name="zero_point")
        mask = (raw - zero).abs() <= float(threshold)
    return boolean_mask_rate(mask, valid_mask=valid_mask, per_sample=per_sample)


def compute_exact_zero_rate(
    integer_mapping: Mapping[str, Any] | Tensor,
    *,
    zero_point: Any = 0.0,
    valid_mask: Any | None = None,
    per_sample: bool = False,
) -> Tensor:
    """Compute exact-zero rate: q_int == zero_point."""

    if isinstance(integer_mapping, Mapping):
        if "exact_zero_mask" in integer_mapping:
            mask = torch.as_tensor(integer_mapping["exact_zero_mask"], dtype=torch.bool)
        else:
            raw_value = integer_mapping.get("x_int_clamped", integer_mapping.get("q_int", integer_mapping.get("x_int_raw")))
            if raw_value is None:
                raise KeyError("integer_mapping must contain exact_zero_mask, x_int_clamped, q_int, or x_int_raw")
            raw = _as_float_tensor(raw_value)
            zero = _broadcast_qparam(integer_mapping.get("zero_point", zero_point), raw, name="zero_point")
            mask = raw == zero
    else:
        raw = _as_float_tensor(integer_mapping)
        zero = _broadcast_qparam(zero_point, raw, name="zero_point")
        mask = raw == zero
    return boolean_mask_rate(mask, valid_mask=valid_mask, per_sample=per_sample)


def compute_topk_token_indices(
    x: Any,
    *,
    topk: int | None = None,
    topk_ratio: float = DEFAULT_SCALAR_OER_TOP_RATIO,
    valid_token_mask: Any | None = None,
    layout: str | None = None,
) -> Tensor:
    """Select FP32 token-norm top-k token ids, optionally restricted to valid tokens."""

    x_btc = normalize_activation_layout(x, layout=layout)
    valid = _normalize_valid_token_mask(valid_token_mask, x_btc)
    if valid is None:
        return token_energy_topk_indices(x_btc, topk=topk, topk_ratio=topk_ratio, batch_dim=0, token_dim=1, channel_dim=2)

    num_tokens = int(x_btc.shape[1])
    if topk is None:
        ratio = float(topk_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {ratio}")
        topk = int(math.ceil(float(num_tokens) * ratio))
    k = min(max(int(topk), 1), num_tokens)
    valid_counts = valid.sum(dim=1)
    if bool((valid_counts < int(k)).any().item()):
        counts = [int(item) for item in valid_counts.detach().cpu().tolist()]
        raise ValueError(f"valid_token_mask has fewer valid tokens than top-k={k}; valid counts={counts}")

    token_energy = x_btc.square().sum(dim=2)
    token_energy = token_energy.masked_fill(~valid, -torch.inf)
    return torch.topk(token_energy, k=int(k), dim=1, largest=True, sorted=False).indices


def _gather_topk_tokens(values: Tensor, topk_token_indices: Any) -> Tensor:
    return gather_token_values(values, topk_token_indices, batch_dim=0, token_dim=1)


def _gather_topk_token_valid_mask(topk_token_indices: Any, valid_token_mask: Any | None, reference: Tensor) -> Tensor | None:
    valid = _normalize_valid_token_mask(valid_token_mask, reference)
    if valid is None:
        return None
    return _gather_topk_tokens(valid, topk_token_indices)


def compute_topk_token_mask_rate(
    integer_mapping: Mapping[str, Any],
    topk_token_indices: Any,
    *,
    mask_name: str,
    valid_token_mask: Any | None = None,
    per_sample: bool = False,
) -> Tensor:
    mask = torch.as_tensor(integer_mapping[mask_name], dtype=torch.bool)
    selected = _gather_topk_tokens(mask, topk_token_indices)
    token_flag = selected.reshape(int(selected.shape[0]), int(selected.shape[1]), -1).any(dim=2)
    selected_valid = _gather_topk_token_valid_mask(topk_token_indices, valid_token_mask, mask)
    return boolean_mask_rate(token_flag, valid_mask=selected_valid, per_sample=per_sample)


def compute_high_energy_scalar_mask_rate(
    integer_mapping: Mapping[str, Any],
    topk_token_indices: Any,
    *,
    mask_name: str,
    valid_token_mask: Any | None = None,
    per_sample: bool = False,
) -> Tensor:
    mask = torch.as_tensor(integer_mapping[mask_name], dtype=torch.bool)
    selected = _gather_topk_tokens(mask, topk_token_indices)
    selected_valid = _gather_topk_token_valid_mask(topk_token_indices, valid_token_mask, mask)
    if selected_valid is not None:
        selected_valid = selected_valid.unsqueeze(-1).expand_as(selected)
    return boolean_mask_rate(selected, valid_mask=selected_valid, per_sample=per_sample)


def compute_topk_token_clip_rate(
    integer_mapping: Mapping[str, Any],
    topk_token_indices: Any,
    *,
    valid_token_mask: Any | None = None,
    per_sample: bool = False,
) -> Tensor:
    """Token-level clipping rate on fixed FP32 top-k tokens.

    A selected token counts as clipped when at least one scalar/channel inside
    that token hits the quantization boundary. The denominator is valid top-k
    tokens, not all scalars inside those tokens.
    """

    return compute_topk_token_mask_rate(integer_mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_token_mask, per_sample=per_sample)


def compute_high_energy_scalar_clip_rate(
    integer_mapping: Mapping[str, Any],
    topk_token_indices: Any,
    *,
    valid_token_mask: Any | None = None,
    per_sample: bool = False,
) -> Tensor:
    """Scalar clipping rate inside fixed FP32 token-norm top-k tokens."""

    return compute_high_energy_scalar_mask_rate(integer_mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_token_mask, per_sample=per_sample)


def compute_topk_token_near_zero_rate(
    integer_mapping: Mapping[str, Any],
    topk_token_indices: Any,
    *,
    valid_token_mask: Any | None = None,
    per_sample: bool = False,
) -> Tensor:
    """Scalar near-zero rate inside fixed FP32 token-norm top-k tokens."""

    return compute_topk_token_mask_rate(integer_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_token_mask, per_sample=per_sample)


def _gather_scalar_topk_mask(mask: Any, topk_indices: Any) -> Tensor:
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    if mask_t.ndim < 2:
        raise ValueError(f"scalar top-k mask must include batch and scalar dimensions, got {tuple(mask_t.shape)}")
    batch_size = int(mask_t.shape[0])
    flat = mask_t.reshape(batch_size, -1)
    indices = _as_long_tensor(topk_indices, device=flat.device)
    if indices.ndim == 1:
        indices = indices.reshape(batch_size, -1)
    else:
        indices = indices.reshape(int(indices.shape[0]), -1)
    if int(indices.shape[0]) != batch_size:
        raise ValueError(f"topk_indices batch size must equal mask batch size, got {int(indices.shape[0])} vs {batch_size}")
    if int(indices.numel()) == 0:
        raise ValueError("topk_indices must not be empty")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(flat.shape[1]):
        raise ValueError("topk_indices contains out-of-range scalar positions")
    return torch.gather(flat, dim=1, index=indices)


def compute_high_energy_mask_rate(
    integer_mapping: Mapping[str, Any],
    topk_indices: Any,
    *,
    mask_name: str,
    per_sample: bool = False,
) -> Tensor:
    """Rate of a danger mask on fixed FP32 OER-oracle scalar positions."""

    selected = _gather_scalar_topk_mask(integer_mapping[mask_name], topk_indices)
    return boolean_mask_rate(selected, per_sample=per_sample)


def compute_high_energy_near_zero_rate(
    integer_mapping: Mapping[str, Any],
    topk_indices: Any,
    *,
    per_sample: bool = False,
) -> Tensor:
    """Near-zero swallow rate on fixed FP32 OER-oracle scalar positions."""

    return compute_high_energy_mask_rate(integer_mapping, topk_indices, mask_name="near_zero_pm1_mask", per_sample=per_sample)


def compute_high_energy_clip_rate(
    integer_mapping: Mapping[str, Any],
    topk_indices: Any,
    *,
    per_sample: bool = False,
) -> Tensor:
    """Boundary-hit rate on fixed FP32 OER-oracle scalar positions."""

    return compute_high_energy_mask_rate(integer_mapping, topk_indices, mask_name="boundary_hit_mask", per_sample=per_sample)

def compute_masked_energy_share(
    x: Any,
    mask: Any,
    *,
    valid_mask: Any | None = None,
    per_sample: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Fraction of FP32 activation energy covered by a danger mask."""

    x_btc = normalize_activation_layout(x)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=x_btc.device)
    if tuple(mask_t.shape) != tuple(x_btc.shape):
        raise ValueError(f"mask shape must match activation shape, got {tuple(mask_t.shape)} vs {tuple(x_btc.shape)}")
    included = torch.ones_like(mask_t, dtype=torch.bool)
    if valid_mask is not None:
        valid = _expand_valid_token_mask(valid_mask, x_btc)
        if valid is None:
            raise ValueError("valid_mask normalization unexpectedly returned None")
        included = included & valid
    energy = x_btc.square()
    if per_sample:
        flat_energy = energy.reshape(int(energy.shape[0]), -1)
        flat_mask = (mask_t & included).reshape(int(energy.shape[0]), -1)
        flat_included = included.reshape(int(energy.shape[0]), -1)
        numerator = flat_energy.masked_fill(~flat_mask, 0.0).sum(dim=1)
        denominator = flat_energy.masked_fill(~flat_included, 0.0).sum(dim=1)
        return torch.where(
            denominator > float(eps),
            numerator / torch.clamp(denominator, min=float(eps)),
            torch.full_like(denominator, float("nan")),
        )
    numerator = energy.masked_fill(~(mask_t & included), 0.0).sum()
    denominator = energy.masked_fill(~included, 0.0).sum()
    if float(denominator.item()) <= float(eps):
        return energy.new_tensor(float("nan"))
    return numerator / torch.clamp(denominator, min=float(eps))


def compute_masked_energy_loss_share(
    x: Any,
    x_q: Any,
    mask: Any,
    *,
    valid_mask: Any | None = None,
    per_sample: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Fraction of FP32 energy represented by positive masked energy loss."""

    x_btc = normalize_activation_layout(x)
    xq_btc = normalize_activation_layout(x_q)
    if tuple(x_btc.shape) != tuple(xq_btc.shape):
        raise ValueError(f"x and x_q shape must match, got {tuple(x_btc.shape)} vs {tuple(xq_btc.shape)}")
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=x_btc.device)
    if tuple(mask_t.shape) != tuple(x_btc.shape):
        raise ValueError(f"mask shape must match activation shape, got {tuple(mask_t.shape)} vs {tuple(x_btc.shape)}")
    included = torch.ones_like(mask_t, dtype=torch.bool)
    if valid_mask is not None:
        valid = _expand_valid_token_mask(valid_mask, x_btc)
        if valid is None:
            raise ValueError("valid_mask normalization unexpectedly returned None")
        included = included & valid
    positive_loss = (x_btc.square() - xq_btc.square()).clamp_min(0.0)
    denominator_energy = x_btc.square()
    if per_sample:
        flat_loss = positive_loss.reshape(int(x_btc.shape[0]), -1)
        flat_energy = denominator_energy.reshape(int(x_btc.shape[0]), -1)
        flat_mask = (mask_t & included).reshape(int(x_btc.shape[0]), -1)
        flat_included = included.reshape(int(x_btc.shape[0]), -1)
        numerator = flat_loss.masked_fill(~flat_mask, 0.0).sum(dim=1)
        denominator = flat_energy.masked_fill(~flat_included, 0.0).sum(dim=1)
        return torch.where(
            denominator > float(eps),
            numerator / torch.clamp(denominator, min=float(eps)),
            torch.full_like(denominator, float("nan")),
        )
    numerator = positive_loss.masked_fill(~(mask_t & included), 0.0).sum()
    denominator = denominator_energy.masked_fill(~included, 0.0).sum()
    if float(denominator.item()) <= float(eps):
        return x_btc.new_tensor(float("nan"))
    return numerator / torch.clamp(denominator, min=float(eps))


def compute_topk_token_masked_energy_share(
    x: Any,
    integer_mapping: Mapping[str, Any],
    topk_token_indices: Any,
    *,
    mask_name: str = "clip_mask",
    valid_token_mask: Any | None = None,
    per_sample: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Energy share inside FP32 high-energy tokens covered by a danger mask."""

    x_btc = normalize_activation_layout(x)
    mask = torch.as_tensor(integer_mapping[mask_name], dtype=torch.bool, device=x_btc.device)
    selected_x = _gather_topk_tokens(x_btc, topk_token_indices)
    selected_mask = _gather_topk_tokens(mask, topk_token_indices)
    selected_valid = _gather_topk_token_valid_mask(topk_token_indices, valid_token_mask, x_btc)
    return compute_masked_energy_share(selected_x, selected_mask, valid_mask=selected_valid, per_sample=per_sample, eps=eps)


def compute_topk_token_masked_energy_loss_share(
    x: Any,
    integer_mapping: Mapping[str, Any],
    topk_token_indices: Any,
    *,
    mask_name: str = "clip_mask",
    valid_token_mask: Any | None = None,
    per_sample: bool = False,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Positive energy-loss share inside FP32 top-k tokens covered by a danger mask."""

    x_btc = normalize_activation_layout(x)
    x_q = _as_float_tensor(integer_mapping["x_q"]).to(device=x_btc.device)
    mask = torch.as_tensor(integer_mapping[mask_name], dtype=torch.bool, device=x_btc.device)
    selected_x = _gather_topk_tokens(x_btc, topk_token_indices)
    selected_xq = _gather_topk_tokens(x_q, topk_token_indices)
    selected_mask = _gather_topk_tokens(mask, topk_token_indices)
    selected_valid = _gather_topk_token_valid_mask(topk_token_indices, valid_token_mask, x_btc)
    return compute_masked_energy_loss_share(selected_x, selected_xq, selected_mask, valid_mask=selected_valid, per_sample=per_sample, eps=eps)

def _scalar_oer_topk_indices_with_valid_mask(
    x_btc: Tensor,
    *,
    topk_indices: Any | None = None,
    topk_ratio: float = DEFAULT_SCALAR_OER_TOP_RATIO,
    valid_token_mask: Any | None = None,
) -> Tensor | None:
    if topk_indices is not None:
        return _as_long_tensor(topk_indices, device=x_btc.device)
    valid = _normalize_valid_token_mask(valid_token_mask, x_btc)
    if valid is None:
        return None
    flat = x_btc.reshape(int(x_btc.shape[0]), -1)
    valid_scalar = valid.unsqueeze(-1).expand_as(x_btc).reshape(int(x_btc.shape[0]), -1)
    ratio = float(topk_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {ratio}")
    k = min(max(int(math.ceil(float(flat.shape[1]) * ratio)), 1), int(flat.shape[1]))
    valid_counts = valid_scalar.sum(dim=1)
    if bool((valid_counts < int(k)).any().item()):
        counts = [int(item) for item in valid_counts.detach().cpu().tolist()]
        raise ValueError(f"valid_token_mask has fewer valid scalars than scalar OER top-k={k}; valid scalar counts={counts}")
    energy = flat.square().masked_fill(~valid_scalar, -torch.inf)
    return torch.topk(energy, k=int(k), dim=1, largest=True, sorted=False).indices


def compute_oer_retention_from_integer_mapping(
    x: Any,
    integer_mapping: Mapping[str, Any],
    *,
    topk_indices: Any | None = None,
    topk_ratio: float = DEFAULT_SCALAR_OER_TOP_RATIO,
    valid_token_mask: Any | None = None,
    eps: float = DEFAULT_EPS,
    reduction: str = "mean",
) -> dict[str, Tensor]:
    """Compute scalar-level OER retention from a fixed FP32 oracle."""

    x_btc = normalize_activation_layout(x)
    x_q = _as_float_tensor(integer_mapping["x_q"]).to(device=x_btc.device)
    if tuple(x_btc.shape) != tuple(x_q.shape):
        raise ValueError(f"x and integer_mapping['x_q'] shape mismatch: {tuple(x_btc.shape)} vs {tuple(x_q.shape)}")
    resolved_topk = _scalar_oer_topk_indices_with_valid_mask(
        x_btc,
        topk_indices=topk_indices,
        topk_ratio=topk_ratio,
        valid_token_mask=valid_token_mask,
    )
    metrics = scalar_oer_energy_metrics(
        x_btc,
        x_q,
        topk_indices=resolved_topk,
        topk_ratio=topk_ratio,
        batch_dim=0,
        reduction=reduction,  # type: ignore[arg-type]
        eps=eps,
    )
    if resolved_topk is None:
        resolved_topk = scalar_oer_topk_indices(x_btc, topk_ratio=topk_ratio, batch_dim=0)
    metrics["topk_indices"] = _as_long_tensor(resolved_topk).detach().cpu()
    return metrics


def _token_energy_for_indices(x_btc: Tensor, topk_token_indices: Tensor) -> Tensor:
    token_energy = x_btc.detach().float().square().sum(dim=-1)
    return torch.gather(token_energy, dim=1, index=topk_token_indices.to(device=token_energy.device))


def _token_any_mask_for_indices(mask_btc: Tensor, topk_token_indices: Tensor) -> Tensor:
    token_mask = mask_btc.detach().bool().any(dim=-1)
    return torch.gather(token_mask, dim=1, index=topk_token_indices.to(device=token_mask.device))


def _compact_topk_token_payload(
    fp32_btc: Tensor,
    quant_btc: Tensor,
    integer_mapping: Mapping[str, Any],
    topk_token_indices: Tensor,
) -> dict[str, Tensor]:
    return {
        "topk_token_indices": topk_token_indices.detach().cpu(),
        "topk_token_energy_fp32": _token_energy_for_indices(fp32_btc, topk_token_indices).detach().cpu().float(),
        "topk_token_energy_quant": _token_energy_for_indices(quant_btc, topk_token_indices).detach().cpu().float(),
        "topk_token_boundary_hit_flags": _token_any_mask_for_indices(
            torch.as_tensor(integer_mapping["boundary_hit_mask"], dtype=torch.bool, device=fp32_btc.device),
            topk_token_indices,
        ).detach().cpu(),
        "topk_token_strict_overflow_flags": _token_any_mask_for_indices(
            torch.as_tensor(integer_mapping["strict_overflow_mask"], dtype=torch.bool, device=fp32_btc.device),
            topk_token_indices,
        ).detach().cpu(),
        "topk_token_near_zero_pm1_flags": _token_any_mask_for_indices(
            torch.as_tensor(integer_mapping["near_zero_pm1_mask"], dtype=torch.bool, device=fp32_btc.device),
            topk_token_indices,
        ).detach().cpu(),
    }


def compute_direct_integer_grid_danger_metrics(
    fp32_activation: Any,
    quant_activation: Any,
    integer_grid: Mapping[str, Any],
    *,
    topk_ratio: float = DEFAULT_SCALAR_OER_TOP_RATIO,
    topk_token_indices: Any | None = None,
    oer_topk_indices: Any | None = None,
    valid_token_mask: Any | None = None,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Compute activation danger metrics from QuantLinear forward diagnostics."""

    fp32_btc = normalize_activation_layout(fp32_activation)
    quant_btc = normalize_activation_layout(quant_activation).to(device=fp32_btc.device)
    if tuple(fp32_btc.shape) != tuple(quant_btc.shape):
        raise ValueError(f"fp32/quant activation shape mismatch: {tuple(fp32_btc.shape)} vs {tuple(quant_btc.shape)}")
    valid_tokens = _normalize_valid_token_mask(valid_token_mask, fp32_btc)
    if valid_tokens is None and "valid_token_mask" in integer_grid:
        valid_tokens = _normalize_valid_token_mask(integer_grid["valid_token_mask"], fp32_btc)
    valid_scalar_mask = _expand_valid_token_mask(valid_tokens, fp32_btc)
    valid_token_count = int(valid_tokens.sum().item()) if valid_tokens is not None else int(fp32_btc.shape[0] * fp32_btc.shape[1])
    total_token_count = int(fp32_btc.shape[0] * fp32_btc.shape[1])

    qmin_value = int(integer_grid.get("qmin", DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN))
    qmax_value = int(integer_grid.get("qmax", DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX))
    boundary_mask = torch.as_tensor(integer_grid["boundary_hit_mask"], dtype=torch.bool, device=fp32_btc.device)
    if "q_preclip" in integer_grid:
        q_preclip = _as_float_tensor(integer_grid["q_preclip"]).to(device=fp32_btc.device)
        low_clip_mask = q_preclip <= float(qmin_value)
        high_clip_mask = q_preclip >= float(qmax_value)
    elif "q_int" in integer_grid:
        q_int = _as_float_tensor(integer_grid["q_int"]).to(device=fp32_btc.device)
        low_clip_mask = q_int <= float(qmin_value)
        high_clip_mask = q_int >= float(qmax_value)
    else:
        low_clip_mask = boundary_mask
        high_clip_mask = boundary_mask
    mapping = {
        "x_q": quant_btc,
        "low_clip_mask": low_clip_mask,
        "high_clip_mask": high_clip_mask,
        "boundary_hit_mask": boundary_mask,
        "strict_overflow_mask": torch.as_tensor(integer_grid["strict_overflow_mask"], dtype=torch.bool, device=fp32_btc.device),
        "exact_zero_mask": torch.as_tensor(integer_grid["exact_zero_mask"], dtype=torch.bool, device=fp32_btc.device),
        "near_zero_pm1_mask": torch.as_tensor(integer_grid["near_zero_pm1_mask"], dtype=torch.bool, device=fp32_btc.device),
        "near_zero_mask": torch.as_tensor(integer_grid["near_zero_pm1_mask"], dtype=torch.bool, device=fp32_btc.device),
        "clip_mask": boundary_mask,
        "scale": _as_float_tensor(integer_grid.get("scale", 1.0)).to(device=fp32_btc.device),
        "zero_point": _as_float_tensor(integer_grid.get("zero_point", 0.0)).to(device=fp32_btc.device),
    }
    for name in ("boundary_hit_mask", "strict_overflow_mask", "exact_zero_mask", "near_zero_pm1_mask"):
        if tuple(mapping[name].shape) != tuple(fp32_btc.shape):
            raise ValueError(f"integer_grid[{name!r}] shape mismatch: {tuple(mapping[name].shape)} vs {tuple(fp32_btc.shape)}")

    if topk_token_indices is None:
        topk_token_indices = compute_topk_token_indices(fp32_btc, topk_ratio=topk_ratio, valid_token_mask=valid_tokens)
        key_token_source = "fp32_token_energy_topk"
    else:
        topk_token_indices = _as_long_tensor(topk_token_indices, device=fp32_btc.device).reshape(int(fp32_btc.shape[0]), -1)
        key_token_source = "fp32_oer_oracle.oer_topk_token_indices_aux"
    scalar_oer_topk = _scalar_oer_topk_indices_with_valid_mask(
        fp32_btc,
        topk_indices=oer_topk_indices,
        topk_ratio=topk_ratio,
        valid_token_mask=valid_tokens,
    )
    if scalar_oer_topk is None:
        scalar_oer_topk = scalar_oer_topk_indices(fp32_btc, topk_ratio=topk_ratio, batch_dim=0)
    clip = compute_clip_rates(mapping, valid_mask=valid_scalar_mask)
    exact_zero = compute_exact_zero_rate(mapping, valid_mask=valid_scalar_mask)
    near_zero = compute_near_zero_rate(mapping, valid_mask=valid_scalar_mask)
    oer = compute_oer_retention_from_integer_mapping(
        fp32_btc,
        mapping,
        topk_indices=scalar_oer_topk,
        topk_ratio=topk_ratio,
        valid_token_mask=valid_tokens,
        eps=eps,
        reduction="mean",
    )

    return {
        "activation_shape": tuple(int(item) for item in fp32_btc.shape),
        "qmin": qmin_value,
        "qmax": qmax_value,
        "topk_ratio": float(topk_ratio),
        "valid_token_count": int(valid_token_count),
        "total_token_count": int(total_token_count),
        "padding_token_ratio": float((total_token_count - valid_token_count) / total_token_count) if total_token_count > 0 else float("nan"),
        "valid_token_mask_applied": bool(valid_tokens is not None),
        "scale": mapping["scale"].detach().cpu().float(),
        "zero_point": mapping["zero_point"].detach().cpu().float(),
        "boundary_hit_rate": clip["boundary_hit_rate"],
        "clip_rate": clip["clip_rate"],
        "clip_low_rate": clip["clip_low_rate"],
        "clip_high_rate": clip["clip_high_rate"],
        "strict_overflow_rate": clip["strict_overflow_rate"],
        "exact_zero_rate": exact_zero,
        "near_zero_pm1_rate": near_zero,
        "near_zero_rate": near_zero,
        "topk_token_exact_zero_rate": compute_topk_token_mask_rate(mapping, topk_token_indices, mask_name="exact_zero_mask", valid_token_mask=valid_tokens),
        "topk_token_near_zero_pm1_rate": compute_topk_token_mask_rate(mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens),
        "topk_token_boundary_hit_rate": compute_topk_token_mask_rate(mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens),
        "topk_token_strict_overflow_rate": compute_topk_token_mask_rate(mapping, topk_token_indices, mask_name="strict_overflow_mask", valid_token_mask=valid_tokens),
        "high_energy_scalar_exact_zero_rate": compute_high_energy_scalar_mask_rate(mapping, topk_token_indices, mask_name="exact_zero_mask", valid_token_mask=valid_tokens),
        "high_energy_scalar_near_zero_pm1_rate": compute_high_energy_scalar_mask_rate(mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens),
        "high_energy_scalar_boundary_hit_rate": compute_high_energy_scalar_mask_rate(mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens),
        "high_energy_scalar_strict_overflow_rate": compute_high_energy_scalar_mask_rate(mapping, topk_token_indices, mask_name="strict_overflow_mask", valid_token_mask=valid_tokens),
        "oer_scalar_clip_rate": compute_high_energy_clip_rate(mapping, scalar_oer_topk),
        "high_energy_near_zero_rate": compute_high_energy_near_zero_rate(mapping, scalar_oer_topk),
        "saturation_clip_energy_share": compute_masked_energy_share(fp32_btc, mapping["boundary_hit_mask"], valid_mask=valid_tokens, eps=eps),
        "saturation_boundary_energy_share": compute_masked_energy_share(fp32_btc, mapping["boundary_hit_mask"], valid_mask=valid_tokens, eps=eps),
        "overflow_energy_proxy": compute_masked_energy_share(fp32_btc, mapping["strict_overflow_mask"], valid_mask=valid_tokens, eps=eps),
        "near_zero_swallow_energy_share": compute_masked_energy_share(fp32_btc, mapping["near_zero_pm1_mask"], valid_mask=valid_tokens, eps=eps),
        "near_zero_swallow_loss_share": compute_masked_energy_loss_share(fp32_btc, quant_btc, mapping["near_zero_pm1_mask"], valid_mask=valid_tokens, eps=eps),
        "key_token_clip_energy_share": compute_topk_token_masked_energy_share(fp32_btc, mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens, eps=eps),
        "key_token_near_zero_energy_share": compute_topk_token_masked_energy_share(fp32_btc, mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens, eps=eps),
        "key_token_near_zero_loss_share": compute_topk_token_masked_energy_loss_share(fp32_btc, mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens, eps=eps),
        "oer_retention": oer["oer_retention"],
        "oer_loss_total": oer["oer_loss"],
        "oer_loss": oer["oer_loss"],
        "off_oracle_energy_ratio": oer["off_oracle_energy_ratio"],
        "oer_topk_indices": oer["topk_indices"],
        "key_token_source": key_token_source,
        "aux_token_source": "fp32_oer_oracle.topk_scalar_indices" if oer_topk_indices is not None else "scalar_oer_topk_aggregated_to_token",
        "oer_oracle_source": "fp32_oer_oracle.topk_scalar_indices" if oer_topk_indices is not None else "runtime_fp32_scalar_oer_topk",
        "compact_topk": _compact_topk_token_payload(fp32_btc, quant_btc, mapping, topk_token_indices),
        "integer_grid_source": str(integer_grid.get("integer_grid_source", "quantlinear_forward")),
        "is_direct_quantlinear_forward_grid": bool(integer_grid.get("is_direct_quantlinear_forward_grid", True)),
    }


def compute_projection_damage_metrics(
    fp32_projection: Any,
    quant_projection: Any,
    *,
    topk_ratio: float = DEFAULT_SCALAR_OER_TOP_RATIO,
    oer_topk_indices: Any | None = None,
    valid_token_mask: Any | None = None,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Compute projection-level distortion without activation integer-grid fields."""

    fp32_btc = normalize_activation_layout(fp32_projection)
    quant_btc = normalize_activation_layout(quant_projection).to(device=fp32_btc.device)
    if tuple(fp32_btc.shape) != tuple(quant_btc.shape):
        raise ValueError(f"fp32/quant projection shape mismatch: {tuple(fp32_btc.shape)} vs {tuple(quant_btc.shape)}")
    valid_tokens = _normalize_valid_token_mask(valid_token_mask, fp32_btc)
    valid_scalar = _expand_valid_token_mask(valid_tokens, fp32_btc)
    if valid_scalar is not None:
        fp32_values = fp32_btc[valid_scalar]
        quant_values = quant_btc[valid_scalar]
    else:
        fp32_values = fp32_btc.reshape(-1)
        quant_values = quant_btc.reshape(-1)
    diff = quant_values.float() - fp32_values.float()
    mse = diff.square().mean() if int(diff.numel()) else torch.tensor(float("nan"), device=fp32_btc.device)
    rel_l2 = torch.linalg.vector_norm(diff, ord=2) / torch.linalg.vector_norm(fp32_values.float(), ord=2).clamp_min(float(eps))
    cos = torch.nn.functional.cosine_similarity(
        fp32_values.float().reshape(1, -1),
        quant_values.float().reshape(1, -1),
        dim=1,
        eps=float(eps),
    ).reshape(())
    fp32_energy = fp32_values.float().square().sum()
    quant_energy = quant_values.float().square().sum()
    topk_indices = _scalar_oer_topk_indices_with_valid_mask(
        fp32_btc,
        topk_indices=oer_topk_indices,
        topk_ratio=topk_ratio,
        valid_token_mask=valid_tokens,
    )
    oer = scalar_oer_energy_metrics(
        fp32_btc,
        quant_btc,
        topk_indices=topk_indices,
        topk_ratio=topk_ratio,
        batch_dim=0,
        reduction="mean",
        eps=eps,
    )
    return {
        "activation_shape": tuple(int(item) for item in fp32_btc.shape),
        "valid_token_count": int(valid_tokens.sum().item()) if valid_tokens is not None else int(fp32_btc.shape[0] * fp32_btc.shape[1]),
        "total_token_count": int(fp32_btc.shape[0] * fp32_btc.shape[1]),
        "padding_token_ratio": float(0.0) if valid_tokens is None else float((valid_tokens.numel() - valid_tokens.sum().item()) / max(int(valid_tokens.numel()), 1)),
        "valid_token_mask_applied": bool(valid_tokens is not None),
        "projection_rel_l2_error": rel_l2.detach().cpu().float(),
        "projection_cosine_drift": (1.0 - cos).detach().cpu().float(),
        "projection_mse": mse.detach().cpu().float(),
        "projection_oer_shift": (1.0 - oer["oer_retention"]).detach().cpu().float(),
        "projection_oer_retention": oer["oer_retention"].detach().cpu().float(),
        "projection_oer_loss": oer["oer_loss"].detach().cpu().float(),
        "projection_off_oracle_energy_ratio": oer["off_oracle_energy_ratio"].detach().cpu().float(),
        "output_energy_distortion": ((quant_energy - fp32_energy) / fp32_energy.clamp_min(float(eps))).detach().cpu().float(),
        "oer_topk_indices": _as_long_tensor(topk_indices).detach().cpu() if topk_indices is not None else scalar_oer_topk_indices(fp32_btc, topk_ratio=topk_ratio, batch_dim=0).detach().cpu(),
        "oer_oracle_source": "fp32_oer_oracle.topk_scalar_indices" if oer_topk_indices is not None else "runtime_fp32_scalar_oer_topk",
    }


def compute_single_activation_danger_metrics(
    x: Any,
    *,
    percentile: float = 0.999,
    topk_ratio: float = DEFAULT_SCALAR_OER_TOP_RATIO,
    qmin: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    zero_point: Any = 0.0,
    valid_token_mask: Any | None = None,
    valid_token_policy: str = "require_mask",
    return_intermediates: bool = False,
) -> dict[str, Any]:
    """Compute single-tensor danger-zone diagnostics.

    Group aggregation, tail gaps, and OER-loss correlations are intentionally
    left to quant_danger_zone_summary.py.
    """

    require_valid_token_policy(
        padding_mask=None if valid_token_mask is None else torch.as_tensor(valid_token_mask),
        valid_token_policy=valid_token_policy,
        context="quant_danger_zone_metrics",
    )
    x_btc = normalize_activation_layout(x)
    valid_tokens = _normalize_valid_token_mask(valid_token_mask, x_btc)
    valid_scalar_mask = _expand_valid_token_mask(valid_tokens, x_btc)
    valid_token_count = int(valid_tokens.sum().item()) if valid_tokens is not None else int(x_btc.shape[0] * x_btc.shape[1])
    total_token_count = int(x_btc.shape[0] * x_btc.shape[1])

    per_tensor_scale = compute_per_tensor_scale(x_btc, percentile=percentile, qmax=qmax)
    per_channel_scale = compute_per_channel_scale(x_btc, percentile=percentile, qmax=qmax)
    tensor_mapping = compute_a4_integer_mapping(
        x_btc,
        scale=per_tensor_scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
    )
    channel_mapping = compute_a4_integer_mapping(
        x_btc,
        scale=per_channel_scale,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
    )

    topk_token_indices = compute_topk_token_indices(x_btc, topk_ratio=topk_ratio, valid_token_mask=valid_tokens)
    tensor_oer = compute_oer_retention_from_integer_mapping(x_btc, tensor_mapping, topk_ratio=topk_ratio, valid_token_mask=valid_tokens, reduction="mean")
    channel_oer = compute_oer_retention_from_integer_mapping(x_btc, channel_mapping, topk_ratio=topk_ratio, valid_token_mask=valid_tokens, reduction="mean")
    tensor_oer_topk = tensor_oer["topk_indices"]
    channel_oer_topk = channel_oer["topk_indices"]

    tensor_clip = compute_clip_rates(tensor_mapping, valid_mask=valid_scalar_mask)
    channel_clip = compute_clip_rates(channel_mapping, valid_mask=valid_scalar_mask)
    tensor_exact_zero = compute_exact_zero_rate(tensor_mapping, valid_mask=valid_scalar_mask)
    channel_exact_zero = compute_exact_zero_rate(channel_mapping, valid_mask=valid_scalar_mask)
    tensor_near_zero = compute_near_zero_rate(tensor_mapping, valid_mask=valid_scalar_mask)
    channel_near_zero = compute_near_zero_rate(channel_mapping, valid_mask=valid_scalar_mask)
    tensor_topk_exact_zero = compute_topk_token_mask_rate(tensor_mapping, topk_token_indices, mask_name="exact_zero_mask", valid_token_mask=valid_tokens)
    channel_topk_exact_zero = compute_topk_token_mask_rate(channel_mapping, topk_token_indices, mask_name="exact_zero_mask", valid_token_mask=valid_tokens)
    tensor_topk_near_zero = compute_topk_token_mask_rate(tensor_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens)
    channel_topk_near_zero = compute_topk_token_mask_rate(channel_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens)
    tensor_topk_boundary_hit = compute_topk_token_mask_rate(tensor_mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens)
    channel_topk_boundary_hit = compute_topk_token_mask_rate(channel_mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens)
    tensor_topk_strict_overflow = compute_topk_token_mask_rate(tensor_mapping, topk_token_indices, mask_name="strict_overflow_mask", valid_token_mask=valid_tokens)
    channel_topk_strict_overflow = compute_topk_token_mask_rate(channel_mapping, topk_token_indices, mask_name="strict_overflow_mask", valid_token_mask=valid_tokens)
    tensor_high_energy_scalar_exact_zero = compute_high_energy_scalar_mask_rate(tensor_mapping, topk_token_indices, mask_name="exact_zero_mask", valid_token_mask=valid_tokens)
    channel_high_energy_scalar_exact_zero = compute_high_energy_scalar_mask_rate(channel_mapping, topk_token_indices, mask_name="exact_zero_mask", valid_token_mask=valid_tokens)
    tensor_high_energy_scalar_near_zero = compute_high_energy_scalar_mask_rate(tensor_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens)
    channel_high_energy_scalar_near_zero = compute_high_energy_scalar_mask_rate(channel_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens)
    tensor_high_energy_scalar_boundary_hit = compute_high_energy_scalar_mask_rate(tensor_mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens)
    channel_high_energy_scalar_boundary_hit = compute_high_energy_scalar_mask_rate(channel_mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens)
    tensor_high_energy_scalar_strict_overflow = compute_high_energy_scalar_mask_rate(tensor_mapping, topk_token_indices, mask_name="strict_overflow_mask", valid_token_mask=valid_tokens)
    channel_high_energy_scalar_strict_overflow = compute_high_energy_scalar_mask_rate(channel_mapping, topk_token_indices, mask_name="strict_overflow_mask", valid_token_mask=valid_tokens)

    tensor_oer_scalar_clip = compute_high_energy_clip_rate(tensor_mapping, tensor_oer_topk)
    channel_oer_scalar_clip = compute_high_energy_clip_rate(channel_mapping, channel_oer_topk)
    tensor_high_energy_near_zero = compute_high_energy_near_zero_rate(tensor_mapping, tensor_oer_topk)
    channel_high_energy_near_zero = compute_high_energy_near_zero_rate(channel_mapping, channel_oer_topk)

    tensor_boundary_energy_share = compute_masked_energy_share(x_btc, tensor_mapping["boundary_hit_mask"], valid_mask=valid_tokens)
    channel_boundary_energy_share = compute_masked_energy_share(x_btc, channel_mapping["boundary_hit_mask"], valid_mask=valid_tokens)
    tensor_overflow_energy_proxy = compute_masked_energy_share(x_btc, tensor_mapping["strict_overflow_mask"], valid_mask=valid_tokens)
    channel_overflow_energy_proxy = compute_masked_energy_share(x_btc, channel_mapping["strict_overflow_mask"], valid_mask=valid_tokens)
    tensor_near_zero_energy_share = compute_masked_energy_share(x_btc, tensor_mapping["near_zero_pm1_mask"], valid_mask=valid_tokens)
    channel_near_zero_energy_share = compute_masked_energy_share(x_btc, channel_mapping["near_zero_pm1_mask"], valid_mask=valid_tokens)
    tensor_near_zero_loss_share = compute_masked_energy_loss_share(x_btc, tensor_mapping["x_q"], tensor_mapping["near_zero_pm1_mask"], valid_mask=valid_tokens)
    channel_near_zero_loss_share = compute_masked_energy_loss_share(x_btc, channel_mapping["x_q"], channel_mapping["near_zero_pm1_mask"], valid_mask=valid_tokens)
    tensor_topk_clip_energy_share = compute_topk_token_masked_energy_share(x_btc, tensor_mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens)
    channel_topk_clip_energy_share = compute_topk_token_masked_energy_share(x_btc, channel_mapping, topk_token_indices, mask_name="boundary_hit_mask", valid_token_mask=valid_tokens)
    tensor_topk_near_zero_energy_share = compute_topk_token_masked_energy_share(x_btc, tensor_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens)
    channel_topk_near_zero_energy_share = compute_topk_token_masked_energy_share(x_btc, channel_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens)
    tensor_topk_near_zero_loss_share = compute_topk_token_masked_energy_loss_share(x_btc, tensor_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens)
    channel_topk_near_zero_loss_share = compute_topk_token_masked_energy_loss_share(x_btc, channel_mapping, topk_token_indices, mask_name="near_zero_pm1_mask", valid_token_mask=valid_tokens)

    result: dict[str, Any] = {
        "activation_shape": tuple(int(item) for item in x_btc.shape),
        "qmin": int(qmin),
        "qmax": int(qmax),
        "percentile": float(percentile),
        "topk_ratio": float(topk_ratio),
        "valid_token_count": int(valid_token_count),
        "total_token_count": int(total_token_count),
        "padding_token_ratio": float((total_token_count - valid_token_count) / total_token_count) if total_token_count > 0 else float("nan"),
        "valid_token_mask_applied": bool(valid_tokens is not None),
        "per_tensor_scale": per_tensor_scale,
        "per_channel_scale": per_channel_scale,
        "per_channel_scale_mean": per_channel_scale.float().mean(),
        "per_channel_scale_std": per_channel_scale.float().std(unbiased=False),
        "scale_ratio_mean": (per_channel_scale / torch.clamp(per_tensor_scale, min=float(DEFAULT_EPS))).float().mean(),
        "scale_ratio_std": (per_channel_scale / torch.clamp(per_tensor_scale, min=float(DEFAULT_EPS))).float().std(unbiased=False),
        "boundary_hit_rate_per_tensor": tensor_clip["boundary_hit_rate"],
        "clip_rate_per_tensor": tensor_clip["clip_rate"],
        "clip_low_rate_per_tensor": tensor_clip["clip_low_rate"],
        "clip_high_rate_per_tensor": tensor_clip["clip_high_rate"],
        "strict_overflow_rate_per_tensor": tensor_clip["strict_overflow_rate"],
        "saturation_boundary_hit_rate_per_tensor": tensor_clip["boundary_hit_rate"],
        "saturation_strict_overflow_rate_per_tensor": tensor_clip["strict_overflow_rate"],
        "saturation_clip_rate_per_tensor": tensor_clip["clip_rate"],
        "saturation_overflow_rate_per_tensor": tensor_clip["strict_overflow_rate"],
        "boundary_hit_rate_per_channel": channel_clip["boundary_hit_rate"],
        "clip_rate_per_channel": channel_clip["clip_rate"],
        "clip_low_rate_per_channel": channel_clip["clip_low_rate"],
        "clip_high_rate_per_channel": channel_clip["clip_high_rate"],
        "strict_overflow_rate_per_channel": channel_clip["strict_overflow_rate"],
        "saturation_boundary_hit_rate_per_channel": channel_clip["boundary_hit_rate"],
        "saturation_strict_overflow_rate_per_channel": channel_clip["strict_overflow_rate"],
        "saturation_clip_rate_per_channel": channel_clip["clip_rate"],
        "saturation_overflow_rate_per_channel": channel_clip["strict_overflow_rate"],
        "exact_zero_rate_per_tensor": tensor_exact_zero,
        "exact_zero_rate_per_channel": channel_exact_zero,
        "near_zero_pm1_rate_per_tensor": tensor_near_zero,
        "near_zero_pm1_rate_per_channel": channel_near_zero,
        "near_zero_rate_per_tensor": tensor_near_zero,
        "near_zero_rate_per_channel": channel_near_zero,
        "high_energy_near_zero_rate_per_tensor": tensor_high_energy_near_zero,
        "high_energy_near_zero_rate_per_channel": channel_high_energy_near_zero,
        "oer_scalar_clip_rate_per_tensor": tensor_oer_scalar_clip,
        "oer_scalar_clip_rate_per_channel": channel_oer_scalar_clip,
        "topk_token_indices": topk_token_indices.detach().cpu(),
        "topk_token_exact_zero_rate_per_tensor": tensor_topk_exact_zero,
        "topk_token_exact_zero_rate_per_channel": channel_topk_exact_zero,
        "topk_token_near_zero_pm1_rate_per_tensor": tensor_topk_near_zero,
        "topk_token_near_zero_pm1_rate_per_channel": channel_topk_near_zero,
        "topk_token_boundary_hit_rate_per_tensor": tensor_topk_boundary_hit,
        "topk_token_boundary_hit_rate_per_channel": channel_topk_boundary_hit,
        "topk_token_strict_overflow_rate_per_tensor": tensor_topk_strict_overflow,
        "topk_token_strict_overflow_rate_per_channel": channel_topk_strict_overflow,
        "high_energy_scalar_exact_zero_rate_per_tensor": tensor_high_energy_scalar_exact_zero,
        "high_energy_scalar_exact_zero_rate_per_channel": channel_high_energy_scalar_exact_zero,
        "high_energy_scalar_near_zero_pm1_rate_per_tensor": tensor_high_energy_scalar_near_zero,
        "high_energy_scalar_near_zero_pm1_rate_per_channel": channel_high_energy_scalar_near_zero,
        "high_energy_scalar_boundary_hit_rate_per_tensor": tensor_high_energy_scalar_boundary_hit,
        "high_energy_scalar_boundary_hit_rate_per_channel": channel_high_energy_scalar_boundary_hit,
        "high_energy_scalar_strict_overflow_rate_per_tensor": tensor_high_energy_scalar_strict_overflow,
        "high_energy_scalar_strict_overflow_rate_per_channel": channel_high_energy_scalar_strict_overflow,
        "topk_token_clip_rate_per_tensor": tensor_topk_boundary_hit,
        "topk_token_clip_rate_per_channel": channel_topk_boundary_hit,
        "high_energy_scalar_clip_rate_per_tensor": tensor_high_energy_scalar_boundary_hit,
        "high_energy_scalar_clip_rate_per_channel": channel_high_energy_scalar_boundary_hit,
        "topk_token_near_zero_rate_per_tensor": tensor_topk_near_zero,
        "topk_token_near_zero_rate_per_channel": channel_topk_near_zero,
        "key_token_near_zero_swallow_rate_per_tensor": tensor_high_energy_near_zero,
        "key_token_near_zero_swallow_rate_per_channel": channel_high_energy_near_zero,
        "saturation_boundary_energy_share_per_tensor": tensor_boundary_energy_share,
        "saturation_boundary_energy_share_per_channel": channel_boundary_energy_share,
        "overflow_energy_proxy_per_tensor": tensor_overflow_energy_proxy,
        "overflow_energy_proxy_per_channel": channel_overflow_energy_proxy,
        "saturation_clip_energy_share_per_tensor": tensor_boundary_energy_share,
        "saturation_clip_energy_share_per_channel": channel_boundary_energy_share,
        "near_zero_swallow_energy_share_per_tensor": tensor_near_zero_energy_share,
        "near_zero_swallow_energy_share_per_channel": channel_near_zero_energy_share,
        "near_zero_swallow_loss_share_per_tensor": tensor_near_zero_loss_share,
        "near_zero_swallow_loss_share_per_channel": channel_near_zero_loss_share,
        "key_token_clip_energy_share_per_tensor": tensor_topk_clip_energy_share,
        "key_token_clip_energy_share_per_channel": channel_topk_clip_energy_share,
        "key_token_near_zero_energy_share_per_tensor": tensor_topk_near_zero_energy_share,
        "key_token_near_zero_energy_share_per_channel": channel_topk_near_zero_energy_share,
        "key_token_near_zero_loss_share_per_tensor": tensor_topk_near_zero_loss_share,
        "key_token_near_zero_loss_share_per_channel": channel_topk_near_zero_loss_share,
        "oer_retention_per_tensor": tensor_oer["oer_retention"],
        "oer_loss_total_per_tensor": tensor_oer["oer_loss"],
        "oer_loss_per_tensor": tensor_oer["oer_loss"],
        "off_oracle_energy_ratio_per_tensor": tensor_oer["off_oracle_energy_ratio"],
        "oer_retention_per_channel": channel_oer["oer_retention"],
        "oer_loss_total_per_channel": channel_oer["oer_loss"],
        "oer_loss_per_channel": channel_oer["oer_loss"],
        "off_oracle_energy_ratio_per_channel": channel_oer["off_oracle_energy_ratio"],
    }
    if return_intermediates:
        result["per_tensor_mapping"] = tensor_mapping
        result["per_channel_mapping"] = channel_mapping
        result["oer_topk_indices_per_tensor"] = tensor_oer_topk
        result["oer_topk_indices_per_channel"] = channel_oer_topk
        if valid_tokens is not None:
            result["valid_token_mask"] = valid_tokens.detach().cpu()
    return result


__all__ = [
    "compute_a4_integer_mapping",
    "compute_clip_rates",
    "compute_direct_integer_grid_danger_metrics",
    "compute_high_energy_clip_rate",
    "compute_high_energy_scalar_clip_rate",
    "compute_high_energy_scalar_mask_rate",
    "compute_exact_zero_rate",
    "compute_high_energy_mask_rate",
    "compute_high_energy_near_zero_rate",
    "compute_masked_energy_loss_share",
    "compute_masked_energy_share",
    "compute_near_zero_rate",
    "compute_projection_damage_metrics",
    "compute_oer_retention_from_integer_mapping",
    "compute_per_channel_scale",
    "compute_per_tensor_scale",
    "compute_single_activation_danger_metrics",
    "compute_topk_token_clip_rate",
    "compute_topk_token_mask_rate",
    "compute_topk_token_masked_energy_loss_share",
    "compute_topk_token_masked_energy_share",
    "compute_topk_token_indices",
    "compute_topk_token_near_zero_rate",
    "normalize_activation_layout",
]

