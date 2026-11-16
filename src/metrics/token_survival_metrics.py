from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

try:
    from .common_metrics import (
        boolean_mask_rate,
        embedding_distortion,
        gather_token_values,
    )
    from .metric_config import (
        DEFAULT_EPS,
        DEFAULT_SCALAR_OER_TOP_RATIO,
        DEFAULT_TOKEN_SURVIVAL_MIN_NORM_RETENTION,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    )
    from ..utils.group_utils import read_canonical_bucket, tail_included
    from ..utils.layout_utils import canonicalize_activation_btc, canonicalize_padding_mask_bt, ensure_canonical_btc
except ImportError:  # pragma: no cover - script execution fallback.
    from src.metrics.common_metrics import (
        boolean_mask_rate,
        embedding_distortion,
        gather_token_values,
    )
    from src.metrics.metric_config import (
        DEFAULT_EPS,
        DEFAULT_SCALAR_OER_TOP_RATIO,
        DEFAULT_TOKEN_SURVIVAL_MIN_NORM_RETENTION,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    )
    from src.utils.group_utils import read_canonical_bucket, tail_included
    from src.utils.layout_utils import canonicalize_activation_btc, canonicalize_padding_mask_bt, ensure_canonical_btc


SAMPLE_OUTPUT_COLUMNS = [
    "sample_id",
    "label",
    "bucket",
    "shot_group",
    "tail_included",
    "layer_idx",
    "module_name",
    "tensor_position",
    "quant_mode",
    "granularity",
    "activation_shape",
    "original_layout",
    "original_shape",
    "canonical_layout",
    "canonical_shape",
    "layout_inference_source",
    "padding_mask_available",
    "padding_mask_aligned",
    "mask_status",
    "valid_token_policy",
    "integer_mapping_source",
    "num_tokens",
    "valid_token_count",
    "hidden_dim",
    "topk",
    "topk_k",
    "survival_min_norm_retention",
    "fp32_topk_token_ids",
    "int4_topk_token_ids",
    "topk_overlap",
    "topk_survival_rate",
    "token_survival_drop",
    "topk_norm_retention_values",
    "mean_norm_retention",
    "mean_token_distortion",
    "topk_token_exact_zero_rate",
    "topk_token_near_zero_pm1_rate",
    "topk_token_boundary_hit_rate",
    "topk_token_strict_overflow_rate",
    "high_energy_scalar_exact_zero_rate",
    "high_energy_scalar_near_zero_pm1_rate",
    "high_energy_scalar_boundary_hit_rate",
    "high_energy_scalar_strict_overflow_rate",
    "topk_token_clip_rate",
    "topk_token_near_zero_rate",
    "high_energy_scalar_clip_rate",
    "high_energy_scalar_near_zero_rate",
    "topk_near_zero_rate",
    "topk_clip_rate",
    "token_oer_retention",
    "token_oer_loss",
    "embedding_distortion",
]


def _as_float_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _as_bool_tensor(value: Any, *, device: torch.device | None = None) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach().bool()
    else:
        result = torch.as_tensor(value, dtype=torch.bool)
    if device is not None:
        result = result.to(device=device)
    return result


def _as_sequence(value: Any, *, batch_size: int, name: str) -> list[Any]:
    if isinstance(value, Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value] * int(batch_size)
    if len(values) != int(batch_size):
        raise ValueError(f"{name} length must equal batch size, got {len(values)} vs {batch_size}")
    return values


def _normalize_dim(dim: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("scalar tensor has no dimensions")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim={dim} is out of range for ndim={ndim}")
    return dim


def _finite_mean(values: Tensor) -> Tensor:
    values = values.float()
    finite = torch.isfinite(values)
    if not finite.any().item():
        return values.new_tensor(float("nan"))
    return values[finite].mean()


def _resolve_topk_count(valid_count: int, *, topk: int | None, topk_ratio: float) -> int:
    if int(valid_count) <= 0:
        raise ValueError("valid token count is zero; cannot compute Token Survival")
    if topk is None:
        ratio = float(topk_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
        resolved = int(math.ceil(float(valid_count) * ratio))
    else:
        resolved = int(topk)
    if resolved <= 0:
        raise ValueError(f"topk must be positive, got {resolved}")
    return min(resolved, int(valid_count))


def normalize_token_tensor_layout(
    x: Any,
    *,
    batch_dim: int = 0,
    token_dim: int = 1,
    channel_dim: int = -1,
    layout: str | None = None,
) -> Tensor:
    """Token survival helper."""

    del batch_dim, token_dim, channel_dim
    tensor = _as_float_tensor(x)
    if int(tensor.numel()) <= 0:
        raise ValueError("token tensor must not be empty")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("token tensor contains NaN or Inf")
    if layout is None:
        return ensure_canonical_btc(tensor, context="token_survival.tokens")
    layout_key = str(layout).upper().replace("D", "C")
    return canonicalize_activation_btc(tensor, raw_layout=layout_key, context="token_survival.tokens").tensor


def normalize_padding_mask(padding_mask: Any, *, batch_size: int, num_tokens: int) -> Tensor:
    """Token survival helper."""

    return canonicalize_padding_mask_bt(
        padding_mask,
        batch_size=int(batch_size),
        token_count=int(num_tokens),
        context="token_survival.padding_mask",
    )


def _token_norm(x_btd: Tensor) -> Tensor:
    return torch.linalg.vector_norm(x_btd.float(), ord=2, dim=-1)


def _topk_token_ids_from_norm(
    norms: Tensor,
    valid_token_mask: Tensor,
    *,
    topk: int | None,
    topk_ratio: float,
) -> list[Tensor]:
    if tuple(norms.shape) != tuple(valid_token_mask.shape):
        raise ValueError(f"norms and valid_token_mask shapes differ: {tuple(norms.shape)} vs {tuple(valid_token_mask.shape)}")

    result: list[Tensor] = []
    token_ids = torch.arange(int(norms.shape[1]), device=norms.device)
    for sample_idx in range(int(norms.shape[0])):
        valid = valid_token_mask[sample_idx]
        valid_count = int(valid.sum().item())
        k = _resolve_topk_count(valid_count, topk=topk, topk_ratio=topk_ratio)
        valid_ids = token_ids[valid]
        valid_norms = norms[sample_idx, valid]
        selected = torch.topk(valid_norms, k=k, largest=True, sorted=True).indices
        result.append(valid_ids[selected].long().cpu())
    return result


def _normalise_external_fp32_topk_token_ids(
    value: Any,
    *,
    valid_token_mask: Tensor,
    batch_size: int,
    num_tokens: int,
) -> list[Tensor]:
    if value is None:
        raise ValueError("fp32_topk_token_ids must not be None")
    if isinstance(value, Tensor):
        rows = [item.reshape(-1) for item in value.detach().cpu().long().reshape(int(batch_size), -1)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != int(batch_size):
            raise ValueError(f"fp32_topk_token_ids batch size mismatch: {len(value)} vs {batch_size}")
        rows = [torch.as_tensor(item, dtype=torch.long).detach().cpu().reshape(-1) for item in value]
    else:
        raise TypeError(f"fp32_topk_token_ids must be a Tensor or sequence, got {type(value)!r}")

    result: list[Tensor] = []
    valid_cpu = valid_token_mask.detach().cpu().bool()
    for sample_idx, ids in enumerate(rows):
        ids = ids.long().reshape(-1)
        if int(ids.numel()) <= 0:
            raise ValueError("fp32_topk_token_ids must not contain empty rows")
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(num_tokens):
            raise ValueError("fp32_topk_token_ids contains an out-of-range token id")
        selected_valid = valid_cpu[int(sample_idx)].index_select(0, ids)
        if not bool(selected_valid.all().item()):
            raise ValueError("fp32_topk_token_ids contains padding token ids")
        result.append(ids.contiguous())
    return result


def _validate_external_fp32_topk_protocol(*, topk_unit: str | None, topk_source: str | None) -> None:
    if topk_unit is not None and str(topk_unit) != "token":
        raise ValueError(f"Token Survival formal top-k requires topk_unit='token', got {topk_unit!r}")
    if topk_source is not None and str(topk_source) != "fp32_token_energy":
        raise ValueError(f"Token Survival formal top-k requires topk_source='fp32_token_energy', got {topk_source!r}")


def select_fp32_topk_token_ids(
    fp32_tokens: Any,
    padding_mask: Any,
    *,
    topk: int | None = None,
    topk_ratio: float = DEFAULT_SCALAR_OER_TOP_RATIO,
    layout: str | None = None,
) -> list[Tensor]:
    """Token survival helper."""

    fp32 = normalize_token_tensor_layout(fp32_tokens, layout=layout)
    mask = normalize_padding_mask(padding_mask, batch_size=int(fp32.shape[0]), num_tokens=int(fp32.shape[1]))
    return _topk_token_ids_from_norm(_token_norm(fp32), ~mask, topk=topk, topk_ratio=topk_ratio)


def _ids_to_json(ids: Tensor) -> str:
    return json.dumps([int(item) for item in ids.detach().cpu().reshape(-1).tolist()], ensure_ascii=False)


def _intersection_rate(left: Tensor, right: Tensor) -> Tensor:
    left_set = {int(item) for item in left.detach().cpu().reshape(-1).tolist()}
    right_set = {int(item) for item in right.detach().cpu().reshape(-1).tolist()}
    if not left_set:
        return torch.tensor(float("nan"), dtype=torch.float32)
    return torch.tensor(float(len(left_set & right_set)) / float(len(left_set)), dtype=torch.float32)


def _validate_survival_min_norm_retention(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError(
            "survival_min_norm_retention must be a finite value in (0, 1]; 0.0 would degenerate survival into nonzero-rate"
        )
    return threshold


_MISSING_ACTIVATION_MAPPING_MESSAGE = (
    "Token Survival topk_token_near_zero_rate/topk_token_clip_rate and "
    "high_energy_scalar_near_zero_rate/high_energy_scalar_clip_rate require integer_mapping "
    "from either reconstructed fixed calibration qparams or a validated QuantLinear.forward integer grid."
)

_ACTIVATION_GRID_MODES = {"a4", "a4-only", "a4_only", "w4a4", "w4-a4", "w4_a4"}
_NO_ACTIVATION_GRID_MODES = {"fp32", "w4", "w4-only", "w4_only"}


def _quant_mode_key(quant_mode: str) -> str:
    return str(quant_mode).strip().lower()


def _requires_activation_integer_mapping(quant_mode: str) -> bool:
    return _quant_mode_key(quant_mode) in _ACTIVATION_GRID_MODES


def _allows_missing_integer_mapping(quant_mode: str) -> bool:
    return _quant_mode_key(quant_mode) in _NO_ACTIVATION_GRID_MODES


def _mapping_source(integer_mapping: Mapping[str, Any] | None) -> str:
    if integer_mapping is None:
        return "none"
    return str(integer_mapping.get("mapping_source", ""))


_DIRECT_QUANTLINEAR_MAPPING_SOURCE = "quantlinear_forward"
_ACCEPTED_INTEGER_MAPPING_SOURCES = {
    "calibration_qparams_reconstructed_from_quant_input",
    "int4_actual",
    _DIRECT_QUANTLINEAR_MAPPING_SOURCE,
}
_QUANTLINEAR_FORWARD_REQUIRED_FIELDS = (
    "x",
    "q_preclip",
    "q_int",
    "boundary_hit_mask",
    "strict_overflow_mask",
    "exact_zero_mask",
    "near_zero_pm1_mask",
    "scale",
    "zero_point",
    "qmin",
    "qmax",
    "mapping_source",
    "integer_grid_source",
    "is_direct_quantlinear_forward_grid",
    "input_layout_info",
    "canonical_layout",
)


def _mapping_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().reshape(-1)
        if int(tensor.numel()) != 1:
            raise ValueError(f"integer_mapping[{name!r}] must be scalar bool-like, got shape={tuple(value.shape)}")
        return bool(tensor[0].item())
    if isinstance(value, bool):
        return bool(value)
    raise ValueError(f"integer_mapping[{name!r}] must be a bool, got {type(value)!r}")


def _require_quantlinear_forward_payload(integer_mapping: Mapping[str, Any]) -> None:
    missing = [key for key in _QUANTLINEAR_FORWARD_REQUIRED_FIELDS if key not in integer_mapping]
    if missing:
        raise KeyError(f"quantlinear_forward integer_mapping is missing required direct payload fields: {missing}")
    if str(integer_mapping.get("integer_grid_source", "")) != _DIRECT_QUANTLINEAR_MAPPING_SOURCE:
        raise ValueError(
            "quantlinear_forward integer_mapping must have integer_grid_source='quantlinear_forward'; "
            f"got {integer_mapping.get('integer_grid_source')!r}"
        )
    if not _mapping_bool(integer_mapping.get("is_direct_quantlinear_forward_grid"), name="is_direct_quantlinear_forward_grid"):
        raise ValueError("quantlinear_forward integer_mapping must set is_direct_quantlinear_forward_grid=True")
    if str(integer_mapping.get("canonical_layout", "")).upper() != "BTC":
        raise ValueError(
            "quantlinear_forward integer_mapping must be canonical BTC; "
            f"got canonical_layout={integer_mapping.get('canonical_layout')!r}"
        )
    if not isinstance(integer_mapping.get("input_layout_info"), Mapping):
        raise TypeError("quantlinear_forward integer_mapping must include input_layout_info mapping")

    q_int = torch.as_tensor(integer_mapping["q_int"]).detach()
    if q_int.ndim != 3:
        raise ValueError(f"quantlinear_forward integer_mapping['q_int'] must be [B,T,C], got shape={tuple(q_int.shape)}")
    expected_shape = tuple(int(item) for item in q_int.shape)
    x = _mapping_tensor(integer_mapping, "x", expected_shape=expected_shape).float()
    q_preclip = _mapping_tensor(integer_mapping, "q_preclip", expected_shape=expected_shape).float()
    q_int_float = _mapping_tensor(integer_mapping, "q_int", expected_shape=expected_shape).float()
    qmin = _scalar_int(integer_mapping["qmin"], name="qmin")
    qmax = _scalar_int(integer_mapping["qmax"], name="qmax")
    expected_q_int = torch.clamp(q_preclip, min=float(qmin), max=float(qmax))
    if not torch.allclose(q_int_float.cpu(), expected_q_int.cpu(), atol=1e-6, rtol=0.0):
        raise ValueError("quantlinear_forward integer_mapping['q_int'] must equal clamp(q_preclip, qmin, qmax)")

    scale = torch.as_tensor(integer_mapping["scale"], dtype=torch.float32, device=x.device).detach()
    zero = _zero_point_tensor(integer_mapping, device=x.device)
    try:
        expected_q_preclip = torch.round(x / torch.clamp(scale, min=float(DEFAULT_EPS)) + zero)
    except RuntimeError as exc:
        raise ValueError("quantlinear_forward scale/zero_point cannot broadcast to x for q_preclip validation") from exc
    if not torch.allclose(q_preclip.cpu(), expected_q_preclip.cpu(), atol=1e-5, rtol=0.0):
        raise ValueError("quantlinear_forward integer_mapping['q_preclip'] must equal round(x / scale + zero_point)")

    _normalise_or_derive_exact_zero_mask(integer_mapping, expected_shape=expected_shape)
    _normalise_or_derive_near_zero_pm1_mask(integer_mapping, expected_shape=expected_shape)
    _normalise_or_derive_boundary_hit_mask(integer_mapping, expected_shape=expected_shape)
    _normalise_or_derive_strict_overflow_mask(integer_mapping, expected_shape=expected_shape)


def _require_int4_actual_mapping(integer_mapping: Mapping[str, Any]) -> None:
    missing = [
        key
        for key in ("x", "q_preclip", "q_int", "scale", "qmin", "qmax", "mapping_source")
        if key not in integer_mapping
    ]
    if missing:
        raise KeyError(f"integer_mapping is missing required fields: {missing}")
    source = _mapping_source(integer_mapping)
    if source not in _ACCEPTED_INTEGER_MAPPING_SOURCES:
        raise ValueError(
            "Token Survival near_zero/clip metrics require accepted integer_mapping source "
            f"{sorted(_ACCEPTED_INTEGER_MAPPING_SOURCES)}; got mapping_source={source!r}"
        )
    if source == _DIRECT_QUANTLINEAR_MAPPING_SOURCE:
        _require_quantlinear_forward_payload(integer_mapping)


def _prepare_integer_mapping(
    *,
    quant_mode: str,
    integer_mapping: Mapping[str, Any] | None,
    require_integer_mapping: bool | None = None,
) -> Mapping[str, Any] | None:
    if integer_mapping is None:
        if require_integer_mapping is False:
            return None
        if require_integer_mapping is True:
            raise ValueError(_MISSING_ACTIVATION_MAPPING_MESSAGE)
        if _requires_activation_integer_mapping(quant_mode):
            raise ValueError(_MISSING_ACTIVATION_MAPPING_MESSAGE)
        if _allows_missing_integer_mapping(quant_mode):
            return None
        raise ValueError(f"unknown quant_mode={quant_mode!r}; cannot decide whether INT4 activation integer_mapping is required")

    _require_int4_actual_mapping(integer_mapping)
    return integer_mapping


def _mapping_tensor(
    integer_mapping: Mapping[str, Any],
    name: str,
    *,
    expected_shape: tuple[int, int, int],
    dtype: torch.dtype | None = None,
) -> Tensor:
    if name not in integer_mapping:
        raise KeyError(f"integer_mapping is missing {name}")
    tensor = torch.as_tensor(integer_mapping[name]).detach()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(f"integer_mapping[{name!r}] shape must equal token shape, got {tuple(tensor.shape)} vs {expected_shape}")
    return tensor


def _scalar_int(value: Any, *, name: str) -> int:
    tensor = torch.as_tensor(value).detach().reshape(-1)
    if int(tensor.numel()) != 1:
        raise ValueError(f"integer_mapping[{name!r}] must be scalar, got shape={tuple(torch.as_tensor(value).shape)}")
    return int(tensor[0].item())


def _normalise_mapping_mask(value: Any, *, shape: tuple[int, int, int], name: str) -> Tensor:
    mask = torch.as_tensor(value).detach().to(dtype=torch.bool)
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"integer_mapping[{name!r}] shape must equal token shape, got {tuple(mask.shape)} vs {shape}")
    return mask.contiguous()


def _zero_point_tensor(integer_mapping: Mapping[str, Any], *, device: torch.device) -> Tensor:
    zero = integer_mapping.get("zero_point", 0.0)
    return torch.as_tensor(zero, dtype=torch.float32, device=device).detach()


def _normalise_or_derive_exact_zero_mask(
    integer_mapping: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int, int],
) -> Tensor:
    q_int = _mapping_tensor(integer_mapping, "q_int", expected_shape=expected_shape).float()
    zero = _zero_point_tensor(integer_mapping, device=q_int.device)
    derived = q_int == zero
    for key in ("exact_zero_mask", "near_zero_mask"):
        if key not in integer_mapping:
            continue
        provided = _normalise_mapping_mask(integer_mapping[key], shape=expected_shape, name=key)
        if not torch.equal(provided.cpu(), derived.cpu()):
            raise ValueError(f"integer_mapping[{key!r}] must equal q_int == zero_point for Token Survival exact-zero semantics")
        break
    return derived.bool()


def _normalise_or_derive_near_zero_pm1_mask(
    integer_mapping: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int, int],
) -> Tensor:
    q_int = _mapping_tensor(integer_mapping, "q_int", expected_shape=expected_shape).float()
    zero = _zero_point_tensor(integer_mapping, device=q_int.device)
    derived = (q_int - zero).abs() <= 1.0
    if "near_zero_pm1_mask" in integer_mapping:
        provided = _normalise_mapping_mask(integer_mapping["near_zero_pm1_mask"], shape=expected_shape, name="near_zero_pm1_mask")
        if not torch.equal(provided.cpu(), derived.cpu()):
            raise ValueError("integer_mapping['near_zero_pm1_mask'] must equal abs(q_int - zero_point) <= 1")
    return derived.bool()


def _normalise_or_derive_strict_overflow_mask(
    integer_mapping: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int, int],
) -> Tensor:
    q_preclip = _mapping_tensor(integer_mapping, "q_preclip", expected_shape=expected_shape)
    qmin = _scalar_int(integer_mapping["qmin"], name="qmin")
    qmax = _scalar_int(integer_mapping["qmax"], name="qmax")
    derived = (q_preclip < qmin) | (q_preclip > qmax)
    for key in ("strict_overflow_mask", "clip_mask"):
        if key not in integer_mapping:
            continue
        provided = _normalise_mapping_mask(integer_mapping[key], shape=expected_shape, name=key)
        if not torch.equal(provided.cpu(), derived.cpu()):
            raise ValueError(f"integer_mapping[{key!r}] must be based on q_preclip < qmin or q_preclip > qmax")
        break
    return derived.bool()


def _normalise_or_derive_boundary_hit_mask(
    integer_mapping: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int, int],
) -> Tensor:
    q_int = _mapping_tensor(integer_mapping, "q_int", expected_shape=expected_shape)
    qmin = _scalar_int(integer_mapping["qmin"], name="qmin")
    qmax = _scalar_int(integer_mapping["qmax"], name="qmax")
    derived = (q_int <= qmin) | (q_int >= qmax)
    if "boundary_hit_mask" in integer_mapping:
        provided = _normalise_mapping_mask(integer_mapping["boundary_hit_mask"], shape=expected_shape, name="boundary_hit_mask")
        if not torch.equal(provided.cpu(), derived.cpu()):
            raise ValueError("integer_mapping['boundary_hit_mask'] must equal q_int <= qmin or q_int >= qmax")
    return derived.bool()


def _slice_integer_mapping(integer_mapping: Mapping[str, Any], sample_idx: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    batched_keys = {
        "x", "q_preclip", "q_int",
        "exact_zero_mask", "near_zero_pm1_mask", "boundary_hit_mask", "strict_overflow_mask",
        "near_zero_mask", "clip_mask",
    }
    for key, value in integer_mapping.items():
        if isinstance(value, Tensor):
            tensor = value.detach().cpu()
            if key in batched_keys and tensor.ndim >= 1 and int(tensor.shape[0]) > int(sample_idx):
                result[key] = tensor[int(sample_idx): int(sample_idx) + 1].contiguous()
            else:
                result[key] = tensor
        else:
            result[key] = value
    return result


def compute_topk_integer_rate_summary(
    integer_mapping: Mapping[str, Any],
    fp32_topk_token_ids: Any,
    *,
    valid_token_mask: Any | None = None,
) -> dict[str, Tensor]:
    q_int = torch.as_tensor(integer_mapping["q_int"]).detach()
    if q_int.ndim != 3:
        raise ValueError(f"integer_mapping['q_int'] must be [B,T,C], got shape={tuple(q_int.shape)}")
    expected_shape = tuple(int(item) for item in q_int.shape)
    ids = torch.as_tensor(fp32_topk_token_ids, dtype=torch.long, device=q_int.device).reshape(int(q_int.shape[0]), -1)
    if ids.numel() <= 0:
        raise ValueError("fp32_topk_token_ids must not be empty")
    if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(q_int.shape[1]):
        raise ValueError("fp32_topk_token_ids contains an out-of-range token id")
    if valid_token_mask is not None:
        valid = torch.as_tensor(valid_token_mask, dtype=torch.bool, device=q_int.device).reshape(int(q_int.shape[0]), int(q_int.shape[1]))
        selected_valid = gather_token_values(valid, ids, batch_dim=0, token_dim=1)
        if not bool(selected_valid.all().item()):
            raise ValueError("fp32_topk_token_ids contains padding token ids")
    exact_zero = _normalise_or_derive_exact_zero_mask(integer_mapping, expected_shape=expected_shape).to(device=q_int.device)
    near_zero_pm1 = _normalise_or_derive_near_zero_pm1_mask(integer_mapping, expected_shape=expected_shape).to(device=q_int.device)
    boundary_hit = _normalise_or_derive_boundary_hit_mask(integer_mapping, expected_shape=expected_shape).to(device=q_int.device)
    strict_overflow = _normalise_or_derive_strict_overflow_mask(integer_mapping, expected_shape=expected_shape).to(device=q_int.device)
    selected_exact_zero = gather_token_values(exact_zero, ids, batch_dim=0, token_dim=1)
    selected_near_zero_pm1 = gather_token_values(near_zero_pm1, ids, batch_dim=0, token_dim=1)
    selected_boundary_hit = gather_token_values(boundary_hit, ids, batch_dim=0, token_dim=1)
    selected_strict_overflow = gather_token_values(strict_overflow, ids, batch_dim=0, token_dim=1)
    token_exact_zero = selected_exact_zero.any(dim=-1)
    token_near_zero_pm1 = selected_near_zero_pm1.any(dim=-1)
    token_boundary_hit = selected_boundary_hit.any(dim=-1)
    token_strict_overflow = selected_strict_overflow.any(dim=-1)
    exact_zero_rate = boolean_mask_rate(token_exact_zero, per_sample=True)
    near_zero_pm1_rate = boolean_mask_rate(token_near_zero_pm1, per_sample=True)
    boundary_hit_rate = boolean_mask_rate(token_boundary_hit, per_sample=True)
    strict_overflow_rate = boolean_mask_rate(token_strict_overflow, per_sample=True)
    scalar_exact_zero = boolean_mask_rate(selected_exact_zero, per_sample=True)
    scalar_near_zero_pm1 = boolean_mask_rate(selected_near_zero_pm1, per_sample=True)
    scalar_boundary_hit = boolean_mask_rate(selected_boundary_hit, per_sample=True)
    scalar_strict_overflow = boolean_mask_rate(selected_strict_overflow, per_sample=True)
    return {
        "topk_token_exact_zero_rate": exact_zero_rate,
        "topk_token_near_zero_pm1_rate": near_zero_pm1_rate,
        "topk_token_boundary_hit_rate": boundary_hit_rate,
        "topk_token_strict_overflow_rate": strict_overflow_rate,
        "high_energy_scalar_exact_zero_rate": scalar_exact_zero,
        "high_energy_scalar_near_zero_pm1_rate": scalar_near_zero_pm1,
        "high_energy_scalar_boundary_hit_rate": scalar_boundary_hit,
        "high_energy_scalar_strict_overflow_rate": scalar_strict_overflow,
        # Deprecated Token Survival aliases keep their original local semantics.
        "topk_token_near_zero_rate": exact_zero_rate,
        "topk_token_clip_rate": strict_overflow_rate,
        "high_energy_scalar_near_zero_rate": scalar_exact_zero,
        "high_energy_scalar_clip_rate": scalar_strict_overflow,
    }


def compute_topk_integer_rates(
    integer_mapping: Mapping[str, Any],
    fp32_topk_token_ids: Any,
    *,
    valid_token_mask: Any | None = None,
) -> tuple[Tensor, Tensor]:
    summary = compute_topk_integer_rate_summary(
        integer_mapping,
        fp32_topk_token_ids,
        valid_token_mask=valid_token_mask,
    )
    return (
        summary["high_energy_scalar_exact_zero_rate"],
        summary["high_energy_scalar_strict_overflow_rate"],
    )

def compute_token_oer_retention(
    fp32_tokens: Any,
    quant_tokens: Any,
    fp32_topk_token_ids: Any,
    *,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    fp32 = normalize_token_tensor_layout(fp32_tokens)
    quant = normalize_token_tensor_layout(quant_tokens).to(device=fp32.device)
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"fp32_tokens and quant_tokens shapes differ: {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    ids = torch.as_tensor(fp32_topk_token_ids, dtype=torch.long, device=fp32.device).reshape(int(fp32.shape[0]), -1)
    fp32_selected = gather_token_values(fp32, ids, batch_dim=0, token_dim=1)
    quant_selected = gather_token_values(quant, ids, batch_dim=0, token_dim=1)
    fp32_energy = fp32_selected.square().sum(dim=(1, 2))
    quant_energy = quant_selected.square().sum(dim=(1, 2))
    retention = quant_energy / torch.clamp(fp32_energy, min=float(eps))
    return torch.where(fp32_energy > float(eps), retention, torch.full_like(retention, float("nan")))


def compute_token_direction_distortion(
    fp32_selected_tokens: Tensor,
    quant_selected_tokens: Tensor,
    *,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    if tuple(fp32_selected_tokens.shape) != tuple(quant_selected_tokens.shape):
        raise ValueError("fp32_selected_tokens and quant_selected_tokens shapes differ")
    numerator = (fp32_selected_tokens * quant_selected_tokens).sum(dim=-1)
    fp32_norm = torch.linalg.vector_norm(fp32_selected_tokens, ord=2, dim=-1)
    quant_norm = torch.linalg.vector_norm(quant_selected_tokens, ord=2, dim=-1)
    denom = torch.clamp(fp32_norm * quant_norm, min=float(eps))
    cosine = numerator / denom
    valid = (fp32_norm > float(eps)) & (quant_norm > float(eps))
    distortion = 1.0 - torch.clamp(cosine, min=-1.0, max=1.0)
    return torch.where(valid, distortion, torch.full_like(distortion, float("nan")))


def _embedding_distortion_per_sample(
    *,
    fp32_embedding: Any | None,
    quant_embedding: Any | None,
    batch_size: int,
) -> Tensor:
    if fp32_embedding is None or quant_embedding is None:
        raise ValueError("embedding_distortion requires fp32_embedding and int4_embedding final embeddings")
    dist = embedding_distortion(fp32_embedding, quant_embedding, dim=-1, reduction="none").detach().cpu().float().reshape(-1)
    if int(dist.numel()) != int(batch_size):
        raise ValueError(f"embedding_distortion sample count mismatch: {int(dist.numel())} vs {batch_size}")
    return dist


def compute_token_survival_rows(
    fp32_tokens: Any,
    int4_tokens: Any,
    padding_mask: Any,
    *,
    sample_id: Any,
    label: Any,
    bucket: Any,
    shot_group: Any | None = None,
    layer_idx: int,
    module_name: str,
    tensor_position: str,
    quant_mode: str,
    granularity: str = "per_tensor",
    integer_mapping: Mapping[str, Any] | None = None,
    require_integer_mapping: bool | None = None,
    fp32_embedding: Any | None = None,
    int4_embedding: Any | None = None,
    topk: int | None = None,
    topk_ratio: float = DEFAULT_SCALAR_OER_TOP_RATIO,
    fp32_topk_token_ids: Any | None = None,
    fp32_topk_unit: str | None = None,
    fp32_topk_source: str | None = None,
    survival_min_norm_retention: float = DEFAULT_TOKEN_SURVIVAL_MIN_NORM_RETENTION,
    eps: float = DEFAULT_EPS,
) -> list[dict[str, Any]]:
    fp32 = normalize_token_tensor_layout(fp32_tokens)
    quant = normalize_token_tensor_layout(int4_tokens).to(device=fp32.device)
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"fp32_tokens and int4_tokens shapes differ: {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    batch_size, num_tokens, hidden_dim = (int(fp32.shape[0]), int(fp32.shape[1]), int(fp32.shape[2]))
    pad = normalize_padding_mask(padding_mask, batch_size=batch_size, num_tokens=num_tokens).to(device=fp32.device)
    valid_token_mask = ~pad
    sample_ids = [str(item) for item in _as_sequence(sample_id, batch_size=batch_size, name="sample_id")]
    labels = _as_sequence(label, batch_size=batch_size, name="label")
    bucket_values = [str(item) for item in _as_sequence(bucket, batch_size=batch_size, name="bucket")]
    shot_values = None if shot_group is None else [str(item) for item in _as_sequence(shot_group, batch_size=batch_size, name="shot_group")]
    bucket_result = read_canonical_bucket(
        bucket=bucket_values,
        shot_group=shot_values,
        context="token_survival sample rows",
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in bucket_result.bucket]
    shot_groups = list(buckets) if shot_values is None else [str(item) for item in shot_values]
    tail_flags = [bool(item) for item in tail_included(buckets)]
    survival_threshold = _validate_survival_min_norm_retention(float(survival_min_norm_retention))
    prepared_mapping = _prepare_integer_mapping(
        quant_mode=str(quant_mode),
        integer_mapping=integer_mapping,
        require_integer_mapping=require_integer_mapping,
    )
    integer_mapping_source = _mapping_source(prepared_mapping)
    if fp32_topk_token_ids is None:
        fp32_topk_ids = _topk_token_ids_from_norm(_token_norm(fp32), valid_token_mask, topk=topk, topk_ratio=topk_ratio)
    else:
        _validate_external_fp32_topk_protocol(topk_unit=fp32_topk_unit, topk_source=fp32_topk_source)
        fp32_topk_ids = _normalise_external_fp32_topk_token_ids(
            fp32_topk_token_ids,
            valid_token_mask=valid_token_mask,
            batch_size=batch_size,
            num_tokens=num_tokens,
        )
    int4_topk_ids = _topk_token_ids_from_norm(_token_norm(quant), valid_token_mask, topk=topk, topk_ratio=topk_ratio)
    embedding_dist = _embedding_distortion_per_sample(
        fp32_embedding=fp32_embedding,
        quant_embedding=int4_embedding,
        batch_size=batch_size,
    )

    rows: list[dict[str, Any]] = []
    for sample_idx in range(batch_size):
        fp_ids = fp32_topk_ids[sample_idx]
        int4_ids = int4_topk_ids[sample_idx]
        fp_ids_b = fp_ids.to(device=fp32.device, dtype=torch.long).reshape(1, -1)
        fp32_sample = fp32[sample_idx: sample_idx + 1]
        quant_sample = quant[sample_idx: sample_idx + 1]
        if prepared_mapping is None:
            integer_rate_values = {
                "topk_token_exact_zero_rate": float("nan"),
                "topk_token_near_zero_pm1_rate": float("nan"),
                "topk_token_boundary_hit_rate": float("nan"),
                "topk_token_strict_overflow_rate": float("nan"),
                "high_energy_scalar_exact_zero_rate": float("nan"),
                "high_energy_scalar_near_zero_pm1_rate": float("nan"),
                "high_energy_scalar_boundary_hit_rate": float("nan"),
                "high_energy_scalar_strict_overflow_rate": float("nan"),
                "topk_token_near_zero_rate": float("nan"),
                "topk_token_clip_rate": float("nan"),
                "high_energy_scalar_near_zero_rate": float("nan"),
                "high_energy_scalar_clip_rate": float("nan"),
            }
        else:
            sample_mapping = _slice_integer_mapping(prepared_mapping, sample_idx)
            integer_rates = compute_topk_integer_rate_summary(
                sample_mapping,
                fp_ids_b,
                valid_token_mask=valid_token_mask[sample_idx: sample_idx + 1],
            )
            integer_rate_values = {
                name: float(value.reshape(-1)[0].detach().cpu().item())
                for name, value in integer_rates.items()
            }
        token_oer = compute_token_oer_retention(fp32_sample, quant_sample, fp_ids_b, eps=eps)
        fp32_selected = gather_token_values(fp32_sample, fp_ids_b, batch_dim=0, token_dim=1)
        quant_selected = gather_token_values(quant_sample, fp_ids_b, batch_dim=0, token_dim=1)
        selected_fp32_norm = torch.linalg.vector_norm(fp32_selected, ord=2, dim=-1)
        selected_quant_norm = torch.linalg.vector_norm(quant_selected, ord=2, dim=-1)
        norm_retention = selected_quant_norm / torch.clamp(selected_fp32_norm, min=float(eps))
        norm_retention = torch.where(
            selected_fp32_norm > float(eps),
            norm_retention,
            torch.full_like(norm_retention, float("nan")),
        )
        survived = (selected_fp32_norm > float(eps)) & (norm_retention >= survival_threshold)
        direction_distortion = compute_token_direction_distortion(fp32_selected, quant_selected, eps=eps)
        oer_value = float(token_oer.reshape(-1)[0].detach().cpu().item())
        rows.append(
            {
                "sample_id": sample_ids[sample_idx],
                "label": labels[sample_idx],
                "bucket": buckets[sample_idx],
                "shot_group": shot_groups[sample_idx],
                "tail_included": tail_flags[sample_idx],
                "layer_idx": int(layer_idx),
                "module_name": str(module_name),
                "tensor_position": str(tensor_position),
                "quant_mode": str(quant_mode),
                "granularity": str(granularity),
                "activation_shape": tuple(int(item) for item in fp32_sample.shape),
                "original_layout": "BTC",
                "original_shape": tuple(int(item) for item in fp32_sample.shape),
                "canonical_layout": "BTC",
                "canonical_shape": tuple(int(item) for item in fp32_sample.shape),
                "layout_inference_source": "canonical_metric_input",
                "padding_mask_available": True,
                "padding_mask_aligned": True,
                "mask_status": "aligned",
                "valid_token_policy": "require_mask",
                "integer_mapping_source": integer_mapping_source,
                "num_tokens": int(valid_token_mask[sample_idx].sum().item()),
                "valid_token_count": int(valid_token_mask[sample_idx].sum().item()),
                "hidden_dim": int(hidden_dim),
                "topk": int(fp_ids.numel()),
                "topk_k": int(fp_ids.numel()),
                "survival_min_norm_retention": float(survival_threshold),
                "fp32_topk_token_ids": _ids_to_json(fp_ids),
                "int4_topk_token_ids": _ids_to_json(int4_ids),
                "topk_overlap": float(_intersection_rate(fp_ids, int4_ids).item()),
                "topk_survival_rate": float(survived.float().mean().item()),
                "token_survival_drop": float(1.0 - survived.float().mean().item()),
                "topk_norm_retention_values": json.dumps([
                    None if not math.isfinite(float(item)) else float(item)
                    for item in norm_retention.detach().cpu().reshape(-1).tolist()
                ], ensure_ascii=False),
                "mean_norm_retention": float(_finite_mean(norm_retention.reshape(-1)).item()),
                "mean_token_distortion": float(_finite_mean(direction_distortion.reshape(-1)).item()),
                "topk_token_exact_zero_rate": integer_rate_values["topk_token_exact_zero_rate"],
                "topk_token_near_zero_pm1_rate": integer_rate_values["topk_token_near_zero_pm1_rate"],
                "topk_token_boundary_hit_rate": integer_rate_values["topk_token_boundary_hit_rate"],
                "topk_token_strict_overflow_rate": integer_rate_values["topk_token_strict_overflow_rate"],
                "high_energy_scalar_exact_zero_rate": integer_rate_values["high_energy_scalar_exact_zero_rate"],
                "high_energy_scalar_near_zero_pm1_rate": integer_rate_values["high_energy_scalar_near_zero_pm1_rate"],
                "high_energy_scalar_boundary_hit_rate": integer_rate_values["high_energy_scalar_boundary_hit_rate"],
                "high_energy_scalar_strict_overflow_rate": integer_rate_values["high_energy_scalar_strict_overflow_rate"],
                "topk_token_near_zero_rate": integer_rate_values["topk_token_near_zero_rate"],
                "topk_token_clip_rate": integer_rate_values["topk_token_clip_rate"],
                "high_energy_scalar_near_zero_rate": integer_rate_values["high_energy_scalar_near_zero_rate"],
                "high_energy_scalar_clip_rate": integer_rate_values["high_energy_scalar_clip_rate"],
                "topk_near_zero_rate": integer_rate_values["high_energy_scalar_near_zero_rate"],
                "topk_clip_rate": integer_rate_values["high_energy_scalar_clip_rate"],
                "token_oer_retention": oer_value,
                "token_oer_loss": float(1.0 - oer_value),
                "embedding_distortion": float(embedding_dist[sample_idx].detach().cpu().item()),
            }
        )
    return rows


def compute_single_token_survival_row(
    fp32_tokens: Any,
    int4_tokens: Any,
    padding_mask: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    rows = compute_token_survival_rows(fp32_tokens, int4_tokens, padding_mask, **kwargs)
    if len(rows) != 1:
        raise ValueError(f"compute_single_token_survival_row expected one sample, got {len(rows)} rows")
    return rows[0]


__all__ = [
    "SAMPLE_OUTPUT_COLUMNS",
    "compute_single_token_survival_row",
    "compute_token_direction_distortion",
    "compute_token_oer_retention",
    "compute_token_survival_rows",
    "compute_topk_integer_rate_summary",
    "compute_topk_integer_rates",
    "normalize_padding_mask",
    "normalize_token_tensor_layout",
    "select_fp32_topk_token_ids",
]

