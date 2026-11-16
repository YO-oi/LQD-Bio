from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


CANONICAL_LAYOUT = "BTC"
SUPPORTED_RAW_LAYOUTS = frozenset(("BTC", "TBC"))
VALID_TOKEN_POLICIES = frozenset(("require_mask", "explicit_no_padding"))


@dataclass(frozen=True)
class CanonicalTensor:
    tensor: Tensor
    layout_info: dict[str, Any]
    padding_mask: Tensor | None = None


def shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, Tensor):
        return tuple(int(item) for item in value.shape)
    return tuple(int(item) for item in torch.as_tensor(value).shape)


def layout_info_to_public_fields(layout_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "original_layout": layout_info.get("original_layout", ""),
        "original_shape": layout_info.get("original_shape", ""),
        "canonical_layout": layout_info.get("canonical_layout", CANONICAL_LAYOUT),
        "canonical_shape": layout_info.get("canonical_shape", ""),
        "layout_inference_source": layout_info.get("layout_inference_source", ""),
        "padding_mask_available": layout_info.get("padding_mask_available", False),
        "padding_mask_aligned": layout_info.get("padding_mask_aligned", False),
        "mask_status": layout_info.get("mask_status", ""),
    }


def validate_valid_token_policy(value: Any | None, *, default: str = "require_mask") -> str:
    policy = default if value is None else str(value)
    if policy not in VALID_TOKEN_POLICIES:
        raise ValueError(
            f"VALID_TOKEN_POLICY must be one of {sorted(VALID_TOKEN_POLICIES)}, got {policy!r}"
        )
    return policy


def require_valid_token_policy(
    *,
    padding_mask: Tensor | None,
    valid_token_policy: Any | None,
    context: str,
) -> str:
    policy = validate_valid_token_policy(valid_token_policy)
    if padding_mask is None and policy != "explicit_no_padding":
        raise RuntimeError(
            f"{context}: formal token/OER/clipping/near-zero metrics require padding_mask "
            "or VALID_TOKEN_POLICY='explicit_no_padding'"
        )
    return policy


def _as_tensor(value: Any, *, context: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except Exception as exc:  # pragma: no cover - defensive error enrichment.
        raise TypeError(f"{context}: cannot convert value to Tensor") from exc


def _normalize_raw_layout(raw_layout: str | None, *, context: str) -> str | None:
    if raw_layout is None or str(raw_layout).strip() == "":
        return None
    layout = str(raw_layout).strip().upper().replace(",", "").replace(" ", "")
    if layout not in SUPPORTED_RAW_LAYOUTS:
        raise ValueError(
            f"{context}: raw/original layout must be BTC or TBC, got {raw_layout!r}"
        )
    return layout


def _layout_matches(
    shape: tuple[int, int, int],
    layout: str,
    *,
    batch_size: int | None,
    token_count: int | None,
    feature_dim: int | None,
    padding_mask_shape: tuple[int, int] | None,
) -> bool:
    if layout == "BTC":
        batch_dim, token_dim, channel_dim = shape
    elif layout == "TBC":
        token_dim, batch_dim, channel_dim = shape
    else:  # pragma: no cover - guarded by caller.
        return False

    checks: list[bool] = []
    if batch_size is not None:
        checks.append(int(batch_dim) == int(batch_size))
    if token_count is not None:
        checks.append(int(token_dim) == int(token_count))
    if feature_dim is not None:
        checks.append(int(channel_dim) == int(feature_dim))
    if padding_mask_shape is not None:
        checks.append(tuple(padding_mask_shape) == (int(batch_dim), int(token_dim)))
    if not checks:
        return False
    return all(checks)


def _infer_layout(
    shape: tuple[int, int, int],
    *,
    batch_size: int | None,
    token_count: int | None,
    feature_dim: int | None,
    padding_mask_shape: tuple[int, int] | None,
    context: str,
) -> tuple[str, str]:
    candidates = [
        layout
        for layout in ("BTC", "TBC")
        if _layout_matches(
            shape,
            layout,
            batch_size=batch_size,
            token_count=token_count,
            feature_dim=feature_dim,
            padding_mask_shape=padding_mask_shape,
        )
    ]
    if len(candidates) == 1:
        if padding_mask_shape is not None:
            return candidates[0], "padding_mask_shape"
        return candidates[0], "batch_size_shape_unique"
    if len(candidates) > 1:
        raise ValueError(
            f"{context}: layout inference is ambiguous for shape={shape}, "
            f"batch_size={batch_size}, token_count={token_count}, feature_dim={feature_dim}, "
            f"padding_mask_shape={padding_mask_shape}; candidates={candidates}"
        )
    raise ValueError(
        f"{context}: cannot infer BTC/TBC layout for shape={shape}, "
        f"batch_size={batch_size}, token_count={token_count}, feature_dim={feature_dim}, "
        f"padding_mask_shape={padding_mask_shape}"
    )


def _canonicalize_mask_for_shape(
    padding_mask: Any | None,
    *,
    expected_bt: tuple[int, int],
    context: str,
) -> tuple[Tensor | None, bool, bool, str, tuple[int, ...] | None]:
    if padding_mask is None:
        return None, False, False, "missing", None
    mask = _as_tensor(padding_mask, context=f"{context}.padding_mask").detach().to(dtype=torch.bool)
    if mask.ndim == 3 and int(mask.shape[-1]) == 1:
        mask = mask.squeeze(-1)
    raw_shape = tuple(int(item) for item in mask.shape)
    if mask.ndim != 2:
        raise ValueError(f"{context}: padding_mask must be rank-2 [B,T], got shape={raw_shape}")
    if tuple(mask.shape) != tuple(expected_bt):
        raise ValueError(
            f"{context}: padding_mask must align with canonical [B,T]={expected_bt}, "
            f"got shape={raw_shape}"
        )
    return mask.contiguous(), True, True, "aligned", raw_shape


def canonicalize_activation_btc(
    value: Any,
    *,
    raw_layout: str | None = None,
    original_layout: str | None = None,
    padding_mask: Any | None = None,
    batch_size: int | None = None,
    token_count: int | None = None,
    feature_dim: int | None = None,
    context: str = "activation",
) -> CanonicalTensor:
    tensor = _as_tensor(value, context=context)
    if tensor.ndim != 3:
        raise ValueError(f"{context}: activation/token tensor must be rank-3 [B,T,C], got shape={tuple(tensor.shape)}")
    if int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) <= 0 or int(tensor.shape[2]) <= 0:
        raise ValueError(f"{context}: activation/token tensor dimensions must be non-empty, got shape={tuple(tensor.shape)}")

    original_shape = tuple(int(item) for item in tensor.shape)
    layout = _normalize_raw_layout(original_layout if original_layout is not None else raw_layout, context=context)
    mask_shape = None if padding_mask is None else tuple(int(item) for item in torch.as_tensor(padding_mask).shape[:2])
    if layout is None:
        layout, inference_source = _infer_layout(
            original_shape,
            batch_size=None if batch_size is None else int(batch_size),
            token_count=None if token_count is None else int(token_count),
            feature_dim=None if feature_dim is None else int(feature_dim),
            padding_mask_shape=mask_shape if padding_mask is not None else None,
            context=context,
        )
    else:
        has_validation_context = any(
            item is not None
            for item in (batch_size, token_count, feature_dim)
        )
        if has_validation_context and not _layout_matches(
            original_shape,
            layout,
            batch_size=None if batch_size is None else int(batch_size),
            token_count=None if token_count is None else int(token_count),
            feature_dim=None if feature_dim is None else int(feature_dim),
            padding_mask_shape=None,
        ):
            raise ValueError(
                f"{context}: explicit layout={layout!r} does not match shape/context; "
                f"shape={original_shape}, batch_size={batch_size}, token_count={token_count}, "
                f"feature_dim={feature_dim}, padding_mask_shape={mask_shape}"
            )
        inference_source = "explicit_layout"

    if layout == "BTC":
        canonical = tensor.contiguous()
    else:
        canonical = tensor.transpose(0, 1).contiguous()

    canonical_shape = tuple(int(item) for item in canonical.shape)
    mask_bt, mask_available, mask_aligned, mask_status, mask_raw_shape = _canonicalize_mask_for_shape(
        padding_mask,
        expected_bt=(canonical_shape[0], canonical_shape[1]),
        context=context,
    )
    if not mask_available and inference_source == "batch_size_shape_unique":
        mask_status = "missing_but_layout_inferred_unique"

    layout_info = {
        "original_layout": layout,
        "original_shape": original_shape,
        "canonical_layout": CANONICAL_LAYOUT,
        "canonical_shape": canonical_shape,
        "layout_inference_source": inference_source,
        "padding_mask_available": bool(mask_available),
        "padding_mask_aligned": bool(mask_aligned),
        "mask_status": mask_status,
        "padding_mask_original_shape": mask_raw_shape or "",
        "padding_mask_canonical_shape": tuple(int(item) for item in mask_bt.shape) if mask_bt is not None else "",
    }
    return CanonicalTensor(tensor=canonical, padding_mask=mask_bt, layout_info=layout_info)


def canonicalize_padding_mask_bt(
    padding_mask: Any,
    *,
    batch_size: int,
    token_count: int,
    context: str = "padding_mask",
) -> Tensor:
    mask, _, _, _, _ = _canonicalize_mask_for_shape(
        padding_mask,
        expected_bt=(int(batch_size), int(token_count)),
        context=context,
    )
    if mask is None:  # pragma: no cover - padding_mask is required by signature.
        raise RuntimeError(f"{context}: padding_mask unexpectedly resolved to None")
    return mask


def ensure_canonical_btc(value: Any, *, context: str = "activation") -> Tensor:
    tensor = _as_tensor(value, context=context)
    if tensor.ndim != 3:
        raise ValueError(f"{context}: metric input must already be canonical [B,T,C], got shape={tuple(tensor.shape)}")
    if int(tensor.shape[0]) <= 0 or int(tensor.shape[1]) <= 0 or int(tensor.shape[2]) <= 0:
        raise ValueError(f"{context}: canonical [B,T,C] dimensions must be non-empty, got shape={tuple(tensor.shape)}")
    return tensor.contiguous()


def canonicalize_already_filtered_tokens_btc(value: Any, *, context: str = "activation") -> CanonicalTensor:
    tensor = _as_tensor(value, context=context)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
        raise ValueError(
            f"{context}: already-filtered valid tokens must be [T,C] or [1,T,C], got shape={tuple(tensor.shape)}"
        )
    canonical = ensure_canonical_btc(tensor, context=context)
    shape = tuple(int(item) for item in canonical.shape)
    layout_info = {
        "original_layout": "BTC",
        "original_shape": shape,
        "canonical_layout": CANONICAL_LAYOUT,
        "canonical_shape": shape,
        "layout_inference_source": "already_filtered_valid_tokens",
        "padding_mask_available": False,
        "padding_mask_aligned": False,
        "mask_status": "already_filtered_valid_tokens",
    }
    return CanonicalTensor(tensor=canonical, layout_info=layout_info, padding_mask=None)


__all__ = [
    "CANONICAL_LAYOUT",
    "SUPPORTED_RAW_LAYOUTS",
    "VALID_TOKEN_POLICIES",
    "CanonicalTensor",
    "canonicalize_activation_btc",
    "canonicalize_already_filtered_tokens_btc",
    "canonicalize_padding_mask_bt",
    "ensure_canonical_btc",
    "layout_info_to_public_fields",
    "require_valid_token_policy",
    "shape_tuple",
    "validate_valid_token_policy",
]

