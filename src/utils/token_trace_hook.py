from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor

try:
    from src.utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
except ImportError:  # pragma: no cover
    from .layout_utils import canonicalize_activation_btc, layout_info_to_public_fields


FULL_LAYER_INDICES: tuple[int, ...] = tuple(range(12))
FFN_MODULE_NAMES = ("fc1", "fc2")
TOKEN_TRACE_POSITIONS = ("block_output", "input", "output")

TensorPosition = Literal["block_output", "input", "output"]


@dataclass(frozen=True)
class TokenTraceRecord:
    """Captured canonical token feature record."""

    layer_idx: int
    module_name: str
    tensor_position: TensorPosition
    target_name: str
    tensor: Tensor
    activation_shape: tuple[int, int, int]
    original_layout: str
    original_shape: tuple[int, int, int]
    canonical_layout: str
    canonical_shape: tuple[int, int, int]
    layout_inference_source: str
    padding_mask_available: bool
    padding_mask_aligned: bool
    mask_status: str
    sample_id: list[str]
    padding_mask: Tensor
    valid_token_mask: Tensor


def _validate_module_name(module_name: str) -> str:
    module_name = str(module_name)
    if module_name not in FFN_MODULE_NAMES:
        raise ValueError(f"module_name must be 'fc1' or 'fc2', got {module_name!r}")
    return module_name


def _get_encoder_layers(model: nn.Module) -> nn.ModuleList:
    if not hasattr(model, "encoder"):
        raise AttributeError("model is missing encoder; cannot locate BEATs Transformer blocks")
    encoder = model.encoder
    if not hasattr(encoder, "layers"):
        raise AttributeError("model.encoder is missing layers; cannot locate BEATs Transformer blocks")
    layers = encoder.layers
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"model.encoder.layers must be nn.ModuleList, got {type(layers)!r}")
    return layers


def resolve_full_layer_indices(model: nn.Module) -> tuple[int, ...]:
    """Resolve Transformer layer indices from the live model."""

    return tuple(range(len(_get_encoder_layers(model))))


def get_transformer_block(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return encoder.layers[layer_idx]."""

    layers = _get_encoder_layers(model)
    layer_idx = int(layer_idx)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx={layer_idx} is out of range [0, {len(layers) - 1}]")
    return layers[layer_idx]


def get_ffn_module(model: nn.Module, layer_idx: int, module_name: str) -> nn.Module:
    """Return encoder.layers[layer_idx].fc1/fc2."""

    module_name = _validate_module_name(module_name)
    block = get_transformer_block(model, int(layer_idx))
    if not hasattr(block, module_name):
        raise AttributeError(f"encoder.layers.{int(layer_idx)}.{module_name} does not exist")
    return getattr(block, module_name)


def _normalise_layer_indices(model: nn.Module, layer_indices: Sequence[int] | None) -> tuple[int, ...]:
    values = resolve_full_layer_indices(model) if layer_indices is None else tuple(int(item) for item in layer_indices)
    if not values:
        raise ValueError("layer_indices cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"layer_indices contains duplicate values: {values}")
    layer_count = len(_get_encoder_layers(model))
    invalid = [int(item) for item in values if int(item) < 0 or int(item) >= layer_count]
    if invalid:
        raise IndexError(f"layer_indices out of range for model with {layer_count} layers: {invalid}")
    return values


def _normalise_trace_positions(positions: Sequence[str] | None) -> tuple[TensorPosition, ...]:
    values = TOKEN_TRACE_POSITIONS if positions is None else tuple(str(item) for item in positions)
    if not values:
        raise ValueError("positions cannot be empty")
    invalid = [item for item in values if item not in TOKEN_TRACE_POSITIONS]
    if invalid:
        raise ValueError(f"positions must be in {TOKEN_TRACE_POSITIONS}, got {invalid}")
    if len(set(values)) != len(values):
        raise ValueError(f"positions contains duplicate values: {values}")
    return tuple(values)  # type: ignore[return-value]


def _normalise_target_modules(
    target_modules: Sequence[Any] | None,
    *,
    layer_indices: Sequence[int],
    require_ffn_targets: bool,
) -> tuple[tuple[int, str], ...]:
    if target_modules is None:
        if not require_ffn_targets:
            return ()
        return tuple((int(layer_idx), module_name) for layer_idx in layer_indices for module_name in FFN_MODULE_NAMES)

    result: list[tuple[int, str]] = []
    for item in target_modules:
        if isinstance(item, Mapping):
            if "layer_idx" not in item or "module_name" not in item:
                raise KeyError("target_modules mapping must include layer_idx and module_name")
            layer_idx = item["layer_idx"]
            module_name = item["module_name"]
        else:
            try:
                layer_idx, module_name = item
            except (TypeError, ValueError) as exc:
                raise TypeError(f"target_modules item must be (layer_idx, module_name), got {item!r}") from exc
        result.append((int(layer_idx), _validate_module_name(str(module_name))))
    if not result and require_ffn_targets:
        raise ValueError("target_modules cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"target_modules contains duplicate entries: {result}")
    return tuple(result)


def _first_feature_tensor(value: Any, *, target_name: str) -> Tensor:
    """Extract the feature tensor from a hook output."""

    if isinstance(value, Tensor):
        return value
    if isinstance(value, Mapping):
        preferred_keys = ("x", "features", "feature", "encoder_out", "hidden_states", "last_hidden_state")
        for key in preferred_keys:
            item = value.get(key)
            if isinstance(item, Tensor):
                return item
        candidates = list(value.values())
    elif isinstance(value, (tuple, list)):
        candidates = list(value)
    else:
        raise TypeError(f"{target_name}: hook output must be Tensor/tuple/list/dict, got {type(value)!r}")

    tensor_candidates = [item for item in candidates if isinstance(item, Tensor)]
    for item in tensor_candidates:
        if item.is_floating_point() and item.ndim >= 3:
            return item
    for item in tensor_candidates:
        if item.is_floating_point() and item.ndim >= 2:
            return item
    if tensor_candidates:
        return tensor_candidates[0]
    raise TypeError(f"{target_name}: hook output did not contain a feature Tensor")


def _to_sample_id_list(sample_id: Sequence[Any] | Tensor, *, batch_size: int) -> list[str]:
    if isinstance(sample_id, Tensor):
        values = sample_id.detach().cpu().reshape(-1).tolist()
    elif isinstance(sample_id, Sequence) and not isinstance(sample_id, (str, bytes)):
        values = list(sample_id)
    else:
        values = [sample_id]
    if len(values) != int(batch_size):
        raise ValueError(f"sample_id count must equal batch_size, got {len(values)} vs {batch_size}")
    return [str(item) for item in values]


def _normalise_padding_mask(padding_mask: Any, *, batch_size: int) -> Tensor:
    if not isinstance(padding_mask, Tensor):
        padding_mask = torch.as_tensor(padding_mask)
    mask = padding_mask.detach().to(device=torch.device("cpu"), dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError(f"padding_mask must have shape [B,T], got shape={tuple(mask.shape)}")
    if int(mask.shape[0]) != int(batch_size):
        raise ValueError(f"padding_mask first dimension must equal batch_size, got {tuple(mask.shape)} vs {batch_size}")
    return mask.contiguous()


def _feature_to_btd(
    value: Any,
    *,
    batch_size: int,
    padding_mask: Tensor | None,
    target_name: str,
):
    tensor = _first_feature_tensor(value, target_name=target_name).detach()
    return canonicalize_activation_btc(
        tensor,
        padding_mask=padding_mask,
        batch_size=int(batch_size),
        context=target_name,
    )


class TokenTraceHookManager:
    """Capture Transformer block outputs and FFN input/output token features."""

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_indices: Sequence[int] | None = None,
        target_modules: Sequence[Any] | None = None,
        positions: Sequence[str] | None = None,
        require_padding_mask: bool = True,
        on_capture: Callable[[TokenTraceRecord], None] | None = None,
        cache_records: bool = True,
    ) -> None:
        self.model = model
        self.layer_indices = _normalise_layer_indices(model, layer_indices)
        self.positions = _normalise_trace_positions(positions)
        self.target_modules = _normalise_target_modules(
            target_modules,
            layer_indices=self.layer_indices,
            require_ffn_targets=bool({"input", "output"} & set(self.positions)),
        )
        self.require_padding_mask = bool(require_padding_mask)
        self.on_capture = on_capture
        self.cache_records = bool(cache_records)
        self._handles: list[Any] = []
        self._cache: list[TokenTraceRecord] = []
        self._current_sample_id: list[str] | None = None
        self._current_padding_mask: Tensor | None = None

    def set_batch_context(
        self,
        *,
        sample_id: Sequence[Any] | Tensor,
        padding_mask: Any,
        clear_cache: bool = True,
    ) -> None:
        """Set current batch sample ids and token-level padding mask."""

        mask = _normalise_padding_mask(padding_mask, batch_size=len(sample_id) if not isinstance(sample_id, Tensor) else int(sample_id.numel()))
        sample_ids = _to_sample_id_list(sample_id, batch_size=int(mask.shape[0]))
        if clear_cache:
            self.clear_cache()
        self._current_sample_id = sample_ids
        self._current_padding_mask = mask

    def clear_batch_context(self) -> None:
        """Clear current batch context."""

        self._current_sample_id = None
        self._current_padding_mask = None

    def register_hooks(self) -> None:
        """Register selected block output and FFN input/output hooks."""

        if self._handles:
            raise RuntimeError("TokenTraceHookManager hooks are already registered; call clear_hooks() first")

        if "block_output" in self.positions:
            for layer_idx in self.layer_indices:
                block = get_transformer_block(self.model, int(layer_idx))
                target_name = f"Layer{int(layer_idx)}.block_output"

                def block_hook(
                    _module: nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    hook_layer_idx: int = int(layer_idx),
                    hook_target_name: str = target_name,
                ) -> None:
                    del _module, _inputs
                    self._capture(
                        output,
                        layer_idx=hook_layer_idx,
                        module_name="block",
                        tensor_position="block_output",
                        target_name=hook_target_name,
                    )

                self._handles.append(block.register_forward_hook(block_hook))

        if {"input", "output"} & set(self.positions):
            for layer_idx, module_name in self.target_modules:
                module = get_ffn_module(self.model, int(layer_idx), module_name)
                target_name = f"Layer{int(layer_idx)}.{module_name}"

                def ffn_hook(
                    _module: nn.Module,
                    inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    hook_layer_idx: int = int(layer_idx),
                    hook_module_name: str = module_name,
                    hook_target_name: str = target_name,
                ) -> None:
                    del _module
                    if "input" in self.positions:
                        if not inputs:
                            raise RuntimeError(f"{hook_target_name}: forward hook received no inputs")
                        self._capture(
                            inputs[0],
                            layer_idx=hook_layer_idx,
                            module_name=hook_module_name,
                            tensor_position="input",
                            target_name=f"{hook_target_name}.input",
                        )
                    if "output" in self.positions:
                        self._capture(
                            output,
                            layer_idx=hook_layer_idx,
                            module_name=hook_module_name,
                            tensor_position="output",
                            target_name=f"{hook_target_name}.output",
                        )

                self._handles.append(module.register_forward_hook(ffn_hook))

    def _capture(
        self,
        value: Any,
        *,
        layer_idx: int,
        module_name: str,
        tensor_position: TensorPosition,
        target_name: str,
    ) -> None:
        if self._current_sample_id is None:
            raise RuntimeError(f"{target_name}: token trace capture requires set_batch_context(sample_id, padding_mask)")
        batch_size = len(self._current_sample_id)
        padding_mask = self._current_padding_mask
        if padding_mask is None and self.require_padding_mask:
            raise RuntimeError(f"{target_name}: require_padding_mask=True, but current batch has no padding_mask")
        canonical = _feature_to_btd(value, batch_size=batch_size, padding_mask=padding_mask, target_name=target_name)
        tensor = canonical.tensor.to(device=torch.device("cpu")).contiguous()
        public_layout = layout_info_to_public_fields(canonical.layout_info)
        if padding_mask is None:
            padding_mask = torch.zeros((batch_size, int(tensor.shape[1])), dtype=torch.bool)
        if tuple(padding_mask.shape) != (batch_size, int(tensor.shape[1])):
            raise ValueError(
                f"{target_name}: padding_mask must be [B,T] and align with feature; "
                f"mask_shape={tuple(padding_mask.shape)}, feature_shape={tuple(tensor.shape)}"
            )

        record = TokenTraceRecord(
            layer_idx=int(layer_idx),
            module_name=str(module_name),
            tensor_position=tensor_position,
            target_name=str(target_name),
            tensor=tensor,
            activation_shape=tuple(int(item) for item in tensor.shape),
            original_layout=str(public_layout["original_layout"]),
            original_shape=tuple(int(item) for item in public_layout["original_shape"]),
            canonical_layout=str(public_layout["canonical_layout"]),
            canonical_shape=tuple(int(item) for item in public_layout["canonical_shape"]),
            layout_inference_source=str(public_layout["layout_inference_source"]),
            padding_mask_available=bool(public_layout["padding_mask_available"]),
            padding_mask_aligned=bool(public_layout["padding_mask_aligned"]),
            mask_status=str(public_layout["mask_status"]),
            sample_id=list(self._current_sample_id),
            padding_mask=padding_mask.clone(),
            valid_token_mask=(~padding_mask).clone(),
        )
        if self.cache_records:
            self._cache.append(record)
        if self.on_capture is not None:
            self.on_capture(record)

    def get_cache(self) -> list[TokenTraceRecord]:
        """Return cached token trace records."""

        return list(self._cache)

    def clear_cache(self) -> None:
        """Clear cached token trace records."""

        self._cache.clear()

    def pop_cache(self) -> list[TokenTraceRecord]:
        """Pop cached token trace records and clear the cache."""

        records = self.get_cache()
        self.clear_cache()
        return records

    def clear_hooks(self) -> None:
        """Remove registered hooks."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()


__all__ = [
    "FFN_MODULE_NAMES",
    "FULL_LAYER_INDICES",
    "TOKEN_TRACE_POSITIONS",
    "TokenTraceHookManager",
    "TokenTraceRecord",
    "get_ffn_module",
    "get_transformer_block",
    "resolve_full_layer_indices",
]