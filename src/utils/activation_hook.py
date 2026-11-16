from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor


_ALLOWED_FFN_MODULE_NAMES = {"fc1", "fc2"}


def _validate_module_name(module_name: str) -> str:
    module_name = str(module_name)
    if module_name not in _ALLOWED_FFN_MODULE_NAMES:
        raise ValueError(f"module_name 只允许 'fc1' 或 'fc2'，当前为 {module_name!r}")
    return module_name


def _get_encoder_layers(model: nn.Module) -> nn.ModuleList:
    if not hasattr(model, "encoder"):
        raise AttributeError("model 缺少 encoder，无法定位 BEATs Transformer FFN")
    encoder = model.encoder
    if not hasattr(encoder, "layers"):
        raise AttributeError("model.encoder 缺少 layers，无法定位 BEATs Transformer FFN")
    layers = encoder.layers
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"model.encoder.layers 必须是 nn.ModuleList，当前为 {type(layers)!r}")
    return layers


def get_target_ffn_module(model: nn.Module, layer_idx: int, module_name: str) -> nn.Module:
    """只允许 fc1 / fc2。"""

    module_name = _validate_module_name(module_name)
    layers = _get_encoder_layers(model)
    layer_idx = int(layer_idx)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx={layer_idx} 越界，合法范围是 [0, {len(layers) - 1}]")
    layer = layers[layer_idx]
    if not hasattr(layer, module_name):
        raise AttributeError(f"encoder.layers.{layer_idx}.{module_name} 不存在")
    return getattr(layer, module_name)


def _normalize_target(target: Any) -> tuple[int, str]:
    if isinstance(target, Mapping):
        if "layer_idx" not in target or "module_name" not in target:
            raise KeyError("target mapping 必须包含 layer_idx 和 module_name")
        return int(target["layer_idx"]), _validate_module_name(str(target["module_name"]))
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)) and len(target) == 2:
        layer_idx, module_name = target
        return int(layer_idx), _validate_module_name(str(module_name))
    raise TypeError(
        "target_modules 中每个 target 必须是 (layer_idx, module_name) "
        "或包含 layer_idx/module_name 的 mapping"
    )


def _detach_activation(
    value: Any,
    *,
    target_name: str,
    activation_name: str,
    cache_device: torch.device | str | None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{target_name}: {activation_name} 必须是 Tensor，当前为 {type(value)!r}")
    detached = value.detach()
    if cache_device is None:
        return detached
    return detached.to(device=torch.device(str(cache_device)))


class ActivationHookManager:
    """统一管理 FFN activation forward hook。"""

    def __init__(
        self,
        model: nn.Module,
        target_modules: Sequence[Any],
        *,
        cache_activations: bool = True,
        cache_device: torch.device | str | None = "cpu",
        max_cache_batches: int | None = None,
        on_capture: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.targets = [_normalize_target(target) for target in target_modules]
        self.cache_activations = bool(cache_activations)
        self.cache_device = cache_device
        self.max_cache_batches = None if max_cache_batches is None else int(max_cache_batches)
        if self.max_cache_batches is not None and self.max_cache_batches <= 0:
            raise ValueError(f"max_cache_batches 必须为正数或 None，当前为 {max_cache_batches}")
        self.on_capture = on_capture
        self._handles: list[Any] = []
        self._cache: dict[tuple[int, str], dict[str, list[Tensor]]] = {
            target: {"input_activation": [], "output_activation": []}
            for target in self.targets
        }

    def register_hooks(self) -> None:
        """给目标 fc1/fc2 注册 forward hook。"""

        if self._handles:
            raise RuntimeError("hooks 已经注册，请先调用 clear_hooks()")

        for layer_idx, module_name in self.targets:
            module = get_target_ffn_module(self.model, layer_idx, module_name)
            target_key = (int(layer_idx), module_name)
            target_name = f"encoder.layers.{int(layer_idx)}.{module_name}"

            def hook(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                *,
                cache_key: tuple[int, str] = target_key,
                cache_name: str = target_name,
            ) -> None:
                del _module
                if not inputs:
                    raise RuntimeError(f"{cache_name}: forward hook 没有捕获到 inputs")
                captured_input = _detach_activation(
                    inputs[0],
                    target_name=cache_name,
                    activation_name="input_activation",
                    cache_device=self.cache_device,
                )
                captured_output = _detach_activation(
                    output,
                    target_name=cache_name,
                    activation_name="output_activation",
                    cache_device=self.cache_device,
                )

                if self.cache_activations:
                    current_batches = len(self._cache[cache_key]["input_activation"])
                    if self.max_cache_batches is not None and current_batches >= self.max_cache_batches:
                        raise RuntimeError(
                            f"{cache_name}: hook cache 已达到 max_cache_batches={self.max_cache_batches}，"
                            "大 eval 集请使用 cache_activations=False 并通过 on_capture 流式消费"
                        )
                    self._cache[cache_key]["input_activation"].append(captured_input)
                    self._cache[cache_key]["output_activation"].append(captured_output)

                if self.on_capture is not None:
                    self.on_capture(
                        {
                            "layer_idx": cache_key[0],
                            "module_name": cache_key[1],
                            "target_name": cache_name,
                            "input_activation": captured_input,
                            "output_activation": captured_output,
                            "raw_shape": tuple(int(item) for item in captured_input.shape),
                            "raw_layout": None,
                            "input_raw_shape": tuple(int(item) for item in captured_input.shape),
                            "input_raw_layout": None,
                            "output_raw_shape": tuple(int(item) for item in captured_output.shape),
                            "output_raw_layout": None,
                        }
                    )

            self._handles.append(module.register_forward_hook(hook))

    def clear_hooks(self) -> None:
        """移除已注册 hook；不清空 cache，便于推理结束后读取。"""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear_cache(self) -> None:
        """Release cached input/output activation tensors without touching hooks."""

        self._cache = {
            target: {"input_activation": [], "output_activation": []}
            for target in self.targets
        }

    def get_cache(self) -> dict[tuple[int, str], dict[str, list[Tensor]]]:
        """返回已缓存的 input_activation / output_activation。"""

        return self._cache


__all__ = [
    "ActivationHookManager",
    "get_target_ffn_module",
]