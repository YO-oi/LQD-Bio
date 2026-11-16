from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .observer import ObserverConfig, load_project_observer_configs, observe_weight
except ImportError:  # pragma: no cover - 兼容直接以脚本方式调试。
    from src.quant.observer import ObserverConfig, load_project_observer_configs, observe_weight


QuantMode = Literal["none", "w4", "a4", "w4a4"]
PercentileMode = Literal["retained_mass"]


@dataclass(frozen=True)
class Int4QuantConfig:
    """FFN INT4 fake quant 的固定协议配置。"""

    weight_bits: int = 4
    activation_bits: int = 4
    weight_qmin: int = -7
    weight_qmax: int = 7
    activation_qmin: int = -7
    activation_qmax: int = 7
    weight_symmetric: bool = True
    activation_symmetric: bool = True
    activation_percentile_mode: PercentileMode = "retained_mass"
    activation_percentile: float = 99.9
    activation_observer_bins: int = 2048
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.weight_bits != 4:
            raise ValueError(f"Only W4 is supported, got weight_bits={self.weight_bits}")
        if self.activation_bits != 4:
            raise ValueError(f"Only A4 is supported, got activation_bits={self.activation_bits}")
        if not self.weight_symmetric:
            raise ValueError("Weight quantization must be symmetric for this FFN INT4 protocol.")
        if self.weight_qmin >= self.weight_qmax:
            raise ValueError("weight_qmin must be smaller than weight_qmax")
        if self.activation_qmin >= self.activation_qmax:
            raise ValueError("activation_qmin must be smaller than activation_qmax")
        if self.weight_symmetric and int(self.weight_qmin) != -int(self.weight_qmax):
            raise ValueError(
                "symmetric W4 must use the effective signed INT4 codebook [-7, 7], "
                f"got [{self.weight_qmin}, {self.weight_qmax}]"
            )
        if self.activation_symmetric and int(self.activation_qmin) != -int(self.activation_qmax):
            raise ValueError(
                "symmetric A4 must use the effective signed INT4 codebook [-7, 7], "
                f"got [{self.activation_qmin}, {self.activation_qmax}]"
            )
        if self.weight_symmetric and int(self.weight_qmax) != 7:
            raise ValueError("symmetric W4 must use qmax=7 so scale=amax/7 and saturation=[-7S, 7S]")
        if self.activation_symmetric and int(self.activation_qmax) != 7:
            raise ValueError("symmetric A4 must use qmax=7 so scale=threshold/7 and saturation=[-7S, 7S]")
        if self.weight_qmax <= 0:
            raise ValueError("symmetric weight quantization requires weight_qmax > 0")
        if not (0.0 < float(self.activation_percentile) <= 100.0):
            raise ValueError(
                "activation_percentile must be in (0, 100], "
                f"got {self.activation_percentile}"
            )
        if self.activation_percentile_mode != "retained_mass":
            raise ValueError(
                "activation_percentile_mode must be 'retained_mass', "
                f"got {self.activation_percentile_mode!r}"
            )
        if int(self.activation_observer_bins) <= 1:
            raise ValueError("activation_observer_bins must be > 1")
        if float(self.eps) <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    def symmetric_abs_percentile(self) -> float:
        """symmetric observer 使用 abs activation 的 retained-mass 分位。"""

        return float(self.activation_percentile)

    def asymmetric_percentile_bounds(self) -> tuple[float, float]:
        """asymmetric observer 的中心保留区间。"""

        tail_mass = max(0.0, 100.0 - float(self.activation_percentile))
        lower = tail_mass / 2.0
        upper = 100.0 - tail_mass / 2.0
        return lower, upper


def int4_quant_config_from_mapping(quant_cfg: Mapping[str, Any]) -> Int4QuantConfig:
    """从 config.QUANT 风格的映射构造 Int4QuantConfig。"""

    weight_cfg = dict(quant_cfg.get("WEIGHT", {}))
    activation_cfg = dict(quant_cfg.get("ACTIVATION", {}))

    weight_granularity = str(weight_cfg.get("GRANULARITY", "per_channel"))
    activation_granularity = str(activation_cfg.get("GRANULARITY", "per_tensor"))
    activation_method = str(activation_cfg.get("CALIBRATION_METHOD", "percentile"))
    activation_percentile_mode = str(activation_cfg.get("PERCENTILE_MODE", "retained_mass"))
    weight_channel_axis = int(weight_cfg.get("CHANNEL_AXIS", 0))

    if str(quant_cfg.get("TARGET_SCOPE", "beats_transformer_ffn")) != "beats_transformer_ffn":
        raise ValueError("QUANT.TARGET_SCOPE must be 'beats_transformer_ffn'")
    if tuple(quant_cfg.get("TARGET_LINEAR_NAMES", ("fc1", "fc2"))) != ("fc1", "fc2"):
        raise ValueError("QUANT.TARGET_LINEAR_NAMES must be ('fc1', 'fc2')")
    if weight_granularity != "per_channel":
        raise ValueError("QUANT.WEIGHT.GRANULARITY must be 'per_channel'")
    if weight_channel_axis != 0:
        raise ValueError("QUANT.WEIGHT.CHANNEL_AXIS must be 0 for nn.Linear out_features")
    if activation_granularity != "per_tensor":
        raise ValueError("QUANT.ACTIVATION.GRANULARITY must be 'per_tensor'")
    if activation_method != "percentile":
        raise ValueError("QUANT.ACTIVATION.CALIBRATION_METHOD must be 'percentile'")
    if activation_percentile_mode != "retained_mass":
        raise ValueError("QUANT.ACTIVATION.PERCENTILE_MODE must be 'retained_mass'")

    weight_eps = float(weight_cfg.get("EPS", 1e-8))
    activation_eps = float(activation_cfg.get("EPS", weight_eps))
    if weight_eps != activation_eps:
        raise ValueError("QUANT.WEIGHT.EPS and QUANT.ACTIVATION.EPS must match")

    return Int4QuantConfig(
        weight_bits=int(weight_cfg.get("BITS", 4)),
        activation_bits=int(activation_cfg.get("BITS", 4)),
        weight_qmin=int(weight_cfg.get("QMIN", -7)),
        weight_qmax=int(weight_cfg.get("QMAX", 7)),
        activation_qmin=int(activation_cfg.get("QMIN", -7)),
        activation_qmax=int(activation_cfg.get("QMAX", 7)),
        weight_symmetric=bool(weight_cfg.get("SYMMETRIC", True)),
        activation_symmetric=bool(activation_cfg.get("SYMMETRIC", True)),
        activation_percentile_mode=activation_percentile_mode,  # type: ignore[arg-type]
        activation_percentile=float(activation_cfg.get("PERCENTILE", 99.9)),
        activation_observer_bins=int(activation_cfg.get("OBSERVER_BINS", 2048)),
        eps=weight_eps,
    )


def default_quant_mode_from_mapping(quant_cfg: Mapping[str, Any]) -> QuantMode:
    mode = str(quant_cfg.get("DEFAULT_MODE", "w4a4")).lower()
    if mode not in {"none", "w4", "a4", "w4a4"}:
        raise ValueError(f"QUANT.DEFAULT_MODE must be one of none/w4/a4/w4a4, got {mode!r}")
    return mode  # type: ignore[return-value]


def load_project_quant_defaults() -> tuple[QuantMode, Int4QuantConfig]:
    """惰性读取项目根 config.QUANT。"""

    import config as project_config

    quant_cfg = getattr(project_config, "QUANT")
    return default_quant_mode_from_mapping(quant_cfg), int4_quant_config_from_mapping(quant_cfg)


def _coerce_quant_config(config: Int4QuantConfig | Mapping[str, Any] | None) -> Int4QuantConfig:
    if config is None:
        return Int4QuantConfig()
    if isinstance(config, Int4QuantConfig):
        return config
    if isinstance(config, Mapping):
        return int4_quant_config_from_mapping(config)
    raise TypeError(f"Unsupported quant config type: {type(config)!r}")


def _default_observer_configs_from_quant_config(config: Int4QuantConfig) -> dict[str, ObserverConfig]:
    activation_percentiles = {99.0, float(config.activation_percentile), 99.99}
    activation_low: float | None = None
    activation_high = float(config.activation_percentile)
    if not config.activation_symmetric:
        activation_low, activation_high = config.asymmetric_percentile_bounds()
        activation_percentiles.update({activation_low, activation_high})

    return {
        "activation": ObserverConfig(
            observer_type="activation",
            granularity="per_tensor",
            qmin=int(config.activation_qmin),
            qmax=int(config.activation_qmax),
            symmetric=bool(config.activation_symmetric),
            percentiles=tuple(activation_percentiles),
            percentile_low=activation_low,
            percentile_high=activation_high,
            eps=float(config.eps),
        ),
        "weight": ObserverConfig(
            observer_type="weight",
            granularity="per_channel",
            qmin=int(config.weight_qmin),
            qmax=int(config.weight_qmax),
            symmetric=bool(config.weight_symmetric),
            percentiles=(99.0, 99.9, 99.99),
            percentile_low=None,
            percentile_high=99.9,
            channel_axis=0,
            eps=float(config.eps),
        ),
    }


def _coerce_observer_configs(
    observer_configs: Mapping[str, ObserverConfig] | None,
    quant_config: Int4QuantConfig,
) -> dict[str, ObserverConfig]:
    defaults = _default_observer_configs_from_quant_config(quant_config)
    if observer_configs is None:
        return defaults

    merged = dict(defaults)
    for key in ("activation", "weight"):
        if key in observer_configs:
            value = observer_configs[key]
            if not isinstance(value, ObserverConfig):
                raise TypeError(f"observer_configs[{key!r}] must be ObserverConfig, got {type(value)!r}")
            merged[key] = value
    return merged


def _quant_linear_class() -> type[nn.Module]:
    # 延迟导入，避免 quant_linear.py 读取 Int4QuantConfig 时形成循环导入。
    try:
        from .quant_linear import QuantLinear
    except ImportError:  # pragma: no cover - 兼容直接以脚本方式调试。
        from src.quant.quant_linear import QuantLinear
    return QuantLinear


def _module_cache_valid(module: nn.Module, attr_name: str) -> bool:
    return bool(getattr(module, attr_name, False))


class QuantizationController:
    """统一管理 FFN fake quant 状态机和一组 QuantLinear 模块。"""

    def __init__(
        self,
        config: Int4QuantConfig | Mapping[str, Any] | None = None,
        mode: QuantMode = "none",
        observer_configs: Mapping[str, ObserverConfig] | None = None,
    ) -> None:
        self.config = _coerce_quant_config(config)
        self.observer_configs = _coerce_observer_configs(observer_configs, self.config)
        self.observer_enabled = False
        self.fake_quant_enabled = False
        self.qparams_frozen = False
        self.weight_quant_enabled = False
        self.activation_quant_enabled = False
        self.modules: dict[str, nn.Module] = {}
        self._activation_padding_mask_stack: list[Any] = []
        self._padding_mask_hook_handles: list[Any] = []
        self.set_quant_mode(mode)

    def register_module(self, name: str, module: nn.Module) -> None:
        """注册一个 QuantLinear；controller 不再创建第二套 Linear wrapper。"""

        if name in self.modules and self.modules[name] is not module:
            raise ValueError(f"duplicate quantized module name: {name}")
        if not hasattr(module, "activation_observer"):
            raise TypeError("QuantizationController 只能注册带 activation_observer 的 QuantLinear 模块")
        self.modules[name] = module
        self._sync_module_runtime_flags(module)
        if hasattr(module, "set_activation_padding_mask_provider"):
            module.set_activation_padding_mask_provider(lambda self=self: self.current_activation_padding_mask)
        module.activation_observer.set_enabled(self.observer_enabled)  # type: ignore[attr-defined]

    def _sync_module_runtime_flags(self, module: nn.Module) -> None:
        if hasattr(module, "weight_quant_enabled"):
            module.weight_quant_enabled = bool(self.weight_quant_enabled)  # type: ignore[attr-defined]
        if hasattr(module, "activation_quant_enabled"):
            module.activation_quant_enabled = bool(self.activation_quant_enabled)  # type: ignore[attr-defined]
        if hasattr(module, "fake_quant_enabled"):
            module.fake_quant_enabled = bool(self.fake_quant_enabled)  # type: ignore[attr-defined]
        if hasattr(module, "calibration_enabled"):
            module.calibration_enabled = bool(self.observer_enabled and self.activation_quant_enabled)  # type: ignore[attr-defined]

    def _sync_all_module_runtime_flags(self) -> None:
        for module in self.modules.values():
            self._sync_module_runtime_flags(module)

    def set_quant_mode(self, mode: QuantMode, *, reset_activation_state: bool = True) -> None:
        mode = mode.lower()  # type: ignore[assignment]
        if mode not in {"none", "w4", "a4", "w4a4"}:
            raise ValueError(f"Unsupported quant mode={mode!r}; use none/w4/a4/w4a4")
        previous_weight_quant_enabled = bool(self.weight_quant_enabled)
        previous_activation_quant_enabled = bool(self.activation_quant_enabled)
        self.weight_quant_enabled = mode in {"w4", "w4a4"}
        self.activation_quant_enabled = mode in {"a4", "w4a4"}
        self.fake_quant_enabled = False
        self.set_observer_enabled(False)
        self._sync_all_module_runtime_flags()

        if previous_weight_quant_enabled or self.weight_quant_enabled:
            self.invalidate_weight_qparams_cache()

        if reset_activation_state and (
            previous_activation_quant_enabled or self.activation_quant_enabled or self.qparams_frozen
        ):
            self.invalidate_activation_qparams(reset_observers=True)

    def set_observer_enabled(self, enabled: bool) -> None:
        self.observer_enabled = bool(enabled)
        for module in self.modules.values():
            module.activation_observer.set_enabled(self.observer_enabled)  # type: ignore[attr-defined]
            if hasattr(module, "calibration_enabled"):
                module.calibration_enabled = bool(self.observer_enabled and self.activation_quant_enabled)  # type: ignore[attr-defined]

    def set_fake_quant_enabled(self, enabled: bool) -> None:
        if enabled and self.activation_quant_enabled and not self.qparams_frozen:
            raise RuntimeError("activation fake quant requires frozen calibration qparams")
        self.fake_quant_enabled = bool(enabled)
        for module in self.modules.values():
            if self.fake_quant_enabled:
                module.enable_fake_quant()  # type: ignore[attr-defined]
            else:
                module.disable_fake_quant()  # type: ignore[attr-defined]

    def prepare_calibration(self, *, reset_observers: bool = True) -> None:
        """兼容入口：开始第一遍 range calibration。"""

        self.prepare_calibration_range(reset_observers=reset_observers)

    def prepare_calibration_range(self, *, reset_observers: bool = True) -> None:
        """第一遍 calibration：observer 只统计 FP32 activation 范围和基础矩。"""

        if reset_observers:
            self.reset_observers()
        for module in self.modules.values():
            module.activation_observer.start_range_pass(reset=False)  # type: ignore[attr-defined]
            if hasattr(module, "qparams_frozen"):
                module.qparams_frozen = False  # type: ignore[attr-defined]
        self.set_observer_enabled(True)
        self.fake_quant_enabled = False
        self.qparams_frozen = False
        self._sync_all_module_runtime_flags()

    def prepare_calibration_histogram(self) -> None:
        """第二遍 calibration：用第一遍范围固定 histogram 并累积分布。"""

        if self.activation_quant_enabled:
            for module in self.modules.values():
                module.activation_observer.start_histogram_pass()  # type: ignore[attr-defined]
        self.set_observer_enabled(True)
        self.fake_quant_enabled = False
        self.qparams_frozen = False
        self._sync_all_module_runtime_flags()

    def freeze_qparams(self) -> None:
        if self.activation_quant_enabled:
            for module in self.modules.values():
                module.freeze_calibration()  # type: ignore[attr-defined]
        self.qparams_frozen = True

    def prepare_quant_forward(self) -> None:
        self.set_observer_enabled(False)
        self.set_fake_quant_enabled(True)

    def disable_all_runtime_quant(self) -> None:
        self.set_observer_enabled(False)
        self.set_fake_quant_enabled(False)

    def reset_observers(self) -> None:
        for module in self.modules.values():
            if hasattr(module, "reset_activation_observer"):
                module.reset_activation_observer()  # type: ignore[attr-defined]
            else:
                module.activation_observer.reset()  # type: ignore[attr-defined]

    def invalidate_activation_qparams(self, *, reset_observers: bool = True) -> None:
        """显式失效 activation qparams，避免 ablation 间复用旧 calibration。"""

        self.qparams_frozen = False
        if reset_observers:
            self.reset_observers()
        else:
            for module in self.modules.values():
                if hasattr(module, "invalidate_activation_qparams"):
                    module.invalidate_activation_qparams()  # type: ignore[attr-defined]

    def invalidate_weight_qparams_cache(self) -> None:
        """显式清理 weight qparams 缓存，供权重 data 级外部改写后调用。"""

        for module in self.modules.values():
            if hasattr(module, "invalidate_weight_qparams_cache"):
                module.invalidate_weight_qparams_cache()  # type: ignore[attr-defined]

    def push_activation_padding_mask(self, padding_mask: Tensor | Mapping[str, Any] | None) -> None:
        if isinstance(padding_mask, Mapping):
            payload = dict(padding_mask)
            mask = payload.get("padding_mask", payload.get("mask"))
            if mask is not None:
                payload["padding_mask"] = torch.as_tensor(mask).detach().to(dtype=torch.bool)
                payload["padding_mask_shape"] = tuple(int(item) for item in payload["padding_mask"].shape)
                payload.setdefault("mask_layout", "BT")
            self._activation_padding_mask_stack.append(payload)
            return
        if padding_mask is not None:
            padding_mask = padding_mask.detach().to(dtype=torch.bool)
        self._activation_padding_mask_stack.append(padding_mask)

    def pop_activation_padding_mask(self) -> None:
        if not self._activation_padding_mask_stack:
            raise RuntimeError("activation padding mask stack underflow")
        self._activation_padding_mask_stack.pop()

    @property
    def current_activation_padding_mask(self) -> Tensor | None:
        if not self._activation_padding_mask_stack:
            return None
        return self._activation_padding_mask_stack[-1]

    def register_padding_mask_hook_handle(self, handle: Any) -> None:
        self._padding_mask_hook_handles.append(handle)

    def remove_padding_mask_hooks(self) -> None:
        for handle in self._padding_mask_hook_handles:
            handle.remove()
        self._padding_mask_hook_handles.clear()

    def observer_summary(self) -> dict[str, dict[str, Any]]:
        return {name: module.activation_observer.state_summary() for name, module in self.modules.items()}  # type: ignore[attr-defined]

    def weight_observer_summary(self) -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        for name, module in self.modules.items():
            linear = getattr(module, "linear", None)
            weight = getattr(linear, "weight", None)
            if weight is None:
                continue
            summaries[name] = observe_weight(
                weight,
                config=self.observer_configs["weight"],
                tensor_name=f"{name}.weight",
                module_name=name,
                layer_idx=_infer_layer_idx_from_name(name),
                module_type="nn.Linear",
            )
        return summaries

    def state_summary(self) -> dict[str, Any]:
        observer_summaries = list(self.observer_summary().values())
        return {
            "observer_enabled": bool(self.observer_enabled),
            "fake_quant_enabled": bool(self.fake_quant_enabled),
            "qparams_frozen": bool(self.qparams_frozen),
            "weight_quant_enabled": bool(self.weight_quant_enabled),
            "activation_quant_enabled": bool(self.activation_quant_enabled),
            "num_quantized_modules": int(len(self.modules)),
            "target_names": sorted(self.modules),
            "observer_backend": "two_pass_histogram",
            "observer_num_updates": int(sum(item["num_batches"] for item in observer_summaries)),
            "observer_num_values": int(sum(item["num_elements"] for item in observer_summaries)),
            "observer_num_valid_tokens": int(sum(item["num_valid_tokens"] for item in observer_summaries)),
            "observer_num_padding_tokens": int(sum(item["num_padding_tokens"] for item in observer_summaries)),
            "observer_cpu_transfer_num_values": int(sum(item["cpu_transfer_num_elements"] for item in observer_summaries)),
            "observer_cached_chunks": int(sum(item["num_cached_chunks"] for item in observer_summaries)),
            "observer_may_include_padding_tokens": bool(any(item["may_include_padding_tokens"] for item in observer_summaries)),
            "weight_qparams_cached_modules": int(
                sum(1 for module in self.modules.values() if _module_cache_valid(module, "weight_qparams_cache_is_valid"))
            ),
            "activation_qparams_cached_modules": int(
                sum(1 for module in self.modules.values() if _module_cache_valid(module, "activation_qparams_cache_is_valid"))
            ),
        }


def _infer_layer_idx_from_name(name: str) -> int | None:
    parts = str(name).split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers" and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def _has_layers_index_prefix(parts: list[str], end_index: int) -> bool:
    for idx in range(0, end_index - 1):
        if parts[idx] == "layers" and parts[idx + 1].isdigit():
            return True
    return False


def is_beats_ffn_linear_name(name: str) -> bool:
    """只匹配 BEATs Transformer block 的 FFN fc1/fc2 线性层。"""

    parts = str(name).split(".")
    if len(parts) < 3:
        return False

    if parts[-1] in {"fc1", "fc2"} and _has_layers_index_prefix(parts, len(parts) - 1):
        return True

    # activation_name="glu" 时，fc1 是 GLU_Linear，真正的线性层在 fc1.linear。
    if len(parts) >= 4 and parts[-2] == "fc1" and parts[-1] == "linear":
        return _has_layers_index_prefix(parts, len(parts) - 2)

    return False


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    if not parts:
        raise ValueError("module_name must not be empty")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _layer_name_for_ffn_target(name: str) -> str | None:
    parts = str(name).split(".")
    if parts[-1] in {"fc1", "fc2"}:
        return ".".join(parts[:-1])
    if len(parts) >= 2 and parts[-2] == "fc1" and parts[-1] == "linear":
        return ".".join(parts[:-2])
    return None


def _extract_self_attn_padding_mask(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Tensor | None:
    if "self_attn_padding_mask" in kwargs:
        return kwargs["self_attn_padding_mask"]
    # TransformerSentenceEncoderLayer.forward(x, self_attn_mask, self_attn_padding_mask, ...)
    if len(args) >= 3:
        candidate = args[2]
        if candidate is None or isinstance(candidate, Tensor):
            return candidate
    return None


def _register_layer_padding_mask_hooks(
    layer: nn.Module,
    *,
    controller: QuantizationController,
) -> None:
    def pre_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del module
        padding_mask = _extract_self_attn_padding_mask(args, kwargs)
        activation = args[0] if args and isinstance(args[0], Tensor) else None
        controller.push_activation_padding_mask(
            {
                "padding_mask": padding_mask,
                "activation_layout": None,
                "mask_layout": "BT" if padding_mask is not None else None,
                "raw_shape": None if activation is None else tuple(int(item) for item in activation.shape),
                "padding_mask_shape": None if padding_mask is None else tuple(int(item) for item in padding_mask.shape),
            }
        )

    def post_hook(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs, output
        controller.pop_activation_padding_mask()

    controller.register_padding_mask_hook_handle(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
    controller.register_padding_mask_hook_handle(layer.register_forward_hook(post_hook, with_kwargs=True, always_call=True))


def register_layer_padding_mask_hooks(
    layer: nn.Module,
    *,
    controller: QuantizationController,
) -> None:
    """为单点 FFN 量化注入复用公共 padding mask hook。"""

    _register_layer_padding_mask_hooks(layer, controller=controller)


def replace_beats_ffn_linear_with_quantized(
    model: nn.Module,
    *,
    controller: QuantizationController,
    allow_no_targets: bool = False,
) -> list[str]:
    """把 BEATs Transformer FFN 的 fc1/fc2 替换为 QuantLinear。"""

    QuantLinear = _quant_linear_class()
    replaced: list[str] = []
    named_modules = list(model.named_modules())
    module_lookup = dict(named_modules)
    hooked_layer_names: set[str] = set()
    for name, module in named_modules:
        if not name or not is_beats_ffn_linear_name(name):
            continue
        if isinstance(module, QuantLinear):
            controller.register_module(name, module)
            replaced.append(name)
            layer_name = _layer_name_for_ffn_target(name)
            if layer_name is not None and layer_name not in hooked_layer_names:
                layer_module = module_lookup.get(layer_name)
                if layer_module is not None:
                    _register_layer_padding_mask_hooks(layer_module, controller=controller)
                    hooked_layer_names.add(layer_name)
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent, child_name = _get_parent_module(model, name)
        wrapped = QuantLinear(
            module,
            quant_config=controller.config,
            activation_observer_config=controller.observer_configs["activation"],
            target_name=name,
        )
        setattr(parent, child_name, wrapped)
        controller.register_module(name, wrapped)
        replaced.append(name)
        layer_name = _layer_name_for_ffn_target(name)
        if layer_name is not None and layer_name not in hooked_layer_names:
            layer_module = module_lookup.get(layer_name)
            if layer_module is not None:
                _register_layer_padding_mask_hooks(layer_module, controller=controller)
                hooked_layer_names.add(layer_name)

    if not replaced and not allow_no_targets:
        raise RuntimeError("No BEATs Transformer FFN fc1/fc2 nn.Linear targets were found.")
    return replaced


def prepare_beats_ffn_int4_quantization(
    model: nn.Module,
    *,
    mode: QuantMode,
    config: Int4QuantConfig | Mapping[str, Any] | None = None,
    observer_configs: Mapping[str, ObserverConfig] | None = None,
    allow_no_targets: bool = False,
) -> QuantizationController:
    """创建控制器并替换 BEATs FFN fc1/fc2 线性层。"""

    controller = QuantizationController(config=config, mode=mode, observer_configs=observer_configs)
    replace_beats_ffn_linear_with_quantized(model, controller=controller, allow_no_targets=allow_no_targets)
    return controller


def prepare_beats_ffn_int4_quantization_from_project_config(
    model: nn.Module,
    *,
    allow_no_targets: bool = False,
) -> QuantizationController:
    """使用项目根 config.QUANT 默认协议准备 BEATs FFN INT4 fake quant。"""

    mode, quant_config = load_project_quant_defaults()
    observer_configs = load_project_observer_configs()
    return prepare_beats_ffn_int4_quantization(
        model,
        mode=mode,
        config=quant_config,
        observer_configs=observer_configs,
        allow_no_targets=allow_no_targets,
    )


def iter_quantized_linears(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    QuantLinear = _quant_linear_class()
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            yield name, module


__all__ = [
    "Int4QuantConfig",
    "ObserverConfig",
    "QuantizationController",
    "default_quant_mode_from_mapping",
    "int4_quant_config_from_mapping",
    "is_beats_ffn_linear_name",
    "iter_quantized_linears",
    "register_layer_padding_mask_hooks",
    "load_project_quant_defaults",
    "prepare_beats_ffn_int4_quantization",
    "prepare_beats_ffn_int4_quantization_from_project_config",
    "replace_beats_ffn_linear_with_quantized",
]
