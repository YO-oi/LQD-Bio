from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .quant_core import Int4QuantConfig, QuantMode, QuantizationController, default_quant_mode_from_mapping, register_layer_padding_mask_hooks
    from .quant_linear import QuantLinear
except ImportError:  # pragma: no cover - 兼容直接脚本调试。
    from src.quant.quant_core import Int4QuantConfig, QuantMode, QuantizationController, default_quant_mode_from_mapping, register_layer_padding_mask_hooks
    from src.quant.quant_linear import QuantLinear


_ALLOWED_FFN_MODULE_NAMES = {"fc1", "fc2"}
_QUANT_LINEAR_TYPES = (QuantLinear,)
_MODE_ALIASES: dict[str, QuantMode] = {
    "none": "none",
    "fp32": "none",
    "w4": "w4",
    "w4only": "w4",
    "w4_only": "w4",
    "w4-only": "w4",
    "weight": "w4",
    "weight_only": "w4",
    "weight-only": "w4",
    "a4": "a4",
    "a4only": "a4",
    "a4_only": "a4",
    "a4-only": "a4",
    "activation": "a4",
    "activation_only": "a4",
    "activation-only": "a4",
    "w4a4": "w4a4",
    "w4_a4": "w4a4",
    "w4-a4": "w4a4",
}


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


def _get_ffn_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    layers = _get_encoder_layers(model)
    layer_idx = int(layer_idx)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx={layer_idx} 越界，合法范围是 [0, {len(layers) - 1}]")
    return layers[layer_idx]


def _target_name(layer_idx: int, module_name: str) -> str:
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _iter_quant_linears(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, _QUANT_LINEAR_TYPES)]


def _default_mode_from_config(quant_config: Int4QuantConfig | Mapping[str, Any] | None) -> QuantMode:
    if isinstance(quant_config, Mapping):
        if "QUANT" in quant_config and isinstance(quant_config["QUANT"], Mapping):
            return default_quant_mode_from_mapping(quant_config["QUANT"])
        return default_quant_mode_from_mapping(quant_config)
    return "w4a4"


def normalize_quant_mode(mode: str | QuantMode | None, quant_config: Int4QuantConfig | Mapping[str, Any] | None = None) -> QuantMode:
    """把 W4-only / A4-only / W4A4 等实验名统一成 quant_core 的模式名。"""

    if mode is None:
        return _default_mode_from_config(quant_config)
    raw = str(mode).strip().lower().replace(" ", "_")
    normalized = _MODE_ALIASES.get(raw)
    if normalized is None:
        compact = raw.replace("_", "").replace("-", "")
        normalized = _MODE_ALIASES.get(compact)
    if normalized is None:
        raise ValueError(f"quant mode 只允许 none / W4-only / A4-only / W4A4，当前为 {mode!r}")
    return normalized


def get_ffn_module(model: nn.Module, layer_idx: int, module_name: str) -> nn.Module:
    """只按 BEATs FFN 路径定位目标模块：model.encoder.layers[layer_idx].fc1/fc2。"""

    module_name = _validate_module_name(module_name)
    layer = _get_ffn_layer(model, int(layer_idx))
    if not hasattr(layer, module_name):
        raise AttributeError(f"{_target_name(layer_idx, module_name)} 不存在")
    return getattr(layer, module_name)


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是 mapping，当前为 {type(value)!r}")
    return value


def _select_target_qparams(qparams: Mapping[str, Any] | None, *, layer_idx: int, module_name: str) -> Mapping[str, Any]:
    """从多种常见校准文件结构中取出单个目标模块的 qparams。"""

    data = _as_mapping(qparams, name="calibrated_qparams")
    if not data:
        return {}
    target_path = _target_name(layer_idx, module_name)
    tuple_key = (int(layer_idx), str(module_name))
    for container_key in ("modules", "module_qparams", "qparams"):
        nested = data.get(container_key)
        if isinstance(nested, Mapping):
            if target_path in nested and isinstance(nested[target_path], Mapping):
                return nested[target_path]
            if tuple_key in nested and isinstance(nested[tuple_key], Mapping):
                return nested[tuple_key]
    if target_path in data and isinstance(data[target_path], Mapping):
        return data[target_path]
    if tuple_key in data and isinstance(data[tuple_key], Mapping):
        return data[tuple_key]
    if any(key in data for key in ("weight", "weight_qparams", "activation", "activation_qparams")):
        return data
    return {}


def _kind_qparams(module_qparams: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    for key in (kind, f"{kind}_qparams"):
        value = module_qparams.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _tensor_from_qparams(qparams: Mapping[str, Any], key: str, *, default: Any | None = None) -> Tensor:
    if key not in qparams:
        if default is None:
            raise KeyError(f"qparams 缺少字段 {key!r}")
        value = default
    else:
        value = qparams[key]
    if isinstance(value, Tensor):
        return value.detach().clone()
    return torch.as_tensor(value, dtype=torch.float32)


def _optional_qrange(qparams: Mapping[str, Any]) -> tuple[int | None, int | None]:
    qmin = qparams.get("qmin", None)
    qmax = qparams.get("qmax", None)
    return (None if qmin is None else int(qmin), None if qmax is None else int(qmax))


def quant_mode_uses_weight(mode: str | QuantMode | None, quant_config: Int4QuantConfig | Mapping[str, Any] | None = None) -> bool:
    return normalize_quant_mode(mode, quant_config) in {"w4", "w4a4"}


def quant_mode_uses_activation(mode: str | QuantMode | None, quant_config: Int4QuantConfig | Mapping[str, Any] | None = None) -> bool:
    return normalize_quant_mode(mode, quant_config) in {"a4", "w4a4"}


def set_quant_module_mode(module: QuantLinear, mode: str | QuantMode) -> QuantMode:
    """设置单个 QuantLinear 的 W4-only / A4-only / W4A4 运行开关。"""

    normalized = normalize_quant_mode(mode, module.config)
    module.weight_quant_enabled = quant_mode_uses_weight(normalized, module.config)
    module.activation_quant_enabled = quant_mode_uses_activation(normalized, module.config)
    module.calibration_enabled = False
    module.activation_observer.disable()
    module.disable_fake_quant()
    if not module.activation_quant_enabled:
        module.qparams_frozen = True
    return normalized


def apply_fixed_qparams_to_quant_module(
    module: QuantLinear,
    *,
    mode: str | QuantMode,
    weight_qparams: Mapping[str, Any] | None = None,
    activation_qparams: Mapping[str, Any] | None = None,
    require_qparams: bool = True,
) -> None:
    """把固定校准 qparams 写入 QuantLinear；不实现量化公式，只调用 QuantLinear 的公共入口。"""

    normalized = set_quant_module_mode(module, mode)
    weight_qparams = _as_mapping(weight_qparams, name="weight_qparams")
    activation_qparams = _as_mapping(activation_qparams, name="activation_qparams")

    if normalized in {"w4", "w4a4"}:
        if not weight_qparams:
            if require_qparams:
                raise KeyError(f"{module.target_name}: {normalized} 需要固定 weight qparams")
        else:
            qmin, qmax = _optional_qrange(weight_qparams)
            module.set_fixed_weight_qparams(
                _tensor_from_qparams(weight_qparams, "scale"),
                _tensor_from_qparams(weight_qparams, "zero_point", default=0.0),
                qmin=qmin,
                qmax=qmax,
            )

    if normalized in {"a4", "w4a4"}:
        if not activation_qparams:
            if require_qparams:
                raise KeyError(f"{module.target_name}: {normalized} 需要固定 activation qparams")
        else:
            qmin, qmax = _optional_qrange(activation_qparams)
            module.set_fixed_activation_qparams(
                _tensor_from_qparams(activation_qparams, "scale"),
                _tensor_from_qparams(activation_qparams, "zero_point", default=0.0),
                qmin=qmin,
                qmax=qmax,
            )


def replace_ffn_module_with_quant(
    model: nn.Module,
    layer_idx: int,
    module_name: str,
    quant_config: Int4QuantConfig | Mapping[str, Any] | None,
    *,
    quant_mode: str | QuantMode | None = None,
    calibrated_qparams: Mapping[str, Any] | None = None,
    weight_qparams: Mapping[str, Any] | None = None,
    activation_qparams: Mapping[str, Any] | None = None,
    activation_observer_config: Any | None = None,
    require_calibrated_qparams: bool = False,
    enable_fake_quant: bool = False,
) -> QuantLinear:
    """只替换一个 BEATs FFN fc1/fc2 为 QuantLinear。

    量化细节由 QuantLinear / quant_core / calibration / ste 负责，本函数只做：
    - 定位 model.encoder.layers[layer_idx].fc1/fc2；
    - 包装普通 nn.Linear；
    - 写入外部固定 qparams；
    - 切换 W4-only / A4-only / W4A4 模式。
    """

    module_name = _validate_module_name(module_name)
    layer_idx = int(layer_idx)
    target = get_ffn_module(model, layer_idx, module_name)
    target_path = _target_name(layer_idx, module_name)

    existing_quantized = _iter_quant_linears(model)
    if existing_quantized:
        names = ", ".join(name for name, _ in existing_quantized)
        raise RuntimeError(
            "replace_ffn_module_with_quant 要求输入是干净模型；"
            f"当前已存在量化 Linear 包装: {names}"
        )
    if isinstance(target, _QUANT_LINEAR_TYPES):
        raise RuntimeError(f"{target_path} 已经是 QuantLinear，不能重复注入")
    if not isinstance(target, nn.Linear):
        raise TypeError(
            f"{target_path} 必须是 nn.Linear 才能用 QuantLinear 包装，"
            f"当前为 {type(target)!r}。这个 injector 不会改写 attention/LayerNorm/softmax/声学前端。"
        )

    wrapped = QuantLinear(
        target,
        quant_config=quant_config,
        activation_observer_config=activation_observer_config,
        target_name=target_path,
    )
    selected_qparams = _select_target_qparams(calibrated_qparams, layer_idx=layer_idx, module_name=module_name)
    weight_qparams = weight_qparams or _kind_qparams(selected_qparams, "weight")
    activation_qparams = activation_qparams or _kind_qparams(selected_qparams, "activation")
    active_mode = normalize_quant_mode(quant_mode, quant_config)
    apply_fixed_qparams_to_quant_module(
        wrapped,
        mode=active_mode,
        weight_qparams=weight_qparams,
        activation_qparams=activation_qparams,
        require_qparams=bool(require_calibrated_qparams),
    )
    if enable_fake_quant:
        wrapped.enable_fake_quant()

    setattr(_get_ffn_layer(model, layer_idx), module_name, wrapped)
    replaced = _iter_quant_linears(model)
    if len(replaced) != 1 or replaced[0][0] != target_path:
        names = ", ".join(name for name, _ in replaced)
        raise RuntimeError(f"单模块注入后发现量化模块集合异常: {names}")
    return wrapped


def replace_ffn_module_with_calibrated_quant(
    model: nn.Module,
    layer_idx: int,
    module_name: str,
    quant_config: Int4QuantConfig | Mapping[str, Any] | None,
    calibrated_qparams: Mapping[str, Any],
    *,
    quant_mode: str | QuantMode = "w4a4",
    enable_fake_quant: bool = True,
) -> QuantLinear:
    """使用固定校准 qparams 注入单个 FFN 量化模块，默认直接进入 fake quant forward。"""

    return replace_ffn_module_with_quant(
        model,
        layer_idx,
        module_name,
        quant_config,
        quant_mode=quant_mode,
        calibrated_qparams=calibrated_qparams,
        require_calibrated_qparams=True,
        enable_fake_quant=enable_fake_quant,
    )


def make_single_ffn_quant_controller(
    model: nn.Module,
    quant_module: QuantLinear,
    *,
    layer_idx: int,
    module_name: str,
    quant_config: Int4QuantConfig | Mapping[str, Any] | None,
    quant_mode: str | QuantMode,
) -> QuantizationController:
    """为单点 FFN 注入创建 quant_core 控制器，并注册 padding mask hook。"""

    controller = QuantizationController(config=quant_config, mode=normalize_quant_mode(quant_mode, quant_config))
    controller.register_module(_target_name(layer_idx, module_name), quant_module)
    register_layer_padding_mask_hooks(_get_ffn_layer(model, int(layer_idx)), controller=controller)
    return controller


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_checkpoint_path(cfg: Any) -> Path:
    beats_cfg = _cfg_get(cfg, "BEATS", None)
    if isinstance(beats_cfg, Mapping) and "CHECKPOINT" in beats_cfg:
        return Path(beats_cfg["CHECKPOINT"])
    checkpoint = _cfg_get(cfg, "CHECKPOINT", None)
    if checkpoint is not None:
        return Path(checkpoint)
    raise AttributeError("cfg 必须提供 BEATS['CHECKPOINT'] 或 CHECKPOINT")


def restore_clean_model_from_checkpoint(cfg: Any) -> nn.Module:
    """从 BEATs checkpoint 还原未注入量化包装的干净 FP32 模型。"""

    checkpoint_path = _resolve_checkpoint_path(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"BEATs checkpoint 不存在: {checkpoint_path}")

    map_location = _cfg_get(cfg, "MAP_LOCATION", "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint 必须是 mapping，当前为 {type(checkpoint)!r}")
    if "cfg" not in checkpoint:
        raise KeyError("BEATs checkpoint 缺少 'cfg'")
    if "model" not in checkpoint:
        raise KeyError("BEATs checkpoint 缺少 'model'")

    # 懒加载 BEATs，避免普通 injector 导入时触发声学前端依赖。
    from BEATs import BEATs, BEATsConfig

    beats_cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(beats_cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    if _iter_quant_linears(model):
        raise RuntimeError("restore_clean_model_from_checkpoint 还原后不应包含 QuantLinear 包装")
    return model


__all__ = [
    "apply_fixed_qparams_to_quant_module",
    "get_ffn_module",
    "make_single_ffn_quant_controller",
    "normalize_quant_mode",
    "quant_mode_uses_activation",
    "quant_mode_uses_weight",
    "replace_ffn_module_with_calibrated_quant",
    "replace_ffn_module_with_quant",
    "restore_clean_model_from_checkpoint",
    "set_quant_module_mode",
]
