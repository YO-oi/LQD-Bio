from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from src.utils.semantic_geometry_cache import (
        DEFAULT_EMBEDDING_DIM,
        FP32_GRANULARITY,
        FP32_QUANT_MODE,
        SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION,
        build_fp32_embedding_cache,
        build_quant_embedding_cache,
        build_semantic_embedding_identity,
        atomic_torch_save,
        file_identity,
        qparams_identity_manifest,
        save_embedding_cache,
    )
    from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from src.export.export_fp32_core_cache import (
        _pooled_logits as core_pooled_logits,
        backbone_features_for_cache,
        mask_aware_mean as core_mask_aware_mean,
    )
    from src.quant.calibration import freeze_weight_qparams_from_tensor
    from src.quant.observer import ObserverConfig
    from src.quant.quant_core import QuantizationController, register_layer_padding_mask_hooks
    from src.quant.quant_linear import QuantLinear
    from src.utils.group_utils import read_canonical_bucket, tail_included
    from src.quant.ffn_quant_injector import (
        apply_fixed_qparams_to_quant_module,
        get_ffn_module,
        make_single_ffn_quant_controller,
        normalize_quant_mode,
        replace_ffn_module_with_quant,
        restore_clean_model_from_checkpoint,
    )
except ImportError:  # pragma: no cover - 兼容直接以脚本方式调试。
    from ..utils.semantic_geometry_cache import (
        DEFAULT_EMBEDDING_DIM,
        FP32_GRANULARITY,
        FP32_QUANT_MODE,
        SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION,
        build_fp32_embedding_cache,
        build_quant_embedding_cache,
        build_semantic_embedding_identity,
        atomic_torch_save,
        file_identity,
        qparams_identity_manifest,
        save_embedding_cache,
    )
    from ..data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from ..export.export_fp32_core_cache import (
        _pooled_logits as core_pooled_logits,
        backbone_features_for_cache,
        mask_aware_mean as core_mask_aware_mean,
    )
    from ..quant.calibration import freeze_weight_qparams_from_tensor
    from ..quant.observer import ObserverConfig
    from ..quant.quant_core import QuantizationController, register_layer_padding_mask_hooks
    from ..quant.quant_linear import QuantLinear
    from ..utils.group_utils import read_canonical_bucket, tail_included
    from ..quant.ffn_quant_injector import (
        apply_fixed_qparams_to_quant_module,
        get_ffn_module,
        make_single_ffn_quant_controller,
        normalize_quant_mode,
        replace_ffn_module_with_quant,
        restore_clean_model_from_checkpoint,
    )


TARGET_MODULE_NAMES = ("fc1", "fc2")
QUANT_MODE_LABELS = {
    "none": "FP32",
    "w4": "W4-only",
    "a4": "A4-only",
    "w4a4": "W4A4",
}
SUPPORTED_ACTIVATION_GRANULARITIES = ("per_tensor", "per_channel")
DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES = ("per_tensor", "per_channel")
DEFAULT_RUNNER_CFG_NAME = "SEMANTIC_GEOMETRY_MODE_RUNNER"


@dataclass(frozen=True)
class ForwardResult:
    """模型 forward 的标准输出。

    features 用于最终 embedding pooling；logits 只在模型显式返回时保存，不在这里计算指标。
    """

    features: Tensor
    padding_mask: Tensor | None = None
    logits: Tensor | None = None


@dataclass(frozen=True)
class ModeRunResult:
    """单次模式运行结果。

    embedding_cache 是后续 semantic_geometry_metrics.py 的标准输入；qparams 只在量化模式下存在。
    """

    embedding_cache: dict[str, Any]
    qparams: dict[str, Any] | None
    runtime_summary: dict[str, Any]
    cache_path: Path | None = None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def _runner_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, DEFAULT_RUNNER_CFG_NAME, {})
    return value if isinstance(value, Mapping) else {}


def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}


def _quant_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "QUANT", {})
    if not isinstance(value, Mapping):
        raise TypeError("cfg.QUANT 必须是 mapping，用于 FFN INT4 注入")
    return value


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    runner = _runner_cfg(cfg)
    if key in runner:
        return runner[key]
    geometry_probe = _cfg_get(cfg, "SEMANTIC_GEOMETRY_PROJECTION", {})
    if isinstance(geometry_probe, Mapping) and key in geometry_probe:
        return geometry_probe[key]
    return _cfg_get(cfg, key, default)


def _compute_logits_if_available(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "COMPUTE_LOGITS_IF_AVAILABLE", True))


def _dataset_name(cfg: Any) -> str:
    dataset_cfg = _dataset_cfg(cfg)
    for key in ("NAME", "name", "DATASET_NAME", "dataset_name"):
        value = _nested_get(dataset_cfg, key, None)
        if value:
            return str(value)
    active_dataset = _cfg_get(cfg, "ACTIVE_DATASET", None)
    if active_dataset:
        return str(active_dataset)
    dataset_root = _nested_get(dataset_cfg, "ROOT", None)
    if dataset_root is not None:
        return Path(dataset_root).name
    current_result_dir = _cfg_get(cfg, "CURRENT_RESULT_DIR", None)
    if current_result_dir is not None:
        return Path(current_result_dir).name
    raise AttributeError("无法确定数据集名称：请配置 ACTIVE_DATASET 或 CURRENT_DATASET['ROOT']")


def _artifact_root(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "semantic_geometry"


def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", _cfg_get(cfg, "DEVICE", None))
    if configured is None:
        raise AttributeError("semantic_geometry_mode_runner 必须从 config.py 获取 DEVICE")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py 配置 DEVICE={configured!r}，但当前环境 CUDA 不可用")
    return device


def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


def _make_audio_config(cfg: Any) -> AudioInputConfig:
    audio = _cfg_get(cfg, "AUDIO", {})
    if not isinstance(audio, Mapping):
        audio = {}
    return AudioInputConfig(
        sample_rate=int(audio.get("TARGET_SR", 16000)),
        target_seconds=float(audio.get("TARGET_SECONDS", 10.0)),
        normalization=str(audio.get("NORMALIZATION", "peak")),
        eps=float(audio.get("EPS", 1e-8)),
        deterministic_crop=str(audio.get("CROP_MODE", "center")),  # type: ignore[arg-type]
        fixed_crop_start=int(audio.get("FIXED_CROP_START", 0)),
    )


def _split_csv(cfg: Any, name: str) -> Path:
    key = f"{name.upper()}_CSV"
    value = _setting_get(cfg, key, None)
    if value is not None:
        return Path(value)
    dataset_value = _nested_get(_dataset_cfg(cfg), key, None)
    if dataset_value is not None:
        return Path(dataset_value)
    raise AttributeError(f"无法找到 {key}，请在 cfg.CURRENT_DATASET 或 cfg.{DEFAULT_RUNNER_CFG_NAME} 中配置")


def _split_csv_optional(cfg: Any, name: str) -> Path | None:
    try:
        return _split_csv(cfg, name)
    except AttributeError:
        return None


def _checkpoint_identity(cfg: Any) -> dict[str, Any] | None:
    for key in ("CHECKPOINT_PATH", "CKPT_PATH", "MODEL_CHECKPOINT", "MODEL_CHECKPOINT_PATH", "BEATS_CHECKPOINT"):
        value = _setting_get(cfg, key, _cfg_get(cfg, key, None))
        if value is not None:
            return file_identity(value)
    return None


def _target_modules_identity(target_modules: Sequence[tuple[int, str]]) -> list[list[Any]]:
    return [[int(layer_idx), str(module_name)] for layer_idx, module_name in target_modules]


def _qparams_identity_for_target_map(
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    target_modules: Sequence[tuple[int, str]],
) -> dict[str, Any]:
    detail: dict[str, dict[str, Any]] = {}
    identity_map: dict[str, Any] = {}
    source_map: dict[str, Any] = {}
    type_map: dict[str, Any] = {}
    for layer_idx, module_name in target_modules:
        target = _target_name(int(layer_idx), str(module_name))
        qparams = _qparams_for_target(qparams_by_target, layer_idx=int(layer_idx), module_name=str(module_name))
        item = qparams_identity_manifest(qparams, source_hint=target)
        detail[target] = item
        identity_map[target] = item.get("qparams_identity")
        source_map[target] = item.get("qparams_source")
        type_map[target] = item.get("qparams_identity_type")
    return {
        "qparams_identity_type": "target_map",
        "qparams_identity": identity_map,
        "qparams_source": source_map,
        "qparams_identity_detail": detail,
        "qparams_identity_type_by_target": type_map,
    }


def _embedding_identity_manifest_fields(
    cfg: Any,
    *,
    scope: str,
    quant_mode: str,
    granularity: str,
    target_modules: Sequence[tuple[int, str]],
    sample_ids: Sequence[Any],
    qparams_identity_info: Mapping[str, Any],
    embedding_dim: int | None = DEFAULT_EMBEDDING_DIM,
) -> dict[str, Any]:
    identity = build_semantic_embedding_identity(
        scope=scope,
        quant_mode=normalize_mode_label(quant_mode),
        granularity=granularity,
        target_modules=_target_modules_identity(target_modules),
        sample_ids=sample_ids,
        eval_csv=_split_csv_optional(cfg, "eval"),
        calibration_csv=None if normalize_mode_label(quant_mode) == FP32_QUANT_MODE else _split_csv_optional(cfg, "calibration"),
        checkpoint=_checkpoint_identity(cfg),
        quant_config=_cfg_get(cfg, "QUANT", {}),
        compute_logits_if_available=_compute_logits_if_available(cfg),
        embedding_dim=embedding_dim,
        qparams_identity_type=str(qparams_identity_info.get("qparams_identity_type")),
        qparams_identity=qparams_identity_info.get("qparams_identity"),
        qparams_source=qparams_identity_info.get("qparams_source"),
    )
    fields = {
        "embedding_cache_identity_version": SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION,
        "embedding_identity": identity,
        "qparams_identity_type": qparams_identity_info.get("qparams_identity_type"),
        "qparams_identity": qparams_identity_info.get("qparams_identity"),
        "qparams_source": qparams_identity_info.get("qparams_source"),
    }
    for optional_key in ("qparams_identity_detail", "qparams_identity_type_by_target"):
        if optional_key in qparams_identity_info:
            fields[optional_key] = qparams_identity_info[optional_key]
    return fields


def _assert_full_model_default_runtime_supported(
    target_modules: Sequence[tuple[int, str]],
    custom_quant_runner: Callable[..., tuple[Any, Any]] | None,
) -> None:
    _ = (target_modules, custom_quant_runner)


def _batch_size(cfg: Any, name: str) -> int:
    return int(_setting_get(cfg, f"{name.upper()}_BATCH_SIZE", _setting_get(cfg, "BATCH_SIZE", 4)))


def _num_workers(cfg: Any) -> int:
    return int(_setting_get(cfg, "NUM_WORKERS", 0))


def _pin_memory(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "PIN_MEMORY", _device(cfg).type == "cuda"))


def build_audio_loader(cfg: Any, *, name: str, mode: str) -> DataLoader:
    """按统一 waveform 输入协议构建固定 loader。

    eval/export/geometry 都应传 deterministic 模式，不能在语义几何阶段使用随机裁剪。
    """

    override = _cfg_get(cfg, f"build_{name}_loader", None)
    if callable(override):
        return override(cfg)
    dataset = AudioInputDataset(
        _split_csv(cfg, name),
        mode=mode,
        dataset_root=_nested_get(_dataset_cfg(cfg), "ROOT", None),
        project_root=_cfg_get(cfg, "BASE_DIR", None),
        audio_config=_make_audio_config(cfg),
        global_seed=_seed(cfg),
        epoch=0,
        return_dict=True,
        require_label_id=True,
        include_audio_stats=False,
    )
    return DataLoader(
        dataset,
        batch_size=_batch_size(cfg, name),
        shuffle=False,
        num_workers=_num_workers(cfg),
        pin_memory=_pin_memory(cfg),
        collate_fn=partial(audio_input_collate_fn, include_audio_stats=False),
    )


def build_eval_loader(cfg: Any) -> DataLoader:
    """构建固定 eval.csv loader。"""

    return build_audio_loader(cfg, name="eval", mode="eval")


def build_calibration_loader(cfg: Any) -> DataLoader:
    """构建固定 calibration.csv loader。"""

    return build_audio_loader(cfg, name="calibration", mode="calibration")


def load_sample_metadata_from_csv(path: str | Path) -> dict[str, dict[str, Any]]:
    """读取 split CSV 中的 label / bucket 元信息，用 sample_id 对齐。"""

    import pandas as pd

    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise KeyError(f"{path} 缺少 sample_id 列")
    if "bucket" not in frame.columns:
        raise KeyError(f"{path} 缺少样本级 bucket 列；shot_group 只能作为可选兼容列")
    duplicated = frame["sample_id"].astype(str).duplicated()
    if bool(duplicated.any()):
        preview = frame.loc[duplicated, "sample_id"].astype(str).head(8).tolist()
        raise ValueError(f"{path} sample_id 重复: {preview}")

    bucket_result = read_canonical_bucket(
        bucket=frame["bucket"],
        shot_group=frame["shot_group"] if "shot_group" in frame.columns else None,
        context=f"{path} semantic geometry metadata",
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in bucket_result.bucket]
    tail_flags = [bool(item) for item in tail_included(buckets)]

    result: dict[str, dict[str, Any]] = {}
    for idx, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        label = row["label"] if "label" in frame.columns else row["primary_label"] if "primary_label" in frame.columns else row.get("label_id", sample_id)
        bucket = buckets[int(idx)]
        result[sample_id] = {
            "label": label,
            "bucket": bucket,
            "shot_group": bucket,
            "tail_included": tail_flags[int(idx)],
        }
    return result


def load_eval_metadata(cfg: Any) -> dict[str, dict[str, Any]]:
    """从 eval.csv 读取样本级元信息。"""

    return load_sample_metadata_from_csv(_split_csv(cfg, "eval"))


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
    return moved


def _extract_forward_result(output: Any) -> ForwardResult:
    if isinstance(output, Tensor):
        return ForwardResult(features=output)
    if isinstance(output, Mapping):
        feature_tensor = None
        for key in ("embedding", "embeddings", "features", "feature", "x", "encoder_out"):
            value = output.get(key)
            if isinstance(value, Tensor):
                feature_tensor = value
                break
        if feature_tensor is None:
            raise TypeError("模型输出 dict 中没有可用于 embedding pooling 的 tensor 字段")
        padding_mask = output.get("padding_mask")
        logits = output.get("logits")
        return ForwardResult(
            features=feature_tensor,
            padding_mask=padding_mask if isinstance(padding_mask, Tensor) else None,
            logits=logits if isinstance(logits, Tensor) else None,
        )
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if not isinstance(first, Tensor):
            raise TypeError(f"模型输出第一个元素必须是 Tensor，当前为 {type(first)!r}")
        padding_mask = output[1] if len(output) > 1 and isinstance(output[1], Tensor) else None
        logits = output[2] if len(output) > 2 and isinstance(output[2], Tensor) else None
        return ForwardResult(features=first, padding_mask=padding_mask, logits=logits)
    raise TypeError(f"不支持的模型输出类型: {type(output)!r}")


def _forward_model(model: nn.Module, batch: Mapping[str, Any], cfg: Any) -> ForwardResult:
    for name in ("semantic_geometry_forward", "ffn_quant_mode_forward", "ffn_fp32_reference_forward", "ffn_heatmap_forward"):
        forward_fn = _cfg_get(cfg, name, None)
        if callable(forward_fn):
            return _extract_forward_result(forward_fn(model, batch, cfg))
    waveform = batch["waveform"]
    padding_mask = batch.get("padding_mask")
    if all(hasattr(model, name) for name in ("preprocess", "patch_embedding", "encoder")):
        features, token_padding_mask = backbone_features_for_cache(model, waveform, padding_mask)
        embedding, _valid_token_count = core_mask_aware_mean(features, token_padding_mask)
        logits = None
        if _compute_logits_if_available(cfg):
            try:
                logits = core_pooled_logits(model, features, token_padding_mask)
            except AttributeError:
                logits = None
        return ForwardResult(features=embedding, logits=logits)
    if hasattr(model, "extract_features"):
        return _extract_forward_result(model.extract_features(waveform, padding_mask=padding_mask))
    try:
        return _extract_forward_result(model(waveform, padding_mask=padding_mask))
    except TypeError:
        return _extract_forward_result(model(waveform))


def _pool_embedding(features: Tensor, padding_mask: Tensor | None, *, require_mask_aware: bool = True) -> Tensor:
    """最终 embedding 统一使用 mask-aware mean；不能保存成 [1,768]。"""

    if features.ndim == 2:
        return features.detach()
    if features.ndim == 3:
        if padding_mask is not None and tuple(padding_mask.shape) == tuple(features.shape[:2]):
            mask = padding_mask.to(device=features.device, dtype=torch.bool)
            valid = (~mask).unsqueeze(-1)
            summed = features.masked_fill(~valid, 0.0).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1).to(dtype=features.dtype)
            return (summed / denom).detach()
        if require_mask_aware:
            raise RuntimeError("语义几何 embedding pooling 要求 mask-aware mean，但 padding_mask 无法匹配 features")
        return features.mean(dim=1).detach()
    if features.ndim > 3:
        return features.flatten(start_dim=1).detach()
    return features.reshape(features.shape[0], -1).detach()


def _autocast_context(cfg: Any, device: torch.device) -> Any:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    use_amp = bool(beats_cfg.get("USE_AMP", False)) if isinstance(beats_cfg, Mapping) else False
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def restore_clean_model(cfg: Any) -> nn.Module:
    """每次模式运行前都重新加载干净 FP32 模型。"""

    for name in ("restore_clean_model", "restore_clean_model_from_checkpoint"):
        fn = _cfg_get(cfg, name, None)
        if callable(fn):
            model = fn(cfg)
            if not isinstance(model, nn.Module):
                raise TypeError(f"{name} 必须返回 nn.Module，当前为 {type(model)!r}")
            return model
    return restore_clean_model_from_checkpoint(cfg)


def normalize_mode_label(mode: str) -> str:
    normalized = normalize_quant_mode(mode, None)
    return QUANT_MODE_LABELS[normalized]


def normalize_activation_granularity(granularity: str | None) -> str:
    value = str(granularity or "per_tensor").strip().lower().replace("-", "_")
    aliases = {
        "tensor": "per_tensor",
        "per_tensor": "per_tensor",
        "activation_per_tensor": "per_tensor",
        "channel": "per_channel",
        "per_channel": "per_channel",
        "activation_per_channel": "per_channel",
    }
    normalized = aliases.get(value)
    if normalized not in SUPPORTED_ACTIVATION_GRANULARITIES:
        raise ValueError(f"activation granularity 只允许 per_tensor / per_channel，当前为 {granularity!r}")
    return normalized


def activation_observer_config_for_granularity(cfg: Any, activation_granularity: str | None) -> ObserverConfig:
    """Build an activation observer config for an explicit diagnostic granularity."""

    granularity = normalize_activation_granularity(activation_granularity)
    quant_cfg = _quant_cfg(cfg)
    activation_cfg = quant_cfg.get("ACTIVATION", {})
    activation_cfg = activation_cfg if isinstance(activation_cfg, Mapping) else {}
    observer_root = _cfg_get(cfg, "OBSERVER", {})
    observer_activation = {}
    if isinstance(observer_root, Mapping):
        observer_activation = observer_root.get("ACTIVATION", {})
        observer_activation = observer_activation if isinstance(observer_activation, Mapping) else {}

    percentile_high = float(activation_cfg.get("PERCENTILE", 99.9))
    percentiles = set(float(item) for item in observer_activation.get("PERCENTILES", (99.0, percentile_high, 99.99)))
    percentiles.update({99.0, percentile_high, 99.99})
    channel_axis = -1 if granularity == "per_channel" else int(observer_activation.get("CHANNEL_AXIS", 0))
    return ObserverConfig(
        observer_type="activation",
        granularity=granularity,  # type: ignore[arg-type]
        qmin=int(activation_cfg.get("QMIN", -7)),
        qmax=int(activation_cfg.get("QMAX", 7)),
        symmetric=bool(activation_cfg.get("SYMMETRIC", True)),
        percentiles=tuple(sorted(percentiles)),
        percentile_low=observer_activation.get("PERCENTILE_LOW", None),
        percentile_high=observer_activation.get("PERCENTILE_HIGH", percentile_high),
        channel_axis=channel_axis,
        eps=float(activation_cfg.get("EPS", 1e-8)),
        histogram_bins=int(observer_activation.get("HISTOGRAM_BINS", activation_cfg.get("OBSERVER_BINS", 2048))),
        percentile_backend=str(observer_activation.get("PERCENTILE_BACKEND", "two_pass_histogram")),  # type: ignore[arg-type]
    )


def _mode_uses_weight(mode: str) -> bool:
    return normalize_quant_mode(mode, None) in {"w4", "w4a4"}


def _mode_uses_activation(mode: str) -> bool:
    return normalize_quant_mode(mode, None) in {"a4", "w4a4"}


def _target_name(layer_idx: int, module_name: str) -> str:
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _ffn_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise AttributeError("model.encoder.layers does not exist")
    layers = model.encoder.layers
    layer_idx = int(layer_idx)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx={layer_idx} out of range [0, {len(layers) - 1}]")
    return layers[layer_idx]


def _all_target_modules(model: nn.Module) -> list[tuple[int, str]]:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise AttributeError("model.encoder.layers 不存在，无法遍历 BEATs Transformer FFN")
    targets: list[tuple[int, str]] = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        for module_name in TARGET_MODULE_NAMES:
            if not hasattr(layer, module_name):
                raise AttributeError(f"encoder.layers.{layer_idx}.{module_name} 不存在")
            targets.append((int(layer_idx), module_name))
    if not targets:
        raise ValueError("没有找到任何 FFN fc1/fc2 目标模块")
    return targets


def target_modules_from_config(cfg: Any, model: nn.Module | None = None) -> list[tuple[int, str]]:
    """读取目标 FFN 模块；未配置时默认全部 fc1/fc2。"""

    probe_model = model if model is not None else restore_clean_model(cfg)
    configured = _setting_get(cfg, "TARGET_MODULES", None)
    if configured is None:
        return _all_target_modules(probe_model)
    result: list[tuple[int, str]] = []
    for item in configured:
        layer_idx, module_name = item
        module_name = str(module_name)
        if module_name not in TARGET_MODULE_NAMES:
            raise ValueError(f"module_name 只允许 fc1/fc2，当前为 {module_name!r}")
        get_ffn_module(probe_model, int(layer_idx), module_name)
        result.append((int(layer_idx), module_name))
    if not result:
        raise ValueError("TARGET_MODULES 不能为空")
    return result


def _run_forward_only(model: nn.Module, loader: DataLoader, cfg: Any) -> None:
    device = _device(cfg)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            with _autocast_context(cfg, device):
                _forward_model(model, moved, cfg)


def _run_two_pass_calibration(controller: Any, model: nn.Module, loader: DataLoader, cfg: Any) -> None:
    """两遍 activation calibration：range pass + histogram pass。"""

    if controller.activation_quant_enabled:
        controller.prepare_calibration_range(reset_observers=True)
        _run_forward_only(model, loader, cfg)
        controller.prepare_calibration_histogram()
        _run_forward_only(model, loader, cfg)
        controller.freeze_qparams()
    controller.qparams_frozen = True
    controller.prepare_quant_forward()


def _qrange_from_config(qconfig: Any, *, kind: str) -> tuple[int, int]:
    if kind == "weight":
        return int(qconfig.weight_qmin), int(qconfig.weight_qmax)
    return int(qconfig.activation_qmin), int(qconfig.activation_qmax)


def _activation_percentile_from_quant_cfg(cfg: Any) -> float:
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    if not isinstance(activation_cfg, Mapping):
        return 99.9
    return float(activation_cfg.get("PERCENTILE", 99.9))


def _activation_percentile_mode_from_quant_cfg(cfg: Any) -> str:
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    if not isinstance(activation_cfg, Mapping):
        return "retained_mass"
    return str(activation_cfg.get("PERCENTILE_MODE", "retained_mass"))


def _canonical_path_text(path: str | Path) -> str:
    """把路径规范成可比较字符串，用于校验 qparams 是否来自同一 calibration.csv。"""

    value = Path(path).expanduser()
    try:
        return str(value.resolve(strict=False))
    except Exception:
        return str(value)


def _expected_calibration_provenance(cfg: Any, activation_granularity: str) -> dict[str, Any]:
    """当前公平对比要求的 calibration 元信息。"""

    return {
        "calibration_csv": _canonical_path_text(_split_csv(cfg, "calibration")),
        "activation_percentile": float(_activation_percentile_from_quant_cfg(cfg)),
        "activation_percentile_mode": _activation_percentile_mode_from_quant_cfg(cfg),
        "activation_granularity": normalize_activation_granularity(activation_granularity),
    }


def _mapping_section(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    section = value.get(key, {})
    return section if isinstance(section, Mapping) else {}


def _qparam_meta_value(qparams: Mapping[str, Any], *keys: str) -> Any:
    """从 qparams 顶层、manifest、calibration_provenance、activation 区块读取元信息。"""

    sections = [
        qparams,
        _mapping_section(qparams, "calibration_provenance"),
        _mapping_section(qparams, "calibration"),
        _mapping_section(qparams, "manifest"),
        _mapping_section(qparams, "activation"),
    ]
    for section in sections:
        for key in keys:
            if key in section and section[key] is not None:
                return section[key]
    return None


def _same_path_text(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return _canonical_path_text(str(left)).lower() == _canonical_path_text(str(right)).lower()


def attach_calibration_provenance(
    qparams: Mapping[str, Any],
    cfg: Any,
    *,
    activation_granularity: str,
    source: str,
) -> dict[str, Any]:
    """把公平校准元信息写进 qparams，保证落盘文件可独立审计。"""

    result = dict(qparams)
    expected = _expected_calibration_provenance(cfg, activation_granularity)
    provenance = dict(_mapping_section(result, "calibration_provenance"))
    provenance.update(expected)
    provenance.update({"source": str(source), "runner": "semantic_geometry_mode_runner"})
    result["calibration_provenance"] = provenance
    result["calibration_csv"] = expected["calibration_csv"]
    result["activation_percentile"] = expected["activation_percentile"]
    result["activation_percentile_mode"] = expected["activation_percentile_mode"]
    result["activation_granularity"] = expected["activation_granularity"]

    activation = dict(_mapping_section(result, "activation"))
    if activation:
        activation.setdefault("calibration_csv", expected["calibration_csv"])
        activation.setdefault("percentile", expected["activation_percentile"])
        activation.setdefault("percentile_mode", expected["activation_percentile_mode"])
        activation.setdefault("granularity", expected["activation_granularity"])
        result["activation"] = activation
    return result


def validate_calibration_provenance(
    qparams: Mapping[str, Any],
    cfg: Any,
    *,
    activation_granularity: str,
    require_activation: bool,
    context: str,
) -> None:
    """强制校验 activation qparams 是否来自同一 calibration.csv 与同一 percentile。"""

    if not require_activation:
        return
    expected = _expected_calibration_provenance(cfg, activation_granularity)

    actual_csv = _qparam_meta_value(qparams, "calibration_csv", "csv", "calibration_path")
    if actual_csv is None:
        raise RuntimeError(f"{context}: activation qparams 必须记录 calibration_csv，不能只依赖 runner manifest 声明")
    if not _same_path_text(actual_csv, expected["calibration_csv"]):
        raise RuntimeError(
            f"{context}: activation qparams 的 calibration_csv 与当前配置不一致；"
            f"qparams={actual_csv!r}, expected={expected['calibration_csv']!r}"
        )

    actual_percentile = _qparam_meta_value(qparams, "activation_percentile", "percentile")
    if actual_percentile is None:
        raise RuntimeError(f"{context}: activation qparams 必须记录 activation_percentile")
    if abs(float(actual_percentile) - float(expected["activation_percentile"])) > 1e-9:
        raise RuntimeError(
            f"{context}: activation_percentile 与当前配置不一致；"
            f"qparams={float(actual_percentile)}, expected={float(expected['activation_percentile'])}"
        )

    actual_granularity = _qparam_meta_value(qparams, "activation_granularity", "granularity")
    if actual_granularity is None:
        raise RuntimeError(f"{context}: activation qparams 必须记录 activation_granularity")
    if normalize_activation_granularity(str(actual_granularity)) != expected["activation_granularity"]:
        raise RuntimeError(
            f"{context}: activation_granularity 与当前运行配置不一致；"
            f"qparams={actual_granularity!r}, expected={expected['activation_granularity']!r}"
        )

    actual_mode = _qparam_meta_value(qparams, "activation_percentile_mode", "percentile_mode")
    if actual_mode is not None and str(actual_mode) != str(expected["activation_percentile_mode"]):
        raise RuntimeError(
            f"{context}: activation_percentile_mode 与当前配置不一致；"
            f"qparams={actual_mode!r}, expected={expected['activation_percentile_mode']!r}"
        )


def _assert_current_runtime_supports_granularity(granularity: str) -> None:
    """校验默认 QuantLinear runtime 是否支持该 activation 粒度。"""

    if granularity in DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES:
        return
    raise NotImplementedError(
        "当前默认 QuantLinear / quant_core runtime 只实现 activation per_tensor。"
        "per_channel 是协议级可选项，正式运行必须传入 custom_quant_runner，"
        "并提供带 calibration_csv / activation_percentile / activation_granularity 的固定 qparams。"
    )


def calibrate_target_qparams(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    calibration_loader: DataLoader,
    activation_granularity: str = "per_tensor",
) -> dict[str, Any]:
    """对单个目标模块计算一次固定 qparams。

    W4-only / A4-only / W4A4 复用同一套 qparams；activation per-tensor 和 per-channel 必须使用同一
    calibration_loader 与同一 percentile。
    """

    activation_granularity = normalize_activation_granularity(activation_granularity)
    _assert_current_runtime_supports_granularity(activation_granularity)
    activation_observer_config = activation_observer_config_for_granularity(cfg, activation_granularity)

    device = _device(cfg)
    model = restore_clean_model(cfg).to(device)
    quant_module = replace_ffn_module_with_quant(
        model,
        int(layer_idx),
        module_name,
        _quant_cfg(cfg),
        quant_mode="W4A4",
        activation_observer_config=activation_observer_config,
        enable_fake_quant=False,
    )
    controller = make_single_ffn_quant_controller(
        model,
        quant_module,
        layer_idx=int(layer_idx),
        module_name=module_name,
        quant_config=_quant_cfg(cfg),
        quant_mode="W4A4",
    )
    try:
        _run_two_pass_calibration(controller, model, calibration_loader, cfg)
        w_qmin, w_qmax = _qrange_from_config(quant_module.config, kind="weight")
        a_qmin, a_qmax = _qrange_from_config(quant_module.config, kind="activation")
        weight_qparams = freeze_weight_qparams_from_tensor(
            quant_module.linear.weight.detach().cpu(),
            qmin=w_qmin,
            qmax=w_qmax,
            symmetric=True,
            eps=float(quant_module.config.eps),
            channel_axis=0,
        )
        activation_qparams = {
            "scale": quant_module.activation_scale.detach().cpu().clone(),
            "zero_point": quant_module.activation_zero_point.detach().cpu().clone(),
            "qmin": a_qmin,
            "qmax": a_qmax,
            "symmetric": bool(quant_module.config.activation_symmetric),
            "granularity": activation_granularity,
            "observer_type": "activation",
            "method": "calibration_two_pass_percentile",
            "channel_axis": int(activation_observer_config.channel_axis),
            "percentile": _activation_percentile_from_quant_cfg(cfg),
        }
        qparams = {
            "layer_idx": int(layer_idx),
            "module_name": str(module_name),
            "target_name": _target_name(layer_idx, module_name),
            "activation_granularity": activation_granularity,
            "weight": weight_qparams,
            "activation": activation_qparams,
            "observer_summary": controller.observer_summary(),
            "weight_observer_summary": controller.weight_observer_summary(),
            "controller_summary": controller.state_summary(),
        }
        return attach_calibration_provenance(
            qparams,
            cfg,
            activation_granularity=activation_granularity,
            source="default_two_pass_calibration",
        )
    finally:
        controller.remove_padding_mask_hooks()


def qparams_cache_path(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    activation_granularity: str = "per_tensor",
) -> Path:
    activation_granularity = normalize_activation_granularity(activation_granularity)
    return _artifact_root(cfg) / "qparams" / f"layer_{int(layer_idx):02d}_{module_name}_{activation_granularity}_qparams.pt"


def save_qparams(qparams: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(dict(qparams), path)
    return path


def load_qparams(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"qparams 文件不存在: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"qparams 文件必须保存 mapping，当前为 {type(payload)!r}")
    return dict(payload)


def load_or_calibrate_qparams(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    calibration_loader: DataLoader,
    activation_granularity: str = "per_tensor",
    force_recalibrate: bool = False,
    qparams_path: str | Path | None = None,
) -> dict[str, Any]:
    """复用固定校准参数；不存在或强制重算时才跑 calibration.csv。"""

    path = Path(qparams_path) if qparams_path is not None else qparams_cache_path(
        cfg,
        layer_idx=int(layer_idx),
        module_name=module_name,
        activation_granularity=activation_granularity,
    )
    if path.exists() and not bool(force_recalibrate):
        qparams = load_qparams(path)
    else:
        qparams = calibrate_target_qparams(
            cfg,
            layer_idx=int(layer_idx),
            module_name=module_name,
            calibration_loader=calibration_loader,
            activation_granularity=activation_granularity,
        )
        save_qparams(qparams, path)
    _validate_qparams_for_target(
        qparams,
        layer_idx=int(layer_idx),
        module_name=module_name,
        activation_granularity=activation_granularity,
        require_weight=True,
        require_activation=True,
    )
    validate_calibration_provenance(
        qparams,
        cfg,
        activation_granularity=activation_granularity,
        require_activation=True,
        context=f"{_target_name(layer_idx, module_name)} qparams",
    )
    qparams["qparams_path"] = str(path)
    return qparams


def _validate_qparams_for_target(
    qparams: Mapping[str, Any],
    *,
    layer_idx: int,
    module_name: str,
    activation_granularity: str,
    require_weight: bool = True,
    require_activation: bool = True,
) -> None:
    expected_target = _target_name(layer_idx, module_name)
    if str(qparams.get("target_name", expected_target)) != expected_target:
        raise ValueError(f"qparams target_name 不匹配: {qparams.get('target_name')} vs {expected_target}")
    if require_activation and str(qparams.get("activation_granularity", activation_granularity)) != normalize_activation_granularity(activation_granularity):
        raise ValueError("qparams activation_granularity 与当前运行配置不一致")
    required_kinds: list[str] = []
    if require_weight:
        required_kinds.append("weight")
    if require_activation:
        required_kinds.append("activation")
    for kind in required_kinds:
        section = qparams.get(kind)
        if not isinstance(section, Mapping):
            raise KeyError(f"qparams 缺少 {kind} 区块")
        for field in ("scale", "zero_point", "qmin", "qmax"):
            if field not in section:
                raise KeyError(f"qparams.{kind} 缺少字段 {field!r}")


def _require_qparams_for_quant_mode(
    *,
    quant_mode: str,
    weight_qparams: Mapping[str, Any] | None,
    activation_qparams: Mapping[str, Any] | None,
    target_name: str,
) -> None:
    if _mode_uses_weight(quant_mode):
        if not isinstance(weight_qparams, Mapping) or "scale" not in weight_qparams or "zero_point" not in weight_qparams:
            raise RuntimeError(f"{target_name} {quant_mode}: W4 路径必须提供固定 weight qparams")
    if _mode_uses_activation(quant_mode):
        if not isinstance(activation_qparams, Mapping) or "scale" not in activation_qparams or "zero_point" not in activation_qparams:
            raise RuntimeError(f"{target_name} {quant_mode}: A4 路径必须提供固定 activation qparams")


def _assert_quant_module_has_fixed_qparams(quant_module: Any, quant_mode: str, target_name: str) -> None:
    if _mode_uses_weight(quant_mode):
        if not bool(getattr(quant_module, "fixed_weight_qparams_enabled", False)):
            raise RuntimeError(f"{target_name} {quant_mode}: weight fake quant 未启用固定 qparams")
        weight_scale = getattr(quant_module, "weight_scale", None)
        if not isinstance(weight_scale, Tensor) or int(weight_scale.numel()) <= 0:
            raise RuntimeError(f"{target_name} {quant_mode}: fixed weight scale 为空")
    if _mode_uses_activation(quant_mode):
        if not bool(getattr(quant_module, "qparams_frozen", False)):
            raise RuntimeError(f"{target_name} {quant_mode}: activation qparams 未冻结")
        activation_scale = getattr(quant_module, "activation_scale", None)
        if not isinstance(activation_scale, Tensor) or not torch.isfinite(activation_scale.detach()).all().item():
            raise RuntimeError(f"{target_name} {quant_mode}: activation scale 非有限")


def _metadata_for_batch(
    batch: Mapping[str, Any],
    *,
    sample_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[Any], list[str]]:
    sample_ids = [str(item) for item in batch["sample_id"]]
    label_id = batch.get("label_id")
    labels: list[Any] = []
    bucket_values: list[Any] = []
    shot_values: list[Any] = []
    for index, sample_id in enumerate(sample_ids):
        metadata = sample_metadata.get(sample_id, {}) if sample_metadata is not None else {}
        if "label" in metadata:
            labels.append(metadata["label"])
        elif "primary_label" in metadata:
            labels.append(metadata["primary_label"])
        elif isinstance(label_id, Tensor):
            labels.append(str(int(label_id.detach().cpu().reshape(-1)[index].item())))
        else:
            labels.append(sample_id)
        bucket_values.append(metadata.get("bucket"))
        shot_values.append(metadata.get("shot_group"))

    bucket_result = read_canonical_bucket(
        bucket=bucket_values,
        shot_group=shot_values if any(item is not None for item in shot_values) else None,
        context="semantic_geometry batch metadata",
        allow_shot_group_fallback=False,
    )
    return labels, [str(item) for item in bucket_result.bucket]


def collect_final_embeddings(
    cfg: Any,
    *,
    model: nn.Module,
    eval_loader: DataLoader,
    sample_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """跑固定 eval set，输出最终 embedding、logits 与样本对齐元信息。

    本函数只收集模型输出，不计算 Fisher、margin 或任何几何指标。
    """

    device = _device(cfg)
    require_mask_aware = bool(_setting_get(cfg, "REQUIRE_MASK_AWARE_POOLING", True))
    compute_logits = _compute_logits_if_available(cfg)
    model.eval()
    sample_ids: list[str] = []
    labels: list[Any] = []
    buckets: list[str] = []
    label_chunks: list[Tensor] = []
    embedding_chunks: list[Tensor] = []
    logits_chunks: list[Tensor] = []

    with torch.no_grad():
        for batch in eval_loader:
            batch_sample_ids = [str(item) for item in batch["sample_id"]]
            if "label_id" not in batch or not isinstance(batch["label_id"], Tensor):
                raise KeyError("eval_loader batch 必须包含 tensor 类型 label_id")
            batch_label_id = batch["label_id"].detach().cpu().long().reshape(-1)
            if len(batch_sample_ids) != int(batch_label_id.numel()):
                raise RuntimeError("sample_id 数量必须和 label_id 数量一致")

            batch_labels, batch_buckets = _metadata_for_batch(batch, sample_metadata=sample_metadata)
            moved = _move_batch_to_device(batch, device)
            with _autocast_context(cfg, device):
                output = _forward_model(model, moved, cfg)
            padding_mask = output.padding_mask if output.padding_mask is not None else moved.get("padding_mask")
            pooled = _pool_embedding(output.features.float(), padding_mask, require_mask_aware=require_mask_aware)
            if int(pooled.shape[0]) != len(batch_sample_ids):
                raise RuntimeError(f"pooled embedding B 维与 sample_id 数量不一致: {tuple(pooled.shape)} vs {len(batch_sample_ids)}")
            embedding_chunks.append(pooled.cpu())
            label_chunks.append(batch_label_id)
            sample_ids.extend(batch_sample_ids)
            labels.extend(batch_labels)
            buckets.extend(batch_buckets)
            if compute_logits and output.logits is not None:
                logits = output.logits.detach().float().cpu()
                if logits.ndim != 2 or int(logits.shape[0]) != len(batch_sample_ids):
                    raise RuntimeError(f"logits 必须是 [B,C]，当前 shape={tuple(logits.shape)}")
                logits_chunks.append(logits)

    if not embedding_chunks:
        raise RuntimeError("固定 eval_loader 没有产生任何 batch")
    result: dict[str, Any] = {
        "sample_id": sample_ids,
        "label": labels,
        "bucket": buckets,
        "shot_group": list(buckets),
        "tail_included": [bool(item) for item in tail_included(buckets)],
        "label_id": torch.cat(label_chunks, dim=0),
        "embedding": torch.cat(embedding_chunks, dim=0),
    }
    if logits_chunks:
        result["logits"] = torch.cat(logits_chunks, dim=0)
    return result


def _build_cache_from_embedding_payload(
    payload: Mapping[str, Any],
    *,
    quant_mode: str,
    granularity: str,
    expected_dim: int | None,
    extra_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if normalize_quant_mode(quant_mode, None) == "none":
        return build_fp32_embedding_cache(
            sample_id=payload["sample_id"],
            label=payload["label"],
            bucket=payload["bucket"],
            shot_group=payload.get("shot_group"),
            embedding=payload["embedding"],
            label_id=payload.get("label_id"),
            logits=payload.get("logits"),
            expected_dim=expected_dim,
            extra_manifest=extra_manifest,
        )
    return build_quant_embedding_cache(
        sample_id=payload["sample_id"],
        label=payload["label"],
        bucket=payload["bucket"],
        shot_group=payload.get("shot_group"),
        embedding=payload["embedding"],
        quant_mode=normalize_mode_label(quant_mode),
        granularity=granularity,
        label_id=payload.get("label_id"),
        logits=payload.get("logits"),
        expected_dim=expected_dim,
        extra_manifest=extra_manifest,
    )


def run_fp32_embedding_mode(
    cfg: Any,
    *,
    eval_loader: DataLoader | None = None,
    sample_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    save_path: str | Path | None = None,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
) -> ModeRunResult:
    """运行干净 FP32 teacher，输出最终 embedding cache。"""

    loader = eval_loader if eval_loader is not None else build_eval_loader(cfg)
    metadata = sample_metadata if sample_metadata is not None else load_eval_metadata(cfg)
    device = _device(cfg)
    model = restore_clean_model(cfg).to(device)
    payload = collect_final_embeddings(cfg, model=model, eval_loader=loader, sample_metadata=metadata)
    identity_fields = _embedding_identity_manifest_fields(
        cfg,
        scope="fp32",
        quant_mode=FP32_QUANT_MODE,
        granularity=FP32_GRANULARITY,
        target_modules=[(-1, "fp32")],
        sample_ids=payload["sample_id"],
        qparams_identity_info=qparams_identity_manifest(None, source_hint="fp32_no_qparams"),
        embedding_dim=expected_dim,
    )
    cache = _build_cache_from_embedding_payload(
        payload,
        quant_mode=FP32_QUANT_MODE,
        granularity=FP32_GRANULARITY,
        expected_dim=expected_dim,
        extra_manifest={
            "runner": "semantic_geometry_mode_runner",
            "mode_stage": "fp32_embedding_export",
            "eval_csv": str(_split_csv(cfg, "eval")),
            "embedding_pooling": "mask_aware_mean",
            "compute_logits_if_available": _compute_logits_if_available(cfg),
            **identity_fields,
        },
    )
    cache_path = save_embedding_cache(cache, save_path, expected_dim=expected_dim) if save_path is not None else None
    return ModeRunResult(
        embedding_cache=cache,
        qparams=None,
        cache_path=cache_path,
        runtime_summary={
            "quant_mode": FP32_QUANT_MODE,
            "granularity": FP32_GRANULARITY,
            "device": str(device),
            "num_samples": len(cache["sample_id"]),
            "compute_logits_if_available": _compute_logits_if_available(cfg),
            "logits_saved": "logits" in cache,
        },
    )


def _run_quant_with_default_runtime(
    cfg: Any,
    *,
    model: nn.Module,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    qparams: Mapping[str, Any],
    activation_granularity: str,
) -> tuple[Any, Any]:
    weight_qparams = qparams["weight"] if _mode_uses_weight(quant_mode) else None
    activation_qparams = qparams["activation"] if _mode_uses_activation(quant_mode) else None
    target_name = _target_name(layer_idx, module_name)
    activation_observer_config = activation_observer_config_for_granularity(cfg, activation_granularity)
    _require_qparams_for_quant_mode(
        quant_mode=quant_mode,
        weight_qparams=weight_qparams,
        activation_qparams=activation_qparams,
        target_name=target_name,
    )
    quant_module = replace_ffn_module_with_quant(
        model,
        int(layer_idx),
        module_name,
        _quant_cfg(cfg),
        quant_mode=quant_mode,
        weight_qparams=weight_qparams,
        activation_qparams=activation_qparams,
        activation_observer_config=activation_observer_config,
        require_calibrated_qparams=True,
        enable_fake_quant=False,
    )
    controller = make_single_ffn_quant_controller(
        model,
        quant_module,
        layer_idx=int(layer_idx),
        module_name=module_name,
        quant_config=_quant_cfg(cfg),
        quant_mode=quant_mode,
    )
    controller.qparams_frozen = True
    controller.prepare_quant_forward()
    _assert_quant_module_has_fixed_qparams(quant_module, quant_mode, target_name)
    return quant_module, controller


def _target_qparams_keys(layer_idx: int, module_name: str) -> tuple[Any, ...]:
    return (
        (int(layer_idx), str(module_name)),
        f"{int(layer_idx)}:{module_name}",
        f"layer_{int(layer_idx):02d}_{module_name}",
        _target_name(int(layer_idx), str(module_name)),
    )


def _qparams_for_target(
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    *,
    layer_idx: int,
    module_name: str,
) -> dict[str, Any]:
    for key in _target_qparams_keys(layer_idx, module_name):
        value = qparams_by_target.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    known = ", ".join(str(key) for key in list(qparams_by_target.keys())[:8])
    raise KeyError(f"missing qparams for {_target_name(layer_idx, module_name)}; known keys: {known}")


def _normalize_controller_list(controller: Any) -> list[Any]:
    if controller is None:
        return []
    if isinstance(controller, (list, tuple)):
        return [item for item in controller if item is not None]
    return [controller]


def _remove_controller_hooks(controllers: Sequence[Any]) -> None:
    for controller in controllers:
        if hasattr(controller, "remove_padding_mask_hooks"):
            controller.remove_padding_mask_hooks()


def _run_full_model_quant_with_default_runtime(
    cfg: Any,
    *,
    model: nn.Module,
    target_modules: Sequence[tuple[int, str]],
    quant_mode: str,
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    activation_granularity: str,
) -> tuple[list[Any], list[Any]]:
    quant_modules: list[Any] = []
    target_qparams: list[tuple[int, str, dict[str, Any]]] = []
    seen_targets: set[tuple[int, str]] = set()
    for layer_idx, module_name in target_modules:
        layer_idx = int(layer_idx)
        module_name = str(module_name)
        target_key = (layer_idx, module_name)
        if target_key in seen_targets:
            raise ValueError(f"duplicate full_model target: {_target_name(layer_idx, module_name)}")
        seen_targets.add(target_key)
        fixed_qparams = _qparams_for_target(
            qparams_by_target,
            layer_idx=layer_idx,
            module_name=module_name,
        )
        _validate_qparams_for_target(
            fixed_qparams,
            layer_idx=layer_idx,
            module_name=module_name,
            activation_granularity=activation_granularity,
            require_weight=_mode_uses_weight(quant_mode),
            require_activation=_mode_uses_activation(quant_mode),
        )
        validate_calibration_provenance(
            fixed_qparams,
            cfg,
            activation_granularity=activation_granularity,
            require_activation=_mode_uses_activation(quant_mode),
            context=f"{_target_name(layer_idx, module_name)} {normalize_mode_label(quant_mode)} full_model",
        )
        target = get_ffn_module(model, layer_idx, module_name)
        if isinstance(target, QuantLinear):
            raise RuntimeError(f"{_target_name(layer_idx, module_name)} is already QuantLinear")
        if not isinstance(target, nn.Linear):
            raise TypeError(f"{_target_name(layer_idx, module_name)} must be nn.Linear, got {type(target)!r}")
        target_qparams.append((layer_idx, module_name, fixed_qparams))

    activation_observer_config = activation_observer_config_for_granularity(cfg, activation_granularity)
    controller = QuantizationController(config=_quant_cfg(cfg), mode=normalize_quant_mode(quant_mode, None))
    hooked_layers: set[int] = set()
    for layer_idx, module_name, fixed_qparams in target_qparams:
        weight_qparams = fixed_qparams["weight"] if _mode_uses_weight(quant_mode) else None
        activation_qparams = fixed_qparams["activation"] if _mode_uses_activation(quant_mode) else None
        target_name = _target_name(layer_idx, module_name)
        _require_qparams_for_quant_mode(
            quant_mode=quant_mode,
            weight_qparams=weight_qparams,
            activation_qparams=activation_qparams,
            target_name=target_name,
        )
        layer = _ffn_layer(model, layer_idx)
        wrapped = QuantLinear(
            getattr(layer, module_name),
            quant_config=_quant_cfg(cfg),
            activation_observer_config=activation_observer_config,
            target_name=target_name,
        )
        setattr(layer, module_name, wrapped)
        apply_fixed_qparams_to_quant_module(
            wrapped,
            mode=quant_mode,
            weight_qparams=weight_qparams,
            activation_qparams=activation_qparams,
            require_qparams=True,
        )
        controller.register_module(target_name, wrapped)
        if layer_idx not in hooked_layers:
            register_layer_padding_mask_hooks(layer, controller=controller)
            hooked_layers.add(layer_idx)
        controller.qparams_frozen = True
        _assert_quant_module_has_fixed_qparams(wrapped, quant_mode, target_name)
        quant_modules.append(wrapped)

    controller.prepare_quant_forward()
    return quant_modules, [controller]


def run_full_model_quant_embedding_mode(
    cfg: Any,
    *,
    target_modules: Sequence[tuple[int, str]],
    quant_mode: str,
    activation_granularity: str = "per_tensor",
    qparams_by_target: Mapping[Any, Mapping[str, Any]] | None = None,
    calibration_loader: DataLoader | None = None,
    eval_loader: DataLoader | None = None,
    sample_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    save_path: str | Path | None = None,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    custom_quant_runner: Callable[..., tuple[Any, Any]] | None = None,
) -> ModeRunResult:
    """Run a full-model FFN quantization scope and export final embedding cache."""

    normalized_mode = normalize_quant_mode(quant_mode, None)
    if normalized_mode == "none":
        return run_fp32_embedding_mode(
            cfg,
            eval_loader=eval_loader,
            sample_metadata=sample_metadata,
            save_path=save_path,
            expected_dim=expected_dim,
        )
    activation_granularity = normalize_activation_granularity(activation_granularity)
    if activation_granularity != "per_tensor" and custom_quant_runner is None:
        _assert_current_runtime_supports_granularity(activation_granularity)
    if not target_modules:
        raise ValueError("full_model scope requires at least one target module")
    target_modules = [(int(layer_idx), str(module_name)) for layer_idx, module_name in target_modules]
    _assert_full_model_default_runtime_supported(target_modules, custom_quant_runner)
    if qparams_by_target is None:
        raise ValueError("full_model scope requires qparams_by_target; calibrate per target before running")
    effective_target_count = len(target_modules)
    default_single_target_equivalent = custom_quant_runner is None and effective_target_count == 1
    if custom_quant_runner is not None:
        runtime_name = "custom_quant_runner"
    elif effective_target_count == 1:
        runtime_name = "default_single_target_injector"
    else:
        runtime_name = "default_multi_ffn_injector"
    effective_scope = "single_target_equivalent" if default_single_target_equivalent else "full_model"

    loader = eval_loader if eval_loader is not None else build_eval_loader(cfg)
    metadata = sample_metadata if sample_metadata is not None else load_eval_metadata(cfg)
    device = _device(cfg)
    model = restore_clean_model(cfg).to(device)
    controllers: list[Any] = []
    quant_modules: list[Any] = []
    qparams_paths: dict[str, Any] = {}
    for layer_idx, module_name in target_modules:
        fixed_qparams = _qparams_for_target(qparams_by_target, layer_idx=int(layer_idx), module_name=str(module_name))
        qparams_paths[_target_name(layer_idx, module_name)] = fixed_qparams.get("qparams_path")

    if custom_quant_runner is not None:
        quant_module, controller = custom_quant_runner(
            cfg=cfg,
            model=model,
            scope="full_model",
            target_modules=list(target_modules),
            quant_mode=normalize_mode_label(quant_mode),
            activation_granularity=activation_granularity,
            qparams_by_target=qparams_by_target,
        )
        quant_modules = list(quant_module) if isinstance(quant_module, (list, tuple)) else [quant_module]
        controllers = _normalize_controller_list(controller)
    else:
        quant_modules, controllers = _run_full_model_quant_with_default_runtime(
            cfg,
            model=model,
            target_modules=list(target_modules),
            quant_mode=normalize_mode_label(quant_mode),
            qparams_by_target=qparams_by_target,
            activation_granularity=activation_granularity,
        )

    try:
        payload = collect_final_embeddings(cfg, model=model, eval_loader=loader, sample_metadata=metadata)
        identity_fields = _embedding_identity_manifest_fields(
            cfg,
            scope="full_model",
            quant_mode=normalize_mode_label(quant_mode),
            granularity=activation_granularity,
            target_modules=target_modules,
            sample_ids=payload["sample_id"],
            qparams_identity_info=_qparams_identity_for_target_map(qparams_by_target, target_modules),
            embedding_dim=expected_dim,
        )
        cache = _build_cache_from_embedding_payload(
            payload,
            quant_mode=normalize_mode_label(quant_mode),
            granularity=activation_granularity,
            expected_dim=expected_dim,
            extra_manifest={
                "runner": "semantic_geometry_mode_runner",
                "mode_stage": "full_model_quant_embedding_export",
                "scope": "full_model",
                "effective_scope": effective_scope,
                "effective_target_count": effective_target_count,
                "runtime": runtime_name,
                "eval_csv": str(_split_csv(cfg, "eval")),
                "calibration_csv": str(_split_csv(cfg, "calibration")),
                "embedding_pooling": "mask_aware_mean",
                "compute_logits_if_available": _compute_logits_if_available(cfg),
                "target_modules": [[int(layer_idx), str(module_name)] for layer_idx, module_name in target_modules],
                "activation_granularity": activation_granularity,
                "activation_percentile": _activation_percentile_from_quant_cfg(cfg),
                "uses_weight_quant": bool(_mode_uses_weight(quant_mode)),
                "uses_activation_quant": bool(_mode_uses_activation(quant_mode)),
                "qparams_paths": qparams_paths,
                "custom_quant_runner": custom_quant_runner is not None,
                "default_runtime_activation_granularity_supported": activation_granularity in DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES,
                **identity_fields,
            },
        )
        cache_path = save_embedding_cache(cache, save_path, expected_dim=expected_dim) if save_path is not None else None
        return ModeRunResult(
            embedding_cache=cache,
            qparams={str(key): dict(value) for key, value in qparams_by_target.items()},
            cache_path=cache_path,
            runtime_summary={
                "scope": "full_model",
                "effective_scope": effective_scope,
                "effective_target_count": effective_target_count,
                "runtime": runtime_name,
                "quant_mode": normalize_mode_label(quant_mode),
                "granularity": activation_granularity,
                "device": str(device),
                "num_samples": len(cache["sample_id"]),
                "num_targets": len(target_modules),
                "target_modules": [[int(layer_idx), str(module_name)] for layer_idx, module_name in target_modules],
                "fake_quant_enabled": all(bool(getattr(module, "fake_quant_enabled", True)) for module in quant_modules),
                "weight_quant_enabled": bool(_mode_uses_weight(quant_mode)),
                "activation_quant_enabled": bool(_mode_uses_activation(quant_mode)),
                "compute_logits_if_available": _compute_logits_if_available(cfg),
                "logits_saved": "logits" in cache,
            },
        )
    finally:
        _remove_controller_hooks(controllers)

def run_quant_embedding_mode(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    activation_granularity: str = "per_tensor",
    qparams: Mapping[str, Any] | None = None,
    qparams_path: str | Path | None = None,
    calibration_loader: DataLoader | None = None,
    eval_loader: DataLoader | None = None,
    sample_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    save_path: str | Path | None = None,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    force_recalibrate: bool = False,
    custom_quant_runner: Callable[..., tuple[Any, Any]] | None = None,
) -> ModeRunResult:
    """运行一种量化模式，输出最终 embedding cache。

    A4-only 只启用 activation fake quant；W4-only 只启用 weight fake quant；W4A4 同时启用二者。
    weight 始终保留 FP32 master 参数，forward 内临时 fake quant，严禁 weight.data.copy_。
    """

    normalized_mode = normalize_quant_mode(quant_mode, None)
    if normalized_mode == "none":
        return run_fp32_embedding_mode(
            cfg,
            eval_loader=eval_loader,
            sample_metadata=sample_metadata,
            save_path=save_path,
            expected_dim=expected_dim,
        )

    activation_granularity = normalize_activation_granularity(activation_granularity)
    if activation_granularity != "per_tensor" and custom_quant_runner is None:
        _assert_current_runtime_supports_granularity(activation_granularity)

    loader = eval_loader if eval_loader is not None else build_eval_loader(cfg)
    metadata = sample_metadata if sample_metadata is not None else load_eval_metadata(cfg)
    calib_loader = calibration_loader if calibration_loader is not None else build_calibration_loader(cfg)
    fixed_qparams = dict(qparams) if qparams is not None else load_or_calibrate_qparams(
        cfg,
        layer_idx=int(layer_idx),
        module_name=module_name,
        calibration_loader=calib_loader,
        activation_granularity=activation_granularity,
        force_recalibrate=force_recalibrate,
        qparams_path=qparams_path,
    )
    _validate_qparams_for_target(
        fixed_qparams,
        layer_idx=int(layer_idx),
        module_name=module_name,
        activation_granularity=activation_granularity,
        require_weight=_mode_uses_weight(quant_mode),
        require_activation=_mode_uses_activation(quant_mode),
    )
    validate_calibration_provenance(
        fixed_qparams,
        cfg,
        activation_granularity=activation_granularity,
        require_activation=_mode_uses_activation(quant_mode),
        context=f"{_target_name(layer_idx, module_name)} {normalize_mode_label(quant_mode)}",
    )

    device = _device(cfg)
    model = restore_clean_model(cfg).to(device)
    controller = None
    if custom_quant_runner is not None:
        quant_module, controller = custom_quant_runner(
            cfg=cfg,
            model=model,
            layer_idx=int(layer_idx),
            module_name=module_name,
            quant_mode=normalize_mode_label(quant_mode),
            activation_granularity=activation_granularity,
            qparams=fixed_qparams,
        )
    else:
        quant_module, controller = _run_quant_with_default_runtime(
            cfg,
            model=model,
            layer_idx=int(layer_idx),
            module_name=module_name,
            quant_mode=normalize_mode_label(quant_mode),
            qparams=fixed_qparams,
            activation_granularity=activation_granularity,
        )

    try:
        payload = collect_final_embeddings(cfg, model=model, eval_loader=loader, sample_metadata=metadata)
        identity_fields = _embedding_identity_manifest_fields(
            cfg,
            scope="single_target",
            quant_mode=normalize_mode_label(quant_mode),
            granularity=activation_granularity,
            target_modules=[(int(layer_idx), str(module_name))],
            sample_ids=payload["sample_id"],
            qparams_identity_info=qparams_identity_manifest(fixed_qparams, source_hint=_target_name(layer_idx, module_name)),
            embedding_dim=expected_dim,
        )
        cache = _build_cache_from_embedding_payload(
            payload,
            quant_mode=normalize_mode_label(quant_mode),
            granularity=activation_granularity,
            expected_dim=expected_dim,
            extra_manifest={
                "runner": "semantic_geometry_mode_runner",
                "mode_stage": "quant_embedding_export",
                "eval_csv": str(_split_csv(cfg, "eval")),
                "calibration_csv": str(_split_csv(cfg, "calibration")),
                "embedding_pooling": "mask_aware_mean",
                "compute_logits_if_available": _compute_logits_if_available(cfg),
                "layer_idx": int(layer_idx),
                "module_name": str(module_name),
                "target_name": _target_name(layer_idx, module_name),
                "activation_granularity": activation_granularity,
                "activation_percentile": _activation_percentile_from_quant_cfg(cfg),
                "uses_weight_quant": bool(_mode_uses_weight(quant_mode)),
                "uses_activation_quant": bool(_mode_uses_activation(quant_mode)),
                "qparams_path": fixed_qparams.get("qparams_path"),
                "qparams_calibration_csv": _qparam_meta_value(fixed_qparams, "calibration_csv", "csv", "calibration_path"),
                "qparams_activation_percentile": _qparam_meta_value(fixed_qparams, "activation_percentile", "percentile"),
                "qparams_activation_percentile_mode": _qparam_meta_value(fixed_qparams, "activation_percentile_mode", "percentile_mode"),
                "qparams_activation_granularity": _qparam_meta_value(fixed_qparams, "activation_granularity", "granularity"),
                "custom_quant_runner": custom_quant_runner is not None,
                "runtime": "custom_quant_runner" if custom_quant_runner is not None else "default_single_target_injector",
                "default_runtime_activation_granularity_supported": activation_granularity in DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES,
                **identity_fields,
            },
        )
        cache_path = save_embedding_cache(cache, save_path, expected_dim=expected_dim) if save_path is not None else None
        return ModeRunResult(
            embedding_cache=cache,
            qparams=fixed_qparams,
            cache_path=cache_path,
            runtime_summary={
                "quant_mode": normalize_mode_label(quant_mode),
                "granularity": activation_granularity,
                "device": str(device),
                "num_samples": len(cache["sample_id"]),
                "layer_idx": int(layer_idx),
                "module_name": str(module_name),
                "target_name": _target_name(layer_idx, module_name),
                "fake_quant_enabled": bool(getattr(quant_module, "fake_quant_enabled", True)),
                "weight_quant_enabled": bool(_mode_uses_weight(quant_mode)),
                "activation_quant_enabled": bool(_mode_uses_activation(quant_mode)),
                "compute_logits_if_available": _compute_logits_if_available(cfg),
                "logits_saved": "logits" in cache,
            },
        )
    finally:
        if controller is not None and hasattr(controller, "remove_padding_mask_hooks"):
            controller.remove_padding_mask_hooks()


def default_embedding_cache_path(
    cfg: Any,
    *,
    quant_mode: str,
    activation_granularity: str = "per_tensor",
    layer_idx: int | None = None,
    module_name: str | None = None,
) -> Path:
    label = normalize_mode_label(quant_mode).lower().replace("-", "_")
    granularity = normalize_activation_granularity(activation_granularity) if normalize_quant_mode(quant_mode, None) != "none" else FP32_GRANULARITY
    if normalize_quant_mode(quant_mode, None) == "none":
        filename = "fp32_embedding.pt"
    else:
        if layer_idx is None or module_name is None:
            raise ValueError("量化 embedding cache 路径需要 layer_idx 和 module_name")
        filename = f"layer_{int(layer_idx):02d}_{module_name}_{label}_{granularity}_embedding.pt"
    return _artifact_root(cfg) / "embedding_cache" / filename


def run_embedding_mode(
    cfg: Any,
    *,
    quant_mode: str,
    layer_idx: int | None = None,
    module_name: str | None = None,
    activation_granularity: str = "per_tensor",
    qparams: Mapping[str, Any] | None = None,
    qparams_path: str | Path | None = None,
    qparams_by_target: Mapping[Any, Mapping[str, Any]] | None = None,
    target_modules: Sequence[tuple[int, str]] | None = None,
    scope: str = "single_target",
    calibration_loader: DataLoader | None = None,
    eval_loader: DataLoader | None = None,
    sample_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    save: bool = False,
    save_path: str | Path | None = None,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    force_recalibrate: bool = False,
    custom_quant_runner: Callable[..., tuple[Any, Any]] | None = None,
) -> ModeRunResult:
    """Unified entry for FP32, single-target quantization, or full-model quantization."""

    normalized_scope = str(scope).strip().lower().replace("-", "_")
    if normalized_scope not in {"single_target", "full_model"}:
        raise ValueError(f"unsupported semantic geometry scope: {scope!r}")
    normalized_mode = normalize_quant_mode(quant_mode, None)
    if normalized_mode == "none":
        resolved_save_path = save_path if save_path is not None else default_embedding_cache_path(
            cfg,
            quant_mode=FP32_QUANT_MODE,
        ) if save else None
        return run_fp32_embedding_mode(
            cfg,
            eval_loader=eval_loader,
            sample_metadata=sample_metadata,
            save_path=resolved_save_path,
            expected_dim=expected_dim,
        )

    if normalized_scope == "full_model":
        resolved_save_path = save_path
        if resolved_save_path is None and save:
            label = normalize_mode_label(quant_mode).lower().replace("-", "_")
            granularity = normalize_activation_granularity(activation_granularity)
            resolved_save_path = _artifact_root(cfg) / "embedding_cache" / "full_model" / f"full_model_{label}_{granularity}_embedding.pt"
        return run_full_model_quant_embedding_mode(
            cfg,
            target_modules=[] if target_modules is None else [(int(item_layer), str(item_module)) for item_layer, item_module in target_modules],
            quant_mode=normalize_mode_label(quant_mode),
            activation_granularity=activation_granularity,
            qparams_by_target=qparams_by_target,
            calibration_loader=calibration_loader,
            eval_loader=eval_loader,
            sample_metadata=sample_metadata,
            save_path=resolved_save_path,
            expected_dim=expected_dim,
            custom_quant_runner=custom_quant_runner,
        )

    if layer_idx is None or module_name is None:
        raise ValueError("A4-only / W4-only / W4A4 run requires layer_idx and module_name")
    resolved_save_path = save_path
    if resolved_save_path is None and save:
        resolved_save_path = default_embedding_cache_path(
            cfg,
            quant_mode=normalize_mode_label(quant_mode),
            activation_granularity=activation_granularity,
            layer_idx=int(layer_idx),
            module_name=module_name,
        )
    return run_quant_embedding_mode(
        cfg,
        layer_idx=int(layer_idx),
        module_name=module_name,
        quant_mode=normalize_mode_label(quant_mode),
        activation_granularity=activation_granularity,
        qparams=qparams,
        qparams_path=qparams_path,
        calibration_loader=calibration_loader,
        eval_loader=eval_loader,
        sample_metadata=sample_metadata,
        save_path=resolved_save_path,
        expected_dim=expected_dim,
        force_recalibrate=force_recalibrate,
        custom_quant_runner=custom_quant_runner,
    )

__all__ = [
    "DEFAULT_RUNNER_CFG_NAME",
    "DEFAULT_RUNTIME_ACTIVATION_GRANULARITIES",
    "ModeRunResult",
    "ForwardResult",
    "QUANT_MODE_LABELS",
    "SUPPORTED_ACTIVATION_GRANULARITIES",
    "TARGET_MODULE_NAMES",
    "attach_calibration_provenance",
    "activation_observer_config_for_granularity",
    "build_audio_loader",
    "build_calibration_loader",
    "build_eval_loader",
    "calibrate_target_qparams",
    "collect_final_embeddings",
    "_compute_logits_if_available",
    "default_embedding_cache_path",
    "load_eval_metadata",
    "load_or_calibrate_qparams",
    "load_qparams",
    "load_sample_metadata_from_csv",
    "normalize_activation_granularity",
    "normalize_mode_label",
    "qparams_cache_path",
    "restore_clean_model",
    "run_embedding_mode",
    "run_fp32_embedding_mode",
    "run_quant_embedding_mode",
    "run_full_model_quant_embedding_mode",
    "save_qparams",
    "target_modules_from_config",
    "validate_calibration_provenance",
]
