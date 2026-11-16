from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from src.analysis.build_fp32_oer_oracle import (
        ORACLE_STORAGE_FORMAT as FP32_OER_STORAGE_FORMAT,
        iter_fp32_oer_oracle_shards,
        load_fp32_oer_oracle,
    )
    from src.export.export_fp32_core_cache import (
        CORE_STORAGE_FORMAT as FP32_CORE_STORAGE_FORMAT,
        forward_core_outputs,
        load_fp32_core_cache,
    )
    from src.metrics.ffn_quant_mode_metrics import (
        QUANT_MODE_COMPARISON_COLUMNS,
        QUANT_MODE_GROUP_METRIC_SPECS,
        QUANT_MODE_METRIC_COLUMNS,
        QUANT_MODE_TARGET_DETAIL_COLUMNS,
        QUANT_MODE_TARGET_DETAIL_SPECS,
        compute_quant_mode_metric_tables_from_sample_metrics,
    )
    from src.metrics.common_metrics import scalar_oer_energy_metrics, scalar_oer_topk_indices
    from src.plots.ffn_quant_mode_plotter import plot_all_quant_mode_plots
    from src.quant.calibration import freeze_weight_qparams_from_tensor
    from src.quant.ffn_quant_injector import (
        get_ffn_module,
        make_single_ffn_quant_controller,
        normalize_quant_mode,
        quant_mode_uses_activation,
        quant_mode_uses_weight,
        replace_ffn_module_with_quant,
        restore_clean_model_from_checkpoint,
    )
    from src.utils.ffn_fp32_reference_cache import OER_FEATURE_LAYOUT
    from src.utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
except ImportError:  # pragma: no cover - script execution fallback.
    from ..data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from ..analysis.build_fp32_oer_oracle import (
        ORACLE_STORAGE_FORMAT as FP32_OER_STORAGE_FORMAT,
        iter_fp32_oer_oracle_shards,
        load_fp32_oer_oracle,
    )
    from ..export.export_fp32_core_cache import (
        CORE_STORAGE_FORMAT as FP32_CORE_STORAGE_FORMAT,
        forward_core_outputs,
        load_fp32_core_cache,
    )
    from ..metrics.ffn_quant_mode_metrics import (
        QUANT_MODE_COMPARISON_COLUMNS,
        QUANT_MODE_GROUP_METRIC_SPECS,
        QUANT_MODE_METRIC_COLUMNS,
        QUANT_MODE_TARGET_DETAIL_COLUMNS,
        QUANT_MODE_TARGET_DETAIL_SPECS,
        compute_quant_mode_metric_tables_from_sample_metrics,
    )
    from ..metrics.common_metrics import scalar_oer_energy_metrics, scalar_oer_topk_indices
    from ..plots.ffn_quant_mode_plotter import plot_all_quant_mode_plots
    from ..quant.calibration import freeze_weight_qparams_from_tensor
    from ..quant.ffn_quant_injector import (
        get_ffn_module,
        make_single_ffn_quant_controller,
        normalize_quant_mode,
        quant_mode_uses_activation,
        quant_mode_uses_weight,
        replace_ffn_module_with_quant,
        restore_clean_model_from_checkpoint,
    )
    from ..utils.ffn_fp32_reference_cache import OER_FEATURE_LAYOUT
    from ..utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields


CSV_FILENAME = "ffn_quant_mode_ablation.csv"
COMPARISON_SUMMARY_CSV_FILENAME = "ffn_quant_mode_comparison_summary.csv"
TARGET_DETAIL_CSV_FILENAME = "ffn_quant_mode_target_detail.csv"
QPARAMS_SUMMARY_CSV_FILENAME = "ffn_quant_mode_qparams_summary.csv"
MANIFEST_FILENAME = "ffn_quant_mode_ablation_manifest.json"
QPARAMS_DIRNAME = "qparams"
ARTIFACT_DIRNAME = "quant_mode_artifacts"
FP32_REFERENCE_CORE_FILENAME = "core_eval.pt"
FP32_REFERENCE_OER_ORACLE_FILENAME = "oer_oracle_eval.pt"
FP32_CORE_CACHE_TYPE = "fp32_core_cache"
FP32_CORE_STAGE = "quant_injection_pre_fp32_reference"
FP32_OER_CACHE_TYPE = "fp32_oer_oracle"
FP32_OER_STAGE = "quant_injection_pre_fp32_oer_oracle"
FP32_OER_ORACLE_SOURCE = "fp32_real_activation"
PROTOCOL_VERSION = "ffn_quant_mode_ablation.v5"
MODE_SHARDS_DIRNAME = "mode_shards"
DONE_DIRNAME = "done"
TARGET_MODULE_NAMES = ("fc1", "fc2")
QUANT_MODES = ("W4-only", "A4-only", "W4A4")
EXTRA_CSV_COLUMNS = [
    "target_name",
    "qparams_path",
    "artifact_path",
    "embedding_path",
    "output_num_elements",
    "output_mean",
    "output_std",
    "output_max_abs",
    "activation_fp32_shape",
    "activation_quant_shape",
    "projection_fp32_shape",
    "projection_quant_shape",
    "activation_layout_info",
    "projection_layout_info",
    "input_oer_activation_kind",
    "input_oer_oracle_source",
    "projection_oer_activation_kind",
    "projection_oer_oracle_source",
    "activation_clipping_rate_runtime",
    "activation_clipping_rate_by_sample_mean",
    "activation_clipping_rate_by_sample_std",
    "activation_clipping_rate_by_sample_max",
    "weight_qparams_enabled",
    "activation_qparams_enabled",
]
CSV_COLUMNS = list(dict.fromkeys([*QUANT_MODE_METRIC_COLUMNS, *EXTRA_CSV_COLUMNS]))
COMPARISON_CSV_COLUMNS = list(dict.fromkeys([*QUANT_MODE_COMPARISON_COLUMNS, *EXTRA_CSV_COLUMNS]))
TARGET_DETAIL_CSV_COLUMNS = list(dict.fromkeys([*QUANT_MODE_TARGET_DETAIL_COLUMNS, *EXTRA_CSV_COLUMNS]))
QPARAMS_SUMMARY_COLUMNS = [
    "row_type",
    "group_protocol_version",
    "layer_idx",
    "module_name",
    "target_name",
    "quant_mode",
    "granularity",
    "qparams_path",
    "weight_qparams_enabled",
    "activation_qparams_enabled",
    "weight_scale_shape",
    "weight_scale_min",
    "weight_scale_max",
    "weight_scale_mean",
    "weight_scale_std",
    "weight_scale_finite_check",
    "weight_zero_point_shape",
    "weight_zero_point_min",
    "weight_zero_point_max",
    "weight_zero_point_mean",
    "weight_zero_point_std",
    "weight_zero_point_finite_check",
    "weight_qmin",
    "weight_qmax",
    "weight_symmetric",
    "weight_granularity",
    "weight_channel_axis",
    "weight_clip_rate_proxy",
    "activation_scale_shape",
    "activation_scale_min",
    "activation_scale_max",
    "activation_scale_mean",
    "activation_scale_std",
    "activation_scale_finite_check",
    "activation_zero_point_shape",
    "activation_zero_point_min",
    "activation_zero_point_max",
    "activation_zero_point_mean",
    "activation_zero_point_std",
    "activation_zero_point_finite_check",
    "activation_qmin",
    "activation_qmax",
    "activation_symmetric",
    "activation_granularity",
    "activation_channel_axis",
    "activation_min",
    "activation_max",
    "activation_abs_max",
    "activation_histogram_built",
    "activation_histogram_bins",
    "activation_histogram_num_elements",
    "activation_percentiles",
    "activation_abs_percentiles",
    "observer_num_batches",
    "observer_num_samples",
    "observer_range_num_batches",
    "observer_range_num_samples",
    "observer_histogram_num_batches",
    "observer_histogram_num_samples",
    "calibration_sample_count",
    "controller_observer_backend",
    "controller_observer_num_updates",
    "controller_observer_num_values",
    "activation_clipping_rate_runtime",
    "activation_clipping_rate_by_sample_mean",
    "activation_clipping_rate_by_sample_std",
    "activation_clipping_rate_by_sample_max",
    "activation_clipping_rate_by_sample_count",
    "activation_fp32_shape",
    "activation_quant_shape",
    "projection_fp32_shape",
    "projection_quant_shape",
    "activation_layout_info",
    "projection_layout_info",
    "observer_summary",
    "weight_observer_summary",
    "controller_summary",
]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def _ablation_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FFN_QUANT_MODE_ABLATION", {})
    return value if isinstance(value, Mapping) else {}


def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}


def _quant_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "QUANT", {})
    if not isinstance(value, Mapping):
        raise TypeError("cfg.QUANT must be a mapping for FFN INT4 settings")
    return value


def _metrics_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "METRICS", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    ablation = _ablation_cfg(cfg)
    if key in ablation:
        return ablation[key]
    return _cfg_get(cfg, key, default)


def _oer_top_ratio(cfg: Any) -> float:
    return float(
        _setting_get(
            cfg,
            "SCALAR_OER_TOP_RATIO",
            _metrics_cfg(cfg).get("SCALAR_OER_TOP_RATIO", _cfg_get(cfg, "OER_SCALAR_TOP_RATIO", 0.001)),
        )
    )


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
    raise AttributeError("missing dataset name; configure ACTIVE_DATASET or CURRENT_DATASET['ROOT']")


def _result_dir(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "quant_mode_ablation"


def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", _cfg_get(cfg, "DEVICE", None))
    if configured is None:
        raise AttributeError("ffn_quant_mode_ablation requires config.py DEVICE")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py DEVICE={configured!r}, but CUDA is not available")
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
    raise AttributeError(f"missing {key}; configure it in CURRENT_DATASET or FFN_QUANT_MODE_ABLATION")


def _batch_size(cfg: Any, name: str) -> int:
    key = f"{name.upper()}_BATCH_SIZE"
    return int(_setting_get(cfg, key, _setting_get(cfg, "BATCH_SIZE", 4)))


def _num_workers(cfg: Any) -> int:
    return int(_setting_get(cfg, "NUM_WORKERS", 0))


def _pin_memory(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "PIN_MEMORY", _device(cfg).type == "cuda"))


def _build_audio_loader(cfg: Any, *, name: str, mode: str) -> DataLoader:
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


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
    return moved


def _extract_tensor_and_mask(output: Any) -> tuple[Tensor, Tensor | None]:
    if isinstance(output, Tensor):
        return output, None
    if isinstance(output, Mapping):
        tensor = None
        for key in ("embedding", "embeddings", "features", "feature", "x", "logits", "lprobs"):
            value = output.get(key)
            if isinstance(value, Tensor):
                tensor = value
                break
        if tensor is None:
            raise TypeError("embedding output dict does not contain a tensor value")
        mask = output.get("padding_mask")
        return tensor, mask if isinstance(mask, Tensor) else None
    if isinstance(output, (tuple, list)) and output:
        tensor = output[0]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"embedding output value must be a Tensor, got {type(tensor)!r}")
        mask = output[1] if len(output) > 1 and isinstance(output[1], Tensor) else None
        return tensor, mask
    raise TypeError(f"cannot extract embedding tensor from output type {type(output)!r}")


def _forward_features(model: nn.Module, batch: Mapping[str, Any], cfg: Any) -> tuple[Tensor, Tensor | None]:
    for name in ("ffn_quant_mode_forward", "ffn_fp32_reference_forward", "ffn_heatmap_forward"):
        forward_fn = _cfg_get(cfg, name, None)
        if callable(forward_fn):
            return _extract_tensor_and_mask(forward_fn(model, batch, cfg))
    waveform = batch["waveform"]
    padding_mask = batch.get("padding_mask")
    if hasattr(model, "extract_features"):
        return _extract_tensor_and_mask(model.extract_features(waveform, padding_mask=padding_mask))
    try:
        return _extract_tensor_and_mask(model(waveform, padding_mask=padding_mask))
    except TypeError:
        return _extract_tensor_and_mask(model(waveform))


def _pool_embedding(features: Tensor, padding_mask: Tensor | None, cfg: Any) -> Tensor:
    if features.ndim == 2:
        return features.detach()
    if features.ndim == 3:
        if padding_mask is not None and tuple(padding_mask.shape) == tuple(features.shape[:2]):
            mask = padding_mask.to(device=features.device, dtype=torch.bool)
            valid = (~mask).unsqueeze(-1)
            summed = features.masked_fill(~valid, 0.0).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1).to(dtype=features.dtype)
            return (summed / denom).detach()
        if bool(_setting_get(cfg, "REQUIRE_MASK_AWARE_POOLING", False)):
            raise RuntimeError("mask-aware mean pooling requires padding_mask and feature tensor with matching token dimension")
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

def _restore_clean_model(cfg: Any) -> nn.Module:
    for name in ("restore_clean_model", "restore_clean_model_from_checkpoint"):
        fn = _cfg_get(cfg, name, None)
        if callable(fn):
            model = fn(cfg)
            if not isinstance(model, nn.Module):
                raise TypeError(f"{name} must return nn.Module, got {type(model)!r}")
            return model
    return restore_clean_model_from_checkpoint(cfg)


def _all_target_modules(model: nn.Module) -> list[tuple[int, str]]:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise AttributeError("model.encoder.layers is required to locate BEATs Transformer FFN modules")
    targets: list[tuple[int, str]] = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        for module_name in TARGET_MODULE_NAMES:
            if not hasattr(layer, module_name):
                raise AttributeError(f"encoder.layers.{layer_idx}.{module_name} does not exist")
            targets.append((int(layer_idx), module_name))
    if not targets:
        raise ValueError("no FFN fc1/fc2 target modules were found")
    return targets


def _target_modules(cfg: Any, model: nn.Module) -> list[tuple[int, str]]:
    configured = _setting_get(cfg, "TARGET_MODULES", None)
    if configured is None:
        return _all_target_modules(model)
    result: list[tuple[int, str]] = []
    for item in configured:
        layer_idx, module_name = item
        module_name = str(module_name)
        if module_name not in TARGET_MODULE_NAMES:
            raise ValueError(f"module_name must be fc1/fc2, got {module_name!r}")
        get_ffn_module(model, int(layer_idx), module_name)
        result.append((int(layer_idx), module_name))
    if not result:
        raise ValueError("TARGET_MODULES cannot be empty")
    return result


def _target_name(layer_idx: int, module_name: str) -> str:
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _activation_to_btc(activation: Tensor, batch_size: int, target_name: str) -> Tensor:
    """Quant mode ablation helper."""

    return canonicalize_activation_btc(
        activation.detach().float().cpu(),
        batch_size=int(batch_size),
        context=target_name,
    ).tensor


def _activation_to_btc_with_layout(activation: Tensor, batch_size: int, target_name: str):
    return canonicalize_activation_btc(
        activation.detach().float().cpu(),
        batch_size=int(batch_size),
        context=target_name,
    )


def _assert_btc_feature_layout(x: Tensor, *, batch_size: int, target_name: str) -> Tensor:
    if x.ndim != 3:
        raise ValueError(f"{target_name}: expected canonical [B,T,C], got shape={tuple(x.shape)}")
    if int(x.shape[0]) != int(batch_size):
        raise ValueError(f"{target_name}: first dimension must equal batch_size={batch_size}, got shape={tuple(x.shape)}")
    return x.contiguous()


class _TensorStatsAccumulator:
    """Quant mode ablation helper."""

    def __init__(self) -> None:
        self.num_batches = 0
        self.num_elements = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0

    def update(self, x: Tensor) -> None:
        values = x.detach().float().cpu().reshape(-1)
        if int(values.numel()) == 0:
            return
        self.num_batches += 1
        self.num_elements += int(values.numel())
        self.sum += float(values.sum().item())
        self.sum_sq += float(values.square().sum().item())
        self.max_abs = max(self.max_abs, float(values.abs().max().item()))

    def summary(self) -> dict[str, float | int]:
        if self.num_elements <= 0:
            return {
                "num_batches": int(self.num_batches),
                "num_elements": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "max_abs": float("nan"),
            }
        mean = self.sum / float(self.num_elements)
        variance = max(self.sum_sq / float(self.num_elements) - mean * mean, 0.0)
        return {
            "num_batches": int(self.num_batches),
            "num_elements": int(self.num_elements),
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "max_abs": float(self.max_abs),
        }


def _run_forward_only(model: nn.Module, loader: DataLoader, cfg: Any) -> None:
    device = _device(cfg)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            with _autocast_context(cfg, device):
                _forward_features(model, moved, cfg)


def _run_two_pass_calibration(controller: Any, model: nn.Module, loader: DataLoader, cfg: Any) -> None:
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


def calibrate_target_module(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    calibration_loader: DataLoader,
) -> dict[str, Any]:
    """Quant mode ablation helper."""

    device = _device(cfg)
    model = _restore_clean_model(cfg).to(device)
    quant_module = replace_ffn_module_with_quant(
        model,
        int(layer_idx),
        module_name,
        _quant_cfg(cfg),
        quant_mode="W4A4",
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
            "granularity": "per_tensor",
            "observer_type": "activation",
            "method": "calibration_two_pass_percentile",
            "channel_axis": -1,
        }
        return {
            "layer_idx": int(layer_idx),
            "module_name": module_name,
            "target_name": _target_name(layer_idx, module_name),
            "weight": weight_qparams,
            "activation": activation_qparams,
            "observer_summary": controller.observer_summary(),
            "weight_observer_summary": controller.weight_observer_summary(),
            "controller_summary": controller.state_summary(),
        }
    finally:
        controller.remove_padding_mask_hooks()


def _save_qparams(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    qparams: Mapping[str, Any],
    *,
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> Path:
    path = _qparams_path(cfg, layer_idx, module_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    payload = dict(qparams)
    payload["protocol_version"] = protocol_version
    payload["experiment_signature"] = experiment_signature
    payload["experiment_signature_payload"] = _json_safe(signature_payload)
    payload["target"] = {"layer_idx": int(layer_idx), "module_name": str(module_name)}
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def _load_qparams(path: Path, protocol_version: str, experiment_signature: str) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        value = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive resume path.
        print(f"[resume] ignore invalid qparams file {path}: {exc}")
        return None
    if not isinstance(value, Mapping):
        return None
    payload = dict(value)
    if not _cache_protocol_matches(payload, protocol_version, experiment_signature):
        print(f"[resume] ignore stale qparams file {path}: protocol/signature mismatch")
        return None
    if "weight" not in payload or "activation" not in payload:
        print(f"[resume] ignore invalid qparams file {path}: missing weight/activation")
        return None
    return payload


def _mode_label(mode: str) -> str:
    normalized = normalize_quant_mode(mode, None)
    return {"w4": "W4-only", "a4": "A4-only", "w4a4": "W4A4", "none": "FP32"}[normalized]


def _mode_slug(mode: str) -> str:
    return _mode_label(mode).lower().replace("-", "_")


def _mode_uses_weight(mode: str) -> bool:
    return quant_mode_uses_weight(mode, None)


def _mode_uses_activation(mode: str) -> bool:
    return quant_mode_uses_activation(mode, None)

def _require_qparams_for_quant_mode(
    *,
    quant_mode: str,
    weight_qparams: Mapping[str, Any] | None,
    activation_qparams: Mapping[str, Any] | None,
    target_name: str,
) -> None:
    """Quant mode ablation helper."""

    if _mode_uses_weight(quant_mode):
        if not isinstance(weight_qparams, Mapping) or "scale" not in weight_qparams or "zero_point" not in weight_qparams:
            raise RuntimeError(f"{target_name} {quant_mode}: fixed weight qparams are required")
    if _mode_uses_activation(quant_mode):
        if not isinstance(activation_qparams, Mapping) or "scale" not in activation_qparams or "zero_point" not in activation_qparams:
            raise RuntimeError(f"{target_name} {quant_mode}: fixed activation qparams are required")


def _assert_quant_module_has_fixed_qparams(quant_module: Any, quant_mode: str, target_name: str) -> None:
    """Quant mode ablation helper."""

    if _mode_uses_weight(quant_mode):
        if not bool(getattr(quant_module, "fixed_weight_qparams_enabled", False)):
            raise RuntimeError(f"{target_name} {quant_mode}: fixed weight fake quant qparams are not enabled")
        weight_scale = getattr(quant_module, "weight_scale", None)
        if not isinstance(weight_scale, Tensor) or int(weight_scale.numel()) <= 0:
            raise RuntimeError(f"{target_name} {quant_mode}: fixed weight scale is empty")
    if _mode_uses_activation(quant_mode):
        if not bool(getattr(quant_module, "qparams_frozen", False)):
            raise RuntimeError(f"{target_name} {quant_mode}: activation qparams are not frozen")
        activation_scale = getattr(quant_module, "activation_scale", None)
        if not isinstance(activation_scale, Tensor) or not torch.isfinite(activation_scale.detach()).all().item():
            raise RuntimeError(f"{target_name} {quant_mode}: activation scale is not finite")



def _metric_eps(cfg: Any) -> float:
    return float(_setting_get(cfg, "METRIC_EPS", 1e-8))


def _nan_vector(length: int) -> Tensor:
    return torch.full((int(length),), float("nan"), dtype=torch.float32)


def _relative_l2_per_sample(fp32: Tensor, quant: Tensor, *, eps: float) -> Tensor:
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"per-sample relative L2 shape mismatch: {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    if fp32.ndim < 1:
        raise ValueError("per-sample relative L2 requires a batch dimension")
    batch_size = int(fp32.shape[0])
    if batch_size <= 0:
        return _nan_vector(0)
    fp32_flat = fp32.detach().float().reshape(batch_size, -1)
    quant_flat = quant.detach().float().reshape(batch_size, -1).to(device=fp32_flat.device)
    numerator = torch.linalg.vector_norm(quant_flat - fp32_flat, ord=2, dim=1)
    denominator = torch.linalg.vector_norm(fp32_flat, ord=2, dim=1).clamp_min(float(eps))
    return (numerator / denominator).detach().cpu().float()


def _oer_metrics_from_topk_indices(
    fp32: Tensor,
    quant: Tensor,
    topk_indices: Tensor | Sequence[Any],
    *,
    eps: float,
) -> dict[str, Tensor]:
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"OER shape mismatch: {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    batch_size = int(fp32.shape[0])
    fp32_flat = fp32.detach().float().reshape(batch_size, -1)
    quant_flat = quant.detach().float().reshape(batch_size, -1).to(device=fp32_flat.device)

    def compute_one(sample_idx: int, indices_value: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indices = torch.as_tensor(indices_value, dtype=torch.long, device=fp32_flat.device).reshape(-1)
        common_values = scalar_oer_energy_metrics(
            fp32_flat[int(sample_idx) : int(sample_idx) + 1],
            quant_flat[int(sample_idx) : int(sample_idx) + 1],
            topk_indices=indices,
            batch_dim=0,
            reduction="none",
            eps=eps,
        )
        fp32_values = fp32_flat[int(sample_idx)].index_select(0, indices)
        quant_values = quant_flat[int(sample_idx)].index_select(0, indices)
        fp32_oracle_energy = fp32_values.square().sum()
        quant_oracle_energy = quant_values.square().sum()
        quant_total_energy = quant_flat[int(sample_idx)].square().sum()
        retention = torch.as_tensor(common_values["oer_retention"], device=fp32_flat.device).reshape(())
        off_oracle_ratio = torch.as_tensor(common_values["off_oracle_energy_ratio"], device=fp32_flat.device).reshape(())
        return retention, off_oracle_ratio, fp32_oracle_energy, quant_oracle_energy, quant_total_energy

    if isinstance(topk_indices, Tensor):
        indices_tensor = topk_indices.detach().long()
        if indices_tensor.ndim == 1:
            indices_tensor = indices_tensor.reshape(batch_size, -1)
        if int(indices_tensor.shape[0]) != batch_size:
            raise ValueError(f"OER top-k batch mismatch: {int(indices_tensor.shape[0])} vs {batch_size}")
        index_rows: Sequence[Any] = [indices_tensor[idx] for idx in range(batch_size)]
    else:
        index_rows = list(topk_indices)
        if len(index_rows) != batch_size:
            raise ValueError(f"OER top-k batch mismatch: {len(index_rows)} vs {batch_size}")

    retention_rows: list[Tensor] = []
    off_ratio_rows: list[Tensor] = []
    fp32_energy_rows: list[Tensor] = []
    quant_energy_rows: list[Tensor] = []
    quant_total_rows: list[Tensor] = []
    for sample_idx, indices_value in enumerate(index_rows):
        retention, off_ratio, fp32_energy, quant_energy, quant_total = compute_one(sample_idx, indices_value)
        retention_rows.append(retention.reshape(()))
        off_ratio_rows.append(off_ratio.reshape(()))
        fp32_energy_rows.append(fp32_energy.reshape(()))
        quant_energy_rows.append(quant_energy.reshape(()))
        quant_total_rows.append(quant_total.reshape(()))

    retention = torch.stack(retention_rows, dim=0)
    off_oracle_ratio = torch.stack(off_ratio_rows, dim=0)
    fp32_oracle_energy = torch.stack(fp32_energy_rows, dim=0)
    quant_oracle_energy = torch.stack(quant_energy_rows, dim=0)
    quant_total_energy = torch.stack(quant_total_rows, dim=0)
    return {
        "oer_retention": retention.detach().cpu().float(),
        "oer_loss": (1.0 - retention).detach().cpu().float(),
        "off_oracle_energy_ratio": off_oracle_ratio.detach().cpu().float(),
        "fp32_oracle_energy": fp32_oracle_energy.detach().cpu().float(),
        "quant_oracle_energy": quant_oracle_energy.detach().cpu().float(),
        "quant_total_energy": quant_total_energy.detach().cpu().float(),
    }


def _oer_metrics_from_fp32_topk(fp32: Tensor, quant: Tensor, *, topk_ratio: float, eps: float) -> dict[str, Tensor]:
    topk_indices = scalar_oer_topk_indices(fp32, topk_ratio=topk_ratio, batch_dim=0).detach().cpu().long()
    values = _oer_metrics_from_topk_indices(fp32, quant, topk_indices, eps=eps)
    values["topk_indices"] = topk_indices
    return values


def _record_btc_shape(shape_records: dict[str, tuple[int, ...]], name: str, tensor: Tensor) -> None:
    shape = tuple(int(item) for item in tensor.shape)
    previous = shape_records.get(name)
    if previous is None:
        shape_records[name] = shape
        return
    if previous[1:] != shape[1:]:
        raise RuntimeError(f"{name} non-batch activation shape changed across batches: {previous} vs {shape}")


def _merged_shape(shape_records: Mapping[str, tuple[int, ...]], name: str, batch_total: int) -> tuple[int, ...]:
    shape = shape_records.get(name)
    if shape is None:
        return ()
    return (int(batch_total), *tuple(int(item) for item in shape[1:]))


def _topk_count_stats(topk_indices: Tensor | Sequence[Any] | None) -> dict[str, int | float | None]:
    if topk_indices is None:
        return {"count": None, "count_min": None, "count_max": None, "count_mean": None}
    if isinstance(topk_indices, Tensor):
        if topk_indices.ndim >= 2:
            counts = [int(topk_indices.shape[1])] * int(topk_indices.shape[0])
        else:
            counts = [int(topk_indices.numel())]
    else:
        counts = [int(torch.as_tensor(item).numel()) for item in topk_indices]
    if not counts:
        counts = [0]
    return {
        "count": int(counts[0]),
        "count_min": int(min(counts)),
        "count_max": int(max(counts)),
        "count_mean": float(sum(counts) / max(1, len(counts))),
    }


def _oer_topk_summary(
    input_oracle_topk_indices: Tensor | Sequence[Any] | None,
    projection_oracle_topk_indices: Tensor | Sequence[Any] | None,
    *,
    oer_oracle: Mapping[str, Any] | None,
    enabled: bool,
    projection_topk_ratio: float,
    sample_count: int,
    cfg: Any,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "input_topk_count": None,
            "input_topk_ratio": None,
            "projection_topk_count": None,
            "projection_topk_ratio": None,
            "sample_count": int(sample_count),
        }
    if input_oracle_topk_indices is None:
        raise RuntimeError("OER top-k summary requested without input oracle indices")
    if projection_oracle_topk_indices is None:
        raise RuntimeError("OER top-k summary requested without output oracle indices")
    input_counts = _topk_count_stats(input_oracle_topk_indices)
    projection_counts = _topk_count_stats(projection_oracle_topk_indices)
    return {
        "enabled": True,
        "input_topk_count": input_counts["count"],
        "input_topk_count_min": input_counts["count_min"],
        "input_topk_count_max": input_counts["count_max"],
        "input_topk_count_mean": input_counts["count_mean"],
        "input_topk_ratio": float(oer_oracle.get("input_topk_ratio", oer_oracle.get("topk_ratio", _oer_top_ratio(cfg)))) if oer_oracle is not None else float(_oer_top_ratio(cfg)),
        "projection_topk_count": projection_counts["count"],
        "projection_topk_count_min": projection_counts["count_min"],
        "projection_topk_count_max": projection_counts["count_max"],
        "projection_topk_count_mean": projection_counts["count_mean"],
        "projection_topk_ratio": float(projection_topk_ratio),
        "sample_count": int(sample_count),
    }


def _normalise_oracle_topk_indices(
    value: Any,
    *,
    target_name: str,
    oracle_key: str,
) -> Tensor | list[Tensor]:
    if value is None:
        raise KeyError(f"{target_name}: OER oracle missing {oracle_key}")
    if isinstance(value, Tensor):
        return value.detach().cpu().long()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [torch.as_tensor(item, dtype=torch.long).cpu().reshape(-1) for item in value]
    return torch.as_tensor(value, dtype=torch.long).cpu()


def _collect_quant_eval(
    cfg: Any,
    *,
    model: nn.Module,
    quant_module: Any,
    eval_loader: DataLoader,
    layer_idx: int,
    module_name: str,
    oer_oracle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    device = _device(cfg)
    model.eval()
    sample_ids: list[str] = []
    labels: list[Tensor] = []
    embeddings: list[Tensor] = []
    layout_debug_records: list[dict[str, Any]] = []
    activation_clipping_rates: list[Tensor] = []
    activation_clipping_rate_by_sample_chunks: list[Tensor] = []
    sample_metric_chunks: dict[str, list[Tensor]] = {
        "activation_quant_error": [],
        "oer_retention": [],
        "oer_loss": [],
        "off_oracle_energy_ratio": [],
        "projection_quant_error": [],
        "projection_oer_retention": [],
        "projection_oer_loss": [],
        "projection_off_oracle_energy_ratio": [],
        "clipping_rate": [],
        "exact_zero_rate": [],
        "input_fp32_oracle_energy": [],
        "input_quant_oracle_energy": [],
        "input_quant_total_energy": [],
        "projection_fp32_oracle_energy": [],
        "projection_quant_oracle_energy": [],
        "projection_quant_total_energy": [],
    }
    output_stats = _TensorStatsAccumulator()
    hook_state: dict[str, Any] = {"batch_size": None}
    target_name = _target_name(layer_idx, module_name)
    shape_records: dict[str, tuple[int, ...]] = {}
    enable_oer_metrics = oer_oracle is not None
    oracle_sample_ids: list[str] = []
    input_oracle_topk_indices: Tensor | list[Tensor] | None = None
    projection_oracle_topk_indices: Tensor | list[Tensor] | None = None
    projection_topk_ratio = _oer_top_ratio(cfg)
    if enable_oer_metrics:
        oracle_sample_ids = [str(item) for item in oer_oracle.get("sample_id", [])]
        raw_input_topk_indices = oer_oracle.get("input_topk_indices", oer_oracle.get("topk_indices", oer_oracle.get("flat_index")))
        raw_projection_topk_indices = oer_oracle.get("output_topk_indices", oer_oracle.get("projection_topk_indices"))
        input_oracle_topk_indices = _normalise_oracle_topk_indices(
            raw_input_topk_indices,
            target_name=target_name,
            oracle_key="input_topk_indices/topk_indices/flat_index",
        )
        projection_oracle_topk_indices = _normalise_oracle_topk_indices(
            raw_projection_topk_indices,
            target_name=target_name,
            oracle_key="output_topk_indices/projection_topk_indices",
        )
        projection_topk_ratio = float(oer_oracle.get("output_topk_ratio", oer_oracle.get("topk_ratio", _oer_top_ratio(cfg))))
    sample_offset = 0
    eps = _metric_eps(cfg)
    activation_quant_active = bool(getattr(quant_module, "fake_quant_enabled", False)) and bool(
        getattr(quant_module, "activation_quant_enabled", False)
    )

    def output_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, Tensor):
            raise TypeError(f"{target_name}: forward output must be Tensor, got {type(output)!r}")
        batch_size = hook_state.get("batch_size")
        if batch_size is None:
            raise RuntimeError(f"{target_name}: output hook is missing batch_size")
        output_stats.update(_activation_to_btc(output, int(batch_size), f"{target_name}.output"))

    handle = quant_module.register_forward_hook(output_hook)
    try:
        with torch.no_grad():
            for batch in eval_loader:
                batch_sample_ids = [str(item) for item in batch["sample_id"]]
                batch_label = batch["label_id"].detach().cpu().long().reshape(-1)
                if len(batch_sample_ids) != int(batch_label.numel()):
                    raise RuntimeError("sample_id count must match label_id count")
                batch_size = int(batch_label.numel())
                expected_oracle_ids = oracle_sample_ids[sample_offset : sample_offset + batch_size]
                if expected_oracle_ids and expected_oracle_ids != batch_sample_ids:
                    raise RuntimeError(
                        f"{target_name}: OER oracle sample_id order mismatch at offset {sample_offset}: "
                        f"oracle={expected_oracle_ids[:3]} batch={batch_sample_ids[:3]}"
                    )
                hook_state["batch_size"] = batch_size
                moved = _move_batch_to_device(batch, device)
                logits, pooled, token_padding_mask, valid_token_count = forward_core_outputs(model, moved, cfg)
                if pooled.ndim != 2:
                    raise RuntimeError(f"{target_name}: quant embedding must be [B,D], got shape={tuple(pooled.shape)}")
                if int(pooled.shape[0]) != batch_size:
                    raise RuntimeError(
                        f"{target_name}: quant embedding batch mismatch {int(pooled.shape[0])} vs {batch_size}"
                    )
                embeddings.append(pooled.float().cpu())
                labels.append(batch_label)
                sample_ids.extend(batch_sample_ids)

                fp32_activation, quant_activation = quant_module.get_oer_activation_tensors()
                if fp32_activation is None or quant_activation is None:
                    raise RuntimeError(f"{target_name}: fake quant eval missing OER input activation cache")
                fp32_projection, quant_projection = quant_module.get_oer_projection_tensors()
                if fp32_projection is None or quant_projection is None:
                    raise RuntimeError(f"{target_name}: fake quant eval missing projection output cache")

                fp32_activation_layout = _activation_to_btc_with_layout(fp32_activation, batch_size, f"{target_name}.fp32_input")
                quant_activation_layout = _activation_to_btc_with_layout(quant_activation, batch_size, f"{target_name}.quant_input")
                fp32_projection_layout = _activation_to_btc_with_layout(fp32_projection, batch_size, f"{target_name}.fp32_projection")
                quant_projection_layout = _activation_to_btc_with_layout(quant_projection, batch_size, f"{target_name}.quant_projection")
                fp32_btc = fp32_activation_layout.tensor
                quant_btc = quant_activation_layout.tensor
                fp32_projection_btc = fp32_projection_layout.tensor
                quant_projection_btc = quant_projection_layout.tensor
                _record_btc_shape(shape_records, "activation_fp32_shape", fp32_btc)
                _record_btc_shape(shape_records, "activation_quant_shape", quant_btc)
                _record_btc_shape(shape_records, "projection_fp32_shape", fp32_projection_btc)
                _record_btc_shape(shape_records, "projection_quant_shape", quant_projection_btc)

                layout_debug_records.append(
                    {
                        "sample_id": list(batch_sample_ids),
                        "fp32_activation": layout_info_to_public_fields(fp32_activation_layout.layout_info),
                        "quant_activation": layout_info_to_public_fields(quant_activation_layout.layout_info),
                        "fp32_projection": layout_info_to_public_fields(fp32_projection_layout.layout_info),
                        "quant_projection": layout_info_to_public_fields(quant_projection_layout.layout_info),
                    }
                )

                sample_metric_chunks["activation_quant_error"].append(
                    _relative_l2_per_sample(fp32_btc, quant_btc, eps=eps)
                )
                if enable_oer_metrics:
                    if input_oracle_topk_indices is None:
                        raise RuntimeError(f"{target_name}: input OER oracle unexpectedly disabled")
                    input_topk = input_oracle_topk_indices[sample_offset : sample_offset + batch_size]
                    input_oer = _oer_metrics_from_topk_indices(fp32_btc, quant_btc, input_topk, eps=eps)
                    sample_metric_chunks["oer_retention"].append(input_oer["oer_retention"])
                    sample_metric_chunks["oer_loss"].append(input_oer["oer_loss"])
                    sample_metric_chunks["off_oracle_energy_ratio"].append(input_oer["off_oracle_energy_ratio"])
                    sample_metric_chunks["input_fp32_oracle_energy"].append(input_oer["fp32_oracle_energy"])
                    sample_metric_chunks["input_quant_oracle_energy"].append(input_oer["quant_oracle_energy"])
                    sample_metric_chunks["input_quant_total_energy"].append(input_oer["quant_total_energy"])
                else:
                    sample_metric_chunks["oer_retention"].append(_nan_vector(batch_size))
                    sample_metric_chunks["oer_loss"].append(_nan_vector(batch_size))
                    sample_metric_chunks["off_oracle_energy_ratio"].append(_nan_vector(batch_size))
                    sample_metric_chunks["input_fp32_oracle_energy"].append(_nan_vector(batch_size))
                    sample_metric_chunks["input_quant_oracle_energy"].append(_nan_vector(batch_size))
                    sample_metric_chunks["input_quant_total_energy"].append(_nan_vector(batch_size))

                sample_metric_chunks["projection_quant_error"].append(
                    _relative_l2_per_sample(fp32_projection_btc, quant_projection_btc, eps=eps)
                )
                if enable_oer_metrics:
                    if projection_oracle_topk_indices is None:
                        raise RuntimeError(f"{target_name}: output OER oracle unexpectedly disabled")
                    projection_topk = projection_oracle_topk_indices[sample_offset : sample_offset + batch_size]
                    projection_oer = _oer_metrics_from_topk_indices(
                        fp32_projection_btc,
                        quant_projection_btc,
                        projection_topk,
                        eps=eps,
                    )
                    sample_metric_chunks["projection_oer_retention"].append(projection_oer["oer_retention"])
                    sample_metric_chunks["projection_oer_loss"].append(projection_oer["oer_loss"])
                    sample_metric_chunks["projection_off_oracle_energy_ratio"].append(
                        projection_oer["off_oracle_energy_ratio"]
                    )
                    sample_metric_chunks["projection_fp32_oracle_energy"].append(projection_oer["fp32_oracle_energy"])
                    sample_metric_chunks["projection_quant_oracle_energy"].append(projection_oer["quant_oracle_energy"])
                    sample_metric_chunks["projection_quant_total_energy"].append(projection_oer["quant_total_energy"])
                else:
                    sample_metric_chunks["projection_oer_retention"].append(_nan_vector(batch_size))
                    sample_metric_chunks["projection_oer_loss"].append(_nan_vector(batch_size))
                    sample_metric_chunks["projection_off_oracle_energy_ratio"].append(_nan_vector(batch_size))
                    sample_metric_chunks["projection_fp32_oracle_energy"].append(_nan_vector(batch_size))
                    sample_metric_chunks["projection_quant_oracle_energy"].append(_nan_vector(batch_size))
                    sample_metric_chunks["projection_quant_total_energy"].append(_nan_vector(batch_size))

                clipping_rate = getattr(quant_module, "last_activation_clipping_rate", None)
                if activation_quant_active and isinstance(clipping_rate, Tensor):
                    activation_clipping_rates.append(clipping_rate.detach().float().cpu().reshape(()))
                clipping_by_sample = getattr(quant_module, "last_activation_clipping_rate_by_sample", None)
                if activation_quant_active and isinstance(clipping_by_sample, Tensor):
                    clipping_by_sample_cpu = clipping_by_sample.detach().float().cpu().reshape(-1)
                    if int(clipping_by_sample_cpu.numel()) != batch_size:
                        raise RuntimeError(
                            f"{target_name}: activation clipping by-sample count mismatch "
                            f"{int(clipping_by_sample_cpu.numel())} vs {batch_size}"
                        )
                else:
                    clipping_by_sample_cpu = _nan_vector(batch_size)
                activation_clipping_rate_by_sample_chunks.append(clipping_by_sample_cpu)
                sample_metric_chunks["clipping_rate"].append(clipping_by_sample_cpu)
                if activation_quant_active:
                    exact_zero = (quant_btc.abs().reshape(batch_size, -1) <= 0.0).float().mean(dim=1).detach().cpu()
                else:
                    exact_zero = _nan_vector(batch_size)
                sample_metric_chunks["exact_zero_rate"].append(exact_zero)

                sample_offset += batch_size
                hook_state["batch_size"] = None
                if hasattr(quant_module, "clear_oer_activation_tensors"):
                    quant_module.clear_oer_activation_tensors()
                del (
                    moved,
                    logits,
                    pooled,
                    token_padding_mask,
                    valid_token_count,
                    fp32_activation,
                    quant_activation,
                    fp32_projection,
                    quant_projection,
                    fp32_btc,
                    quant_btc,
                    fp32_projection_btc,
                    quant_projection_btc,
                )
    finally:
        handle.remove()

    if not embeddings:
        raise RuntimeError("eval_loader produced no batches")
    if oracle_sample_ids and sample_offset != len(oracle_sample_ids):
        raise RuntimeError(
            f"{target_name}: OER oracle sample count mismatch after eval: consumed {sample_offset}, oracle has {len(oracle_sample_ids)}"
        )
    batch_total = len(sample_ids)
    sample_metrics = {
        name: torch.cat(chunks, dim=0) if chunks else _nan_vector(batch_total)
        for name, chunks in sample_metric_chunks.items()
    }
    activation_clipping = torch.stack(activation_clipping_rates).mean() if activation_clipping_rates else torch.tensor(float("nan"))
    activation_clipping_by_sample = (
        torch.cat(activation_clipping_rate_by_sample_chunks, dim=0)
        if activation_clipping_rate_by_sample_chunks
        else _nan_vector(batch_total)
    )
    fp32_weight = quant_module.linear.weight.detach().cpu().float()
    quant_weight = quant_module.weight_quant(quant_module.linear.weight).detach().cpu().float()
    return {
        "sample_id": sample_ids,
        "label_id": torch.cat(labels, dim=0),
        "embedding": torch.cat(embeddings, dim=0),
        "sample_metrics": sample_metrics,
        "fp32_weight": fp32_weight,
        "quant_weight": quant_weight,
        "activation_clipping_rate_runtime": activation_clipping,
        "activation_clipping_rate_by_sample": activation_clipping_by_sample,
        "layout_debug_records": layout_debug_records,
        "activation_layout_info": layout_debug_records[0]["fp32_activation"] if layout_debug_records else {},
        "projection_layout_info": layout_debug_records[0]["fp32_projection"] if layout_debug_records else {},
        "activation_fp32_shape": _merged_shape(shape_records, "activation_fp32_shape", batch_total),
        "activation_quant_shape": _merged_shape(shape_records, "activation_quant_shape", batch_total),
        "projection_fp32_shape": _merged_shape(shape_records, "projection_fp32_shape", batch_total),
        "projection_quant_shape": _merged_shape(shape_records, "projection_quant_shape", batch_total),
        "output_stats": output_stats.summary(),
        "oer_feature_layout": OER_FEATURE_LAYOUT,
        "input_oer_activation_kind": "input_activation",
        "input_oer_oracle_source": "oer_oracle_eval.pt.input.topk_scalar_indices" if enable_oer_metrics else "disabled",
        "projection_oer_activation_kind": "linear_projection_output",
        "projection_oer_oracle_source": "oer_oracle_eval.pt.output.topk_scalar_indices" if enable_oer_metrics else "disabled",
        "oer_topk_summary": _oer_topk_summary(
            input_oracle_topk_indices,
            projection_oracle_topk_indices,
            oer_oracle=oer_oracle,
            enabled=enable_oer_metrics,
            projection_topk_ratio=projection_topk_ratio,
            sample_count=batch_total,
            cfg=cfg,
        ),
    }


def _assert_same_eval_order(fp32_embedding_cache: Mapping[str, Any], quant_cache: Mapping[str, Any]) -> None:
    if list(fp32_embedding_cache["sample_id"]) != list(quant_cache["sample_id"]):
        raise RuntimeError("quant eval sample_id order does not match FP32 reference cache")
    fp32_labels = fp32_embedding_cache["label_id"].detach().cpu().long().reshape(-1)
    quant_labels = quant_cache["label_id"].detach().cpu().long().reshape(-1)
    if not torch.equal(fp32_labels, quant_labels):
        raise RuntimeError("quant eval label_id does not match FP32 reference cache")
    fp32_embedding = torch.as_tensor(fp32_embedding_cache["embedding"]).detach().cpu()
    quant_embedding = torch.as_tensor(quant_cache["embedding"]).detach().cpu()
    if tuple(fp32_embedding.shape) != tuple(quant_embedding.shape):
        raise RuntimeError(
            "quant eval embedding shape must match core_eval.pt embedding shape before geometry metrics: "
            f"fp32={tuple(fp32_embedding.shape)}, quant={tuple(quant_embedding.shape)}"
        )


def run_single_quant_mode(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    qparams: Mapping[str, Any],
    eval_loader: DataLoader,
    oer_oracle: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Quant mode ablation helper."""

    device = _device(cfg)
    model = _restore_clean_model(cfg).to(device)
    weight_qparams = qparams["weight"] if _mode_uses_weight(quant_mode) else None
    activation_qparams = qparams["activation"] if _mode_uses_activation(quant_mode) else None
    target_name = _target_name(layer_idx, module_name)
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
    try:
        controller.qparams_frozen = True
        controller.prepare_quant_forward()
        _assert_quant_module_has_fixed_qparams(quant_module, quant_mode, target_name)
        return _collect_quant_eval(
            cfg,
            model=model,
            quant_module=quant_module,
            eval_loader=eval_loader,
            layer_idx=int(layer_idx),
            module_name=module_name,
            oer_oracle=oer_oracle,
        )
    finally:
        controller.remove_padding_mask_hooks()


def _save_mode_artifacts(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    quant_cache: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    target_detail_rows: Sequence[Mapping[str, Any]],
    qparams_path: Path,
) -> tuple[Path, Path]:
    base_dir = _result_dir(cfg) / ARTIFACT_DIRNAME / f"layer_{int(layer_idx):02d}_{module_name}_{_mode_slug(quant_mode)}"
    base_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = base_dir / "embedding.pt"
    artifact_path = base_dir / "eval_stats.pt"
    embedding_tmp_path = embedding_path.with_name(f"{embedding_path.name}.tmp")
    artifact_tmp_path = artifact_path.with_name(f"{artifact_path.name}.tmp")
    torch.save(
        {
            "layer_idx": int(layer_idx),
            "module_name": module_name,
            "quant_mode": _mode_label(quant_mode),
            "sample_id": list(quant_cache["sample_id"]),
            "label_id": quant_cache["label_id"].detach().cpu(),
            "embedding": quant_cache["embedding"].detach().cpu(),
            "oer_feature_layout": OER_FEATURE_LAYOUT,
        },
        embedding_tmp_path,
    )
    os.replace(embedding_tmp_path, embedding_path)
    torch.save(
        {
            "layer_idx": int(layer_idx),
            "module_name": module_name,
            "quant_mode": _mode_label(quant_mode),
            "qparams_path": str(qparams_path),
            "sample_id": list(quant_cache["sample_id"]),
            "label_id": quant_cache["label_id"].detach().cpu(),
            "activation_fp32_shape": tuple(quant_cache.get("activation_fp32_shape", ())),
            "activation_quant_shape": tuple(quant_cache.get("activation_quant_shape", ())),
            "projection_fp32_shape": tuple(quant_cache.get("projection_fp32_shape", ())),
            "projection_quant_shape": tuple(quant_cache.get("projection_quant_shape", ())),
            "activation_layout_info": quant_cache.get("activation_layout_info", {}),
            "projection_layout_info": quant_cache.get("projection_layout_info", {}),
            "layout_debug_records": list(quant_cache.get("layout_debug_records", [])),
            "input_oer_activation_kind": quant_cache.get("input_oer_activation_kind", "input_activation"),
            "input_oer_oracle_source": quant_cache.get("input_oer_oracle_source", "oer_oracle_eval.pt.input.topk_scalar_indices"),
            "projection_oer_activation_kind": quant_cache.get("projection_oer_activation_kind", "linear_projection_output"),
            "projection_oer_oracle_source": quant_cache.get("projection_oer_oracle_source", "oer_oracle_eval.pt.output.topk_scalar_indices"),
            "oer_topk_summary": dict(quant_cache.get("oer_topk_summary", {})),
            "output_stats": dict(quant_cache["output_stats"]),
            "activation_clipping_rate_runtime": quant_cache["activation_clipping_rate_runtime"],
            "activation_clipping_rate_by_sample": quant_cache["activation_clipping_rate_by_sample"].detach().cpu(),
            "sample_metrics": {
                str(name): value.detach().cpu() if isinstance(value, Tensor) else value
                for name, value in dict(quant_cache.get("sample_metrics", {})).items()
            },
            "metric_rows": list(metric_rows),
            "comparison_rows": list(comparison_rows),
            "target_detail_rows": list(target_detail_rows),
            "energy_stats": [
                {
                    "analysis_group": row.get("analysis_group", ""),
                    "metric_name": row.get("metric_name", ""),
                    "metric_value": row.get("metric_value", ""),
                    "aggregation_unit": row.get("aggregation_unit", ""),
                    "aggregation_fn": row.get("aggregation_fn", ""),
                }
                for row in metric_rows
                if str(row.get("metric_name", ""))
                in {
                    "oer_retention",
                    "oer_loss",
                    "off_oracle_energy_ratio",
                    "projection_oer_retention",
                    "projection_oer_loss",
                    "projection_off_oracle_energy_ratio",
                }
            ],
        },
        artifact_tmp_path,
    )
    os.replace(artifact_tmp_path, artifact_path)
    return embedding_path, artifact_path


def _to_csv_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return json.dumps(_json_safe(value), ensure_ascii=False)
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _write_rows(csv_path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] = CSV_COLUMNS) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns)
    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_csv_value(row.get(key, "")) for key in fieldnames})
    os.replace(tmp_path, csv_path)


def _fp32_reference_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_REFERENCE_CACHE", {})
    return value if isinstance(value, Mapping) else {}


def _enable_oer_metrics(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "ENABLE_OER_METRICS", _setting_get(cfg, "OER_METRICS_ENABLED", True)))


def _fp32_reference_cache_root(cfg: Any) -> Path:
    configured = _setting_get(cfg, "FP32_REFERENCE_CACHE_ROOT", None)
    if configured is None:
        configured = _nested_get(_fp32_reference_cfg(cfg), "CACHE_ROOT", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "cache" / "fp32_reference"


def _fp32_core_eval_path(cfg: Any) -> Path:
    for key in ("CORE_CACHE_PATH", "CORE_EVAL_PATH", "FP32_CORE_CACHE_PATH"):
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
    reference_cfg = _fp32_reference_cfg(cfg)
    for key in ("CORE_CACHE_PATH", "CORE_EVAL_PATH", "FP32_CORE_CACHE_PATH"):
        value = _nested_get(reference_cfg, key, None)
        if value is not None:
            return Path(value)
    return _fp32_reference_cache_root(cfg) / FP32_REFERENCE_CORE_FILENAME


def _fp32_oer_oracle_path(cfg: Any) -> Path:
    for key in ("OER_ORACLE_PATH", "OER_ORACLE_EVAL_PATH", "FP32_OER_ORACLE_PATH"):
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
    reference_cfg = _fp32_reference_cfg(cfg)
    for key in ("OER_ORACLE_PATH", "OER_ORACLE_EVAL_PATH", "FP32_OER_ORACLE_PATH"):
        value = _nested_get(reference_cfg, key, None)
        if value is not None:
            return Path(value)
    return _fp32_reference_cache_root(cfg) / FP32_REFERENCE_OER_ORACLE_FILENAME


def _artifact_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _read_json_mapping_required(path: Path, *, context: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{context} manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} manifest must be a mapping: {path}")
    return dict(payload)


def _normalise_sample_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Tensor):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_id_from_metadata_row(row: Mapping[str, Any]) -> str | None:
    for key in ("sample_id", "audio_id", "file_id", "id"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            return text
    return None


def _read_eval_csv_sample_ids(cfg: Any) -> list[str]:
    csv_path = _split_csv(cfg, "eval")
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"eval CSV does not exist: {csv_path}")
    sample_ids: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"eval CSV has no header: {csv_path}")
        for row_idx, row in enumerate(reader, start=2):
            sample_id = _sample_id_from_metadata_row(row)
            if sample_id is None:
                raise ValueError(f"{csv_path}:{row_idx}: missing sample_id/audio_id/file_id")
            sample_ids.append(str(sample_id))
            if str(sample_id) in seen:
                duplicates.append(str(sample_id))
            seen.add(str(sample_id))
    if not sample_ids:
        raise ValueError(f"eval CSV has no samples: {csv_path}")
    if duplicates:
        raise ValueError(f"eval CSV has duplicate sample_id values: {sorted(set(duplicates))[:8]}")
    return sample_ids


def _resolve_reference_relative_path(root_file: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root_file.parent / path


def _validate_sharded_reference_manifest(
    artifact_path: Path,
    manifest: Mapping[str, Any],
    *,
    expected_cache_type: str,
    expected_stage: str,
    expected_storage_format: str,
    context: str,
) -> None:
    mismatches: list[str] = []
    required_pairs = {
        "cache_type": expected_cache_type,
        "stage": expected_stage,
        "storage_format": expected_storage_format,
    }
    for key, expected in required_pairs.items():
        actual = manifest.get(key)
        if actual != expected:
            mismatches.append(f"manifest[{key}] expected {expected!r}, got {actual!r}")
    if not bool(manifest.get("sharded", False)):
        mismatches.append("manifest[sharded] expected True")
    shard_paths = manifest.get("shard_paths")
    if not isinstance(shard_paths, list) or not shard_paths:
        mismatches.append("manifest[shard_paths] must be a non-empty list")
    declared_shards = manifest.get("num_shards")
    if declared_shards is None:
        mismatches.append("manifest[num_shards] is required")
    elif isinstance(shard_paths, list) and int(declared_shards) != len(shard_paths):
        mismatches.append(f"manifest[num_shards]={declared_shards} but shard_paths has {len(shard_paths)} entries")
    if isinstance(shard_paths, list):
        missing = [str(item) for item in shard_paths if not _resolve_reference_relative_path(artifact_path, item).is_file()]
        if missing:
            mismatches.append(f"missing shard files: {missing[:8]}")
    if mismatches:
        preview = "; ".join(mismatches[:8])
        raise RuntimeError(f"{context} is not a valid sharded FP32 reference artifact: {artifact_path}; {preview}")


def _validate_reference_identity(cfg: Any, manifest: Mapping[str, Any], *, context: str) -> tuple[str, str]:
    eval_csv = _split_csv(cfg, "eval")
    eval_csv_hash = _sha256_file(eval_csv)
    if manifest.get("eval_csv_sha256") != eval_csv_hash:
        raise RuntimeError(
            f"{context} eval_csv_sha256 mismatch: "
            f"manifest={manifest.get('eval_csv_sha256')!r}, current={eval_csv_hash!r}, eval_csv={eval_csv}"
        )
    checkpoint_path = _optional_checkpoint_path(cfg)
    checkpoint_hash = None if checkpoint_path is None else _sha256_file(checkpoint_path)
    if checkpoint_hash is None:
        raise RuntimeError(f"{context} checkpoint_sha256 cannot be validated because checkpoint path is missing or unreadable")
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(
            f"{context} checkpoint_sha256 mismatch: "
            f"manifest={manifest.get('checkpoint_sha256')!r}, current={checkpoint_hash!r}, checkpoint={checkpoint_path}"
        )
    return str(eval_csv_hash), str(checkpoint_hash)


def _load_required_core_eval_reference(cfg: Any) -> dict[str, Any]:
    core_path = _fp32_core_eval_path(cfg)
    manifest_path = _artifact_manifest_path(core_path)
    manifest = _read_json_mapping_required(manifest_path, context="core_eval.pt")
    _validate_sharded_reference_manifest(
        core_path,
        manifest,
        expected_cache_type=FP32_CORE_CACHE_TYPE,
        expected_stage=FP32_CORE_STAGE,
        expected_storage_format=FP32_CORE_STORAGE_FORMAT,
        context="core_eval.pt",
    )
    eval_csv_hash, checkpoint_hash = _validate_reference_identity(cfg, manifest, context="core_eval.pt")
    eval_sample_ids = _read_eval_csv_sample_ids(cfg)
    expected_sample_hash = _sample_order_hash(eval_sample_ids)
    manifest_sample_hash = manifest.get("sample_order_hash")
    if manifest_sample_hash != expected_sample_hash:
        raise RuntimeError(
            "core_eval.pt sample_order_hash does not match eval CSV order: "
            f"manifest={manifest_sample_hash!r}, current={expected_sample_hash!r}"
        )

    payload = load_fp32_core_cache(core_path, materialize=True, include_padding_mask=False, include_logits=False)
    core_sample_ids = _normalise_sample_ids(payload.get("sample_id"))
    if core_sample_ids != eval_sample_ids:
        first_mismatches = [
            {"index": idx, "core_eval": observed, "eval_csv": expected}
            for idx, (observed, expected) in enumerate(zip(core_sample_ids, eval_sample_ids))
            if observed != expected
        ][:8]
        raise RuntimeError(
            "core_eval.pt sample_id order must exactly match eval CSV for quant mode ablation; "
            f"core_count={len(core_sample_ids)}, eval_count={len(eval_sample_ids)}, first_mismatches={first_mismatches}"
        )
    embedding = torch.as_tensor(payload["embedding"], dtype=torch.float32).cpu().contiguous()
    label_id = torch.as_tensor(payload["label_id"], dtype=torch.long).cpu().reshape(-1).contiguous()
    if int(embedding.shape[0]) != len(core_sample_ids) or int(label_id.numel()) != len(core_sample_ids):
        raise RuntimeError("core_eval.pt embedding/label_id rows must match sample_id count")
    embedding_cache = {
        "sample_id": list(core_sample_ids),
        "label_id": label_id,
        "embedding": embedding,
        "bucket": list(payload.get("bucket", [])),
        "shot_group": list(payload.get("shot_group", payload.get("bucket", []))),
        "tail_included": list(payload.get("tail_included", [])),
    }
    if len(embedding_cache["bucket"]) != len(core_sample_ids):
        raise RuntimeError("core_eval.pt bucket rows must match sample_id count")
    if len(embedding_cache["shot_group"]) != len(core_sample_ids):
        embedding_cache["shot_group"] = list(embedding_cache["bucket"])
    if len(embedding_cache["tail_included"]) != len(core_sample_ids):
        embedding_cache["tail_included"] = [str(bucket) in {"medium", "few"} for bucket in embedding_cache["bucket"]]

    return {
        "required": True,
        "status": "validated",
        "path": core_path,
        "manifest_path": manifest_path,
        "identity": _path_identity(core_path),
        "manifest_identity": _path_identity(manifest_path),
        "manifest": dict(manifest),
        "cache_type": manifest.get("cache_type"),
        "stage": manifest.get("stage"),
        "storage_format": manifest.get("storage_format"),
        "sharded": bool(manifest.get("sharded", False)),
        "num_shards": int(manifest.get("num_shards", 0)),
        "eval_csv_sha256": eval_csv_hash,
        "checkpoint_sha256": checkpoint_hash,
        "sample_order_hash": manifest_sample_hash,
        "sample_count": len(core_sample_ids),
        "sample_id_preview": core_sample_ids[:5],
        "embedding_cache": embedding_cache,
    }


def _target_manifest_set(value: Any) -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return result
    for item in value:
        if isinstance(item, Mapping):
            result.add((int(item.get("layer_idx", -1)), str(item.get("module_name", ""))))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            result.add((int(item[0]), str(item[1])))
    return result


def _load_required_oer_oracle_reference(
    cfg: Any,
    *,
    core_reference: Mapping[str, Any],
    target_modules: Sequence[tuple[int, str]],
) -> dict[str, Any]:
    oracle_path = _fp32_oer_oracle_path(cfg)
    manifest_path = _artifact_manifest_path(oracle_path)
    manifest = _read_json_mapping_required(manifest_path, context="oer_oracle_eval.pt")
    _validate_sharded_reference_manifest(
        oracle_path,
        manifest,
        expected_cache_type=FP32_OER_CACHE_TYPE,
        expected_stage=FP32_OER_STAGE,
        expected_storage_format=FP32_OER_STORAGE_FORMAT,
        context="oer_oracle_eval.pt",
    )
    eval_csv_hash, checkpoint_hash = _validate_reference_identity(cfg, manifest, context="oer_oracle_eval.pt")
    if eval_csv_hash != core_reference.get("eval_csv_sha256"):
        raise RuntimeError("oer_oracle_eval.pt eval_csv_sha256 differs from core_eval.pt")
    if checkpoint_hash != core_reference.get("checkpoint_sha256"):
        raise RuntimeError("oer_oracle_eval.pt checkpoint_sha256 differs from core_eval.pt")
    if manifest.get("sample_order_hash") != core_reference.get("sample_order_hash"):
        raise RuntimeError(
            "oer_oracle_eval.pt sample_order_hash differs from core_eval.pt: "
            f"oer={manifest.get('sample_order_hash')!r}, core={core_reference.get('sample_order_hash')!r}"
        )
    cached_ratio = manifest.get("topk_ratio")
    if cached_ratio is None or abs(float(cached_ratio) - float(_oer_top_ratio(cfg))) > 1e-12:
        raise RuntimeError(f"oer_oracle_eval.pt topk_ratio mismatch: manifest={cached_ratio!r}, current={_oer_top_ratio(cfg)!r}")
    positions = {str(item) for item in manifest.get("activation_positions", [])}
    missing_positions = [position for position in ("input", "output") if position not in positions]
    if missing_positions:
        raise RuntimeError(
            "oer_oracle_eval.pt must include activation_position='input' and activation_position='output' "
            f"for quant mode OER metrics; missing={missing_positions}"
        )
    cached_targets = _target_manifest_set(manifest.get("target_modules"))
    missing_targets = [
        {"layer_idx": int(layer_idx), "module_name": str(module_name)}
        for layer_idx, module_name in target_modules
        if (int(layer_idx), str(module_name)) not in cached_targets
    ]
    if missing_targets:
        raise RuntimeError(f"oer_oracle_eval.pt missing target modules required by this experiment: {missing_targets[:8]}")

    oracle = load_fp32_oer_oracle(oracle_path, materialize=False)
    oracle_sample_ids = _normalise_sample_ids(oracle.get("sample_id_order", oracle.get("sample_id")))
    core_sample_ids = _normalise_sample_ids(_nested_get(core_reference, "embedding_cache", {}).get("sample_id"))
    if oracle_sample_ids and oracle_sample_ids != core_sample_ids:
        raise RuntimeError("oer_oracle_eval.pt sample_id_order must exactly match core_eval.pt sample_id order")
    return {
        "required": True,
        "status": "validated",
        "path": oracle_path,
        "manifest_path": manifest_path,
        "identity": _path_identity(oracle_path),
        "manifest_identity": _path_identity(manifest_path),
        "manifest": dict(manifest),
        "oracle": oracle,
        "cache_type": manifest.get("cache_type"),
        "stage": manifest.get("stage"),
        "storage_format": manifest.get("storage_format"),
        "sharded": bool(manifest.get("sharded", False)),
        "num_shards": int(manifest.get("num_shards", 0)),
        "sample_order_hash": manifest.get("sample_order_hash"),
        "topk_ratio": float(cached_ratio),
        "activation_positions": list(manifest.get("activation_positions", [])),
    }


def _describe_optional_oer_oracle_reference(cfg: Any) -> dict[str, Any]:
    path = _fp32_oer_oracle_path(cfg)
    manifest_path = _artifact_manifest_path(path)
    result: dict[str, Any] = {
        "required": False,
        "path": path,
        "manifest_path": manifest_path,
        "exists": bool(path.exists()),
        "status": "missing",
    }
    if not path.exists():
        return result
    result["status"] = "present_without_manifest"
    if not manifest_path.exists():
        return result
    try:
        manifest = _read_json_mapping_required(manifest_path, context="oer_oracle_eval.pt")
    except Exception as exc:  # noqa: BLE001 - optional diagnostic only.
        result["status"] = "invalid_manifest"
        result["error"] = str(exc)
        return result
    result.update(
        {
            "status": "present",
            "identity": _path_identity(path),
            "manifest_identity": _path_identity(manifest_path),
            "cache_type": manifest.get("cache_type"),
            "oracle_source": manifest.get("oracle_source"),
            "stage": manifest.get("stage"),
            "storage_format": manifest.get("storage_format"),
            "sharded": bool(manifest.get("sharded", False)),
            "num_shards": manifest.get("num_shards"),
            "sample_order_hash": manifest.get("sample_order_hash"),
            "topk_ratio": manifest.get("topk_ratio"),
            "activation_positions": manifest.get("activation_positions"),
        }
    )
    return result


def _build_reference_cache(cfg: Any, _eval_loader: DataLoader, target_modules: Sequence[tuple[int, str]]) -> dict[str, Any]:
    core_reference = _load_required_core_eval_reference(cfg)
    enable_oer = _enable_oer_metrics(cfg)
    if enable_oer:
        oer_reference = _load_required_oer_oracle_reference(
            cfg,
            core_reference=core_reference,
            target_modules=target_modules,
        )
        reference_state = {
            "required": {"core_eval": {key: value for key, value in core_reference.items() if key != "embedding_cache"}},
            "required_if_enable_oer": {"oer_oracle_eval": {key: value for key, value in oer_reference.items() if key != "oracle"}},
            "optional": {},
        }
    else:
        oer_reference = _describe_optional_oer_oracle_reference(cfg)
        reference_state = {
            "required": {"core_eval": {key: value for key, value in core_reference.items() if key != "embedding_cache"}},
            "required_if_enable_oer": {},
            "optional": {"oer_oracle_eval": oer_reference},
        }
    return {
        "embedding_cache": core_reference["embedding_cache"],
        "oer_oracle_reference": oer_reference,
        "oer_oracle_cache": {},
        "enable_oer_metrics": bool(enable_oer),
        "reference_state": reference_state,
        "manifest": reference_state,
    }

def _oer_position_key(value: Any) -> str | None:
    text = str(value)
    if text in {"input", "input_activation", "pre_linear", "linear_input"}:
        return "input"
    if text in {"output", "output_activation", "post_linear", "linear_output", "linear_projection_output"}:
        return "output"
    return None


def _load_target_oer_oracle(
    fp32_reference_cache: Mapping[str, Any],
    *,
    layer_idx: int,
    module_name: str,
) -> dict[str, Any]:
    reference = fp32_reference_cache.get("oer_oracle_reference")
    if not isinstance(reference, Mapping) or not bool(reference.get("required", False)):
        raise RuntimeError("OER metrics are enabled but oer_oracle_eval.pt was not validated")
    oracle = reference.get("oracle")
    if not isinstance(oracle, Mapping):
        raise RuntimeError("validated OER oracle reference is missing its top-level shard manifest")
    oracle_path = Path(reference["path"])
    expected_sample_ids = _normalise_sample_ids(fp32_reference_cache["embedding_cache"].get("sample_id"))
    required_positions = ("input", "output")
    sample_ids_by_position: dict[str, list[str]] = {position: [] for position in required_positions}
    topk_rows_by_position: dict[str, list[Tensor]] = {position: [] for position in required_positions}
    topk_ratios_by_position: dict[str, list[float]] = {position: [] for position in required_positions}
    layout_records_by_position: dict[str, list[dict[str, Any]]] = {position: [] for position in required_positions}
    target_name = _target_name(layer_idx, module_name)

    def row_value(shard: Mapping[str, Any], key: str, row_idx: int, default: Any = "") -> Any:
        values = shard.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) > row_idx:
            return values[row_idx]
        return default

    for shard in iter_fp32_oer_oracle_shards(oracle, oracle_path, layer_idx=int(layer_idx)):
        shard_sample_ids = [str(item) for item in shard.get("sample_id", [])]
        row_count = len(shard_sample_ids)
        if row_count <= 0:
            continue
        layer_values = torch.as_tensor(shard.get("layer_idx"), dtype=torch.long).reshape(-1)
        module_values = [str(item) for item in shard.get("module_name", [])]
        position_values = [str(item) for item in shard.get("activation_position", [])]
        topk_values = shard.get("topk_scalar_indices")
        ratio_values = torch.as_tensor(shard.get("topk_ratio"), dtype=torch.float32).reshape(-1)
        if len(module_values) != row_count or len(position_values) != row_count or int(layer_values.numel()) != row_count:
            raise RuntimeError(f"{target_name}: OER shard row metadata length mismatch")
        if not isinstance(topk_values, Sequence) or isinstance(topk_values, (str, bytes)) or len(topk_values) != row_count:
            raise RuntimeError(f"{target_name}: OER shard topk_scalar_indices length mismatch")
        if int(ratio_values.numel()) != row_count:
            raise RuntimeError(f"{target_name}: OER shard topk_ratio length mismatch")
        for row_idx, sample_id in enumerate(shard_sample_ids):
            if int(layer_values[row_idx].item()) != int(layer_idx):
                continue
            if module_values[row_idx] != str(module_name):
                continue
            position_key = _oer_position_key(position_values[row_idx])
            if position_key not in required_positions:
                continue
            sample_ids_by_position[position_key].append(str(sample_id))
            topk_rows_by_position[position_key].append(
                torch.as_tensor(topk_values[row_idx], dtype=torch.long).cpu().reshape(-1).contiguous()
            )
            topk_ratios_by_position[position_key].append(float(ratio_values[row_idx].item()))
            if not layout_records_by_position[position_key]:
                layout_records_by_position[position_key].append(
                    {
                        "activation_shape": row_value(shard, "activation_shape", row_idx),
                        "original_layout": row_value(shard, "original_layout", row_idx),
                        "original_shape": row_value(shard, "original_shape", row_idx),
                        "canonical_layout": row_value(shard, "canonical_layout", row_idx),
                        "canonical_shape": row_value(shard, "canonical_shape", row_idx),
                        "layout_inference_source": row_value(shard, "layout_inference_source", row_idx),
                        "padding_mask_available": row_value(shard, "padding_mask_available", row_idx, False),
                        "padding_mask_aligned": row_value(shard, "padding_mask_aligned", row_idx, False),
                        "mask_status": row_value(shard, "mask_status", row_idx),
                    }
                )

    for position in required_positions:
        position_sample_ids = sample_ids_by_position[position]
        if not position_sample_ids:
            raise KeyError(f"oer_oracle_eval.pt has no {position} OER rows for {target_name}")
        if position_sample_ids != expected_sample_ids:
            first_mismatches = [
                {"index": idx, "oracle": observed, "core_eval": expected}
                for idx, (observed, expected) in enumerate(zip(position_sample_ids, expected_sample_ids))
                if observed != expected
            ][:8]
            raise RuntimeError(
                f"{target_name}: {position} OER oracle sample_id order must match core_eval.pt; "
                f"oracle_count={len(position_sample_ids)}, core_count={len(expected_sample_ids)}, "
                f"first_mismatches={first_mismatches}"
            )
        if len(set(position_sample_ids)) != len(position_sample_ids):
            raise RuntimeError(f"{target_name}: {position} OER oracle contains duplicate sample_id rows")
        topk_ratios = topk_ratios_by_position[position]
        if topk_ratios and any(abs(float(item) - float(topk_ratios[0])) > 1e-12 for item in topk_ratios):
            raise RuntimeError(f"{target_name}: {position} OER oracle topk_ratio changed across samples")

    input_topk_ratio = (
        float(topk_ratios_by_position["input"][0])
        if topk_ratios_by_position["input"]
        else float(reference.get("topk_ratio", 0.0))
    )
    output_topk_ratio = (
        float(topk_ratios_by_position["output"][0])
        if topk_ratios_by_position["output"]
        else float(reference.get("topk_ratio", 0.0))
    )
    if abs(input_topk_ratio - output_topk_ratio) > 1e-12:
        raise RuntimeError(
            f"{target_name}: input/output OER oracle topk_ratio mismatch: "
            f"input={input_topk_ratio}, output={output_topk_ratio}"
        )
    return {
        "layout": OER_FEATURE_LAYOUT,
        "contains_full_activation": False,
        "oracle_source": "oer_oracle_eval.pt.input+output.topk_scalar_indices",
        "input_oracle_source": "oer_oracle_eval.pt.input.topk_scalar_indices",
        "output_oracle_source": "oer_oracle_eval.pt.output.topk_scalar_indices",
        "activation_kind": "input_activation",
        "output_activation_kind": "linear_projection_output",
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "target_name": target_name,
        "sample_id": list(expected_sample_ids),
        "input_sample_id": sample_ids_by_position["input"],
        "output_sample_id": sample_ids_by_position["output"],
        "topk_indices": topk_rows_by_position["input"],
        "flat_index": topk_rows_by_position["input"],
        "input_topk_indices": topk_rows_by_position["input"],
        "output_topk_indices": topk_rows_by_position["output"],
        "projection_topk_indices": topk_rows_by_position["output"],
        "topk_ratio": input_topk_ratio,
        "input_topk_ratio": input_topk_ratio,
        "output_topk_ratio": output_topk_ratio,
        "layout_debug": {
            "input": layout_records_by_position["input"][0] if layout_records_by_position["input"] else {},
            "output": layout_records_by_position["output"][0] if layout_records_by_position["output"] else {},
        },
        "input_layout_debug": layout_records_by_position["input"][0] if layout_records_by_position["input"] else {},
        "output_layout_debug": layout_records_by_position["output"][0] if layout_records_by_position["output"] else {},
    }


def _load_target_input_oer_oracle(
    fp32_reference_cache: Mapping[str, Any],
    *,
    layer_idx: int,
    module_name: str,
) -> dict[str, Any]:
    return _load_target_oer_oracle(
        fp32_reference_cache,
        layer_idx=int(layer_idx),
        module_name=module_name,
    )


def _tensor_shape(value: Any) -> tuple[int, ...] | str:
    if value is None:
        return ""
    try:
        tensor = value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError):
        return ""
    return tuple(int(item) for item in tensor.shape)


def _tensor_summary(prefix: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {
            f"{prefix}_shape": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
            f"{prefix}_mean": "",
            f"{prefix}_std": "",
            f"{prefix}_finite_check": False,
        }
    try:
        tensor = value.detach().cpu().float() if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return {
            f"{prefix}_shape": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
            f"{prefix}_mean": "",
            f"{prefix}_std": "",
            f"{prefix}_finite_check": False,
        }
    flat = tensor.reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite = flat[finite_mask]
    if int(finite.numel()) == 0:
        return {
            f"{prefix}_shape": tuple(int(item) for item in tensor.shape),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_finite_check": False,
        }
    std = finite.std(unbiased=False) if int(finite.numel()) > 1 else finite.new_tensor(0.0)
    return {
        f"{prefix}_shape": tuple(int(item) for item in tensor.shape),
        f"{prefix}_min": float(finite.min().item()),
        f"{prefix}_max": float(finite.max().item()),
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_std": float(std.item()),
        f"{prefix}_finite_check": bool(finite_mask.all().item()),
    }


def _vector_stats(prefix: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_std": "",
            f"{prefix}_max": "",
            f"{prefix}_count": 0,
        }
    tensor = value.detach().cpu().float().reshape(-1) if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_count": 0,
        }
    std = finite.std(unbiased=False) if int(finite.numel()) > 1 else finite.new_tensor(0.0)
    return {
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_std": float(std.item()),
        f"{prefix}_max": float(finite.max().item()),
        f"{prefix}_count": int(finite.numel()),
    }


def _first_mapping_value(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        return {}
    first = next(iter(value.values()))
    return first if isinstance(first, Mapping) else {}


def _target_summary(summary_by_target: Any, target_name: str) -> Mapping[str, Any]:
    if not isinstance(summary_by_target, Mapping):
        return {}
    direct = summary_by_target.get(target_name)
    if isinstance(direct, Mapping):
        return direct
    return _first_mapping_value(summary_by_target)


def _qparams_for_mode(qparams: Mapping[str, Any], quant_mode: str, kind: str) -> Mapping[str, Any] | None:
    if kind == "weight" and not _mode_uses_weight(quant_mode):
        return None
    if kind == "activation" and not _mode_uses_activation(quant_mode):
        return None
    value = qparams.get(kind)
    return value if isinstance(value, Mapping) else None


def _clip_rate_proxy(values: Any, qparams: Mapping[str, Any] | None) -> float:
    if qparams is None or "scale" not in qparams:
        return float("nan")
    tensor = values.detach().cpu().float() if isinstance(values, Tensor) else torch.as_tensor(values, dtype=torch.float32)
    scale = qparams["scale"].detach().cpu().float() if isinstance(qparams["scale"], Tensor) else torch.as_tensor(qparams["scale"], dtype=torch.float32)
    zero_point_value = qparams.get("zero_point", 0.0)
    zero_point = zero_point_value.detach().cpu().float() if isinstance(zero_point_value, Tensor) else torch.as_tensor(zero_point_value, dtype=torch.float32)
    qmin = float(qparams.get("qmin", -7))
    qmax = float(qparams.get("qmax", 7))
    scale = torch.clamp(scale, min=1e-12)
    raw = tensor / scale + zero_point
    clipped = (raw < qmin) | (raw > qmax)
    return float(clipped.float().mean().item()) if int(clipped.numel()) else float("nan")


def _build_qparams_summary_row(
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    qparams: Mapping[str, Any],
    qparams_path: Path,
    quant_cache: Mapping[str, Any],
) -> dict[str, Any]:
    target_name = _target_name(layer_idx, module_name)
    observer_summary = _target_summary(qparams.get("observer_summary"), target_name)
    weight_observer_summary = _target_summary(qparams.get("weight_observer_summary"), target_name)
    controller_summary = qparams.get("controller_summary", {}) if isinstance(qparams.get("controller_summary"), Mapping) else {}
    weight_qparams = _qparams_for_mode(qparams, quant_mode, "weight")
    activation_qparams = _qparams_for_mode(qparams, quant_mode, "activation")
    clip_stats = _vector_stats("activation_clipping_rate_by_sample", quant_cache.get("activation_clipping_rate_by_sample"))
    row: dict[str, Any] = {
        "row_type": "target_qparams_summary",
        "group_protocol_version": 2,
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "target_name": target_name,
        "quant_mode": _mode_label(quant_mode),
        "granularity": "per-channel-weight/per-tensor-activation",
        "qparams_path": str(qparams_path),
        "weight_qparams_enabled": bool(_mode_uses_weight(quant_mode)),
        "activation_qparams_enabled": bool(_mode_uses_activation(quant_mode)),
        "weight_qmin": "" if weight_qparams is None else weight_qparams.get("qmin", ""),
        "weight_qmax": "" if weight_qparams is None else weight_qparams.get("qmax", ""),
        "weight_symmetric": "" if weight_qparams is None else weight_qparams.get("symmetric", ""),
        "weight_granularity": "" if weight_qparams is None else weight_qparams.get("granularity", ""),
        "weight_channel_axis": "" if weight_qparams is None else weight_qparams.get("channel_axis", ""),
        "weight_clip_rate_proxy": _clip_rate_proxy(quant_cache.get("fp32_weight"), weight_qparams),
        "activation_qmin": "" if activation_qparams is None else activation_qparams.get("qmin", ""),
        "activation_qmax": "" if activation_qparams is None else activation_qparams.get("qmax", ""),
        "activation_symmetric": "" if activation_qparams is None else activation_qparams.get("symmetric", ""),
        "activation_granularity": "" if activation_qparams is None else activation_qparams.get("granularity", ""),
        "activation_channel_axis": "" if activation_qparams is None else activation_qparams.get("channel_axis", ""),
        "activation_min": observer_summary.get("min", ""),
        "activation_max": observer_summary.get("max", ""),
        "activation_abs_max": observer_summary.get("abs_max", ""),
        "activation_histogram_built": observer_summary.get("histogram_built", ""),
        "activation_histogram_bins": observer_summary.get("histogram_bins", ""),
        "activation_histogram_num_elements": observer_summary.get("histogram_num_elements", ""),
        "activation_percentiles": observer_summary.get("percentiles", {}),
        "activation_abs_percentiles": observer_summary.get("abs_percentiles", {}),
        "observer_num_batches": observer_summary.get("num_batches", ""),
        "observer_num_samples": observer_summary.get("num_samples", ""),
        "observer_range_num_batches": observer_summary.get("range_num_batches", ""),
        "observer_range_num_samples": observer_summary.get("range_num_samples", ""),
        "observer_histogram_num_batches": observer_summary.get("histogram_num_batches", ""),
        "observer_histogram_num_samples": observer_summary.get("histogram_num_samples", ""),
        "calibration_sample_count": observer_summary.get("range_num_samples", ""),
        "controller_observer_backend": controller_summary.get("observer_backend", ""),
        "controller_observer_num_updates": controller_summary.get("observer_num_updates", ""),
        "controller_observer_num_values": controller_summary.get("observer_num_values", ""),
        "activation_clipping_rate_runtime": quant_cache.get("activation_clipping_rate_runtime", ""),
        "activation_fp32_shape": tuple(quant_cache.get("activation_fp32_shape", ())),
        "activation_quant_shape": tuple(quant_cache.get("activation_quant_shape", ())),
        "projection_fp32_shape": tuple(quant_cache.get("projection_fp32_shape", ())),
        "projection_quant_shape": tuple(quant_cache.get("projection_quant_shape", ())),
        "observer_summary": observer_summary,
        "weight_observer_summary": weight_observer_summary,
        "controller_summary": controller_summary,
    }
    row.update(_tensor_summary("weight_scale", None if weight_qparams is None else weight_qparams.get("scale")))
    row.update(_tensor_summary("weight_zero_point", None if weight_qparams is None else weight_qparams.get("zero_point")))
    row.update(_tensor_summary("activation_scale", None if activation_qparams is None else activation_qparams.get("scale")))
    row.update(_tensor_summary("activation_zero_point", None if activation_qparams is None else activation_qparams.get("zero_point")))
    row.update(clip_stats)
    return row


def _row_extras(
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    qparams_path: Path,
    embedding_path: Path,
    artifact_path: Path,
    quant_cache: Mapping[str, Any],
) -> dict[str, Any]:
    output_stats = quant_cache["output_stats"]
    clipping_stats = _vector_stats("activation_clipping_rate_by_sample", quant_cache.get("activation_clipping_rate_by_sample"))
    return {
        "target_name": _target_name(layer_idx, module_name),
        "quant_mode": _mode_label(quant_mode),
        "qparams_path": str(qparams_path),
        "artifact_path": str(artifact_path),
        "embedding_path": str(embedding_path),
        "output_num_elements": output_stats.get("num_elements", ""),
        "output_mean": output_stats.get("mean", ""),
        "output_std": output_stats.get("std", ""),
        "output_max_abs": output_stats.get("max_abs", ""),
        "activation_fp32_shape": tuple(quant_cache.get("activation_fp32_shape", ())),
        "activation_quant_shape": tuple(quant_cache.get("activation_quant_shape", ())),
        "projection_fp32_shape": tuple(quant_cache.get("projection_fp32_shape", ())),
        "projection_quant_shape": tuple(quant_cache.get("projection_quant_shape", ())),
        "activation_layout_info": quant_cache.get("activation_layout_info", {}),
        "projection_layout_info": quant_cache.get("projection_layout_info", {}),
        "input_oer_activation_kind": quant_cache.get("input_oer_activation_kind", "input_activation"),
        "input_oer_oracle_source": quant_cache.get("input_oer_oracle_source", "oer_oracle_eval.pt.input.topk_scalar_indices"),
        "projection_oer_activation_kind": quant_cache.get("projection_oer_activation_kind", "linear_projection_output"),
        "projection_oer_oracle_source": quant_cache.get("projection_oer_oracle_source", "oer_oracle_eval.pt.output.topk_scalar_indices"),
        "activation_clipping_rate_runtime": quant_cache.get("activation_clipping_rate_runtime", ""),
        "activation_clipping_rate_by_sample_mean": clipping_stats["activation_clipping_rate_by_sample_mean"],
        "activation_clipping_rate_by_sample_std": clipping_stats["activation_clipping_rate_by_sample_std"],
        "activation_clipping_rate_by_sample_max": clipping_stats["activation_clipping_rate_by_sample_max"],
        "weight_qparams_enabled": bool(_mode_uses_weight(quant_mode)),
        "activation_qparams_enabled": bool(_mode_uses_activation(quant_mode)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value



def _target_stem(layer_idx: int, module_name: str) -> str:
    return f"layer_{int(layer_idx):02d}_{module_name}"


def _mode_stem(layer_idx: int, module_name: str, quant_mode: str) -> str:
    return f"{_target_stem(layer_idx, module_name)}_{_mode_slug(quant_mode)}"


def _qparams_path(cfg: Any, layer_idx: int, module_name: str) -> Path:
    return _result_dir(cfg) / QPARAMS_DIRNAME / f"{_target_stem(layer_idx, module_name)}_qparams.pt"


def _mode_shard_path(cfg: Any, layer_idx: int, module_name: str, quant_mode: str, table_name: str) -> Path:
    return _result_dir(cfg) / MODE_SHARDS_DIRNAME / f"{_mode_stem(layer_idx, module_name, quant_mode)}.{table_name}.csv"


def _done_marker_path(cfg: Any, layer_idx: int, module_name: str, quant_mode: str) -> Path:
    return _result_dir(cfg) / DONE_DIRNAME / f"{_mode_stem(layer_idx, module_name, quant_mode)}.done.json"


def _table_columns(table_name: str) -> Sequence[str]:
    if table_name == "metric":
        return CSV_COLUMNS
    if table_name == "comparison":
        return COMPARISON_CSV_COLUMNS
    if table_name == "target_detail":
        return TARGET_DETAIL_CSV_COLUMNS
    if table_name == "qparams_summary":
        return QPARAMS_SUMMARY_COLUMNS
    raise ValueError(f"unknown quant-mode table name: {table_name!r}")


def _mode_shard_paths(cfg: Any, layer_idx: int, module_name: str, quant_mode: str) -> dict[str, Path]:
    return {
        "metric": _mode_shard_path(cfg, layer_idx, module_name, quant_mode, "metric"),
        "comparison": _mode_shard_path(cfg, layer_idx, module_name, quant_mode, "comparison"),
        "target_detail": _mode_shard_path(cfg, layer_idx, module_name, quant_mode, "target_detail"),
        "qparams_summary": _mode_shard_path(cfg, layer_idx, module_name, quant_mode, "qparams_summary"),
    }


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [dict(row) for row in reader]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive resume path.
        print(f"[resume] ignore invalid done marker {path}: {exc}")
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path_value: Any) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_identity(path_value: Any) -> dict[str, Any]:
    if path_value is None:
        return {"path": None, "sha256": None}
    path = Path(path_value)
    return {
        "path": str(path.expanduser().resolve(strict=False)),
        "sha256": _sha256_file(path),
    }


def _optional_checkpoint_path(cfg: Any) -> Path | None:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    if isinstance(beats_cfg, Mapping):
        for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT"):
            value = beats_cfg.get(key)
            if value is not None:
                return Path(value)
    for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT"):
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return Path(value)
    return None


def _fp32_reference_signature_payload(reference_state: Mapping[str, Any]) -> dict[str, Any]:
    required = _nested_get(reference_state, "required", {})
    core = _nested_get(required, "core_eval", {})
    required_oer = _nested_get(reference_state, "required_if_enable_oer", {})
    oer = _nested_get(required_oer, "oer_oracle_eval", {})
    optional = _nested_get(reference_state, "optional", {})
    optional_oer = _nested_get(optional, "oer_oracle_eval", {})
    return {
        "required_core_eval": {
            "path": _nested_get(_nested_get(core, "identity", {}), "path", None),
            "sha256": _nested_get(_nested_get(core, "identity", {}), "sha256", None),
            "manifest_sha256": _nested_get(_nested_get(core, "manifest_identity", {}), "sha256", None),
            "storage_format": _nested_get(core, "storage_format", None),
            "eval_csv_sha256": _nested_get(core, "eval_csv_sha256", None),
            "checkpoint_sha256": _nested_get(core, "checkpoint_sha256", None),
            "sample_order_hash": _nested_get(core, "sample_order_hash", None),
            "sample_count": _nested_get(core, "sample_count", None),
        },
        "required_if_enable_oer": {
            "path": _nested_get(_nested_get(oer, "identity", {}), "path", None),
            "sha256": _nested_get(_nested_get(oer, "identity", {}), "sha256", None),
            "manifest_sha256": _nested_get(_nested_get(oer, "manifest_identity", {}), "sha256", None),
            "storage_format": _nested_get(oer, "storage_format", None),
            "sample_order_hash": _nested_get(oer, "sample_order_hash", None),
            "topk_ratio": _nested_get(oer, "topk_ratio", None),
            "activation_positions": _nested_get(oer, "activation_positions", None),
        },
        "optional_oer_oracle_eval": {
            "status": _nested_get(optional_oer, "status", None),
            "path": _nested_get(optional_oer, "path", None),
            "exists": _nested_get(optional_oer, "exists", None),
            "manifest_sha256": _nested_get(_nested_get(optional_oer, "manifest_identity", {}), "sha256", None),
        },
    }


def _experiment_signature_payload(
    cfg: Any,
    target_modules: Sequence[tuple[int, str]],
    *,
    fp32_reference_state: Mapping[str, Any],
) -> dict[str, Any]:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": "ffn_quant_mode_ablation",
        "dataset_name": _dataset_name(cfg),
        "active_dataset": _cfg_get(cfg, "ACTIVE_DATASET", None),
        "checkpoint": _path_identity(_optional_checkpoint_path(cfg)),
        "calibration_csv": _path_identity(_split_csv(cfg, "calibration")),
        "eval_csv": _path_identity(_split_csv(cfg, "eval")),
        "quant": _json_safe(_quant_cfg(cfg)),
        "ablation": _json_safe(_ablation_cfg(cfg)),
        "audio": _json_safe(_cfg_get(cfg, "AUDIO", {})),
        "amp": bool(beats_cfg.get("USE_AMP", False)) if isinstance(beats_cfg, Mapping) else False,
        "enable_oer_metrics": bool(_enable_oer_metrics(cfg)),
        "oer_top_ratio": float(_oer_top_ratio(cfg)),
        "quant_modes": list(QUANT_MODES),
        "fp32_reference": _fp32_reference_signature_payload(fp32_reference_state),
        "target_modules": [
            {"layer_idx": int(layer_idx), "module_name": str(module_name)}
            for layer_idx, module_name in target_modules
        ],
        "save_full_activation": False,
    }

def _experiment_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_protocol_matches(payload: Mapping[str, Any], protocol_version: str, experiment_signature: str) -> bool:
    return (
        str(payload.get("protocol_version", "")) == str(protocol_version)
        and str(payload.get("experiment_signature", "")) == str(experiment_signature)
    )


def _mode_done_valid(
    cfg: Any,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    protocol_version: str,
    experiment_signature: str,
) -> bool:
    payload = _read_json(_done_marker_path(cfg, layer_idx, module_name, quant_mode))
    if payload is None or not _cache_protocol_matches(payload, protocol_version, experiment_signature):
        return False
    target = payload.get("target", {})
    if int(_nested_get(target, "layer_idx", -1)) != int(layer_idx):
        return False
    if str(_nested_get(target, "module_name", "")) != str(module_name):
        return False
    if str(payload.get("quant_mode", "")) != _mode_label(quant_mode):
        return False
    for shard_path in _mode_shard_paths(cfg, layer_idx, module_name, quant_mode).values():
        if not shard_path.exists() or shard_path.stat().st_size <= 0:
            return False
    qparams_path = _qparams_path(cfg, layer_idx, module_name)
    if not qparams_path.exists() or qparams_path.stat().st_size <= 0:
        return False
    return True


def _write_mode_done_marker(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    quant_mode: str,
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
    qparams_path: Path,
    embedding_path: Path,
    artifact_path: Path,
    row_counts: Mapping[str, int],
) -> None:
    _write_json_atomic(
        _done_marker_path(cfg, layer_idx, module_name, quant_mode),
        {
            "protocol_version": protocol_version,
            "experiment_signature": experiment_signature,
            "experiment_signature_payload": signature_payload,
            "target": {"layer_idx": int(layer_idx), "module_name": str(module_name)},
            "quant_mode": _mode_label(quant_mode),
            "qparams_path": qparams_path,
            "embedding_path": embedding_path,
            "artifact_path": artifact_path,
            "shards": _mode_shard_paths(cfg, layer_idx, module_name, quant_mode),
            "row_counts": dict(row_counts),
        },
    )

def _metric_specs_manifest() -> dict[str, Any]:
    def encode(spec: Any) -> dict[str, Any]:
        if isinstance(spec, Mapping):
            return {
                "metric_direction": str(spec.get("metric_direction", "")),
                "aggregation_unit": str(spec.get("aggregation_unit", "")),
                "aggregation_fn": str(spec.get("aggregation_fn", "")),
                "metric_role": str(spec.get("metric_role", "")),
            }
        return {
            "metric_direction": str(spec.metric_direction),
            "aggregation_unit": str(spec.aggregation_unit),
            "aggregation_fn": str(spec.aggregation_fn),
            "metric_role": str(spec.metric_role),
        }

    return {
        "group_metrics": {name: encode(spec) for name, spec in QUANT_MODE_GROUP_METRIC_SPECS.items()},
        "target_detail_metrics": {name: encode(spec) for name, spec in QUANT_MODE_TARGET_DETAIL_SPECS.items()},
    }


def _write_experiment_manifest(
    cfg: Any,
    *,
    manifest_path: Path,
    csv_path: Path,
    comparison_csv_path: Path,
    target_detail_csv_path: Path,
    qparams_summary_csv_path: Path,
    target_modules: Sequence[tuple[int, str]],
    fp32_reference_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    target_detail_rows: Sequence[Mapping[str, Any]],
    qparams_summary_rows: Sequence[Mapping[str, Any]],
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> None:
    manifest = {
        "experiment_name": "ffn_quant_mode_ablation",
        "protocol_version": protocol_version,
        "experiment_signature": experiment_signature,
        "experiment_signature_payload": _json_safe(signature_payload),
        "resume_protocol": {
            "unit": "target_module_plus_quant_mode",
            "done_marker_requires": ["metric_shard", "comparison_shard", "target_detail_shard", "qparams_summary_shard", "qparams"],
            "mode_shards_dir": str(_result_dir(cfg) / MODE_SHARDS_DIRNAME),
            "done_dir": str(_result_dir(cfg) / DONE_DIRNAME),
        },
        "output_protocol_version": 2,
        "group_protocol_version": 2,
        "group_protocol_name": "bucket_analysis_group.v2",
        "bucket_field": "bucket",
        "analysis_group_field": "analysis_group",
        "shot_group_rule": "aggregate rows keep shot_group empty; sample rows must use bucket and must not use tail",
        "tail_definition": "tail = medium + few",
        "primary_gap": "tail_vs_many_gap",
        "supplementary_gap": "few_vs_many_gap",
        "quant_modes": list(QUANT_MODES),
        "target_modules": [_target_name(layer_idx, module_name) for layer_idx, module_name in target_modules],
        "target_module_scope": list(TARGET_MODULE_NAMES),
        "qparams_source": "single W4A4 two-pass calibration per target, reused by W4-only/A4-only/W4A4 according to mode",
        "fisher_normalization": "raw_scatter_ratio",
        "fisher_definition": "raw_between_within_scatter",
        "normalize_fisher_by_dof": False,
        "csv_path": str(csv_path),
        "comparison_summary_csv_path": str(comparison_csv_path),
        "target_detail_csv_path": str(target_detail_csv_path),
        "qparams_summary_csv_path": str(qparams_summary_csv_path),
        "metric_specs": _metric_specs_manifest(),
        "fp32_reference_cache_path": str(_fp32_core_eval_path(cfg)),
        "fp32_reference_core_path": str(_fp32_core_eval_path(cfg)),
        "fp32_reference_oer_oracle_path": str(_fp32_oer_oracle_path(cfg)),
        "enable_oer_metrics": bool(_enable_oer_metrics(cfg)),
        "fp32_reference_manifest": _json_safe(fp32_reference_manifest),
        "row_counts": {
            "metric_rows": int(len(rows)),
            "comparison_rows": int(len(comparison_rows)),
            "target_detail_rows": int(len(target_detail_rows)),
            "qparams_summary_rows": int(len(qparams_summary_rows)),
        },
        "oer_feature_layout": OER_FEATURE_LAYOUT,
        "input_oer_activation_kind": "input_activation",
        "input_oer_oracle_source": "oer_oracle_eval.pt.input.topk_scalar_indices" if _enable_oer_metrics(cfg) else "disabled",
        "projection_oer_activation_kind": "linear_projection_output",
        "projection_oer_oracle_source": "oer_oracle_eval.pt.output.topk_scalar_indices" if _enable_oer_metrics(cfg) else "disabled",
        "projection_oer_top_ratio": float(_oer_top_ratio(cfg)),
        "w4_only_activation_metrics_note": "W4-only disables activation fake quant; activation/input-OER metrics are expected to be near identity and are diagnostic only for weight-only runs.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def run_single_target(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    calibration_loader: DataLoader,
    eval_loader: DataLoader,
    fp32_reference_cache: Mapping[str, Any],
    protocol_version: str,
    experiment_signature: str,
    signature_payload: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    embedding_cache = fp32_reference_cache["embedding_cache"]
    enable_oer_metrics = bool(fp32_reference_cache.get("enable_oer_metrics", True))
    target_oracle = (
        _load_target_oer_oracle(fp32_reference_cache, layer_idx=int(layer_idx), module_name=module_name)
        if enable_oer_metrics
        else None
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_comparison_rows: list[dict[str, Any]] = []
    all_target_detail_rows: list[dict[str, Any]] = []
    all_qparams_summary_rows: list[dict[str, Any]] = []
    qparams_path = _qparams_path(cfg, int(layer_idx), module_name)
    qparams: dict[str, Any] | None = None

    for quant_mode in QUANT_MODES:
        shard_paths = _mode_shard_paths(cfg, int(layer_idx), module_name, quant_mode)
        if _mode_done_valid(cfg, int(layer_idx), module_name, quant_mode, protocol_version, experiment_signature):
            all_metric_rows.extend(_read_csv_rows(shard_paths["metric"]))
            all_comparison_rows.extend(_read_csv_rows(shard_paths["comparison"]))
            all_target_detail_rows.extend(_read_csv_rows(shard_paths["target_detail"]))
            all_qparams_summary_rows.extend(_read_csv_rows(shard_paths["qparams_summary"]))
            print(f"[resume] skip completed quant-mode target {_target_name(layer_idx, module_name)} {_mode_label(quant_mode)}")
            continue

        if qparams is None:
            qparams = _load_qparams(qparams_path, protocol_version, experiment_signature)
            if qparams is None:
                qparams = calibrate_target_module(cfg, int(layer_idx), module_name, calibration_loader)
                qparams_path = _save_qparams(
                    cfg,
                    int(layer_idx),
                    module_name,
                    qparams,
                    protocol_version=protocol_version,
                    experiment_signature=experiment_signature,
                    signature_payload=signature_payload,
                )

        quant_cache = run_single_quant_mode(
            cfg,
            layer_idx=int(layer_idx),
            module_name=module_name,
            quant_mode=quant_mode,
            qparams=qparams,
            eval_loader=eval_loader,
            oer_oracle=target_oracle,
        )
        _assert_same_eval_order(embedding_cache, quant_cache)
        metric_tables = compute_quant_mode_metric_tables_from_sample_metrics(
            fp32_embeddings=embedding_cache["embedding"],
            quant_embeddings=quant_cache["embedding"],
            label_id=embedding_cache["label_id"],
            bucket=embedding_cache["bucket"],
            shot_group=embedding_cache.get("shot_group"),
            sample_id=embedding_cache["sample_id"],
            quant_mode=_mode_label(quant_mode),
            layer_idx=int(layer_idx),
            module_name=module_name,
            sample_metrics=quant_cache["sample_metrics"],
            fp32_weight=quant_cache["fp32_weight"],
            quant_weight=quant_cache["quant_weight"],
            weight_qparams=qparams["weight"] if _mode_uses_weight(quant_mode) else None,
            normalize_fisher_by_dof=False,
        )
        embedding_path, artifact_path = _save_mode_artifacts(
            cfg,
            layer_idx=int(layer_idx),
            module_name=module_name,
            quant_mode=quant_mode,
            quant_cache=quant_cache,
            metric_rows=metric_tables["metric_rows"],
            comparison_rows=metric_tables["comparison_rows"],
            target_detail_rows=metric_tables["target_detail_rows"],
            qparams_path=qparams_path,
        )
        extras = _row_extras(
            layer_idx=int(layer_idx),
            module_name=module_name,
            quant_mode=quant_mode,
            qparams_path=qparams_path,
            embedding_path=embedding_path,
            artifact_path=artifact_path,
            quant_cache=quant_cache,
        )
        mode_metric_rows: list[dict[str, Any]] = []
        mode_comparison_rows: list[dict[str, Any]] = []
        mode_target_detail_rows: list[dict[str, Any]] = []
        mode_qparams_summary_rows = [
            _build_qparams_summary_row(
                layer_idx=int(layer_idx),
                module_name=module_name,
                quant_mode=quant_mode,
                qparams=qparams,
                qparams_path=qparams_path,
                quant_cache=quant_cache,
            )
        ]
        for table_name, destination in (
            ("metric_rows", mode_metric_rows),
            ("comparison_rows", mode_comparison_rows),
            ("target_detail_rows", mode_target_detail_rows),
        ):
            for row in metric_tables[table_name]:
                merged = dict(row)
                merged.update(extras)
                destination.append(merged)

        _write_rows(shard_paths["metric"], mode_metric_rows, CSV_COLUMNS)
        _write_rows(shard_paths["comparison"], mode_comparison_rows, COMPARISON_CSV_COLUMNS)
        _write_rows(shard_paths["target_detail"], mode_target_detail_rows, TARGET_DETAIL_CSV_COLUMNS)
        _write_rows(shard_paths["qparams_summary"], mode_qparams_summary_rows, QPARAMS_SUMMARY_COLUMNS)
        _write_mode_done_marker(
            cfg,
            layer_idx=int(layer_idx),
            module_name=module_name,
            quant_mode=quant_mode,
            protocol_version=protocol_version,
            experiment_signature=experiment_signature,
            signature_payload=signature_payload,
            qparams_path=qparams_path,
            embedding_path=embedding_path,
            artifact_path=artifact_path,
            row_counts={
                "metric": len(mode_metric_rows),
                "comparison": len(mode_comparison_rows),
                "target_detail": len(mode_target_detail_rows),
                "qparams_summary": len(mode_qparams_summary_rows),
            },
        )
        all_metric_rows.extend(mode_metric_rows)
        all_comparison_rows.extend(mode_comparison_rows)
        all_target_detail_rows.extend(mode_target_detail_rows)
        all_qparams_summary_rows.extend(mode_qparams_summary_rows)
        del quant_cache, metric_tables
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "metric_rows": all_metric_rows,
        "comparison_rows": all_comparison_rows,
        "target_detail_rows": all_target_detail_rows,
        "qparams_summary_rows": all_qparams_summary_rows,
    }


def main(cfg: Any) -> dict[str, Any]:
    """Quant mode ablation helper."""

    seed_everything = _cfg_get(cfg, "seed_everything", None)
    if callable(seed_everything):
        seed_everything(_seed(cfg))

    output_dir = _result_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_FILENAME
    comparison_csv_path = output_dir / COMPARISON_SUMMARY_CSV_FILENAME
    target_detail_csv_path = output_dir / TARGET_DETAIL_CSV_FILENAME
    qparams_summary_csv_path = output_dir / QPARAMS_SUMMARY_CSV_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    calibration_loader = _build_audio_loader(cfg, name="calibration", mode="calibration")
    eval_loader = _build_audio_loader(cfg, name="eval", mode="eval")

    probe_model = _restore_clean_model(cfg)
    target_modules = _target_modules(cfg, probe_model)
    del probe_model

    fp32_reference_cache = _build_reference_cache(cfg, eval_loader, target_modules)
    reference_manifest = fp32_reference_cache.get("manifest", {})
    reference_state = fp32_reference_cache.get("reference_state", reference_manifest)

    signature_payload = _experiment_signature_payload(
        cfg,
        target_modules,
        fp32_reference_state=reference_state if isinstance(reference_state, Mapping) else {},
    )
    protocol_version = PROTOCOL_VERSION
    experiment_signature = _experiment_signature(signature_payload)
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    target_detail_rows: list[dict[str, Any]] = []
    qparams_summary_rows: list[dict[str, Any]] = []
    _write_rows(csv_path, metric_rows, CSV_COLUMNS)
    _write_rows(comparison_csv_path, comparison_rows, COMPARISON_CSV_COLUMNS)
    _write_rows(target_detail_csv_path, target_detail_rows, TARGET_DETAIL_CSV_COLUMNS)
    _write_rows(qparams_summary_csv_path, qparams_summary_rows, QPARAMS_SUMMARY_COLUMNS)
    for layer_idx, module_name in target_modules:
        target_tables = run_single_target(
            cfg,
            layer_idx=int(layer_idx),
            module_name=module_name,
            calibration_loader=calibration_loader,
            eval_loader=eval_loader,
            fp32_reference_cache=fp32_reference_cache,
            protocol_version=protocol_version,
            experiment_signature=experiment_signature,
            signature_payload=signature_payload,
        )
        metric_rows.extend(target_tables["metric_rows"])
        comparison_rows.extend(target_tables["comparison_rows"])
        target_detail_rows.extend(target_tables["target_detail_rows"])
        qparams_summary_rows.extend(target_tables["qparams_summary_rows"])
        _write_rows(csv_path, metric_rows, CSV_COLUMNS)
        _write_rows(comparison_csv_path, comparison_rows, COMPARISON_CSV_COLUMNS)
        _write_rows(target_detail_csv_path, target_detail_rows, TARGET_DETAIL_CSV_COLUMNS)
        _write_rows(qparams_summary_csv_path, qparams_summary_rows, QPARAMS_SUMMARY_COLUMNS)
        _write_experiment_manifest(
            cfg,
            manifest_path=manifest_path,
            csv_path=csv_path,
            comparison_csv_path=comparison_csv_path,
            target_detail_csv_path=target_detail_csv_path,
            qparams_summary_csv_path=qparams_summary_csv_path,
            target_modules=target_modules,
            fp32_reference_manifest=reference_manifest if isinstance(reference_manifest, Mapping) else {},
            rows=metric_rows,
            comparison_rows=comparison_rows,
            target_detail_rows=target_detail_rows,
            qparams_summary_rows=qparams_summary_rows,
            protocol_version=protocol_version,
            experiment_signature=experiment_signature,
            signature_payload=signature_payload,
        )

    import pandas as pd

    metric_df = pd.read_csv(csv_path)
    comparison_df = pd.read_csv(comparison_csv_path)
    plot_paths = plot_all_quant_mode_plots(metric_df, output_dir, comparison_df=comparison_df)
    _write_experiment_manifest(
        cfg,
        manifest_path=manifest_path,
        csv_path=csv_path,
        comparison_csv_path=comparison_csv_path,
        target_detail_csv_path=target_detail_csv_path,
        qparams_summary_csv_path=qparams_summary_csv_path,
        target_modules=target_modules,
        fp32_reference_manifest=reference_manifest if isinstance(reference_manifest, Mapping) else {},
        rows=metric_rows,
        comparison_rows=comparison_rows,
        target_detail_rows=target_detail_rows,
        qparams_summary_rows=qparams_summary_rows,
        protocol_version=protocol_version,
        experiment_signature=experiment_signature,
        signature_payload=signature_payload,
    )
    return {
        "result_dir": output_dir,
        "csv_path": csv_path,
        "comparison_summary_csv_path": comparison_csv_path,
        "target_detail_csv_path": target_detail_csv_path,
        "qparams_summary_csv_path": qparams_summary_csv_path,
        "manifest_path": manifest_path,
        "plot_paths": plot_paths,
        "reference_cache_path": _fp32_core_eval_path(cfg),
        "oer_oracle_path": _fp32_oer_oracle_path(cfg),
        "enable_oer_metrics": bool(_enable_oer_metrics(cfg)),
        "num_targets": len(target_modules),
        "num_quant_modes": len(QUANT_MODES),
        "num_rows": len(metric_rows),
        "num_comparison_rows": len(comparison_rows),
        "num_target_detail_rows": len(target_detail_rows),
        "num_qparams_summary_rows": len(qparams_summary_rows),
        "protocol_version": protocol_version,
        "experiment_signature": experiment_signature,
    }


if __name__ == "__main__":
    import config as project_config

    summary = main(project_config)
    print(f"FFN Quant Mode Ablation complete: {summary['csv_path']}")



