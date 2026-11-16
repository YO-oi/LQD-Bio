from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from src.export.export_fp32_core_cache import forward_core_outputs
    from src.metrics.metric_config import (
        DEFAULT_EPS,
        DEFAULT_TOKEN_SURVIVAL_MIN_NORM_RETENTION,
    )
    from src.metrics.semantic_geometry_metrics import (
        build_fp32_semantic_geometry_reference,
        build_semantic_geometry_summary_v2_long,
        compute_semantic_geometry_tables,
    )
    from src.metrics.token_survival_metrics import compute_token_survival_rows
    from src.quant.ffn_quant_injector import apply_fixed_qparams_to_quant_module, normalize_quant_mode
    from src.quant.quant_core import QuantizationController, register_layer_padding_mask_hooks
    from src.quant.quant_linear import QuantLinear
    from src.utils.semantic_geometry_cache import (
        DEFAULT_EMBEDDING_DIM,
        build_quant_embedding_cache,
        file_identity,
        qparams_identity_manifest,
        save_embedding_cache,
        stable_json_hash,
    )
    from src.utils.semantic_geometry_mode_runner import (
        activation_observer_config_for_granularity,
        build_eval_loader,
        load_eval_metadata,
        load_qparams,
        qparams_cache_path,
        restore_clean_model,
        run_embedding_mode,
        validate_calibration_provenance,
    )
    from src.utils.group_utils import bucket_from_train_label_n, group_protocol_metadata, read_canonical_bucket
    from src.utils.token_trace_hook import TokenTraceHookManager, TokenTraceRecord
    from src.verify.ffn_quant_mode_ablation import (
        _build_reference_cache as build_quant_mode_reference_cache,
        _load_target_input_oer_oracle as load_target_input_oer_oracle,
        _normalise_oracle_topk_indices as normalise_oracle_topk_indices,
        _oer_metrics_from_topk_indices as oer_metrics_from_topk_indices,
        _relative_l2_per_sample as relative_l2_per_sample,
    )
except ImportError:  # pragma: no cover - direct script fallback.
    from .export_fp32_core_cache import forward_core_outputs
    from ..metrics.metric_config import (
        DEFAULT_EPS,
        DEFAULT_TOKEN_SURVIVAL_MIN_NORM_RETENTION,
    )
    from ..metrics.semantic_geometry_metrics import (
        build_fp32_semantic_geometry_reference,
        build_semantic_geometry_summary_v2_long,
        compute_semantic_geometry_tables,
    )
    from ..metrics.token_survival_metrics import compute_token_survival_rows
    from ..quant.ffn_quant_injector import apply_fixed_qparams_to_quant_module, normalize_quant_mode
    from ..quant.quant_core import QuantizationController, register_layer_padding_mask_hooks
    from ..quant.quant_linear import QuantLinear
    from ..utils.semantic_geometry_cache import (
        DEFAULT_EMBEDDING_DIM,
        build_quant_embedding_cache,
        file_identity,
        qparams_identity_manifest,
        save_embedding_cache,
        stable_json_hash,
    )
    from ..utils.semantic_geometry_mode_runner import (
        activation_observer_config_for_granularity,
        build_eval_loader,
        load_eval_metadata,
        load_qparams,
        qparams_cache_path,
        restore_clean_model,
        run_embedding_mode,
        validate_calibration_provenance,
    )
    from ..utils.group_utils import bucket_from_train_label_n, group_protocol_metadata, read_canonical_bucket
    from ..utils.token_trace_hook import TokenTraceHookManager, TokenTraceRecord
    from ..verify.ffn_quant_mode_ablation import (
        _build_reference_cache as build_quant_mode_reference_cache,
        _load_target_input_oer_oracle as load_target_input_oer_oracle,
        _normalise_oracle_topk_indices as normalise_oracle_topk_indices,
        _oer_metrics_from_topk_indices as oer_metrics_from_topk_indices,
        _relative_l2_per_sample as relative_l2_per_sample,
    )


EXPERIMENT_NAME = "l9_high_energy_boundary_counterfactual"
PROTOCOL_VERSION = "l9_high_energy_boundary_counterfactual.v1"
RUNNER_CFG_NAME = "L9_BOUNDARY_COUNTERFACTUAL"
QUANT_MODE = "W4A4"
ACTIVATION_GRANULARITY = "per_tensor"
WEIGHT_GRANULARITY = "per_output_channel"
TARGET_LAYER_IDX = 9
TARGET_MODULE_NAME = "fc1"
TARGET = (TARGET_LAYER_IDX, TARGET_MODULE_NAME)
TARGET_LINEAR_NAMES = ("fc1", "fc2")
TRACE_SPECS = (
    {"scope": "l9_block_output", "layer_idx": 9, "module_name": "block", "tensor_position": "block_output"},
    {"scope": "l11_final_block_output", "layer_idx": 11, "module_name": "block", "tensor_position": "block_output"},
)
RANGE_MULTIPLIERS = (1.0, 1.25, 1.5, 2.0)
EXPECTED_ACTIVATION_QRANGE = (-7, 7)
EXPECTED_ZERO_POINT = 0.0
GROUP_NAMES = ("overall", "many", "medium", "few", "tail")
DIAGNOSTIC_DELTA_EPS = 1e-6

SUMMARY_FILENAME = "l9_boundary_counterfactual_summary.csv"
DANGER_ZONE_FILENAME = "l9_boundary_danger_zone.csv"
TOKEN_METRICS_FILENAME = "l9_boundary_token_metrics.csv"
GEOMETRY_FILENAME = "l9_boundary_geometry.csv"
TASK_METRICS_FILENAME = "l9_boundary_task_metrics.csv"
MANIFEST_FILENAME = "l9_boundary_manifest.json"
QPARAMS_JSON_FILENAME = "l9_boundary_counterfactual_qparams.json"

DANGER_ZONE_METRICS = (
    "strict_overflow_rate",
    "overflow_energy_proxy",
    "boundary_hit_rate",
    "clipping_rate",
    "near_zero_rate",
    "exact_zero_rate",
    "activation_quant_error",
    "oer_retention",
    "oer_loss",
    "off_oracle_energy_ratio",
    "teacher_topk_relative_distortion",
    "topk_overlap",
    "mean_norm_retention",
)
TOKEN_METRICS = (
    "topk_overlap",
    "topk_survival_rate",
    "token_survival_drop",
    "mean_norm_retention",
    "mean_token_distortion",
    "token_oer_retention",
    "token_oer_loss",
    "embedding_distortion",
)


@dataclass(frozen=True)
class MultiplierSpec:
    k: float

    @property
    def name(self) -> str:
        return f"k={self.k:g}"

    @property
    def slug(self) -> str:
        return _slug_float(self.k)


@dataclass(frozen=True)
class CandidateRunResult:
    spec: MultiplierSpec
    qparams_by_target: dict[tuple[int, str], dict[str, Any]]
    embedding_cache: dict[str, Any]
    danger_zone_rows: list[dict[str, Any]]
    token_metric_rows: list[dict[str, Any]]
    runtime_summary: dict[str, Any]
    cache_path: Path


def multiplier_specs() -> tuple[MultiplierSpec, ...]:
    return tuple(MultiplierSpec(float(k)) for k in RANGE_MULTIPLIERS)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else default


def _runner_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, RUNNER_CFG_NAME, {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    runner = _runner_cfg(cfg)
    if key in runner:
        return runner[key]
    return _cfg_get(cfg, key, default)


def _quant_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "QUANT", {})
    if not isinstance(value, Mapping):
        raise TypeError("cfg.QUANT must be a mapping")
    return value


def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}


def _dataset_root(cfg: Any) -> Path:
    value = _nested_get(_dataset_cfg(cfg), "ROOT", None)
    if value is not None:
        return Path(value)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / "local_data"


def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", _cfg_get(cfg, "DEVICE", None))
    if configured is None:
        raise AttributeError(f"{EXPERIMENT_NAME} requires cfg.DEVICE")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"cfg.DEVICE={configured!r} but CUDA is not available")
    return device


def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


def _metric_eps(cfg: Any) -> float:
    metrics_cfg = _cfg_get(cfg, "METRICS", {})
    configured = metrics_cfg.get("EPS", DEFAULT_EPS) if isinstance(metrics_cfg, Mapping) else DEFAULT_EPS
    return float(_setting_get(cfg, "METRIC_EPS", configured))


def _token_topk_ratio(cfg: Any) -> float:
    configured = _cfg_get(cfg, "TOKEN_TOPK_RATIO", 0.05)
    return float(_setting_get(cfg, "TOKEN_TOPK_RATIO", configured))


def _survival_min_norm_retention(cfg: Any) -> float:
    metrics_cfg = _cfg_get(cfg, "METRICS", {})
    configured = (
        metrics_cfg.get("TOKEN_SURVIVAL_MIN_NORM_RETENTION", DEFAULT_TOKEN_SURVIVAL_MIN_NORM_RETENTION)
        if isinstance(metrics_cfg, Mapping)
        else DEFAULT_TOKEN_SURVIVAL_MIN_NORM_RETENTION
    )
    return float(_setting_get(cfg, "TOKEN_SURVIVAL_MIN_NORM_RETENTION", configured))


def _expected_dim(cfg: Any) -> int:
    return int(_setting_get(cfg, "EXPECTED_EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM))


def _split_csv(cfg: Any, split: str) -> Path:
    key = f"{split.upper()}_CSV"
    configured = _setting_get(cfg, key, None)
    if configured is not None:
        return Path(configured)
    dataset_value = _nested_get(_dataset_cfg(cfg), key, None)
    if dataset_value is not None:
        return Path(dataset_value)
    raise AttributeError(f"Cannot resolve {key}")


def _checkpoint_path(cfg: Any) -> Path | None:
    for key in ("CHECKPOINT_PATH", "CKPT_PATH", "MODEL_CHECKPOINT", "MODEL_CHECKPOINT_PATH", "BEATS_CHECKPOINT"):
        value = _setting_get(cfg, key, _cfg_get(cfg, key, None))
        if value is not None:
            return Path(value)
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    if isinstance(beats_cfg, Mapping):
        for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT_PATH", "MODEL_CHECKPOINT", "MODEL_CHECKPOINT_PATH"):
            value = beats_cfg.get(key)
            if value is not None:
                return Path(value)
    return None


def _label_mapping_path(cfg: Any) -> Path | None:
    value = _setting_get(cfg, "LABEL_MAPPING", None)
    if value is not None:
        return Path(value)
    dataset_value = _nested_get(_dataset_cfg(cfg), "LABEL_MAPPING", None)
    return None if dataset_value is None else Path(dataset_value)


def _result_dir(cfg: Any, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return _dataset_root(cfg) / "export" / EXPERIMENT_NAME


def _activation_qrange(cfg: Any) -> tuple[int, int]:
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    activation_cfg = activation_cfg if isinstance(activation_cfg, Mapping) else {}
    return int(activation_cfg.get("QMIN", -7)), int(activation_cfg.get("QMAX", 7))


def _assert_quant_injection_qrange(cfg: Any) -> None:
    qrange = _activation_qrange(cfg)
    if qrange != EXPECTED_ACTIVATION_QRANGE:
        raise RuntimeError(
            f"{EXPERIMENT_NAME} is locked to Quant Injection activation qrange "
            f"{EXPECTED_ACTIVATION_QRANGE}; current activation qrange is {qrange}"
        )


def _target_key(layer_idx: int, module_name: str) -> tuple[int, str]:
    return (int(layer_idx), str(module_name))


def _target_name(layer_idx: int, module_name: str) -> str:
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _ffn_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    return model.encoder.layers[int(layer_idx)]


def discover_ffn_targets(model: nn.Module, cfg: Any) -> tuple[tuple[int, str], ...]:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise AttributeError("model.encoder.layers does not exist")
    target_names = tuple(_quant_cfg(cfg).get("TARGET_LINEAR_NAMES", TARGET_LINEAR_NAMES))
    if target_names != TARGET_LINEAR_NAMES:
        raise ValueError(f"Expected QUANT.TARGET_LINEAR_NAMES={TARGET_LINEAR_NAMES}, got {target_names!r}")
    targets: list[tuple[int, str]] = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        for module_name in target_names:
            module = getattr(layer, str(module_name), None)
            if isinstance(module, nn.Linear):
                targets.append((int(layer_idx), str(module_name)))
    if TARGET not in targets:
        raise RuntimeError(f"Required target {_target_name(*TARGET)} was not found")
    if not any(int(item["layer_idx"]) == 11 for item in TRACE_SPECS):
        raise RuntimeError("TRACE_SPECS must include the final L11 block output diagnostic")
    return tuple(targets)


def _clone_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return value


def _clone_qparams(qparams: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _clone_value(value) for key, value in qparams.items()}


def _as_float_tensor(value: Any, *, name: str) -> Tensor:
    tensor = value.detach().cpu().float() if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if int(tensor.numel()) <= 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN or Inf")
    return tensor


def _as_scalar_float(value: Any, *, name: str) -> float:
    tensor = _as_float_tensor(value, name=name).reshape(-1)
    if int(tensor.numel()) != 1:
        raise ValueError(f"{name} must be a per-tensor scalar, got shape={tuple(_as_float_tensor(value, name=name).shape)}")
    result = float(tensor[0].item())
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {result}")
    return result


def _zero_point_is_zero(value: Any) -> bool:
    tensor = _as_float_tensor(value, name="zero_point").reshape(-1)
    return bool(torch.allclose(tensor, torch.zeros_like(tensor), atol=0.0, rtol=0.0))


def _qparams_root(cfg: Any, qparams_root: str | Path | None = None) -> Path | None:
    if qparams_root is not None:
        return Path(qparams_root)
    configured = _setting_get(cfg, "BASELINE_QPARAMS_ROOT", _setting_get(cfg, "QPARAMS_ROOT", None))
    return None if configured is None else Path(configured)


def _baseline_qparams_path(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    qparams_root: str | Path | None = None,
) -> Path:
    root = _qparams_root(cfg, qparams_root)
    if root is not None:
        return root / f"layer_{int(layer_idx):02d}_{module_name}_{ACTIVATION_GRANULARITY}_qparams.pt"
    return qparams_cache_path(
        cfg,
        layer_idx=int(layer_idx),
        module_name=str(module_name),
        activation_granularity=ACTIVATION_GRANULARITY,
    )


def _validate_baseline_qparams(qparams: Mapping[str, Any], cfg: Any, *, layer_idx: int, module_name: str) -> None:
    target_name = _target_name(layer_idx, module_name)
    if str(qparams.get("target_name", target_name)) != target_name:
        raise ValueError(f"{target_name}: qparams target mismatch: {qparams.get('target_name')!r}")
    if str(qparams.get("activation_granularity", ACTIVATION_GRANULARITY)) != ACTIVATION_GRANULARITY:
        raise ValueError(f"{target_name}: activation_granularity must be {ACTIVATION_GRANULARITY}")

    activation = qparams.get("activation")
    weight = qparams.get("weight")
    if not isinstance(activation, Mapping):
        raise KeyError(f"{target_name}: qparams missing activation section")
    if not isinstance(weight, Mapping):
        raise KeyError(f"{target_name}: qparams missing weight section")

    if (int(activation.get("qmin")), int(activation.get("qmax"))) != EXPECTED_ACTIVATION_QRANGE:
        raise RuntimeError(
            f"{target_name}: activation qrange must be {EXPECTED_ACTIVATION_QRANGE}, "
            f"got {(activation.get('qmin'), activation.get('qmax'))}"
        )
    if not bool(activation.get("symmetric", True)):
        raise RuntimeError(f"{target_name}: activation qparams must be symmetric")
    if str(activation.get("granularity", ACTIVATION_GRANULARITY)) != ACTIVATION_GRANULARITY:
        raise RuntimeError(f"{target_name}: activation section granularity must be {ACTIVATION_GRANULARITY}")
    if not _zero_point_is_zero(activation.get("zero_point", 0.0)):
        raise RuntimeError(f"{target_name}: activation zero_point must remain 0")
    _as_scalar_float(activation.get("scale"), name=f"{target_name}.activation.scale")

    if (int(weight.get("qmin")), int(weight.get("qmax"))) != EXPECTED_ACTIVATION_QRANGE:
        raise RuntimeError(
            f"{target_name}: weight qrange must be {EXPECTED_ACTIVATION_QRANGE}, "
            f"got {(weight.get('qmin'), weight.get('qmax'))}"
        )
    if not bool(weight.get("symmetric", True)):
        raise RuntimeError(f"{target_name}: weight qparams must be symmetric")
    if int(weight.get("channel_axis", 0)) != 0:
        raise RuntimeError(f"{target_name}: weight channel_axis must be 0 for per-output-channel W4")
    weight_granularity = str(weight.get("granularity", "per_channel"))
    if weight_granularity not in {"per_channel", "per_output_channel"}:
        raise RuntimeError(f"{target_name}: weight granularity must be per-output-channel/per_channel")

    validate_calibration_provenance(
        qparams,
        cfg,
        activation_granularity=ACTIVATION_GRANULARITY,
        require_activation=True,
        context=f"{target_name} baseline qparams",
    )


def load_baseline_qparams_by_target(
    cfg: Any,
    *,
    target_modules: Sequence[tuple[int, str]],
    qparams_root: str | Path | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    _assert_quant_injection_qrange(cfg)
    loaded: dict[tuple[int, str], dict[str, Any]] = {}
    missing: list[str] = []
    for layer_idx, module_name in target_modules:
        path = _baseline_qparams_path(
            cfg,
            layer_idx=int(layer_idx),
            module_name=str(module_name),
            qparams_root=qparams_root,
        )
        if not path.exists():
            missing.append(str(path))
            continue
        qparams = load_qparams(path)
        qparams["qparams_path"] = str(path)
        _validate_baseline_qparams(qparams, cfg, layer_idx=int(layer_idx), module_name=str(module_name))
        loaded[_target_key(layer_idx, module_name)] = qparams
    if missing:
        preview = "\n".join(missing[:8])
        raise FileNotFoundError(
            "Missing frozen per-tensor W4A4 qparams. This diagnostic will not recalibrate automatically.\n"
            f"First missing paths:\n{preview}"
        )
    return loaded


def _multiplied_activation_qparams(
    activation: Mapping[str, Any],
    *,
    multiplier: float,
) -> dict[str, Any]:
    result = {key: _clone_value(value) for key, value in activation.items()}
    scale = _as_float_tensor(result["scale"], name="L9.fc1 activation.scale")
    result["scale"] = scale * float(multiplier)
    result["zero_point"] = _clone_value(activation.get("zero_point", torch.tensor(0.0)))
    result["qmin"] = EXPECTED_ACTIVATION_QRANGE[0]
    result["qmax"] = EXPECTED_ACTIVATION_QRANGE[1]
    result["symmetric"] = True
    result["granularity"] = ACTIVATION_GRANULARITY
    result["range_multiplier"] = float(multiplier)
    result["method"] = "frozen_per_tensor_scale_multiplier"
    return result


def build_multiplier_qparams_by_target(
    baseline_qparams_by_target: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    spec: MultiplierSpec,
) -> dict[tuple[int, str], dict[str, Any]]:
    if _target_key(*TARGET) not in baseline_qparams_by_target:
        raise KeyError(f"baseline qparams are missing {_target_name(*TARGET)}")
    result = {key: _clone_qparams(qparams) for key, qparams in baseline_qparams_by_target.items()}
    target_qparams = result[_target_key(*TARGET)]
    activation = target_qparams.get("activation")
    if not isinstance(activation, Mapping):
        raise KeyError(f"{_target_name(*TARGET)} qparams missing activation section")
    baseline_scale = _as_scalar_float(activation["scale"], name="L9.fc1 baseline activation scale")
    target_qparams["activation"] = _multiplied_activation_qparams(activation, multiplier=spec.k)
    target_qparams["range_multiplier"] = float(spec.k)
    target_qparams["baseline_scale"] = float(baseline_scale)
    target_qparams["counterfactual_scale"] = float(baseline_scale * float(spec.k))
    target_qparams["counterfactual_target"] = _target_name(*TARGET)
    target_qparams["counterfactual_intervention"] = "activation_scale_multiplier_only"
    return result


def _qparams_for_target(
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    *,
    layer_idx: int,
    module_name: str,
) -> dict[str, Any]:
    keys = (
        _target_key(layer_idx, module_name),
        f"{int(layer_idx)}:{module_name}",
        f"layer_{int(layer_idx):02d}_{module_name}",
        _target_name(layer_idx, module_name),
    )
    for key in keys:
        value = qparams_by_target.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    known = ", ".join(str(item) for item in list(qparams_by_target.keys())[:8])
    raise KeyError(f"Missing qparams for {_target_name(layer_idx, module_name)}; known keys: {known}")


def _assert_quant_module_has_fixed_qparams(module: QuantLinear, *, target_name: str) -> None:
    if not bool(getattr(module, "fixed_weight_qparams_enabled", False)):
        raise RuntimeError(f"{target_name}: fixed weight qparams are not enabled")
    if not bool(getattr(module, "qparams_frozen", False)):
        raise RuntimeError(f"{target_name}: activation qparams are not frozen")
    scale = getattr(module, "activation_scale", None)
    if not isinstance(scale, Tensor) or int(scale.numel()) <= 0 or not torch.isfinite(scale).all().item():
        raise RuntimeError(f"{target_name}: activation scale is invalid")


def apply_l9_scale_multiplier_quantization(
    cfg: Any,
    *,
    model: nn.Module,
    target_modules: Sequence[tuple[int, str]],
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
) -> tuple[dict[tuple[int, str], QuantLinear], list[QuantizationController]]:
    controller = QuantizationController(config=_quant_cfg(cfg), mode=normalize_quant_mode(QUANT_MODE, None))
    quant_modules: dict[tuple[int, str], QuantLinear] = {}
    hooked_layers: set[int] = set()
    seen: set[tuple[int, str]] = set()
    activation_observer_config = activation_observer_config_for_granularity(cfg, ACTIVATION_GRANULARITY)

    for layer_idx, module_name in target_modules:
        key = _target_key(layer_idx, module_name)
        if key in seen:
            raise ValueError(f"Duplicate target {_target_name(layer_idx, module_name)}")
        seen.add(key)
        fixed_qparams = _qparams_for_target(qparams_by_target, layer_idx=int(layer_idx), module_name=str(module_name))
        validate_calibration_provenance(
            fixed_qparams,
            cfg,
            activation_granularity=ACTIVATION_GRANULARITY,
            require_activation=True,
            context=f"{_target_name(layer_idx, module_name)} runtime qparams",
        )
        layer = _ffn_layer(model, int(layer_idx))
        original = getattr(layer, str(module_name))
        if isinstance(original, QuantLinear):
            raise RuntimeError(f"{_target_name(layer_idx, module_name)} is already QuantLinear")
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{_target_name(layer_idx, module_name)} must be nn.Linear, got {type(original)!r}")
        wrapped = QuantLinear(
            original,
            quant_config=_quant_cfg(cfg),
            activation_observer_config=activation_observer_config,
            target_name=_target_name(layer_idx, module_name),
        )
        setattr(layer, str(module_name), wrapped)
        apply_fixed_qparams_to_quant_module(
            wrapped,
            mode=QUANT_MODE,
            weight_qparams=fixed_qparams["weight"],
            activation_qparams=fixed_qparams["activation"],
            require_qparams=True,
        )
        if key == _target_key(*TARGET):
            wrapped.set_integer_grid_capture(
                True,
                scope="l9_boundary_counterfactual",
                save_device="cpu",
                detach=True,
                canonical_layout="BTC",
            )
        controller.register_module(_target_name(layer_idx, module_name), wrapped)
        if int(layer_idx) not in hooked_layers:
            register_layer_padding_mask_hooks(layer, controller=controller)
            hooked_layers.add(int(layer_idx))
        controller.qparams_frozen = True
        _assert_quant_module_has_fixed_qparams(wrapped, target_name=_target_name(layer_idx, module_name))
        quant_modules[key] = wrapped

    controller.prepare_quant_forward()
    if _target_key(*TARGET) not in quant_modules:
        raise RuntimeError(f"{_target_name(*TARGET)} was not quantized")
    return quant_modules, [controller]


def _remove_controller_hooks(controllers: Sequence[Any]) -> None:
    for controller in controllers:
        if hasattr(controller, "remove_padding_mask_hooks"):
            controller.remove_padding_mask_hooks()


def load_l9_input_oer_oracle(cfg: Any, *, eval_loader: DataLoader) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        reference_cache = build_quant_mode_reference_cache(cfg, eval_loader, (TARGET,))
        state = reference_cache.get("reference_state", reference_cache.get("manifest", {}))
        state = dict(state) if isinstance(state, Mapping) else {"status": "unknown"}
        if not bool(reference_cache.get("enable_oer_metrics", False)):
            state["status"] = "oer_metrics_disabled"
            return None, state
        oracle = dict(
            load_target_input_oer_oracle(
                reference_cache,
                layer_idx=TARGET_LAYER_IDX,
                module_name=TARGET_MODULE_NAME,
            )
        )
        raw_topk = oracle.get("input_topk_indices", oracle.get("topk_indices", oracle.get("flat_index")))
        oracle["input_topk_indices"] = normalise_oracle_topk_indices(
            raw_topk,
            target_name=_target_name(*TARGET),
            oracle_key="input_topk_indices/topk_indices/flat_index",
        )
        state["status"] = "validated"
        return oracle, state
    except Exception as exc:  # noqa: BLE001 - surfaced as invalid local fidelity rows.
        return None, {
            "status": "invalid",
            "target": _target_name(*TARGET),
            "error": str(exc),
        }


def _batch_label_id(batch: Mapping[str, Any]) -> Tensor:
    for key in ("label_id", "target", "targets", "labels", "label"):
        value = batch.get(key)
        if value is None:
            continue
        if isinstance(value, Tensor):
            tensor = value.detach().cpu().long().reshape(-1)
            if int(tensor.numel()) > 0:
                return tensor
    raise KeyError("eval batch must include tensor label_id/target/labels")


def _batch_sample_ids(batch: Mapping[str, Any], *, batch_size: int) -> list[str]:
    for key in ("sample_id", "sample_ids", "id", "ids"):
        value = batch.get(key)
        if value is None:
            continue
        if isinstance(value, Tensor):
            items = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = list(value)
        else:
            items = [value]
        if len(items) != int(batch_size):
            raise ValueError(f"batch {key} count {len(items)} != batch_size {batch_size}")
        return [str(item) for item in items]
    raise KeyError("eval batch must include sample_id")


def _batch_labels_and_buckets(
    batch: Mapping[str, Any],
    *,
    sample_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[Any], list[str]]:
    label_id = _batch_label_id(batch)
    sample_ids = _batch_sample_ids(batch, batch_size=int(label_id.numel()))
    labels: list[Any] = []
    buckets: list[str] = []
    for idx, sample_id in enumerate(sample_ids):
        if sample_id not in sample_metadata:
            raise KeyError(f"sample_id={sample_id!r} is missing from eval metadata")
        metadata = sample_metadata[sample_id]
        label = metadata.get("label", metadata.get("primary_label", str(int(label_id[idx].item()))))
        bucket = str(metadata.get("bucket", metadata.get("shot_group", ""))).strip().lower()
        if bucket not in {"many", "medium", "few"}:
            raise ValueError(f"sample_id={sample_id!r} has invalid bucket={bucket!r}")
        labels.append(label)
        buckets.append(bucket)
    return sample_ids, labels, buckets


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        str(key): value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _as_btc_tensor(value: Any, *, batch_size: int, target_name: str, tensor_name: str) -> Tensor:
    tensor = value.detach().cpu().float() if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"{target_name}: {tensor_name} must be canonical [B,T,C], got shape={tuple(tensor.shape)}")
    if int(tensor.shape[0]) != int(batch_size):
        raise ValueError(f"{target_name}: {tensor_name} batch mismatch {int(tensor.shape[0])} vs {batch_size}")
    return tensor.contiguous()


def _expand_valid_mask(valid_mask: Any, value: Tensor, *, target_name: str) -> Tensor:
    if valid_mask is None:
        return torch.ones_like(value, dtype=torch.bool)
    valid = valid_mask.detach().cpu().bool() if isinstance(valid_mask, Tensor) else torch.as_tensor(valid_mask, dtype=torch.bool)
    if tuple(valid.shape) == tuple(value.shape):
        return valid.contiguous()
    if tuple(valid.shape) == tuple(value.shape[:2]):
        return valid.unsqueeze(-1).expand_as(value).contiguous()
    raise ValueError(f"{target_name}: valid mask shape {tuple(valid.shape)} does not match {tuple(value.shape)}")


def _per_sample_mask_rate(mask_value: Any, valid_mask: Any, *, batch_size: int, target_name: str) -> Tensor:
    mask = mask_value.detach().cpu().bool() if isinstance(mask_value, Tensor) else torch.as_tensor(mask_value, dtype=torch.bool)
    if mask.ndim != 3 or int(mask.shape[0]) != int(batch_size):
        raise ValueError(f"{target_name}: mask must be [B,T,C], got {tuple(mask.shape)}")
    valid = _expand_valid_mask(valid_mask, mask, target_name=target_name)
    flat_mask = (mask & valid).reshape(batch_size, -1)
    flat_valid = valid.reshape(batch_size, -1)
    denom = flat_valid.sum(dim=1)
    values = flat_mask.sum(dim=1).float() / denom.clamp_min(1).float()
    return torch.where(denom > 0, values, torch.full_like(values, float("nan"))).detach().cpu().float()


def _per_sample_energy_share(
    fp32_btc: Tensor,
    mask_value: Any,
    valid_mask: Any,
    *,
    eps: float,
    target_name: str,
) -> Tensor:
    mask = mask_value.detach().cpu().bool() if isinstance(mask_value, Tensor) else torch.as_tensor(mask_value, dtype=torch.bool)
    if tuple(mask.shape) != tuple(fp32_btc.shape):
        raise ValueError(f"{target_name}: energy mask shape {tuple(mask.shape)} != activation {tuple(fp32_btc.shape)}")
    valid = _expand_valid_mask(valid_mask, mask, target_name=target_name)
    energy = fp32_btc.detach().cpu().float().square()
    numerator = energy.masked_fill(~(mask & valid), 0.0).reshape(int(fp32_btc.shape[0]), -1).sum(dim=1)
    denominator = energy.masked_fill(~valid, 0.0).reshape(int(fp32_btc.shape[0]), -1).sum(dim=1)
    values = numerator / denominator.clamp_min(float(eps))
    return torch.where(denominator > float(eps), values, torch.full_like(values, float("nan"))).detach().cpu().float()


def _mean_norm_retention(fp32_btc: Tensor, quant_btc: Tensor, valid_mask: Any, *, eps: float, target_name: str) -> Tensor:
    if tuple(fp32_btc.shape) != tuple(quant_btc.shape):
        raise ValueError(f"{target_name}: fp32/quant activation shapes differ")
    if valid_mask is None:
        token_valid = torch.ones(tuple(fp32_btc.shape[:2]), dtype=torch.bool)
    elif isinstance(valid_mask, Tensor):
        token_valid = valid_mask.detach().cpu().bool()
    else:
        token_valid = torch.as_tensor(valid_mask, dtype=torch.bool)
    if tuple(token_valid.shape) == tuple(fp32_btc.shape):
        token_valid = token_valid.any(dim=-1)
    if tuple(token_valid.shape) != tuple(fp32_btc.shape[:2]):
        token_valid = torch.ones(tuple(fp32_btc.shape[:2]), dtype=torch.bool)
    fp32_norm = torch.linalg.vector_norm(fp32_btc.float(), ord=2, dim=-1)
    quant_norm = torch.linalg.vector_norm(quant_btc.float(), ord=2, dim=-1)
    retention = quant_norm / fp32_norm.clamp_min(float(eps))
    retention = torch.where(fp32_norm > float(eps), retention, torch.full_like(retention, float("nan")))
    values: list[Tensor] = []
    for sample_idx in range(int(fp32_btc.shape[0])):
        sample_values = retention[sample_idx][token_valid[sample_idx]]
        finite = sample_values[torch.isfinite(sample_values)]
        values.append(finite.mean() if int(finite.numel()) > 0 else retention.new_tensor(float("nan")))
    return torch.stack(values).detach().cpu().float()


def _nan_vector(length: int) -> Tensor:
    return torch.full((int(length),), float("nan"), dtype=torch.float32)


def _topk_rows(topk_indices: Tensor | Sequence[Any], *, batch_size: int, target_name: str) -> list[Tensor]:
    if isinstance(topk_indices, Tensor):
        tensor = topk_indices.detach().cpu().long()
        if tensor.ndim == 1:
            if int(batch_size) != 1:
                raise ValueError(f"{target_name}: 1D top-k indices only support batch_size=1")
            tensor = tensor.reshape(1, -1)
        else:
            tensor = tensor.reshape(int(tensor.shape[0]), -1)
        if int(tensor.shape[0]) != int(batch_size):
            raise ValueError(f"{target_name}: top-k rows {int(tensor.shape[0])} != batch_size {batch_size}")
        return [tensor[idx].reshape(-1) for idx in range(int(batch_size))]
    if isinstance(topk_indices, (str, bytes)):
        raise TypeError(f"{target_name}: top-k indices must not be a string")
    rows = [torch.as_tensor(item, dtype=torch.long).cpu().reshape(-1) for item in topk_indices]
    if len(rows) != int(batch_size):
        raise ValueError(f"{target_name}: top-k rows {len(rows)} != batch_size {batch_size}")
    return rows


def _slice_topk_indices(
    topk_indices: Tensor | Sequence[Any],
    *,
    sample_offset: int,
    batch_size: int,
    target_name: str,
) -> Tensor | list[Tensor]:
    if isinstance(topk_indices, Tensor):
        tensor = topk_indices.detach().cpu().long()
        if tensor.ndim == 1:
            if int(sample_offset) != 0 or int(batch_size) != 1:
                raise ValueError(f"{target_name}: cannot slice 1D top-k indices except for one-sample oracle")
            return tensor.reshape(1, -1)
        end = int(sample_offset) + int(batch_size)
        if end > int(tensor.shape[0]):
            raise ValueError(f"{target_name}: top-k tensor slice exceeds oracle rows")
        return tensor[int(sample_offset):end].contiguous()
    rows = list(topk_indices)
    end = int(sample_offset) + int(batch_size)
    if end > len(rows):
        raise ValueError(f"{target_name}: top-k list slice exceeds oracle rows")
    return [torch.as_tensor(rows[idx], dtype=torch.long).cpu().reshape(-1) for idx in range(int(sample_offset), end)]


def _validate_oracle_sample_ids(
    oracle: Mapping[str, Any],
    *,
    sample_offset: int,
    batch_sample_ids: Sequence[str],
    target_name: str,
) -> None:
    oracle_ids = [str(item) for item in oracle.get("sample_id", [])]
    if not oracle_ids:
        return
    expected = oracle_ids[int(sample_offset):int(sample_offset) + len(batch_sample_ids)]
    if expected != [str(item) for item in batch_sample_ids]:
        raise RuntimeError(
            f"{target_name}: OER oracle sample order mismatch at offset {sample_offset}; "
            f"oracle={expected[:3]} batch={list(batch_sample_ids)[:3]}"
        )


def _teacher_topk_relative_distortion(
    fp32: Tensor,
    quant: Tensor,
    topk_indices: Tensor | Sequence[Any],
    *,
    eps: float,
    target_name: str,
) -> tuple[Tensor, Tensor]:
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"{target_name}: OER activation shape mismatch")
    batch_size = int(fp32.shape[0])
    fp32_flat = fp32.detach().float().reshape(batch_size, -1)
    quant_flat = quant.detach().float().reshape(batch_size, -1).to(device=fp32_flat.device)
    rows = _topk_rows(topk_indices, batch_size=batch_size, target_name=target_name)
    values: list[Tensor] = []
    counts: list[int] = []
    for sample_idx, index_row in enumerate(rows):
        indices = index_row.to(device=fp32_flat.device, dtype=torch.long).reshape(-1)
        counts.append(int(indices.numel()))
        if int(indices.numel()) == 0:
            values.append(fp32_flat.new_tensor(float("nan")))
            continue
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(fp32_flat.shape[1]):
            raise ValueError(f"{target_name}: top-k indices contain out-of-range scalar ids")
        fp32_values = fp32_flat[sample_idx].index_select(0, indices)
        quant_values = quant_flat[sample_idx].index_select(0, indices)
        denom = torch.linalg.vector_norm(fp32_values, ord=2)
        value = torch.linalg.vector_norm(quant_values - fp32_values, ord=2) / denom.clamp_min(float(eps))
        values.append(value if float(denom.item()) > float(eps) else fp32_flat.new_tensor(float("nan")))
    return torch.stack(values).detach().cpu().float(), torch.as_tensor(counts, dtype=torch.float32)


def _topk_overlap(
    fp32: Tensor,
    quant: Tensor,
    teacher_topk_indices: Tensor | Sequence[Any],
    *,
    target_name: str,
) -> Tensor:
    del fp32
    batch_size = int(quant.shape[0])
    quant_flat = quant.detach().float().reshape(batch_size, -1)
    rows = _topk_rows(teacher_topk_indices, batch_size=batch_size, target_name=target_name)
    values: list[Tensor] = []
    for sample_idx, index_row in enumerate(rows):
        teacher = index_row.to(device=quant_flat.device, dtype=torch.long).reshape(-1)
        if int(teacher.numel()) == 0:
            values.append(quant_flat.new_tensor(float("nan")))
            continue
        if int(teacher.min().item()) < 0 or int(teacher.max().item()) >= int(quant_flat.shape[1]):
            raise ValueError(f"{target_name}: teacher top-k indices contain out-of-range scalar ids")
        k = min(int(teacher.numel()), int(quant_flat.shape[1]))
        quant_topk = torch.topk(quant_flat[sample_idx].square(), k=k, largest=True, sorted=False).indices
        teacher_mask = torch.zeros(int(quant_flat.shape[1]), dtype=torch.bool, device=quant_flat.device)
        quant_mask = torch.zeros_like(teacher_mask)
        teacher_mask.scatter_(0, teacher[:k], True)
        quant_mask.scatter_(0, quant_topk, True)
        values.append((teacher_mask & quant_mask).sum().float() / float(k))
    return torch.stack(values).detach().cpu().float()


def _oer_topk_vectors(
    fp32: Tensor,
    quant: Tensor,
    topk_indices: Tensor | Sequence[Any] | None,
    *,
    batch_size: int,
    target_name: str,
    eps: float,
) -> dict[str, Tensor]:
    if topk_indices is None:
        return {
            "oer_available": torch.zeros(batch_size, dtype=torch.float32),
            "oer_retention": _nan_vector(batch_size),
            "oer_loss": _nan_vector(batch_size),
            "off_oracle_energy_ratio": _nan_vector(batch_size),
            "teacher_topk_relative_distortion": _nan_vector(batch_size),
            "topk_overlap": _nan_vector(batch_size),
            "teacher_topk_count": _nan_vector(batch_size),
        }
    rows = _topk_rows(topk_indices, batch_size=batch_size, target_name=target_name)
    oer_values = oer_metrics_from_topk_indices(fp32, quant, rows, eps=eps)
    distortion, counts = _teacher_topk_relative_distortion(
        fp32,
        quant,
        rows,
        eps=eps,
        target_name=target_name,
    )
    return {
        "oer_available": torch.ones(batch_size, dtype=torch.float32),
        "oer_retention": oer_values["oer_retention"],
        "oer_loss": oer_values["oer_loss"],
        "off_oracle_energy_ratio": oer_values["off_oracle_energy_ratio"],
        "teacher_topk_relative_distortion": distortion,
        "topk_overlap": _topk_overlap(fp32, quant, rows, target_name=target_name),
        "teacher_topk_count": counts,
    }


def _danger_vectors_from_quantlinear(
    fp32_btc: Tensor,
    quant_btc: Tensor,
    diagnostics: Mapping[str, Any],
    *,
    batch_size: int,
    target_name: str,
    eps: float,
    fallback_valid_mask: Any = None,
) -> dict[str, Tensor]:
    qmin = int(diagnostics.get("qmin", EXPECTED_ACTIVATION_QRANGE[0]))
    qmax = int(diagnostics.get("qmax", EXPECTED_ACTIVATION_QRANGE[1]))
    if (qmin, qmax) != EXPECTED_ACTIVATION_QRANGE:
        raise RuntimeError(f"{target_name}: integer-grid qrange must be {EXPECTED_ACTIVATION_QRANGE}, got {(qmin, qmax)}")
    valid_mask = diagnostics.get("valid_token_mask", fallback_valid_mask)
    strict = diagnostics.get("strict_overflow_mask", diagnostics.get("clip_mask"))
    boundary = diagnostics.get("boundary_hit_mask")
    exact_zero = diagnostics.get("exact_zero_mask", diagnostics.get("near_zero_mask"))
    near_zero = diagnostics.get("near_zero_pm1_mask")
    q_preclip = diagnostics.get("q_preclip")
    if strict is None or boundary is None or exact_zero is None or near_zero is None:
        raise KeyError(f"{target_name}: integer-grid diagnostics are missing required masks")
    if q_preclip is not None:
        q_preclip_t = q_preclip.detach().cpu().float() if isinstance(q_preclip, Tensor) else torch.as_tensor(q_preclip, dtype=torch.float32)
        clipping = (q_preclip_t <= float(qmin)) | (q_preclip_t >= float(qmax))
    else:
        clipping = boundary
    return {
        "strict_overflow_rate": _per_sample_mask_rate(strict, valid_mask, batch_size=batch_size, target_name=target_name),
        "overflow_energy_proxy": _per_sample_energy_share(fp32_btc, strict, valid_mask, eps=eps, target_name=target_name),
        "boundary_hit_rate": _per_sample_mask_rate(boundary, valid_mask, batch_size=batch_size, target_name=target_name),
        "clipping_rate": _per_sample_mask_rate(clipping, valid_mask, batch_size=batch_size, target_name=target_name),
        "near_zero_rate": _per_sample_mask_rate(near_zero, valid_mask, batch_size=batch_size, target_name=target_name),
        "exact_zero_rate": _per_sample_mask_rate(exact_zero, valid_mask, batch_size=batch_size, target_name=target_name),
        "activation_quant_error": relative_l2_per_sample(fp32_btc, quant_btc, eps=eps),
        "mean_norm_retention": _mean_norm_retention(
            fp32_btc,
            quant_btc,
            valid_mask,
            eps=eps,
            target_name=target_name,
        ),
    }


def _l9_sample_rows(
    *,
    spec: MultiplierSpec,
    quant_module: QuantLinear,
    batch_sample_ids: Sequence[str],
    batch_labels: Sequence[Any],
    batch_buckets: Sequence[str],
    sample_offset: int,
    oer_oracle: Mapping[str, Any] | None,
    baseline_scale: float,
    counterfactual_scale: float,
    eps: float,
) -> list[dict[str, Any]]:
    target_name = _target_name(*TARGET)
    batch_size = len(batch_sample_ids)
    fp32_activation, quant_activation = quant_module.get_oer_activation_tensors()
    if fp32_activation is None or quant_activation is None:
        raise RuntimeError(f"{target_name}: fake quant eval missing OER activation cache")
    fp32_btc = _as_btc_tensor(fp32_activation, batch_size=batch_size, target_name=target_name, tensor_name="fp32_activation")
    quant_btc = _as_btc_tensor(quant_activation, batch_size=batch_size, target_name=target_name, tensor_name="quant_activation")
    diagnostics = quant_module.get_integer_grid_diagnostics()
    if diagnostics is None:
        raise RuntimeError(f"{target_name}: integer-grid diagnostics were not captured")
    danger_vectors = _danger_vectors_from_quantlinear(
        fp32_btc,
        quant_btc,
        diagnostics,
        batch_size=batch_size,
        target_name=target_name,
        eps=eps,
        fallback_valid_mask=getattr(quant_module, "last_activation_valid_mask_for_oer", None),
    )
    topk_indices: Tensor | list[Tensor] | None = None
    if oer_oracle is not None:
        _validate_oracle_sample_ids(
            oer_oracle,
            sample_offset=sample_offset,
            batch_sample_ids=batch_sample_ids,
            target_name=target_name,
        )
        raw_topk = oer_oracle.get("input_topk_indices", oer_oracle.get("topk_indices", oer_oracle.get("flat_index")))
        if raw_topk is not None:
            topk_indices = _slice_topk_indices(
                raw_topk,
                sample_offset=sample_offset,
                batch_size=batch_size,
                target_name=target_name,
            )
    oer_vectors = _oer_topk_vectors(
        fp32_btc,
        quant_btc,
        topk_indices,
        batch_size=batch_size,
        target_name=target_name,
        eps=eps,
    )

    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(batch_sample_ids):
        row: dict[str, Any] = {
            "experiment_name": EXPERIMENT_NAME,
            "candidate": spec.name,
            "range_multiplier": float(spec.k),
            "layer_idx": TARGET_LAYER_IDX,
            "module_name": TARGET_MODULE_NAME,
            "target_name": target_name,
            "activation_position": "FFN fc1 input",
            "activation_granularity": ACTIVATION_GRANULARITY,
            "weight_granularity": WEIGHT_GRANULARITY,
            "baseline_scale": float(baseline_scale),
            "counterfactual_scale": float(counterfactual_scale),
            "qmin": EXPECTED_ACTIVATION_QRANGE[0],
            "qmax": EXPECTED_ACTIVATION_QRANGE[1],
            "zero_point": EXPECTED_ZERO_POINT,
            "sample_id": str(sample_id),
            "label": batch_labels[idx],
            "bucket": str(batch_buckets[idx]),
            "row_type": "sample",
            "oer_available": float(oer_vectors["oer_available"][idx].item()),
            "teacher_topk_count": float(oer_vectors["teacher_topk_count"][idx].item()),
        }
        for metric in (
            "strict_overflow_rate",
            "overflow_energy_proxy",
            "boundary_hit_rate",
            "clipping_rate",
            "near_zero_rate",
            "exact_zero_rate",
            "activation_quant_error",
            "mean_norm_retention",
        ):
            row[metric] = _finite_float(danger_vectors[metric][idx])
        for metric in (
            "oer_retention",
            "oer_loss",
            "off_oracle_energy_ratio",
            "teacher_topk_relative_distortion",
            "topk_overlap",
        ):
            row[metric] = _finite_float(oer_vectors[metric][idx])
        rows.append(row)
    return rows


def _bucket_in_group(bucket: Any, group_name: str) -> bool:
    value = str(bucket).strip().lower()
    if group_name == "overall":
        return True
    if group_name == "tail":
        return value in {"medium", "few"}
    return value == str(group_name)


def _finite_float(value: Any) -> float:
    try:
        if isinstance(value, Tensor):
            value = value.detach().cpu().reshape(-1)[0].item()
        result = float(value)
    except (TypeError, ValueError, IndexError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _finite_mean(values: Sequence[Any]) -> float:
    finite = [_finite_float(item) for item in values]
    finite = [item for item in finite if math.isfinite(item)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _finite_count(values: Sequence[Any]) -> int:
    return sum(1 for item in values if math.isfinite(_finite_float(item)))


def aggregate_danger_zone_rows(
    *,
    spec: MultiplierSpec,
    sample_rows: Sequence[Mapping[str, Any]],
    baseline_scale: float,
    counterfactual_scale: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUP_NAMES:
        group_samples = [row for row in sample_rows if _bucket_in_group(row.get("bucket", ""), group)]
        row: dict[str, Any] = {
            "experiment_name": EXPERIMENT_NAME,
            "candidate": spec.name,
            "range_multiplier": float(spec.k),
            "analysis_group": group,
            "layer_idx": TARGET_LAYER_IDX,
            "module_name": TARGET_MODULE_NAME,
            "target_name": _target_name(*TARGET),
            "activation_position": "FFN fc1 input",
            "activation_granularity": ACTIVATION_GRANULARITY,
            "weight_granularity": WEIGHT_GRANULARITY,
            "baseline_scale": float(baseline_scale),
            "counterfactual_scale": float(counterfactual_scale),
            "qmin": EXPECTED_ACTIVATION_QRANGE[0],
            "qmax": EXPECTED_ACTIVATION_QRANGE[1],
            "zero_point": EXPECTED_ZERO_POINT,
            "sample_count": len(group_samples),
            "aggregation_unit": "sample",
            "aggregation_fn": "nanmean",
            "metric_status": "empty_group" if not group_samples else "ok",
            "row_type": "aggregate",
        }
        for metric in DANGER_ZONE_METRICS:
            values = [sample_row.get(metric) for sample_row in group_samples]
            row[metric] = _finite_mean(values)
            row[f"{metric}_valid_count"] = _finite_count(values)
        row["oer_available_count"] = sum(1 for sample_row in group_samples if _finite_float(sample_row.get("oer_available")) > 0.5)
        rows.append(row)
    return rows


def _records_by_key(records: Sequence[TokenTraceRecord]) -> dict[tuple[int, str, str], TokenTraceRecord]:
    result: dict[tuple[int, str, str], TokenTraceRecord] = {}
    for record in records:
        key = (int(record.layer_idx), str(record.module_name), str(record.tensor_position))
        result[key] = record
    return result


def _forward_with_trace(
    model: nn.Module,
    batch: Mapping[str, Any],
    cfg: Any,
    *,
    sample_ids: Sequence[str],
    token_padding_mask: Tensor,
) -> tuple[Tensor, Tensor, list[TokenTraceRecord]]:
    manager = TokenTraceHookManager(
        model,
        layer_indices=tuple(sorted({int(spec["layer_idx"]) for spec in TRACE_SPECS})),
        positions=("block_output",),
        require_padding_mask=True,
    )
    manager.register_hooks()
    try:
        manager.set_batch_context(sample_id=list(sample_ids), padding_mask=token_padding_mask.detach().cpu())
        logits, embedding, returned_mask, _valid_count = forward_core_outputs(model, batch, cfg)
        returned_mask = returned_mask.detach().cpu().bool().contiguous()
        if tuple(returned_mask.shape) != tuple(token_padding_mask.shape):
            raise RuntimeError("token padding_mask shape changed during trace forward")
        if not torch.equal(returned_mask, token_padding_mask.detach().cpu().bool()):
            raise RuntimeError("token padding_mask changed during trace forward")
        records = manager.pop_cache()
        expected = {(int(spec["layer_idx"]), str(spec["module_name"]), str(spec["tensor_position"])) for spec in TRACE_SPECS}
        captured = set(_records_by_key(records))
        missing = sorted(expected - captured)
        if missing:
            raise RuntimeError(f"TokenTraceHookManager missing expected records: {missing}")
        return logits.detach().cpu().float(), embedding.detach().cpu().float(), records
    finally:
        manager.clear_hooks()
        manager.clear_cache()
        manager.clear_batch_context()


def _token_rows_for_batch(
    *,
    spec: MultiplierSpec,
    fp32_records: Sequence[TokenTraceRecord],
    quant_records: Sequence[TokenTraceRecord],
    token_padding_mask: Tensor,
    batch_sample_ids: Sequence[str],
    batch_labels: Sequence[Any],
    batch_buckets: Sequence[str],
    fp32_embedding: Tensor,
    quant_embedding: Tensor,
    cfg: Any,
) -> list[dict[str, Any]]:
    fp32_by_key = _records_by_key(fp32_records)
    quant_by_key = _records_by_key(quant_records)
    rows: list[dict[str, Any]] = []
    for trace_spec in TRACE_SPECS:
        key = (
            int(trace_spec["layer_idx"]),
            str(trace_spec["module_name"]),
            str(trace_spec["tensor_position"]),
        )
        fp32_record = fp32_by_key[key]
        quant_record = quant_by_key[key]
        metric_rows = compute_token_survival_rows(
            fp32_record.tensor,
            quant_record.tensor,
            token_padding_mask,
            sample_id=list(batch_sample_ids),
            label=list(batch_labels),
            bucket=list(batch_buckets),
            shot_group=list(batch_buckets),
            layer_idx=int(trace_spec["layer_idx"]),
            module_name=str(trace_spec["module_name"]),
            tensor_position=str(trace_spec["tensor_position"]),
            quant_mode=QUANT_MODE,
            granularity=f"l9_fc1_scale_{spec.slug}",
            require_integer_mapping=False,
            fp32_embedding=fp32_embedding,
            int4_embedding=quant_embedding,
            topk_ratio=_token_topk_ratio(cfg),
            survival_min_norm_retention=_survival_min_norm_retention(cfg),
            eps=_metric_eps(cfg),
        )
        for row in metric_rows:
            enriched = dict(row)
            enriched.update(
                {
                    "experiment_name": EXPERIMENT_NAME,
                    "candidate": spec.name,
                    "range_multiplier": float(spec.k),
                    "trace_scope": str(trace_spec["scope"]),
                    "trace_target_name": fp32_record.target_name,
                    "tracked_token_ids_fixed": True,
                    "topk_selected_on": "fp32",
                    "metric_source": "src.metrics.token_survival_metrics.compute_token_survival_rows",
                    "row_type": "sample",
                }
            )
            rows.append(enriched)
    return rows


def aggregate_token_metric_rows(*, spec: MultiplierSpec, sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = [str(item["scope"]) for item in TRACE_SPECS]
    for trace_scope in scopes:
        trace_rows = [row for row in sample_rows if str(row.get("trace_scope", "")) == trace_scope]
        if trace_rows:
            first = trace_rows[0]
            layer_idx = int(first.get("layer_idx", -1))
            module_name = str(first.get("module_name", ""))
            tensor_position = str(first.get("tensor_position", ""))
        else:
            trace_spec = next(item for item in TRACE_SPECS if str(item["scope"]) == trace_scope)
            layer_idx = int(trace_spec["layer_idx"])
            module_name = str(trace_spec["module_name"])
            tensor_position = str(trace_spec["tensor_position"])
        for group in GROUP_NAMES:
            group_rows = [row for row in trace_rows if _bucket_in_group(row.get("bucket", ""), group)]
            row: dict[str, Any] = {
                "experiment_name": EXPERIMENT_NAME,
                "candidate": spec.name,
                "range_multiplier": float(spec.k),
                "analysis_group": group,
                "trace_scope": trace_scope,
                "layer_idx": layer_idx,
                "module_name": module_name,
                "tensor_position": tensor_position,
                "sample_count": len(group_rows),
                "aggregation_unit": "sample",
                "aggregation_fn": "nanmean",
                "metric_status": "empty_group" if not group_rows else "ok",
                "row_type": "aggregate",
            }
            for metric in TOKEN_METRICS:
                values = [sample_row.get(metric) for sample_row in group_rows]
                row[metric] = _finite_mean(values)
                row[f"{metric}_valid_count"] = _finite_count(values)
            rows.append(row)
    return rows


def run_multiplier_candidate(
    cfg: Any,
    *,
    spec: MultiplierSpec,
    target_modules: Sequence[tuple[int, str]],
    baseline_qparams_by_target: Mapping[tuple[int, str], Mapping[str, Any]],
    eval_loader: DataLoader,
    sample_metadata: Mapping[str, Mapping[str, Any]],
    oer_oracle: Mapping[str, Any] | None,
    output_dir: Path,
) -> CandidateRunResult:
    qparams_by_target = build_multiplier_qparams_by_target(baseline_qparams_by_target, spec=spec)
    l9_qparams = qparams_by_target[_target_key(*TARGET)]
    baseline_scale = float(l9_qparams["baseline_scale"])
    counterfactual_scale = float(l9_qparams["counterfactual_scale"])
    device = _device(cfg)
    quant_model = restore_clean_model(cfg).to(device)
    fp32_model = restore_clean_model(cfg).to(device)
    quant_modules, controllers = apply_l9_scale_multiplier_quantization(
        cfg,
        model=quant_model,
        target_modules=target_modules,
        qparams_by_target=qparams_by_target,
    )
    l9_module = quant_modules[_target_key(*TARGET)]
    sample_ids: list[str] = []
    labels: list[Any] = []
    buckets: list[str] = []
    label_chunks: list[Tensor] = []
    logits_chunks: list[Tensor] = []
    embedding_chunks: list[Tensor] = []
    danger_sample_rows: list[dict[str, Any]] = []
    token_sample_rows: list[dict[str, Any]] = []
    sample_offset = 0
    eps = _metric_eps(cfg)

    try:
        fp32_model.eval()
        quant_model.eval()
        for batch in eval_loader:
            batch_sample_ids, batch_labels, batch_buckets = _batch_labels_and_buckets(
                batch,
                sample_metadata=sample_metadata,
            )
            label_id = _batch_label_id(batch)
            batch_size = int(label_id.numel())
            moved = _move_batch_to_device(batch, device)

            _mask_logits, _mask_embedding, token_padding_mask, _mask_valid_count = forward_core_outputs(fp32_model, moved, cfg)
            token_padding_mask = token_padding_mask.detach().cpu().bool().contiguous()
            fp32_logits, fp32_embedding, fp32_records = _forward_with_trace(
                fp32_model,
                moved,
                cfg,
                sample_ids=batch_sample_ids,
                token_padding_mask=token_padding_mask,
            )
            quant_logits, quant_embedding, quant_records = _forward_with_trace(
                quant_model,
                moved,
                cfg,
                sample_ids=batch_sample_ids,
                token_padding_mask=token_padding_mask,
            )
            if int(quant_logits.shape[0]) != batch_size or int(quant_embedding.shape[0]) != batch_size:
                raise RuntimeError(f"{spec.name}: quant output batch mismatch")

            sample_ids.extend(batch_sample_ids)
            labels.extend(batch_labels)
            buckets.extend(batch_buckets)
            label_chunks.append(label_id.detach().cpu().long())
            logits_chunks.append(quant_logits.detach().cpu().float().contiguous())
            embedding_chunks.append(quant_embedding.detach().cpu().float().contiguous())
            danger_sample_rows.extend(
                _l9_sample_rows(
                    spec=spec,
                    quant_module=l9_module,
                    batch_sample_ids=batch_sample_ids,
                    batch_labels=batch_labels,
                    batch_buckets=batch_buckets,
                    sample_offset=sample_offset,
                    oer_oracle=oer_oracle,
                    baseline_scale=baseline_scale,
                    counterfactual_scale=counterfactual_scale,
                    eps=eps,
                )
            )
            token_sample_rows.extend(
                _token_rows_for_batch(
                    spec=spec,
                    fp32_records=fp32_records,
                    quant_records=quant_records,
                    token_padding_mask=token_padding_mask,
                    batch_sample_ids=batch_sample_ids,
                    batch_labels=batch_labels,
                    batch_buckets=batch_buckets,
                    fp32_embedding=fp32_embedding,
                    quant_embedding=quant_embedding,
                    cfg=cfg,
                )
            )
            sample_offset += batch_size
            if hasattr(l9_module, "clear_oer_activation_tensors"):
                l9_module.clear_oer_activation_tensors()
            del moved, fp32_logits, fp32_embedding, quant_logits, quant_embedding

        if not sample_ids:
            raise RuntimeError("eval_loader produced no samples")
        if len(set(sample_ids)) != len(sample_ids):
            raise RuntimeError(f"{spec.name}: duplicate sample_id values encountered")
        label_all = torch.cat(label_chunks, dim=0)
        logits_all = torch.cat(logits_chunks, dim=0)
        embedding_all = torch.cat(embedding_chunks, dim=0)
        qparams_identity_by_target = {
            _target_name(*target): qparams_identity_manifest(qparams, source_hint=_target_name(*target))
            for target, qparams in qparams_by_target.items()
        }
        cache = build_quant_embedding_cache(
            sample_id=sample_ids,
            label=labels,
            bucket=buckets,
            shot_group=buckets,
            embedding=embedding_all,
            quant_mode=QUANT_MODE,
            granularity=f"l9_fc1_scale_{spec.slug}",
            label_id=label_all,
            logits=logits_all,
            expected_dim=_expected_dim(cfg),
            extra_manifest={
                "runner": EXPERIMENT_NAME,
                "protocol_version": PROTOCOL_VERSION,
                "range_multiplier": float(spec.k),
                "target": _target_name(*TARGET),
                "activation_position": "FFN fc1 input",
                "baseline_scale": baseline_scale,
                "counterfactual_scale": counterfactual_scale,
                "activation_scale_multiplier_applied_to": _target_name(*TARGET),
                "other_activation_qparams_unchanged": True,
                "weight_qparams_unchanged": True,
                "qparams_identity_by_target": qparams_identity_by_target,
            },
        )
        cache_path = output_dir / "embedding_cache" / f"l9_fc1_scale_{spec.slug}_embedding.pt"
        save_embedding_cache(cache, cache_path, expected_dim=_expected_dim(cfg))
        danger_rows = aggregate_danger_zone_rows(
            spec=spec,
            sample_rows=danger_sample_rows,
            baseline_scale=baseline_scale,
            counterfactual_scale=counterfactual_scale,
        )
        token_rows = aggregate_token_metric_rows(spec=spec, sample_rows=token_sample_rows)
        return CandidateRunResult(
            spec=spec,
            qparams_by_target=qparams_by_target,
            embedding_cache=cache,
            danger_zone_rows=danger_rows,
            token_metric_rows=token_rows,
            cache_path=cache_path,
            runtime_summary={
                "candidate": spec.name,
                "range_multiplier": float(spec.k),
                "baseline_scale": baseline_scale,
                "counterfactual_scale": counterfactual_scale,
                "num_samples": len(sample_ids),
                "danger_sample_rows": len(danger_sample_rows),
                "danger_zone_rows": len(danger_rows),
                "token_sample_rows": len(token_sample_rows),
                "token_metric_rows": len(token_rows),
                "embedding_cache_path": str(cache_path),
            },
        )
    finally:
        _remove_controller_hooks(controllers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _macro_f1(labels: Tensor, predictions: Tensor, *, num_classes: int) -> float:
    values: list[float] = []
    for class_idx in range(int(num_classes)):
        true_positive = int(((labels == class_idx) & (predictions == class_idx)).sum().item())
        false_positive = int(((labels != class_idx) & (predictions == class_idx)).sum().item())
        false_negative = int(((labels == class_idx) & (predictions != class_idx)).sum().item())
        support = true_positive + false_negative
        if support <= 0:
            continue
        denom = (2 * true_positive) + false_positive + false_negative
        values.append(float((2 * true_positive) / denom) if denom > 0 else float("nan"))
    finite = [item for item in values if math.isfinite(item)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def compute_task_metric_rows(spec: MultiplierSpec, embedding_cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "logits" not in embedding_cache:
        return [
            {
                "experiment_name": EXPERIMENT_NAME,
                "candidate": spec.name,
                "range_multiplier": float(spec.k),
                "analysis_group": group,
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "support_count": 0,
                "metric_status": "missing_logits",
            }
            for group in GROUP_NAMES
        ]
    logits = torch.as_tensor(embedding_cache["logits"], dtype=torch.float32)
    labels = torch.as_tensor(embedding_cache["label_id"], dtype=torch.long).reshape(-1)
    predictions = logits.argmax(dim=1).detach().cpu().long().reshape(-1)
    buckets = list(embedding_cache["bucket"])
    rows: list[dict[str, Any]] = []
    for group in GROUP_NAMES:
        mask = torch.tensor([_bucket_in_group(bucket, group) for bucket in buckets], dtype=torch.bool)
        support = int(mask.sum().item())
        if support <= 0:
            accuracy = float("nan")
            macro_f1 = float("nan")
        else:
            accuracy = float((predictions[mask] == labels[mask]).float().mean().item())
            macro_f1 = _macro_f1(labels[mask], predictions[mask], num_classes=int(logits.shape[1]))
        rows.append(
            {
                "experiment_name": EXPERIMENT_NAME,
                "candidate": spec.name,
                "range_multiplier": float(spec.k),
                "analysis_group": group,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "support_count": support,
                "metric_status": "ok",
            }
        )
    return rows


def _with_candidate(row: Mapping[str, Any], spec: MultiplierSpec) -> dict[str, Any]:
    result = dict(row)
    result["experiment_name"] = EXPERIMENT_NAME
    result["candidate"] = spec.name
    result["range_multiplier"] = float(spec.k)
    result["target"] = _target_name(*TARGET)
    return result


def _metric_lookup(rows: Sequence[Mapping[str, Any]], *, group: str, name: str, value_key: str = "value") -> float:
    for row in rows:
        if str(row.get("analysis_group", "")) == group and str(row.get("metric_name", name)) == name:
            return _finite_float(row.get(value_key))
    return float("nan")


def _global_lookup(rows: Sequence[Mapping[str, Any]], *, name: str) -> float:
    for row in rows:
        if name in row:
            return _finite_float(row.get(name))
    return float("nan")


def _danger_lookup(rows: Sequence[Mapping[str, Any]], *, group: str, metric: str) -> float:
    for row in rows:
        if str(row.get("analysis_group", "")) == group:
            return _finite_float(row.get(metric))
    return float("nan")


def _token_lookup(rows: Sequence[Mapping[str, Any]], *, scope: str, group: str, metric: str) -> float:
    for row in rows:
        if str(row.get("trace_scope", "")) == scope and str(row.get("analysis_group", "")) == group:
            return _finite_float(row.get(metric))
    return float("nan")


def _task_lookup(rows: Sequence[Mapping[str, Any]], *, group: str, metric: str) -> float:
    for row in rows:
        if str(row.get("analysis_group", "")) == group:
            return _finite_float(row.get(metric))
    return float("nan")


def _delta(value: Any, baseline: Any) -> float:
    current = _finite_float(value)
    base = _finite_float(baseline)
    return current - base if math.isfinite(current) and math.isfinite(base) else float("nan")


def _positive_delta(row: Mapping[str, Any], key: str, *, direction: str) -> bool:
    value = _finite_float(row.get(key))
    if not math.isfinite(value):
        return False
    if direction == "higher":
        return value > DIAGNOSTIC_DELTA_EPS
    if direction == "lower":
        return value < -DIAGNOSTIC_DELTA_EPS
    raise ValueError(f"unknown direction={direction!r}")


def _negative_delta(row: Mapping[str, Any], key: str, *, direction: str) -> bool:
    value = _finite_float(row.get(key))
    if not math.isfinite(value):
        return False
    if direction == "higher":
        return value < -DIAGNOSTIC_DELTA_EPS
    if direction == "lower":
        return value > DIAGNOSTIC_DELTA_EPS
    raise ValueError(f"unknown direction={direction!r}")


def _classify_diagnostic_status(row: Mapping[str, Any]) -> str:
    required = (
        "strict_overflow_rate_tail",
        "overflow_energy_proxy_tail",
        "boundary_hit_rate_tail",
        "oer_retention_tail",
        "topk_overlap_tail",
        "teacher_topk_relative_distortion_tail",
        "l9_block_output_topk_overlap_tail",
        "l11_final_block_output_topk_overlap_tail",
        "embedding_distortion_tail",
        "fisher_retention_tail",
        "variance_inflation_deviation_tail",
        "tail_collapse_gap",
    )
    if any(not math.isfinite(_finite_float(row.get(key))) for key in required):
        return "invalid"

    boundary_improved = any(
        (
            _positive_delta(row, "delta_strict_overflow_rate_tail", direction="lower"),
            _positive_delta(row, "delta_overflow_energy_proxy_tail", direction="lower"),
            _positive_delta(row, "delta_boundary_hit_rate_tail", direction="lower"),
            _positive_delta(row, "delta_clipping_rate_tail", direction="lower"),
        )
    )
    local_improved = any(
        (
            _positive_delta(row, "delta_OER", direction="higher"),
            _positive_delta(row, "delta_TopK_overlap", direction="higher"),
            _positive_delta(row, "delta_relative_distortion", direction="lower"),
            _positive_delta(row, "delta_mean_norm_retention_tail", direction="higher"),
        )
    )
    token_improved = any(
        (
            _positive_delta(row, "delta_l9_block_output_topk_overlap_tail", direction="higher"),
            _positive_delta(row, "delta_l11_final_block_output_topk_overlap_tail", direction="higher"),
            _positive_delta(row, "delta_l9_block_output_mean_norm_retention_tail", direction="higher"),
            _positive_delta(row, "delta_l11_final_block_output_mean_norm_retention_tail", direction="higher"),
        )
    )
    geometry_improved = any(
        (
            _positive_delta(row, "delta_embedding_distortion", direction="lower"),
            _positive_delta(row, "delta_tail_fisher_retention", direction="higher"),
            _positive_delta(row, "delta_many_fisher_retention", direction="higher"),
            _positive_delta(row, "delta_tail_variance_inflation_deviation", direction="lower"),
            _positive_delta(row, "delta_many_variance_inflation_deviation", direction="lower"),
            _positive_delta(row, "delta_tail_collapse_gap", direction="lower"),
            _positive_delta(row, "delta_tail_geometry_margin_retention", direction="higher"),
        )
    )
    task_positive = any(
        _positive_delta(row, key, direction="higher")
        for key in (
            "delta_overall_accuracy",
            "delta_macro_f1",
            "delta_many_accuracy",
            "delta_medium_accuracy",
            "delta_few_accuracy",
            "delta_tail_accuracy",
        )
    )
    task_negative = any(
        _negative_delta(row, key, direction="higher")
        for key in (
            "delta_overall_accuracy",
            "delta_macro_f1",
            "delta_many_accuracy",
            "delta_medium_accuracy",
            "delta_few_accuracy",
            "delta_tail_accuracy",
        )
    )
    negative_tradeoff = any(
        (
            _negative_delta(row, "delta_OER", direction="higher"),
            _negative_delta(row, "delta_TopK_overlap", direction="higher"),
            _negative_delta(row, "delta_relative_distortion", direction="lower"),
            _negative_delta(row, "delta_l9_block_output_topk_overlap_tail", direction="higher"),
            _negative_delta(row, "delta_l11_final_block_output_topk_overlap_tail", direction="higher"),
            _negative_delta(row, "delta_embedding_distortion", direction="lower"),
            _negative_delta(row, "delta_tail_fisher_retention", direction="higher"),
            _negative_delta(row, "delta_tail_variance_inflation_deviation", direction="lower"),
            _negative_delta(row, "delta_near_zero_rate_tail", direction="lower"),
            _negative_delta(row, "delta_exact_zero_rate_tail", direction="lower"),
            task_negative,
        )
    )
    any_positive = boundary_improved or local_improved or token_improved or geometry_improved or task_positive
    if any_positive and negative_tradeoff:
        return "mixed"
    if boundary_improved and (token_improved or geometry_improved):
        return "boundary_and_downstream_positive"
    if boundary_improved and local_improved and not geometry_improved:
        return "local_fidelity_only"
    if boundary_improved:
        return "boundary_only"
    if geometry_improved and not task_positive and not task_negative:
        return "geometry_positive_task_neutral"
    if not any_positive and not negative_tradeoff:
        return "no_effect"
    return "mixed"


def build_summary_rows(candidate_tables: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_name, tables in candidate_tables.items():
        danger_rows = list(tables.get("danger", ()))
        token_rows = list(tables.get("token", ()))
        geometry_rows = list(tables.get("geometry_summary", ()))
        geometry_global_rows = list(tables.get("geometry_global", ()))
        task_rows = list(tables.get("task", ()))
        first_danger = danger_rows[0] if danger_rows else {}
        row: dict[str, Any] = {
            "experiment_name": EXPERIMENT_NAME,
            "candidate": candidate_name,
            "range_multiplier": _finite_float(first_danger.get("range_multiplier")),
            "target": _target_name(*TARGET),
            "baseline_scale": _finite_float(first_danger.get("baseline_scale")),
            "counterfactual_scale": _finite_float(first_danger.get("counterfactual_scale")),
            "qmin": EXPECTED_ACTIVATION_QRANGE[0],
            "qmax": EXPECTED_ACTIVATION_QRANGE[1],
            "zero_point": EXPECTED_ZERO_POINT,
            "diagnostic_status": "pending_delta",
        }
        for group in ("many", "medium", "few", "tail"):
            for metric in (
                "strict_overflow_rate",
                "overflow_energy_proxy",
                "boundary_hit_rate",
                "clipping_rate",
                "near_zero_rate",
                "exact_zero_rate",
                "oer_retention",
                "topk_overlap",
                "teacher_topk_relative_distortion",
                "mean_norm_retention",
            ):
                row[f"{metric}_{group}"] = _danger_lookup(danger_rows, group=group, metric=metric)
        row["OER_tail"] = row["oer_retention_tail"]
        row["TopK_overlap_tail"] = row["topk_overlap_tail"]
        row["relative_distortion_tail"] = row["teacher_topk_relative_distortion_tail"]

        for scope in ("l9_block_output", "l11_final_block_output"):
            for group in ("many", "tail"):
                row[f"{scope}_topk_overlap_{group}"] = _token_lookup(
                    token_rows,
                    scope=scope,
                    group=group,
                    metric="topk_overlap",
                )
                row[f"{scope}_mean_norm_retention_{group}"] = _token_lookup(
                    token_rows,
                    scope=scope,
                    group=group,
                    metric="mean_norm_retention",
                )

        for group in ("many", "tail"):
            row[f"embedding_distortion_{group}"] = _metric_lookup(
                geometry_rows,
                group=group,
                name="embedding_distortion",
            )
            row[f"fisher_retention_{group}"] = _metric_lookup(
                geometry_rows,
                group=group,
                name="fisher_retention",
            )
            row[f"variance_inflation_deviation_{group}"] = _metric_lookup(
                geometry_rows,
                group=group,
                name="variance_inflation_deviation",
            )
            row[f"geometry_margin_retention_{group}"] = _metric_lookup(
                geometry_rows,
                group=group,
                name="margin_retention",
            )
        row["tail_collapse_gap"] = _global_lookup(geometry_global_rows, name="tail_collapse_gap")

        for group in GROUP_NAMES:
            row[f"{group}_accuracy"] = _task_lookup(task_rows, group=group, metric="accuracy")
            row[f"{group}_macro_f1"] = _task_lookup(task_rows, group=group, metric="macro_f1")
        row["overall_accuracy"] = row["overall_accuracy"]
        row["macro_f1"] = row["overall_macro_f1"]
        rows.append(row)

    baseline = next((row for row in rows if abs(_finite_float(row.get("range_multiplier")) - 1.0) <= 1e-12), None)
    if baseline is None:
        return rows

    delta_keys = [
        key
        for key in rows[0].keys()
        if key
        not in {
            "experiment_name",
            "candidate",
            "range_multiplier",
            "target",
            "qmin",
            "qmax",
            "zero_point",
            "diagnostic_status",
        }
    ]
    for row in rows:
        for key in delta_keys:
            row[f"delta_{key}"] = _delta(row.get(key), baseline.get(key))
        row["delta_OER"] = row["delta_oer_retention_tail"]
        row["delta_TopK_overlap"] = row["delta_topk_overlap_tail"]
        row["delta_relative_distortion"] = row["delta_teacher_topk_relative_distortion_tail"]
        row["delta_embedding_distortion"] = row["delta_embedding_distortion_tail"]
        row["delta_tail_fisher_retention"] = row["delta_fisher_retention_tail"]
        row["delta_many_fisher_retention"] = row["delta_fisher_retention_many"]
        row["delta_tail_variance_inflation_deviation"] = row["delta_variance_inflation_deviation_tail"]
        row["delta_many_variance_inflation_deviation"] = row["delta_variance_inflation_deviation_many"]
        row["delta_tail_geometry_margin_retention"] = row["delta_geometry_margin_retention_tail"]
        row["delta_many_geometry_margin_retention"] = row["delta_geometry_margin_retention_many"]
        row["diagnostic_status"] = _classify_diagnostic_status(row)
    return rows


def build_counterfactual_qparams_manifest(
    *,
    baseline_qparams_by_target: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    target_qparams = baseline_qparams_by_target[_target_key(*TARGET)]
    activation = target_qparams["activation"]
    baseline_scale = _as_scalar_float(activation["scale"], name="L9.fc1 baseline activation scale")
    entries = []
    for spec in multiplier_specs():
        entries.append(
            {
                "candidate": spec.name,
                "range_multiplier": float(spec.k),
                "baseline_scale": float(baseline_scale),
                "counterfactual_scale": float(baseline_scale * spec.k),
                "qmin": EXPECTED_ACTIVATION_QRANGE[0],
                "qmax": EXPECTED_ACTIVATION_QRANGE[1],
                "zero_point": EXPECTED_ZERO_POINT,
                "symmetric": True,
                "activation_granularity": ACTIVATION_GRANULARITY,
                "target": _target_name(*TARGET),
                "baseline_qparams_identity": qparams_identity_manifest(target_qparams, source_hint=_target_name(*TARGET)),
            }
        )
    return {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "target": _target_name(*TARGET),
        "activation_position": "FFN fc1 input",
        "training_run": False,
        "test_touched": False,
        "entries": entries,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if int(tensor.numel()) == 1:
            return tensor.reshape(()).item()
        return tensor.reshape(-1).tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _csv_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False)
    if isinstance(value, Tensor):
        return json.dumps(_json_safe(value), ensure_ascii=False)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(str(key))
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_safe(row.get(key, "")) for key in fields})
    os.replace(tmp, path)


def _slug_float(value: float) -> str:
    return str(float(value)).rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _file_id(path: str | Path | None) -> dict[str, Any] | None:
    return None if path is None else file_identity(path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity_required_hash(path: str | Path | None, *, context: str) -> dict[str, Any]:
    if path is None:
        raise FileNotFoundError(f"{context}: path is not configured")
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{context}: file does not exist: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_sha256(resolved),
        "sha256_required": True,
    }


def _checkpoint_identity(cfg: Any) -> dict[str, Any]:
    return _file_identity_required_hash(_checkpoint_path(cfg), context="checkpoint")


def _read_csv_rows(path: str | Path, *, context: str) -> tuple[list[dict[str, str]], list[str]]:
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{context}: CSV file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(item) for item in (reader.fieldnames or [])]
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"{context}: CSV header is empty")
    if not rows:
        raise ValueError(f"{context}: CSV has no rows")
    return rows, fieldnames


def _label_column(fieldnames: Sequence[str], *, context: str) -> str:
    for column in ("label", "primary_label", "ebird_code", "label_id"):
        if column in fieldnames:
            return column
    raise KeyError(f"{context}: cannot find label column in {tuple(fieldnames)}")


def _required_column(fieldnames: Sequence[str], column: str, *, context: str) -> None:
    if column not in fieldnames:
        raise KeyError(f"{context}: required column {column!r} is missing")


def _canonical_train_label_counts(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    label_column: str,
    fieldnames: Sequence[str],
) -> tuple[dict[str, int], dict[str, int], str]:
    row_counts: dict[str, int] = {}
    label_n_values: dict[str, set[int]] = {}
    for row in train_rows:
        label = str(row.get(label_column, "")).strip()
        if not label:
            raise ValueError("train.csv contains an empty label value")
        row_counts[label] = row_counts.get(label, 0) + 1
        if "label_n" in fieldnames:
            parsed = _optional_int(row.get("label_n"))
            if parsed is None:
                raise ValueError(f"train.csv label_n is missing for label={label!r}")
            label_n_values.setdefault(label, set()).add(int(parsed))
    if "label_n" not in fieldnames:
        return dict(row_counts), row_counts, "train.csv row count"
    inconsistent = {label: sorted(values) for label, values in label_n_values.items() if len(values) != 1}
    if inconsistent:
        raise RuntimeError(f"train.csv label_n is not unique per label: {dict(list(inconsistent.items())[:8])}")
    canonical = {label: next(iter(values)) for label, values in label_n_values.items()}
    return canonical, row_counts, "train.csv label_n column"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    number = float(text)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"label_n must be an integer, got {value!r}")
    return int(number)


def build_bucket_provenance_manifest(cfg: Any) -> dict[str, Any]:
    train_csv = _split_csv(cfg, "train")
    eval_csv = _split_csv(cfg, "eval")
    train_rows, train_fields = _read_csv_rows(train_csv, context="train split")
    eval_rows, eval_fields = _read_csv_rows(eval_csv, context="eval split")
    train_label_column = _label_column(train_fields, context="train split")
    eval_label_column = _label_column(eval_fields, context="eval split")
    _required_column(eval_fields, "bucket", context="eval split")
    counts, physical_train_row_counts, train_count_source = _canonical_train_label_counts(
        train_rows,
        label_column=train_label_column,
        fieldnames=train_fields,
    )

    observed_buckets: list[str] = []
    train_label_counts_for_eval: list[int] = []
    missing_labels: list[str] = []
    label_n_mismatches: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for row_idx, row in enumerate(eval_rows):
        label = str(row.get(eval_label_column, "")).strip()
        if label not in counts:
            missing_labels.append(label)
            continue
        train_count = int(counts[label])
        observed_bucket = str(row.get("bucket", "")).strip().lower()
        expected_bucket = bucket_from_train_label_n(train_count)
        observed_buckets.append(observed_bucket)
        train_label_counts_for_eval.append(train_count)
        if "label_n" in eval_fields:
            label_n = _optional_int(row.get("label_n"))
            if label_n is not None and int(label_n) != train_count:
                label_n_mismatches.append(
                    {
                        "row_idx": int(row_idx),
                        "sample_id": str(row.get("sample_id", "")),
                        "label": label,
                        "eval_label_n": int(label_n),
                        "train_label_count": train_count,
                    }
                )
        assignments.append(
            {
                "sample_id": str(row.get("sample_id", row_idx)),
                "label": label,
                "train_label_count": train_count,
                "observed_bucket": observed_bucket,
                "expected_bucket": expected_bucket,
            }
        )
    if missing_labels:
        raise RuntimeError(f"eval split contains labels absent from train split: {sorted(set(missing_labels))[:8]}")
    read_canonical_bucket(
        bucket=observed_buckets,
        train_label_n=train_label_counts_for_eval,
        context="EnergyBound eval bucket provenance",
        allow_shot_group_fallback=False,
    )
    if label_n_mismatches:
        raise RuntimeError(f"eval label_n does not match train label counts: {label_n_mismatches[:8]}")

    train_count_values = list(counts.values())
    bucket_histogram = {bucket: 0 for bucket in ("many", "medium", "few")}
    for item in assignments:
        bucket_histogram[str(item["observed_bucket"])] = bucket_histogram.get(str(item["observed_bucket"]), 0) + 1
    return {
        "status": "validated",
        "canonical_bucket_source": "train split canonical label count",
        "canonical_train_label_count_source": train_count_source,
        "train_csv_identity": _file_id(train_csv),
        "eval_csv_identity": _file_id(eval_csv),
        "group_protocol": group_protocol_metadata(
            tail_rule="tail is recomputed from eval rows whose train-count-derived bucket is medium or few",
            canonical_group_field="bucket",
            legacy_group_alias="shot_group",
            main_comparison_group="tail",
            supplementary_extreme_group="few",
        ),
        "train_label_column": train_label_column,
        "eval_label_column": eval_label_column,
        "train_label_n_column_checked": "label_n" in train_fields,
        "eval_bucket_column": "bucket",
        "eval_label_n_column_checked": "label_n" in eval_fields,
        "eval_bucket_validated_against_train_label_count": True,
        "eval_label_n_matches_train_label_count": True,
        "train_label_count_min": min(train_count_values),
        "train_label_count_max": max(train_count_values),
        "train_label_count_num_classes": len(counts),
        "eval_sample_count": len(eval_rows),
        "eval_bucket_histogram": bucket_histogram,
        "train_label_count_hash": stable_json_hash(dict(sorted(counts.items()))),
        "train_physical_row_count_hash": stable_json_hash(dict(sorted(physical_train_row_counts.items()))),
        "train_physical_row_count_matches_canonical_label_n": dict(sorted(physical_train_row_counts.items()))
        == dict(sorted(counts.items())),
        "eval_bucket_assignment_hash": stable_json_hash(assignments),
    }


def _paths_manifest(cfg: Any, *, output_dir: Path, qparams_root: str | Path | None = None) -> dict[str, Any]:
    qroot = _qparams_root(cfg, qparams_root)
    return {
        "output_dir": str(output_dir),
        "baseline_qparams_root": str(qroot) if qroot is not None else str(_baseline_qparams_path(cfg, layer_idx=TARGET_LAYER_IDX, module_name=TARGET_MODULE_NAME).parent),
        "l9_baseline_qparams": str(
            _baseline_qparams_path(
                cfg,
                layer_idx=TARGET_LAYER_IDX,
                module_name=TARGET_MODULE_NAME,
                qparams_root=qparams_root,
            )
        ),
        "summary": str(output_dir / SUMMARY_FILENAME),
        "danger_zone": str(output_dir / DANGER_ZONE_FILENAME),
        "token_metrics": str(output_dir / TOKEN_METRICS_FILENAME),
        "geometry": str(output_dir / GEOMETRY_FILENAME),
        "task_metrics": str(output_dir / TASK_METRICS_FILENAME),
        "manifest": str(output_dir / MANIFEST_FILENAME),
        "counterfactual_qparams": str(output_dir / QPARAMS_JSON_FILENAME),
    }


def build_protocol_manifest(
    cfg: Any,
    *,
    output_dir: str | Path | None = None,
    qparams_root: str | Path | None = None,
    target_modules: Sequence[tuple[int, str]] | None = None,
    formal_experiment_ran: bool = False,
    qparams_manifest: Mapping[str, Any] | None = None,
    runtime_summaries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_output_dir = _result_dir(cfg, output_dir)
    qmin, qmax = _activation_qrange(cfg)
    manifest: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "diagnostic_only": True,
        "formal_experiment_ran": bool(formal_experiment_ran),
        "training_run": False,
        "test_touched": False,
        "target": _target_name(*TARGET),
        "activation_position": "FFN fc1 input",
        "activation_granularity": ACTIVATION_GRANULARITY,
        "weight_granularity": WEIGHT_GRANULARITY,
        "quant_mode": QUANT_MODE,
        "range_multipliers": list(RANGE_MULTIPLIERS),
        "qmin": qmin,
        "qmax": qmax,
        "zero_point": EXPECTED_ZERO_POINT,
        "symmetric": True,
        "qrange_user_confirmed": EXPECTED_ACTIVATION_QRANGE,
        "scale_intervention": {
            "definition": "s'_L9 = k * s_L9",
            "target_only": _target_name(*TARGET),
            "other_activation_qparams_unchanged": True,
            "weight_qparams_unchanged": True,
            "dynamic_forward_minmax": False,
        },
        "metric_sources": {
            "danger_zone": "QuantLinear integer-grid diagnostics captured at L9.fc1 forward input",
            "l9_oer_topk": "validated FP32 teacher oer_oracle_eval.pt input top-k scalar indices",
            "token_survival": "src.metrics.token_survival_metrics.compute_token_survival_rows",
            "geometry": "src.metrics.semantic_geometry_metrics.compute_semantic_geometry_tables",
            "task": "canonical eval logits from forward_core_outputs",
        },
        "danger_zone_metrics": list(DANGER_ZONE_METRICS),
        "token_metrics": list(TOKEN_METRICS),
        "diagnostic_status_labels": [
            "boundary_and_downstream_positive",
            "boundary_only",
            "local_fidelity_only",
            "geometry_positive_task_neutral",
            "mixed",
            "no_effect",
            "invalid",
        ],
        "summary_delta_fields": [
            "delta_embedding_distortion",
            "delta_tail_fisher_retention",
            "delta_many_fisher_retention",
            "delta_tail_variance_inflation_deviation",
            "delta_many_variance_inflation_deviation",
            "delta_tail_collapse_gap",
            "delta_OER",
            "delta_TopK_overlap",
            "delta_relative_distortion",
        ],
        "target_modules": [] if target_modules is None else [_target_name(*target) for target in target_modules],
        "calibration_identity": _file_id(_split_csv(cfg, "calibration")),
        "train_split_identity": _file_id(_split_csv(cfg, "train")),
        "eval_split_identity": _file_id(_split_csv(cfg, "eval")),
        "label_mapping_identity": _file_id(_label_mapping_path(cfg)),
        "checkpoint_identity": _checkpoint_identity(cfg),
        "bucket_provenance": build_bucket_provenance_manifest(cfg),
        "seed": _seed(cfg),
        "paths": _paths_manifest(cfg, output_dir=resolved_output_dir, qparams_root=qparams_root),
        "run_commands": [
            "python -m src.export.export_engergybound --audit-only",
            "python -m src.export.export_engergybound --write-audit-manifest",
            "python -m src.export.export_engergybound --run",
            "python -m src.export.export_engergybound --run --qparams-root local_data/semantic_geometry/qparams",
        ],
    }
    if (qmin, qmax) != EXPECTED_ACTIVATION_QRANGE:
        manifest["protocol_status"] = "invalid_qrange"
        manifest["protocol_error"] = f"Expected {EXPECTED_ACTIVATION_QRANGE}, got {(qmin, qmax)}"
    else:
        manifest["protocol_status"] = "ready"
    if qparams_manifest is not None:
        manifest["counterfactual_qparams"] = dict(qparams_manifest)
        manifest["baseline_qparams_hash"] = qparams_manifest["entries"][0]["baseline_qparams_identity"]
    if runtime_summaries is not None:
        manifest["runtime_summaries"] = list(runtime_summaries)
    return manifest


def run_engergybound_diagnostic(
    cfg: Any,
    *,
    output_dir: str | Path | None = None,
    qparams_root: str | Path | None = None,
) -> dict[str, Any]:
    _assert_quant_injection_qrange(cfg)
    resolved_output_dir = _result_dir(cfg, output_dir)
    eval_loader = build_eval_loader(cfg)
    sample_metadata = load_eval_metadata(cfg)
    probe_model = restore_clean_model(cfg)
    target_modules = discover_ffn_targets(probe_model, cfg)
    del probe_model

    baseline_qparams_by_target = load_baseline_qparams_by_target(
        cfg,
        target_modules=target_modules,
        qparams_root=qparams_root,
    )
    qparams_manifest = build_counterfactual_qparams_manifest(
        baseline_qparams_by_target=baseline_qparams_by_target,
    )
    _write_json(resolved_output_dir / QPARAMS_JSON_FILENAME, qparams_manifest)

    oer_oracle, oer_state = load_l9_input_oer_oracle(cfg, eval_loader=eval_loader)
    fp32_result = run_embedding_mode(
        cfg,
        quant_mode="none",
        eval_loader=eval_loader,
        sample_metadata=sample_metadata,
        save=True,
        save_path=resolved_output_dir / "embedding_cache" / "fp32_reference_embedding.pt",
        expected_dim=_expected_dim(cfg),
    )
    fp32_cache = fp32_result.embedding_cache
    fp32_reference = build_fp32_semantic_geometry_reference(
        sample_id=fp32_cache["sample_id"],
        label_id=fp32_cache["label_id"],
        bucket=fp32_cache["bucket"],
        shot_group=fp32_cache.get("shot_group"),
        fp32_embeddings=fp32_cache["embedding"],
        fp32_logits=fp32_cache.get("logits"),
    )

    all_danger_rows: list[dict[str, Any]] = []
    all_token_rows: list[dict[str, Any]] = []
    all_geometry_rows: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    candidate_tables: dict[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {}
    runtime_summaries: list[dict[str, Any]] = []

    for spec in multiplier_specs():
        result = run_multiplier_candidate(
            cfg,
            spec=spec,
            target_modules=target_modules,
            baseline_qparams_by_target=baseline_qparams_by_target,
            eval_loader=eval_loader,
            sample_metadata=sample_metadata,
            oer_oracle=oer_oracle,
            output_dir=resolved_output_dir,
        )
        all_danger_rows.extend(result.danger_zone_rows)
        all_token_rows.extend(result.token_metric_rows)
        runtime_summaries.append(result.runtime_summary)
        geometry_tables = compute_semantic_geometry_tables(
            fp32_reference=fp32_reference,
            quant_embeddings=result.embedding_cache["embedding"],
            quant_logits=result.embedding_cache.get("logits"),
            sample_id=result.embedding_cache["sample_id"],
            label_id=result.embedding_cache["label_id"],
            bucket=result.embedding_cache["bucket"],
            shot_group=result.embedding_cache.get("shot_group"),
            quant_mode=QUANT_MODE,
            granularity=f"l9_fc1_scale_{spec.slug}",
            require_logits=False,
        )
        geometry_summary = build_semantic_geometry_summary_v2_long(geometry_tables)
        geometry_rows = [_with_candidate(row, spec) for row in geometry_summary]
        geometry_global_rows = [
            _with_candidate({"table_level": "global_level", **dict(row)}, spec)
            for row in geometry_tables.get("global_level", [])
        ]
        all_geometry_rows.extend(geometry_rows)
        all_geometry_rows.extend(geometry_global_rows)
        task_rows = compute_task_metric_rows(spec, result.embedding_cache)
        all_task_rows.extend(task_rows)
        candidate_tables[result.spec.name] = {
            "danger": result.danger_zone_rows,
            "token": result.token_metric_rows,
            "geometry_summary": geometry_rows,
            "geometry_global": geometry_global_rows,
            "task": task_rows,
        }

    summary_rows = build_summary_rows(candidate_tables)
    manifest = build_protocol_manifest(
        cfg,
        output_dir=resolved_output_dir,
        qparams_root=qparams_root,
        target_modules=target_modules,
        formal_experiment_ran=True,
        qparams_manifest=qparams_manifest,
        runtime_summaries=runtime_summaries,
    )
    manifest["oer_reference_state"] = oer_state
    manifest["row_counts"] = {
        "summary": len(summary_rows),
        "danger_zone": len(all_danger_rows),
        "token_metrics": len(all_token_rows),
        "geometry": len(all_geometry_rows),
        "task_metrics": len(all_task_rows),
    }
    manifest["candidate_table_hash"] = stable_json_hash({
        "summary_rows": summary_rows,
        "row_counts": manifest["row_counts"],
    })

    _write_csv(resolved_output_dir / SUMMARY_FILENAME, summary_rows)
    _write_csv(resolved_output_dir / DANGER_ZONE_FILENAME, all_danger_rows)
    _write_csv(resolved_output_dir / TOKEN_METRICS_FILENAME, all_token_rows)
    _write_csv(resolved_output_dir / GEOMETRY_FILENAME, all_geometry_rows)
    _write_csv(resolved_output_dir / TASK_METRICS_FILENAME, all_task_rows)
    _write_json(resolved_output_dir / MANIFEST_FILENAME, manifest)
    return manifest


def write_audit_manifest(
    cfg: Any,
    *,
    output_dir: str | Path | None = None,
    qparams_root: str | Path | None = None,
) -> dict[str, Any]:
    probe_model = restore_clean_model(cfg)
    target_modules = discover_ffn_targets(probe_model, cfg)
    del probe_model
    manifest = build_protocol_manifest(
        cfg,
        output_dir=output_dir,
        qparams_root=qparams_root,
        target_modules=target_modules,
        formal_experiment_ran=False,
    )
    _write_json(_result_dir(cfg, output_dir) / MANIFEST_FILENAME, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--run", action="store_true", help="Run the formal L9 boundary counterfactual diagnostic.")
    parser.add_argument("--audit-only", action="store_true", help="Print the protocol manifest without running eval.")
    parser.add_argument("--write-audit-manifest", action="store_true", help="Write audit-only manifest JSON.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override.")
    parser.add_argument("--qparams-root", type=str, default=None, help="Directory containing frozen per-tensor W4A4 qparams.")
    args = parser.parse_args(argv)

    import config as project_config

    if args.run:
        manifest = run_engergybound_diagnostic(
            project_config,
            output_dir=args.output_dir,
            qparams_root=args.qparams_root,
        )
    elif args.write_audit_manifest:
        manifest = write_audit_manifest(
            project_config,
            output_dir=args.output_dir,
            qparams_root=args.qparams_root,
        )
    else:
        manifest = build_protocol_manifest(
            project_config,
            output_dir=args.output_dir,
            qparams_root=args.qparams_root,
            formal_experiment_ran=False,
        )
    print(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DANGER_ZONE_FILENAME",
    "EXPERIMENT_NAME",
    "MANIFEST_FILENAME",
    "PROTOCOL_VERSION",
    "QPARAMS_JSON_FILENAME",
    "RANGE_MULTIPLIERS",
    "SUMMARY_FILENAME",
    "TARGET",
    "TOKEN_METRICS_FILENAME",
    "CandidateRunResult",
    "MultiplierSpec",
    "aggregate_danger_zone_rows",
    "aggregate_token_metric_rows",
    "apply_l9_scale_multiplier_quantization",
    "build_counterfactual_qparams_manifest",
    "build_bucket_provenance_manifest",
    "build_multiplier_qparams_by_target",
    "build_protocol_manifest",
    "build_summary_rows",
    "discover_ffn_targets",
    "load_baseline_qparams_by_target",
    "load_l9_input_oer_oracle",
    "multiplier_specs",
    "run_engergybound_diagnostic",
    "run_multiplier_candidate",
    "write_audit_manifest",
]
