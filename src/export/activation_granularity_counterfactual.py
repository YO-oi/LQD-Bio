from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from src.export.export_fp32_core_cache import forward_core_outputs
    from src.metrics.semantic_geometry_metrics import (
        build_fp32_semantic_geometry_reference,
        build_semantic_geometry_summary_v2_long,
        compute_semantic_geometry_tables,
    )
    from src.quant.ffn_quant_injector import apply_fixed_qparams_to_quant_module, normalize_quant_mode
    from src.quant.quant_core import QuantizationController, register_layer_padding_mask_hooks
    from src.quant.quant_linear import QuantLinear
    from src.utils.semantic_geometry_cache import (
        DEFAULT_EMBEDDING_DIM,
        build_quant_embedding_cache,
        file_identity,
        qparams_identity_manifest,
        save_embedding_cache,
    )
    from src.utils.semantic_geometry_mode_runner import (
        activation_observer_config_for_granularity,
        build_calibration_loader,
        build_eval_loader,
        load_eval_metadata,
        load_or_calibrate_qparams,
        restore_clean_model,
        run_embedding_mode,
        save_qparams,
        validate_calibration_provenance,
    )
    from src.verify.ffn_quant_mode_ablation import (
        _build_reference_cache as build_quant_mode_reference_cache,
        _load_target_input_oer_oracle as load_target_input_oer_oracle,
        _normalise_oracle_topk_indices as normalise_oracle_topk_indices,
        _oer_metrics_from_topk_indices as oer_metrics_from_topk_indices,
        _relative_l2_per_sample as relative_l2_per_sample,
    )
except ImportError:  # pragma: no cover - direct script fallback.
    from ..export.export_fp32_core_cache import forward_core_outputs
    from ..metrics.semantic_geometry_metrics import (
        build_fp32_semantic_geometry_reference,
        build_semantic_geometry_summary_v2_long,
        compute_semantic_geometry_tables,
    )
    from ..quant.ffn_quant_injector import apply_fixed_qparams_to_quant_module, normalize_quant_mode
    from ..quant.quant_core import QuantizationController, register_layer_padding_mask_hooks
    from ..quant.quant_linear import QuantLinear
    from ..utils.semantic_geometry_cache import (
        DEFAULT_EMBEDDING_DIM,
        build_quant_embedding_cache,
        file_identity,
        qparams_identity_manifest,
        save_embedding_cache,
    )
    from ..utils.semantic_geometry_mode_runner import (
        activation_observer_config_for_granularity,
        build_calibration_loader,
        build_eval_loader,
        load_eval_metadata,
        load_or_calibrate_qparams,
        restore_clean_model,
        run_embedding_mode,
        save_qparams,
        validate_calibration_provenance,
    )
    from ..verify.ffn_quant_mode_ablation import (
        _build_reference_cache as build_quant_mode_reference_cache,
        _load_target_input_oer_oracle as load_target_input_oer_oracle,
        _normalise_oracle_topk_indices as normalise_oracle_topk_indices,
        _oer_metrics_from_topk_indices as oer_metrics_from_topk_indices,
        _relative_l2_per_sample as relative_l2_per_sample,
    )


EXPERIMENT_NAME = "activation_granularity_counterfactual"
PROTOCOL_VERSION = "activation_granularity_counterfactual.v2"
BASELINE_GRANULARITY = "per_tensor"
COUNTERFACTUAL_GRANULARITY = "per_channel"
QUANT_MODE = "W4A4"
ACTIVATION_HOOK_SEMANTIC = "FFN input activation"
INTERVENTION_TARGETS: tuple[tuple[int, str], ...] = ((3, "fc1"), (4, "fc1"), (0, "fc2"))
EXPECTED_ACTIVATION_QRANGE = (-7, 7)
TARGET_LINEAR_NAMES = ("fc1", "fc2")
GROUP_NAMES = ("overall", "many", "medium", "few", "tail")
DIAGNOSTIC_DELTA_EPS = 1e-6
LOCAL_MECHANISM_METRICS = (
    "clipping_rate",
    "strict_overflow_rate",
    "boundary_hit_rate",
    "near_zero_rate",
    "near_zero_pm1_rate",
    "activation_quant_error",
    "oer_retention",
    "oer_loss",
    "off_oracle_energy_ratio",
    "teacher_topk_relative_distortion",
    "topk_overlap",
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    intervention_targets: tuple[tuple[int, str], ...]
    isolated_or_joint: str

    @property
    def is_baseline(self) -> bool:
        return not self.intervention_targets


@dataclass(frozen=True)
class CounterfactualCandidateRunResult:
    qparams_by_target: dict[tuple[int, str], dict[str, Any]]
    embedding_cache: dict[str, Any]
    local_metric_rows: list[dict[str, Any]]
    runtime_summary: dict[str, Any]
    cache_path: Path | None = None


def candidate_specs(*, include_joint: bool = True) -> tuple[CandidateSpec, ...]:
    specs: list[CandidateSpec] = [
        CandidateSpec("baseline", (), "baseline"),
        CandidateSpec("L3.fc1-only", ((3, "fc1"),), "isolated"),
        CandidateSpec("L4.fc1-only", ((4, "fc1"),), "isolated"),
        CandidateSpec("L0.fc2-only", ((0, "fc2"),), "isolated"),
    ]
    if include_joint:
        specs.append(CandidateSpec("joint", INTERVENTION_TARGETS, "joint"))
    return tuple(specs)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else default


def _quant_cfg(cfg: Any) -> Mapping[str, Any]:
    quant_cfg = _cfg_get(cfg, "QUANT", {})
    if not isinstance(quant_cfg, Mapping):
        raise TypeError("cfg.QUANT must be a mapping")
    return quant_cfg


def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    dataset_cfg = _cfg_get(cfg, "CURRENT_DATASET", {})
    return dataset_cfg if isinstance(dataset_cfg, Mapping) else {}


def _dataset_root(cfg: Any) -> Path:
    configured = _nested_get(_dataset_cfg(cfg), "ROOT", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / "local_data"


def _runner_cfg(cfg: Any) -> Mapping[str, Any]:
    runner_cfg = _cfg_get(cfg, "ACTIVATION_GRANULARITY_COUNTERFACTUAL", {})
    return runner_cfg if isinstance(runner_cfg, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    runner_cfg = _runner_cfg(cfg)
    if key in runner_cfg:
        return runner_cfg[key]
    return _cfg_get(cfg, key, default)


def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", _cfg_get(cfg, "DEVICE", None))
    if configured is None:
        raise AttributeError(f"{EXPERIMENT_NAME} requires cfg.DEVICE or ACTIVATION_GRANULARITY_COUNTERFACTUAL['DEVICE']")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"cfg.DEVICE={configured!r} but CUDA is not available")
    return device


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        str(key): value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _split_csv(cfg: Any, split: str) -> Path:
    key = f"{split.upper()}_CSV"
    configured = _nested_get(_runner_cfg(cfg), key, None)
    if configured is not None:
        return Path(configured)
    configured = _nested_get(_dataset_cfg(cfg), key, None)
    if configured is not None:
        return Path(configured)
    raise AttributeError(f"Cannot resolve {key}")


def _result_dir(cfg: Any) -> Path:
    configured = _nested_get(_runner_cfg(cfg), "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return _dataset_root(cfg) / "export" / EXPERIMENT_NAME


def _checkpoint_path(cfg: Any) -> Path | None:
    beats = _cfg_get(cfg, "BEATS", {})
    if isinstance(beats, Mapping) and beats.get("CHECKPOINT") is not None:
        return Path(beats["CHECKPOINT"])
    for key in ("CHECKPOINT", "CHECKPOINT_PATH", "MODEL_CHECKPOINT"):
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return Path(value)
    return None


def _label_mapping_path(cfg: Any) -> Path | None:
    value = _nested_get(_dataset_cfg(cfg), "LABEL_MAPPING", None)
    return None if value is None else Path(value)


def _target_name(layer_idx: int, module_name: str) -> str:
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _target_key(layer_idx: int, module_name: str) -> tuple[int, str]:
    return int(layer_idx), str(module_name)


def _candidate_label(candidate: CandidateSpec) -> str:
    return candidate.name.replace(".", "_").replace("-", "_")


def _activation_qrange(cfg: Any) -> tuple[int, int]:
    activation_cfg = _quant_cfg(cfg).get("ACTIVATION", {})
    activation_cfg = activation_cfg if isinstance(activation_cfg, Mapping) else {}
    return int(activation_cfg.get("QMIN", -7)), int(activation_cfg.get("QMAX", 7))


def assert_quant_injection_qrange(cfg: Any) -> None:
    qrange = _activation_qrange(cfg)
    if qrange != EXPECTED_ACTIVATION_QRANGE:
        raise RuntimeError(
            f"{EXPERIMENT_NAME} is locked to Quant Injection qrange "
            f"{EXPECTED_ACTIVATION_QRANGE}; current activation qrange is {qrange}"
        )


def discover_ffn_targets(model: nn.Module, cfg: Any) -> tuple[tuple[int, str], ...]:
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise AttributeError("model.encoder.layers does not exist")
    quant_cfg = _quant_cfg(cfg)
    target_names = tuple(quant_cfg.get("TARGET_LINEAR_NAMES", TARGET_LINEAR_NAMES))
    if target_names != TARGET_LINEAR_NAMES:
        raise ValueError(f"Expected QUANT.TARGET_LINEAR_NAMES={TARGET_LINEAR_NAMES}, got {target_names!r}")
    targets: list[tuple[int, str]] = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        for module_name in target_names:
            module = getattr(layer, str(module_name), None)
            if isinstance(module, nn.Linear):
                targets.append((int(layer_idx), str(module_name)))
    if not targets:
        raise RuntimeError("No BEATs FFN fc1/fc2 Linear targets were found")
    return tuple(targets)


def _ffn_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    return model.encoder.layers[int(layer_idx)]


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


def _mode_uses_weight(quant_mode: str) -> bool:
    return normalize_quant_mode(quant_mode, None) in {"w4", "w4a4"}


def _mode_uses_activation(quant_mode: str) -> bool:
    return normalize_quant_mode(quant_mode, None) in {"a4", "w4a4"}


def _clone_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_value(item) for item in value)
    return value


def _clone_qparams(qparams: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _clone_value(value) for key, value in qparams.items()}


def diagnostic_qparams_path(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    activation_granularity: str,
) -> Path:
    return (
        _result_dir(cfg)
        / "diagnostic_qparams"
        / f"layer_{int(layer_idx):02d}_{module_name}_{activation_granularity}_qparams.pt"
    )


def load_or_calibrate_diagnostic_qparams(
    cfg: Any,
    *,
    layer_idx: int,
    module_name: str,
    activation_granularity: str,
    calibration_loader: DataLoader,
    force_recalibrate: bool = False,
) -> dict[str, Any]:
    path = diagnostic_qparams_path(
        cfg,
        layer_idx=int(layer_idx),
        module_name=str(module_name),
        activation_granularity=str(activation_granularity),
    )
    qparams = load_or_calibrate_qparams(
        cfg,
        layer_idx=int(layer_idx),
        module_name=str(module_name),
        calibration_loader=calibration_loader,
        activation_granularity=str(activation_granularity),
        force_recalibrate=bool(force_recalibrate),
        qparams_path=path,
    )
    qparams["diagnostic_only"] = True
    qparams["experiment_name"] = EXPERIMENT_NAME
    qparams["qparams_path"] = str(path)
    save_qparams(qparams, path)
    return qparams


def build_counterfactual_qparams_by_target(
    cfg: Any,
    *,
    target_modules: Sequence[tuple[int, str]],
    candidate: CandidateSpec,
    calibration_loader: DataLoader,
    force_recalibrate: bool = False,
) -> dict[tuple[int, str], dict[str, Any]]:
    assert_quant_injection_qrange(cfg)
    intervention_set = {_target_key(*target) for target in candidate.intervention_targets}
    baseline_by_target: dict[tuple[int, str], dict[str, Any]] = {}
    for layer_idx, module_name in target_modules:
        baseline = load_or_calibrate_diagnostic_qparams(
            cfg,
            layer_idx=int(layer_idx),
            module_name=str(module_name),
            activation_granularity=BASELINE_GRANULARITY,
            calibration_loader=calibration_loader,
            force_recalibrate=force_recalibrate,
        )
        validate_calibration_provenance(
            baseline,
            cfg,
            activation_granularity=BASELINE_GRANULARITY,
            require_activation=True,
            context=f"{_target_name(layer_idx, module_name)} baseline qparams",
        )
        baseline_by_target[_target_key(layer_idx, module_name)] = baseline

    result = {key: _clone_qparams(value) for key, value in baseline_by_target.items()}
    for layer_idx, module_name in sorted(intervention_set):
        key = _target_key(layer_idx, module_name)
        if key not in baseline_by_target:
            raise KeyError(f"Intervention target {_target_name(layer_idx, module_name)} is not in target_modules")
        per_channel = load_or_calibrate_diagnostic_qparams(
            cfg,
            layer_idx=layer_idx,
            module_name=module_name,
            activation_granularity=COUNTERFACTUAL_GRANULARITY,
            calibration_loader=calibration_loader,
            force_recalibrate=force_recalibrate,
        )
        validate_calibration_provenance(
            per_channel,
            cfg,
            activation_granularity=COUNTERFACTUAL_GRANULARITY,
            require_activation=True,
            context=f"{_target_name(layer_idx, module_name)} counterfactual activation qparams",
        )
        composed = _clone_qparams(baseline_by_target[key])
        composed["activation"] = _clone_value(per_channel["activation"])
        composed["activation_granularity"] = COUNTERFACTUAL_GRANULARITY
        composed["calibration_provenance"] = _clone_value(per_channel.get("calibration_provenance", {}))
        composed["qparams_path"] = per_channel.get("qparams_path")
        composed["baseline_weight_qparams_path"] = baseline_by_target[key].get("qparams_path")
        composed["counterfactual_activation_qparams_path"] = per_channel.get("qparams_path")
        composed["counterfactual_weight_source"] = "baseline_per_tensor_qparams_weight_section"
        composed["diagnostic_only"] = True
        composed["experiment_name"] = EXPERIMENT_NAME
        result[key] = composed
    return result


def apply_counterfactual_quantization(
    cfg: Any,
    *,
    model: nn.Module,
    target_modules: Sequence[tuple[int, str]],
    quant_mode: str,
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    intervention_targets: Sequence[tuple[int, str]],
    capture_targets: Sequence[tuple[int, str]] = INTERVENTION_TARGETS,
) -> tuple[dict[tuple[int, str], QuantLinear], list[QuantizationController]]:
    """Apply this experiment's mixed activation granularity without touching old runners."""

    intervention_set = {_target_key(*target) for target in intervention_targets}
    capture_set = {_target_key(*target) for target in capture_targets}
    normalized_mode = normalize_quant_mode(quant_mode, None)
    controller = QuantizationController(config=_quant_cfg(cfg), mode=normalized_mode)
    quant_modules: dict[tuple[int, str], QuantLinear] = {}
    hooked_layers: set[int] = set()
    seen: set[tuple[int, str]] = set()

    for layer_idx, module_name in target_modules:
        key = _target_key(layer_idx, module_name)
        if key in seen:
            raise ValueError(f"Duplicate target {_target_name(layer_idx, module_name)}")
        seen.add(key)
        target_granularity = COUNTERFACTUAL_GRANULARITY if key in intervention_set else BASELINE_GRANULARITY
        qparams = _qparams_for_target(qparams_by_target, layer_idx=layer_idx, module_name=module_name)
        validate_calibration_provenance(
            qparams,
            cfg,
            activation_granularity=target_granularity,
            require_activation=_mode_uses_activation(normalized_mode),
            context=f"{_target_name(layer_idx, module_name)} {target_granularity} runtime qparams",
        )

        layer = _ffn_layer(model, layer_idx)
        original = getattr(layer, module_name)
        if isinstance(original, QuantLinear):
            raise RuntimeError(f"{_target_name(layer_idx, module_name)} is already QuantLinear")
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{_target_name(layer_idx, module_name)} must be nn.Linear, got {type(original)!r}")

        wrapped = QuantLinear(
            original,
            quant_config=_quant_cfg(cfg),
            activation_observer_config=activation_observer_config_for_granularity(cfg, target_granularity),
            target_name=_target_name(layer_idx, module_name),
        )
        setattr(layer, module_name, wrapped)
        apply_fixed_qparams_to_quant_module(
            wrapped,
            mode=normalized_mode,
            weight_qparams=qparams.get("weight") if _mode_uses_weight(normalized_mode) else None,
            activation_qparams=qparams.get("activation") if _mode_uses_activation(normalized_mode) else None,
            require_qparams=True,
        )
        if key in capture_set and hasattr(wrapped, "set_integer_grid_capture"):
            wrapped.set_integer_grid_capture(
                True,
                scope=EXPERIMENT_NAME,
                save_device="cpu",
                detach=True,
                canonical_layout="BTC",
            )
        controller.register_module(_target_name(layer_idx, module_name), wrapped)
        if int(layer_idx) not in hooked_layers:
            register_layer_padding_mask_hooks(layer, controller=controller)
            hooked_layers.add(int(layer_idx))
        quant_modules[key] = wrapped

    controller.qparams_frozen = True
    controller.prepare_quant_forward()
    return quant_modules, [controller]


def make_counterfactual_quant_runner(
    intervention_targets: Sequence[tuple[int, str]],
    *,
    capture_targets: Sequence[tuple[int, str]] = INTERVENTION_TARGETS,
) -> Callable[..., tuple[list[QuantLinear], list[QuantizationController]]]:

    def _runner(
        *,
        cfg: Any,
        model: nn.Module,
        scope: str,
        target_modules: Sequence[tuple[int, str]],
        quant_mode: str,
        activation_granularity: str,
        qparams_by_target: Mapping[Any, Mapping[str, Any]],
        **_: Any,
    ) -> tuple[list[QuantLinear], list[QuantizationController]]:
        del scope, activation_granularity
        quant_modules, controllers = apply_counterfactual_quantization(
            cfg,
            model=model,
            target_modules=target_modules,
            quant_mode=quant_mode,
            qparams_by_target=qparams_by_target,
            intervention_targets=intervention_targets,
            capture_targets=capture_targets,
        )
        return list(quant_modules.values()), controllers

    return _runner


def _tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _scale_stats(prefix: str, value: Any) -> dict[str, Any]:
    tensor = _tensor(value, dtype=torch.float32).reshape(-1)
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) == 0:
        return {
            f"{prefix}_shape": tuple(int(item) for item in _tensor(value).shape),
            f"{prefix}_numel": int(tensor.numel()),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
        }
    return {
        f"{prefix}_shape": tuple(int(item) for item in _tensor(value).shape),
        f"{prefix}_numel": int(tensor.numel()),
        f"{prefix}_min": float(finite.min().item()),
        f"{prefix}_max": float(finite.max().item()),
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_std": float(finite.std(unbiased=False).item()) if int(finite.numel()) > 1 else 0.0,
    }


def build_local_qparams_rows(
    *,
    candidate: CandidateSpec,
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    target_modules: Sequence[tuple[int, str]] = INTERVENTION_TARGETS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    intervention_set = {_target_key(*target) for target in candidate.intervention_targets}
    for layer_idx, module_name in target_modules:
        qparams = _qparams_for_target(qparams_by_target, layer_idx=layer_idx, module_name=module_name)
        activation = qparams.get("activation", {})
        weight = qparams.get("weight", {})
        activation = activation if isinstance(activation, Mapping) else {}
        weight = weight if isinstance(weight, Mapping) else {}
        row = {
            "experiment_name": EXPERIMENT_NAME,
            "candidate": candidate.name,
            "isolated_or_joint": candidate.isolated_or_joint,
            "layer_idx": int(layer_idx),
            "module_name": str(module_name),
            "target_name": _target_name(layer_idx, module_name),
            "is_intervention_target": _target_key(layer_idx, module_name) in intervention_set,
            "activation_granularity": activation.get("granularity", qparams.get("activation_granularity", "")),
            "weight_granularity": weight.get("granularity", "per_channel"),
            "activation_qmin": activation.get("qmin", ""),
            "activation_qmax": activation.get("qmax", ""),
            "activation_zero_point_all_zero": bool(torch.all(_tensor(activation.get("zero_point", 0.0)) == 0).item()),
            "activation_channel_axis": activation.get("channel_axis", ""),
            "activation_qparams_path": qparams.get("counterfactual_activation_qparams_path", qparams.get("qparams_path", "")),
            "weight_qparams_source": qparams.get("counterfactual_weight_source", "own_qparams_weight_section"),
            "baseline_weight_qparams_path": qparams.get("baseline_weight_qparams_path", qparams.get("qparams_path", "")),
        }
        if "scale" in activation:
            row.update(_scale_stats("activation_scale", activation["scale"]))
        if "scale" in weight:
            row.update(_scale_stats("weight_scale", weight["scale"]))
        rows.append(row)
    return rows


def _metric_eps(cfg: Any) -> float:
    metrics_cfg = _cfg_get(cfg, "METRICS", {})
    metrics_cfg = metrics_cfg if isinstance(metrics_cfg, Mapping) else {}
    return float(_setting_get(cfg, "METRIC_EPS", metrics_cfg.get("METRIC_EPS", 1e-8)))


def _nan_vector(length: int) -> Tensor:
    return torch.full((int(length),), float("nan"), dtype=torch.float32)


def _batch_label_id(batch: Mapping[str, Any]) -> Tensor:
    value = batch.get("label_id")
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.long)
    return value.detach().cpu().long().reshape(-1)


def _batch_sample_ids(batch: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in batch["sample_id"]]


def _batch_labels_and_buckets(
    batch: Mapping[str, Any],
    *,
    sample_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[Any], list[str]]:
    sample_ids = _batch_sample_ids(batch)
    label_id = _batch_label_id(batch)
    if len(sample_ids) != int(label_id.numel()):
        raise RuntimeError("sample_id count must match label_id count")
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


def _as_btc_tensor(value: Any, *, batch_size: int, target_name: str, tensor_name: str) -> Tensor:
    tensor = _tensor(value, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"{target_name}: {tensor_name} must be canonical [B,T,C], got shape={tuple(tensor.shape)}")
    if int(tensor.shape[0]) != int(batch_size):
        raise ValueError(
            f"{target_name}: {tensor_name} batch mismatch, got {int(tensor.shape[0])} vs {int(batch_size)}"
        )
    return tensor.contiguous()


def _expand_valid_mask(valid_mask: Any, value_mask: Tensor, *, target_name: str, mask_name: str) -> Tensor:
    if valid_mask is None:
        return torch.ones_like(value_mask, dtype=torch.bool)
    valid = valid_mask.detach().cpu().bool() if isinstance(valid_mask, Tensor) else torch.as_tensor(valid_mask, dtype=torch.bool)
    if tuple(valid.shape) == tuple(value_mask.shape):
        return valid.contiguous()
    if tuple(valid.shape) == tuple(value_mask.shape[:2]):
        return valid.unsqueeze(-1).expand_as(value_mask).contiguous()
    raise ValueError(
        f"{target_name}: {mask_name} valid mask shape {tuple(valid.shape)} must match "
        f"{tuple(value_mask.shape)} or token shape {tuple(value_mask.shape[:2])}"
    )


def _per_sample_mask_rate(mask_value: Any, valid_mask: Any, *, batch_size: int, target_name: str, mask_name: str) -> Tensor:
    mask = mask_value.detach().cpu().bool() if isinstance(mask_value, Tensor) else torch.as_tensor(mask_value, dtype=torch.bool)
    if mask.ndim != 3 or int(mask.shape[0]) != int(batch_size):
        raise ValueError(f"{target_name}: {mask_name} must be [B,T,C], got shape={tuple(mask.shape)}")
    valid = _expand_valid_mask(valid_mask, mask, target_name=target_name, mask_name=mask_name)
    flat_mask = (mask & valid).reshape(batch_size, -1)
    flat_valid = valid.reshape(batch_size, -1)
    denom = flat_valid.sum(dim=1)
    values = flat_mask.sum(dim=1).float() / denom.clamp_min(1).float()
    return torch.where(denom > 0, values, torch.full_like(values, float("nan"))).detach().cpu().float()


def _integer_grid_metric_vectors(
    diagnostics: Mapping[str, Any] | None,
    *,
    batch_size: int,
    target_name: str,
    fallback_valid_mask: Any = None,
) -> dict[str, Tensor]:
    if diagnostics is None:
        return {
            "integer_grid_available": torch.zeros(batch_size, dtype=torch.float32),
            "clipping_rate": _nan_vector(batch_size),
            "strict_overflow_rate": _nan_vector(batch_size),
            "boundary_hit_rate": _nan_vector(batch_size),
            "near_zero_rate": _nan_vector(batch_size),
            "near_zero_pm1_rate": _nan_vector(batch_size),
        }

    qmin = diagnostics.get("qmin")
    qmax = diagnostics.get("qmax")
    if qmin is not None and qmax is not None and (int(qmin), int(qmax)) != EXPECTED_ACTIVATION_QRANGE:
        raise RuntimeError(f"{target_name}: integer-grid qrange must be {EXPECTED_ACTIVATION_QRANGE}, got {(qmin, qmax)}")
    valid_mask = diagnostics.get("valid_token_mask")
    if valid_mask is None:
        valid_mask = fallback_valid_mask
    strict = diagnostics.get("strict_overflow_mask", diagnostics.get("clip_mask"))
    boundary = diagnostics.get("boundary_hit_mask")
    exact_zero = diagnostics.get("exact_zero_mask", diagnostics.get("near_zero_mask"))
    near_zero_pm1 = diagnostics.get("near_zero_pm1_mask")
    missing = [
        name
        for name, value in (
            ("strict_overflow_mask", strict),
            ("boundary_hit_mask", boundary),
            ("exact_zero_mask", exact_zero),
            ("near_zero_pm1_mask", near_zero_pm1),
        )
        if value is None
    ]
    if missing:
        raise KeyError(f"{target_name}: integer-grid diagnostics missing masks: {missing}")
    strict_rate = _per_sample_mask_rate(
        strict,
        valid_mask,
        batch_size=batch_size,
        target_name=target_name,
        mask_name="strict_overflow_mask",
    )
    return {
        "integer_grid_available": torch.ones(batch_size, dtype=torch.float32),
        "clipping_rate": strict_rate,
        "strict_overflow_rate": strict_rate,
        "boundary_hit_rate": _per_sample_mask_rate(
            boundary,
            valid_mask,
            batch_size=batch_size,
            target_name=target_name,
            mask_name="boundary_hit_mask",
        ),
        "near_zero_rate": _per_sample_mask_rate(
            exact_zero,
            valid_mask,
            batch_size=batch_size,
            target_name=target_name,
            mask_name="exact_zero_mask",
        ),
        "near_zero_pm1_rate": _per_sample_mask_rate(
            near_zero_pm1,
            valid_mask,
            batch_size=batch_size,
            target_name=target_name,
            mask_name="near_zero_pm1_mask",
        ),
    }


def _topk_rows(topk_indices: Tensor | Sequence[Any], *, batch_size: int, target_name: str) -> list[Tensor]:
    if isinstance(topk_indices, Tensor):
        tensor = topk_indices.detach().cpu().long()
        if tensor.ndim == 1:
            if int(batch_size) != 1:
                raise ValueError(f"{target_name}: 1D top-k indices are only valid for batch_size=1")
            tensor = tensor.reshape(1, -1)
        else:
            tensor = tensor.reshape(int(tensor.shape[0]), -1)
        if int(tensor.shape[0]) != int(batch_size):
            raise ValueError(f"{target_name}: top-k row count {int(tensor.shape[0])} != batch_size {batch_size}")
        return [tensor[idx].reshape(-1) for idx in range(int(batch_size))]
    if isinstance(topk_indices, (str, bytes)):
        raise TypeError(f"{target_name}: top-k indices must be tensor rows, not a string")
    rows = [torch.as_tensor(item, dtype=torch.long).cpu().reshape(-1) for item in topk_indices]
    if len(rows) != int(batch_size):
        raise ValueError(f"{target_name}: top-k row count {len(rows)} != batch_size {batch_size}")
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
                raise ValueError(f"{target_name}: cannot slice 1D top-k indices except for a single-sample oracle")
            return tensor.reshape(1, -1)
        end = int(sample_offset) + int(batch_size)
        if end > int(tensor.shape[0]):
            raise ValueError(f"{target_name}: top-k tensor slice exceeds oracle rows")
        return tensor[int(sample_offset) : end].contiguous()
    if isinstance(topk_indices, (str, bytes)):
        raise TypeError(f"{target_name}: top-k indices must be tensor rows, not a string")
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
    expected = oracle_ids[int(sample_offset) : int(sample_offset) + len(batch_sample_ids)]
    if expected != [str(item) for item in batch_sample_ids]:
        raise RuntimeError(
            f"{target_name}: OER oracle sample_id order mismatch at offset {sample_offset}; "
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
        raise ValueError(f"{target_name}: OER activation shape mismatch {tuple(fp32.shape)} vs {tuple(quant.shape)}")
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
        if float(denom.item()) <= float(eps):
            value = fp32_flat.new_tensor(float("nan"))
        values.append(value.reshape(()))
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
    topk_rows = _topk_rows(topk_indices, batch_size=batch_size, target_name=target_name)
    oer_values = oer_metrics_from_topk_indices(fp32, quant, topk_rows, eps=eps)
    distortion, counts = _teacher_topk_relative_distortion(
        fp32,
        quant,
        topk_rows,
        eps=eps,
        target_name=target_name,
    )
    return {
        "oer_available": torch.ones(batch_size, dtype=torch.float32),
        "oer_retention": oer_values["oer_retention"],
        "oer_loss": oer_values["oer_loss"],
        "off_oracle_energy_ratio": oer_values["off_oracle_energy_ratio"],
        "teacher_topk_relative_distortion": distortion,
        "topk_overlap": _topk_overlap(fp32, quant, topk_rows, target_name=target_name),
        "teacher_topk_count": counts,
    }


def _local_sample_rows_for_target(
    *,
    candidate: CandidateSpec,
    layer_idx: int,
    module_name: str,
    quant_module: QuantLinear,
    batch_sample_ids: Sequence[str],
    batch_labels: Sequence[Any],
    batch_buckets: Sequence[str],
    sample_offset: int,
    oer_oracle: Mapping[str, Any] | None,
    eps: float,
) -> list[dict[str, Any]]:
    target_name = _target_name(layer_idx, module_name)
    batch_size = len(batch_sample_ids)
    fp32_activation, quant_activation = quant_module.get_oer_activation_tensors()
    if fp32_activation is None or quant_activation is None:
        raise RuntimeError(f"{target_name}: fake quant eval missing OER activation cache")
    fp32_btc = _as_btc_tensor(fp32_activation, batch_size=batch_size, target_name=target_name, tensor_name="fp32_activation")
    quant_btc = _as_btc_tensor(quant_activation, batch_size=batch_size, target_name=target_name, tensor_name="quant_activation")
    diagnostics = quant_module.get_integer_grid_diagnostics()
    fallback_valid_mask = getattr(quant_module, "last_activation_valid_mask_for_oer", None)
    integer_vectors = _integer_grid_metric_vectors(
        diagnostics,
        batch_size=batch_size,
        target_name=target_name,
        fallback_valid_mask=fallback_valid_mask,
    )
    activation_error = relative_l2_per_sample(fp32_btc, quant_btc, eps=eps)

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
    intervention_set = {_target_key(*target) for target in candidate.intervention_targets}
    for idx, sample_id in enumerate(batch_sample_ids):
        row = {
            "experiment_name": EXPERIMENT_NAME,
            "candidate": candidate.name,
            "isolated_or_joint": candidate.isolated_or_joint,
            "layer_idx": int(layer_idx),
            "module_name": str(module_name),
            "target_name": target_name,
            "is_intervention_target": _target_key(layer_idx, module_name) in intervention_set,
            "sample_id": str(sample_id),
            "label": batch_labels[idx],
            "bucket": str(batch_buckets[idx]),
            "integer_grid_available": _finite_float(integer_vectors["integer_grid_available"][idx]),
            "oer_available": _finite_float(oer_vectors["oer_available"][idx]),
            "activation_quant_error": _finite_float(activation_error[idx]),
            "teacher_topk_count": _finite_float(oer_vectors["teacher_topk_count"][idx]),
        }
        for metric_name in (
            "clipping_rate",
            "strict_overflow_rate",
            "boundary_hit_rate",
            "near_zero_rate",
            "near_zero_pm1_rate",
        ):
            row[metric_name] = _finite_float(integer_vectors[metric_name][idx])
        for metric_name in (
            "oer_retention",
            "oer_loss",
            "off_oracle_energy_ratio",
            "teacher_topk_relative_distortion",
            "topk_overlap",
        ):
            row[metric_name] = _finite_float(oer_vectors[metric_name][idx])
        rows.append(row)
    return rows


def _bucket_in_group(bucket: Any, group_name: str) -> bool:
    value = str(bucket).strip().lower()
    if group_name == "overall":
        return True
    if group_name == "tail":
        return value in {"medium", "few"}
    return value == str(group_name)


def _finite_mean(values: Sequence[Any]) -> float:
    finite = [_finite_float(item) for item in values]
    finite = [item for item in finite if math.isfinite(item)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _finite_count(values: Sequence[Any]) -> int:
    return sum(1 for item in values if math.isfinite(_finite_float(item)))


def build_local_metric_rows(
    *,
    candidate: CandidateSpec,
    qparams_by_target: Mapping[Any, Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    target_modules: Sequence[tuple[int, str]] = INTERVENTION_TARGETS,
) -> list[dict[str, Any]]:
    qparam_rows = build_local_qparams_rows(
        candidate=candidate,
        qparams_by_target=qparams_by_target,
        target_modules=target_modules,
    )
    sample_by_target: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for sample_row in sample_rows:
        key = _target_key(int(sample_row["layer_idx"]), str(sample_row["module_name"]))
        sample_by_target.setdefault(key, []).append(sample_row)

    rows: list[dict[str, Any]] = []
    for base_row in qparam_rows:
        key = _target_key(int(base_row["layer_idx"]), str(base_row["module_name"]))
        target_samples = sample_by_target.get(key, [])
        for group in GROUP_NAMES:
            group_samples = [row for row in target_samples if _bucket_in_group(row.get("bucket", ""), group)]
            sample_count = len(group_samples)
            integer_count = sum(1 for row in group_samples if _finite_float(row.get("integer_grid_available")) > 0.5)
            oer_count = sum(1 for row in group_samples if _finite_float(row.get("oer_available")) > 0.5)
            row = dict(base_row)
            row.update(
                {
                    "row_type": "aggregate",
                    "analysis_group": group,
                    "sample_count": int(sample_count),
                    "integer_grid_available_count": int(integer_count),
                    "oer_available_count": int(oer_count),
                    "local_metric_source": "QuantLinear.forward integer-grid diagnostics + OER activation cache",
                    "aggregation_unit": "sample",
                    "aggregation_fn": "nanmean",
                    "tail_definition": "tail = medium + few",
                }
            )
            for metric_name in LOCAL_MECHANISM_METRICS:
                values = [sample_row.get(metric_name) for sample_row in group_samples]
                row[metric_name] = _finite_mean(values)
                row[f"{metric_name}_valid_count"] = _finite_count(values)
            row["teacher_topk_count_mean"] = _finite_mean([sample_row.get("teacher_topk_count") for sample_row in group_samples])
            if sample_count <= 0:
                row["metric_status"] = "empty_group"
            elif integer_count < sample_count:
                row["metric_status"] = "partial_integer_grid_capture"
            elif oer_count <= 0:
                row["metric_status"] = "missing_oer_oracle"
            elif oer_count < sample_count:
                row["metric_status"] = "partial_oer_oracle"
            else:
                row["metric_status"] = "ok"
            rows.append(row)
    return rows


def _mask_for_group(buckets: Sequence[Any], group_name: str) -> Tensor:
    values = [str(item) for item in buckets]
    if group_name == "overall":
        return torch.ones(len(values), dtype=torch.bool)
    if group_name == "tail":
        return torch.tensor([item in {"medium", "few"} for item in values], dtype=torch.bool)
    return torch.tensor([item == group_name for item in values], dtype=torch.bool)


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


def compute_task_metric_rows(candidate: CandidateSpec, embedding_cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "logits" not in embedding_cache:
        return [
            {
                "experiment_name": EXPERIMENT_NAME,
                "candidate": candidate.name,
                "analysis_group": group,
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "support_count": 0,
                "metric_status": "missing_logits",
            }
            for group in GROUP_NAMES
        ]
    logits = _tensor(embedding_cache["logits"], dtype=torch.float32)
    labels = _tensor(embedding_cache["label_id"], dtype=torch.long).reshape(-1)
    predictions = logits.argmax(dim=1).detach().cpu().long().reshape(-1)
    if int(labels.numel()) != int(predictions.numel()):
        raise RuntimeError("label_id and logits.argmax sample counts do not match")
    buckets = list(embedding_cache["bucket"])
    rows: list[dict[str, Any]] = []
    for group in GROUP_NAMES:
        mask = _mask_for_group(buckets, group)
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
                "candidate": candidate.name,
                "analysis_group": group,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "support_count": support,
                "metric_status": "ok",
            }
        )
    return rows


def _with_candidate(row: Mapping[str, Any], candidate: CandidateSpec) -> dict[str, Any]:
    result = dict(row)
    result["experiment_name"] = EXPERIMENT_NAME
    result["candidate"] = candidate.name
    result["isolated_or_joint"] = candidate.isolated_or_joint
    result["intervention_targets"] = [_target_name(*target) for target in candidate.intervention_targets]
    return result


def _metric_lookup(rows: Sequence[Mapping[str, Any]], *, group: str, name: str, value_key: str) -> float:
    for row in rows:
        if str(row.get("analysis_group", "")) == group and str(row.get("metric_name", name)) == name:
            return _finite_float(row.get(value_key))
    return float("nan")


def _global_lookup(rows: Sequence[Mapping[str, Any]], *, name: str) -> float:
    for row in rows:
        if name in row:
            return _finite_float(row.get(name))
    return float("nan")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _summary_local_rows(rows: Sequence[Mapping[str, Any]], *, group: str) -> list[Mapping[str, Any]]:
    grouped = [row for row in rows if str(row.get("analysis_group", "")) == group]
    intervention_rows = [row for row in grouped if _truthy(row.get("is_intervention_target", False))]
    return intervention_rows if intervention_rows else grouped


def _local_metric_lookup(rows: Sequence[Mapping[str, Any]], *, group: str, metric: str) -> float:
    return _finite_mean([row.get(metric) for row in _summary_local_rows(rows, group=group)])


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
        "oer_retention_tail",
        "topk_overlap_tail",
        "relative_distortion_tail",
        "embedding_distortion_tail",
        "fisher_retention_tail",
        "variance_inflation_deviation_tail",
        "tail_collapse_gap",
    )
    if any(not math.isfinite(_finite_float(row.get(key))) for key in required):
        return "invalid"

    local_positive = any(
        (
            _positive_delta(row, "delta_OER", direction="higher"),
            _positive_delta(row, "delta_TopK_overlap", direction="higher"),
            _positive_delta(row, "delta_relative_distortion", direction="lower"),
            _positive_delta(row, "delta_clipping_rate_tail", direction="lower"),
            _positive_delta(row, "delta_strict_overflow_rate_tail", direction="lower"),
            _positive_delta(row, "delta_boundary_hit_rate_tail", direction="lower"),
            _positive_delta(row, "delta_near_zero_rate_tail", direction="lower"),
        )
    )
    local_negative = any(
        (
            _negative_delta(row, "delta_OER", direction="higher"),
            _negative_delta(row, "delta_TopK_overlap", direction="higher"),
            _negative_delta(row, "delta_relative_distortion", direction="lower"),
            _negative_delta(row, "delta_clipping_rate_tail", direction="lower"),
            _negative_delta(row, "delta_strict_overflow_rate_tail", direction="lower"),
            _negative_delta(row, "delta_boundary_hit_rate_tail", direction="lower"),
            _negative_delta(row, "delta_near_zero_rate_tail", direction="lower"),
        )
    )
    downstream_positive = any(
        (
            _positive_delta(row, "delta_embedding_distortion_tail", direction="lower"),
            _positive_delta(row, "delta_Fisher_retention", direction="higher"),
            _positive_delta(row, "delta_variance_inflation_deviation_tail", direction="lower"),
            _positive_delta(row, "delta_tail_collapse_gap", direction="lower"),
        )
    )
    downstream_negative = any(
        (
            _negative_delta(row, "delta_embedding_distortion_tail", direction="lower"),
            _negative_delta(row, "delta_Fisher_retention", direction="higher"),
            _negative_delta(row, "delta_variance_inflation_deviation_tail", direction="lower"),
            _negative_delta(row, "delta_tail_collapse_gap", direction="lower"),
        )
    )
    if local_positive and downstream_positive:
        return "local_and_downstream_positive"
    if local_positive and not downstream_negative:
        return "local_only"
    if downstream_positive or downstream_negative or local_negative:
        return "geometry_only_or_mixed"
    return "no_effect"


def _task_positive(row: Mapping[str, Any]) -> bool:
    return any(
        _positive_delta(row, key, direction="higher")
        for key in (
            "delta_overall_accuracy",
            "delta_many_accuracy",
            "delta_medium_accuracy",
            "delta_tail_accuracy",
            "delta_macro_f1",
            "delta_tail_macro_f1",
        )
    )


def build_summary_rows(
    *,
    candidate_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_name, tables in candidate_rows.items():
        local_rows = list(tables.get("local", ()))
        geometry_rows = list(tables.get("geometry_summary", ()))
        geometry_global_rows = list(tables.get("geometry_global", ()))
        task_rows = list(tables.get("task", ()))
        row = {
            "experiment_name": EXPERIMENT_NAME,
            "candidate": candidate_name,
            "clipping_rate_tail": _local_metric_lookup(local_rows, group="tail", metric="clipping_rate"),
            "strict_overflow_rate_tail": _local_metric_lookup(local_rows, group="tail", metric="strict_overflow_rate"),
            "boundary_hit_rate_tail": _local_metric_lookup(local_rows, group="tail", metric="boundary_hit_rate"),
            "near_zero_rate_tail": _local_metric_lookup(local_rows, group="tail", metric="near_zero_rate"),
            "near_zero_pm1_rate_tail": _local_metric_lookup(local_rows, group="tail", metric="near_zero_pm1_rate"),
            "oer_retention_tail": _local_metric_lookup(local_rows, group="tail", metric="oer_retention"),
            "topk_overlap_tail": _local_metric_lookup(local_rows, group="tail", metric="topk_overlap"),
            "teacher_topk_relative_distortion_tail": _local_metric_lookup(
                local_rows,
                group="tail",
                metric="teacher_topk_relative_distortion",
            ),
            "relative_distortion_tail": _local_metric_lookup(
                local_rows,
                group="tail",
                metric="teacher_topk_relative_distortion",
            ),
            "embedding_distortion_tail": _metric_lookup(
                geometry_rows, group="tail", name="embedding_distortion", value_key="value"
            ),
            "fisher_retention_tail": _metric_lookup(
                geometry_rows, group="tail", name="fisher_retention", value_key="value"
            ),
            "variance_inflation_deviation_tail": _metric_lookup(
                geometry_rows, group="tail", name="variance_inflation_deviation", value_key="value"
            ),
            "tail_collapse_gap": _global_lookup(geometry_global_rows, name="tail_collapse_gap"),
            "overall_accuracy": _task_lookup(task_rows, group="overall", metric="accuracy"),
            "many_accuracy": _task_lookup(task_rows, group="many", metric="accuracy"),
            "medium_accuracy": _task_lookup(task_rows, group="medium", metric="accuracy"),
            "tail_accuracy": _task_lookup(task_rows, group="tail", metric="accuracy"),
            "macro_f1": _task_lookup(task_rows, group="overall", metric="macro_f1"),
            "tail_macro_f1": _task_lookup(task_rows, group="tail", metric="macro_f1"),
            "diagnostic_status": "pending_delta",
            "task_positive": "",
        }
        rows.append(row)

    baseline = next((row for row in rows if row["candidate"] == "baseline"), None)
    if baseline is not None:
        delta_keys = (
            "clipping_rate_tail",
            "strict_overflow_rate_tail",
            "boundary_hit_rate_tail",
            "near_zero_rate_tail",
            "near_zero_pm1_rate_tail",
            "oer_retention_tail",
            "topk_overlap_tail",
            "teacher_topk_relative_distortion_tail",
            "relative_distortion_tail",
            "embedding_distortion_tail",
            "fisher_retention_tail",
            "variance_inflation_deviation_tail",
            "tail_collapse_gap",
            "overall_accuracy",
            "many_accuracy",
            "medium_accuracy",
            "tail_accuracy",
            "macro_f1",
            "tail_macro_f1",
        )
        for row in rows:
            for key in delta_keys:
                row[f"delta_{key}"] = _delta(row.get(key), baseline.get(key))
            row["delta_OER"] = row["delta_oer_retention_tail"]
            row["delta_TopK_overlap"] = row["delta_topk_overlap_tail"]
            row["delta_relative_distortion"] = row["delta_relative_distortion_tail"]
            row["delta_Fisher_retention"] = row["delta_fisher_retention_tail"]
            row["diagnostic_status"] = _classify_diagnostic_status(row)
            row["task_positive"] = bool(_task_positive(row))
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if int(tensor.numel()) == 1:
            return tensor.item()
        return tensor.reshape(-1).tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def build_protocol_manifest(
    cfg: Any,
    *,
    target_modules: Sequence[tuple[int, str]] | None = None,
    include_joint: bool = True,
    formal_experiment_ran: bool = False,
) -> dict[str, Any]:
    qmin, qmax = _activation_qrange(cfg)
    checkpoint = _checkpoint_path(cfg)
    label_mapping = _label_mapping_path(cfg)
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "output_protocol_version": 2,
        "diagnostic_only": True,
        "formal_experiment_ran": bool(formal_experiment_ran),
        "activation_baseline_granularity": BASELINE_GRANULARITY,
        "activation_counterfactual_granularity": COUNTERFACTUAL_GRANULARITY,
        "weight_granularity": "per_output_channel",
        "activation_hook_semantic": ACTIVATION_HOOK_SEMANTIC,
        "intervention_targets": [_target_name(*target) for target in INTERVENTION_TARGETS],
        "candidates": [
            {
                "name": spec.name,
                "isolated_or_joint": spec.isolated_or_joint,
                "intervention_targets": [_target_name(*target) for target in spec.intervention_targets],
            }
            for spec in candidate_specs(include_joint=include_joint)
        ],
        "target_modules": [] if target_modules is None else [_target_name(*target) for target in target_modules],
        "qmin": qmin,
        "qmax": qmax,
        "symmetric": True,
        "zero_point": 0,
        "activation_per_channel_definition": {
            "scale_shape": "[C]",
            "channel_dimension": "feature/hidden dimension, last dim of canonical [B,T,C] FFN input activation",
            "reduction_axes": ["batch", "token", "sample"],
            "calibration_source": "calibration.csv",
            "dynamic_forward_minmax": False,
        },
        "local_mechanism_metrics": {
            "aggregation_groups": list(GROUP_NAMES),
            "aggregation_unit": "sample",
            "integer_grid_source": "QuantLinear.last_integer_grid_diagnostics captured during quant forward",
            "oer_source": "validated fp32 oer_oracle_eval.pt input top-k scalar indices",
            "metrics": list(LOCAL_MECHANISM_METRICS),
            "definitions": {
                "clipping_rate": "per-sample mean of strict_overflow_mask over valid activation scalars",
                "strict_overflow_rate": "q_preclip outside [qmin, qmax] before clamp",
                "boundary_hit_rate": "q_int equals qmin or qmax after clamp",
                "near_zero_rate": "q_int equals zero_point",
                "near_zero_pm1_rate": "|q_int - zero_point| <= 1",
                "oer_retention": "existing scalar OER energy retention on fixed FP32 teacher top-k scalar positions",
                "teacher_topk_relative_distortion": "relative L2 distortion restricted to fixed FP32 teacher top-k scalar positions",
                "topk_overlap": "overlap between fixed FP32 teacher top-k scalar positions and quant activation top-k scalar positions",
            },
        },
        "summary_delta_fields": [
            "delta_OER",
            "delta_TopK_overlap",
            "delta_relative_distortion",
            "delta_tail_collapse_gap",
            "delta_embedding_distortion_tail",
            "delta_Fisher_retention",
            "delta_variance_inflation_deviation_tail",
            "delta_overall_accuracy",
            "delta_many_accuracy",
            "delta_medium_accuracy",
            "delta_tail_accuracy",
            "delta_macro_f1",
            "delta_tail_macro_f1",
        ],
        "diagnostic_status_labels": [
            "local_and_downstream_positive",
            "local_only",
            "geometry_only_or_mixed",
            "no_effect",
            "invalid",
        ],
        "calibration_csv": str(_split_csv(cfg, "calibration")),
        "eval_csv": str(_split_csv(cfg, "eval")),
        "test_touched": False,
        "checkpoint_identity": None if checkpoint is None else file_identity(checkpoint),
        "label_mapping_identity": None if label_mapping is None else file_identity(label_mapping),
        "output_files": {
            "summary": str(_result_dir(cfg) / "activation_granularity_summary.csv"),
            "local_metrics": str(_result_dir(cfg) / "activation_granularity_local_metrics.csv"),
            "geometry": str(_result_dir(cfg) / "activation_granularity_geometry.csv"),
            "task_metrics": str(_result_dir(cfg) / "activation_granularity_task_metrics.csv"),
            "manifest": str(_result_dir(cfg) / "activation_granularity_manifest.json"),
        },
        "run_commands": [
            "python -m src.export.activation_granularity_counterfactual --audit-only --write-audit-manifest",
            "python -m src.export.activation_granularity_counterfactual --run",
            "python -m src.export.activation_granularity_counterfactual --run --skip-joint",
        ],
    }
    if (qmin, qmax) != EXPECTED_ACTIVATION_QRANGE:
        manifest["protocol_status"] = "invalid_qrange"
        manifest["protocol_error"] = f"Expected qrange {EXPECTED_ACTIVATION_QRANGE}, got {(qmin, qmax)}"
    else:
        manifest["protocol_status"] = "ready"
    return manifest


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
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
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fields})
    os.replace(tmp, path)


def _candidate_granularity_label(candidate: CandidateSpec) -> str:
    if candidate.is_baseline:
        return "activation_per_tensor"
    return "activation_per_tensor_with_selected_per_channel"


def _remove_controller_hooks(controllers: Sequence[Any]) -> None:
    for controller in controllers:
        if hasattr(controller, "remove_padding_mask_hooks"):
            controller.remove_padding_mask_hooks()


def load_local_oer_oracles(
    cfg: Any,
    *,
    eval_loader: DataLoader,
    target_modules: Sequence[tuple[int, str]],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    target_set = {_target_key(*target) for target in target_modules}
    capture_targets = tuple(target for target in INTERVENTION_TARGETS if _target_key(*target) in target_set)
    missing_targets = [target for target in INTERVENTION_TARGETS if _target_key(*target) not in target_set]
    if missing_targets:
        raise RuntimeError(f"Intervention targets are missing from discovered FFN targets: {missing_targets}")

    try:
        reference_cache = build_quant_mode_reference_cache(cfg, eval_loader, capture_targets)
    except Exception as exc:  # noqa: BLE001 - keep geometry runnable and mark local OER invalid.
        return {}, {
            "status": "invalid",
            "error": str(exc),
            "required_targets": [_target_name(*target) for target in capture_targets],
        }

    state = reference_cache.get("reference_state", reference_cache.get("manifest", {}))
    state = dict(state) if isinstance(state, Mapping) else {"status": "unknown"}
    if not bool(reference_cache.get("enable_oer_metrics", False)):
        state["status"] = "oer_metrics_disabled"
        return {}, state

    oracles: dict[tuple[int, str], dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for layer_idx, module_name in capture_targets:
        target_name = _target_name(layer_idx, module_name)
        try:
            oracle = dict(
                load_target_input_oer_oracle(
                    reference_cache,
                    layer_idx=int(layer_idx),
                    module_name=str(module_name),
                )
            )
            raw_topk = oracle.get("input_topk_indices", oracle.get("topk_indices", oracle.get("flat_index")))
            oracle["input_topk_indices"] = normalise_oracle_topk_indices(
                raw_topk,
                target_name=target_name,
                oracle_key="input_topk_indices/topk_indices/flat_index",
            )
            oracles[_target_key(layer_idx, module_name)] = oracle
        except Exception as exc:  # noqa: BLE001 - target-level status is surfaced in local metrics.
            errors[target_name] = str(exc)
    if errors:
        state["status"] = "partial_target_oracle_error" if oracles else "target_oracle_error"
        state["target_oracle_errors"] = errors
    else:
        state["status"] = "validated"
    return oracles, state


def run_counterfactual_candidate(
    cfg: Any,
    *,
    candidate: CandidateSpec,
    target_modules: Sequence[tuple[int, str]],
    calibration_loader: DataLoader,
    eval_loader: DataLoader,
    sample_metadata: Mapping[str, Mapping[str, Any]],
    force_recalibrate: bool = False,
) -> tuple[dict[str, Any], Any]:
    qparams_by_target = build_counterfactual_qparams_by_target(
        cfg,
        target_modules=target_modules,
        candidate=candidate,
        calibration_loader=calibration_loader,
        force_recalibrate=force_recalibrate,
    )
    save_path = _result_dir(cfg) / "embedding_cache" / f"{_candidate_label(candidate)}_embedding.pt"
    result = run_embedding_mode(
        cfg,
        scope="full_model",
        target_modules=target_modules,
        quant_mode=QUANT_MODE,
        activation_granularity=BASELINE_GRANULARITY,
        qparams_by_target=qparams_by_target,
        calibration_loader=calibration_loader,
        eval_loader=eval_loader,
        sample_metadata=sample_metadata,
        save=True,
        save_path=save_path,
        custom_quant_runner=make_counterfactual_quant_runner(candidate.intervention_targets),
    )
    return qparams_by_target, result


def run_counterfactual_candidate_with_local_metrics(
    cfg: Any,
    *,
    candidate: CandidateSpec,
    target_modules: Sequence[tuple[int, str]],
    calibration_loader: DataLoader,
    eval_loader: DataLoader,
    sample_metadata: Mapping[str, Mapping[str, Any]],
    oer_oracles_by_target: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
    force_recalibrate: bool = False,
) -> CounterfactualCandidateRunResult:
    qparams_by_target = build_counterfactual_qparams_by_target(
        cfg,
        target_modules=target_modules,
        candidate=candidate,
        calibration_loader=calibration_loader,
        force_recalibrate=force_recalibrate,
    )
    device = _device(cfg)
    model = restore_clean_model(cfg).to(device)
    quant_modules, controllers = apply_counterfactual_quantization(
        cfg,
        model=model,
        target_modules=target_modules,
        quant_mode=QUANT_MODE,
        qparams_by_target=qparams_by_target,
        intervention_targets=candidate.intervention_targets,
        capture_targets=INTERVENTION_TARGETS,
    )
    capture_modules = {
        key: module
        for key, module in quant_modules.items()
        if key in {_target_key(*target) for target in INTERVENTION_TARGETS}
    }
    if set(capture_modules) != {_target_key(*target) for target in INTERVENTION_TARGETS}:
        missing = sorted({_target_key(*target) for target in INTERVENTION_TARGETS} - set(capture_modules))
        raise RuntimeError(f"Missing capture modules after quant injection: {missing}")

    sample_ids: list[str] = []
    labels: list[Any] = []
    buckets: list[str] = []
    label_chunks: list[Tensor] = []
    logits_chunks: list[Tensor] = []
    embedding_chunks: list[Tensor] = []
    local_sample_rows: list[dict[str, Any]] = []
    sample_offset = 0
    eps = _metric_eps(cfg)

    try:
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                batch_sample_ids, batch_labels, batch_buckets = _batch_labels_and_buckets(
                    batch,
                    sample_metadata=sample_metadata,
                )
                batch_label_id = _batch_label_id(batch)
                batch_size = int(batch_label_id.numel())
                moved = _move_batch_to_device(batch, device)
                logits, embedding, _token_padding_mask, _valid_token_count = forward_core_outputs(model, moved, cfg)
                if int(logits.shape[0]) != batch_size or int(embedding.shape[0]) != batch_size:
                    raise RuntimeError(
                        f"{candidate.name}: logits/embedding batch mismatch, "
                        f"logits={tuple(logits.shape)}, embedding={tuple(embedding.shape)}, batch={batch_size}"
                    )

                sample_ids.extend(batch_sample_ids)
                labels.extend(batch_labels)
                buckets.extend(batch_buckets)
                label_chunks.append(batch_label_id)
                logits_chunks.append(logits.detach().cpu().float().contiguous())
                embedding_chunks.append(embedding.detach().cpu().float().contiguous())

                for (layer_idx, module_name), quant_module in sorted(capture_modules.items()):
                    oracle = None if oer_oracles_by_target is None else oer_oracles_by_target.get((layer_idx, module_name))
                    local_sample_rows.extend(
                        _local_sample_rows_for_target(
                            candidate=candidate,
                            layer_idx=int(layer_idx),
                            module_name=str(module_name),
                            quant_module=quant_module,
                            batch_sample_ids=batch_sample_ids,
                            batch_labels=batch_labels,
                            batch_buckets=batch_buckets,
                            sample_offset=sample_offset,
                            oer_oracle=oracle,
                            eps=eps,
                        )
                    )

                sample_offset += batch_size
                for quant_module in capture_modules.values():
                    if hasattr(quant_module, "clear_oer_activation_tensors"):
                        quant_module.clear_oer_activation_tensors()
                del moved, logits, embedding

        if not sample_ids:
            raise RuntimeError("eval_loader produced no samples")
        if len(set(sample_ids)) != len(sample_ids):
            raise RuntimeError(f"{candidate.name}: duplicate sample_id values encountered")
        label_all = torch.cat(label_chunks, dim=0)
        logits_all = torch.cat(logits_chunks, dim=0)
        embedding_all = torch.cat(embedding_chunks, dim=0)
        expected_dim = int(_setting_get(cfg, "EXPECTED_EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM))
        granularity = _candidate_granularity_label(candidate)
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
            granularity=granularity,
            label_id=label_all,
            logits=logits_all,
            expected_dim=expected_dim,
            extra_manifest={
                "runner": EXPERIMENT_NAME,
                "mode_stage": "counterfactual_quant_embedding_export_with_local_metrics",
                "scope": "full_model",
                "activation_hook_semantic": ACTIVATION_HOOK_SEMANTIC,
                "activation_baseline_granularity": BASELINE_GRANULARITY,
                "activation_counterfactual_granularity": COUNTERFACTUAL_GRANULARITY,
                "intervention_targets": [_target_name(*target) for target in candidate.intervention_targets],
                "capture_targets": [_target_name(*target) for target in INTERVENTION_TARGETS],
                "target_modules": [_target_name(*target) for target in target_modules],
                "qparams_identity_by_target": qparams_identity_by_target,
                "integer_grid_capture": True,
                "oer_oracle_required_for": [
                    "oer_retention",
                    "teacher_topk_relative_distortion",
                    "topk_overlap",
                ],
            },
        )
        save_path = _result_dir(cfg) / "embedding_cache" / f"{_candidate_label(candidate)}_embedding.pt"
        cache_path = save_embedding_cache(cache, save_path, expected_dim=expected_dim)
        local_metric_rows = build_local_metric_rows(
            candidate=candidate,
            qparams_by_target=qparams_by_target,
            sample_rows=local_sample_rows,
        )
        return CounterfactualCandidateRunResult(
            qparams_by_target=qparams_by_target,
            embedding_cache=cache,
            local_metric_rows=local_metric_rows,
            cache_path=cache_path,
            runtime_summary={
                "runner": EXPERIMENT_NAME,
                "runtime": "counterfactual_local_metric_runner",
                "quant_mode": QUANT_MODE,
                "granularity": granularity,
                "device": str(device),
                "num_samples": len(sample_ids),
                "num_targets": len(target_modules),
                "capture_targets": [_target_name(*target) for target in INTERVENTION_TARGETS],
                "local_sample_rows": len(local_sample_rows),
                "local_metric_rows": len(local_metric_rows),
                "embedding_cache_path": str(cache_path),
            },
        )
    finally:
        _remove_controller_hooks(controllers)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_counterfactual_diagnostic(
    cfg: Any,
    *,
    include_joint: bool = True,
    force_recalibrate: bool = False,
) -> dict[str, Any]:
    assert_quant_injection_qrange(cfg)
    calibration_loader = build_calibration_loader(cfg)
    eval_loader = build_eval_loader(cfg)
    sample_metadata = load_eval_metadata(cfg)
    probe_model = restore_clean_model(cfg)
    target_modules = discover_ffn_targets(probe_model, cfg)
    del probe_model

    output_dir = _result_dir(cfg)
    oer_oracles_by_target, oer_reference_state = load_local_oer_oracles(
        cfg,
        eval_loader=eval_loader,
        target_modules=target_modules,
    )
    fp32_result = run_embedding_mode(
        cfg,
        quant_mode="none",
        eval_loader=eval_loader,
        sample_metadata=sample_metadata,
        save=True,
        save_path=output_dir / "embedding_cache" / "fp32_reference_embedding.pt",
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

    all_local_rows: list[dict[str, Any]] = []
    all_geometry_rows: list[dict[str, Any]] = []
    all_task_rows: list[dict[str, Any]] = []
    candidate_tables: dict[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {}
    qparams_identity_by_candidate: dict[str, Any] = {}

    for candidate in candidate_specs(include_joint=include_joint):
        result = run_counterfactual_candidate_with_local_metrics(
            cfg,
            candidate=candidate,
            target_modules=target_modules,
            calibration_loader=calibration_loader,
            eval_loader=eval_loader,
            sample_metadata=sample_metadata,
            oer_oracles_by_target=oer_oracles_by_target,
            force_recalibrate=force_recalibrate,
        )
        qparams_by_target = result.qparams_by_target
        qparams_identity_by_candidate[candidate.name] = {
            _target_name(*target): qparams_identity_manifest(qparams, source_hint=_target_name(*target))
            for target, qparams in qparams_by_target.items()
        }
        local_rows = result.local_metric_rows
        all_local_rows.extend(local_rows)
        cache = result.embedding_cache
        geometry_tables = compute_semantic_geometry_tables(
            fp32_reference=fp32_reference,
            quant_embeddings=cache["embedding"],
            quant_logits=cache.get("logits"),
            sample_id=cache["sample_id"],
            label_id=cache["label_id"],
            bucket=cache["bucket"],
            shot_group=cache.get("shot_group"),
            quant_mode=QUANT_MODE,
            granularity=(
                "activation_per_tensor"
                if candidate.is_baseline
                else "activation_per_tensor_with_selected_per_channel"
            ),
            require_logits=False,
        )
        geometry_summary = build_semantic_geometry_summary_v2_long(geometry_tables)
        geometry_rows = [_with_candidate(row, candidate) for row in geometry_summary]
        geometry_global_rows = [
            _with_candidate({"table_level": "global_level", **dict(row)}, candidate)
            for row in geometry_tables.get("global_level", [])
        ]
        all_geometry_rows.extend(geometry_rows)
        all_geometry_rows.extend(geometry_global_rows)
        task_rows = compute_task_metric_rows(candidate, cache)
        all_task_rows.extend(task_rows)
        candidate_tables[candidate.name] = {
            "local": local_rows,
            "geometry_summary": geometry_rows,
            "geometry_global": geometry_global_rows,
            "task": task_rows,
        }

    summary_rows = build_summary_rows(candidate_rows=candidate_tables)
    manifest = build_protocol_manifest(
        cfg,
        target_modules=target_modules,
        include_joint=include_joint,
        formal_experiment_ran=True,
    )
    manifest["qparams_identity_by_candidate"] = qparams_identity_by_candidate
    manifest["oer_reference_state"] = oer_reference_state
    manifest["oer_oracle_targets_loaded"] = [_target_name(*target) for target in sorted(oer_oracles_by_target)]
    manifest["row_counts"] = {
        "summary": len(summary_rows),
        "local_metrics": len(all_local_rows),
        "geometry": len(all_geometry_rows),
        "task_metrics": len(all_task_rows),
    }

    _write_csv(output_dir / "activation_granularity_summary.csv", summary_rows)
    _write_csv(output_dir / "activation_granularity_local_metrics.csv", all_local_rows)
    _write_csv(output_dir / "activation_granularity_geometry.csv", all_geometry_rows)
    _write_csv(output_dir / "activation_granularity_task_metrics.csv", all_task_rows)
    _write_json(output_dir / "activation_granularity_manifest.json", manifest)
    return manifest


def write_audit_manifest(cfg: Any, *, include_joint: bool = True) -> dict[str, Any]:
    probe_model = restore_clean_model(cfg)
    target_modules = discover_ffn_targets(probe_model, cfg)
    del probe_model
    manifest = build_protocol_manifest(
        cfg,
        target_modules=target_modules,
        include_joint=include_joint,
        formal_experiment_ran=False,
    )
    _write_json(_result_dir(cfg) / "activation_granularity_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--run", action="store_true", help="Run the formal counterfactual diagnostic.")
    parser.add_argument("--audit-only", action="store_true", help="Print protocol manifest without running eval.")
    parser.add_argument("--write-audit-manifest", action="store_true", help="Write audit-only manifest JSON.")
    parser.add_argument("--skip-joint", action="store_true", help="Skip the optional joint intervention.")
    parser.add_argument("--force-recalibrate", action="store_true", help="Recompute diagnostic qparams.")
    args = parser.parse_args(argv)

    import config as project_config

    include_joint = not bool(args.skip_joint)
    if args.run:
        manifest = run_counterfactual_diagnostic(
            project_config,
            include_joint=include_joint,
            force_recalibrate=bool(args.force_recalibrate),
        )
    elif args.write_audit_manifest:
        manifest = write_audit_manifest(project_config, include_joint=include_joint)
    else:
        manifest = build_protocol_manifest(project_config, include_joint=include_joint, formal_experiment_ran=False)
    print(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_GRANULARITY",
    "COUNTERFACTUAL_GRANULARITY",
    "EXPERIMENT_NAME",
    "INTERVENTION_TARGETS",
    "PROTOCOL_VERSION",
    "CandidateSpec",
    "CounterfactualCandidateRunResult",
    "apply_counterfactual_quantization",
    "assert_quant_injection_qrange",
    "build_counterfactual_qparams_by_target",
    "build_local_metric_rows",
    "build_local_qparams_rows",
    "build_protocol_manifest",
    "candidate_specs",
    "diagnostic_qparams_path",
    "discover_ffn_targets",
    "load_local_oer_oracles",
    "load_or_calibrate_diagnostic_qparams",
    "make_counterfactual_quant_runner",
    "run_counterfactual_candidate",
    "run_counterfactual_candidate_with_local_metrics",
    "run_counterfactual_diagnostic",
    "write_audit_manifest",
]
