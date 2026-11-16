from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline.export_eval import load_trained_checkpoint, pooled_logits_from_features, prepare_encoder_input_for_export
from src.quant.ffn_quant_injector import (
    get_ffn_module,
    make_single_ffn_quant_controller,
    replace_ffn_module_with_calibrated_quant,
)
from src.quant.quant_linear import QuantLinear
from src.train.train_beats_classifier_head import build_loader, dataset_paths
from src.utils.semantic_geometry_cache import file_identity, qparams_identity_manifest, stable_json_hash
from src.utils.semantic_geometry_mode_runner import load_qparams


EXPERIMENT_NAME = "single_layer_causal_injection"
PROTOCOL_VERSION = "single_layer_causal_injection.v1"
DEFAULT_DATASET = "local_data"
DEFAULT_CHECKPOINT_NAME = "BEATs_local_data.pt"
DEFAULT_QPARAMS_SUBDIR = Path("semantic_geometry") / "qparams"
DEFAULT_OUTPUT_SUBDIR = EXPERIMENT_NAME
QUANT_MODE = "w4a4"
EXPECTED_QRANGE = (-7, 7)
SUMMARY_GROUPS = ("overall", "many", "medium", "tail")


@dataclass(frozen=True)
class TargetSpec:
    layer_idx: int
    module_name: str

    @property
    def target(self) -> str:
        return f"L{int(self.layer_idx)}.{self.module_name}"

    @property
    def module_path(self) -> str:
        return f"encoder.layers.{int(self.layer_idx)}.{self.module_name}"

    @property
    def qparams_stem(self) -> str:
        return f"layer_{int(self.layer_idx):02d}_{self.module_name}"

    @property
    def activation_hook_position(self) -> str:
        return f"{self.module_name}_input_activation"


TARGET_SPECS = (
    TargetSpec(1, "fc1"),
    TargetSpec(4, "fc1"),
    TargetSpec(9, "fc2"),
)


SAMPLE_COLUMNS = [
    "sample_index",
    "sample_id",
    "audio_path",
    "label_id",
    "bucket",
    "target",
    "quantized_module",
    "activation_hook_position",
    "fp32_pred",
    "injected_pred",
    "j_star",
    "m_fp32",
    "m_injected",
    "margin_drop",
    "fp32_dynamic_margin",
    "injected_dynamic_margin",
    "classification_flip",
]


SUMMARY_COLUMNS = [
    "target",
    "group",
    "n_fp32_correct",
    "mean_margin_drop",
    "median_margin_drop",
    "margin_drop_std",
    "flip_count",
    "flip_rate",
]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if int(tensor.numel()) == 1:
            return tensor.reshape(()).item()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def _device_from_args(value: str | None) -> torch.device:
    configured = str(value or _cfg_get(project_config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={configured!r} requires CUDA, but CUDA is not available")
    return device


def _default_batch_size() -> int:
    cfg = _cfg_get(project_config, "FFN_LAYER_HEATMAP", {})
    if isinstance(cfg, Mapping):
        return int(cfg.get("EVAL_BATCH_SIZE", cfg.get("BATCH_SIZE", 8)))
    return 8


def _default_num_workers() -> int:
    cfg = _cfg_get(project_config, "FFN_LAYER_HEATMAP", {})
    if isinstance(cfg, Mapping):
        return int(cfg.get("NUM_WORKERS", 0))
    return 0


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(strict=False)


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    if str(args.dataset) != DEFAULT_DATASET:
        raise ValueError(f"{EXPERIMENT_NAME} only accepts dataset={DEFAULT_DATASET!r}; got {args.dataset!r}")
    paths = dataset_paths(DEFAULT_DATASET, None)
    checkpoint = Path(args.checkpoint) if args.checkpoint is not None else paths.output_checkpoint
    expected_checkpoint = paths.output_checkpoint
    if checkpoint.name != DEFAULT_CHECKPOINT_NAME or not _same_path(checkpoint, expected_checkpoint):
        raise ValueError(
            f"checkpoint must be the Quant Injection system checkpoint {expected_checkpoint}; got {checkpoint}. "
            "Seed FP32 teacher, Vanilla KD, QVF, PER, or copied checkpoints are refused."
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    qparams_root = Path(args.qparams_root) if args.qparams_root is not None else paths.root / DEFAULT_QPARAMS_SUBDIR
    output_dir = Path(args.output_dir) if args.output_dir is not None else paths.root / DEFAULT_OUTPUT_SUBDIR
    return {
        "dataset_root": paths.root,
        "eval_csv": paths.eval_csv,
        "train_csv": paths.train_csv,
        "test_csv": paths.test_csv,
        "full_csv": paths.full_csv,
        "label_mapping": paths.label_mapping,
        "checkpoint": checkpoint,
        "qparams_root": qparams_root,
        "output_dir": output_dir,
        "calibration_csv": Path(project_config.DATASETS[DEFAULT_DATASET]["CALIBRATION_CSV"]),
        "calibration_manifest": Path(project_config.DATASETS[DEFAULT_DATASET]["CALIBRATION_MANIFEST"]),
    }


def _load_eval_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise KeyError(f"eval csv missing sample_id column: {path}")
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"eval csv contains an empty sample_id: {path}")
            bucket = str(row.get("bucket", "")).strip().lower()
            if bucket not in {"many", "medium", "few"}:
                raise ValueError(f"{sample_id}: expected bucket many/medium/few, got {bucket!r}")
            rows[sample_id] = {
                "sample_id": sample_id,
                "label_id": int(row.get("label_id", -1)),
                "bucket": bucket,
                "audio_path": str(row.get("filepath") or row.get("audio_path") or row.get("path") or ""),
            }
    return rows


def _batch_sample_ids(batch: Mapping[str, Any]) -> list[str]:
    value = batch.get("sample_id")
    if value is None:
        raise KeyError("batch missing sample_id")
    if isinstance(value, Tensor):
        return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def _batch_audio_paths(batch: Mapping[str, Any], sample_ids: Sequence[str], metadata: Mapping[str, Mapping[str, Any]]) -> list[str]:
    for key in ("audio_path", "filepath", "path", "file_id"):
        value = batch.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Tensor)):
            return [str(item) for item in value]
    return [str(metadata.get(str(sample_id), {}).get("audio_path", "")) for sample_id in sample_ids]


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
    return result


def _set_no_param_grad(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _quant_modules(model: nn.Module) -> list[tuple[str, QuantLinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, QuantLinear)]


def _assert_clean_model(model: nn.Module, *, context: str) -> None:
    quantized = _quant_modules(model)
    if quantized:
        names = [name for name, _module in quantized]
        raise RuntimeError(f"{context}: clean checkpoint reload unexpectedly contains QuantLinear modules: {names[:8]}")


def _assert_no_param_grads(model: nn.Module, *, context: str) -> None:
    offenders = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if offenders:
        raise RuntimeError(f"{context}: expected no parameter gradients, found {offenders[:8]}")


def _as_float_tensor(value: Any, *, name: str) -> Tensor:
    tensor = value.detach().cpu().float() if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if int(tensor.numel()) <= 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains NaN/Inf")
    return tensor


def _zero_point_is_zero(value: Any) -> bool:
    tensor = _as_float_tensor(value, name="zero_point")
    return bool(torch.allclose(tensor, torch.zeros_like(tensor), atol=0.0, rtol=0.0))


def _validate_qparams(qparams: Mapping[str, Any], spec: TargetSpec) -> None:
    if str(qparams.get("target_name", spec.module_path)) != spec.module_path:
        raise ValueError(f"{spec.target}: qparams target mismatch: {qparams.get('target_name')!r}")
    activation = qparams.get("activation")
    weight = qparams.get("weight")
    if not isinstance(activation, Mapping):
        raise KeyError(f"{spec.target}: qparams missing activation section")
    if not isinstance(weight, Mapping):
        raise KeyError(f"{spec.target}: qparams missing weight section")
    if (int(activation.get("qmin")), int(activation.get("qmax"))) != EXPECTED_QRANGE:
        raise RuntimeError(f"{spec.target}: activation qrange must be {EXPECTED_QRANGE}")
    if (int(weight.get("qmin")), int(weight.get("qmax"))) != EXPECTED_QRANGE:
        raise RuntimeError(f"{spec.target}: weight qrange must be {EXPECTED_QRANGE}")
    if str(activation.get("granularity", "per_tensor")) != "per_tensor":
        raise RuntimeError(f"{spec.target}: activation granularity must be per_tensor")
    if str(weight.get("granularity", "per_channel")) not in {"per_channel", "per_output_channel"}:
        raise RuntimeError(f"{spec.target}: weight granularity must be per-output-channel/per_channel")
    if int(weight.get("channel_axis", 0)) != 0:
        raise RuntimeError(f"{spec.target}: weight channel_axis must be 0")
    if not bool(activation.get("symmetric", True)):
        raise RuntimeError(f"{spec.target}: activation qparams must be symmetric")
    if not bool(weight.get("symmetric", True)):
        raise RuntimeError(f"{spec.target}: weight qparams must be symmetric")
    if not _zero_point_is_zero(activation.get("zero_point", 0.0)):
        raise RuntimeError(f"{spec.target}: activation zero_point must be 0")
    if not _zero_point_is_zero(weight.get("zero_point", 0.0)):
        raise RuntimeError(f"{spec.target}: weight zero_point must be 0")
    _as_float_tensor(activation.get("scale"), name=f"{spec.target}.activation.scale")
    _as_float_tensor(weight.get("scale"), name=f"{spec.target}.weight.scale")


def _qparams_path(qparams_root: Path, spec: TargetSpec) -> Path:
    candidates = (
        qparams_root / f"{spec.qparams_stem}_per_tensor_qparams.pt",
        qparams_root / f"{spec.qparams_stem}_qparams.pt",
        qparams_root / f"{spec.qparams_stem}_fixed_qparams.pt",
    )
    for path in candidates:
        if path.exists():
            return path
    preview = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"missing frozen qparams for {spec.target}; this script will not recalibrate.\n"
        f"Tried:\n{preview}"
    )


def _load_qparams_by_target(qparams_root: Path) -> dict[TargetSpec, dict[str, Any]]:
    result: dict[TargetSpec, dict[str, Any]] = {}
    for spec in TARGET_SPECS:
        path = _qparams_path(qparams_root, spec)
        qparams = load_qparams(path)
        qparams["qparams_path"] = str(path)
        _validate_qparams(qparams, spec)
        result[spec] = qparams
    return result


def _load_clean_fp32_model(checkpoint_path: Path, *, device: torch.device, context: str) -> tuple[nn.Module, Mapping[str, Any]]:
    model, checkpoint = load_trained_checkpoint(checkpoint_path, device=device)
    _assert_clean_model(model, context=context)
    _set_no_param_grad(model)
    _assert_no_param_grads(model, context=context)
    model.eval()
    return model, checkpoint


def _enabled_quant_module_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, module in _quant_modules(model):
        if (
            bool(getattr(module, "fake_quant_enabled", False))
            or bool(getattr(module, "weight_quant_enabled", False))
            or bool(getattr(module, "activation_quant_enabled", False))
            or bool(getattr(module, "calibration_enabled", False))
        ):
            names.append(name)
    return names


def _assert_single_target_quant_state(model: nn.Module, spec: TargetSpec) -> list[dict[str, Any]]:
    quantized = _quant_modules(model)
    names = [name for name, _module in quantized]
    if names != [spec.module_path]:
        raise RuntimeError(
            f"{spec.target}: exactly one QuantLinear is allowed and it must be {spec.module_path}; got {names}"
        )
    enabled = _enabled_quant_module_names(model)
    if enabled != [spec.module_path]:
        raise RuntimeError(f"{spec.target}: non-target quant module entered quant state: enabled={enabled}")
    module = quantized[0][1]
    checks = {
        "fake_quant_enabled": bool(getattr(module, "fake_quant_enabled", False)),
        "weight_quant_enabled": bool(getattr(module, "weight_quant_enabled", False)),
        "activation_quant_enabled": bool(getattr(module, "activation_quant_enabled", False)),
        "calibration_enabled": bool(getattr(module, "calibration_enabled", False)),
        "qparams_frozen": bool(getattr(module, "qparams_frozen", False)),
        "fixed_weight_qparams_enabled": bool(getattr(module, "fixed_weight_qparams_enabled", False)),
    }
    if not checks["fake_quant_enabled"] or not checks["weight_quant_enabled"] or not checks["activation_quant_enabled"]:
        raise RuntimeError(f"{spec.target}: W4A4 fake quant is not fully enabled: {checks}")
    if checks["calibration_enabled"]:
        raise RuntimeError(f"{spec.target}: calibration must be disabled during injection forward: {checks}")
    if not checks["qparams_frozen"] or not checks["fixed_weight_qparams_enabled"]:
        raise RuntimeError(f"{spec.target}: fixed qparams are not active: {checks}")
    return [{"module_path": spec.module_path, **checks}]


def _prepare_injection_model(
    checkpoint_path: Path,
    qparams: Mapping[str, Any],
    *,
    spec: TargetSpec,
    device: torch.device,
) -> tuple[nn.Module, Any, list[dict[str, Any]]]:
    model, _checkpoint = _load_clean_fp32_model(
        checkpoint_path,
        device=device,
        context=f"{spec.target}.injection_clean_reload",
    )
    quant_module = replace_ffn_module_with_calibrated_quant(
        model,
        spec.layer_idx,
        spec.module_name,
        _cfg_get(project_config, "QUANT", {}),
        qparams,
        quant_mode=QUANT_MODE,
        enable_fake_quant=True,
    )
    if not isinstance(quant_module, QuantLinear):
        raise TypeError(f"{spec.target}: expected QuantLinear after injection")
    controller = make_single_ffn_quant_controller(
        model,
        quant_module,
        layer_idx=spec.layer_idx,
        module_name=spec.module_name,
        quant_config=_cfg_get(project_config, "QUANT", {}),
        quant_mode=QUANT_MODE,
    )
    controller.qparams_frozen = True
    controller.prepare_quant_forward()
    quant_state = _assert_single_target_quant_state(model, spec)
    return model, controller, quant_state


def _forward_logits(model: nn.Module, batch: Mapping[str, Any], *, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    _set_no_param_grad(model)
    moved = _move_batch_to_device(batch, device)
    labels = moved["label_id"].long()
    with torch.inference_mode():
        encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
            model,
            moved["waveform"],
            moved["padding_mask"],
        )
        features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
        logits = pooled_logits_from_features(model, features.float(), token_padding_mask)
    _assert_no_param_grads(model, context="forward_logits")
    return logits.detach().cpu().float(), labels.detach().cpu().long(), token_padding_mask.detach().cpu().bool()


def _competitor_indices(logits: Tensor, labels: Tensor) -> Tensor:
    work = logits.detach().clone()
    rows = torch.arange(int(work.shape[0]), device=work.device)
    work[rows, labels.long()] = -torch.inf
    return work.argmax(dim=1)


def _fixed_margin(logits: Tensor, labels: Tensor, competitors: Tensor) -> Tensor:
    rows = torch.arange(int(logits.shape[0]), device=logits.device)
    return logits[rows, labels.long()] - logits[rows, competitors.long()]


def _dynamic_margin(logits: Tensor, labels: Tensor) -> Tensor:
    competitors = _competitor_indices(logits, labels)
    return _fixed_margin(logits, labels, competitors)


@dataclass(frozen=True)
class TargetRunResult:
    spec: TargetSpec
    sample_order: tuple[str, ...]
    fp32_logits: Tensor
    fp32_preds: Tensor
    rows: tuple[dict[str, Any], ...]
    quant_state: tuple[dict[str, Any], ...]
    checkpoint_reloads: int


def _run_single_target(
    *,
    spec: TargetSpec,
    checkpoint_path: Path,
    qparams: Mapping[str, Any],
    loader: Any,
    device: torch.device,
    metadata: Mapping[str, Mapping[str, Any]],
) -> TargetRunResult:
    fp32_model, _fp32_checkpoint = _load_clean_fp32_model(
        checkpoint_path,
        device=device,
        context=f"{spec.target}.fp32_clean_reload",
    )
    injection_model, controller, quant_state = _prepare_injection_model(
        checkpoint_path,
        qparams,
        spec=spec,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    sample_order: list[str] = []
    fp32_logits_chunks: list[Tensor] = []
    fp32_pred_chunks: list[Tensor] = []
    sample_offset = 0
    try:
        for batch in loader:
            sample_ids = _batch_sample_ids(batch)
            audio_paths = _batch_audio_paths(batch, sample_ids, metadata)
            fp32_logits, labels, fp32_mask = _forward_logits(fp32_model, batch, device=device)
            injected_logits, injected_labels, injected_mask = _forward_logits(injection_model, batch, device=device)
            if not torch.equal(labels, injected_labels):
                raise RuntimeError(f"{spec.target}: FP32 and injected labels differ in the same eval batch")
            if not torch.equal(fp32_mask, injected_mask):
                raise RuntimeError(f"{spec.target}: FP32 and injected token padding_mask differ")

            fp32_preds = fp32_logits.argmax(dim=1)
            injected_preds = injected_logits.argmax(dim=1)
            j_star = _competitor_indices(fp32_logits, labels)
            m_fp32 = _fixed_margin(fp32_logits, labels, j_star)
            m_injected = _fixed_margin(injected_logits, labels, j_star)
            fp32_dynamic = _dynamic_margin(fp32_logits, labels)
            injected_dynamic = _dynamic_margin(injected_logits, labels)
            fp32_correct = fp32_preds.eq(labels)

            sample_order.extend(sample_ids)
            fp32_logits_chunks.append(fp32_logits)
            fp32_pred_chunks.append(fp32_preds.detach().cpu().long())

            for local_index, sample_id in enumerate(sample_ids):
                if not bool(fp32_correct[local_index].item()):
                    continue
                sid = str(sample_id)
                meta = metadata.get(sid, {})
                label_id = int(labels[local_index].item())
                injected_pred = int(injected_preds[local_index].item())
                row = {
                    "sample_index": int(sample_offset + local_index),
                    "sample_id": sid,
                    "audio_path": str(audio_paths[local_index]),
                    "label_id": label_id,
                    "bucket": str(meta.get("bucket", "")),
                    "target": spec.target,
                    "quantized_module": spec.module_path,
                    "activation_hook_position": spec.activation_hook_position,
                    "fp32_pred": int(fp32_preds[local_index].item()),
                    "injected_pred": injected_pred,
                    "j_star": int(j_star[local_index].item()),
                    "m_fp32": float(m_fp32[local_index].item()),
                    "m_injected": float(m_injected[local_index].item()),
                    "margin_drop": float((m_fp32[local_index] - m_injected[local_index]).item()),
                    "fp32_dynamic_margin": float(fp32_dynamic[local_index].item()),
                    "injected_dynamic_margin": float(injected_dynamic[local_index].item()),
                    "classification_flip": int(injected_pred != label_id),
                }
                rows.append(row)
            sample_offset += len(sample_ids)
        _assert_single_target_quant_state(injection_model, spec)
    finally:
        controller.remove_padding_mask_hooks()
    return TargetRunResult(
        spec=spec,
        sample_order=tuple(sample_order),
        fp32_logits=torch.cat(fp32_logits_chunks, dim=0).contiguous(),
        fp32_preds=torch.cat(fp32_pred_chunks, dim=0).contiguous(),
        rows=tuple(rows),
        quant_state=tuple(quant_state),
        checkpoint_reloads=2,
    )


def _assert_fp32_invariants(results: Sequence[TargetRunResult]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("no target results to validate")
    baseline = results[0]
    for result in results[1:]:
        if result.sample_order != baseline.sample_order:
            raise RuntimeError(
                f"FP32 sample order differs: {baseline.spec.target} vs {result.spec.target}"
            )
        if not torch.equal(result.fp32_logits, baseline.fp32_logits):
            diff = torch.max(torch.abs(result.fp32_logits - baseline.fp32_logits)).item()
            raise RuntimeError(
                f"FP32 logits are not exactly identical across targets: "
                f"{baseline.spec.target} vs {result.spec.target}, max_abs_diff={float(diff):.8g}"
            )
        if not torch.equal(result.fp32_preds, baseline.fp32_preds):
            raise RuntimeError(
                f"FP32 predictions differ across targets: {baseline.spec.target} vs {result.spec.target}"
            )
    return {
        "fp32_logits_identical_across_targets": True,
        "fp32_predictions_identical_across_targets": True,
        "sample_order_identical_across_targets": True,
        "sample_order_hash": stable_json_hash(list(baseline.sample_order)),
        "num_eval_samples_seen": int(len(baseline.sample_order)),
    }


def _summary_group_rows(rows: Sequence[Mapping[str, Any]], target: str, group: str) -> list[Mapping[str, Any]]:
    target_rows = [row for row in rows if str(row.get("target")) == str(target)]
    if group == "overall":
        return target_rows
    if group == "tail":
        return [row for row in target_rows if str(row.get("bucket", "")).lower() in {"medium", "few"}]
    return [row for row in target_rows if str(row.get("bucket", "")).lower() == group]


def _finite_margin_drops(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get("margin_drop", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _build_summary_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in TARGET_SPECS:
        for group in SUMMARY_GROUPS:
            rows = _summary_group_rows(sample_rows, spec.target, group)
            drops = _finite_margin_drops(rows)
            flips = int(sum(int(row.get("classification_flip", 0)) for row in rows))
            n = int(len(rows))
            result.append(
                {
                    "target": spec.target,
                    "group": group,
                    "n_fp32_correct": n,
                    "mean_margin_drop": float(np.mean(drops)) if drops else float("nan"),
                    "median_margin_drop": float(np.median(drops)) if drops else float("nan"),
                    "margin_drop_std": float(np.std(drops, ddof=0)) if drops else float("nan"),
                    "flip_count": flips,
                    "flip_rate": float(flips / n) if n > 0 else float("nan"),
                }
            )
    return result


def _build_manifest(
    *,
    paths: Mapping[str, Path],
    args: argparse.Namespace,
    device: torch.device,
    qparams_by_target: Mapping[TargetSpec, Mapping[str, Any]],
    results: Sequence[TargetRunResult],
    fp32_validation: Mapping[str, Any],
) -> dict[str, Any]:
    target_payloads: list[dict[str, Any]] = []
    for result in results:
        spec = result.spec
        qparams = qparams_by_target[spec]
        target_payloads.append(
            {
                "target": spec.target,
                "layer_idx": int(spec.layer_idx),
                "module_name": spec.module_name,
                "actual_quantized_module": spec.module_path,
                "activation_hook_position": spec.activation_hook_position,
                "quantized_weight": True,
                "quantized_activation": "module_input_activation_only",
                "other_modules_fp32": True,
                "full_w4a4_forward_used": False,
                "clean_checkpoint_reloaded_before_target": True,
                "checkpoint_reloads_for_target": int(result.checkpoint_reloads),
                "quant_state": list(result.quant_state),
                "qparams_identity": qparams_identity_manifest(qparams, source_hint=spec.target),
            }
        )
    return {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "analysis_only": True,
        "dataset": DEFAULT_DATASET,
        "checkpoint": str(paths["checkpoint"].resolve(strict=False)),
        "checkpoint_name_required": DEFAULT_CHECKPOINT_NAME,
        "checkpoint_sha256": _file_sha256(paths["checkpoint"]),
        "analysis_split": "eval",
        "eval_csv": file_identity(paths["eval_csv"]),
        "test_touched": False,
        "train_touched": False,
        "test_csv_not_read": True,
        "training_performed": False,
        "recalibration_performed": False,
        "same_eval_csv": True,
        "same_sample_order": True,
        "same_waveform": True,
        "crop_mode": "center",
        "same_padding_mask": True,
        "same_pooling": "mask_aware_mean",
        "same_classifier": True,
        "checkpoint_reload_policy": "fresh clean FP32 checkpoint reload for every target before injection",
        "same_model_instance_reused_across_targets": False,
        "weight_data_copy_used": False,
        "full_w4a4_forward_used": False,
        "full_w4a4_intermediate_activation_used_as_injection_input": False,
        "quantization": {
            "scope": "single_ffn_module_per_experiment",
            "mode": QUANT_MODE.upper(),
            "activation": {
                "bits": 4,
                "granularity": "per_tensor",
                "symmetric": True,
                "qmin": EXPECTED_QRANGE[0],
                "qmax": EXPECTED_QRANGE[1],
                "zero_point": 0,
            },
            "weight": {
                "bits": 4,
                "granularity": "per_output_channel",
                "symmetric": True,
                "qmin": EXPECTED_QRANGE[0],
                "qmax": EXPECTED_QRANGE[1],
                "zero_point": 0,
            },
            "non_target_layers_quantized": False,
            "target_predecessor_layers_quantized": False,
        },
        "qparams_root": str(paths["qparams_root"].resolve(strict=False)),
        "calibration_csv": file_identity(paths["calibration_csv"]),
        "calibration_manifest": file_identity(paths["calibration_manifest"]),
        "targets": target_payloads,
        "validations": {
            **dict(fp32_validation),
            "only_target_module_quantized_per_experiment": True,
            "clean_checkpoint_reload_per_target_verified": True,
            "test_touched": False,
        },
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "outputs": {
            "sample_csv": str((paths["output_dir"] / "single_layer_causal_injection.csv").resolve(strict=False)),
            "summary_csv": str((paths["output_dir"] / "summary.csv").resolve(strict=False)),
            "manifest_json": str((paths["output_dir"] / "single_layer_causal_injection_manifest.json").resolve(strict=False)),
        },
    }


def _print_report(sample_rows: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]], paths: Mapping[str, Path]) -> None:
    print("[Single-Layer Causal Injection]")
    print(f"checkpoint: {paths['checkpoint']}")
    print(f"output_dir: {paths['output_dir']}")
    print("test_touched: False")
    print(f"fp32_correct_rows: {len(sample_rows)}")
    for row in summary_rows:
        if row.get("group") != "overall":
            continue
        print(
            f"{row['target']}: n={row['n_fp32_correct']} "
            f"mean_margin_drop={_csv_value(row['mean_margin_drop'])} "
            f"flip_count={row['flip_count']} flip_rate={_csv_value(row['flip_rate'])}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _resolve_paths(args)
    device = _device_from_args(args.device)
    metadata = _load_eval_metadata(paths["eval_csv"])
    qparams_by_target = _load_qparams_by_target(paths["qparams_root"])
    loader = build_loader(
        dataset_paths(DEFAULT_DATASET, None),
        split="eval",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    results: list[TargetRunResult] = []
    for spec in TARGET_SPECS:
        result = _run_single_target(
            spec=spec,
            checkpoint_path=paths["checkpoint"],
            qparams=qparams_by_target[spec],
            loader=loader,
            device=device,
            metadata=metadata,
        )
        results.append(result)

    fp32_validation = _assert_fp32_invariants(results)
    sample_rows = [row for result in results for row in result.rows]
    summary_rows = _build_summary_rows(sample_rows)
    manifest = _build_manifest(
        paths=paths,
        args=args,
        device=device,
        qparams_by_target=qparams_by_target,
        results=results,
        fp32_validation=fp32_validation,
    )

    output_dir = paths["output_dir"]
    _write_csv(output_dir / "single_layer_causal_injection.csv", sample_rows, SAMPLE_COLUMNS)
    _write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    _write_json(output_dir / "single_layer_causal_injection_manifest.json", manifest)
    _print_report(sample_rows, summary_rows, paths)
    return {
        "output_dir": str(output_dir),
        "sample_rows": int(len(sample_rows)),
        "summary_rows": int(len(summary_rows)),
        "fp32_validation": dict(fp32_validation),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Only local_data is accepted.")
    parser.add_argument("--checkpoint", default=None, help=f"Must be local_data/{DEFAULT_CHECKPOINT_NAME}.")
    parser.add_argument("--qparams-root", default=None, help="Directory containing frozen W4A4 qparams.")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV and manifest outputs.")
    parser.add_argument("--device", default=None, help="Override config.DEVICE, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=_default_batch_size())
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument("--seed", type=int, default=int(_cfg_get(project_config, "SPLIT_SEED", 42)))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
