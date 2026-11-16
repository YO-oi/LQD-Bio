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
    apply_fixed_qparams_to_quant_module,
    get_ffn_module,
)
from src.quant.quant_core import prepare_beats_ffn_int4_quantization_from_project_config
from src.quant.quant_linear import QuantLinear
from src.train.train_beats_classifier_head import build_loader, dataset_paths
from src.utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
from src.utils.semantic_geometry_cache import file_identity, qparams_identity_manifest, stable_json_hash
from src.utils.semantic_geometry_mode_runner import load_qparams


EXPERIMENT_NAME = "decision_relevant_quantization_perturbation_sanity"
PROTOCOL_VERSION = "decision_relevant_quantization_perturbation_sanity.v1"
DEFAULT_DATASET = "local_data"
DEFAULT_CHECKPOINT_NAME = "BEATs_local_data.pt"
DEFAULT_QPARAMS_SUBDIR = Path("semantic_geometry") / "qparams"
DEFAULT_OUTPUT_SUBDIR = EXPERIMENT_NAME
QUANT_MODE = "w4a4"
EXPECTED_QRANGE = (-7, 7)
EPS = 1e-12
MAIN_GROUPS = ("overall_fp32_correct", "many", "medium", "tail")
ALL_GROUPS = (*MAIN_GROUPS, "few")


@dataclass(frozen=True)
class TargetSpec:
    layer_idx: int
    module_name: str
    hook_position: str

    @property
    def target_name(self) -> str:
        return f"encoder.layers.{int(self.layer_idx)}.{self.module_name}"

    @property
    def public_name(self) -> str:
        return f"L{int(self.layer_idx)}.{self.module_name}.{self.hook_position}"

    @property
    def qparams_stem(self) -> str:
        return f"layer_{int(self.layer_idx):02d}_{self.module_name}"


TARGET_SPECS = (
    TargetSpec(1, "fc1", "input_activation"),
    TargetSpec(4, "fc1", "input_activation"),
    TargetSpec(9, "fc2", "output_representation"),
)


SAMPLE_COLUMNS = [
    "sample_index",
    "sample_id",
    "audio_path",
    "label_id",
    "label_name",
    "bucket",
    "group_tail_included",
    "target",
    "layer_idx",
    "module_name",
    "hook_position",
    "fp32_pred",
    "int4_pred",
    "fp32_correct",
    "int4_correct",
    "fp32_correct_int4_flip",
    "j_star",
    "m_fp32_fixed",
    "m_int4_fixed",
    "m_fp32_dynamic",
    "m_int4_dynamic",
    "actual_fixed_margin_change",
    "actual_fixed_margin_drop",
    "valid_token_count",
    "D_abs",
    "D_rel",
    "mean_gradient_norm",
    "signed_projection_sum",
    "linear_margin_damage",
    "harmful_projection_sum",
    "beneficial_projection_sum",
    "harmful_projection_ratio",
    "linear_margin_damage_ratio",
    "signed_risk_ratio",
    "topk_overlap_at_k",
    "topk_k",
]

TOKEN_COLUMNS = [
    "sample_index",
    "sample_id",
    "label_id",
    "bucket",
    "target",
    "layer_idx",
    "module_name",
    "hook_position",
    "token_index",
    "fp32_norm",
    "int4_norm",
    "delta_norm",
    "relative_distortion",
    "gradient_norm",
    "decision_projection",
    "harmful_projection",
    "beneficial_projection",
]

COUPLING_COLUMNS = [
    "group",
    "target",
    "metric",
    "against",
    "sample_count",
    "pearson",
    "spearman",
    "mean",
    "median",
    "std",
    "fp32_correct_int4_flip_count",
    "fp32_correct_int4_nonflip_count",
    "flip_mean",
    "nonflip_mean",
    "flip_median",
    "nonflip_median",
    "flip_vs_nonflip_auroc",
]

TAYLOR_COLUMNS = [
    "group",
    "target",
    "sample_count",
    "mae",
    "rmse",
    "pearson",
    "spearman",
    "sign_agreement",
    "mean_signed_projection_sum",
    "mean_actual_fixed_margin_change",
    "approximation_status",
]

COUNTEREXAMPLE_COLUMNS = [
    "case_type",
    "target",
    "sample_id",
    "paired_sample_id",
    "bucket",
    "paired_bucket",
    "fp32_correct_int4_flip",
    "paired_fp32_correct_int4_flip",
    "D_rel",
    "paired_D_rel",
    "actual_fixed_margin_drop",
    "paired_actual_fixed_margin_drop",
    "linear_margin_damage",
    "paired_linear_margin_damage",
    "harmful_projection_sum",
    "paired_harmful_projection_sum",
    "note",
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, str)):
        return value
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    return value


def _device_from_args(value: str | None) -> torch.device:
    configured = str(value or _cfg_get(project_config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={configured!r} requires CUDA, but torch.cuda.is_available() is False")
    return device


def _default_batch_size() -> int:
    cfg = _cfg_get(project_config, "FFN_TOKEN_SURVIVAL_PROBE", {})
    if isinstance(cfg, Mapping):
        return int(cfg.get("EVAL_BATCH_SIZE", cfg.get("BATCH_SIZE", 8)))
    return 8


def _default_num_workers() -> int:
    cfg = _cfg_get(project_config, "FFN_TOKEN_SURVIVAL_PROBE", {})
    if isinstance(cfg, Mapping):
        return int(cfg.get("NUM_WORKERS", 0))
    return 0


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    if str(args.dataset) != DEFAULT_DATASET:
        raise ValueError(
            f"{EXPERIMENT_NAME} is a local_data sanity check only; got dataset={args.dataset!r}. "
            "It must not be used as a PER formal result."
        )
    paths = dataset_paths(DEFAULT_DATASET, None)
    checkpoint = Path(args.checkpoint) if args.checkpoint is not None else paths.output_checkpoint
    if checkpoint.name != DEFAULT_CHECKPOINT_NAME:
        raise ValueError(
            f"checkpoint must be {DEFAULT_CHECKPOINT_NAME}; got {checkpoint}. "
            "FP32 teacher seed/KD/QVF/PER checkpoints are intentionally refused."
        )
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


def _load_label_names(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"label_mapping must be a non-empty mapping: {path}")
    result = {int(value): str(key) for key, value in raw.items()}
    if len(result) != len(raw):
        raise ValueError(f"label_mapping contains duplicate label ids: {path}")
    return result


def _load_eval_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sample_id" not in (reader.fieldnames or []):
            raise KeyError(f"eval csv missing sample_id column: {path}")
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"eval csv contains empty sample_id: {path}")
            bucket = str(row.get("bucket", "")).strip().lower()
            if bucket not in {"many", "medium", "few"}:
                raise ValueError(f"{sample_id}: expected bucket many/medium/few, got {bucket!r}")
            label_id = int(row.get("label_id", -1))
            rows[sample_id] = {
                "sample_id": sample_id,
                "bucket": bucket,
                "label_id": label_id,
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
    target_name = spec.target_name
    if str(qparams.get("target_name", target_name)) != target_name:
        raise ValueError(f"{target_name}: qparams target mismatch: {qparams.get('target_name')!r}")
    activation = qparams.get("activation")
    weight = qparams.get("weight")
    if not isinstance(activation, Mapping):
        raise KeyError(f"{target_name}: qparams missing activation section")
    if not isinstance(weight, Mapping):
        raise KeyError(f"{target_name}: qparams missing weight section")
    if (int(activation.get("qmin")), int(activation.get("qmax"))) != EXPECTED_QRANGE:
        raise RuntimeError(f"{target_name}: activation qrange must be {EXPECTED_QRANGE}")
    if (int(weight.get("qmin")), int(weight.get("qmax"))) != EXPECTED_QRANGE:
        raise RuntimeError(f"{target_name}: weight qrange must be {EXPECTED_QRANGE}")
    if str(activation.get("granularity", "per_tensor")) != "per_tensor":
        raise RuntimeError(f"{target_name}: activation granularity must be per_tensor")
    if str(weight.get("granularity", "per_channel")) not in {"per_channel", "per_output_channel"}:
        raise RuntimeError(f"{target_name}: weight granularity must be per output channel")
    if int(weight.get("channel_axis", 0)) != 0:
        raise RuntimeError(f"{target_name}: weight channel_axis must be 0")
    if not bool(activation.get("symmetric", True)) or not bool(weight.get("symmetric", True)):
        raise RuntimeError(f"{target_name}: qparams must be symmetric")
    if not _zero_point_is_zero(activation.get("zero_point", 0.0)):
        raise RuntimeError(f"{target_name}: activation zero_point must be 0")
    if not _zero_point_is_zero(weight.get("zero_point", 0.0)):
        raise RuntimeError(f"{target_name}: weight zero_point must be 0")
    _as_float_tensor(activation.get("scale"), name=f"{target_name}.activation.scale")
    _as_float_tensor(weight.get("scale"), name=f"{target_name}.weight.scale")


def _target_key(spec: TargetSpec | tuple[int, str]) -> tuple[int, str]:
    if isinstance(spec, TargetSpec):
        return (int(spec.layer_idx), str(spec.module_name))
    layer_idx, module_name = spec
    return (int(layer_idx), str(module_name))


def _public_quant_target_name(key: tuple[int, str]) -> str:
    layer_idx, module_name = key
    return f"L{int(layer_idx)}.{module_name}"


def _module_path_for_key(key: tuple[int, str]) -> str:
    layer_idx, module_name = key
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _target_spec_from_module_path(module_path: str) -> TargetSpec:
    parts = str(module_path).split(".")
    if len(parts) != 4 or parts[0] != "encoder" or parts[1] != "layers":
        raise ValueError(f"unexpected quantized module path: {module_path!r}")
    return TargetSpec(int(parts[2]), str(parts[3]), "full_quant_module")


def _all_ffn_specs_for_model(model: nn.Module) -> tuple[TargetSpec, ...]:
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        raise AttributeError("model.encoder.layers is required for full-W4A4 FFN quantization")
    specs: list[TargetSpec] = []
    for layer_idx, layer in enumerate(layers):
        for module_name in ("fc1", "fc2"):
            if not hasattr(layer, module_name):
                raise AttributeError(f"encoder.layers.{layer_idx}.{module_name} is missing")
            specs.append(TargetSpec(int(layer_idx), module_name, "full_quant_module"))
    if not specs:
        raise RuntimeError("no FFN fc1/fc2 targets found for full-W4A4 quantization")
    return tuple(specs)


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
        f"missing frozen qparams for {spec.public_name}. This script will not recalibrate.\n"
        f"Tried:\n{preview}"
    )


def _load_qparams_by_target(qparams_root: Path, specs: Sequence[TargetSpec]) -> dict[tuple[int, str], dict[str, Any]]:
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for spec in specs:
        key = _target_key(spec)
        if key in result:
            continue
        path = _qparams_path(qparams_root, spec)
        qparams = load_qparams(path)
        qparams["qparams_path"] = str(path)
        _validate_qparams(qparams, spec)
        result[key] = qparams
    return result


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


def _canonicalize(value: Tensor, token_padding_mask: Tensor, *, context: str) -> tuple[Tensor, dict[str, Any]]:
    canonical = canonicalize_activation_btc(
        value,
        padding_mask=token_padding_mask,
        batch_size=int(token_padding_mask.shape[0]),
        token_count=int(token_padding_mask.shape[1]),
        context=context,
    )
    return canonical.tensor, layout_info_to_public_fields(canonical.layout_info)


def _canonicalize_like(value: Tensor, token_padding_mask: Tensor, layout_info: Mapping[str, Any], *, context: str) -> Tensor:
    raw_layout = str(layout_info.get("original_layout") or "")
    if not raw_layout:
        raise RuntimeError(f"{context}: missing original_layout for gradient canonicalization")
    canonical = canonicalize_activation_btc(
        value,
        raw_layout=raw_layout,
        padding_mask=token_padding_mask,
        batch_size=int(token_padding_mask.shape[0]),
        token_count=int(token_padding_mask.shape[1]),
        context=context,
    )
    return canonical.tensor


def _capture_for_grad(value: Tensor) -> Tensor:
    if value.requires_grad:
        return value
    return value.detach().requires_grad_(True)


def _forward_fp32_with_gradients(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    specs: Sequence[TargetSpec],
) -> dict[str, Any]:
    model.eval()
    _set_no_param_grad(model)
    moved = _move_batch_to_device(batch, device)
    labels = moved["label_id"].long()
    captured: dict[TargetSpec, Tensor] = {}
    handles: list[Any] = []

    for spec in specs:
        target_module = get_ffn_module(model, spec.layer_idx, spec.module_name)
        if spec.hook_position == "input_activation":

            def pre_hook(
                module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                hook_spec: TargetSpec = spec,
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                del module
                if not args or not isinstance(args[0], Tensor):
                    raise RuntimeError(f"{hook_spec.public_name}: expected tensor as first module input")
                replacement = _capture_for_grad(args[0])
                captured[hook_spec] = replacement
                return (replacement, *args[1:]), kwargs

            handles.append(target_module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        elif spec.hook_position == "output_representation":

            def post_hook(
                module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: Any,
                *,
                hook_spec: TargetSpec = spec,
            ) -> Tensor:
                del module, args, kwargs
                if not isinstance(output, Tensor):
                    raise RuntimeError(f"{hook_spec.public_name}: expected tensor module output")
                replacement = _capture_for_grad(output)
                captured[hook_spec] = replacement
                return replacement

            handles.append(target_module.register_forward_hook(post_hook, with_kwargs=True))
        else:
            raise ValueError(f"unsupported hook_position={spec.hook_position!r}")

    try:
        with torch.enable_grad():
            encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
                model,
                moved["waveform"],
                moved["padding_mask"],
            )
            features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
            features = features.float()
            logits = pooled_logits_from_features(model, features, token_padding_mask)
            missing = [spec.public_name for spec in specs if spec not in captured]
            if missing:
                raise RuntimeError(f"FP32 activation hook(s) did not fire: {missing}")
            h_fp32_by_target: dict[TargetSpec, Tensor] = {}
            layout_info_by_target: dict[TargetSpec, dict[str, Any]] = {}
            for spec in specs:
                h_fp32, layout_info = _canonicalize(captured[spec], token_padding_mask, context=f"{spec.public_name}.fp32")
                h_fp32_by_target[spec] = h_fp32
                layout_info_by_target[spec] = dict(layout_info)
            preds = logits.detach().argmax(dim=1)
            fp32_correct = preds.eq(labels)
            competitors = _competitor_indices(logits.detach(), labels)
            fixed_margins = _fixed_margin(logits, labels, competitors)
            gradient_by_target: dict[TargetSpec, Tensor] = {}
            if bool(fp32_correct.any().item()):
                selected_margin = fixed_margins[fp32_correct].sum()
                raw_tensors = tuple(captured[spec] for spec in specs)
                grad_tensors = torch.autograd.grad(
                    selected_margin,
                    raw_tensors,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                for spec, grad_raw in zip(specs, grad_tensors):
                    gradient_by_target[spec] = _canonicalize_like(
                        grad_raw.detach(),
                        token_padding_mask,
                        layout_info_by_target[spec],
                        context=f"{spec.public_name}.gradient",
                    )
            else:
                for spec in specs:
                    gradient_by_target[spec] = torch.zeros_like(h_fp32_by_target[spec])
    finally:
        for handle in handles:
            handle.remove()

    _assert_no_param_grads(model, context="fp32_multi_target_gradient_forward")
    return {
        "logits": logits.detach().cpu().float(),
        "labels": labels.detach().cpu().long(),
        "preds": preds.detach().cpu().long(),
        "competitors": competitors.detach().cpu().long(),
        "fixed_margins": fixed_margins.detach().cpu().float(),
        "dynamic_margins": _dynamic_margin(logits.detach(), labels).detach().cpu().float(),
        "fp32_correct": fp32_correct.detach().cpu().bool(),
        "h_fp32_by_target": {spec: value.detach().cpu().float() for spec, value in h_fp32_by_target.items()},
        "gradient_by_target": {spec: value.detach().cpu().float() for spec, value in gradient_by_target.items()},
        "token_padding_mask": token_padding_mask.detach().cpu().bool(),
        "layout_info_by_target": layout_info_by_target,
    }


def _prepare_full_int4_model(
    checkpoint_path: Path,
    *,
    qparams_by_target: Mapping[tuple[int, str], Mapping[str, Any]],
    device: torch.device,
) -> tuple[nn.Module, Any]:
    model, _checkpoint = load_trained_checkpoint(checkpoint_path, device=device)
    model.eval()
    _set_no_param_grad(model)
    controller = prepare_beats_ffn_int4_quantization_from_project_config(model)
    missing: list[str] = []
    for module_path, module in sorted(controller.modules.items()):
        spec = _target_spec_from_module_path(module_path)
        key = _target_key(spec)
        qparams = qparams_by_target.get(key)
        if qparams is None:
            missing.append(module_path)
            continue
        apply_fixed_qparams_to_quant_module(
            module,
            mode=QUANT_MODE,
            weight_qparams=qparams.get("weight"),
            activation_qparams=qparams.get("activation"),
            require_qparams=True,
        )
    if missing:
        raise FileNotFoundError(
            "full-W4A4 INT4 forward requires frozen qparams for every FFN fc1/fc2 module; "
            f"missing examples={missing[:8]}"
        )
    controller.qparams_frozen = True
    controller.prepare_quant_forward()
    for module_path, module in sorted(controller.modules.items()):
        if not bool(getattr(module, "fake_quant_enabled", False)):
            raise RuntimeError(f"{module_path}: fake quant was not enabled")
        if not bool(getattr(module, "qparams_frozen", False)):
            raise RuntimeError(f"{module_path}: activation qparams are not frozen")
    setattr(model, "_decision_sanity_quant_controller", controller)
    return model, controller


def _collect_int4_capture(model: nn.Module, spec: TargetSpec, token_padding_mask: Tensor) -> Tensor:
    module = get_ffn_module(model, spec.layer_idx, spec.module_name)
    if not isinstance(module, QuantLinear):
        raise TypeError(f"{spec.public_name}: full-W4A4 model target is not QuantLinear")
    if spec.hook_position == "input_activation":
        _fp32_capture, quant_capture = module.get_oer_activation_tensors()
    elif spec.hook_position == "output_representation":
        _fp32_capture, quant_capture = module.get_oer_projection_tensors()
    else:
        raise ValueError(f"unsupported hook_position={spec.hook_position!r}")
    if quant_capture is None:
        raise RuntimeError(f"{spec.public_name}: QuantLinear did not expose quantized representation")
    quant_btc = quant_capture.detach().cpu().float().contiguous()
    if tuple(quant_btc.shape[:2]) != tuple(token_padding_mask.shape):
        raise RuntimeError(
            f"{spec.public_name}: quant representation shape {tuple(quant_btc.shape)} "
            f"does not align token mask {tuple(token_padding_mask.shape)}"
        )
    return quant_btc


def _forward_full_int4(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    specs: Sequence[TargetSpec],
) -> dict[str, Any]:
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
        features = features.float()
        logits = pooled_logits_from_features(model, features, token_padding_mask)
    token_padding_mask_cpu = token_padding_mask.detach().cpu().bool()
    h_int4_by_target = {
        spec: _collect_int4_capture(model, spec, token_padding_mask_cpu)
        for spec in specs
    }
    _assert_no_param_grads(model, context="full_int4_forward")
    return {
        "logits": logits.detach().cpu().float(),
        "labels": labels.detach().cpu().long(),
        "preds": logits.detach().argmax(dim=1).cpu().long(),
        "dynamic_margins": _dynamic_margin(logits.detach(), labels).detach().cpu().float(),
        "h_int4_by_target": h_int4_by_target,
        "token_padding_mask": token_padding_mask_cpu,
    }


def _token_topk_overlap(fp32_norm: Tensor, int4_norm: Tensor, valid_mask: Tensor, *, ratio: float = 0.05) -> tuple[float, int]:
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    n = int(valid_indices.numel())
    if n <= 0:
        return float("nan"), 0
    k = max(1, int(math.ceil(float(n) * float(ratio))))
    k = min(k, n)
    fp32_valid = fp32_norm[valid_indices]
    int4_valid = int4_norm[valid_indices]
    fp32_top = set(valid_indices[torch.topk(fp32_valid, k=k).indices].tolist())
    int4_top = set(valid_indices[torch.topk(int4_valid, k=k).indices].tolist())
    return float(len(fp32_top & int4_top) / float(k)), int(k)


def _rows_for_batch(
    *,
    spec: TargetSpec,
    fp32: Mapping[str, Any],
    int4: Mapping[str, Any],
    sample_ids: Sequence[str],
    audio_paths: Sequence[str],
    label_names: Mapping[int, str],
    metadata: Mapping[str, Mapping[str, Any]],
    global_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels: Tensor = fp32["labels"]
    fp32_logits: Tensor = fp32["logits"]
    int4_logits: Tensor = int4["logits"]
    competitors: Tensor = fp32["competitors"]
    fp32_fixed: Tensor = fp32["fixed_margins"]
    int4_fixed = _fixed_margin(int4_logits, labels, competitors)
    fp32_dynamic: Tensor = fp32["dynamic_margins"]
    int4_dynamic: Tensor = int4["dynamic_margins"]
    fp32_correct: Tensor = fp32["fp32_correct"]
    int4_preds: Tensor = int4["preds"]
    h_fp32: Tensor = fp32["h_fp32_by_target"][spec]
    h_int4: Tensor = int4["h_int4_by_target"][spec]
    gradient: Tensor = fp32["gradient_by_target"][spec]
    token_padding_mask: Tensor = fp32["token_padding_mask"]
    int4_mask: Tensor = int4["token_padding_mask"]
    if tuple(token_padding_mask.shape) != tuple(int4_mask.shape) or not torch.equal(token_padding_mask, int4_mask):
        raise RuntimeError(f"{spec.public_name}: FP32 and INT4 token padding masks differ")
    if tuple(h_fp32.shape) != tuple(h_int4.shape) or tuple(h_fp32.shape) != tuple(gradient.shape):
        raise RuntimeError(
            f"{spec.public_name}: representation/gradient shape mismatch: "
            f"fp32={tuple(h_fp32.shape)} int4={tuple(h_int4.shape)} grad={tuple(gradient.shape)}"
        )

    delta = h_int4 - h_fp32
    delta_norm = torch.linalg.vector_norm(delta.float(), dim=-1)
    fp32_norm = torch.linalg.vector_norm(h_fp32.float(), dim=-1)
    int4_norm = torch.linalg.vector_norm(h_int4.float(), dim=-1)
    grad_norm = torch.linalg.vector_norm(gradient.float(), dim=-1)
    rel = delta_norm / fp32_norm.clamp_min(EPS)
    projection = (gradient.float() * delta.float()).sum(dim=-1)
    harmful = torch.clamp(-projection, min=0.0)
    beneficial = torch.clamp(projection, min=0.0)

    sample_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for local_index, sample_id in enumerate(sample_ids):
        if not bool(fp32_correct[local_index].item()):
            continue
        valid = ~token_padding_mask[local_index].bool()
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            raise RuntimeError(f"{sample_id}: zero valid tokens")
        sid = str(sample_id)
        meta = metadata.get(sid, {})
        bucket = str(meta.get("bucket", "")).strip().lower()
        label_id = int(labels[local_index].item())
        fixed_change = float((int4_fixed[local_index] - fp32_fixed[local_index]).item())
        fixed_drop = -fixed_change
        signed_sum = float(projection[local_index][valid].sum().item())
        linear_damage = -signed_sum
        harmful_sum = float(harmful[local_index][valid].sum().item())
        beneficial_sum = float(beneficial[local_index][valid].sum().item())
        margin_denom = max(abs(float(fp32_fixed[local_index].item())), EPS)
        topk_overlap, topk_k = _token_topk_overlap(fp32_norm[local_index], int4_norm[local_index], valid)
        row = {
            "sample_index": int(global_offset + local_index),
            "sample_id": sid,
            "audio_path": str(audio_paths[local_index]),
            "label_id": label_id,
            "label_name": str(label_names.get(label_id, label_id)),
            "bucket": bucket,
            "group_tail_included": int(bucket in {"medium", "few"}),
            "target": spec.public_name,
            "layer_idx": int(spec.layer_idx),
            "module_name": spec.module_name,
            "hook_position": spec.hook_position,
            "fp32_pred": int(fp32["preds"][local_index].item()),
            "int4_pred": int(int4_preds[local_index].item()),
            "fp32_correct": 1,
            "int4_correct": int(int(int4_preds[local_index].item()) == label_id),
            "fp32_correct_int4_flip": int(int(int4_preds[local_index].item()) != label_id),
            "j_star": int(competitors[local_index].item()),
            "m_fp32_fixed": float(fp32_fixed[local_index].item()),
            "m_int4_fixed": float(int4_fixed[local_index].item()),
            "m_fp32_dynamic": float(fp32_dynamic[local_index].item()),
            "m_int4_dynamic": float(int4_dynamic[local_index].item()),
            "actual_fixed_margin_change": fixed_change,
            "actual_fixed_margin_drop": fixed_drop,
            "valid_token_count": valid_count,
            "D_abs": float(delta_norm[local_index][valid].mean().item()),
            "D_rel": float(rel[local_index][valid].mean().item()),
            "mean_gradient_norm": float(grad_norm[local_index][valid].mean().item()),
            "signed_projection_sum": signed_sum,
            "linear_margin_damage": linear_damage,
            "harmful_projection_sum": harmful_sum,
            "beneficial_projection_sum": beneficial_sum,
            "harmful_projection_ratio": harmful_sum / margin_denom,
            "linear_margin_damage_ratio": linear_damage / margin_denom,
            "signed_risk_ratio": linear_damage / margin_denom,
            "topk_overlap_at_k": topk_overlap,
            "topk_k": topk_k,
        }
        sample_rows.append(row)

        valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
        for token_index in valid_indices:
            token_rows.append(
                {
                    "sample_index": int(global_offset + local_index),
                    "sample_id": sid,
                    "label_id": label_id,
                    "bucket": bucket,
                    "target": spec.public_name,
                    "layer_idx": int(spec.layer_idx),
                    "module_name": spec.module_name,
                    "hook_position": spec.hook_position,
                    "token_index": int(token_index),
                    "fp32_norm": float(fp32_norm[local_index, token_index].item()),
                    "int4_norm": float(int4_norm[local_index, token_index].item()),
                    "delta_norm": float(delta_norm[local_index, token_index].item()),
                    "relative_distortion": float(rel[local_index, token_index].item()),
                    "gradient_norm": float(grad_norm[local_index, token_index].item()),
                    "decision_projection": float(projection[local_index, token_index].item()),
                    "harmful_projection": float(harmful[local_index, token_index].item()),
                    "beneficial_projection": float(beneficial[local_index, token_index].item()),
                }
            )
    return sample_rows, token_rows


def _finite_arrays(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        try:
            x = float(row.get(x_key, float("nan")))
            y = float(row.get(y_key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _mean(values: Sequence[float]) -> float:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    return float(np.mean(finite)) if finite else float("nan")


def _median(values: Sequence[float]) -> float:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    return float(np.median(finite)) if finite else float("nan")


def _std(values: Sequence[float]) -> float:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    return float(np.std(finite, ddof=0)) if finite else float("nan")


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    finite_pairs = [
        (float(score), int(label))
        for score, label in zip(scores, labels)
        if math.isfinite(float(score)) and int(label) in {0, 1}
    ]
    positives = sum(label for _score, label in finite_pairs)
    negatives = len(finite_pairs) - positives
    if positives <= 0 or negatives <= 0:
        return float("nan")
    sorted_scores = np.asarray([score for score, _label in finite_pairs], dtype=np.float64)
    sorted_labels = np.asarray([label for _score, label in finite_pairs], dtype=np.int64)
    ranks = _rankdata(sorted_scores)
    rank_sum_pos = float(ranks[sorted_labels == 1].sum())
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def _rows_for_group(rows: Sequence[Mapping[str, Any]], group: str) -> list[Mapping[str, Any]]:
    if group == "overall_fp32_correct":
        return list(rows)
    if group == "tail":
        return [row for row in rows if str(row.get("bucket", "")).lower() in {"medium", "few"}]
    return [row for row in rows if str(row.get("bucket", "")).lower() == group]


def _target_rows(rows: Sequence[Mapping[str, Any]], target: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("target")) == str(target)]


def _build_taylor_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    targets = sorted({str(row["target"]) for row in sample_rows})
    for target in targets:
        rows_for_target = _target_rows(sample_rows, target)
        for group in ALL_GROUPS:
            rows = _rows_for_group(rows_for_target, group)
            signed, actual = _finite_arrays(rows, "signed_projection_sum", "actual_fixed_margin_change")
            if signed.size:
                diff = signed - actual
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff * diff)))
                sign_agreement = float(np.mean(np.sign(signed) == np.sign(actual)))
            else:
                mae = rmse = sign_agreement = float("nan")
            pearson = _pearson(signed, actual)
            spearman = _spearman(signed, actual)
            if signed.size < 3:
                status = "insufficient_data"
            elif (math.isfinite(pearson) and abs(pearson) >= 0.3) or (math.isfinite(spearman) and abs(spearman) >= 0.3):
                status = "first_order_approximation_usable"
            else:
                status = "first_order_approximation_weak"
            result.append(
                {
                    "group": group,
                    "target": target,
                    "sample_count": int(signed.size),
                    "mae": mae,
                    "rmse": rmse,
                    "pearson": pearson,
                    "spearman": spearman,
                    "sign_agreement": sign_agreement,
                    "mean_signed_projection_sum": float(np.mean(signed)) if signed.size else float("nan"),
                    "mean_actual_fixed_margin_change": float(np.mean(actual)) if actual.size else float("nan"),
                    "approximation_status": status,
                }
            )
    return result


def _build_coupling_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("D_abs", "D_rel", "linear_margin_damage", "harmful_projection_sum", "signed_risk_ratio")
    result: list[dict[str, Any]] = []
    targets = sorted({str(row["target"]) for row in sample_rows})
    for target in targets:
        rows_for_target = _target_rows(sample_rows, target)
        for group in ALL_GROUPS:
            rows = _rows_for_group(rows_for_target, group)
            flip_labels = [int(row.get("fp32_correct_int4_flip", 0)) for row in rows]
            for metric in metrics:
                x, y = _finite_arrays(rows, metric, "actual_fixed_margin_drop")
                values = [float(row.get(metric, float("nan"))) for row in rows]
                flip_values = [float(row.get(metric, float("nan"))) for row in rows if int(row.get("fp32_correct_int4_flip", 0)) == 1]
                nonflip_values = [float(row.get(metric, float("nan"))) for row in rows if int(row.get("fp32_correct_int4_flip", 0)) == 0]
                result.append(
                    {
                        "group": group,
                        "target": target,
                        "metric": metric,
                        "against": "actual_fixed_margin_drop",
                        "sample_count": int(x.size),
                        "pearson": _pearson(x, y),
                        "spearman": _spearman(x, y),
                        "mean": _mean(values),
                        "median": _median(values),
                        "std": _std(values),
                        "fp32_correct_int4_flip_count": int(sum(flip_labels)),
                        "fp32_correct_int4_nonflip_count": int(len(flip_labels) - sum(flip_labels)),
                        "flip_mean": _mean(flip_values),
                        "nonflip_mean": _mean(nonflip_values),
                        "flip_median": _median(flip_values),
                        "nonflip_median": _median(nonflip_values),
                        "flip_vs_nonflip_auroc": _auroc(values, flip_labels),
                    }
                )
    return result


def _quantile(values: Sequence[float], q: float) -> float:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    return float(np.quantile(finite, q)) if finite else float("nan")


def _counterexample_row(case_type: str, row: Mapping[str, Any], *, note: str, paired: Mapping[str, Any] | None = None) -> dict[str, Any]:
    paired = paired or {}
    return {
        "case_type": case_type,
        "target": row.get("target", ""),
        "sample_id": row.get("sample_id", ""),
        "paired_sample_id": paired.get("sample_id", ""),
        "bucket": row.get("bucket", ""),
        "paired_bucket": paired.get("bucket", ""),
        "fp32_correct_int4_flip": row.get("fp32_correct_int4_flip", ""),
        "paired_fp32_correct_int4_flip": paired.get("fp32_correct_int4_flip", ""),
        "D_rel": row.get("D_rel", ""),
        "paired_D_rel": paired.get("D_rel", ""),
        "actual_fixed_margin_drop": row.get("actual_fixed_margin_drop", ""),
        "paired_actual_fixed_margin_drop": paired.get("actual_fixed_margin_drop", ""),
        "linear_margin_damage": row.get("linear_margin_damage", ""),
        "paired_linear_margin_damage": paired.get("linear_margin_damage", ""),
        "harmful_projection_sum": row.get("harmful_projection_sum", ""),
        "paired_harmful_projection_sum": paired.get("harmful_projection_sum", ""),
        "note": note,
    }


def _build_counterexamples(sample_rows: Sequence[Mapping[str, Any]], *, max_per_case: int = 5) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in sorted({str(row["target"]) for row in sample_rows}):
        rows = _target_rows(sample_rows, target)
        if len(rows) < 4:
            continue
        d_rel_q75 = _quantile([float(row.get("D_rel", float("nan"))) for row in rows], 0.75)
        d_rel_med = _quantile([float(row.get("D_rel", float("nan"))) for row in rows], 0.50)
        drop_q25 = _quantile([float(row.get("actual_fixed_margin_drop", float("nan"))) for row in rows], 0.25)
        drop_q75 = _quantile([float(row.get("actual_fixed_margin_drop", float("nan"))) for row in rows], 0.75)

        case_a = [
            row for row in rows
            if float(row.get("D_rel", float("nan"))) >= d_rel_q75
            and float(row.get("actual_fixed_margin_drop", float("nan"))) <= drop_q25
        ]
        case_a = sorted(case_a, key=lambda row: (-float(row["D_rel"]), float(row["actual_fixed_margin_drop"])))
        for row in case_a[:max_per_case]:
            result.append(
                _counterexample_row(
                    "A_high_generic_low_margin_drop",
                    row,
                    note="Generic relative distortion is high, but realized fixed-margin drop is low.",
                )
            )

        case_b = [
            row for row in rows
            if float(row.get("D_rel", float("nan"))) <= d_rel_med
            and float(row.get("actual_fixed_margin_drop", float("nan"))) >= drop_q75
        ]
        case_b = sorted(case_b, key=lambda row: (-float(row["actual_fixed_margin_drop"]), float(row["D_rel"])))
        for row in case_b[:max_per_case]:
            result.append(
                _counterexample_row(
                    "B_not_high_generic_high_margin_drop",
                    row,
                    note="Generic relative distortion is not high, but realized fixed-margin drop is high.",
                )
            )

        sorted_rows = sorted(rows, key=lambda row: float(row.get("D_rel", float("nan"))))
        pairs: list[tuple[float, float, Mapping[str, Any], Mapping[str, Any]]] = []
        for left, right in zip(sorted_rows, sorted_rows[1:]):
            if int(left.get("fp32_correct_int4_flip", 0)) == int(right.get("fp32_correct_int4_flip", 0)):
                continue
            d_gap = abs(float(left["D_rel"]) - float(right["D_rel"]))
            damage_gap = abs(float(left["linear_margin_damage"]) - float(right["linear_margin_damage"]))
            pairs.append((d_gap, -damage_gap, left, right))
        pairs.sort(key=lambda item: (item[0], item[1]))
        for _d_gap, _neg_damage_gap, left, right in pairs[:max_per_case]:
            result.append(
                _counterexample_row(
                    "C_similar_generic_different_decision_projection",
                    left,
                    paired=right,
                    note="Nearby generic distortion, but decision-aware projection differs and flip status differs.",
                )
            )
    return result


def _metric_score(row: Mapping[str, Any]) -> float:
    values = []
    for key in ("spearman", "pearson"):
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            values.append(abs(value))
    return max(values) if values else float("nan")


def _best_metric(coupling_rows: Sequence[Mapping[str, Any]], *, metrics: set[str]) -> Mapping[str, Any] | None:
    candidates = [
        row for row in coupling_rows
        if str(row.get("group")) == "overall_fp32_correct" and str(row.get("metric")) in metrics
    ]
    candidates = [row for row in candidates if math.isfinite(_metric_score(row))]
    if not candidates:
        return None
    return max(candidates, key=_metric_score)


def _final_status(
    sample_rows: Sequence[Mapping[str, Any]],
    taylor_rows: Sequence[Mapping[str, Any]],
    coupling_rows: Sequence[Mapping[str, Any]],
) -> str:
    if len(sample_rows) < 8:
        return "insufficient_data"
    overall_taylor = [
        row for row in taylor_rows
        if str(row.get("group")) == "overall_fp32_correct" and str(row.get("approximation_status")) != "insufficient_data"
    ]
    if not overall_taylor:
        return "insufficient_data"
    usable_taylor = any(str(row.get("approximation_status")) == "first_order_approximation_usable" for row in overall_taylor)
    if not usable_taylor:
        return "first_order_approximation_weak"
    generic = _best_metric(coupling_rows, metrics={"D_abs", "D_rel"})
    decision = _best_metric(coupling_rows, metrics={"linear_margin_damage", "harmful_projection_sum", "signed_risk_ratio"})
    if decision is None:
        return "decision_signal_weak"
    decision_score = _metric_score(decision)
    generic_score = _metric_score(generic) if generic is not None else float("nan")
    if math.isfinite(decision_score) and (not math.isfinite(generic_score) or decision_score >= max(0.25, generic_score + 0.03)):
        return "decision_signal_present"
    return "decision_signal_weak"


def _build_manifest(
    *,
    paths: Mapping[str, Path],
    qparams_by_target: Mapping[tuple[int, str], Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    sample_order: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    final_status: str,
) -> dict[str, Any]:
    qparams_identity_by_analysis_target = {
        spec.public_name: qparams_identity_manifest(qparams_by_target[_target_key(spec)], source_hint=spec.public_name)
        for spec in TARGET_SPECS
    }
    qparams_identity_by_quantized_target = {
        _public_quant_target_name(key): qparams_identity_manifest(qparams, source_hint=_public_quant_target_name(key))
        for key, qparams in sorted(qparams_by_target.items())
    }
    calibration_sources = sorted(
        {
            str(qparams.get("calibration_csv") or qparams.get("activation", {}).get("calibration_csv") or "")
            for qparams in qparams_by_target.values()
        }
        - {""}
    )
    return {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "analysis_only": True,
        "sanity_check_only": True,
        "formal_paper_result": False,
        "causal_or_method_claim_allowed": False,
        "dataset": DEFAULT_DATASET,
        "analysis_split": "eval",
        "test_touched": False,
        "train_touched": False,
        "checkpoint": str(paths["checkpoint"].resolve(strict=False)),
        "checkpoint_name_required": DEFAULT_CHECKPOINT_NAME,
        "checkpoint_sha256": _file_sha256(paths["checkpoint"]),
        "same_checkpoint_fp32_int4": True,
        "same_classifier": True,
        "same_eval_samples": True,
        "same_waveform": True,
        "same_center_crop": True,
        "same_padding_mask": True,
        "same_pooling": "mask_aware_mean",
        "same_sample_order": True,
        "sample_order_hash": stable_json_hash([str(item) for item in sample_order]),
        "fixed_competitor_j_star": True,
        "competitor_definition": "argmax_j_not_y(z_fp32[j])",
        "main_cohort": "all eval samples where FP32 top-1 is correct",
        "groups": list(ALL_GROUPS),
        "tail_definition": "tail = bucket in {medium, few}; few is diagnostic only",
        "quant_mode": QUANT_MODE.upper(),
        "quantization": {
            "target_scope": "full_beats_transformer_ffn_w4a4",
            "analysis_hook_scope": "fixed_target_representations_from_shared_full_int4_forward",
            "weight_bits": 4,
            "activation_bits": 4,
            "weight_granularity": "per_output_channel",
            "activation_granularity": "per_tensor",
            "symmetric": True,
            "qmin": EXPECTED_QRANGE[0],
            "qmax": EXPECTED_QRANGE[1],
            "zero_point": 0,
            "fake_quant": True,
            "observer_enabled": False,
            "qparams_recalibrated": False,
            "dynamic_minmax_used": False,
            "percentile_changed": False,
        },
        "quantized_targets": [
            {
                "target": _public_quant_target_name(key),
                "module_path": _module_path_for_key(key),
                "layer_idx": int(key[0]),
                "module_name": str(key[1]),
            }
            for key in sorted(qparams_by_target)
        ],
        "num_quantized_ffn_modules": int(len(qparams_by_target)),
        "layers": [
            {
                "target": spec.public_name,
                "layer_idx": int(spec.layer_idx),
                "module_name": spec.module_name,
                "hook_position": spec.hook_position,
                "module_path": spec.target_name,
            }
            for spec in TARGET_SPECS
        ],
        "qparams_root": str(paths["qparams_root"].resolve(strict=False)),
        "qparams_identity_by_target": qparams_identity_by_analysis_target,
        "qparams_identity_by_analysis_target": qparams_identity_by_analysis_target,
        "qparams_identity_by_quantized_target": qparams_identity_by_quantized_target,
        "calibration_source": calibration_sources or [str(paths["calibration_csv"].resolve(strict=False))],
        "calibration_csv": file_identity(paths["calibration_csv"]),
        "calibration_manifest": file_identity(paths["calibration_manifest"]),
        "eval_csv": file_identity(paths["eval_csv"]),
        "label_mapping": file_identity(paths["label_mapping"]),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "num_fp32_correct_rows": int(len(sample_rows)),
        "num_fp32_correct_target_rows": int(len(sample_rows)),
        "num_fp32_correct_samples": int(len({str(row.get("sample_id")) for row in sample_rows})),
        "num_eval_samples_seen": int(len(sample_order)),
        "final_descriptive_status": str(final_status),
        "required_outputs": [
            "decision_perturbation_sample_summary.csv",
            "decision_perturbation_token_detail.csv",
            "decision_perturbation_coupling.csv",
            "decision_perturbation_taylor_check.csv",
            "decision_perturbation_counterexamples.csv",
            "decision_perturbation_manifest.json",
        ],
        "disclaimer": (
            "Descriptive sanity check only: results do not establish causality, do not define a new training method, "
            "and are not formal PER paper evidence."
        ),
    }


def _print_console_report(
    *,
    paths: Mapping[str, Path],
    sample_rows: Sequence[Mapping[str, Any]],
    coupling_rows: Sequence[Mapping[str, Any]],
    taylor_rows: Sequence[Mapping[str, Any]],
    counterexample_rows: Sequence[Mapping[str, Any]],
    final_status: str,
) -> None:
    print("[Decision-Relevant Perturbation Sanity Check]")
    print(f"checkpoint: {paths['checkpoint']}")
    print(f"output_dir: {paths['output_dir']}")
    print(f"fp32_correct_target_rows: {len(sample_rows)}")
    print(f"test_touched: False")
    generic_best = _best_metric(coupling_rows, metrics={"D_abs", "D_rel"})
    decision_best = _best_metric(coupling_rows, metrics={"linear_margin_damage", "harmful_projection_sum", "signed_risk_ratio"})
    for target in sorted({str(row.get("target")) for row in sample_rows}):
        rows = _target_rows(sample_rows, target)
        flips = sum(int(row.get("fp32_correct_int4_flip", 0)) for row in rows)
        taylors = [
            row for row in taylor_rows
            if row.get("target") == target and row.get("group") == "overall_fp32_correct"
        ]
        taylor = taylors[0] if taylors else {}
        print(
            f"{target}: n={len(rows)} flips={flips} "
            f"taylor_spearman={_csv_value(taylor.get('spearman', ''))} "
            f"sign_agreement={_csv_value(taylor.get('sign_agreement', ''))}"
        )
    print(f"best_generic_metric: {dict(generic_best) if generic_best is not None else {}}")
    print(f"best_decision_aware_metric: {dict(decision_best) if decision_best is not None else {}}")
    print(f"counterexample_count: {len(counterexample_rows)}")
    print(f"final_descriptive_status: {final_status}")
    print(
        "Descriptive sanity check only; no causal claim, no method claim, "
        "no quantization protocol change, and no test-set evidence."
    )


def _assert_target_independent_int4_outputs(sample_rows: Sequence[Mapping[str, Any]]) -> None:
    by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for row in sample_rows:
        by_sample.setdefault(str(row.get("sample_id")), []).append(row)
    pred_mismatches: list[str] = []
    margin_mismatches: list[str] = []
    for sample_id, rows in by_sample.items():
        if len(rows) <= 1:
            continue
        preds = {int(row.get("int4_pred")) for row in rows}
        if len(preds) != 1:
            pred_mismatches.append(f"{sample_id}: {sorted(preds)}")
        margins = [float(row.get("m_int4_fixed", float("nan"))) for row in rows]
        finite_margins = [value for value in margins if math.isfinite(value)]
        if finite_margins and (max(finite_margins) - min(finite_margins)) > 1e-7:
            margin_mismatches.append(f"{sample_id}: min={min(finite_margins):.8g} max={max(finite_margins):.8g}")
    if pred_mismatches or margin_mismatches:
        raise RuntimeError(
            "shared full-W4A4 INT4 logits invariant failed; target-specific rows must not change final INT4 output. "
            f"prediction mismatches={pred_mismatches[:8]}, margin mismatches={margin_mismatches[:8]}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _resolve_paths(args)
    device = _device_from_args(args.device)
    label_names = _load_label_names(paths["label_mapping"])
    metadata = _load_eval_metadata(paths["eval_csv"])
    loader = build_loader(
        dataset_paths(DEFAULT_DATASET, None),
        split="eval",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    fp32_model, checkpoint_payload = load_trained_checkpoint(paths["checkpoint"], device=device)
    model_kind = str(checkpoint_payload.get("_baseline_model_kind") or checkpoint_payload.get("model_kind") or "")
    if model_kind != "fp32_teacher":
        raise ValueError(f"checkpoint must be fp32_teacher; got {model_kind!r}")
    _set_no_param_grad(fp32_model)
    full_quant_specs = _all_ffn_specs_for_model(fp32_model)
    qparams_by_target = _load_qparams_by_target(paths["qparams_root"], full_quant_specs)
    int4_model, _int4_controller = _prepare_full_int4_model(
        paths["checkpoint"],
        qparams_by_target=qparams_by_target,
        device=device,
    )

    sample_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    sample_order: list[str] = []
    global_offset = 0
    for batch in loader:
        sample_ids = _batch_sample_ids(batch)
        sample_order.extend(sample_ids)
        audio_paths = _batch_audio_paths(batch, sample_ids, metadata)
        fp32 = _forward_fp32_with_gradients(fp32_model, batch, device=device, specs=TARGET_SPECS)
        int4 = _forward_full_int4(int4_model, batch, device=device, specs=TARGET_SPECS)
        for spec in TARGET_SPECS:
            batch_sample_rows, batch_token_rows = _rows_for_batch(
                spec=spec,
                fp32=fp32,
                int4=int4,
                sample_ids=sample_ids,
                audio_paths=audio_paths,
                label_names=label_names,
                metadata=metadata,
                global_offset=global_offset,
            )
            sample_rows.extend(batch_sample_rows)
            token_rows.extend(batch_token_rows)
        global_offset += len(sample_ids)

    _assert_target_independent_int4_outputs(sample_rows)
    taylor_rows = _build_taylor_rows(sample_rows)
    coupling_rows = _build_coupling_rows(sample_rows)
    counterexample_rows = _build_counterexamples(sample_rows)
    final_status = _final_status(sample_rows, taylor_rows, coupling_rows)
    manifest = _build_manifest(
        paths=paths,
        qparams_by_target=qparams_by_target,
        sample_rows=sample_rows,
        sample_order=sample_order,
        args=args,
        device=device,
        final_status=final_status,
    )

    output_dir = paths["output_dir"]
    _write_csv(output_dir / "decision_perturbation_sample_summary.csv", sample_rows, SAMPLE_COLUMNS)
    _write_csv(output_dir / "decision_perturbation_token_detail.csv", token_rows, TOKEN_COLUMNS)
    _write_csv(output_dir / "decision_perturbation_coupling.csv", coupling_rows, COUPLING_COLUMNS)
    _write_csv(output_dir / "decision_perturbation_taylor_check.csv", taylor_rows, TAYLOR_COLUMNS)
    _write_csv(output_dir / "decision_perturbation_counterexamples.csv", counterexample_rows, COUNTEREXAMPLE_COLUMNS)
    _write_json(output_dir / "decision_perturbation_manifest.json", manifest)
    _print_console_report(
        paths=paths,
        sample_rows=sample_rows,
        coupling_rows=coupling_rows,
        taylor_rows=taylor_rows,
        counterexample_rows=counterexample_rows,
        final_status=final_status,
    )
    return {
        "output_dir": str(output_dir),
        "sample_rows": len(sample_rows),
        "token_rows": len(token_rows),
        "final_status": final_status,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Only local_data is accepted for this sanity check.")
    parser.add_argument("--checkpoint", default=None, help=f"Must point to {DEFAULT_CHECKPOINT_NAME}.")
    parser.add_argument("--qparams-root", default=None, help="Directory containing frozen per-tensor W4A4 qparams.")
    parser.add_argument("--output-dir", default=None, help="Output directory for the sanity-check artifacts.")
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
