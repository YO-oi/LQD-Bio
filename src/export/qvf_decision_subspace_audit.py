from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline.export_eval import (
    DEFAULT_POOLING,
    load_trained_checkpoint,
    move_batch_to_device,
    pooled_logits_from_features,
    prepare_encoder_input_for_export,
)
from src.baseline.manifest import file_sha256
from src.baseline.run_vanilla_kd import (
    QVF_METHOD_VARIANT,
    QVF_MODULE_NAME,
    QVF_PROTOCOL_VERSION,
    QVF_SENSITIVE_LAYERS,
    QVF_TOPK_RATIO,
    _disable_encoder_layerdrop_for_qvf,
    forward_student_qvf,
)
from src.quant.ffn_quant_injector import get_ffn_module
from src.train.train_beats_classifier_head import build_loader, dataset_paths
from src.utils.layout_utils import canonicalize_activation_btc


EXPERIMENT_NAME = "qvf_decision_subspace_audit"
PROTOCOL_VERSION = "qvf_decision_subspace_audit.v1"
TARGET_LAYERS = (1, 9)
TARGETS = tuple(f"L{layer}.{QVF_MODULE_NAME}" for layer in TARGET_LAYERS)
TARGET_MODULE_PATHS = tuple(f"encoder.layers.{layer}.{QVF_MODULE_NAME}" for layer in TARGET_LAYERS)
EPS = 1e-12
DEFAULT_TOLERANCE = 1e-5
FORBIDDEN_CHECKPOINT_NAME = "BEATs_local_data.pt"
SUMMARY_GROUPS = ("overall", "many", "medium", "tail")


SAMPLE_COLUMNS = [
    "sample_id",
    "label_id",
    "bucket",
    "target",
    "teacher_pred",
    "vanilla_pred",
    "qvf_pred",
    "j_star",
    "m_teacher",
    "m_vanilla",
    "m_qvf",
    "vanilla_margin_error",
    "qvf_margin_error",
    "margin_recovery",
    "valid_token_count",
    "gradient_norm",
    "vanilla_total_error",
    "qvf_total_error",
    "vanilla_parallel_error",
    "qvf_parallel_error",
    "vanilla_orthogonal_error",
    "qvf_orthogonal_error",
    "vanilla_decision_error_ratio",
    "qvf_decision_error_ratio",
    "delta_total",
    "delta_parallel",
    "delta_orthogonal",
    "qvf_reduction_case",
    "parallel_reduction_ratio",
    "orthogonal_reduction_ratio",
    "vanilla_signed_projection",
    "qvf_signed_projection",
    "vanilla_predicted_margin_damage",
    "qvf_predicted_margin_damage",
    "vanilla_reconstruction_abs_error",
    "qvf_reconstruction_abs_error",
    "vanilla_margin_orthogonal_dot_abs",
    "qvf_margin_orthogonal_dot_abs",
]


SUMMARY_COLUMNS = [
    "target",
    "group",
    "n",
    "mean_delta_total",
    "median_delta_total",
    "mean_delta_parallel",
    "median_delta_parallel",
    "mean_delta_orthogonal",
    "median_delta_orthogonal",
    "mean_parallel_reduction_ratio",
    "median_parallel_reduction_ratio",
    "mean_orthogonal_reduction_ratio",
    "median_orthogonal_reduction_ratio",
    "mean_margin_recovery",
    "median_margin_recovery",
    "count_parallel_and_orthogonal_down",
    "count_orthogonal_down_parallel_flat",
    "count_total_down_parallel_up",
    "count_other_reduction_case",
    "pearson_delta_total_margin_recovery",
    "spearman_delta_total_margin_recovery",
    "pearson_delta_parallel_margin_recovery",
    "spearman_delta_parallel_margin_recovery",
    "pearson_delta_orthogonal_margin_recovery",
    "spearman_delta_orthogonal_margin_recovery",
]


class ProtocolError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _portable_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _portable_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _batch_sample_ids(batch: Mapping[str, Any]) -> list[str]:
    raw = batch.get("sample_id")
    if raw is None:
        raise KeyError("eval batch missing sample_id")
    if isinstance(raw, Tensor):
        return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    return [str(raw)]


def _float(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _mean(values: Sequence[float]) -> float:
    finite = np.asarray([float(item) for item in values if math.isfinite(float(item))], dtype=np.float64)
    if int(finite.size) == 0:
        return float("nan")
    return float(np.mean(finite))


def _median(values: Sequence[float]) -> float:
    finite = np.asarray([float(item) for item in values if math.isfinite(float(item))], dtype=np.float64)
    if int(finite.size) == 0:
        return float("nan")
    return float(np.median(finite))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    while start < int(order.size):
        end = start + 1
        while end < int(order.size) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _corr(x_values: Sequence[float], y_values: Sequence[float], *, spearman: bool) -> float:
    x = np.asarray([float(item) for item in x_values], dtype=np.float64)
    y = np.asarray([float(item) for item in y_values], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if int(x.size) < 2:
        return float("nan")
    if spearman:
        x = _rankdata(x.tolist())
        y = _rankdata(y.tolist())
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _qvf_reduction_case(
    *,
    delta_total: float,
    delta_parallel: float,
    delta_orthogonal: float,
    tolerance: float,
) -> str:
    tol = float(tolerance)
    if float(delta_total) > tol and float(delta_parallel) < -tol:
        return "total_down_parallel_up"
    if float(delta_parallel) > tol and float(delta_orthogonal) > tol:
        return "parallel_and_orthogonal_down"
    if float(delta_orthogonal) > tol and abs(float(delta_parallel)) <= tol:
        return "orthogonal_down_parallel_flat"
    return "other"


def _group_mask(row: Mapping[str, Any], group: str) -> bool:
    bucket = str(row.get("bucket", "")).strip().lower()
    if group == "overall":
        return True
    if group == "tail":
        return bucket in {"medium", "few", "tail"}
    return bucket == group


def _eval_metadata(eval_csv: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(eval_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "label_id", "bucket"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise KeyError(f"eval.csv must contain columns {sorted(required)}: {eval_csv}")
        for line_number, row in enumerate(reader, start=2):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"{eval_csv}:row={line_number} missing sample_id")
            if sample_id in rows:
                raise ValueError(f"{eval_csv}: duplicate sample_id={sample_id!r}")
            rows[sample_id] = {
                "label_id": int(float(str(row["label_id"]).strip())),
                "bucket": str(row.get("bucket", "")).strip(),
            }
    return rows


def _resolve_manifest_path(value: Any, *, dataset_root: Path, description: str) -> Path:
    if value is None or str(value).strip() == "":
        raise KeyError(f"manifest missing {description}")
    raw = str(value).strip()
    direct = Path(raw)
    candidates: list[Path] = [direct]
    normalized = raw.replace("\\", "/")
    lower = normalized.lower()
    anchor = dataset_root.name.replace("\\", "/").lower()
    marker = f"{anchor}/"
    index = lower.find(marker)
    if index >= 0:
        tail = normalized[index + len(marker) :]
        candidates.append(dataset_root / Path(tail))
        candidates.append(PROJECT_ROOT / dataset_root.name / Path(tail))
    local_marker = "local_data/"
    local_index = lower.find(local_marker)
    if local_index >= 0:
        candidates.append(PROJECT_ROOT / "local_data" / Path(normalized[local_index + len(local_marker) :]))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(strict=True)
    tried = "; ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"could not resolve {description} from manifest value={raw!r}; tried: {tried}")


def _require_file_hash(path: Path, expected_hash: Any, *, description: str) -> str:
    expected = "" if expected_hash is None else str(expected_hash).strip()
    actual = file_sha256(path)
    if not actual:
        raise RuntimeError(f"could not compute SHA256 for {description}: {path}")
    if expected and actual != expected:
        raise RuntimeError(f"{description} SHA256 mismatch: expected={expected}, actual={actual}, path={path}")
    return actual


def _assert_not_quant_injection_checkpoint(path: Path, *, description: str) -> None:
    text = str(path).replace("\\", "/").lower()
    if path.name == FORBIDDEN_CHECKPOINT_NAME or "quant_injection" in text:
        raise ProtocolError(f"{description} resolves to a Quant Injection artifact, which is forbidden: {path}")


def _assert_center_crop_protocol() -> dict[str, Any]:
    audio_cfg = getattr(project_config, "AUDIO", {})
    configured = str(audio_cfg.get("CROP_MODE", "center")).strip().lower() if isinstance(audio_cfg, Mapping) else "center"
    if configured != "center":
        raise ProtocolError(f"AUDIO.CROP_MODE must be center for this audit; got {configured!r}")
    return {"crop_mode": "center", "source": "config.AUDIO.CROP_MODE"}


def _default_qvf_manifest(dataset_root: Path, seed: int, qvf_beta: float) -> Path:
    beta_text = str(float(qvf_beta)).replace(".", "p")
    if beta_text.endswith("0") and "." not in beta_text:
        beta_text = beta_text[:-1]
    if beta_text == "0p05":
        beta_dir = "beta_0p05"
    else:
        beta_dir = f"beta_{beta_text}"
    return dataset_root / "innovation" / "QVF" / f"seed_{int(seed)}" / beta_dir / "qvf_manifest.json"


def _default_kd_manifest(dataset_root: Path, seed: int) -> Path:
    return dataset_root / "baseline" / "kd" / f"seed_{int(seed)}" / "manifests" / "vanilla_kd" / "run_manifest.json"


def _default_teacher_manifest(dataset_root: Path, seed: int) -> Path:
    return dataset_root / "baseline" / "teacher" / f"seed_{int(seed)}" / "metrics" / "fp32_teacher" / "run_manifest.json"


def _manifest_split_hash(manifest: Mapping[str, Any], split: str) -> str:
    split_hashes = manifest.get("split_hashes")
    if isinstance(split_hashes, Mapping) and split in split_hashes:
        return str(split_hashes[split])
    int4_provenance = manifest.get("int4_student_provenance")
    if isinstance(int4_provenance, Mapping) and split == "calibration":
        return str(int4_provenance.get("calibration_csv_hash", ""))
    return ""


def _require_equal(name: str, left: Any, right: Any) -> None:
    if str(left) != str(right):
        raise ProtocolError(f"{name} mismatch: {left!r} != {right!r}")


def _require_float_close(name: str, left: Any, right: float, *, tol: float = 1e-12) -> None:
    value = float(left)
    if abs(value - float(right)) > float(tol):
        raise ProtocolError(f"{name} mismatch: {value!r} != {right!r}")


def _validate_manifests(
    *,
    kd_manifest: Mapping[str, Any],
    qvf_manifest: Mapping[str, Any],
    teacher_manifest: Mapping[str, Any],
    qvf_beta: float,
) -> dict[str, Any]:
    if bool(qvf_manifest.get("test_split_touched", False)):
        raise ProtocolError("QVF manifest reports test_split_touched=true")
    _require_equal("dataset_name", kd_manifest.get("dataset_name"), qvf_manifest.get("dataset_name"))
    if teacher_manifest.get("dataset_name") is not None:
        _require_equal("teacher dataset_name", kd_manifest.get("dataset_name"), teacher_manifest.get("dataset_name"))

    qvf_protocol = qvf_manifest.get("qvf_protocol")
    if not isinstance(qvf_protocol, Mapping):
        raise ProtocolError("QVF manifest missing qvf_protocol mapping")
    _require_equal("qvf protocol_version", qvf_protocol.get("protocol_version"), QVF_PROTOCOL_VERSION)
    _require_equal("qvf method_variant", qvf_protocol.get("method_variant"), QVF_METHOD_VARIANT)
    _require_float_close("qvf beta", qvf_protocol.get("qvf_beta"), float(qvf_beta))
    _require_float_close("qvf topk_ratio", qvf_protocol.get("topk_ratio"), float(QVF_TOPK_RATIO))
    if qvf_protocol.get("qvf_layer_weights") or qvf_manifest.get("qvf_layer_weights"):
        raise ProtocolError("requested canonical equal-weight QVF run, but qvf_layer_weights are present")
    if str(qvf_protocol.get("layer_weight_policy", "")).strip() not in {"", "equal_layer_mean"}:
        raise ProtocolError(f"unexpected QVF layer_weight_policy={qvf_protocol.get('layer_weight_policy')!r}")
    _require_equal("qvf sensitive_layer_list", tuple(qvf_protocol.get("sensitive_layer_list", ())), TARGETS)
    _require_equal("qvf exact_module_paths", tuple(qvf_protocol.get("exact_module_paths", ())), TARGET_MODULE_PATHS)
    _require_equal("qvf activation_hook_semantic", qvf_protocol.get("activation_hook_semantic"), "FFN fc1 input activation")
    _require_equal(
        "qvf student_activation_semantic",
        qvf_protocol.get("student_activation_semantic"),
        "post_fake_quant_dequant_input_activation_with_autograd_graph",
    )
    _require_equal(
        "qvf teacher_activation_semantic",
        qvf_protocol.get("teacher_activation_semantic"),
        "fp32_teacher_fc1_input_activation_detached",
    )

    _require_equal("teacher checkpoint hash KD/QVF", kd_manifest.get("teacher_checkpoint_sha256"), qvf_manifest.get("teacher_checkpoint_sha256"))
    if teacher_manifest.get("checkpoint_sha256"):
        _require_equal("teacher checkpoint hash manifest/KD", teacher_manifest.get("checkpoint_sha256"), kd_manifest.get("teacher_checkpoint_sha256"))
    _require_equal(
        "int4 calibrated student checkpoint hash KD/QVF",
        kd_manifest.get("int4_calibrated_student_checkpoint_sha256"),
        qvf_manifest.get("int4_calibrated_student_checkpoint_sha256"),
    )
    _require_equal("qparams_hash KD/QVF", kd_manifest.get("qparams_hash"), qvf_manifest.get("qparams_hash"))
    _require_equal("qparams_csv_sha256 KD/QVF", kd_manifest.get("qparams_csv_sha256"), qvf_manifest.get("qparams_csv_sha256"))
    _require_equal("eval split hash KD/QVF", _manifest_split_hash(kd_manifest, "eval"), _manifest_split_hash(qvf_manifest, "eval"))
    teacher_eval_hash = _manifest_split_hash(teacher_manifest, "eval")
    if teacher_eval_hash:
        _require_equal("eval split hash teacher/KD", teacher_eval_hash, _manifest_split_hash(kd_manifest, "eval"))
        _require_equal("eval split hash teacher/QVF", teacher_eval_hash, _manifest_split_hash(qvf_manifest, "eval"))
    _require_equal(
        "calibration split hash KD/QVF",
        _manifest_split_hash(kd_manifest, "calibration"),
        _manifest_split_hash(qvf_manifest, "calibration"),
    )
    return {
        "module_name": QVF_MODULE_NAME,
        "hook_position": "forward_pre_hook input",
        "tensor_semantics": {
            "teacher": str(qvf_protocol.get("teacher_activation_semantic")),
            "student": str(qvf_protocol.get("student_activation_semantic")),
            "activation_hook": str(qvf_protocol.get("activation_hook_semantic")),
        },
        "qvf_protocol": dict(qvf_protocol),
    }


def _j_star_from_logits(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,C], got {tuple(logits.shape)}")
    masked = logits.detach().clone()
    row = torch.arange(int(labels.numel()), device=logits.device)
    masked[row, labels.long()] = -torch.inf
    return masked.argmax(dim=1).long()


def _fixed_margin(logits: Tensor, labels: Tensor, j_star: Tensor) -> Tensor:
    row = torch.arange(int(labels.numel()), device=logits.device)
    return logits[row, labels.long()].float() - logits[row, j_star.long()].float()


@contextmanager
def _temporary_no_param_grads(model: torch.nn.Module) -> Iterable[None]:
    previous: list[tuple[torch.nn.Parameter, bool]] = []
    for parameter in model.parameters():
        previous.append((parameter, bool(parameter.requires_grad)))
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in previous:
            parameter.requires_grad_(requires_grad)


def _forward_teacher_decision_tensors(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    layer_indices: Sequence[int],
) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, Tensor], dict[int, Tensor]]:
    layer_indices = tuple(sorted(int(item) for item in layer_indices))
    encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
        model,
        batch["waveform"],
        batch.get("padding_mask"),
    )
    captures: dict[int, Tensor] = {}
    handles: list[Any] = []
    graph_started = {"value": False}
    for layer_idx in layer_indices:
        module = get_ffn_module(model, layer_idx, QVF_MODULE_NAME)

        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], *, layer_idx: int = layer_idx) -> tuple[Any, ...]:
            if not inputs or not isinstance(inputs[0], Tensor):
                raise RuntimeError(f"teacher decision capture expected tensor input at L{layer_idx}.{QVF_MODULE_NAME}")
            if graph_started["value"] and bool(inputs[0].requires_grad):
                captures[layer_idx] = inputs[0]
                return inputs
            replacement = inputs[0].detach().clone().requires_grad_(True)
            captures[layer_idx] = replacement
            graph_started["value"] = True
            return (replacement, *inputs[1:])

        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.enable_grad(), _temporary_no_param_grads(model), _disable_encoder_layerdrop_for_qvf(model):
            features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
            logits = pooled_logits_from_features(model, features, token_padding_mask)
            labels = batch["label_id"].long().to(device=logits.device)
            j_star = _j_star_from_logits(logits, labels)
            margins = _fixed_margin(logits, labels, j_star)
            capture_values = [captures[int(layer_idx)] for layer_idx in layer_indices]
            grad_values = torch.autograd.grad(
                margins.sum(),
                capture_values,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    if set(captures) != set(int(item) for item in layer_indices):
        raise RuntimeError(f"missing teacher decision capture layers: {sorted(set(layer_indices) - set(captures))}")
    activations: dict[int, Tensor] = {}
    gradients: dict[int, Tensor] = {}
    valid_masks: dict[int, Tensor] = {}
    for layer_idx, raw_grad in zip(layer_indices, grad_values):
        canonical_h = canonicalize_activation_btc(
            captures[layer_idx].detach().float(),
            padding_mask=token_padding_mask,
            context=f"qvf_decision.teacher.L{layer_idx}.{QVF_MODULE_NAME}.input",
        )
        canonical_g = canonicalize_activation_btc(
            raw_grad.detach().float(),
            padding_mask=token_padding_mask,
            context=f"qvf_decision.gradient.L{layer_idx}.{QVF_MODULE_NAME}.input",
        )
        if canonical_h.padding_mask is None or canonical_g.padding_mask is None:
            raise RuntimeError(f"teacher capture missing padding mask at L{layer_idx}.{QVF_MODULE_NAME}")
        if not torch.equal(canonical_h.padding_mask, canonical_g.padding_mask):
            raise RuntimeError(f"teacher H/G padding masks differ at L{layer_idx}.{QVF_MODULE_NAME}")
        activations[layer_idx] = canonical_h.tensor.detach()
        gradients[layer_idx] = canonical_g.tensor.detach()
        valid_masks[layer_idx] = (~canonical_h.padding_mask).detach().bool()
    return logits.detach().float(), token_padding_mask.detach().bool(), activations, gradients, valid_masks


def _assert_same_mask(name: str, left: Tensor, right: Tensor) -> None:
    if tuple(left.shape) != tuple(right.shape) or not torch.equal(left.detach().cpu().bool(), right.detach().cpu().bool()):
        raise ProtocolError(f"valid-token mask mismatch for {name}: {tuple(left.shape)} vs {tuple(right.shape)}")


def _decompose_delta(delta: Tensor, gradient: Tensor, *, eps: float, tolerance: float, context: str) -> dict[str, float]:
    delta_vec = delta.reshape(-1).float()
    grad_vec = gradient.reshape(-1).float()
    g2 = torch.dot(grad_vec, grad_vec)
    projection = torch.dot(grad_vec, delta_vec)
    coeff = projection / (g2 + float(eps))
    parallel = coeff * grad_vec
    perpendicular = delta_vec - parallel
    total_error = torch.linalg.vector_norm(delta_vec)
    parallel_error = torch.linalg.vector_norm(parallel)
    orthogonal_error = torch.linalg.vector_norm(perpendicular)
    reconstruction_abs_error = torch.max(torch.abs(delta_vec - (parallel + perpendicular))).detach()
    margin_orthogonal_dot_abs = torch.abs(torch.dot(grad_vec, perpendicular)).detach()
    if _float(reconstruction_abs_error) > float(tolerance):
        raise RuntimeError(
            f"{context}: delta reconstruction failed: abs_error={_float(reconstruction_abs_error):.9g}, "
            f"tolerance={float(tolerance):.9g}"
        )
    if _float(margin_orthogonal_dot_abs) > float(tolerance):
        raise RuntimeError(
            f"{context}: first-order margin-orthogonal component is not orthogonal: "
            f"abs_dot={_float(margin_orthogonal_dot_abs):.9g}, tolerance={float(tolerance):.9g}"
        )
    return {
        "total_error": _float(total_error),
        "parallel_error": _float(parallel_error),
        "orthogonal_error": _float(orthogonal_error),
        "decision_error_ratio": _float(parallel_error.square() / (total_error.square() + float(eps))),
        "signed_projection": _float(projection),
        "predicted_margin_damage": _float(-projection),
        "reconstruction_abs_error": _float(reconstruction_abs_error),
        "margin_orthogonal_dot_abs": _float(margin_orthogonal_dot_abs),
    }


def _rows_for_batch(
    *,
    sample_ids: Sequence[str],
    metadata: Mapping[str, Mapping[str, Any]],
    labels: Tensor,
    target_layers: Sequence[int],
    teacher_logits: Tensor,
    vanilla_logits: Tensor,
    qvf_logits: Tensor,
    teacher_activations: Mapping[int, Tensor],
    teacher_gradients: Mapping[int, Tensor],
    teacher_valid_masks: Mapping[int, Tensor],
    vanilla_activations: Mapping[int, Tensor],
    vanilla_valid_masks: Mapping[int, Tensor],
    qvf_activations: Mapping[int, Tensor],
    qvf_valid_masks: Mapping[int, Tensor],
    tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels_cpu = labels.detach().cpu().long()
    j_star = _j_star_from_logits(teacher_logits, labels.to(device=teacher_logits.device)).detach().cpu().long()
    teacher_pred = teacher_logits.argmax(dim=1).detach().cpu().long()
    vanilla_pred = vanilla_logits.argmax(dim=1).detach().cpu().long()
    qvf_pred = qvf_logits.argmax(dim=1).detach().cpu().long()
    m_teacher = _fixed_margin(teacher_logits, labels.to(device=teacher_logits.device), j_star.to(device=teacher_logits.device)).detach().cpu()
    m_vanilla = _fixed_margin(vanilla_logits, labels.to(device=vanilla_logits.device), j_star.to(device=vanilla_logits.device)).detach().cpu()
    m_qvf = _fixed_margin(qvf_logits, labels.to(device=qvf_logits.device), j_star.to(device=qvf_logits.device)).detach().cpu()

    for layer_idx in tuple(int(item) for item in target_layers):
        target = f"L{layer_idx}.{QVF_MODULE_NAME}"
        _assert_same_mask(f"{target}.teacher_vs_vanilla", teacher_valid_masks[layer_idx], vanilla_valid_masks[layer_idx])
        _assert_same_mask(f"{target}.teacher_vs_qvf", teacher_valid_masks[layer_idx], qvf_valid_masks[layer_idx])
        for sample_index, sample_id in enumerate(sample_ids):
            if sample_id not in metadata:
                raise KeyError(f"eval metadata missing sample_id={sample_id!r}")
            label_id = int(labels_cpu[int(sample_index)].item())
            meta_label = int(metadata[sample_id]["label_id"])
            if label_id != meta_label:
                raise ProtocolError(f"{sample_id}: batch label_id={label_id} differs from eval.csv label_id={meta_label}")
            valid = teacher_valid_masks[layer_idx][int(sample_index)].detach().bool()
            valid_count = int(valid.sum().item())
            if valid_count <= 0:
                raise RuntimeError(f"{sample_id} {target}: zero valid tokens")
            h_t = teacher_activations[layer_idx][int(sample_index)][valid].float()
            g_t = teacher_gradients[layer_idx][int(sample_index)][valid].float()
            h_v = vanilla_activations[layer_idx][int(sample_index)][valid].to(device=h_t.device).float()
            h_q = qvf_activations[layer_idx][int(sample_index)][valid].to(device=h_t.device).float()
            if tuple(h_t.shape) != tuple(h_v.shape) or tuple(h_t.shape) != tuple(h_q.shape):
                raise RuntimeError(
                    f"{sample_id} {target}: tensor shape mismatch teacher={tuple(h_t.shape)} "
                    f"vanilla={tuple(h_v.shape)} qvf={tuple(h_q.shape)}"
                )
            grad_norm = _float(torch.linalg.vector_norm(g_t.reshape(-1)))
            vanilla = _decompose_delta(
                h_v - h_t,
                g_t,
                eps=EPS,
                tolerance=float(tolerance),
                context=f"{sample_id} {target} vanilla",
            )
            qvf = _decompose_delta(
                h_q - h_t,
                g_t,
                eps=EPS,
                tolerance=float(tolerance),
                context=f"{sample_id} {target} qvf",
            )
            margin_error_v = float(m_teacher[int(sample_index)].item() - m_vanilla[int(sample_index)].item())
            margin_error_q = float(m_teacher[int(sample_index)].item() - m_qvf[int(sample_index)].item())
            delta_total = float(vanilla["total_error"] - qvf["total_error"])
            delta_parallel = float(vanilla["parallel_error"] - qvf["parallel_error"])
            delta_orthogonal = float(vanilla["orthogonal_error"] - qvf["orthogonal_error"])
            qvf_reduction_case = _qvf_reduction_case(
                delta_total=delta_total,
                delta_parallel=delta_parallel,
                delta_orthogonal=delta_orthogonal,
                tolerance=float(tolerance),
            )
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "label_id": label_id,
                    "bucket": str(metadata[sample_id]["bucket"]),
                    "target": target,
                    "teacher_pred": int(teacher_pred[int(sample_index)].item()),
                    "vanilla_pred": int(vanilla_pred[int(sample_index)].item()),
                    "qvf_pred": int(qvf_pred[int(sample_index)].item()),
                    "j_star": int(j_star[int(sample_index)].item()),
                    "m_teacher": float(m_teacher[int(sample_index)].item()),
                    "m_vanilla": float(m_vanilla[int(sample_index)].item()),
                    "m_qvf": float(m_qvf[int(sample_index)].item()),
                    "vanilla_margin_error": margin_error_v,
                    "qvf_margin_error": margin_error_q,
                    "margin_recovery": float(abs(margin_error_v) - abs(margin_error_q)),
                    "valid_token_count": valid_count,
                    "gradient_norm": grad_norm,
                    "vanilla_total_error": float(vanilla["total_error"]),
                    "qvf_total_error": float(qvf["total_error"]),
                    "vanilla_parallel_error": float(vanilla["parallel_error"]),
                    "qvf_parallel_error": float(qvf["parallel_error"]),
                    "vanilla_orthogonal_error": float(vanilla["orthogonal_error"]),
                    "qvf_orthogonal_error": float(qvf["orthogonal_error"]),
                    "vanilla_decision_error_ratio": float(vanilla["decision_error_ratio"]),
                    "qvf_decision_error_ratio": float(qvf["decision_error_ratio"]),
                    "delta_total": delta_total,
                    "delta_parallel": delta_parallel,
                    "delta_orthogonal": delta_orthogonal,
                    "qvf_reduction_case": qvf_reduction_case,
                    "parallel_reduction_ratio": delta_parallel / (float(vanilla["parallel_error"]) + EPS),
                    "orthogonal_reduction_ratio": delta_orthogonal / (float(vanilla["orthogonal_error"]) + EPS),
                    "vanilla_signed_projection": float(vanilla["signed_projection"]),
                    "qvf_signed_projection": float(qvf["signed_projection"]),
                    "vanilla_predicted_margin_damage": float(vanilla["predicted_margin_damage"]),
                    "qvf_predicted_margin_damage": float(qvf["predicted_margin_damage"]),
                    "vanilla_reconstruction_abs_error": float(vanilla["reconstruction_abs_error"]),
                    "qvf_reconstruction_abs_error": float(qvf["reconstruction_abs_error"]),
                    "vanilla_margin_orthogonal_dot_abs": float(vanilla["margin_orthogonal_dot_abs"]),
                    "qvf_margin_orthogonal_dot_abs": float(qvf["margin_orthogonal_dot_abs"]),
                }
            )
    return rows


def _summary_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_rows = [row for row in sample_rows if str(row.get("target")) == target]
        for group in SUMMARY_GROUPS:
            group_rows = [row for row in target_rows if _group_mask(row, group)]
            values = {column: [float(row[column]) for row in group_rows] for column in (
                "delta_total",
                "delta_parallel",
                "delta_orthogonal",
                "parallel_reduction_ratio",
                "orthogonal_reduction_ratio",
                "margin_recovery",
            )}
            row = {
                "target": target,
                "group": group,
                "n": int(len(group_rows)),
                "mean_delta_total": _mean(values["delta_total"]),
                "median_delta_total": _median(values["delta_total"]),
                "mean_delta_parallel": _mean(values["delta_parallel"]),
                "median_delta_parallel": _median(values["delta_parallel"]),
                "mean_delta_orthogonal": _mean(values["delta_orthogonal"]),
                "median_delta_orthogonal": _median(values["delta_orthogonal"]),
                "mean_parallel_reduction_ratio": _mean(values["parallel_reduction_ratio"]),
                "median_parallel_reduction_ratio": _median(values["parallel_reduction_ratio"]),
                "mean_orthogonal_reduction_ratio": _mean(values["orthogonal_reduction_ratio"]),
                "median_orthogonal_reduction_ratio": _median(values["orthogonal_reduction_ratio"]),
                "mean_margin_recovery": _mean(values["margin_recovery"]),
                "median_margin_recovery": _median(values["margin_recovery"]),
                "count_parallel_and_orthogonal_down": sum(
                    1 for item in group_rows if str(item.get("qvf_reduction_case")) == "parallel_and_orthogonal_down"
                ),
                "count_orthogonal_down_parallel_flat": sum(
                    1 for item in group_rows if str(item.get("qvf_reduction_case")) == "orthogonal_down_parallel_flat"
                ),
                "count_total_down_parallel_up": sum(
                    1 for item in group_rows if str(item.get("qvf_reduction_case")) == "total_down_parallel_up"
                ),
                "count_other_reduction_case": sum(1 for item in group_rows if str(item.get("qvf_reduction_case")) == "other"),
                "pearson_delta_total_margin_recovery": _corr(values["delta_total"], values["margin_recovery"], spearman=False),
                "spearman_delta_total_margin_recovery": _corr(values["delta_total"], values["margin_recovery"], spearman=True),
                "pearson_delta_parallel_margin_recovery": _corr(values["delta_parallel"], values["margin_recovery"], spearman=False),
                "spearman_delta_parallel_margin_recovery": _corr(values["delta_parallel"], values["margin_recovery"], spearman=True),
                "pearson_delta_orthogonal_margin_recovery": _corr(values["delta_orthogonal"], values["margin_recovery"], spearman=False),
                "spearman_delta_orthogonal_margin_recovery": _corr(values["delta_orthogonal"], values["margin_recovery"], spearman=True),
            }
            rows.append(row)
    return rows


def _load_models(
    *,
    teacher_checkpoint: Path,
    vanilla_checkpoint: Path,
    qvf_checkpoint: Path,
    qparams_csv: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Mapping[str, Any], torch.nn.Module, Mapping[str, Any], torch.nn.Module, Mapping[str, Any]]:
    teacher_model, teacher_payload = load_trained_checkpoint(teacher_checkpoint, device=device)
    vanilla_model, vanilla_payload = load_trained_checkpoint(vanilla_checkpoint, device=device, qparams_csv_override=qparams_csv)
    qvf_model, qvf_payload = load_trained_checkpoint(qvf_checkpoint, device=device, qparams_csv_override=qparams_csv)
    for name, model in (("teacher", teacher_model), ("vanilla", vanilla_model), ("qvf", qvf_model)):
        model.eval()
        if str(getattr(model, "_baseline_model_kind", "")) == "fp32_teacher" and name != "teacher":
            raise ProtocolError(f"{name} model unexpectedly loaded as FP32 teacher")
    return teacher_model, teacher_payload, vanilla_model, vanilla_payload, qvf_model, qvf_payload


def _reference_hash_from_loaded(model: torch.nn.Module, payload: Mapping[str, Any]) -> str:
    return str(
        getattr(model, "_baseline_reference_checkpoint_hash", "")
        or payload.get("_baseline_reference_checkpoint_hash")
        or payload.get("teacher_checkpoint_sha256")
        or payload.get("teacher_checkpoint_hash")
        or ""
    )


def _checkpoint_identity_from_payload(model: torch.nn.Module, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_kind": payload.get("_baseline_model_kind"),
        "reference_checkpoint_hash": _reference_hash_from_loaded(model, payload),
        "qparams_hash": payload.get("qparams_hash"),
        "qparams_csv": payload.get("qparams_csv"),
        "qparams_csv_sha256": payload.get("qparams_csv_sha256") or payload.get("qparams_csv_hash"),
        "quantized_layers_count": len(payload.get("quantized_layers", []) or []),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed)
    qvf_beta = float(args.qvf_beta)
    paths = dataset_paths(args.dataset, None)
    crop_protocol = _assert_center_crop_protocol()
    kd_manifest_path = Path(args.kd_manifest) if args.kd_manifest else _default_kd_manifest(paths.root, seed)
    qvf_manifest_path = Path(args.qvf_manifest) if args.qvf_manifest else _default_qvf_manifest(paths.root, seed, qvf_beta)
    teacher_manifest_path = Path(args.teacher_manifest) if args.teacher_manifest else _default_teacher_manifest(paths.root, seed)
    kd_manifest = _read_json(kd_manifest_path)
    qvf_manifest = _read_json(qvf_manifest_path)
    teacher_manifest = _read_json(teacher_manifest_path)
    tensor_protocol = _validate_manifests(
        kd_manifest=kd_manifest,
        qvf_manifest=qvf_manifest,
        teacher_manifest=teacher_manifest,
        qvf_beta=qvf_beta,
    )

    teacher_checkpoint = _resolve_manifest_path(qvf_manifest.get("teacher_checkpoint") or teacher_manifest.get("checkpoint"), dataset_root=paths.root, description="teacher checkpoint")
    vanilla_checkpoint = _resolve_manifest_path(kd_manifest.get("checkpoint"), dataset_root=paths.root, description="Vanilla KD checkpoint")
    qvf_checkpoint = _resolve_manifest_path(qvf_manifest.get("checkpoint"), dataset_root=paths.root, description="QVF checkpoint")
    qparams_csv = _resolve_manifest_path(qvf_manifest.get("qparams_csv") or kd_manifest.get("qparams_csv"), dataset_root=paths.root, description="shared qparams_csv")
    for description, path in (
        ("teacher checkpoint", teacher_checkpoint),
        ("Vanilla KD checkpoint", vanilla_checkpoint),
        ("QVF checkpoint", qvf_checkpoint),
    ):
        _assert_not_quant_injection_checkpoint(path, description=description)

    hashes = {
        "teacher_checkpoint_sha256": _require_file_hash(
            teacher_checkpoint,
            qvf_manifest.get("teacher_checkpoint_sha256") or teacher_manifest.get("checkpoint_sha256"),
            description="teacher checkpoint",
        ),
        "vanilla_checkpoint_sha256": _require_file_hash(vanilla_checkpoint, kd_manifest.get("checkpoint_sha256"), description="Vanilla KD checkpoint"),
        "qvf_checkpoint_sha256": _require_file_hash(qvf_checkpoint, qvf_manifest.get("checkpoint_sha256"), description="QVF checkpoint"),
        "qparams_csv_sha256": _require_file_hash(qparams_csv, qvf_manifest.get("qparams_csv_sha256"), description="qparams_csv"),
    }

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    teacher_model, teacher_payload, vanilla_model, vanilla_payload, qvf_model, qvf_payload = _load_models(
        teacher_checkpoint=teacher_checkpoint,
        vanilla_checkpoint=vanilla_checkpoint,
        qvf_checkpoint=qvf_checkpoint,
        qparams_csv=qparams_csv,
        device=device,
    )
    vanilla_reference_hash = _reference_hash_from_loaded(vanilla_model, vanilla_payload)
    qvf_reference_hash = _reference_hash_from_loaded(qvf_model, qvf_payload)
    _require_equal(
        "loaded Vanilla/QVF teacher reference hash",
        vanilla_reference_hash,
        qvf_reference_hash,
    )
    _require_equal("loaded KD/QVF qparams_hash", vanilla_payload.get("qparams_hash"), qvf_payload.get("qparams_hash"))
    if vanilla_reference_hash != hashes["teacher_checkpoint_sha256"]:
        raise ProtocolError("loaded Vanilla KD teacher reference hash differs from the resolved teacher checkpoint")
    if qvf_reference_hash != hashes["teacher_checkpoint_sha256"]:
        raise ProtocolError("loaded QVF teacher reference hash differs from the resolved teacher checkpoint")

    metadata = _eval_metadata(paths.eval_csv)
    loader = build_loader(paths, split="eval", batch_size=int(args.batch_size), num_workers=int(args.num_workers), seed=seed)

    sample_rows: list[dict[str, Any]] = []
    seen_sample_ids: list[str] = []
    teacher_j_star_by_sample: dict[str, int] = {}
    for batch in loader:
        moved = move_batch_to_device(batch, device)
        sample_ids = _batch_sample_ids(moved)
        labels = moved["label_id"].long()
        with torch.no_grad():
            vanilla_logits, vanilla_padding, vanilla_activations, vanilla_valid_masks = forward_student_qvf(
                vanilla_model,
                moved,
                layer_indices=TARGET_LAYERS,
            )
            qvf_logits, qvf_padding, qvf_activations, qvf_valid_masks = forward_student_qvf(
                qvf_model,
                moved,
                layer_indices=TARGET_LAYERS,
            )
        teacher_logits, teacher_padding, teacher_activations, teacher_gradients, teacher_valid_masks = _forward_teacher_decision_tensors(
            teacher_model,
            moved,
            layer_indices=TARGET_LAYERS,
        )
        _assert_same_mask("token_padding.teacher_vs_vanilla", ~teacher_padding, ~vanilla_padding)
        _assert_same_mask("token_padding.teacher_vs_qvf", ~teacher_padding, ~qvf_padding)
        j_star = _j_star_from_logits(teacher_logits, labels.to(device=teacher_logits.device)).detach().cpu().long()
        for sample_id, competitor in zip(sample_ids, j_star.tolist()):
            if sample_id in teacher_j_star_by_sample and teacher_j_star_by_sample[sample_id] != int(competitor):
                raise ProtocolError(f"{sample_id}: j_star changed within eval pass")
            teacher_j_star_by_sample[str(sample_id)] = int(competitor)
        seen_sample_ids.extend(str(item) for item in sample_ids)
        sample_rows.extend(
            _rows_for_batch(
                sample_ids=sample_ids,
                metadata=metadata,
                labels=labels,
                target_layers=TARGET_LAYERS,
                teacher_logits=teacher_logits,
                vanilla_logits=vanilla_logits.detach().float(),
                qvf_logits=qvf_logits.detach().float(),
                teacher_activations=teacher_activations,
                teacher_gradients=teacher_gradients,
                teacher_valid_masks=teacher_valid_masks,
                vanilla_activations=vanilla_activations,
                vanilla_valid_masks=vanilla_valid_masks,
                qvf_activations=qvf_activations,
                qvf_valid_masks=qvf_valid_masks,
                tolerance=float(args.numerical_tolerance),
            )
        )

    expected_eval_order = list(metadata)
    if seen_sample_ids != expected_eval_order:
        raise ProtocolError("DataLoader sample_id order differs from eval.csv order")
    summary_rows = _summary_rows(sample_rows)

    output_dir = Path(args.output_dir) if args.output_dir else qvf_manifest_path.parent / "decision_subspace_audit"
    sample_csv = output_dir / "qvf_decision_subspace_sample.csv"
    summary_csv = output_dir / "qvf_decision_subspace_summary.csv"
    manifest_json = output_dir / "manifest.json"
    _write_csv(sample_csv, sample_rows, SAMPLE_COLUMNS)
    _write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)

    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": paths.name,
        "device": str(device),
        "seed": seed,
        "qvf_beta": qvf_beta,
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": hashes["teacher_checkpoint_sha256"],
        "vanilla_kd_checkpoint": str(vanilla_checkpoint),
        "vanilla_kd_checkpoint_sha256": hashes["vanilla_checkpoint_sha256"],
        "qvf_checkpoint": str(qvf_checkpoint),
        "qvf_checkpoint_sha256": hashes["qvf_checkpoint_sha256"],
        "qparams_csv": str(qparams_csv),
        "qparams_csv_sha256": hashes["qparams_csv_sha256"],
        "manifests": {
            "teacher": str(teacher_manifest_path.resolve(strict=False)),
            "vanilla_kd": str(kd_manifest_path.resolve(strict=False)),
            "qvf": str(qvf_manifest_path.resolve(strict=False)),
        },
        "loaded_checkpoint_identity": {
            "teacher": _checkpoint_identity_from_payload(teacher_model, teacher_payload),
            "vanilla_kd": _checkpoint_identity_from_payload(vanilla_model, vanilla_payload),
            "qvf": _checkpoint_identity_from_payload(qvf_model, qvf_payload),
        },
        "fixed_teacher_gradient": True,
        "fixed_competitor": "j_star from FP32 teacher argmax over j != y",
        "target_tensors": [
            {
                "target": f"L{layer}.{QVF_MODULE_NAME}",
                "module_name": f"encoder.layers.{layer}.{QVF_MODULE_NAME}",
                "hook_position": tensor_protocol["hook_position"],
                "tensor_semantics": tensor_protocol["tensor_semantics"],
            }
            for layer in TARGET_LAYERS
        ],
        "hook_positions": {
            "teacher_gradient": "temporary forward_pre_hook replacement at QVF teacher fc1 input tensor",
            "student_vanilla": "QuantLinear qvf activation capture after activation fake-quant/dequant",
            "student_qvf": "QuantLinear qvf activation capture after activation fake-quant/dequant",
        },
        "decision_subspace_naming": {
            "parallel": "component aligned with the fixed FP32 teacher margin gradient",
            "perpendicular": "first-order margin-orthogonal component under the local FP32 teacher margin gradient",
        },
        "pooling": DEFAULT_POOLING,
        "crop_protocol": crop_protocol,
        "activation_quantization_protocol": "from shared qparams/qparams_hash in Vanilla KD and QVF checkpoints",
        "weight_quantization_protocol": "from shared qparams/qparams_hash in Vanilla KD and QVF checkpoints",
        "split_hashes": {
            "teacher_eval": _manifest_split_hash(teacher_manifest, "eval"),
            "vanilla_kd_eval": _manifest_split_hash(kd_manifest, "eval"),
            "qvf_eval": _manifest_split_hash(qvf_manifest, "eval"),
            "vanilla_kd_calibration": _manifest_split_hash(kd_manifest, "calibration"),
            "qvf_calibration": _manifest_split_hash(qvf_manifest, "calibration"),
        },
        "calibration_qparams": {
            "qparams_hash": str(qvf_manifest.get("qparams_hash")),
            "calibration_csv_hash": _manifest_split_hash(qvf_manifest, "calibration"),
            "calibration_recomputed": False,
        },
        "qvf_reduction_case_definitions": {
            "parallel_and_orthogonal_down": "delta_parallel > tolerance and delta_orthogonal > tolerance",
            "orthogonal_down_parallel_flat": "delta_orthogonal > tolerance and abs(delta_parallel) <= tolerance",
            "total_down_parallel_up": "delta_total > tolerance and delta_parallel < -tolerance; evaluated first",
            "other": "all samples not matching the three predeclared descriptive cases",
        },
        "validations": {
            "vanilla_qvf_same_fp32_teacher": True,
            "teacher_manifest_eval_hash_matches_kd_qvf": True,
            "same_eval_csv": True,
            "same_sample_id_order": True,
            "same_valid_token_mask": True,
            "same_j_star_source": True,
            "no_quant_injection_checkpoint": True,
            "training_run": False,
            "calibration_run": False,
            "test_touched": False,
            "delta_reconstruction_checked": True,
            "first_order_margin_orthogonal_dot_checked": True,
            "numerical_tolerance": float(args.numerical_tolerance),
        },
        "sample_order_hash": _sample_order_hash(seen_sample_ids),
        "eval_sample_count": int(len(seen_sample_ids)),
        "sample_rows": int(len(sample_rows)),
        "outputs": {
            "sample_csv": str(sample_csv.resolve(strict=False)),
            "summary_csv": str(summary_csv.resolve(strict=False)),
            "manifest_json": str(manifest_json.resolve(strict=False)),
        },
        "qvf_protocol": tensor_protocol["qvf_protocol"],
        "test_touched": False,
    }
    _write_json(manifest_json, _portable_json(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QVF Decision-Subspace Audit for Vanilla KD vs canonical QVF beta=0.05.")
    parser.add_argument("--dataset", default=None, help="Dataset key from config.DATASETS; default: config.ACTIVE_DATASET.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--qvf-beta", type=float, default=0.05)
    parser.add_argument("--teacher-manifest", default=None, help="Optional FP32 teacher manifest path.")
    parser.add_argument("--kd-manifest", default=None, help="Optional Vanilla KD run manifest path.")
    parser.add_argument("--qvf-manifest", default=None, help="Optional canonical QVF manifest path.")
    parser.add_argument("--output-dir", default=None, help="Default: QVF manifest directory / decision_subspace_audit.")
    parser.add_argument("--device", default=None, help="Default: cuda if available else cpu.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--numerical-tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = run(args)
    print(json.dumps({"ok": True, "outputs": manifest["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
