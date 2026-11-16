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

import src.export.single_layer_causal_injection as single_layer
from src.baseline.export_eval import pooled_logits_from_features, prepare_encoder_input_for_export
from src.utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
from src.utils.semantic_geometry_cache import file_identity, qparams_identity_manifest, stable_json_hash


EXPERIMENT_NAME = "three_factor_decomposition"
PROTOCOL_VERSION = "three_factor_decomposition.v1"
DEFAULT_OUTPUT_SUBDIR = EXPERIMENT_NAME
HOOK_POSITION = "linear_module_output"
EPS = 1e-12
DEFAULT_NUMERICAL_TOLERANCE = 1e-6
GROUPS = ("overall", "many", "medium", "tail", "few")


SAMPLE_COLUMNS = [
    "sample_index",
    "sample_id",
    "audio_path",
    "label_id",
    "bucket",
    "target",
    "hook_position",
    "m_fp32",
    "m_injected",
    "actual_margin_drop",
    "classification_flip",
    "fp32_pred",
    "injected_pred",
    "j_star",
    "fp32_dynamic_margin",
    "injected_dynamic_margin",
    "valid_token_count",
    "channel_count",
    "flattened_valid_elements",
    "padding_tokens_excluded",
    "quantization_magnitude_Q",
    "decision_sensitivity_S",
    "direction_cosine_A",
    "harmful_alignment",
    "first_order_margin_change",
    "predicted_margin_drop",
    "factor_product_QSA",
    "factor_algebra_abs_error",
    "Q_times_S",
    "Q_times_harmful_alignment",
    "S_times_harmful_alignment",
]


CORRELATION_COLUMNS = [
    "target",
    "group",
    "factor",
    "n",
    "pearson_with_margin_drop",
    "spearman_with_margin_drop",
]


TAYLOR_COLUMNS = [
    "target",
    "group",
    "n",
    "mae",
    "rmse",
    "pearson",
    "spearman",
    "sign_agreement",
]


@dataclass(frozen=True)
class TargetFactorResult:
    spec: single_layer.TargetSpec
    sample_order: tuple[str, ...]
    fp32_logits: Tensor
    injected_logits: Tensor
    fp32_preds: Tensor
    injected_preds: Tensor
    sample_rows: tuple[dict[str, Any], ...]
    quant_state: tuple[dict[str, Any], ...]
    single_layer_reference_validation: Mapping[str, Any]
    checkpoint_clean_reloads: int


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


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = dict(single_layer._resolve_paths(args))
    output_dir = Path(args.output_dir) if args.output_dir is not None else paths["dataset_root"] / DEFAULT_OUTPUT_SUBDIR
    paths["output_dir"] = output_dir
    return paths


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
        raise RuntimeError(f"{context}: missing original_layout")
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


def _assert_equal_tensor(name: str, left: Tensor, right: Tensor) -> None:
    if torch.equal(left, right):
        return
    diff = torch.max(torch.abs(left.detach().cpu().float() - right.detach().cpu().float())).item()
    raise RuntimeError(f"{name}: tensors are not exactly identical; max_abs_diff={float(diff):.8g}")


def _forward_fp32_output_gradient(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    spec: single_layer.TargetSpec,
) -> dict[str, Any]:
    model.eval()
    single_layer._set_no_param_grad(model)
    moved = single_layer._move_batch_to_device(batch, device)
    labels = moved["label_id"].long()
    captured: dict[str, Tensor] = {}
    target_module = single_layer.get_ffn_module(model, spec.layer_idx, spec.module_name)

    def post_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> Tensor:
        del module, args, kwargs
        if not isinstance(output, Tensor):
            raise RuntimeError(f"{spec.target}: expected tensor output from target module")
        replacement = _capture_for_grad(output)
        captured["raw"] = replacement
        return replacement

    handle = target_module.register_forward_hook(post_hook, with_kwargs=True)
    try:
        with torch.enable_grad():
            encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
                model,
                moved["waveform"],
                moved["padding_mask"],
            )
            features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
            logits = pooled_logits_from_features(model, features.float(), token_padding_mask)
            if "raw" not in captured:
                raise RuntimeError(f"{spec.target}: output hook did not fire")
            raw = captured["raw"]
            h_fp32, layout_info = _canonicalize(raw, token_padding_mask, context=f"{spec.target}.fp32_output")
            preds = logits.detach().argmax(dim=1)
            fp32_correct = preds.eq(labels)
            j_star = single_layer._competitor_indices(logits.detach(), labels)
            m_fp32 = single_layer._fixed_margin(logits, labels, j_star)
            gradient = torch.zeros_like(h_fp32)
            if bool(fp32_correct.any().item()):
                selected_margin = m_fp32[fp32_correct].sum()
                (grad_raw,) = torch.autograd.grad(
                    selected_margin,
                    raw,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                gradient = _canonicalize_like(
                    grad_raw.detach(),
                    token_padding_mask,
                    layout_info,
                    context=f"{spec.target}.fp32_output_gradient",
                )
    finally:
        handle.remove()

    single_layer._assert_no_param_grads(model, context=f"{spec.target}.fp32_output_gradient_forward")
    return {
        "logits": logits.detach().cpu().float(),
        "labels": labels.detach().cpu().long(),
        "preds": preds.detach().cpu().long(),
        "j_star": j_star.detach().cpu().long(),
        "m_fp32": m_fp32.detach().cpu().float(),
        "dynamic_margin": single_layer._dynamic_margin(logits.detach(), labels).detach().cpu().float(),
        "fp32_correct": fp32_correct.detach().cpu().bool(),
        "H": h_fp32.detach().cpu().float(),
        "G": gradient.detach().cpu().float(),
        "token_padding_mask": token_padding_mask.detach().cpu().bool(),
        "layout_info": dict(layout_info),
    }


def _forward_injected_output(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    spec: single_layer.TargetSpec,
) -> dict[str, Any]:
    single_layer._assert_single_target_quant_state(model, spec)
    model.eval()
    single_layer._set_no_param_grad(model)
    moved = single_layer._move_batch_to_device(batch, device)
    labels = moved["label_id"].long()
    captured: dict[str, Tensor] = {}
    target_module = single_layer.get_ffn_module(model, spec.layer_idx, spec.module_name)

    def post_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        del module, args, kwargs
        if not isinstance(output, Tensor):
            raise RuntimeError(f"{spec.target}: expected tensor output from injected target module")
        captured["raw"] = output.detach()

    handle = target_module.register_forward_hook(post_hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
                model,
                moved["waveform"],
                moved["padding_mask"],
            )
            features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
            logits = pooled_logits_from_features(model, features.float(), token_padding_mask)
            if "raw" not in captured:
                raise RuntimeError(f"{spec.target}: injected output hook did not fire")
            h_q, layout_info = _canonicalize(captured["raw"], token_padding_mask, context=f"{spec.target}.injected_output")
    finally:
        handle.remove()

    single_layer._assert_single_target_quant_state(model, spec)
    single_layer._assert_no_param_grads(model, context=f"{spec.target}.injected_output_forward")
    return {
        "logits": logits.detach().cpu().float(),
        "labels": labels.detach().cpu().long(),
        "preds": logits.detach().argmax(dim=1).cpu().long(),
        "dynamic_margin": single_layer._dynamic_margin(logits.detach(), labels).detach().cpu().float(),
        "H_q": h_q.detach().cpu().float(),
        "token_padding_mask": token_padding_mask.detach().cpu().bool(),
        "layout_info": dict(layout_info),
    }


def _flatten_valid(
    *,
    h_fp32: Tensor,
    h_q: Tensor,
    gradient: Tensor,
    token_padding_mask: Tensor,
    spec: single_layer.TargetSpec,
    sample_index: int,
) -> tuple[Tensor, Tensor, Tensor, int, int, int]:
    if tuple(h_fp32.shape) != tuple(h_q.shape) or tuple(h_fp32.shape) != tuple(gradient.shape):
        raise RuntimeError(
            f"{spec.target}: H/H_q/G shape mismatch: "
            f"H={tuple(h_fp32.shape)} H_q={tuple(h_q.shape)} G={tuple(gradient.shape)}"
        )
    if h_fp32.ndim != 3:
        raise RuntimeError(f"{spec.target}: expected [B,T,C] output representation, got {tuple(h_fp32.shape)}")
    if tuple(token_padding_mask.shape) != tuple(h_fp32.shape[:2]):
        raise RuntimeError(
            f"{spec.target}: token padding mask {tuple(token_padding_mask.shape)} "
            f"does not align H {tuple(h_fp32.shape)}"
        )
    valid = ~token_padding_mask[int(sample_index)].bool()
    valid_token_count = int(valid.sum().item())
    padding_token_count = int(valid.numel() - valid_token_count)
    if valid_token_count <= 0:
        raise RuntimeError(f"{spec.target}: sample index {sample_index} has zero valid tokens")
    channel_count = int(h_fp32.shape[-1])
    h_valid = h_fp32[int(sample_index)][valid].reshape(-1).double()
    hq_valid = h_q[int(sample_index)][valid].reshape(-1).double()
    g_valid = gradient[int(sample_index)][valid].reshape(-1).double()
    expected_numel = valid_token_count * channel_count
    if int(h_valid.numel()) != expected_numel or int(hq_valid.numel()) != expected_numel or int(g_valid.numel()) != expected_numel:
        raise RuntimeError(f"{spec.target}: padding exclusion flatten size mismatch")
    return h_valid, hq_valid, g_valid, valid_token_count, padding_token_count, channel_count


def _factor_metrics(
    *,
    h_fp32: Tensor,
    h_q: Tensor,
    gradient: Tensor,
    token_padding_mask: Tensor,
    spec: single_layer.TargetSpec,
    sample_index: int,
    numerical_tolerance: float,
) -> dict[str, Any]:
    h_valid, hq_valid, g_valid, valid_token_count, padding_token_count, channel_count = _flatten_valid(
        h_fp32=h_fp32,
        h_q=h_q,
        gradient=gradient,
        token_padding_mask=token_padding_mask,
        spec=spec,
        sample_index=sample_index,
    )
    delta = hq_valid - h_valid
    q = float(torch.linalg.vector_norm(delta).item())
    s = float(torch.linalg.vector_norm(g_valid).item())
    first_order = float(torch.dot(g_valid, delta).item())
    denom = q * s
    if denom > EPS:
        a = float(first_order / denom)
    else:
        if abs(first_order) > float(numerical_tolerance):
            raise RuntimeError(
                f"{spec.target}: nonzero first-order change with zero Q*S: "
                f"first_order={first_order:.8g}, Q={q:.8g}, S={s:.8g}"
            )
        a = 0.0
    factor_product = float(q * s * a)
    algebra_error = abs(first_order - factor_product)
    if algebra_error >= float(numerical_tolerance):
        raise RuntimeError(
            f"{spec.target}: factor algebra check failed: "
            f"|first_order_margin_change - Q*S*A|={algebra_error:.8g} "
            f">= tolerance={float(numerical_tolerance):.8g}"
        )
    return {
        "valid_token_count": valid_token_count,
        "padding_token_count": padding_token_count,
        "channel_count": channel_count,
        "flattened_valid_elements": int(valid_token_count * channel_count),
        "padding_tokens_excluded": True,
        "quantization_magnitude_Q": q,
        "decision_sensitivity_S": s,
        "direction_cosine_A": a,
        "harmful_alignment": -a,
        "first_order_margin_change": first_order,
        "predicted_margin_drop": -first_order,
        "factor_product_QSA": factor_product,
        "factor_algebra_abs_error": algebra_error,
        "Q_times_S": q * s,
        "Q_times_harmful_alignment": q * (-a),
        "S_times_harmful_alignment": s * (-a),
    }


def _run_target_factor(
    *,
    spec: single_layer.TargetSpec,
    checkpoint_path: Path,
    qparams: Mapping[str, Any],
    loader: Any,
    device: torch.device,
    metadata: Mapping[str, Mapping[str, Any]],
    numerical_tolerance: float,
) -> TargetFactorResult:
    fp32_model, _fp32_checkpoint = single_layer._load_clean_fp32_model(
        checkpoint_path,
        device=device,
        context=f"{spec.target}.three_factor_fp32_clean_reload",
    )
    reference_fp32_model, _reference_fp32_checkpoint = single_layer._load_clean_fp32_model(
        checkpoint_path,
        device=device,
        context=f"{spec.target}.single_layer_reference_fp32_clean_reload",
    )
    injected_model, injected_controller, quant_state = single_layer._prepare_injection_model(
        checkpoint_path,
        qparams,
        spec=spec,
        device=device,
    )
    reference_injected_model, reference_injected_controller, _reference_quant_state = single_layer._prepare_injection_model(
        checkpoint_path,
        qparams,
        spec=spec,
        device=device,
    )

    sample_rows: list[dict[str, Any]] = []
    sample_order: list[str] = []
    fp32_logits_chunks: list[Tensor] = []
    injected_logits_chunks: list[Tensor] = []
    fp32_pred_chunks: list[Tensor] = []
    injected_pred_chunks: list[Tensor] = []
    sample_offset = 0
    reference_checks = {
        "fp32_logits_identical_to_single_layer_reference": True,
        "injected_logits_identical_to_single_layer_reference": True,
        "fp32_token_padding_mask_identical_to_reference": True,
        "injected_token_padding_mask_identical_to_reference": True,
    }

    try:
        for batch in loader:
            sample_ids = single_layer._batch_sample_ids(batch)
            audio_paths = single_layer._batch_audio_paths(batch, sample_ids, metadata)

            fp32 = _forward_fp32_output_gradient(fp32_model, batch, device=device, spec=spec)
            injected = _forward_injected_output(injected_model, batch, device=device, spec=spec)
            ref_fp32_logits, ref_labels, ref_fp32_mask = single_layer._forward_logits(reference_fp32_model, batch, device=device)
            ref_injected_logits, ref_injected_labels, ref_injected_mask = single_layer._forward_logits(
                reference_injected_model,
                batch,
                device=device,
            )

            _assert_equal_tensor(f"{spec.target}.fp32_logits_vs_single_layer_reference", fp32["logits"], ref_fp32_logits)
            _assert_equal_tensor(f"{spec.target}.injected_logits_vs_single_layer_reference", injected["logits"], ref_injected_logits)
            _assert_equal_tensor(f"{spec.target}.fp32_labels_vs_reference", fp32["labels"], ref_labels)
            _assert_equal_tensor(f"{spec.target}.injected_labels_vs_reference", injected["labels"], ref_injected_labels)
            _assert_equal_tensor(f"{spec.target}.fp32_token_padding_mask_vs_reference", fp32["token_padding_mask"], ref_fp32_mask)
            _assert_equal_tensor(
                f"{spec.target}.injected_token_padding_mask_vs_reference",
                injected["token_padding_mask"],
                ref_injected_mask,
            )
            _assert_equal_tensor(f"{spec.target}.labels_fp32_vs_injected", fp32["labels"], injected["labels"])
            _assert_equal_tensor(f"{spec.target}.token_padding_mask_fp32_vs_injected", fp32["token_padding_mask"], injected["token_padding_mask"])

            labels: Tensor = fp32["labels"]
            j_star: Tensor = fp32["j_star"]
            m_fp32: Tensor = fp32["m_fp32"]
            m_injected = single_layer._fixed_margin(injected["logits"], labels, j_star)
            fp32_preds: Tensor = fp32["preds"]
            injected_preds: Tensor = injected["preds"]
            fp32_correct: Tensor = fp32["fp32_correct"]

            sample_order.extend(sample_ids)
            fp32_logits_chunks.append(fp32["logits"])
            injected_logits_chunks.append(injected["logits"])
            fp32_pred_chunks.append(fp32_preds)
            injected_pred_chunks.append(injected_preds)

            h_fp32: Tensor = fp32["H"]
            h_q: Tensor = injected["H_q"]
            gradient: Tensor = fp32["G"]
            token_padding_mask: Tensor = fp32["token_padding_mask"]

            for local_index, sample_id in enumerate(sample_ids):
                if not bool(fp32_correct[local_index].item()):
                    continue
                sid = str(sample_id)
                label_id = int(labels[local_index].item())
                injected_pred = int(injected_preds[local_index].item())
                meta = metadata.get(sid, {})
                factors = _factor_metrics(
                    h_fp32=h_fp32,
                    h_q=h_q,
                    gradient=gradient,
                    token_padding_mask=token_padding_mask,
                    spec=spec,
                    sample_index=local_index,
                    numerical_tolerance=numerical_tolerance,
                )
                row = {
                    "sample_index": int(sample_offset + local_index),
                    "sample_id": sid,
                    "audio_path": str(audio_paths[local_index]),
                    "label_id": label_id,
                    "bucket": str(meta.get("bucket", "")),
                    "target": spec.target,
                    "hook_position": HOOK_POSITION,
                    "m_fp32": float(m_fp32[local_index].item()),
                    "m_injected": float(m_injected[local_index].item()),
                    "actual_margin_drop": float((m_fp32[local_index] - m_injected[local_index]).item()),
                    "classification_flip": int(injected_pred != label_id),
                    "fp32_pred": int(fp32_preds[local_index].item()),
                    "injected_pred": injected_pred,
                    "j_star": int(j_star[local_index].item()),
                    "fp32_dynamic_margin": float(fp32["dynamic_margin"][local_index].item()),
                    "injected_dynamic_margin": float(injected["dynamic_margin"][local_index].item()),
                    **factors,
                }
                sample_rows.append(row)
            sample_offset += len(sample_ids)
        single_layer._assert_single_target_quant_state(injected_model, spec)
        single_layer._assert_single_target_quant_state(reference_injected_model, spec)
    finally:
        injected_controller.remove_padding_mask_hooks()
        reference_injected_controller.remove_padding_mask_hooks()

    return TargetFactorResult(
        spec=spec,
        sample_order=tuple(sample_order),
        fp32_logits=torch.cat(fp32_logits_chunks, dim=0).contiguous(),
        injected_logits=torch.cat(injected_logits_chunks, dim=0).contiguous(),
        fp32_preds=torch.cat(fp32_pred_chunks, dim=0).contiguous(),
        injected_preds=torch.cat(injected_pred_chunks, dim=0).contiguous(),
        sample_rows=tuple(sample_rows),
        quant_state=tuple(quant_state),
        single_layer_reference_validation=reference_checks,
        checkpoint_clean_reloads=4,
    )


def _assert_shared_fp32_baseline(results: Sequence[TargetFactorResult]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("no target results to validate")
    baseline = results[0]
    for result in results[1:]:
        if result.sample_order != baseline.sample_order:
            raise RuntimeError(f"FP32 sample order differs: {baseline.spec.target} vs {result.spec.target}")
        _assert_equal_tensor(
            f"fp32_logits_shared_baseline.{baseline.spec.target}_vs_{result.spec.target}",
            baseline.fp32_logits,
            result.fp32_logits,
        )
        _assert_equal_tensor(
            f"fp32_predictions_shared_baseline.{baseline.spec.target}_vs_{result.spec.target}",
            baseline.fp32_preds,
            result.fp32_preds,
        )
    return {
        "same_fp32_baseline_across_targets": True,
        "fp32_logits_identical_across_targets": True,
        "fp32_predictions_identical_across_targets": True,
        "sample_order_identical_across_targets": True,
        "sample_order_hash": stable_json_hash(list(baseline.sample_order)),
        "num_eval_samples_seen": int(len(baseline.sample_order)),
    }


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


def _rows_for_group(rows: Sequence[Mapping[str, Any]], target: str, group: str) -> list[Mapping[str, Any]]:
    target_rows = [row for row in rows if str(row.get("target")) == str(target)]
    if group == "overall":
        return target_rows
    if group == "tail":
        return [row for row in target_rows if str(row.get("bucket", "")).lower() in {"medium", "few"}]
    return [row for row in target_rows if str(row.get("bucket", "")).lower() == group]


FACTOR_COLUMNS = {
    "Q": "quantization_magnitude_Q",
    "S": "decision_sensitivity_S",
    "A_harm": "harmful_alignment",
    "Q_times_S": "Q_times_S",
    "Q_times_A_harm": "Q_times_harmful_alignment",
    "S_times_A_harm": "S_times_harmful_alignment",
    "predicted_margin_drop": "predicted_margin_drop",
}


def _build_correlation_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in single_layer.TARGET_SPECS:
        for group in GROUPS:
            rows = _rows_for_group(sample_rows, spec.target, group)
            for factor, column in FACTOR_COLUMNS.items():
                x, y = _finite_arrays(rows, column, "actual_margin_drop")
                result.append(
                    {
                        "target": spec.target,
                        "group": group,
                        "factor": factor,
                        "n": int(x.size),
                        "pearson_with_margin_drop": _pearson(x, y),
                        "spearman_with_margin_drop": _spearman(x, y),
                    }
                )
    return result


def _build_taylor_rows(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in single_layer.TARGET_SPECS:
        for group in GROUPS:
            rows = _rows_for_group(sample_rows, spec.target, group)
            pred, actual = _finite_arrays(rows, "predicted_margin_drop", "actual_margin_drop")
            if pred.size:
                diff = pred - actual
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff * diff)))
                sign_agreement = float(np.mean(np.sign(pred) == np.sign(actual)))
            else:
                mae = rmse = sign_agreement = float("nan")
            result.append(
                {
                    "target": spec.target,
                    "group": group,
                    "n": int(pred.size),
                    "mae": mae,
                    "rmse": rmse,
                    "pearson": _pearson(pred, actual),
                    "spearman": _spearman(pred, actual),
                    "sign_agreement": sign_agreement,
                }
            )
    return result


def _build_manifest(
    *,
    paths: Mapping[str, Path],
    args: argparse.Namespace,
    device: torch.device,
    qparams_by_target: Mapping[single_layer.TargetSpec, Mapping[str, Any]],
    results: Sequence[TargetFactorResult],
    shared_fp32_validation: Mapping[str, Any],
) -> dict[str, Any]:
    targets: list[dict[str, Any]] = []
    for result in results:
        spec = result.spec
        qparams = qparams_by_target[spec]
        targets.append(
            {
                "target": spec.target,
                "module_path": spec.module_path,
                "layer_idx": int(spec.layer_idx),
                "module_name": spec.module_name,
                "hook_position": HOOK_POSITION,
                "representation": "target_linear_module_output",
                "isolated_w4a4": True,
                "full_w4a4_forward_used": False,
                "quant_state": list(result.quant_state),
                "qparams_identity": qparams_identity_manifest(qparams, source_hint=spec.target),
                "single_layer_reference_validation": dict(result.single_layer_reference_validation),
                "checkpoint_clean_reloads": int(result.checkpoint_clean_reloads),
            }
        )
    return {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "analysis_only": True,
        "dataset": single_layer.DEFAULT_DATASET,
        "checkpoint": str(paths["checkpoint"].resolve(strict=False)),
        "checkpoint_name_required": single_layer.DEFAULT_CHECKPOINT_NAME,
        "checkpoint_sha256": single_layer._file_sha256(paths["checkpoint"]),
        "analysis_split": "eval",
        "eval_csv": file_identity(paths["eval_csv"]),
        "test_touched": False,
        "train_touched": False,
        "training_performed": False,
        "recalibration_performed": False,
        "full_w4a4_forward_used": False,
        "full_w4a4_intermediate_activation_used_as_injection_input": False,
        "same_waveform": True,
        "same_center_crop": True,
        "same_padding_mask": True,
        "same_pooling": "mask_aware_mean",
        "same_classifier": True,
        "fixed_competitor": True,
        "competitor_definition": "j_star = argmax_{j != y} z_FP32[j]",
        "main_cohort": "FP32-correct eval samples",
        "groups": list(GROUPS),
        "tail_definition": "tail = medium + few",
        "medium_few_interpretation": "exploratory_only_on_local_data",
        "quantization": {
            "scope": "isolated_single_ffn_module_per_target",
            "mode": single_layer.QUANT_MODE.upper(),
            "activation": {
                "bits": 4,
                "granularity": "per_tensor",
                "symmetric": True,
                "qmin": single_layer.EXPECTED_QRANGE[0],
                "qmax": single_layer.EXPECTED_QRANGE[1],
                "zero_point": 0,
            },
            "weight": {
                "bits": 4,
                "granularity": "per_output_channel",
                "symmetric": True,
                "qmin": single_layer.EXPECTED_QRANGE[0],
                "qmax": single_layer.EXPECTED_QRANGE[1],
                "zero_point": 0,
            },
            "non_target_layers_quantized": False,
            "target_predecessor_layers_quantized": False,
        },
        "representation_definition": {
            "H": "FP32 target Linear module output",
            "H_q": "isolated W4A4 target Linear module output",
            "delta_H": "H_q - H",
            "padding_policy": "flatten valid tokens x channels only; padding tokens excluded before factor statistics",
            "cosine": "signed cosine; harmful_alignment = -direction_cosine_A",
        },
        "numerical_tolerance": float(args.numerical_tolerance),
        "qparams_root": str(paths["qparams_root"].resolve(strict=False)),
        "calibration_csv": file_identity(paths["calibration_csv"]),
        "calibration_manifest": file_identity(paths["calibration_manifest"]),
        "targets": targets,
        "validations": {
            **dict(shared_fp32_validation),
            "fp32_logits_identical_to_single_layer_causal_injection_reference": True,
            "injected_logits_identical_to_single_layer_causal_injection_reference": True,
            "only_target_module_w4a4_per_experiment": True,
            "padding_tokens_excluded_from_H_G_deltaH_statistics": True,
            "factor_product_algebra_checked": True,
            "test_touched": False,
        },
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "outputs": {
            "sample_csv": str((paths["output_dir"] / "three_factor_sample.csv").resolve(strict=False)),
            "correlation_csv": str((paths["output_dir"] / "three_factor_correlation.csv").resolve(strict=False)),
            "taylor_csv": str((paths["output_dir"] / "three_factor_taylor.csv").resolve(strict=False)),
            "manifest_json": str((paths["output_dir"] / "manifest.json").resolve(strict=False)),
        },
        "disclaimer": "Correlation and Taylor diagnostics are descriptive; this script does not claim causality from correlation.",
    }


def _print_report(
    *,
    paths: Mapping[str, Path],
    sample_rows: Sequence[Mapping[str, Any]],
    taylor_rows: Sequence[Mapping[str, Any]],
) -> None:
    print("[Three-Factor Decomposition]")
    print(f"checkpoint: {paths['checkpoint']}")
    print(f"output_dir: {paths['output_dir']}")
    print("test_touched: False")
    print(f"sample_rows: {len(sample_rows)}")
    for spec in single_layer.TARGET_SPECS:
        overall = [
            row for row in taylor_rows
            if row.get("target") == spec.target and row.get("group") == "overall"
        ]
        taylor = overall[0] if overall else {}
        rows = [row for row in sample_rows if row.get("target") == spec.target]
        flips = sum(int(row.get("classification_flip", 0)) for row in rows)
        print(
            f"{spec.target}: n={len(rows)} flips={flips} "
            f"taylor_pearson={_csv_value(taylor.get('pearson', ''))} "
            f"sign_agreement={_csv_value(taylor.get('sign_agreement', ''))}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _resolve_paths(args)
    device = single_layer._device_from_args(args.device)
    metadata = single_layer._load_eval_metadata(paths["eval_csv"])
    qparams_by_target = single_layer._load_qparams_by_target(paths["qparams_root"])
    loader = single_layer.build_loader(
        single_layer.dataset_paths(single_layer.DEFAULT_DATASET, None),
        split="eval",
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    results: list[TargetFactorResult] = []
    for spec in single_layer.TARGET_SPECS:
        result = _run_target_factor(
            spec=spec,
            checkpoint_path=paths["checkpoint"],
            qparams=qparams_by_target[spec],
            loader=loader,
            device=device,
            metadata=metadata,
            numerical_tolerance=float(args.numerical_tolerance),
        )
        results.append(result)

    shared_fp32_validation = _assert_shared_fp32_baseline(results)
    sample_rows = [row for result in results for row in result.sample_rows]
    correlation_rows = _build_correlation_rows(sample_rows)
    taylor_rows = _build_taylor_rows(sample_rows)
    manifest = _build_manifest(
        paths=paths,
        args=args,
        device=device,
        qparams_by_target=qparams_by_target,
        results=results,
        shared_fp32_validation=shared_fp32_validation,
    )

    output_dir = paths["output_dir"]
    _write_csv(output_dir / "three_factor_sample.csv", sample_rows, SAMPLE_COLUMNS)
    _write_csv(output_dir / "three_factor_correlation.csv", correlation_rows, CORRELATION_COLUMNS)
    _write_csv(output_dir / "three_factor_taylor.csv", taylor_rows, TAYLOR_COLUMNS)
    _write_json(output_dir / "manifest.json", manifest)
    _print_report(paths=paths, sample_rows=sample_rows, taylor_rows=taylor_rows)
    return {
        "output_dir": str(output_dir),
        "sample_rows": int(len(sample_rows)),
        "correlation_rows": int(len(correlation_rows)),
        "taylor_rows": int(len(taylor_rows)),
        "shared_fp32_validation": dict(shared_fp32_validation),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--dataset", default=single_layer.DEFAULT_DATASET, help="Only local_data is accepted.")
    parser.add_argument("--checkpoint", default=None, help=f"Must be local_data/{single_layer.DEFAULT_CHECKPOINT_NAME}.")
    parser.add_argument("--qparams-root", default=None, help="Directory containing frozen W4A4 qparams.")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV and manifest outputs.")
    parser.add_argument("--device", default=None, help="Override config.DEVICE, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=single_layer._default_batch_size())
    parser.add_argument("--num-workers", type=int, default=single_layer._default_num_workers())
    parser.add_argument("--seed", type=int, default=int(single_layer._cfg_get(single_layer.project_config, "SPLIT_SEED", 42)))
    parser.add_argument("--numerical-tolerance", type=float, default=DEFAULT_NUMERICAL_TOLERANCE)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative")
    if not math.isfinite(float(args.numerical_tolerance)) or float(args.numerical_tolerance) <= 0.0:
        raise ValueError("--numerical-tolerance must be a finite positive value")
    run(args)


if __name__ == "__main__":
    main()
