"""Frozen local_data A4 calibration pilot; default execution stops after Phase 1.

Phase 1: first 16 FP32-correct calibration samples, L4 target, three scales.
Phase 2: explicitly requested full A-D pilot, gated on a successful Phase 1.
Phase 3: explicitly requested CBDB, using Phase 2's saved per-sample risks.
No activation arrays, gradients, checkpoints, or datasets are written.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as project_config
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

import src.export.single_layer_causal_injection as single_layer
import src.export.three_factor_decomposition as three_factor
from src.baseline.export_eval import pooled_logits_from_features, prepare_encoder_input_for_export
from src.data.audio_input import AudioInputDataset, audio_input_collate_fn
from src.export.activation_granularity_counterfactual import _macro_f1
from src.metrics.common_metrics import activation_clipping_rate, embedding_distortion
from src.metrics.ffn_quant_mode_metrics import compute_near_zero_rate
from src.quant.calibration import calculate_activation_qparams_from_observer
from src.quant.ffn_quant_injector import (
    make_single_ffn_quant_controller,
    replace_ffn_module_with_calibrated_quant,
    set_quant_module_mode,
)
from src.quant.observer import ActivationObserver, load_project_observer_configs
from src.quant.quant_linear import QuantLinear
from src.quant.ste import quantize_dequantize_ste
from src.train.train_beats_classifier_head import audio_config_from_project

PROTOCOL = "decision_aware_calibration_pilot.a4_input.v1"
FULL_PERCENTILES = (99.0, 99.3, 99.5, 99.7, 99.8, 99.9, 99.95, 99.99, 100.0)
PHASE1_PERCENTILES = (99.5, 99.9, 100.0)
ALPHAS = (0.0, 0.10, 0.25, 0.50)
GROUPS = ("overall", "many", "medium", "few")
TIE_RULE = "primary np.isclose(rtol=1e-9,atol=1e-12); minimum MSE; larger percentile"
TOLERANCE = three_factor.DEFAULT_NUMERICAL_TOLERANCE
MANIFEST = "calibration_experiment_manifest.json"
DEPENDENCIES = (
    "config.py", "src/quant/ste.py", "src/quant/quant_linear.py",
    "src/quant/observer.py", "src/quant/calibration.py", "src/quant/ffn_quant_injector.py",
    "src/data/audio_input.py", "src/baseline/export_eval.py",
    "src/train/train_beats_classifier_head.py", "src/utils/layout_utils.py",
    "src/export/single_layer_causal_injection.py", "src/export/three_factor_decomposition.py",
    "src/metrics/common_metrics.py", "src/metrics/ffn_quant_mode_metrics.py",
    "src/export/activation_granularity_counterfactual.py",
    "src/experiments/run_decision_aware_calibration_pilot.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    # Qparams are buffers on wrappers; verify original FP32 parameters by name.
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        require(not isinstance(value, Tensor) or torch.isfinite(value).all().item(),
                f"nonfinite model state: {name}")
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing an empty result CSV: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class Criterion:
    criterion: str
    alpha: float | None = None


def criteria_for_phase(phase: int) -> list[Criterion]:
    if phase == 1:
        return [Criterion("MSE"), Criterion("DP"), Criterion("DB", 0.0)]
    if phase == 2:
        return [Criterion("Percentile"), Criterion("MSE"), Criterion("DP"),
                *(Criterion("DB", alpha) for alpha in ALPHAS)]
    return [Criterion("CBDB", alpha) for alpha in ALPHAS]


def target_specs(manifest_path: Path) -> tuple[list[single_layer.TargetSpec], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = []
    for layer in (4, 9):
        matches = [item for item in manifest["targets"] if int(item["layer_idx"]) == layer]
        require(len(matches) == 1, f"manifest must identify exactly one L{layer} target")
        item = matches[0]
        spec = single_layer.TargetSpec(int(item["layer_idx"]), str(item["module_name"]))
        require(item["target"] == spec.target and item["module_path"] == spec.module_path,
                f"inconsistent manifest target: {item}")
        specs.append(spec)
    report = {
        "manifest_source": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "targets": [{"target": spec.target, "layer_idx": spec.layer_idx,
                     "module_name": spec.module_name, "module_path": spec.module_path,
                     "hook_position": "forward_pre_hook / " + spec.activation_hook_position,
                     "quantized_tensor": spec.module_path + ".input_activation",
                     "gradient_tensor": spec.module_path + ".input_activation",
                     "quantized_tensor_equals_gradient_tensor": True} for spec in specs],
        "gradient_definition": "gradient of FP32 fixed competitor margin w.r.t. actual Linear INPUT",
        "fixed_margin_definition": "z_y - z_j_star; j_star=argmax(FP32 logits excluding y)",
        "quantizer": "QuantLinear.activation_quant -> src.quant.ste.quantize_dequantize_ste",
        "scale_formula": "clip_threshold / 7, guarded by existing quantizer EPS",
        "candidate_generator": "existing two-pass absolute histogram; P100 uses exact observed abs max",
        "mask_handling": "existing BTC canonicalization and three_factor._flatten_valid; True means padding",
        "old_output_gradient": "NOT used for calibration; first-order algebra helper reused on INPUT tensors",
        "weight_quantization": "disabled (A4-only)",
        "model_loader": "single_layer._load_clean_fp32_model",
        "dataset_loader": "AudioInputDataset(mode=calibration/eval), unchanged CSV and order",
        "pooling": "existing pooled_logits_from_features, mask-aware mean",
    }
    return specs, report


def make_loader(path: Path, args: argparse.Namespace, indices: Sequence[int] | None = None) -> DataLoader:
    dataset = AudioInputDataset(
        path, mode="calibration" if path.name == "calibration.csv" else "eval",
        dataset_root=ROOT / "local_data", project_root=ROOT,
        audio_config=audio_config_from_project(), global_seed=project_config.SPLIT_SEED,
        epoch=0, return_dict=True, require_label_id=True, include_audio_stats=False,
    )
    if indices is not None:
        dataset = Subset(dataset, list(indices))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=args.device.type == "cuda",
                      collate_fn=partial(audio_input_collate_fn, include_audio_stats=False))


def forward(model: nn.Module, batch: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
    moved = single_layer._move_batch_to_device(batch, device)
    encoder_input, encoder_mask, token_mask = prepare_encoder_input_for_export(
        model, moved["waveform"], moved["padding_mask"])
    features, _ = model.encoder(encoder_input, padding_mask=encoder_mask)
    return pooled_logits_from_features(model, features.float(), token_mask), token_mask


def baseline_rows(logits: Tensor, mask: Tensor, batch: Mapping[str, Any],
                  metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    logits = logits.detach().cpu().float()
    labels = batch["label_id"].detach().cpu().long()
    j_star = single_layer._competitor_indices(logits, labels)
    margins = single_layer._fixed_margin(logits, labels, j_star)
    dynamic = single_layer._dynamic_margin(logits, labels)
    result = []
    for i, sample_id in enumerate(single_layer._batch_sample_ids(batch)):
        row = metadata[sample_id]
        require(int(row["label_id"]) == int(labels[i]), f"label mismatch for {sample_id}")
        bucket = str(row["bucket"])
        require(bucket in GROUPS[1:], f"invalid bucket {bucket}")
        result.append({
            "sample_id": sample_id, "label_id": int(labels[i]), "bucket": bucket,
            "fp32_pred": int(logits[i].argmax()), "fp32_correct": int(logits[i].argmax() == labels[i]),
            "j_star": int(j_star[i]), "m_fp32": float(margins[i]),
            "fp32_dynamic_margin": float(dynamic[i]), "fp32_logits": logits[i].tolist(),
            "waveform_sha256": tensor_hash(batch["waveform"][i]),
            "waveform_padding_mask_sha256": tensor_hash(batch["padding_mask"][i]),
            "token_padding_mask_sha256": tensor_hash(mask[i]),
        })
    return result


def check_reference(actual: Mapping[str, Any], expected: Mapping[str, Any], checks: dict[str, Any]) -> None:
    require(actual == expected, f"FP32 reference/waveform/mask changed: {actual['sample_id']}")
    checks["fp32_reference_comparisons"] += 1


def collect_reference(model: nn.Module, loader: DataLoader, metadata: Mapping[str, Any],
                      args: argparse.Namespace, label: str) -> list[dict[str, Any]]:
    result = []
    with torch.no_grad():
        for batch in loader:
            logits, mask = forward(model, batch, args.device)
            result.extend(baseline_rows(logits, mask, batch, metadata))
            if len(result) == 1 or len(result) % 16 < args.batch_size:
                log(f"{label}: {len(result)}/{len(loader.dataset)}")
    return result


def capture(model: nn.Module, spec: single_layer.TargetSpec, batch: Mapping[str, Any],
            args: argparse.Namespace, *, gradient: bool) -> dict[str, Any]:
    """Attach autograd to the SAME raw tensor consumed by activation quantization."""
    values: dict[str, Tensor] = {}
    module = single_layer.get_ffn_module(model, spec.layer_idx, spec.module_name)

    def pre_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        h = inputs[0].detach()
        if gradient:
            h = h.requires_grad_(True)
        values["raw"] = h
        return (h, *inputs[1:])

    handle = module.register_forward_pre_hook(pre_hook)
    try:
        with torch.set_grad_enabled(gradient):
            logits, mask = forward(model, batch, args.device)
            h, layout = three_factor._canonicalize(values["raw"], mask, context=spec.target + ".input")
            labels = batch["label_id"].to(args.device).long()
            competitors = single_layer._competitor_indices(logits.detach(), labels)
            margins = single_layer._fixed_margin(logits, labels, competitors)
            if gradient:
                require(bool((logits.argmax(1).eq(labels) & (margins > 0)).all()),
                        "calibration population must be FP32-correct with strictly positive margins")
                (raw_grad,) = torch.autograd.grad(margins.sum(), values["raw"])
                g = three_factor._canonicalize_like(raw_grad, mask, layout, context=spec.target + ".input_gradient")
            else:
                g = None
    finally:
        handle.remove()
    single_layer._assert_no_param_grads(model, context="input-gradient calibration")
    return {"h": h.detach(), "g": None if g is None else g.detach(), "mask": mask.detach(),
            "logits": logits.detach(), "margins": margins.detach(), "layout": layout}


def numerical_metrics(h: Tensor, q: Tensor, g: Tensor, mask: Tensor,
                      spec: single_layer.TargetSpec, index: int, scale: float) -> dict[str, float]:
    hv, qv, gv, valid_n, padding_n, channels = three_factor._flatten_valid(
        h_fp32=h, h_q=q, gradient=g, token_padding_mask=mask, spec=spec, sample_index=index)
    delta = qv - hv
    first_order = float(torch.dot(gv, delta))
    return {
        "activation_mse": float(delta.square().mean()),
        "relative_distortion": float(embedding_distortion(hv, qv, dim=None)),
        "clipping_rate": float(activation_clipping_rate(hv, scale=scale, qmin=-7, qmax=7, reduction="mean")),
        "near_zero_rate": float(compute_near_zero_rate(qv.unsqueeze(0))),
        "first_order_margin_change": first_order, "decision_projection": max(0.0, -first_order),
        "valid_scalar_count": valid_n * channels, "padding_token_count": padding_n,
    }


def sanity_synthetic(spec: single_layer.TargetSpec) -> dict[str, Any]:
    """Adversarial padding + actual QuantLinear path; no toy model experiment."""
    h = torch.tensor([[[0.31, -0.7], [500.0, -900.0], [1.2, -0.02]]])
    g = torch.tensor([[[0.4, -0.3], [1e9, -1e9], [-0.8, 0.2]]])
    mask = torch.tensor([[False, True, False]])
    scale = 0.1
    q = quantize_dequantize_ste(h, scale, qmin=-7, qmax=7)
    base = numerical_metrics(h, q, g, mask, spec, 0, scale)
    poisoned_h, poisoned_q, poisoned_g = h.clone(), q.clone(), g.clone()
    poisoned_h[:, 1], poisoned_q[:, 1], poisoned_g[:, 1] = -2e9, 3e9, 4e9
    poisoned = numerical_metrics(poisoned_h, poisoned_q, poisoned_g, mask, spec, 0, scale)
    require(base == poisoned, "padding pollution changed MSE/inner product/distortion/rates")
    # Observer must exclude padding as well, in both passes.
    observer = ActivationObserver(load_project_observer_configs()["activation"], enabled=True)
    for phase in ("range", "histogram"):
        if phase == "range":
            observer.start_range_pass()
        else:
            observer.start_histogram_pass()
        observer.update(h, valid_mask=~mask, num_samples=1)
    require(float(observer.abs_max_value.max()) < 2, "percentile observer included padding")
    module = QuantLinear(nn.Linear(2, 2), quant_config=project_config.QUANT, store_oer_activation=True)
    set_quant_module_mode(module, "a4")
    module.set_fixed_activation_qparams(torch.tensor(scale), torch.tensor(0.), qmin=-7, qmax=7)
    module.enable_fake_quant()
    module(h, padding_mask=mask)
    require(torch.equal(q, module.last_quant_activation_for_oer), "STE differs from actual QuantLinear input")
    return {"passed": True, "adversarial_padding_exclusion": True,
            "observer_padding_exclusion": True, "quantlinear_matches_shared_operator": True}


def select_candidate(candidates: Sequence[Mapping[str, Any]], criterion: Criterion) -> dict[str, Any]:
    def objective(item: Mapping[str, Any]) -> float:
        if criterion.criterion == "Percentile":
            return float(item["calibration_decision_projection"])
        if criterion.criterion == "MSE":
            return float(item["calibration_mse"])
        if criterion.criterion == "DP":
            return float(item["calibration_decision_projection"])
        key = "db" if criterion.criterion == "DB" else "cbdb"
        return float(item[key][str(criterion.alpha)])

    if criterion.criterion == "Percentile":
        tied = [item for item in candidates if float(item["candidate_percentile"]) == 99.9]
        require(len(tied) == 1, "fixed P99.9 baseline missing")
    else:
        values = [objective(item) for item in candidates]
        require(all(math.isfinite(value) for value in values), "nonfinite calibration objective")
        best = min(values)
        tied = [item for item in candidates if np.isclose(objective(item), best, rtol=1e-9, atol=1e-12)]
    # Only PRIMARY ties use isclose. Secondary MSE uses its actual numeric order.
    selected = min(tied, key=lambda item: (float(item["calibration_mse"]), -float(item["candidate_percentile"])))
    return {"target": selected["target"], **asdict(criterion),
            "selected_percentile": selected["candidate_percentile"],
            "selected_clip_threshold": selected["clip_threshold"], "selected_scale": selected["scale"],
            "calibration_objective": None if criterion.criterion == "Percentile" else objective(selected),
            "objective_tie_count": len(tied), "tie_break_rule": TIE_RULE,
            "selected_by_tie_break": len(tied) > 1}


def summarize_candidates(rows: Sequence[Mapping[str, Any]], percentiles: Sequence[float],
                         candidates: Sequence[Mapping[str, Any]], alphas: Sequence[float]) -> list[dict[str, Any]]:
    result = []
    for percentile, candidate in zip(percentiles, candidates):
        selected = [row for row in rows if float(row["candidate_percentile"]) == percentile]
        require(bool(selected), "empty calibration candidate")
        db, cbdb = {}, {}
        for alpha in alphas:
            by_class: dict[int, list[float]] = defaultdict(list)
            risks = []
            for row in selected:
                risk = max(0.0, float(row["decision_projection"]) - alpha * float(row["m_fp32"]))
                by_class[int(row["label_id"])].append(risk)
                risks.append(risk)
            db[str(alpha)] = math.fsum(risks) / len(risks)
            cbdb[str(alpha)] = math.fsum(math.fsum(v) / len(v) for v in by_class.values()) / len(by_class)
        dp = math.fsum(float(row["decision_projection"]) for row in selected) / len(selected)
        require(abs(db["0.0"] - dp) < 1e-8, "DB(alpha=0) differs from DP")
        result.append({**candidate, "calibration_mse": math.fsum(float(row["activation_mse"]) for row in selected) / len(selected),
                       "calibration_decision_projection": dp, "db": db, "cbdb": cbdb})
    return result


def calibrate(model: nn.Module, spec: single_layer.TargetSpec, loader: DataLoader,
              references: Mapping[str, Any], metadata: Mapping[str, Any], percentiles: Sequence[float],
              args: argparse.Namespace, checks: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observer = ActivationObserver(load_project_observer_configs()["activation"], enabled=True)
    before = state_hash(model)
    for phase in ("range", "histogram"):
        log(f"{spec.target}: {phase} observer on {len(loader.dataset)} matched calibration samples")
        if phase == "range":
            observer.start_range_pass()
        else:
            observer.start_histogram_pass()
        for batch in loader:
            data = capture(model, spec, batch, args, gradient=False)
            for row in baseline_rows(data["logits"], data["mask"], batch, metadata):
                check_reference(row, references[row["sample_id"]], checks)
            observer.update(data["h"], valid_mask=~data["mask"], num_samples=len(batch["sample_id"]))
            del data
    candidates = []
    for percentile in percentiles:
        if percentile == 100.0:
            scale = torch.clamp(observer.abs_max_value / 7, min=observer.config.eps).reshape(())
        else:
            scale, _ = calculate_activation_qparams_from_observer(
                observer, symmetric=True, retained_percentile=percentile,
                lower_percentile=None, upper_percentile=None, qmin=-7, qmax=7, eps=observer.config.eps)
        candidates.append({"target": spec.target, "candidate_percentile": percentile,
                           "clip_threshold": float(scale) * 7, "scale": float(scale)})
    del observer
    rows = []
    for batch in loader:
        data = capture(model, spec, batch, args, gradient=True)
        for row in baseline_rows(data["logits"], data["mask"], batch, metadata):
            check_reference(row, references[row["sample_id"]], checks)
        h, g, mask = data["h"], data["g"], data["mask"]
        require(h.shape == g.shape, "gradient does not match exact quantized input tensor")
        checks["input_gradient_batches"] += 1
        for candidate in candidates:
            q = quantize_dequantize_ste(h, candidate["scale"], qmin=-7, qmax=7)
            # Independent criterion calls share the same operator, and must be bitwise identical.
            for _criterion in criteria_for_phase(args.phase):
                duplicate = quantize_dequantize_ste(h, candidate["scale"], qmin=-7, qmax=7)
                require(torch.equal(q, duplicate), "criterion changed quantized candidate activation")
                checks["identical_candidate_quantization_comparisons"] += 1
                del duplicate
            for i, sample_id in enumerate(single_layer._batch_sample_ids(batch)):
                metrics = numerical_metrics(h, q, g, mask, spec, i, candidate["scale"])
                # Existing attribution algebra, with INPUT tensors and INPUT gradients.
                factor = three_factor._factor_metrics(
                    h_fp32=h, h_q=q, gradient=g, token_padding_mask=mask, spec=spec,
                    sample_index=i, numerical_tolerance=TOLERANCE)
                difference = abs(-metrics["first_order_margin_change"] - factor["predicted_margin_drop"])
                require(difference < TOLERANCE, "existing attribution first-order definition mismatch")
                checks["first_order_max_abs_diff"] = max(checks["first_order_max_abs_diff"], difference)
                checks["first_order_comparisons"] += 1
                rows.append({**candidate, "sample_id": sample_id,
                             "label_id": references[sample_id]["label_id"],
                             "bucket": references[sample_id]["bucket"],
                             "m_fp32": references[sample_id]["m_fp32"], **metrics})
            del q
        del data, h, g, mask
        log(f"{spec.target}: candidate objectives {len(rows) // len(candidates)}/{len(loader.dataset)}")
    require(state_hash(model) == before, "candidate calibration contaminated model state")
    checks["unchanged_model_state_checks"] += 1
    return summarize_candidates(rows, percentiles, candidates, (0.0,) if args.phase == 1 else ALPHAS), rows


def evaluate_selection(selection: Mapping[str, Any], spec: single_layer.TargetSpec, loader: DataLoader,
                       metadata: Mapping[str, Any], args: argparse.Namespace, checks: dict[str, Any],
                       references: dict[str, Any], clean_hash: str) -> list[dict[str, Any]]:
    label = f"{spec.target} {selection['criterion']} alpha={selection['alpha']}"
    log(f"{label}: independent clean checkpoint reload; full eval")
    model, checkpoint = single_layer._load_clean_fp32_model(args.checkpoint, device=args.device, context=label)
    del checkpoint
    require(state_hash(model) == clean_hash, "independent clean reload differs from reference state")
    checks["clean_checkpoint_reloads"] += 1
    # Complete clean forward BEFORE injecting this criterion; compare every eval sample.
    clean_rows = collect_reference(model, loader, metadata, args, label + " FP32")
    if not references:
        references.update({row["sample_id"]: row for row in clean_rows})
    for row in clean_rows:
        check_reference(row, references[row["sample_id"]], checks)
    scale = float(selection["selected_scale"])
    qparams = {"activation": {"scale": torch.tensor(scale), "zero_point": torch.tensor(0.),
                              "qmin": -7, "qmax": 7}}
    quant_module = replace_ffn_module_with_calibrated_quant(
        model, spec.layer_idx, spec.module_name, project_config.QUANT, qparams, quant_mode="a4")
    controller = make_single_ffn_quant_controller(
        model, quant_module, layer_idx=spec.layer_idx, module_name=spec.module_name,
        quant_config=project_config.QUANT, quant_mode="a4")
    controller.qparams_frozen = True
    controller.prepare_quant_forward()
    require(not quant_module.weight_quant_enabled, "A4-only unexpectedly enabled weight quantization")
    parameter_versions = {name: value._version for name, value in model.named_parameters()}
    result = []
    try:
        with torch.no_grad():
            for batch in loader:
                logits, mask = forward(model, batch, args.device)
                h, q = quant_module.get_oer_activation_tensors()
                require(h is not None and q is not None, "actual quantizer did not capture input activations")
                direct = quantize_dequantize_ste(h, scale, qmin=-7, qmax=7)
                require(torch.equal(direct, q), "injected model quantizer differs from calibrated candidate")
                checks["actual_quantlinear_comparisons"] += 1
                del direct
                labels = batch["label_id"].long()
                logits = logits.detach().cpu().float()
                ids = single_layer._batch_sample_ids(batch)
                competitors = torch.tensor([references[sid]["j_star"] for sid in ids])
                margins = single_layer._fixed_margin(logits, labels, competitors)
                dynamic = single_layer._dynamic_margin(logits, labels)
                zero_gradient = torch.zeros_like(h)
                for i, sid in enumerate(ids):
                    ref = references[sid]
                    require(ref["waveform_sha256"] == tensor_hash(batch["waveform"][i]), "eval waveform changed")
                    require(ref["waveform_padding_mask_sha256"] == tensor_hash(batch["padding_mask"][i]), "waveform mask changed")
                    require(ref["token_padding_mask_sha256"] == tensor_hash(mask[i]), "token mask changed")
                    pred = int(logits[i].argmax())
                    metrics = numerical_metrics(h, q, zero_gradient, mask, spec, i, scale)
                    result.append({
                        "sample_id": sid, "label_id": ref["label_id"], "bucket": ref["bucket"],
                        "target": spec.target, "criterion": selection["criterion"], "alpha": selection["alpha"],
                        "selected_scale": scale, "fp32_pred": ref["fp32_pred"], "int4_pred": pred,
                        "fp32_correct": ref["fp32_correct"], "int4_correct": int(pred == ref["label_id"]),
                        "classification_flip": int(bool(ref["fp32_correct"]) and pred != ref["label_id"]),
                        "j_star": ref["j_star"], "m_fp32_fixed": ref["m_fp32"],
                        "m_int4_fixed": float(margins[i]), "fixed_margin_drop": ref["m_fp32"] - float(margins[i]),
                        "fp32_dynamic_margin": ref["fp32_dynamic_margin"], "int4_dynamic_margin": float(dynamic[i]),
                        "dynamic_margin_drop": ref["fp32_dynamic_margin"] - float(dynamic[i]),
                        **{key: metrics[key] for key in ("activation_mse", "relative_distortion", "clipping_rate", "near_zero_rate")},
                    })
                quant_module.clear_oer_activation_tensors()
                quant_module.clear_oer_projection_tensors()
                del h, q, zero_gradient
                if len(result) == 1 or len(result) % 16 < args.batch_size:
                    log(f"{label} A4: {len(result)}/{len(loader.dataset)}")
    finally:
        controller.remove_padding_mask_hooks()
    require(parameter_versions == {name: value._version for name, value in model.named_parameters()},
            "evaluation modified FP32 parameter storage")
    # Remove only the wrapper on this private, freshly loaded model; verify exact checkpoint state.
    setattr(model.encoder.layers[spec.layer_idx], spec.module_name, quant_module.linear)
    require(state_hash(model) == clean_hash, "evaluation changed checkpoint state")
    checks["unchanged_model_state_checks"] += 1
    require([row["sample_id"] for row in result] == list(references), "full eval sample order mismatch")
    del model, quant_module, controller
    gc.collect()
    return result


def comparison_rows(selection: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], classes: int) -> list[dict[str, Any]]:
    result = []
    for group in GROUPS:
        members = [row for row in rows if group == "overall" or row["bucket"] == group]
        n = len(members)
        correct = sum(int(row["fp32_correct"]) for row in members)
        flips = sum(int(row["classification_flip"]) for row in members)
        def mean(key: str) -> float | None:
            return math.fsum(float(row[key]) for row in members) / n if n else None
        def median(key: str) -> float | None:
            return float(np.median([float(row[key]) for row in members])) if n else None
        result.append({**selection, "group": group, "n": n,
                       "eval_accuracy": mean("int4_correct"),
                       "eval_macro_f1": _macro_f1(torch.tensor([row["label_id"] for row in members]),
                                                 torch.tensor([row["int4_pred"] for row in members]),
                                                 num_classes=classes) if n else None,
                       "n_fp32_correct": correct, "flip_count": flips,
                       "n_fp32_correct_to_int4_wrong": flips,
                       "flip_rate": flips / correct if correct else None,
                       "mean_fixed_margin_drop": mean("fixed_margin_drop"),
                       "median_fixed_margin_drop": median("fixed_margin_drop"),
                       "mean_dynamic_margin_drop": mean("dynamic_margin_drop"),
                       "median_dynamic_margin_drop": median("dynamic_margin_drop"),
                       **{key: mean(key) for key in ("m_fp32_fixed", "m_int4_fixed", "fp32_dynamic_margin",
                                                   "int4_dynamic_margin", "activation_mse", "relative_distortion",
                                                   "clipping_rate", "near_zero_rate")}})
    return result


def identities(args: argparse.Namespace) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in (
        args.checkpoint, args.calibration_csv, args.eval_csv,
        ROOT / "local_data/splits/train.csv", ROOT / "local_data/splits/label_mapping.json",
        args.target_manifest, *(ROOT / path for path in DEPENDENCIES))}


def prerequisite(path: Path, expected_phase: int, current_identity: Mapping[str, str],
                 args: argparse.Namespace) -> dict[str, Any]:
    require((path / MANIFEST).is_file(), f"Phase {expected_phase} prerequisite missing: {path}")
    saved = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    require(saved["phase"] == expected_phase and saved["status"] == "passed", "prerequisite phase has not passed")
    require(saved["protocol"] == PROTOCOL and saved["input_sha256"] == current_identity,
            "prerequisite inputs/code differ; refusing silently changed experiment")
    require(saved["batch_size"] == args.batch_size and saved["device"] == str(args.device),
            "prerequisite batch/device changed; exact FP32 reference comparison would be invalid")
    require(saved["sanity_checks"]["all_passed"], "prerequisite sanity checks failed")
    for filename, digest in saved["output_sha256"].items():
        require(sha256(path / filename) == digest, f"prerequisite output was modified: {filename}")
    return saved


def load_saved_references(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    integer_keys = ("label_id", "fp32_pred", "fp32_correct", "j_star")
    for row in rows:
        for key in integer_keys:
            row[key] = int(row[key])
        for key in ("m_fp32", "fp32_dynamic_margin"):
            row[key] = float(row[key])
        row["fp32_logits"] = json.loads(row["fp32_logits"])
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    specs, inspection = target_specs(args.target_manifest)
    input_ids = identities(args)
    if args.phase in (2, 3):
        prerequisite(args.phase1_dir if args.phase == 2 else args.phase2_dir,
                     1 if args.phase == 2 else 2, input_ids, args)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            f"output directory must be new/empty; refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "phase0_inspection.json", inspection)
    log("Phase 0: manifest targets / actual INPUT gradient / A4-only / padding protocol inspected")
    if args.phase == 0:
        return inspection
    checks: dict[str, Any] = {
        "all_passed": False, "fp32_reference_comparisons": 0, "input_gradient_batches": 0,
        "identical_candidate_quantization_comparisons": 0, "actual_quantlinear_comparisons": 0,
        "first_order_comparisons": 0, "first_order_max_abs_diff": 0.,
        "unchanged_model_state_checks": 0, "clean_checkpoint_reloads": 0,
    }
    manifest: dict[str, Any] = {"protocol": PROTOCOL, "phase": args.phase, "status": "running"}
    try:
        torch.manual_seed(project_config.SPLIT_SEED)
        np.random.seed(project_config.SPLIT_SEED)
        checks["synthetic_checks"] = sanity_synthetic(specs[0])
        calibration_meta_rows = read_csv(args.calibration_csv)
        eval_meta_rows = read_csv(args.eval_csv)
        for rows in (calibration_meta_rows, eval_meta_rows):
            require(len({row["sample_id"] for row in rows}) == len(rows), "duplicate sample IDs")
        require(not ({row["sample_id"] for row in calibration_meta_rows} &
                     {row["sample_id"] for row in eval_meta_rows}), "calibration/eval overlap")
        calibration_meta = {row["sample_id"]: row for row in calibration_meta_rows}
        eval_meta = {row["sample_id"]: row for row in eval_meta_rows}
        model, checkpoint = single_layer._load_clean_fp32_model(args.checkpoint, device=args.device, context="pilot census")
        classes = len(checkpoint["label_mapping"])
        del checkpoint
        clean_hash = state_hash(model)
        for spec in specs:
            require(isinstance(single_layer.get_ffn_module(model, spec.layer_idx, spec.module_name), nn.Linear),
                    f"manifest target is not a clean Linear: {spec.target}")
        if args.phase == 3:
            phase2_manifest = json.loads((args.phase2_dir / MANIFEST).read_text(encoding="utf-8"))
            calibration_refs = load_saved_references(args.phase2_dir / "fp32_calibration_samples.csv")
            selected_ids = phase2_manifest["calibration_sample_ids"]
        else:
            calibration_refs = collect_reference(model, make_loader(args.calibration_csv, args),
                                                   calibration_meta, args, "calibration FP32 census")
            eligible = [row for row in calibration_refs if row["fp32_correct"]]
            require(all(row["m_fp32"] > 0 for row in eligible),
                    "FP32-correct sample has a nonpositive margin; no silent population changes allowed")
            selected_ids = [row["sample_id"] for row in (eligible[:16] if args.phase == 1 else eligible)]
        require(bool(selected_ids), "no eligible calibration samples")
        require([row["sample_id"] for row in calibration_refs] == list(calibration_meta), "calibration order changed")
        correct_total = sum(row["fp32_correct"] for row in calibration_refs)
        calibration_ref_map = {row["sample_id"]: row for row in calibration_refs}
        indices = [i for i, row in enumerate(calibration_meta_rows) if row["sample_id"] in set(selected_ids)]
        require([calibration_meta_rows[i]["sample_id"] for i in indices] == selected_ids, "cohort order changed")
        cohort_loader = make_loader(args.calibration_csv, args, indices)
        log(f"Calibration: total={len(calibration_refs)}, FP32-correct={correct_total}, selected={len(indices)}")
        write_csv(args.output_dir / "fp32_calibration_samples.csv", calibration_refs)
        percentiles = PHASE1_PERCENTILES if args.phase == 1 else FULL_PERCENTILES
        active_specs = specs[:1] if args.phase == 1 else specs
        criterion_list = criteria_for_phase(args.phase)
        candidates_all, scalar_rows = [], []
        if args.phase == 3:
            scalar_rows = read_csv(args.phase2_dir / "calibration_candidate_samples.csv")
            stored_candidates = read_csv(args.phase2_dir / "calibration_candidate_sweep.csv")
            for spec in active_specs:
                by_percentile = {}
                for row in stored_candidates:
                    if row["target"] == spec.target:
                        p = float(row["candidate_percentile"])
                        by_percentile[p] = {"target": spec.target, "candidate_percentile": p,
                                            "scale": float(row["scale"]), "clip_threshold": float(row["clip_threshold"])}
                require(set(by_percentile) == set(FULL_PERCENTILES), "Phase 2 candidate set changed")
                current = [row for row in scalar_rows if row["target"] == spec.target]
                for percentile in FULL_PERCENTILES:
                    require([row["sample_id"] for row in current if float(row["candidate_percentile"]) == percentile]
                            == selected_ids, "CBDB population differs from Phase 2")
                candidates_all.extend(summarize_candidates(current, FULL_PERCENTILES,
                                                          [by_percentile[p] for p in FULL_PERCENTILES], ALPHAS))
            checks["calibration_checks_inherited_from_phase2"] = phase2_manifest["sanity_checks"]
        else:
            for spec in active_specs:
                candidates, scalars = calibrate(model, spec, cohort_loader, calibration_ref_map,
                                                 calibration_meta, percentiles, args, checks)
                candidates_all.extend(candidates)
                scalar_rows.extend(scalars)
        require(state_hash(model) == clean_hash, "calibration changed checkpoint model")
        del model
        gc.collect()
        selections = [select_candidate([c for c in candidates_all if c["target"] == spec.target], criterion)
                      for spec in active_specs for criterion in criterion_list]
        sweep = []
        for candidate in candidates_all:
            for criterion in criterion_list:
                alpha = 0.0 if criterion.alpha is None else criterion.alpha
                sweep.append({**{key: candidate[key] for key in ("target", "candidate_percentile", "clip_threshold", "scale",
                                                               "calibration_mse", "calibration_decision_projection")},
                              **asdict(criterion), "calibration_decision_budget": candidate["db"][str(alpha)],
                              "calibration_cb_decision_budget": candidate["cbdb"][str(alpha)] if args.phase == 3 else None})
        checks["db_alpha0_max_abs_diff"] = max(abs(c["db"]["0.0"] - c["calibration_decision_projection"]) for c in candidates_all)
        write_csv(args.output_dir / "calibration_candidate_sweep.csv", sweep)
        write_csv(args.output_dir / "calibration_candidate_samples.csv", scalar_rows)
        write_csv(args.output_dir / "calibration_selected_scales.csv", selections)
        eval_refs: dict[str, Any] = {}
        if args.phase == 3:
            eval_refs.update({row["sample_id"]: row for row in load_saved_references(args.phase2_dir / "fp32_eval_reference.csv")})
        eval_samples, comparisons = [], []
        eval_loader = make_loader(args.eval_csv, args)
        for selection in selections:
            spec = next(spec for spec in active_specs if spec.target == selection["target"])
            rows = evaluate_selection(selection, spec, eval_loader, eval_meta, args, checks, eval_refs, clean_hash)
            eval_samples.extend(rows)
            comparisons.extend(comparison_rows(selection, rows, classes))
            # Persist finished independent runs so failures do not erase completed evidence.
            write_csv(args.output_dir / "calibration_eval_samples.csv", eval_samples)
            write_csv(args.output_dir / "calibration_method_comparison.csv", comparisons)
        require(list(eval_refs) == list(eval_meta), "eval did not cover the full fixed split")
        write_csv(args.output_dir / "fp32_eval_reference.csv", list(eval_refs.values()))
        require(identities(args) == input_ids, "input files/code changed during experiment")
        checks.update({"all_passed": True, "quantized_tensor_equals_gradient_tensor": True,
                       "padding_excluded_from_all_objectives_and_distortion": True,
                       "fp32_reference_invariant": True, "no_model_state_contamination": True,
                       "same_candidate_same_quantizer": True})
        if args.phase == 1:
            checks["phase1_six_checks"] = {
                "db0_equals_dp": checks["db_alpha0_max_abs_diff"] < 1e-8,
                "same_candidate_identical_activation": checks["identical_candidate_quantization_comparisons"] > 0,
                "fp32_reference_invariant": checks["fp32_reference_comparisons"] > 0,
                "gradient_matches_quantized_tensor": checks["input_gradient_batches"] > 0,
                "padding_excluded": checks["synthetic_checks"]["adversarial_padding_exclusion"],
                "no_model_contamination": checks["unchanged_model_state_checks"] >= 4,
            }
            require(all(checks["phase1_six_checks"].values()), "Phase 1 sanity check incomplete")
        write_json(args.output_dir / "sanity_checks.json", checks)
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                    text=True, check=True).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            commit = None
        manifest.update({
            "status": "passed", "dataset": "local_data", "checkpoint": str(args.checkpoint),
            "train_csv": str(ROOT / "local_data/splits/train.csv"), "calibration_csv": str(args.calibration_csv),
            "eval_csv": str(args.eval_csv), "test_used": False, "input_sha256": input_ids,
            "sample_count": len(eval_refs), "calibration_sample_count": len(calibration_refs),
            "fp32_correct_calibration_sample_count": correct_total,
            "excluded_fp32_error_calibration_sample_count": len(calibration_refs) - correct_total,
            "used_calibration_sample_count": len(selected_ids), "calibration_sample_ids": selected_ids,
            "calibration_selection": "first 16 FP32-correct in original CSV order" if args.phase == 1 else "all FP32-correct",
            "eval_sample_count": len(eval_refs), "seed": project_config.SPLIT_SEED,
            "crop_policy": project_config.AUDIO["CROP_MODE"], "audio_protocol": project_config.AUDIO,
            "sample_rate": project_config.AUDIO["TARGET_SR"], "target_seconds": project_config.AUDIO["TARGET_SECONDS"],
            "pooling": "mask_aware_mean", "padding_mask_policy": "True=padding; excluded before flatten, MSE, dot, distortion",
            "quant_dtype": "signed symmetric INT4", "qmin": -7, "qmax": 7, "zero_point": 0,
            "activation_granularity": "per_tensor", "static_scale": True,
            "weight_granularity": "FP32 unquantized", "weight_quantization": "disabled", "amp": False,
            "targets": [entry for entry in inspection["targets"] if entry["target"] in {s.target for s in active_specs}],
            "candidate_percentiles": list(percentiles), "alphas": [0.0] if args.phase == 1 else list(ALPHAS),
            "full_frozen_candidate_percentiles": list(FULL_PERCENTILES), "full_frozen_alphas": list(ALPHAS),
            "fixed_competitor_definition": inspection["fixed_margin_definition"],
            "gradient_definition": inspection["gradient_definition"], "candidate_generator": inspection["candidate_generator"],
            "tie_break_rule": TIE_RULE, "batch_size": args.batch_size, "num_workers": args.num_workers,
            "device": str(args.device), "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
            "relative_distortion_definition": "existing embedding_distortion on flattened valid activation: L2(delta)/L2(h)",
            "clipping_rate_definition": "existing activation_clipping_rate: h/scale outside [-7,7], padding excluded",
            "near_zero_rate_definition": "existing compute_near_zero_rate alias: exact zero on dequantized valid activation",
            "macro_f1_definition": "existing _macro_f1: classes with true support in reporting group",
            "classification_flip_definition": "FP32 correct AND A4 wrong; denominator n_fp32_correct",
            "clean_model_state_sha256": clean_hash, "sanity_checks": checks,
            "git_commit_if_available": commit,
            "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "method_ranking": "not evaluated in Phase 1" if args.phase == 1 else "factual metrics only; no automatic method redesign",
            "phases_automatically_continued": False,
        })
        files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != MANIFEST)
        manifest["output_sha256"] = {path.name: sha256(path) for path in files}
        write_json(args.output_dir / MANIFEST, manifest)
        report_lines = [f"Phase {args.phase}: PASSED", f"Calibration: {len(selected_ids)}/{correct_total} FP32-correct; original={len(calibration_refs)}",
                        f"Full eval samples: {len(eval_refs)}", "A4-only; exact input gradients; no training; no test data."]
        for selection in selections:
            report_lines.append(f"{selection['target']} {selection['criterion']} alpha={selection['alpha']}: "
                                f"P{selection['selected_percentile']:g}, scale={selection['selected_scale']:.12g}, "
                                f"ties={selection['objective_tie_count']}")
        report_lines.append("Phase 1 tests correctness only; no method ranking or screening claim." if args.phase == 1
                            else "See comparison CSV for downstream metrics; no methods or candidate ranges were changed.")
        # Report is intentionally outside output hashes: it is derived solely from the manifest.
        (args.output_dir / "factual_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        log("\n".join(report_lines))
        return manifest
    except Exception as exc:
        manifest.update({"status": "failed", "error": repr(exc), "sanity_checks": checks})
        write_json(args.output_dir / MANIFEST, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase1-dir", type=Path, default=ROOT / "artifacts/decision_aware_calibration_pilot/phase1")
    parser.add_argument("--phase2-dir", type=Path, default=ROOT / "artifacts/decision_aware_calibration_pilot/phase2")
    parser.add_argument("--device", default=project_config.DEVICE)
    # Batch one gives exact samplewise FP32 references across census/subset passes.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    require(args.batch_size == 1, "frozen pilot uses batch_size=1 for exact census/subset FP32 invariance")
    require(args.num_workers >= 0, "num_workers must be nonnegative")
    require(str(project_config.ACTIVE_DATASET) == "local_data", "pilot refuses any dataset other than local_data")
    args.device = torch.device(args.device)
    require(args.device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    args.checkpoint = ROOT / "local_data/BEATs_local_data.pt"
    args.calibration_csv = ROOT / "local_data/splits/calibration.csv"
    args.eval_csv = ROOT / "local_data/splits/eval.csv"
    args.target_manifest = ROOT / "local_data/matched_single_layer_decision_attribution/manifest.json"
    args.output_dir = (args.output_dir or ROOT / f"artifacts/decision_aware_calibration_pilot/phase{args.phase}").resolve()
    for path in (args.phase1_dir, args.phase2_dir):
        require(args.output_dir != path.resolve() or args.phase == (1 if path == args.phase1_dir else 2),
                "a phase cannot overwrite a prerequisite directory")
    run(args)


if __name__ == "__main__":
    main()
