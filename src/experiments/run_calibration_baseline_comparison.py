"""Matched local_data calibration baselines and a nine-scale eval reference.

Selection is calibration-only. Macro-F1, accuracy and conditional flip each have
their own full eval ranking; eval is never used to change a criterion or alpha.
CEGradSquared denotes per-sample CE-gradient-squared weighted activation error,
not an exact Hessian or a full Fisher-information quadratic form.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import gc
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.experiments.run_decision_aware_calibration_pilot as pilot
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from src.export.export_Vanilla_KD import SOFTMAX_TEMPERATURE, _kl

PROTOCOL = "matched_calibration_baselines.ce_grad_squared_output_kl.v1"
CE_CRITERION = "CEGradSquared"
KL_CRITERION = "OutputKL"
NEW_DEPENDENCIES = (
    "src/experiments/run_calibration_baseline_comparison.py",
    "src/export/export_Vanilla_KD.py",
)
MANIFEST = "baseline_comparison_manifest.json"


def require(condition: bool, message: str) -> None:
    pilot.require(condition, message)


def kl_per_sample(fp32_logits: Tensor, quant_logits: Tensor) -> Tensor:
    require(SOFTMAX_TEMPERATURE == 1.0, "the existing KL helper must use fixed T=1")
    require(fp32_logits.shape == quant_logits.shape, "KL logits shape mismatch")
    # Model forward remains FP32. Only the small classifier-output reduction is FP64.
    result = _kl(fp32_logits.detach().double(), quant_logits.detach().double())
    require(bool(torch.isfinite(result).all()), "nonfinite output KL")
    require(bool((result >= -1e-12).all()), "KL direction/numerics produced a negative divergence")
    return result


def weighted_error(h: Tensor, q: Tensor, ce_gradient: Tensor, mask: Tensor,
                   spec: pilot.single_layer.TargetSpec, sample_index: int) -> tuple[float, float, int]:
    hv, qv, gv, valid_n, _, channels = pilot.three_factor._flatten_valid(
        h_fp32=h, h_q=q, gradient=ce_gradient, token_padding_mask=mask,
        spec=spec, sample_index=sample_index)
    delta_sq = (qv - hv).square()
    require(bool(torch.isfinite(gv).all()), "nonfinite CE input gradient")
    value = float((gv.square() * delta_sq).mean())
    mse = float(delta_sq.mean())
    elements = valid_n * channels
    require(elements == hv.numel() and elements > 0, "D_x must equal the valid scalar count")
    require(math.isfinite(value) and value >= 0, "invalid CE-gradient-squared objective")
    return value, mse, elements


def capture_ce_gradient(model: nn.Module, spec: pilot.single_layer.TargetSpec,
                        batch: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    module = pilot.single_layer.get_ffn_module(model, spec.layer_idx, spec.module_name)
    require(isinstance(module, nn.Linear), "CE capture requires a clean FP32 Linear")
    captured: dict[str, Tensor] = {}

    def pre_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        raw = inputs[0].detach().requires_grad_(True)
        captured["actual_input"] = raw
        # This exact tensor is consumed by the Linear; A4 wraps this same input.
        return (raw, *inputs[1:])

    handle = module.register_forward_pre_hook(pre_hook)
    try:
        with torch.enable_grad():
            logits, mask = pilot.forward(model, batch, args.device)
            labels = batch["label_id"].to(args.device).long()
            require(bool(logits.argmax(1).eq(labels).all()), "CE population differs from FP32-correct cohort")
            h, layout = pilot.three_factor._canonicalize(
                captured["actual_input"], mask, context=spec.target + ".CE_input")
            loss = F.cross_entropy(logits, labels, reduction="sum")
            (raw_grad,) = torch.autograd.grad(loss, captured["actual_input"],
                                             retain_graph=False, create_graph=False)
            gradient = pilot.three_factor._canonicalize_like(
                raw_grad, mask, layout, context=spec.target + ".CE_input_gradient")
            require(h.shape == gradient.shape, "CE gradient must match the quantized activation")
    finally:
        handle.remove()
    pilot.single_layer._assert_no_param_grads(model, context="CE gradient only, no training")
    return {"h": h.detach(), "gradient": gradient.detach(), "mask": mask.detach(),
            "logits": logits.detach(), "ce_loss": float(loss.detach()), "layout": layout}


def add_eval_ranks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for target in dict.fromkeys(row["target"] for row in rows):
        current = [dict(row) for row in rows if row["target"] == target]
        for metric, rank_name, descending in (
            ("eval_macro_f1", "macro_f1_rank", True),
            ("eval_accuracy", "accuracy_rank", True),
            ("flip_rate", "conditional_flip_rank", False),
        ):
            values = sorted({float(row[metric]) for row in current}, reverse=descending)
            for row in current:
                row[rank_name] = values.index(float(row[metric])) + 1
        result.extend(sorted(current, key=lambda row: (
            -float(row["eval_macro_f1"]), -float(row["eval_accuracy"]),
            float(row["candidate_percentile"]))))
    return result


def synthetic_sanity() -> dict[str, Any]:
    spec = pilot.single_layer.TargetSpec(4, "fc1")
    original = pilot.sanity_synthetic(spec)
    h = torch.tensor([[[.31, -.7], [500., -900.], [1.2, -.02]]], dtype=torch.float64,
                     requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    weight = torch.tensor([[.6, -.3], [-.2, .4], [.1, .2]], dtype=torch.float64)
    bias = torch.tensor([.05, -.1, 0.], dtype=torch.float64)

    def ce(value: Tensor) -> Tensor:
        pooled = value[:, ~mask[0]].mean(1)
        return F.cross_entropy(F.linear(pooled, weight, bias), torch.tensor([0]))

    (gradient,) = torch.autograd.grad(ce(h), h)
    probs = F.linear(h.detach()[:, ~mask[0]].mean(1), weight, bias).softmax(1)
    residual = probs.clone()
    residual[:, 0] -= 1
    analytic = (residual @ weight) / 2
    require(torch.allclose(gradient[:, 0], analytic, atol=1e-14, rtol=0), "CE input gradient analytic mismatch")
    require(torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1])), "padding influenced CE gradient")
    max_difference = 0.0
    for token in (0, 2):
        for channel in range(2):
            plus, minus = h.detach().clone(), h.detach().clone()
            plus[0, token, channel] += 1e-5
            minus[0, token, channel] -= 1e-5
            finite_difference = float((ce(plus) - ce(minus)) / 2e-5)
            max_difference = max(max_difference, abs(finite_difference - float(gradient[0, token, channel])))
    require(max_difference < 1e-8, "CE input gradient finite-difference mismatch")
    q = pilot.quantize_dequantize_ste(h.detach(), .1, qmin=-7, qmax=7)
    score, _, count = weighted_error(h.detach(), q, gradient, mask, spec, 0)
    manual = sum(float(gradient[0, t, c]) ** 2 * float(q[0, t, c] - h.detach()[0, t, c]) ** 2
                 for t in (0, 2) for c in range(2)) / 4
    require(count == 4 and abs(score - manual) < 1e-15, "per-sample D_x normalization mismatch")
    hp, qp, gp = h.detach().clone(), q.clone(), gradient.clone()
    hp[:, 1], qp[:, 1], gp[:, 1] = 1e9, -1e9, 3e9
    require(weighted_error(hp, qp, gp, mask, spec, 0)[0] == score, "padding contaminated weighted error")
    # Squared elementwise gradient/error products must not become a squared dot product.
    cancellation = weighted_error(torch.zeros(1, 1, 2), torch.ones(1, 1, 2),
                                  torch.tensor([[[1., -1.]]]), torch.zeros(1, 1, dtype=torch.bool), spec, 0)[0]
    require(cancellation == 1., "weighted error incorrectly reused the DB projection")
    p = torch.tensor([[.7, .2, .1]], dtype=torch.float64)
    other = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    expected = sum(float(a) * math.log(float(a / b)) for a, b in zip(p[0], other[0]))
    actual = float(kl_per_sample(p.log(), other.log())[0])
    require(abs(actual - expected) < 1e-14, "KL direction or temperature mismatch")
    require(float(kl_per_sample(p.log(), p.log())[0]) == 0., "clean FP32 KL identity must be zero")
    rank_rows = [{"target": "test", "candidate_percentile": i, "eval_macro_f1": f1,
                  "eval_accuracy": acc, "flip_rate": flip}
                 for i, (f1, acc, flip) in enumerate(((.2, .9, .3), (.8, .4, .2), (.8, .6, .1)))]
    ranked = add_eval_ranks(rank_rows)
    require(ranked[0]["candidate_percentile"] == 2 and ranked[0]["macro_f1_rank"] == 1,
            "primary Macro-F1 / secondary accuracy ranking mismatch")
    require(next(row for row in ranked if row["candidate_percentile"] == 0)["accuracy_rank"] == 1,
            "independent accuracy ranking was collapsed into a composite score")
    example = [{"target": "test", "candidate_percentile": p, "clip_threshold": s * 7, "scale": s,
                "calibration_mse": mse, "calibration_decision_projection": dp,
                "calibration_ce_gradient_squared_error": ce_value, "calibration_output_kl": kl_value,
                "db": {str(alpha): dp for alpha in pilot.ALPHAS}}
               for p, s, mse, dp, ce_value, kl_value in (
                   (99., .1, 3., 3., 1., 2.), (99.9, .2, 2., 1., 2., 3.), (100., .3, 1., 2., 3., 1.))]
    choices = select_methods(example)
    require(next(row for row in choices if row["criterion"] == CE_CRITERION)["selected_percentile"] == 99.,
            "CE-gradient objective was not used for selection")
    require(next(row for row in choices if row["criterion"] == KL_CRITERION)["selected_percentile"] == 100.,
            "actual KL objective was not used for selection")
    return {"passed": True, "existing_quantizer_and_padding": original,
            "ce_gradient_finite_difference_max_abs_diff": max_difference,
            "weighted_error_manual_max_abs_diff": abs(score - manual),
            "weighted_error_padding_excluded": True, "per_sample_valid_element_normalization": True,
            "not_squared_margin_projection": True, "KL_T1_teacher_to_quant_direction": True,
            "KL_identity_zero": True, "independent_three_metric_rankings": True,
            "new_criteria_use_their_own_primary_objectives": True}


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    input_ids = pilot.identities(args)
    old = pilot.prerequisite(args.phase2_dir, 2, input_ids, args)
    require(str(pilot.project_config.ACTIVE_DATASET) == "local_data", "only local_data is allowed")
    require(str(torch.__version__) == old["torch_version"] and torch.get_num_threads() == old["torch_threads"],
            "runtime differs from the frozen Phase 2 run")
    require(old["candidate_percentiles"] == list(pilot.FULL_PERCENTILES), "frozen candidate set changed")
    require(old["alphas"] == list(pilot.ALPHAS), "frozen DB alphas changed")
    require(old["used_calibration_sample_count"] == 177, "expected the same 177 matched calibration samples")
    require(old["weight_quantization"] == "disabled", "A4-only is required")
    specs, inspection = pilot.target_specs(args.target_manifest)
    cal_refs = pilot.load_saved_references(args.phase2_dir / "fp32_calibration_samples.csv")
    eval_refs = pilot.load_saved_references(args.phase2_dir / "fp32_eval_reference.csv")
    selected_ids = old["calibration_sample_ids"]
    require(selected_ids == [row["sample_id"] for row in cal_refs if row["fp32_correct"]],
            "calibration cohort changed")
    metadata = pilot.read_csv(args.calibration_csv)
    require([row["sample_id"] for row in metadata] == [row["sample_id"] for row in cal_refs],
            "calibration CSV order changed")
    selected_set = set(selected_ids)
    indices = [i for i, row in enumerate(metadata) if row["sample_id"] in selected_set]
    sweep = pilot.read_csv(args.phase2_dir / "calibration_candidate_sweep.csv")
    scalars = pilot.read_csv(args.phase2_dir / "calibration_candidate_samples.csv")
    scalar_map = {(row["target"], float(row["candidate_percentile"]), row["sample_id"]): row for row in scalars}
    require(len(scalar_map) == 2 * 9 * 177, "Phase 2 scalar cache is incomplete")
    candidates = []
    for spec in specs:
        for percentile in pilot.FULL_PERCENTILES:
            entries = [row for row in sweep if row["target"] == spec.target
                       and float(row["candidate_percentile"]) == percentile]
            require(len(entries) == 7, "Phase 2 candidate records incomplete")
            first = entries[0]
            candidate = {"target": spec.target, "candidate_percentile": percentile,
                         **{key: float(first[key]) for key in ("scale", "clip_threshold", "calibration_mse",
                                                               "calibration_decision_projection")}, "db": {}}
            for alpha in pilot.ALPHAS:
                entry = next(row for row in entries if row["criterion"] == "DB" and float(row["alpha"]) == alpha)
                candidate["db"][str(alpha)] = float(entry["calibration_decision_budget"])
            require(candidate["db"]["0.0"] == candidate["calibration_decision_projection"], "DB0/DP mismatch")
            candidates.append(candidate)
    source_files = [args.phase2_dir / pilot.MANIFEST,
                    *(args.phase2_dir / name for name in old["output_sha256"])]
    dependencies = [ROOT / path for path in NEW_DEPENDENCIES]
    all_ids = dict(input_ids)
    all_ids.update({str(path.relative_to(ROOT)): pilot.sha256(path) for path in source_files + dependencies})
    return {"old_manifest": old, "input_sha256": all_ids, "specs": specs, "inspection": inspection,
            "cal_refs": {row["sample_id"]: row for row in cal_refs},
            "eval_refs": {row["sample_id"]: row for row in eval_refs},
            "cal_metadata": {row["sample_id"]: row for row in metadata},
            "eval_metadata": {row["sample_id"]: row for row in pilot.read_csv(args.eval_csv)},
            "cohort_loader": pilot.make_loader(args.calibration_csv, args, indices),
            "eval_loader": pilot.make_loader(args.eval_csv, args),
            "candidates": candidates, "old_scalars": scalar_map}


def load_clean(args: argparse.Namespace, context: Mapping[str, Any], checks: dict[str, Any], label: str):
    model, checkpoint = pilot.single_layer._load_clean_fp32_model(args.checkpoint, device=args.device, context=label)
    require(pilot.state_hash(model) == context["old_manifest"]["clean_model_state_sha256"],
            "clean checkpoint state differs from Phase 2")
    checks["independent_clean_reloads"] += 1
    classes = len(checkpoint["label_mapping"])
    del checkpoint
    return model, classes


def compute_ce_objectives(args: argparse.Namespace, context: dict[str, Any], checks: dict[str, Any]):
    rows = []
    fingerprints: dict[tuple[str, str], str] = {}
    quant_hashes: dict[tuple[str, float, str], str] = {}
    classes = 0
    for spec in context["specs"]:
        model, classes = load_clean(args, context, checks, spec.target + " CE gradient")
        candidates = [row for row in context["candidates"] if row["target"] == spec.target]
        seen = []
        for batch in context["cohort_loader"]:
            data = capture_ce_gradient(model, spec, batch, args)
            for ref in pilot.baseline_rows(data["logits"], data["mask"], batch, context["cal_metadata"]):
                pilot.check_reference(ref, context["cal_refs"][ref["sample_id"]], checks)
            checks["CE_input_gradient_batches"] += 1
            ids = pilot.single_layer._batch_sample_ids(batch)
            for i, sid in enumerate(ids):
                fingerprints[(spec.target, sid)] = pilot.tensor_hash(data["h"][i])
            for candidate in candidates:
                q = pilot.quantize_dequantize_ste(data["h"], candidate["scale"], qmin=-7, qmax=7)
                for i, sid in enumerate(ids):
                    value, mse, elements = weighted_error(data["h"], q, data["gradient"], data["mask"], spec, i)
                    old_scalar = context["old_scalars"][(spec.target, candidate["candidate_percentile"], sid)]
                    require(math.isclose(mse, float(old_scalar["activation_mse"]), rel_tol=1e-12, abs_tol=1e-15),
                            "CE capture changed the matched activation/candidate MSE")
                    require(elements == int(old_scalar["valid_scalar_count"]), "valid element count changed")
                    checks["matched_MSE_sample_comparisons"] += 1
                    quant_hashes[(spec.target, candidate["candidate_percentile"], sid)] = pilot.tensor_hash(q[i])
                    rows.append({"target": spec.target, "candidate_percentile": candidate["candidate_percentile"],
                                 "scale": candidate["scale"], "sample_id": sid,
                                 "label_id": context["cal_refs"][sid]["label_id"],
                                 "bucket": context["cal_refs"][sid]["bucket"],
                                 "valid_element_count_D_x": elements, "calibration_mse": mse,
                                 "ce_gradient_squared_weighted_error": value,
                                 "fp32_cross_entropy": data["ce_loss"],
                                 "quantized_input_sha256": quant_hashes[(spec.target, candidate["candidate_percentile"], sid)]})
                del q
            seen.extend(ids)
            del data
            if len(seen) == 1 or len(seen) % 16 == 0:
                pilot.log(f"{spec.target} CE-gradient-squared: {len(seen)}/177")
        require(seen == context["old_manifest"]["calibration_sample_ids"], "CE population/order mismatch")
        require(pilot.state_hash(model) == context["old_manifest"]["clean_model_state_sha256"],
                "CE gradient pass changed model state")
        checks["unchanged_state_checks"] += 1
        del model
        gc.collect()
        for candidate in candidates:
            values = [row["ce_gradient_squared_weighted_error"] for row in rows
                      if row["target"] == spec.target and row["candidate_percentile"] == candidate["candidate_percentile"]]
            require(len(values) == 177, "CE candidate did not use all matched samples")
            candidate["calibration_ce_gradient_squared_error"] = math.fsum(values) / len(values)
        pilot.write_csv(args.output_dir / "calibration_ce_gradient_samples.csv", rows)
    return fingerprints, quant_hashes, classes


def calibration_kl(args: argparse.Namespace, context: dict[str, Any], fingerprints: Mapping[Any, str],
                   quant_hashes: Mapping[Any, str], checks: dict[str, Any]) -> None:
    all_rows = []
    for spec in context["specs"]:
        for candidate in (row for row in context["candidates"] if row["target"] == spec.target):
            label = f"{spec.target} P{candidate['candidate_percentile']:g} calibration KL"
            model, _ = load_clean(args, context, checks, label)
            qparams = {"activation": {"scale": torch.tensor(candidate["scale"]), "zero_point": torch.tensor(0.),
                                      "qmin": -7, "qmax": 7}}
            module = pilot.replace_ffn_module_with_calibrated_quant(
                model, spec.layer_idx, spec.module_name, pilot.project_config.QUANT, qparams, quant_mode="a4")
            controller = pilot.make_single_ffn_quant_controller(
                model, module, layer_idx=spec.layer_idx, module_name=spec.module_name,
                quant_config=pilot.project_config.QUANT, quant_mode="a4")
            controller.qparams_frozen = True
            controller.prepare_quant_forward()
            require(not module.weight_quant_enabled, "weight quantization must remain disabled")
            seen, values = [], []
            try:
                with torch.no_grad():
                    for batch in context["cohort_loader"]:
                        logits, mask = pilot.forward(model, batch, args.device)
                        ids = pilot.single_layer._batch_sample_ids(batch)
                        h, q = module.get_oer_activation_tensors()
                        require(h is not None and q is not None, "actual A4 input capture missing")
                        for i, sid in enumerate(ids):
                            ref = context["cal_refs"][sid]
                            require(pilot.tensor_hash(h[i]) == fingerprints[(spec.target, sid)],
                                    "actual quantized input differs from the CE-gradient tensor")
                            require(pilot.tensor_hash(q[i]) == quant_hashes[(spec.target, candidate["candidate_percentile"], sid)],
                                    "actual QuantLinear differs from the shared candidate quantizer")
                            require(pilot.tensor_hash(batch["waveform"][i]) == ref["waveform_sha256"], "calibration waveform changed")
                            require(pilot.tensor_hash(batch["padding_mask"][i]) == ref["waveform_padding_mask_sha256"], "waveform mask changed")
                            require(pilot.tensor_hash(mask[i]) == ref["token_padding_mask_sha256"], "calibration token mask changed")
                            teacher = torch.tensor([ref["fp32_logits"]], dtype=torch.float64, device=logits.device)
                            score = float(kl_per_sample(teacher, logits[i:i+1])[0])
                            values.append(score)
                            all_rows.append({"target": spec.target, "candidate_percentile": candidate["candidate_percentile"],
                                             "scale": candidate["scale"], "sample_id": sid, "label_id": ref["label_id"],
                                             "bucket": ref["bucket"], "output_kl_fp32_to_a4_T1": score,
                                             "a4_logits": logits[i].detach().cpu().float().tolist()})
                            checks["actual_KL_candidate_input_and_quantization_comparisons"] += 1
                        seen.extend(ids)
                        module.clear_oer_activation_tensors()
                        module.clear_oer_projection_tensors()
                        del h, q, logits
                        if len(seen) == 1 or len(seen) % 32 == 0:
                            pilot.log(f"{label}: {len(seen)}/177")
            finally:
                controller.remove_padding_mask_hooks()
            require(seen == context["old_manifest"]["calibration_sample_ids"], "KL calibration population/order mismatch")
            candidate["calibration_output_kl"] = math.fsum(values) / len(values)
            setattr(model.encoder.layers[spec.layer_idx], spec.module_name, module.linear)
            require(pilot.state_hash(model) == context["old_manifest"]["clean_model_state_sha256"],
                    "KL candidate forward changed model state")
            checks["unchanged_state_checks"] += 1
            checks["complete_KL_calibration_candidates"] += 1
            del model, module, controller
            gc.collect()
            pilot.write_csv(args.output_dir / "calibration_output_kl_samples.csv", all_rows)
            pilot.log(f"{label}: objective={candidate['calibration_output_kl']:.12g}")


def select_methods(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selections = []
    for target in dict.fromkeys(row["target"] for row in candidates):
        current = [row for row in candidates if row["target"] == target]
        for criterion in [pilot.Criterion("MSE"), pilot.Criterion("DP"),
                          *(pilot.Criterion("DB", alpha) for alpha in pilot.ALPHAS)]:
            choice = pilot.select_candidate(current, criterion)
            choice["role"] = "prior_DP_reference" if criterion.criterion == "DP" else "comparison"
            selections.append(choice)
        # Adapt only the primary objective to the existing, frozen selection/tie rule.
        for label, field in ((CE_CRITERION, "calibration_ce_gradient_squared_error"),
                             (KL_CRITERION, "calibration_output_kl")):
            adapted = [{**row, "calibration_decision_projection": row[field]} for row in current]
            choice = pilot.select_candidate(adapted, pilot.Criterion("DP"))
            choice.update({"criterion": label, "role": "comparison", "objective_definition": field})
            selections.append(choice)
    return selections


def flatten_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**{key: value for key, value in candidate.items() if key != "db"},
             **{f"calibration_DB_alpha_{alpha:g}": candidate["db"][str(alpha)] for alpha in pilot.ALPHAS}}
            for candidate in candidates]


def write_reports(args: argparse.Namespace, selections: Sequence[Mapping[str, Any]],
                  comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = [row for row in comparisons if row["group"] == "overall"]
    ranked = add_eval_ranks(overall)
    require(len(ranked) == 18, "full reference must contain nine candidates for each of two targets")
    pilot.write_csv(args.output_dir / "eval_candidate_rankings.csv", ranked)
    hits, method_groups, winners = [], [], {}
    for target in dict.fromkeys(row["target"] for row in ranked):
        current = [row for row in ranked if row["target"] == target]
        best_f1 = max(float(row["eval_macro_f1"]) for row in current)
        best_accuracy = max(float(row["eval_accuracy"]) for row in current)
        min_flip = min(float(row["flip_rate"]) for row in current)
        primary = [row for row in current if float(row["eval_macro_f1"]) == best_f1]
        secondary = max(float(row["eval_accuracy"]) for row in primary)
        winners[target] = {
            "macro_f1_primary_winners": [row["candidate_percentile"] for row in primary],
            "accuracy_secondary_winners_within_primary_tie": [row["candidate_percentile"] for row in primary
                                                              if float(row["eval_accuracy"]) == secondary],
            "accuracy_winners": [row["candidate_percentile"] for row in current if float(row["eval_accuracy"]) == best_accuracy],
            "conditional_flip_winners": [row["candidate_percentile"] for row in current if float(row["flip_rate"]) == min_flip],
            "reference_scope": "post-hoc comparison only, never calibration selection",
        }
        for selection in (row for row in selections if row["target"] == target):
            reference = next(row for row in current if row["candidate_percentile"] == selection["selected_percentile"])
            hits.append({**selection, **{key: reference[key] for key in (
                "eval_macro_f1", "eval_accuracy", "n_fp32_correct", "flip_count", "flip_rate",
                "mean_fixed_margin_drop", "mean_dynamic_margin_drop", "activation_mse", "relative_distortion",
                "clipping_rate", "near_zero_rate", "macro_f1_rank", "accuracy_rank", "conditional_flip_rank")},
                         "selected_eval_primary_optimum": float(reference["eval_macro_f1"]) == best_f1,
                         "macro_f1_gap_to_best_candidate": best_f1 - float(reference["eval_macro_f1"]),
                         "accuracy_gap_to_best_candidate": best_accuracy - float(reference["eval_accuracy"]),
                         "flip_rate_gap_to_lowest_candidate": float(reference["flip_rate"]) - min_flip})
            for row in comparisons:
                if row["target"] == target and row["candidate_percentile"] == selection["selected_percentile"]:
                    method_groups.append({**row, **selection})
    pilot.write_csv(args.output_dir / "method_comparison.csv", hits)
    pilot.write_csv(args.output_dir / "method_group_comparison.csv", method_groups)
    pilot.write_json(args.output_dir / "eval_reference_winners.json", winners)
    lines = ["# 校准准则与完整候选 eval 对比", "",
             "所有准则只根据相同的 177 个 FP32-correct calibration 样本选择 scale。全部 9 个候选的 eval 仅作事后参照，没有反过来调整准则、alpha 或 temperature。", "",
             "CEGradSquared 是实际量化输入的逐样本 CE 梯度平方加权 activation error，按有效元素数 D_x 平均；它不被命名为精确 Hessian。Output KL 为最终分类概率 KL(FP32||A4)，固定 T=1，每个候选均实际运行 calibration forward。", "",
             "Macro-F1 为 primary，Accuracy 为 secondary。Conditional flip 单独报告。相同指标值并列；没有合成加权总分。DP 仅作为旧实验参照保留。", ""]
    fmt = lambda value: f"{float(value):.9g}"
    method_name = lambda row: str(row["criterion"]) + (f"({row['alpha']:g})" if row.get("alpha") is not None else "")
    for target, winner in winners.items():
        lines.extend([f"## {target}", "",
                      f"Macro-F1 最优 percentile：{winner['macro_f1_primary_winners']}；Macro-F1 并列时 Accuracy 最优：{winner['accuracy_secondary_winners_within_primary_tie']}。", "",
                      f"Accuracy 最优 percentile：{winner['accuracy_winners']}；Conditional flip 最优 percentile：{winner['conditional_flip_winners']}。", "",
                      "| 准则 | 选中 P | Scale | F1 % | Acc % | Flip | F1 排名 | Acc 排名 | Flip 排名 | 命中 F1 最优 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
        for row in (item for item in hits if item["target"] == target):
            lines.append("| " + " | ".join([method_name(row), fmt(row["selected_percentile"]), fmt(row["selected_scale"]),
                f"{100*float(row['eval_macro_f1']):.4f}", f"{100*float(row['eval_accuracy']):.4f}",
                f"{row['flip_count']}/{row['n_fp32_correct']}", str(row["macro_f1_rank"]), str(row["accuracy_rank"]),
                str(row["conditional_flip_rank"]), str(row["selected_eval_primary_optimum"])]) + " |")
        lines.extend(["", "完整九候选排名：", "",
                      "| P | Scale | F1 % | Acc % | Flip | F1 排名 | Acc 排名 | Flip 排名 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in (item for item in ranked if item["target"] == target):
            lines.append("| " + " | ".join([fmt(row["candidate_percentile"]), fmt(row["selected_scale"]),
                f"{100*float(row['eval_macro_f1']):.4f}", f"{100*float(row['eval_accuracy']):.4f}",
                f"{row['flip_count']}/{row['n_fp32_correct']}", str(row["macro_f1_rank"]),
                str(row["accuracy_rank"]), str(row["conditional_flip_rank"])]) + " |")
        lines.append("")
    lines.extend(["many/medium/few 的全部指标见 method_group_comparison.csv；few n=1，仅报告数值，不作组别结论。",
                  "本轮没有运行 Phase 3/CBDB，没有训练或修改模型参数，没有使用 test 或 PER。"])
    (args.output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return winners


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    context = load_context(args)
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            f"output directory must be new/empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {"all_passed": False, "fp32_reference_comparisons": 0,
        "independent_clean_reloads": 0, "unchanged_state_checks": 0,
        "CE_input_gradient_batches": 0, "matched_MSE_sample_comparisons": 0,
        "actual_KL_candidate_input_and_quantization_comparisons": 0,
        "complete_KL_calibration_candidates": 0,
        "clean_checkpoint_reloads": 0, "actual_quantlinear_comparisons": 0,
        "unchanged_model_state_checks": 0, "old_eval_sample_comparisons": 0}
    manifest: dict[str, Any] = {"protocol": PROTOCOL, "status": "running",
                              "input_sha256": context["input_sha256"]}
    pilot.write_json(args.output_dir / MANIFEST, manifest)
    try:
        torch.manual_seed(pilot.project_config.SPLIT_SEED)
        pilot.np.random.seed(pilot.project_config.SPLIT_SEED)
        checks["synthetic_sanity"] = synthetic_sanity()
        pilot.log("Preflight passed: frozen Phase 2 cohort/scales/runtime; CE input gradient and KL T=1 sanity")
        pilot.write_csv(args.output_dir / "fp32_calibration_reference.csv", list(context["cal_refs"].values()))
        pilot.write_csv(args.output_dir / "fp32_eval_reference.csv", list(context["eval_refs"].values()))
        fingerprints, quant_hashes, classes = compute_ce_objectives(args, context, checks)
        calibration_kl(args, context, fingerprints, quant_hashes, checks)
        del fingerprints, quant_hashes
        pilot.write_csv(args.output_dir / "calibration_candidate_objectives.csv", flatten_candidates(context["candidates"]))
        selections = select_methods(context["candidates"])
        # Prove that adding baselines did not silently change the original MSE/DP/DB choices.
        old_selections = pilot.read_csv(args.phase2_dir / "calibration_selected_scales.csv")
        for choice in selections:
            if choice["criterion"] not in ("MSE", "DP", "DB"):
                continue
            previous = next(row for row in old_selections if row["target"] == choice["target"]
                            and row["criterion"] == choice["criterion"]
                            and (row["alpha"] == "" if choice["alpha"] is None else float(row["alpha"]) == choice["alpha"]))
            require(float(previous["selected_scale"]) == choice["selected_scale"], "existing selection changed")
        selection_path = args.output_dir / "calibration_selected_scales.csv"
        pilot.write_csv(selection_path, selections)
        selection_hash = pilot.sha256(selection_path)
        manifest.update({"status": "calibration_complete_eval_pending", "frozen_selection_sha256": selection_hash})
        pilot.write_json(args.output_dir / MANIFEST, manifest)
        pilot.log("All calibration selections frozen BEFORE the nine-candidate eval reference")
        old_samples = pilot.read_csv(args.phase2_dir / "calibration_eval_samples.csv")
        old_lookup = {(row["target"], float(row["selected_scale"]), row["sample_id"]): row for row in old_samples}
        all_samples, comparisons = [], []
        for number, candidate in enumerate(context["candidates"], 1):
            spec = next(spec for spec in context["specs"] if spec.target == candidate["target"])
            selection = {"target": candidate["target"], "criterion": "Candidate", "alpha": None,
                         "candidate_percentile": candidate["candidate_percentile"],
                         "selected_percentile": candidate["candidate_percentile"],
                         "selected_scale": candidate["scale"], "selected_clip_threshold": candidate["clip_threshold"]}
            pilot.log(f"Eval reference candidate {number}/18: {spec.target} P{candidate['candidate_percentile']:g}")
            rows = pilot.evaluate_selection(selection, spec, context["eval_loader"], context["eval_metadata"],
                                            args, checks, context["eval_refs"],
                                            context["old_manifest"]["clean_model_state_sha256"])
            for row in rows:
                old = old_lookup.get((spec.target, candidate["scale"], row["sample_id"]))
                if old is not None:
                    require(all(str(value) == old[key] for key, value in row.items() if key not in ("criterion", "alpha")),
                            "a previously evaluated scale no longer matches Phase 2 sample results")
                    checks["old_eval_sample_comparisons"] += 1
                row["candidate_percentile"] = candidate["candidate_percentile"]
            all_samples.extend(rows)
            comparisons.extend(pilot.comparison_rows(selection, rows, classes))
            pilot.write_csv(args.output_dir / "eval_candidate_samples.csv", all_samples)
            pilot.write_csv(args.output_dir / "eval_candidate_metrics.csv", comparisons)
            require(pilot.sha256(selection_path) == selection_hash, "eval modified calibration selections")
        winners = write_reports(args, selections, comparisons)
        for path, value in context["input_sha256"].items():
            require(pilot.sha256(ROOT / path) == value, f"input/code/source changed during run: {path}")
        require(checks["complete_KL_calibration_candidates"] == 18, "not all candidates ran actual calibration KL")
        require(checks["actual_KL_candidate_input_and_quantization_comparisons"] == 3186, "KL population incomplete")
        require(checks["CE_input_gradient_batches"] == 354 and checks["matched_MSE_sample_comparisons"] == 3186,
                "CE-gradient weighted population incomplete")
        require(checks["clean_checkpoint_reloads"] == 18 and checks["actual_quantlinear_comparisons"] == 2664,
                "nine-candidate full eval incomplete")
        checks.update({"all_passed": True, "quantized_tensor_equals_CE_gradient_tensor": True,
                       "all_criteria_matched_calibration_population": True,
                       "selection_frozen_before_eval": True, "padding_excluded_with_D_x_normalization": True,
                       "all_nine_candidates_actually_injected_for_calibration_KL": True,
                       "model_parameters_unchanged": True, "phase2_sources_unchanged": True})
        pilot.write_json(args.output_dir / "sanity_checks.json", checks)
        manifest.update({"status": "passed", "dataset": "local_data", "source_phase2": str(args.phase2_dir),
            "checkpoint": str(args.checkpoint), "calibration_csv": str(args.calibration_csv), "eval_csv": str(args.eval_csv),
            "calibration_original_count": 227, "calibration_fp32_correct_count": 177, "calibration_excluded_error_count": 50,
            "calibration_sample_ids": context["old_manifest"]["calibration_sample_ids"], "eval_count": 148,
            "targets": context["inspection"]["targets"], "candidate_percentiles": list(pilot.FULL_PERCENTILES),
            "alphas": list(pilot.ALPHAS), "quant_mode": "A4-only", "weight_quantization": "disabled",
            "qmin": -7, "qmax": 7, "zero_point": 0, "activation_granularity": "static_per_tensor",
            "ce_gradient_squared_definition": "mean_x(mean_valid_elements((d CE(z_FP32,y)/d h_i)^2 * (Q_s(h_i)-h_i)^2))",
            "gradient_tensor": "exact input activation consumed by A4 QuantLinear",
            "not_exact_hessian_or_full_fisher": True,
            "output_KL_definition": "mean_x KL(softmax(z_FP32)||softmax(z_A4,s)); actual full classifier forward",
            "temperature": 1., "temperature_searched": False, "KL_reduction_dtype": "FP64 on FP32 model logits",
            "selection_population": "same 177 frozen FP32-correct calibration samples for every criterion",
            "tie_break_rule": pilot.TIE_RULE, "primary_metric": "macro_f1", "secondary_metric": "accuracy",
            "separate_robustness_metric": "FP32-correct conditional flip rate",
            "eval_reference_used_for_calibration": False, "eval_reference_winners": winners,
            "rank_ties": "exact metric equality receives equal dense rank; accuracy resolves primary ties only when identifying a secondary winner",
            "DP_role": "previous criterion reference, not a new baseline",
            "test_used": False, "training_performed": False, "phase3_executed": False,
            "clean_model_state_sha256": context["old_manifest"]["clean_model_state_sha256"],
            "device": str(args.device), "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
            "batch_size": 1, "num_workers": args.num_workers, "seed": pilot.project_config.SPLIT_SEED,
            "audio_protocol": pilot.project_config.AUDIO, "pooling": "mask_aware_mean",
            "sanity_checks": checks, "frozen_selection_sha256": selection_hash,
            "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "elapsed_seconds": time.perf_counter() - start})
        manifest["output_sha256"] = {path.name: pilot.sha256(path) for path in sorted(args.output_dir.iterdir())
                                      if path.is_file() and path.name != MANIFEST}
        pilot.write_json(args.output_dir / MANIFEST, manifest)
        pilot.log("BASELINE COMPARISON: PASSED; all nine candidates ranked separately; selections unchanged")
        pilot.log(json.dumps(winners, ensure_ascii=False))
        return manifest
    except Exception as exc:
        manifest.update({"status": "failed", "error": repr(exc), "sanity_checks": checks})
        pilot.write_json(args.output_dir / MANIFEST, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/decision_aware_calibration_pilot/baseline_comparison")
    parser.add_argument("--phase2-dir", type=Path, default=ROOT / "artifacts/decision_aware_calibration_pilot/phase2")
    parser.add_argument("--device", default=pilot.project_config.DEVICE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-only", action="store_true", help="synthetic correctness checks only; no model experiment")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke_only:
        pilot.log(json.dumps(synthetic_sanity(), indent=2))
        return
    args.device = torch.device(args.device)
    require(args.device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    require(args.num_workers >= 0, "num_workers must be nonnegative")
    args.batch_size = 1
    args.checkpoint = ROOT / "local_data/BEATs_local_data.pt"
    args.calibration_csv = ROOT / "local_data/splits/calibration.csv"
    args.eval_csv = ROOT / "local_data/splits/eval.csv"
    args.target_manifest = ROOT / "local_data/matched_single_layer_decision_attribution/manifest.json"
    args.output_dir = args.output_dir.resolve()
    args.phase2_dir = args.phase2_dir.resolve()
    require(args.output_dir != args.phase2_dir, "refusing to overwrite Phase 2 sources")
    run(args)


if __name__ == "__main__":
    main()
