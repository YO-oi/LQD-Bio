from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline.baseline_metrics import correct_logit_margin
from src.baseline.export_eval import (
    DEFAULT_TOKEN_EXPORT,
    DEFAULT_TOKEN_LAYERS,
    ExportEvalConfig,
    forward_export_batch,
    load_fp32_reference_bundle,
    load_trained_checkpoint,
    move_batch_to_device,
    run_export_eval_config,
)
from src.baseline.manifest import canonical_json_hash, file_sha256, path_str
from src.baseline.run_vanilla_kd import _validate_label_tensor
from src.metrics.common_metrics import DEFAULT_EPS, embedding_distortion, tail_and_few_damage_gaps
from src.metrics.geometry_metrics import class_trace_variance
from src.utils.token_trace_hook import resolve_full_layer_indices
from src.innovation.run_two_stage_decoupling import (
    FORMAL_DEFAULTS,
    StageSpec,
    _append_stage_log,
    _configure_stage,
    _make_scheduler,
    _optimizer_lr,
    _reject_amp,
    _seed_everything,
    _write_train_log,
    forward_pooled_logits,
)
from src.innovation.stage2_class_prior_adjustment import (
    _quant_audit,
    _require_stage1_checkpoint_payload,
    build_train_label_buckets,
    validate_split_bucket_coverage,
)
from src.train.train_beats_classifier_head import (
    BASE_BUCKETS,
    build_loader,
    compute_eval_breakdown,
    dataset_paths,
    label_names_by_id,
    load_label_mapping,
    num_predictor_classes,
    validate_split_labels,
)


TAIL_REPAIR_PROTOCOL_VERSION = "innovation_tail_repair_sweep.v1"
BASELINE_NAME = "tail_repair"
MODEL_KIND = "two_stage_tail_repair"
MODEL_NAME = "stage2_tail_margin"
METHOD_VARIANT = "stage2_ce_tail_margin"
SELECTION_METRIC = "eval_tail_macro_f1"
DEFAULT_LAMBDAS = (0.05, 0.10, 0.20)
OVERALL_ACC_GUARDRAIL = -0.005
OVERALL_MACRO_F1_GUARDRAIL = -0.005
MANY_ACC_GUARDRAIL = -0.010
CANONICAL_RELOAD_LOGIT_ATOL = 1e-5
CANONICAL_RELOAD_LOGIT_RTOL = 1e-5
OFFICIAL_CLASSIFICATION_METRIC_SOURCE = "canonical_reload_eval_no_token_artifacts"
DIAGNOSTIC_TOKEN_HOOK_METRIC_SOURCE = "reload_checkpoint_forward_export_batch_all_token_hooks_no_artifacts"
DIAGNOSTIC_EXPORT_METRIC_SOURCE = "token_export_artifact_pass"
EXPORT_PARITY_METRIC_SOURCE = "canonical_reload_vs_actual_token_export_artifact"

SUMMARY_COLUMNS = [
    "candidate_index",
    "selected",
    "passes_guardrail",
    "lambda_tail_margin",
    "gamma_t",
    "stage2_train_tail_margin_loss",
    "stage2_train_tail_margin_loss_tail_sample_mean",
    "stage2_train_tail_margin_loss_batch_weighted",
    "seed",
    "lambda_dir",
    "stage2_checkpoint",
    "stage2_checkpoint_sha256",
    "stage2_selection_checkpoint",
    "stage2_selection_checkpoint_sha256",
    "stage2_train_log",
    "stage2_eval_metrics",
    "stage2_manifest",
    "selection_metric",
    "selection_metric_value",
    "stage2_checkpoint_selection_metric_value",
    "stage2_checkpoint_selection_metric_source",
    "official_classification_metric_source",
    "diagnostic_export_metric_source",
    "stage2_selection_eval_accuracy",
    "stage2_selection_eval_macro_f1_label_universe_nanmean",
    "stage2_selection_eval_many_accuracy",
    "stage2_selection_eval_many_macro_f1",
    "stage2_selection_eval_medium_accuracy",
    "stage2_selection_eval_medium_macro_f1",
    "stage2_selection_eval_tail_accuracy",
    "stage2_selection_eval_tail_macro_f1",
    "selection_reload_sanity_pred_disagreement_count",
    "selection_reload_sanity_max_abs_logit_diff",
    "selection_reload_sanity_max_rel_logit_diff",
    "selection_reload_sanity_logits_allclose",
    "selection_token_hook_sanity_pred_disagreement_count",
    "selection_token_hook_sanity_max_abs_logit_diff",
    "selection_token_hook_sanity_max_rel_logit_diff",
    "selection_token_hook_sanity_logits_allclose",
    "selection_token_hook_sanity_warning",
    "selection_export_parity_sanity_prediction_mismatch_count",
    "selection_export_parity_sanity_n_prediction_mismatch",
    "selection_export_parity_sanity_prediction_mismatch_rate",
    "selection_export_parity_sanity_sample_identity_match",
    "selection_export_parity_sanity_label_sequence_match",
    "selection_export_parity_sanity_max_abs_logit_diff",
    "selection_export_parity_sanity_max_rel_logit_diff",
    "selection_export_parity_sanity_logits_allclose",
    "selection_export_parity_sanity_warning",
    "overall_accuracy",
    "overall_macro_f1",
    "many_accuracy",
    "many_macro_f1",
    "medium_accuracy",
    "medium_macro_f1",
    "tail_accuracy",
    "tail_macro_f1",
    "delta_overall_accuracy_vs_ce_only",
    "delta_overall_macro_f1_vs_ce_only",
    "delta_many_accuracy_vs_ce_only",
    "m_tail",
    "m_many",
    "logit_margin_retention",
    "fisher",
    "geo_margin",
    "class_trace_variance",
    "embedding_distortion",
    "embedding_cosine_similarity",
    "fisher_retention",
    "geo_margin_retention",
    "variance_deviation_abs_log",
    "variance_deviation_nonfinite_class_count",
    "tail_collapse_gap",
    "few_collapse_gap",
    "correct_logit_margin_global",
    "correct_logit_margin_many",
    "correct_logit_margin_tail",
    "best_stage_epoch",
    "best_global_epoch",
    "run_manifest_hash",
    "final_test_exported",
    "final_export_splits",
    "final_dir",
    "final_checkpoint",
    "final_metrics_csv",
    "final_group_metrics_csv",
    "final_geometry_metrics_csv",
    "final_export_manifest",
    "final_export_run_identity_hash",
]

SINGLE_SUMMARY_COLUMNS = [
    "run_mode",
    "single_mode",
    "selection_enabled",
    "guardrail_selection_enabled",
    "diagnostic_passes_guardrail",
    "guardrail_delta_overall_accuracy_min",
    "guardrail_delta_overall_macro_f1_min",
    "guardrail_delta_many_accuracy_min",
    *SUMMARY_COLUMNS,
]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value!r}")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(f"must be non-negative finite float, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"must be positive finite float, got {value!r}")
    return parsed


def _parse_lambdas(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("lambda list must contain at least one value")
    parsed_values: list[float] = []
    for part in parts:
        parsed = float(part)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise argparse.ArgumentTypeError(f"lambda must be positive and finite, got {part!r}")
        parsed_values.append(float(parsed))
    return parsed_values


def _float_text(value: float) -> str:
    return str(float(value))


def _lambda_dir_name(value: float) -> str:
    text = _float_text(value).lower().replace("-", "m").replace("+", "").replace(".", "p")
    return f"lambda_{text}"


def _selected_final_dir_name(value: float) -> str:
    return f"selected_{_lambda_dir_name(value)}_final"


def _lambda_key(value: float) -> str:
    return _float_text(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a strict eval-only Stage2 CE + Tail Margin lambda sweep from a fixed "
            "Stage1 checkpoint, then export test once for the eval-selected lambda."
        )
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stage1-checkpoint", default=None, help="Default: <dataset_root>/innovation/two_stage/seed_<seed>/stage1_checkpoint.pt.")
    parser.add_argument("--two-stage-root", default=None, help="Default: <dataset_root>/innovation/two_stage.")
    parser.add_argument("--reference-root", default=None, help="Default: <dataset_root>/baseline/teacher/seed_<seed>.")
    parser.add_argument("--baseline-root", default=None, help="Default: <dataset_root>/baseline.")
    parser.add_argument("--output-root", default=None, help="Default: <dataset_root>/innovation/tail_repair.")
    parser.add_argument("--ce-only-stage2-eval-metrics", default=None, help="Default: <dataset_root>/innovation/two_stage/seed_<seed>/stage2_eval_metrics.csv.")
    parser.add_argument("--lambda-values", type=_parse_lambdas, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--single", type=_positive_float, default=None, help="Run exactly one explicit lambda under <output_seed_dir>/single/lambda_<value>; no sweep selection and no test split.")
    parser.add_argument("--force-single", action="store_true", help="Allow clearing an existing --single output directory before running.")
    parser.add_argument("--stage2-epochs", type=_positive_int, default=int(FORMAL_DEFAULTS["stage2_epochs"]))
    parser.add_argument("--stage2-classifier-lr", type=float, default=float(FORMAL_DEFAULTS["stage2_classifier_lr"]))
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=float(FORMAL_DEFAULTS["warmup_ratio"]))
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=8)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--export-name", default=BASELINE_NAME)
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT, help="Compatibility option; this script's diagnostic artifact pass is forced to token_export=all.")
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS, help="Compatibility option; diagnostic artifact export uses all token layers.")
    parser.add_argument("--export-all-token-layers", action="store_true", help="Compatibility option; diagnostic artifact export uses all token layers.")
    parser.add_argument("--max-export-payload-gib", type=float, default=16.0)
    parser.add_argument("--summary-csv", default=None, help="Default: <output_seed_dir>/tail_repair_sweep_eval_summary.csv.")
    parser.add_argument("--manifest-path", default=None, help="Default: <output_seed_dir>/tail_repair_sweep_manifest.json.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.force_single) and args.single is None:
        parser.error("--force-single requires --single")
    if args.single is not None and (args.summary_csv or args.manifest_path):
        parser.error(
            "--single writes fixed single-run artifacts under <output_seed_dir>/single/lambda_<value>; "
            "do not pass --summary-csv or --manifest-path with --single"
        )
    return args


def _output_seed_dir(args: argparse.Namespace) -> Path:
    root = dataset_paths(args.dataset, None).root
    output_root = Path(args.output_root) if args.output_root else root / "innovation" / BASELINE_NAME
    return output_root / f"seed_{int(args.seed)}"


def _two_stage_seed_dir(args: argparse.Namespace) -> Path:
    root = dataset_paths(args.dataset, None).root
    two_stage_root = Path(args.two_stage_root) if args.two_stage_root else root / "innovation" / "two_stage"
    return two_stage_root / f"seed_{int(args.seed)}"


def _stage1_checkpoint(args: argparse.Namespace) -> Path:
    return Path(args.stage1_checkpoint) if args.stage1_checkpoint else _two_stage_seed_dir(args) / "stage1_checkpoint.pt"


def _reference_root(args: argparse.Namespace) -> Path:
    root = dataset_paths(args.dataset, None).root
    baseline_root = Path(args.baseline_root) if args.baseline_root else root / "baseline"
    return Path(args.reference_root) if args.reference_root else baseline_root / "teacher" / f"seed_{int(args.seed)}"


def _ce_only_metrics(args: argparse.Namespace) -> Path:
    return Path(args.ce_only_stage2_eval_metrics) if args.ce_only_stage2_eval_metrics else _two_stage_seed_dir(args) / "stage2_eval_metrics.csv"


def _summary_csv(args: argparse.Namespace, seed_dir: Path) -> Path:
    return Path(args.summary_csv) if args.summary_csv else seed_dir / "tail_repair_sweep_eval_summary.csv"


def _manifest_path(args: argparse.Namespace, seed_dir: Path) -> Path:
    return Path(args.manifest_path) if args.manifest_path else seed_dir / "tail_repair_sweep_manifest.json"


def _single_lambda_dir(seed_dir: Path, value: float) -> Path:
    return seed_dir / "single" / _lambda_dir_name(float(value))


def _single_summary_csv(single_dir: Path) -> Path:
    return single_dir / "tail_repair_single_eval_summary.csv"


def _single_manifest_path(single_dir: Path) -> Path:
    return single_dir / "tail_repair_single_manifest.json"


def _stage2_spec(args: argparse.Namespace) -> StageSpec:
    return StageSpec(
        name=MODEL_NAME,
        index=2,
        objective="stage2_classifier_only_ce_plus_tail_correct_logit_margin_hinge",
        epochs=int(args.stage2_epochs),
        backbone_lr=0.0,
        classifier_lr=float(args.stage2_classifier_lr),
        kd_enabled=False,
        train_backbone=False,
        train_classifier=True,
    )


def _read_single_csv_row(path: str | Path) -> dict[str, str]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"expected CSV is missing: {target}")
    with target.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one CSV row in {target}, got {len(rows)}")
    return rows[0]


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"expected CSV is missing: {target}")
    with target.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"expected CSV rows in {target}")
    return rows


def _read_json_mapping(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"expected JSON is missing: {target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {target}")
    return payload


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, columns: Sequence[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    fieldnames = list(columns or [])
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _prepare_single_output_dir(single_dir: Path, *, force: bool) -> None:
    if single_dir.exists() and any(single_dir.iterdir()):
        if not bool(force):
            raise FileExistsError(
                f"single-lambda output directory already exists and is not empty: {single_dir}. "
                "Pass --force-single to clear and replace it."
            )
        single_root = single_dir.parent.resolve()
        target = single_dir.resolve()
        if single_root.name != "single" or single_root not in target.parents:
            raise RuntimeError(f"refusing to clear unexpected single-lambda directory: {single_dir}")
        shutil.rmtree(target)
    single_dir.mkdir(parents=True, exist_ok=True)


def _float_value(row: Mapping[str, Any], key: str, *, default: float = float("nan")) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return float(default)
    parsed = float(value)
    if math.isinf(parsed):
        raise ValueError(f"{key} must be finite or NaN, got {value!r}")
    return parsed


def _metric_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
    analysis_group: str,
    default: float | None = None,
) -> float:
    for row in rows:
        if str(row.get("metric_name")) == str(metric_name) and str(row.get("analysis_group")) == str(analysis_group):
            return _float_value(row, "metric_value")
    if default is not None:
        return float(default)
    raise KeyError(f"missing metric row: metric_name={metric_name}, analysis_group={analysis_group}")


def _batch_sample_ids_for_canonical_eval(batch: Mapping[str, Any]) -> list[str]:
    raw = batch.get("sample_id")
    if raw is None:
        raise KeyError("canonical reload eval requires stable sample_id in each batch")
    if isinstance(raw, Tensor):
        return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    return [str(raw)]


def _tail_mask(labels: Tensor, label_buckets: Mapping[int, str]) -> Tensor:
    values = [str(label_buckets.get(int(label), "unknown")) in {"medium", "few"} for label in labels.detach().cpu().long().reshape(-1).tolist()]
    return torch.tensor(values, dtype=torch.bool, device=labels.device)


def _bucket_mask(labels: Tensor, label_buckets: Mapping[int, str], group_name: str) -> Tensor:
    if group_name == "tail":
        values = [str(label_buckets.get(int(label), "unknown")) in {"medium", "few"} for label in labels.detach().cpu().long().reshape(-1).tolist()]
    else:
        values = [str(label_buckets.get(int(label), "unknown")) == str(group_name) for label in labels.detach().cpu().long().reshape(-1).tolist()]
    return torch.tensor(values, dtype=torch.bool)


def _mean_finite(values: Tensor) -> float:
    flat = values.detach().cpu().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if int(finite.numel()) <= 0:
        return float("nan")
    return float(finite.mean().item())


def _median_finite(values: Sequence[float]) -> float:
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float32)
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) <= 0:
        return float("nan")
    sorted_values = torch.sort(finite.to(dtype=torch.float64)).values
    count = int(sorted_values.numel())
    midpoint = count // 2
    if count % 2 == 1:
        return float(sorted_values[midpoint].item())
    return float(((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0).item())


def _class_f1(label_id: int, *, true_count: Mapping[int, int], pred_count: Mapping[int, int], correct_count: Mapping[int, int]) -> float:
    support = int(true_count.get(int(label_id), 0))
    if support <= 0:
        return float("nan")
    correct = int(correct_count.get(int(label_id), 0))
    false_positive = max(0, int(pred_count.get(int(label_id), 0)) - correct)
    false_negative = max(0, support - correct)
    denominator = (2 * correct) + false_positive + false_negative
    return (2 * correct) / float(denominator) if denominator > 0 else float("nan")


def _nanmean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    return sum(finite) / float(len(finite))


def _classification_summary(
    *,
    labels: Sequence[int],
    predictions: Sequence[int],
    label_names: Mapping[int, str],
    label_buckets: Mapping[int, str],
) -> dict[str, float]:
    if not labels or len(labels) != len(predictions):
        raise RuntimeError("classification summary requires aligned non-empty labels/predictions")
    labels_list = [int(item) for item in labels]
    preds_list = [int(item) for item in predictions]
    result: dict[str, float] = {}
    all_ids = sorted(int(item) for item in label_names.keys())
    group_to_ids = {
        "global": all_ids,
        "many": [label_id for label_id in all_ids if str(label_buckets.get(label_id)) == "many"],
        "medium": [label_id for label_id in all_ids if str(label_buckets.get(label_id)) == "medium"],
        "few": [label_id for label_id in all_ids if str(label_buckets.get(label_id)) == "few"],
        "tail": [label_id for label_id in all_ids if str(label_buckets.get(label_id)) in {"medium", "few"}],
    }
    for group_name, label_ids in group_to_ids.items():
        label_id_set = set(int(item) for item in label_ids)
        sample_indices = [index for index, label_id in enumerate(labels_list) if label_id in label_id_set]
        true_count: dict[int, int] = {}
        pred_count: dict[int, int] = {}
        correct_count: dict[int, int] = {}
        for index in sample_indices:
            label_id = labels_list[index]
            pred = preds_list[index]
            true_count[label_id] = true_count.get(label_id, 0) + 1
            pred_count[pred] = pred_count.get(pred, 0) + 1
            if label_id == pred:
                correct_count[label_id] = correct_count.get(label_id, 0) + 1
        if sample_indices:
            correct = sum(1 for index in sample_indices if preds_list[index] == labels_list[index])
            accuracy = correct / float(len(sample_indices))
        else:
            accuracy = float("nan")
        f1_values = [
            _class_f1(label_id, true_count=true_count, pred_count=pred_count, correct_count=correct_count)
            for label_id in label_ids
        ]
        result[f"{group_name}_accuracy"] = float(accuracy)
        result[f"{group_name}_macro_f1"] = _nanmean(f1_values)
    return result


def _evaluate_tail_repair(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    label_buckets: Mapping[int, str],
    label_names: Mapping[int, str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    labels_all: list[int] = []
    preds_all: list[int] = []
    margin_chunks: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="tail_repair.eval.label_id")
            logits = forward_pooled_logits(model, batch).float()
            loss = F.cross_entropy(logits, labels)
            predictions = logits.detach().argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size
            labels_all.extend(int(item) for item in labels.detach().cpu().tolist())
            preds_all.extend(int(item) for item in predictions.detach().cpu().tolist())
            margin_chunks.append(correct_logit_margin(logits.detach(), labels).detach().cpu())
    if total_samples <= 0:
        raise RuntimeError("Tail repair eval loader produced no samples")
    margins = torch.cat(margin_chunks, dim=0) if margin_chunks else torch.empty(0)
    labels_tensor = torch.tensor(labels_all, dtype=torch.long)
    breakdown, per_class_rows = compute_eval_breakdown(
        {"labels": labels_all, "predictions": preds_all},
        label_names=label_names,
        label_buckets=label_buckets,
    )
    group_summary = _classification_summary(
        labels=labels_all,
        predictions=preds_all,
        label_names=label_names,
        label_buckets=label_buckets,
    )
    return {
        "loss": total_loss / float(total_samples),
        "accuracy": total_correct / float(total_samples),
        "num_samples": float(total_samples),
        "labels": labels_all,
        "predictions": preds_all,
        "eval_breakdown": breakdown,
        "per_class_rows": per_class_rows,
        "group_summary": group_summary,
        "correct_logit_margin_global": _mean_finite(margins),
        "correct_logit_margin_many": _mean_finite(margins[_bucket_mask(labels_tensor, label_buckets, "many")]),
        "correct_logit_margin_tail": _mean_finite(margins[_bucket_mask(labels_tensor, label_buckets, "tail")]),
    }


def _canonical_reload_eval_checkpoint(
    *,
    checkpoint_path: Path,
    split_name: str = "eval",
    args: argparse.Namespace,
    paths: Any,
    label_buckets: Mapping[int, str],
    label_names: Mapping[int, str],
    num_classes: int,
    device: torch.device,
    token_layer_indices: Sequence[int] = (),
    prediction_path: str = OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
    include_logits: bool = False,
) -> dict[str, Any]:
    model, _payload = load_trained_checkpoint(checkpoint_path, device=device)
    model.eval()
    loader = build_loader(
        paths,
        split=str(split_name),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )
    token_layers = tuple(int(item) for item in token_layer_indices)
    sample_ids: list[str] = []
    labels_chunks: list[Tensor] = []
    logits_chunks: list[Tensor] = []
    seen: set[str] = set()
    for batch in loader:
        batch_sample_ids = _batch_sample_ids_for_canonical_eval(batch)
        duplicates = sorted(set(batch_sample_ids) & seen)
        if duplicates:
            raise ValueError(f"duplicate sample_id values during canonical reload eval: {duplicates[:8]}")
        seen.update(batch_sample_ids)
        labels = _validate_label_tensor(
            batch["label_id"],
            num_classes=num_classes,
            context="tail_repair.canonical_reload_eval.label_id",
        ).detach().cpu().long().reshape(-1)
        if int(labels.numel()) != len(batch_sample_ids):
            raise ValueError("canonical reload eval batch label count differs from sample_id count")
        logits, _embeddings, _token_padding_mask, _valid_token_count, _records, _oer_activations = forward_export_batch(
            model,
            batch,
            sample_ids=batch_sample_ids,
            token_layer_indices=token_layers,
            device=device,
            oer_mode="none",
            oer_hook_list=(),
        )
        sample_ids.extend(batch_sample_ids)
        labels_chunks.append(labels.contiguous())
        logits_chunks.append(logits.detach().cpu().float().contiguous())
    if not sample_ids:
        raise RuntimeError(f"canonical reload eval produced no samples for split={split_name}: {checkpoint_path}")
    labels_all = torch.cat(labels_chunks, dim=0).long().contiguous()
    logits_all = torch.cat(logits_chunks, dim=0).float().contiguous()
    predictions_all = logits_all.argmax(dim=1).cpu().long().contiguous()
    loss = F.cross_entropy(logits_all, labels_all)
    labels_list = [int(item) for item in labels_all.tolist()]
    predictions_list = [int(item) for item in predictions_all.tolist()]
    breakdown, per_class_rows = compute_eval_breakdown(
        {"labels": labels_list, "predictions": predictions_list},
        label_names=label_names,
        label_buckets=label_buckets,
    )
    group_summary = _classification_summary(
        labels=labels_list,
        predictions=predictions_list,
        label_names=label_names,
        label_buckets=label_buckets,
    )
    margins = correct_logit_margin(logits_all, labels_all).detach().cpu().float().reshape(-1)
    result: dict[str, Any] = {
        "loss": float(loss.detach().item()),
        "accuracy": float((predictions_all == labels_all).float().mean().item()),
        "num_samples": float(labels_all.numel()),
        "sample_ids": list(sample_ids),
        "labels": labels_list,
        "predictions": predictions_list,
        "eval_breakdown": breakdown,
        "per_class_rows": per_class_rows,
        "group_summary": group_summary,
        "correct_logit_margin_global": _mean_finite(margins),
        "correct_logit_margin_many": _mean_finite(margins[_bucket_mask(labels_all, label_buckets, "many")]),
        "correct_logit_margin_medium": _mean_finite(margins[_bucket_mask(labels_all, label_buckets, "medium")]),
        "correct_logit_margin_few": _mean_finite(margins[_bucket_mask(labels_all, label_buckets, "few")]),
        "correct_logit_margin_tail": _mean_finite(margins[_bucket_mask(labels_all, label_buckets, "tail")]),
        "split": str(split_name),
        "prediction_path": str(prediction_path),
        "token_layer_indices": [int(item) for item in token_layers],
        "selection_prediction_path": str(prediction_path),
        "selection_checkpoint": path_str(checkpoint_path),
        "selection_checkpoint_sha256": file_sha256(checkpoint_path),
    }
    if include_logits:
        result["logits"] = logits_all
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _logit_diff_stats(first_logits: Tensor, second_logits: Tensor) -> dict[str, Any]:
    if tuple(first_logits.shape) != tuple(second_logits.shape):
        raise RuntimeError(f"logits shape differs: first={tuple(first_logits.shape)}, second={tuple(second_logits.shape)}")
    if int(first_logits.numel()) <= 0:
        return {
            "max_abs_logit_diff": 0.0,
            "max_rel_logit_diff": 0.0,
            "logit_atol": CANONICAL_RELOAD_LOGIT_ATOL,
            "logit_rtol": CANONICAL_RELOAD_LOGIT_RTOL,
            "logits_allclose": True,
        }
    first = first_logits.detach().cpu().float()
    second = second_logits.detach().cpu().float()
    abs_diff = (first - second).abs()
    denom = torch.maximum(torch.maximum(first.abs(), second.abs()), torch.full_like(abs_diff, float(DEFAULT_EPS)))
    rel_diff = abs_diff / denom
    return {
        "max_abs_logit_diff": float(abs_diff.max().item()),
        "max_rel_logit_diff": float(rel_diff.max().item()),
        "logit_atol": CANONICAL_RELOAD_LOGIT_ATOL,
        "logit_rtol": CANONICAL_RELOAD_LOGIT_RTOL,
        "logits_allclose": bool(
            torch.allclose(
                first,
                second,
                atol=float(CANONICAL_RELOAD_LOGIT_ATOL),
                rtol=float(CANONICAL_RELOAD_LOGIT_RTOL),
            )
        ),
    }


def _compare_reload_eval_payloads(
    *,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    first_logits: Tensor,
    second_logits: Tensor,
    context: str,
) -> dict[str, Any]:
    if list(first["sample_ids"]) != list(second["sample_ids"]):
        raise RuntimeError(f"{context} sample order differs between passes")
    if [int(item) for item in first["labels"]] != [int(item) for item in second["labels"]]:
        raise RuntimeError(f"{context} labels differ between passes")
    first_predictions = torch.tensor([int(item) for item in first["predictions"]], dtype=torch.long)
    second_predictions = torch.tensor([int(item) for item in second["predictions"]], dtype=torch.long)
    stats = _logit_diff_stats(first_logits, second_logits)
    stats["prediction_disagreement_count"] = int((first_predictions != second_predictions).sum().item())
    return stats


def _all_token_layer_indices_for_checkpoint(*, checkpoint_path: Path, device: torch.device) -> tuple[int, ...]:
    model, _payload = load_trained_checkpoint(checkpoint_path, device=device)
    try:
        layers = tuple(int(item) for item in resolve_full_layer_indices(model))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not layers:
        raise RuntimeError(f"token-hook sanity requires at least one traceable token layer: {checkpoint_path}")
    return layers


def _canonical_reload_sanity_check(
    *,
    checkpoint_path: Path,
    args: argparse.Namespace,
    paths: Any,
    label_buckets: Mapping[int, str],
    label_names: Mapping[int, str],
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    first = _canonical_reload_eval_checkpoint(
        checkpoint_path=checkpoint_path,
        args=args,
        paths=paths,
        label_buckets=label_buckets,
        label_names=label_names,
        num_classes=num_classes,
        device=device,
        include_logits=True,
    )
    second = _canonical_reload_eval_checkpoint(
        checkpoint_path=checkpoint_path,
        args=args,
        paths=paths,
        label_buckets=label_buckets,
        label_names=label_names,
        num_classes=num_classes,
        device=device,
        include_logits=True,
    )
    first_logits = torch.as_tensor(first.pop("logits")).detach().cpu().float()
    second_logits = torch.as_tensor(second.pop("logits")).detach().cpu().float()
    stats = _compare_reload_eval_payloads(
        first=first,
        second=second,
        first_logits=first_logits,
        second_logits=second_logits,
        context="canonical reload sanity check",
    )
    disagreement = int(stats["prediction_disagreement_count"])
    if disagreement > 0:
        raise RuntimeError(f"canonical reload sanity check prediction disagreement: {disagreement}")
    if not bool(stats["logits_allclose"]):
        raise RuntimeError(
            "canonical reload sanity check logits exceeded numerical tolerance: "
            f"max_abs={float(stats['max_abs_logit_diff']):.9g}, "
            f"max_rel={float(stats['max_rel_logit_diff']):.9g}, "
            f"atol={CANONICAL_RELOAD_LOGIT_ATOL:g}, rtol={CANONICAL_RELOAD_LOGIT_RTOL:g}"
        )
    return {
        "checkpoint": path_str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "path": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "reload_count": 2,
        "prediction_disagreement_count": disagreement,
        "max_abs_logit_diff": float(stats["max_abs_logit_diff"]),
        "max_rel_logit_diff": float(stats["max_rel_logit_diff"]),
        "logit_atol": float(stats["logit_atol"]),
        "logit_rtol": float(stats["logit_rtol"]),
        "logits_allclose": bool(stats["logits_allclose"]),
        "first_tail_macro_f1": float(first["group_summary"]["tail_macro_f1"]),
        "second_tail_macro_f1": float(second["group_summary"]["tail_macro_f1"]),
        "first_tail_accuracy": float(first["group_summary"]["tail_accuracy"]),
        "second_tail_accuracy": float(second["group_summary"]["tail_accuracy"]),
    }


def _token_hook_sanity_check(
    *,
    checkpoint_path: Path,
    args: argparse.Namespace,
    paths: Any,
    label_buckets: Mapping[int, str],
    label_names: Mapping[int, str],
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    token_layers = _all_token_layer_indices_for_checkpoint(checkpoint_path=checkpoint_path, device=device)
    canonical = _canonical_reload_eval_checkpoint(
        checkpoint_path=checkpoint_path,
        args=args,
        paths=paths,
        label_buckets=label_buckets,
        label_names=label_names,
        num_classes=num_classes,
        device=device,
        token_layer_indices=(),
        prediction_path=OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        include_logits=True,
    )
    token_hook = _canonical_reload_eval_checkpoint(
        checkpoint_path=checkpoint_path,
        args=args,
        paths=paths,
        label_buckets=label_buckets,
        label_names=label_names,
        num_classes=num_classes,
        device=device,
        token_layer_indices=token_layers,
        prediction_path=DIAGNOSTIC_TOKEN_HOOK_METRIC_SOURCE,
        include_logits=True,
    )
    canonical_logits = torch.as_tensor(canonical.pop("logits")).detach().cpu().float()
    token_hook_logits = torch.as_tensor(token_hook.pop("logits")).detach().cpu().float()
    stats = _compare_reload_eval_payloads(
        first=canonical,
        second=token_hook,
        first_logits=canonical_logits,
        second_logits=token_hook_logits,
        context="canonical vs token-hook sanity check",
    )
    warning = ""
    if int(stats["prediction_disagreement_count"]) > 0:
        warning = "token_hook_prediction_disagreement"
    elif not bool(stats["logits_allclose"]):
        warning = "token_hook_logit_diff_exceeds_tolerance"
    return {
        "checkpoint": path_str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "canonical_path": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "token_hook_path": DIAGNOSTIC_TOKEN_HOOK_METRIC_SOURCE,
        "token_export": "all",
        "token_layer_indices": [int(item) for item in token_layers],
        "prediction_disagreement_count": int(stats["prediction_disagreement_count"]),
        "max_abs_logit_diff": float(stats["max_abs_logit_diff"]),
        "max_rel_logit_diff": float(stats["max_rel_logit_diff"]),
        "logit_atol": float(stats["logit_atol"]),
        "logit_rtol": float(stats["logit_rtol"]),
        "logits_allclose": bool(stats["logits_allclose"]),
        "warning": warning,
        "canonical_tail_macro_f1": float(canonical["group_summary"]["tail_macro_f1"]),
        "token_hook_tail_macro_f1": float(token_hook["group_summary"]["tail_macro_f1"]),
        "canonical_tail_accuracy": float(canonical["group_summary"]["tail_accuracy"]),
        "token_hook_tail_accuracy": float(token_hook["group_summary"]["tail_accuracy"]),
    }


def _payload_string_sequence(payload: Mapping[str, Any], keys: Sequence[str], *, context: str) -> list[str]:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, Tensor):
            return [str(item) for item in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [str(item) for item in value]
        raise TypeError(f"{context} field {key!r} must be a sequence")
    raise KeyError(f"{context} missing any of fields: {list(keys)}")


def _payload_long_vector(payload: Mapping[str, Any], keys: Sequence[str], *, context: str) -> Tensor:
    for key in keys:
        if key not in payload:
            continue
        return torch.as_tensor(payload[key]).detach().cpu().long().reshape(-1).contiguous()
    raise KeyError(f"{context} missing any of fields: {list(keys)}")


def _export_logits_artifact_payload(export_result: Mapping[str, Any], *, split_name: str) -> tuple[Path, Mapping[str, Any]]:
    split_result = export_result.get("split")
    if not isinstance(split_result, Mapping):
        split_result = export_result
    paths = split_result.get("paths") if isinstance(split_result, Mapping) else None
    if not isinstance(paths, Mapping):
        raise KeyError(f"export parity sanity missing artifact paths for split={split_name}")
    logits_value = paths.get("logits")
    if not logits_value:
        raise KeyError(f"export parity sanity missing logits artifact path for split={split_name}")
    logits_path = Path(str(logits_value))
    payload = _load_tensor_payload(logits_path)
    return logits_path, payload


def _prediction_mismatch_examples(
    *,
    sample_ids: Sequence[str],
    labels: Tensor,
    canonical_logits: Tensor,
    export_logits: Tensor,
    canonical_predictions: Tensor,
    export_predictions: Tensor,
    limit: int = 32,
) -> list[dict[str, Any]]:
    mismatch_indices = torch.nonzero(canonical_predictions != export_predictions, as_tuple=False).reshape(-1).tolist()
    if not mismatch_indices:
        return []
    canonical_margins = correct_logit_margin(canonical_logits, labels).detach().cpu().float().reshape(-1)
    export_margins = correct_logit_margin(export_logits, labels).detach().cpu().float().reshape(-1)

    def boundary_key(index: int) -> tuple[float, int]:
        margin = min(abs(float(canonical_margins[index].item())), abs(float(export_margins[index].item())))
        return (margin, int(index))

    rows: list[dict[str, Any]] = []
    for index in sorted((int(item) for item in mismatch_indices), key=boundary_key)[: int(limit)]:
        sample_logit_abs_diff = (canonical_logits[index] - export_logits[index]).detach().cpu().float().abs()
        rows.append(
            {
                "sample_index": int(index),
                "sample_id": str(sample_ids[index]),
                "label_id": int(labels[index].item()),
                "canonical_prediction": int(canonical_predictions[index].item()),
                "export_prediction": int(export_predictions[index].item()),
                "canonical_correct_logit_margin": float(canonical_margins[index].item()),
                "export_correct_logit_margin": float(export_margins[index].item()),
                "boundary_abs_margin_min": float(boundary_key(index)[0]),
                "max_abs_logit_diff_for_sample": float(sample_logit_abs_diff.max().item()),
            }
        )
    return rows


def _export_parity_sanity_check(
    *,
    checkpoint_path: Path,
    split_name: str,
    export_result: Mapping[str, Any],
    args: argparse.Namespace,
    paths: Any,
    label_buckets: Mapping[int, str],
    label_names: Mapping[int, str],
    num_classes: int,
    device: torch.device,
    hard_fail_on_prediction_mismatch: bool,
) -> dict[str, Any]:
    canonical = _canonical_reload_eval_checkpoint(
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        args=args,
        paths=paths,
        label_buckets=label_buckets,
        label_names=label_names,
        num_classes=num_classes,
        device=device,
        token_layer_indices=(),
        prediction_path=OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        include_logits=True,
    )
    logits_path, export_payload = _export_logits_artifact_payload(export_result, split_name=split_name)
    canonical_logits = torch.as_tensor(canonical.pop("logits")).detach().cpu().float().contiguous()
    export_logits = _payload_tensor(export_payload, ("logits",), name=f"{split_name}.export_parity.logits").float().contiguous()
    export_sample_ids = _payload_string_sequence(export_payload, ("sample_ids", "sample_id"), context=f"{split_name}.export_logits")
    export_labels = _payload_long_vector(export_payload, ("labels", "label_id"), context=f"{split_name}.export_logits")
    if export_logits.ndim != 2:
        raise ValueError(f"{split_name} export parity logits must be [N,C], got shape={tuple(export_logits.shape)}")
    if int(export_logits.shape[0]) != len(export_sample_ids):
        raise ValueError(f"{split_name} export parity logits row count differs from sample_id count")
    if int(export_labels.numel()) != len(export_sample_ids):
        raise ValueError(f"{split_name} export parity label count differs from sample_id count")
    export_predictions = export_logits.argmax(dim=1).detach().cpu().long().contiguous()
    export_eval = {
        "sample_ids": export_sample_ids,
        "labels": [int(item) for item in export_labels.tolist()],
        "predictions": [int(item) for item in export_predictions.tolist()],
    }
    stats = _compare_reload_eval_payloads(
        first=canonical,
        second=export_eval,
        first_logits=canonical_logits,
        second_logits=export_logits,
        context=f"{split_name} export parity sanity check",
    )
    canonical_predictions = torch.tensor([int(item) for item in canonical["predictions"]], dtype=torch.long)
    mismatch_count = int(stats["prediction_disagreement_count"])
    num_samples = int(export_labels.numel())
    examples = _prediction_mismatch_examples(
        sample_ids=export_sample_ids,
        labels=export_labels,
        canonical_logits=canonical_logits,
        export_logits=export_logits,
        canonical_predictions=canonical_predictions,
        export_predictions=export_predictions,
    )
    warning = ""
    if mismatch_count > 0:
        warning = "export_prediction_mismatch"
    elif float(stats["max_abs_logit_diff"]) > float(CANONICAL_RELOAD_LOGIT_ATOL):
        warning = "export_logit_diff_exceeds_abs_tolerance"
    if mismatch_count > 0 and hard_fail_on_prediction_mismatch:
        raise RuntimeError(
            f"{split_name} final export parity prediction mismatch: "
            f"count={mismatch_count}, logits_path={logits_path}, examples={examples[:3]}"
        )
    return {
        "checkpoint": path_str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "split": str(split_name),
        "source": EXPORT_PARITY_METRIC_SOURCE,
        "canonical_path": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "export_path": path_str(logits_path),
        "export_path_sha256": file_sha256(logits_path),
        "num_samples": int(num_samples),
        "sample_identity_match": True,
        "label_sequence_match": True,
        "prediction_mismatch_count": int(mismatch_count),
        "n_prediction_mismatch": int(mismatch_count),
        "prediction_mismatch_rate": float(mismatch_count / float(num_samples)) if num_samples > 0 else float("nan"),
        "max_abs_logit_diff": float(stats["max_abs_logit_diff"]),
        "max_rel_logit_diff": float(stats["max_rel_logit_diff"]),
        "logit_atol": float(stats["logit_atol"]),
        "logit_rtol": float(stats["logit_rtol"]),
        "logits_allclose": bool(stats["logits_allclose"]),
        "warning": warning,
        "hard_fail_on_prediction_mismatch": bool(hard_fail_on_prediction_mismatch),
        "prediction_mismatch_examples": examples,
    }


def _compute_gamma_t(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    label_buckets: Mapping[int, str],
) -> dict[str, Any]:
    model.eval()
    tail_margins: list[float] = []
    total_samples = 0
    tail_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="tail_repair.gamma.label_id")
            logits = forward_pooled_logits(model, batch).float()
            margins = correct_logit_margin(logits, labels)
            mask = _tail_mask(labels, label_buckets)
            total_samples += int(labels.numel())
            tail_samples += int(mask.sum().item())
            tail_margins.extend(float(item) for item in margins[mask].detach().cpu().tolist())
    if not tail_margins:
        raise RuntimeError("cannot compute gamma_T: train split has no tail samples under train-derived label buckets")
    median_margin = _median_finite(tail_margins)
    if not math.isfinite(median_margin):
        raise RuntimeError("cannot compute gamma_T: Stage1 train tail margins are all NaN/Inf")
    gamma_t = max(0.0, float(median_margin))
    return {
        "gamma_t": gamma_t,
        "stage1_train_tail_margin_median": float(median_margin),
        "stage1_train_tail_sample_count": int(tail_samples),
        "stage1_train_sample_count": int(total_samples),
        "margin_name": "correct_logit_margin",
        "margin_definition": "logit_y - max_{j != y} logit_j",
        "gamma_definition": "max(0, median(correct_logit_margin over Stage1 train tail samples))",
    }


def train_tail_repair_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    num_classes: int,
    label_buckets: Mapping[int, str],
    gamma_t: float,
    lambda_tail_margin: float,
) -> dict[str, float]:
    model.eval()
    if hasattr(model, "predictor"):
        model.predictor.train()
    if hasattr(model, "predictor_dropout"):
        model.predictor_dropout.train()
    total_loss = 0.0
    total_ce = 0.0
    total_tail_margin_batch_weighted = 0.0
    total_tail_margin_tail_sample_weighted = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0
    tail_batches = 0
    tail_samples = 0
    for batch in loader:
        total_batches += 1
        batch = move_batch_to_device(batch, device)
        labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="tail_repair.train.label_id")
        optimizer.zero_grad(set_to_none=True)
        logits = forward_pooled_logits(model, batch).float()
        ce_loss = F.cross_entropy(logits, labels)
        margins = correct_logit_margin(logits, labels)
        tail_mask = _tail_mask(labels, label_buckets)
        batch_tail_count = int(tail_mask.sum().item())
        if bool(tail_mask.any().item()):
            tail_margin_loss = torch.relu(logits.new_tensor(float(gamma_t)) - margins[tail_mask]).mean()
            tail_batches += 1
            tail_samples += batch_tail_count
            total_tail_margin_tail_sample_weighted += float(tail_margin_loss.detach().item()) * batch_tail_count
        else:
            tail_margin_loss = logits.new_tensor(0.0)
        loss = ce_loss + float(lambda_tail_margin) * tail_margin_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        predictions = logits.detach().argmax(dim=1)
        batch_size = int(labels.numel())
        total_samples += batch_size
        total_correct += int((predictions == labels).sum().item())
        total_loss += float(loss.detach().item()) * batch_size
        total_ce += float(ce_loss.detach().item()) * batch_size
        total_tail_margin_batch_weighted += float(tail_margin_loss.detach().item()) * batch_size
    if total_samples <= 0:
        raise RuntimeError("Tail repair train loader produced no samples")
    tail_sample_mean = (
        total_tail_margin_tail_sample_weighted / float(tail_samples)
        if tail_samples > 0
        else 0.0
    )
    return {
        "train_loss": total_loss / float(total_samples),
        "train_ce_loss": total_ce / float(total_samples),
        "train_tail_margin_loss": tail_sample_mean,
        "train_tail_margin_loss_tail_sample_mean": tail_sample_mean,
        "train_tail_margin_loss_batch_weighted": total_tail_margin_batch_weighted / float(total_samples),
        "train_kd_loss": 0.0,
        "train_accuracy": total_correct / float(total_samples),
        "train_num_samples": float(total_samples),
        "train_num_batches": float(total_batches),
        "train_tail_batches": float(tail_batches),
        "train_tail_samples": float(tail_samples),
        "teacher_forward_count": 0.0,
    }


def _score_eval(eval_payload: Mapping[str, Any]) -> tuple[float, float, float, float]:
    summary = eval_payload.get("group_summary")
    if not isinstance(summary, Mapping):
        raise KeyError("eval payload missing group_summary")
    return (
        float(summary.get("tail_macro_f1", float("nan"))),
        float(summary.get("tail_accuracy", float("nan"))),
        float(summary.get("medium_macro_f1", float("nan"))),
        float(summary.get("medium_accuracy", float("nan"))),
    )


def _safe_score(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("-inf")


def _is_better_eval(current: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    current_score = tuple(_safe_score(item) for item in _score_eval(current))
    best_score = tuple(_safe_score(item) for item in _score_eval(best))
    return current_score > best_score


def _save_tail_repair_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    source_payload: Mapping[str, Any],
    source_stage1_checkpoint: Path,
    stage_epoch: int,
    global_epoch: int,
    eval_metrics: Mapping[str, Any],
    eval_metrics_source: str,
    lambda_tail_margin: float,
    gamma_payload: Mapping[str, Any],
    stage_contract: Mapping[str, Any],
    quant_audit: Mapping[str, Any],
    best_metric_so_far: float,
) -> dict[str, Any]:
    identity_payload = {
        "protocol_version": TAIL_REPAIR_PROTOCOL_VERSION,
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "model_name": MODEL_NAME,
        "source_stage1_checkpoint_sha256": file_sha256(source_stage1_checkpoint),
        "source_stage1_run_identity_hash": source_payload.get("run_identity_hash"),
        "teacher_checkpoint_sha256": source_payload.get("teacher_checkpoint_sha256"),
        "qparams_hash": source_payload.get("qparams_hash"),
        "quantized_layers_hash": source_payload.get("quantized_layers_hash"),
        "label_mapping_hash": source_payload.get("label_mapping_hash"),
        "stage_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "lambda_tail_margin": float(lambda_tail_margin),
        "gamma_t": float(gamma_payload["gamma_t"]),
        "seed": int(source_payload.get("seed", 0)),
    }
    run_identity_hash = canonical_json_hash(identity_payload)
    payload = {str(key): value for key, value in dict(source_payload).items() if not str(key).startswith("_")}
    payload.update(
        {
            "protocol_version": TAIL_REPAIR_PROTOCOL_VERSION,
            "artifact_type": "innovation_tail_repair_stage2_checkpoint",
            "baseline": BASELINE_NAME,
            "model_kind": MODEL_KIND,
            "method_variant": METHOD_VARIANT,
            "model_name": MODEL_NAME,
            "innovation_name": BASELINE_NAME,
            "stage_name": MODEL_NAME,
            "stage_index": 2,
            "stage_objective": "classifier_only_ce_plus_tail_correct_logit_margin_hinge",
            "epoch": int(global_epoch),
            "stage_epoch": int(stage_epoch),
            "selection_metric": SELECTION_METRIC,
            "best_metric_so_far": float(best_metric_so_far),
            "eval_metrics": dict(eval_metrics),
            "eval_metrics_source": str(eval_metrics_source),
            "model_state_dict": model.state_dict(),
            "source_stage1_checkpoint": path_str(source_stage1_checkpoint),
            "source_stage1_checkpoint_sha256": file_sha256(source_stage1_checkpoint),
            "source_stage1_run_identity_hash": source_payload.get("run_identity_hash"),
            "tail_margin_loss": {
                "lambda_tail_margin": float(lambda_tail_margin),
                "gamma_t": float(gamma_payload["gamma_t"]),
                "margin_name": "correct_logit_margin",
                "margin_definition": "logit_y - max_{j != y} logit_j",
                "hinge_power": 1,
                "formula": "CE(all samples) + lambda * mean_tail(max(0, gamma_T - m_i))",
                "empty_tail_batch_loss": 0.0,
            },
            "gamma_t_payload": dict(gamma_payload),
            "stage2_contract": dict(stage_contract),
            "training_quant_audit": dict(quant_audit),
            "teacher_forward_count": 0,
            "fp32_teacher_loaded_for_training": False,
            "kd_enabled": False,
            "qparams_frozen": True,
            "student_backbone_trainable": False,
            "classifier_head_only": True,
            "run_identity_hash": run_identity_hash,
            "run_identity_payload": identity_payload,
            "load_note": "Load through src.baseline.export_eval.load_trained_checkpoint; INT4 qparams are inherited from the fixed Stage1 checkpoint.",
        }
    )
    for parameter in model.parameters():
        parameter.grad = None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _export_split(
    *,
    checkpoint_path: Path,
    split_name: str,
    split_csv: Path,
    output_dir: Path,
    label_mapping_path: Path,
    label_buckets: Mapping[int, str],
    args: argparse.Namespace,
    reference_root: Path,
) -> dict[str, Any]:
    return run_export_eval_config(
        ExportEvalConfig(
            model_name=str(args.export_name),
            checkpoint_path=checkpoint_path,
            split_csv=split_csv,
            split_name=split_name,
            output_dir=output_dir,
            label_mapping_path=label_mapping_path,
            bucket_protocol=label_buckets,
            pooling="mask_aware_mean",
            token_export="all",
            token_layers=DEFAULT_TOKEN_LAYERS,
            device=args.device,
            seed=int(args.seed),
            dataset_root=dataset_paths(args.dataset, None).root,
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
            amp=False,
            export_all_token_layers=True,
            max_export_payload_gib=float(args.max_export_payload_gib),
            reference_root=reference_root,
        )
    )


def _load_tensor_payload(path: str | Path) -> Mapping[str, Any]:
    target = Path(path)
    payload = torch.load(target, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"tensor artifact must contain a mapping: {target}")
    return payload


def _payload_tensor(payload: Mapping[str, Any], keys: Sequence[str], *, name: str) -> Tensor:
    for key in keys:
        if key in payload:
            tensor = torch.as_tensor(payload[key]).detach().cpu().float().contiguous()
            if tensor.ndim != 2:
                raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}")
            return tensor
    raise KeyError(f"payload missing tensor key for {name}: {keys}")


def _payload_labels(payload: Mapping[str, Any]) -> Tensor:
    for key in ("labels", "label_id"):
        if key in payload:
            labels = torch.as_tensor(payload[key]).detach().cpu().long().reshape(-1)
            if int(labels.numel()) <= 0:
                raise ValueError("payload labels are empty")
            return labels
    raise KeyError("payload missing labels/label_id")


def _payload_sample_ids(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("sample_ids", payload.get("sample_id"))
    if isinstance(raw, Tensor):
        return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    raise KeyError("payload missing sample_ids/sample_id")


def _variance_deviation_metrics(split_result: Mapping[str, Any], *, reference_root: Path) -> dict[str, float | int]:
    paths = split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("split result missing paths")
    embedding_payload = _load_tensor_payload(paths["embeddings"])
    labels = _payload_labels(embedding_payload)
    embeddings = _payload_tensor(embedding_payload, ("embeddings", "embedding"), name="candidate embeddings")
    reference = load_fp32_reference_bundle(reference_root, split=str(split_result.get("split") or "eval"))
    if _payload_sample_ids(embedding_payload) != [str(item) for item in reference.sample_ids]:
        raise ValueError("candidate embeddings and FP32 reference sample order differ")
    if tuple(labels.shape) != tuple(reference.labels.shape) or not bool(torch.equal(labels, reference.labels.long().reshape(-1))):
        raise ValueError("candidate embeddings labels differ from FP32 reference labels")
    ref_var, ref_details = class_trace_variance(reference.embeddings, labels, normalization="dof", reduction="none", return_details=True)
    cur_var, cur_details = class_trace_variance(embeddings, labels, normalization="dof", reduction="none", return_details=True)
    ref_ids = torch.as_tensor(ref_details["class_ids"]).detach().cpu().long()
    cur_ids = torch.as_tensor(cur_details["class_ids"]).detach().cpu().long()
    if tuple(ref_ids.tolist()) != tuple(cur_ids.tolist()):
        raise ValueError("candidate and FP32 reference class variance ids differ")
    ref_values = torch.as_tensor(ref_var).detach().cpu().float()
    cur_values = torch.as_tensor(cur_var).detach().cpu().float()
    valid_input = torch.isfinite(ref_values) & torch.isfinite(cur_values)
    ratio = cur_values / torch.clamp(ref_values, min=float(DEFAULT_EPS))
    deviation = torch.abs(torch.log(ratio))
    selected = deviation[valid_input]
    nonfinite_count = int((~torch.isfinite(selected)).sum().item()) if int(selected.numel()) else 0
    if int(selected.numel()) <= 0:
        value = float("nan")
    elif nonfinite_count > 0:
        value = float("inf")
    else:
        value = _mean_finite(selected)
    return {
        "variance_deviation_abs_log": float(value),
        "variance_deviation_nonfinite_class_count": int(nonfinite_count),
    }


def _few_collapse_gap(split_result: Mapping[str, Any], *, reference_root: Path, label_buckets: Mapping[int, str]) -> dict[str, float]:
    paths = split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("split result missing paths")
    embedding_payload = _load_tensor_payload(paths["embeddings"])
    labels = _payload_labels(embedding_payload)
    embeddings = _payload_tensor(embedding_payload, ("embeddings", "embedding"), name="candidate embeddings")
    reference = load_fp32_reference_bundle(reference_root, split=str(split_result.get("split") or "eval"))
    if _payload_sample_ids(embedding_payload) != [str(item) for item in reference.sample_ids]:
        raise ValueError("candidate embeddings and FP32 reference sample order differ")
    distortion = embedding_distortion(reference.embeddings, embeddings, reduction="none")
    buckets = [str(label_buckets.get(int(label), "unknown")) for label in labels.detach().cpu().long().tolist()]
    gap_payload = tail_and_few_damage_gaps(distortion, buckets, higher_is_better=False)
    return {
        "tail_collapse_gap": float(torch.as_tensor(gap_payload["tail_vs_many_gap"]).item()),
        "few_collapse_gap": float(torch.as_tensor(gap_payload["few_vs_many_gap"]).item()),
    }


def _export_metric_summary(
    *,
    split_result: Mapping[str, Any],
    reference_root: Path,
    label_buckets: Mapping[int, str],
) -> dict[str, float]:
    paths = split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("split result missing paths")
    metrics = _read_csv_rows(paths["metrics"])
    group_metrics = _read_csv_rows(paths["group_metrics"])
    geometry = _read_csv_rows(paths["geometry_metrics"])
    collapse = _few_collapse_gap(split_result, reference_root=reference_root, label_buckets=label_buckets)
    variance = _variance_deviation_metrics(split_result, reference_root=reference_root)
    return {
        "overall_accuracy": _metric_value(metrics, metric_name="accuracy", analysis_group="global"),
        "overall_macro_f1": _metric_value(metrics, metric_name="macro_f1_label_universe_nanmean", analysis_group="global"),
        "many_accuracy": _metric_value(group_metrics, metric_name="accuracy", analysis_group="many"),
        "many_macro_f1": _metric_value(group_metrics, metric_name="macro_f1_label_universe_nanmean", analysis_group="many"),
        "medium_accuracy": _metric_value(group_metrics, metric_name="accuracy", analysis_group="medium"),
        "medium_macro_f1": _metric_value(group_metrics, metric_name="macro_f1_label_universe_nanmean", analysis_group="medium"),
        "tail_accuracy": _metric_value(group_metrics, metric_name="accuracy", analysis_group="tail"),
        "tail_macro_f1": _metric_value(group_metrics, metric_name="macro_f1_label_universe_nanmean", analysis_group="tail"),
        "correct_logit_margin_global": _metric_value(geometry, metric_name="correct_logit_margin_detail", analysis_group="global"),
        "correct_logit_margin_many": _metric_value(geometry, metric_name="correct_logit_margin_detail", analysis_group="many"),
        "correct_logit_margin_tail": _metric_value(geometry, metric_name="correct_logit_margin_detail", analysis_group="tail"),
        "logit_margin_retention": _metric_value(metrics, metric_name="logit_margin_retention", analysis_group="global", default=float("nan")),
        "fisher": _metric_value(geometry, metric_name="fisher", analysis_group="global"),
        "geo_margin": _metric_value(geometry, metric_name="geo_margin_main", analysis_group="global"),
        "class_trace_variance": _metric_value(geometry, metric_name="class_trace_variance", analysis_group="global"),
        "embedding_distortion": _metric_value(metrics, metric_name="embedding_distortion", analysis_group="global", default=float("nan")),
        "embedding_cosine_similarity": _metric_value(metrics, metric_name="embedding_cosine_similarity", analysis_group="global", default=float("nan")),
        "fisher_retention": _metric_value(metrics, metric_name="fisher_retention", analysis_group="global", default=float("nan")),
        "geo_margin_retention": _metric_value(metrics, metric_name="geo_margin_retention", analysis_group="global", default=float("nan")),
        "variance_deviation_abs_log": float(variance["variance_deviation_abs_log"]),
        "variance_deviation_nonfinite_class_count": int(variance["variance_deviation_nonfinite_class_count"]),
        "tail_collapse_gap": collapse["tail_collapse_gap"],
        "few_collapse_gap": collapse["few_collapse_gap"],
    }


def _canonical_official_metric_summary(eval_payload: Mapping[str, Any]) -> dict[str, float]:
    group = eval_payload["group_summary"]
    return {
        "overall_accuracy": float(eval_payload["accuracy"]),
        "overall_macro_f1": float(eval_payload["eval_breakdown"]["eval_macro_f1_label_universe_nanmean"]),
        "many_accuracy": float(group["many_accuracy"]),
        "many_macro_f1": float(group["many_macro_f1"]),
        "medium_accuracy": float(group["medium_accuracy"]),
        "medium_macro_f1": float(group["medium_macro_f1"]),
        "tail_accuracy": float(group["tail_accuracy"]),
        "tail_macro_f1": float(group["tail_macro_f1"]),
        "m_tail": float(eval_payload["correct_logit_margin_tail"]),
        "m_many": float(eval_payload["correct_logit_margin_many"]),
        "correct_logit_margin_global": float(eval_payload["correct_logit_margin_global"]),
        "correct_logit_margin_many": float(eval_payload["correct_logit_margin_many"]),
        "correct_logit_margin_medium": float(eval_payload.get("correct_logit_margin_medium", float("nan"))),
        "correct_logit_margin_few": float(eval_payload.get("correct_logit_margin_few", float("nan"))),
        "correct_logit_margin_tail": float(eval_payload["correct_logit_margin_tail"]),
    }


def _group_support_count(labels: Sequence[int], label_buckets: Mapping[int, str], group_name: str) -> int:
    if group_name == "global":
        return len(labels)
    if group_name == "tail":
        return sum(1 for label in labels if str(label_buckets.get(int(label), "unknown")) in {"medium", "few"})
    return sum(1 for label in labels if str(label_buckets.get(int(label), "unknown")) == str(group_name))


def _write_canonical_final_metric_tables(
    *,
    split_evals: Mapping[str, Mapping[str, Any]],
    checkpoint_path: Path,
    label_buckets: Mapping[int, str],
    metrics_path: Path,
    group_metrics_path: Path,
    geometry_metrics_path: Path,
) -> None:
    checkpoint_hash = file_sha256(checkpoint_path)
    metrics_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for split_name, eval_payload in split_evals.items():
        labels = [int(item) for item in eval_payload["labels"]]
        official = _canonical_official_metric_summary(eval_payload)
        metrics_rows.extend(
            [
                {
                    "split": str(split_name),
                    "metric_name": "loss",
                    "analysis_group": "global",
                    "metric_value": float(eval_payload["loss"]),
                    "metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
                    "support_count": len(labels),
                    "checkpoint_sha256": checkpoint_hash,
                },
                {
                    "split": str(split_name),
                    "metric_name": "accuracy",
                    "analysis_group": "global",
                    "metric_value": official["overall_accuracy"],
                    "metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
                    "support_count": len(labels),
                    "checkpoint_sha256": checkpoint_hash,
                },
                {
                    "split": str(split_name),
                    "metric_name": "macro_f1_label_universe_nanmean",
                    "analysis_group": "global",
                    "metric_value": official["overall_macro_f1"],
                    "metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
                    "support_count": len(labels),
                    "checkpoint_sha256": checkpoint_hash,
                },
            ]
        )
        for group_name in ("many", "medium", "few", "tail"):
            support_count = _group_support_count(labels, label_buckets, group_name)
            group_rows.append(
                {
                    "split": str(split_name),
                    "metric_name": "accuracy",
                    "analysis_group": group_name,
                    "metric_value": float(eval_payload["group_summary"][f"{group_name}_accuracy"]),
                    "metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
                    "support_count": support_count,
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
            group_rows.append(
                {
                    "split": str(split_name),
                    "metric_name": "macro_f1_label_universe_nanmean",
                    "analysis_group": group_name,
                    "metric_value": float(eval_payload["group_summary"][f"{group_name}_macro_f1"]),
                    "metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
                    "support_count": support_count,
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
        for group_name in ("global", "many", "medium", "few", "tail"):
            geometry_rows.append(
                {
                    "split": str(split_name),
                    "metric_name": "correct_logit_margin_detail",
                    "analysis_group": group_name,
                    "metric_value": float(eval_payload[f"correct_logit_margin_{group_name}"]),
                    "metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
                    "support_count": _group_support_count(labels, label_buckets, group_name),
                    "checkpoint_sha256": checkpoint_hash,
                    "definition": "mean correct-class logit margin from canonical no-token-hook inference",
                }
            )
    _write_csv(metrics_path, metrics_rows)
    _write_csv(group_metrics_path, group_rows)
    _write_csv(geometry_metrics_path, geometry_rows)


def _canonical_eval_manifest_summary(eval_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "split": str(eval_payload.get("split", "eval")),
        "checkpoint": str(eval_payload.get("selection_checkpoint") or ""),
        "checkpoint_sha256": str(eval_payload.get("selection_checkpoint_sha256") or ""),
        "metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "num_samples": float(eval_payload["num_samples"]),
        "loss": float(eval_payload["loss"]),
        "accuracy": float(eval_payload["accuracy"]),
        "eval_breakdown": dict(eval_payload["eval_breakdown"]),
        "group_summary": dict(eval_payload["group_summary"]),
        "correct_logit_margin_global": float(eval_payload["correct_logit_margin_global"]),
        "correct_logit_margin_many": float(eval_payload["correct_logit_margin_many"]),
        "correct_logit_margin_medium": float(eval_payload.get("correct_logit_margin_medium", float("nan"))),
        "correct_logit_margin_few": float(eval_payload.get("correct_logit_margin_few", float("nan"))),
        "correct_logit_margin_tail": float(eval_payload["correct_logit_margin_tail"]),
        "sample_order_hash": canonical_json_hash({"sample_ids": [str(item) for item in eval_payload["sample_ids"]]}),
    }


def _candidate_eval_metrics_row(
    *,
    seed: int,
    lambda_tail_margin: float,
    gamma_payload: Mapping[str, Any],
    stage_epoch: int,
    global_epoch: int,
    checkpoint_path: Path,
    train_row: Mapping[str, Any],
    quick_eval: Mapping[str, Any],
    selection_eval: Mapping[str, Any],
    selection_sanity: Mapping[str, Any],
    token_hook_sanity: Mapping[str, Any],
    export_parity_sanity: Mapping[str, Any],
    export_summary: Mapping[str, float],
    export_result: Mapping[str, Any],
) -> dict[str, Any]:
    selection_group = selection_eval["group_summary"]
    official_summary = _canonical_official_metric_summary(selection_eval)
    return {
        "seed": int(seed),
        "stage": MODEL_NAME,
        "stage2_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "stage2_checkpoint": path_str(checkpoint_path),
        "stage2_checkpoint_sha256": file_sha256(checkpoint_path),
        "stage2_selection_checkpoint": str(selection_eval.get("selection_checkpoint") or ""),
        "stage2_selection_metric": SELECTION_METRIC,
        "stage2_selection_metric_value": float(selection_group["tail_macro_f1"]),
        "stage2_checkpoint_selection_metric_value": float(selection_group["tail_macro_f1"]),
        "stage2_checkpoint_selection_metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "stage2_selection_checkpoint_sha256": str(selection_eval.get("selection_checkpoint_sha256") or ""),
        "stage2_selection_eval_accuracy": float(selection_eval["accuracy"]),
        "stage2_selection_eval_macro_f1_label_universe_nanmean": float(selection_eval["eval_breakdown"]["eval_macro_f1_label_universe_nanmean"]),
        "stage2_selection_eval_many_accuracy": float(selection_group["many_accuracy"]),
        "stage2_selection_eval_many_macro_f1": float(selection_group["many_macro_f1"]),
        "stage2_selection_eval_medium_accuracy": float(selection_group["medium_accuracy"]),
        "stage2_selection_eval_medium_macro_f1": float(selection_group["medium_macro_f1"]),
        "stage2_selection_eval_tail_accuracy": float(selection_group["tail_accuracy"]),
        "stage2_selection_eval_tail_macro_f1": float(selection_group["tail_macro_f1"]),
        "selection_reload_sanity_pred_disagreement_count": int(selection_sanity["prediction_disagreement_count"]),
        "selection_reload_sanity_max_abs_logit_diff": float(selection_sanity["max_abs_logit_diff"]),
        "selection_reload_sanity_max_rel_logit_diff": float(selection_sanity["max_rel_logit_diff"]),
        "selection_reload_sanity_logits_allclose": bool(selection_sanity["logits_allclose"]),
        "selection_token_hook_sanity_pred_disagreement_count": int(token_hook_sanity["prediction_disagreement_count"]),
        "selection_token_hook_sanity_max_abs_logit_diff": float(token_hook_sanity["max_abs_logit_diff"]),
        "selection_token_hook_sanity_max_rel_logit_diff": float(token_hook_sanity["max_rel_logit_diff"]),
        "selection_token_hook_sanity_logits_allclose": bool(token_hook_sanity["logits_allclose"]),
        "selection_token_hook_sanity_warning": str(token_hook_sanity.get("warning") or ""),
        "selection_export_parity_sanity_prediction_mismatch_count": int(export_parity_sanity["prediction_mismatch_count"]),
        "selection_export_parity_sanity_n_prediction_mismatch": int(export_parity_sanity["n_prediction_mismatch"]),
        "selection_export_parity_sanity_prediction_mismatch_rate": float(export_parity_sanity["prediction_mismatch_rate"]),
        "selection_export_parity_sanity_sample_identity_match": bool(export_parity_sanity["sample_identity_match"]),
        "selection_export_parity_sanity_label_sequence_match": bool(export_parity_sanity["label_sequence_match"]),
        "selection_export_parity_sanity_max_abs_logit_diff": float(export_parity_sanity["max_abs_logit_diff"]),
        "selection_export_parity_sanity_max_rel_logit_diff": float(export_parity_sanity["max_rel_logit_diff"]),
        "selection_export_parity_sanity_logits_allclose": bool(export_parity_sanity["logits_allclose"]),
        "selection_export_parity_sanity_warning": str(export_parity_sanity.get("warning") or ""),
        "lambda_tail_margin": float(lambda_tail_margin),
        "gamma_t": float(gamma_payload["gamma_t"]),
        "stage1_train_tail_margin_median": float(gamma_payload["stage1_train_tail_margin_median"]),
        "stage2_train_ce_loss": float(train_row["train_ce_loss"]),
        "stage2_train_tail_margin_loss": float(train_row["train_tail_margin_loss"]),
        "stage2_train_tail_margin_loss_tail_sample_mean": float(train_row["train_tail_margin_loss_tail_sample_mean"]),
        "stage2_train_tail_margin_loss_batch_weighted": float(train_row["train_tail_margin_loss_batch_weighted"]),
        "stage2_train_total_loss": float(train_row["train_loss"]),
        "stage2_train_kd_loss": 0.0,
        "stage2_teacher_kd_enabled": False,
        "stage2_eval_accuracy_quick": float(quick_eval["accuracy"]),
        "stage2_eval_macro_f1_label_universe_nanmean_quick": float(quick_eval["eval_breakdown"]["eval_macro_f1_label_universe_nanmean"]),
        "stage2_eval_tail_macro_f1_quick": float(quick_eval["group_summary"]["tail_macro_f1"]),
        "stage2_eval_tail_accuracy_quick": float(quick_eval["group_summary"]["tail_accuracy"]),
        "overall_accuracy": float(official_summary["overall_accuracy"]),
        "overall_macro_f1": float(official_summary["overall_macro_f1"]),
        "many_accuracy": float(official_summary["many_accuracy"]),
        "many_macro_f1": float(official_summary["many_macro_f1"]),
        "medium_accuracy": float(official_summary["medium_accuracy"]),
        "medium_macro_f1": float(official_summary["medium_macro_f1"]),
        "tail_accuracy": float(official_summary["tail_accuracy"]),
        "tail_macro_f1": float(official_summary["tail_macro_f1"]),
        "m_tail": float(official_summary["m_tail"]),
        "m_many": float(official_summary["m_many"]),
        "correct_logit_margin_global": float(official_summary["correct_logit_margin_global"]),
        "correct_logit_margin_many": float(official_summary["correct_logit_margin_many"]),
        "correct_logit_margin_tail": float(official_summary["correct_logit_margin_tail"]),
        "official_classification_metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "diagnostic_export_metric_source": DIAGNOSTIC_EXPORT_METRIC_SOURCE,
        "logit_margin_retention": float(export_summary["logit_margin_retention"]),
        "fisher": float(export_summary["fisher"]),
        "geo_margin": float(export_summary["geo_margin"]),
        "class_trace_variance": float(export_summary["class_trace_variance"]),
        "embedding_distortion": float(export_summary["embedding_distortion"]),
        "embedding_cosine_similarity": float(export_summary["embedding_cosine_similarity"]),
        "fisher_retention": float(export_summary["fisher_retention"]),
        "geo_margin_retention": float(export_summary["geo_margin_retention"]),
        "variance_deviation_abs_log": float(export_summary["variance_deviation_abs_log"]),
        "variance_deviation_nonfinite_class_count": int(export_summary["variance_deviation_nonfinite_class_count"]),
        "tail_collapse_gap": float(export_summary["tail_collapse_gap"]),
        "few_collapse_gap": float(export_summary["few_collapse_gap"]),
        "stage2_export_run_identity_hash": str(export_result.get("run_identity_hash") or ""),
        "stage2_export_manifest_hash": str(export_result.get("run_manifest_hash") or ""),
    }


def _candidate_summary_row(
    *,
    index: int,
    lambda_tail_margin: float,
    lambda_dir: Path,
    checkpoint_path: Path,
    train_log: Path,
    eval_metrics_path: Path,
    manifest_path: Path,
    eval_row: Mapping[str, Any],
    ce_only: Mapping[str, Any],
    export_result: Mapping[str, Any],
) -> dict[str, Any]:
    delta_overall_acc = float(eval_row["overall_accuracy"]) - float(ce_only["overall_accuracy"])
    delta_overall_macro_f1 = float(eval_row["overall_macro_f1"]) - float(ce_only["overall_macro_f1"])
    delta_many_acc = float(eval_row["many_accuracy"]) - float(ce_only["many_accuracy"])
    passes = (
        delta_overall_acc >= OVERALL_ACC_GUARDRAIL
        and delta_overall_macro_f1 >= OVERALL_MACRO_F1_GUARDRAIL
        and delta_many_acc >= MANY_ACC_GUARDRAIL
    )
    return {
        "candidate_index": int(index),
        "selected": False,
        "passes_guardrail": bool(passes),
        "lambda_tail_margin": float(lambda_tail_margin),
        "gamma_t": float(eval_row["gamma_t"]),
        "seed": int(eval_row["seed"]),
        "lambda_dir": path_str(lambda_dir),
        "stage2_checkpoint": path_str(checkpoint_path),
        "stage2_checkpoint_sha256": file_sha256(checkpoint_path),
        "stage2_train_log": path_str(train_log),
        "stage2_eval_metrics": path_str(eval_metrics_path),
        "stage2_manifest": path_str(manifest_path),
        "selection_metric": SELECTION_METRIC,
        "selection_metric_value": float(eval_row["tail_macro_f1"]),
        "delta_overall_accuracy_vs_ce_only": delta_overall_acc,
        "delta_overall_macro_f1_vs_ce_only": delta_overall_macro_f1,
        "delta_many_accuracy_vs_ce_only": delta_many_acc,
        "run_manifest_hash": str(export_result.get("run_manifest_hash") or ""),
        "final_test_exported": False,
        "final_export_splits": "",
        "final_dir": "",
        "final_checkpoint": "",
        "final_metrics_csv": "",
        "final_group_metrics_csv": "",
        "final_geometry_metrics_csv": "",
        "final_export_manifest": "",
        "final_export_run_identity_hash": "",
        **{key: eval_row.get(key, "") for key in SUMMARY_COLUMNS if key in eval_row},
    }


def _single_summary_row(
    *,
    lambda_tail_margin: float,
    lambda_dir: Path,
    checkpoint_path: Path,
    train_log: Path,
    eval_metrics_path: Path,
    manifest_path: Path,
    eval_row: Mapping[str, Any],
    ce_only: Mapping[str, Any],
    export_result: Mapping[str, Any],
) -> dict[str, Any]:
    row = _candidate_summary_row(
        index=1,
        lambda_tail_margin=lambda_tail_margin,
        lambda_dir=lambda_dir,
        checkpoint_path=checkpoint_path,
        train_log=train_log,
        eval_metrics_path=eval_metrics_path,
        manifest_path=manifest_path,
        eval_row=eval_row,
        ce_only=ce_only,
        export_result=export_result,
    )
    diagnostic_passes_guardrail = bool(row.get("passes_guardrail"))
    row.update(
        {
            "run_mode": "single",
            "single_mode": True,
            "selection_enabled": False,
            "guardrail_selection_enabled": False,
            "diagnostic_passes_guardrail": diagnostic_passes_guardrail,
            "guardrail_delta_overall_accuracy_min": OVERALL_ACC_GUARDRAIL,
            "guardrail_delta_overall_macro_f1_min": OVERALL_MACRO_F1_GUARDRAIL,
            "guardrail_delta_many_accuracy_min": MANY_ACC_GUARDRAIL,
            "selected": "",
            "passes_guardrail": diagnostic_passes_guardrail,
            "final_test_exported": False,
            "final_export_splits": "eval",
        }
    )
    return row


def _select_best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [row for row in rows if bool(row.get("passes_guardrail"))]
    if not eligible:
        raise RuntimeError("tail repair sweep produced no candidates passing the three performance guardrails")
    return max(
        eligible,
        key=lambda row: (
            _safe_score(float(row["tail_macro_f1"])),
            _safe_score(float(row["tail_accuracy"])),
            _safe_score(float(row["medium_macro_f1"])),
            _safe_score(float(row["medium_accuracy"])),
            -int(row["candidate_index"]),
        ),
    )


def _write_combined_export_table(split_results: Mapping[str, Any], *, table_key: str, path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for split, result in split_results.items():
        paths = result.get("paths")
        if not isinstance(paths, Mapping) or table_key not in paths:
            raise KeyError(f"final export split result missing {table_key!r} path for split={split!r}")
        source_csv = Path(paths[table_key])
        with source_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(str(name))
            for row in reader:
                out = dict(row)
                out.setdefault("split", str(split))
                rows.append(out)
    if not rows:
        raise RuntimeError(f"final export produced no rows for table={table_key!r}")
    if "split" not in fieldnames:
        fieldnames.insert(0, "split")
    _write_csv(path, rows, columns=fieldnames)


def _ce_only_reference_metrics(path: Path, *, seed: int, stage1_checkpoint: Path) -> dict[str, Any]:
    row = _read_single_csv_row(path)
    observed_seed = int(float(row.get("seed", "nan")))
    if observed_seed != int(seed):
        raise RuntimeError(
            f"CE-only Stage2 eval metrics seed mismatch: existing={observed_seed}, expected={int(seed)}"
        )
    checkpoint_value = str(row.get("stage2_checkpoint") or "").strip()
    if not checkpoint_value:
        raise RuntimeError(f"CE-only Stage2 eval metrics missing stage2_checkpoint: {path}")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CE-only Stage2 checkpoint recorded by eval metrics does not exist: {checkpoint_path}")
    stored_checkpoint_hash = str(row.get("stage2_checkpoint_sha256") or "").strip()
    if not stored_checkpoint_hash:
        raise RuntimeError(f"CE-only Stage2 eval metrics missing stage2_checkpoint_sha256: {path}")
    actual_checkpoint_hash = file_sha256(checkpoint_path)
    if str(actual_checkpoint_hash) != stored_checkpoint_hash:
        raise RuntimeError(
            "CE-only Stage2 eval metrics checkpoint hash mismatch: "
            f"metrics={stored_checkpoint_hash}, actual={actual_checkpoint_hash}, checkpoint={checkpoint_path}"
        )
    current_stage1_hash = file_sha256(stage1_checkpoint)
    if not current_stage1_hash:
        raise FileNotFoundError(f"current Stage1 checkpoint is not hashable: {stage1_checkpoint}")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError(f"CE-only Stage2 checkpoint must contain a mapping: {checkpoint_path}")
    if str(checkpoint_payload.get("model_kind") or "") != "two_stage_decoupling":
        raise RuntimeError(
            "CE-only guardrail reference must be a two_stage_decoupling checkpoint, "
            f"got model_kind={checkpoint_payload.get('model_kind')!r}: {checkpoint_path}"
        )
    if int(checkpoint_payload.get("stage_index", -1)) != 2:
        raise RuntimeError(
            f"CE-only guardrail reference must be Stage2, got stage_index={checkpoint_payload.get('stage_index')!r}: {checkpoint_path}"
        )
    if int(checkpoint_payload.get("seed", -1)) != int(seed):
        raise RuntimeError(
            f"CE-only Stage2 checkpoint seed mismatch: existing={checkpoint_payload.get('seed')!r}, expected={int(seed)}"
        )
    source_stage1_hash = str(checkpoint_payload.get("source_stage1_checkpoint_sha256") or "").strip()
    source_stage1_hash_source = "checkpoint_payload"
    manifest_path = path.with_name("stage2_manifest.json")
    manifest_source_stage1_hash = ""
    if manifest_path.exists():
        manifest_payload = _read_json_mapping(manifest_path)
        manifest_seed_value = manifest_payload.get("seed")
        if manifest_seed_value is not None and str(manifest_seed_value).strip():
            manifest_seed = int(float(str(manifest_seed_value)))
            if manifest_seed != int(seed):
                raise RuntimeError(
                    f"CE-only Stage2 manifest seed mismatch: existing={manifest_seed}, expected={int(seed)}"
                )
        manifest_checkpoint_hash = str(manifest_payload.get("stage2_checkpoint_sha256") or "").strip()
        if not manifest_checkpoint_hash:
            raise RuntimeError(f"CE-only Stage2 manifest missing stage2_checkpoint_sha256: {manifest_path}")
        if manifest_checkpoint_hash != stored_checkpoint_hash:
            raise RuntimeError(
                "CE-only Stage2 manifest checkpoint hash mismatch: "
                f"manifest={manifest_checkpoint_hash}, metrics={stored_checkpoint_hash}, manifest_path={manifest_path}"
            )
        manifest_checkpoint_value = str(manifest_payload.get("stage2_checkpoint") or "").strip()
        if manifest_checkpoint_value:
            if Path(manifest_checkpoint_value).resolve() != checkpoint_path.resolve():
                raise RuntimeError(
                    "CE-only Stage2 manifest checkpoint path mismatch: "
                    f"manifest={manifest_checkpoint_value}, metrics={checkpoint_path}"
                )
        manifest_source_stage1_hash = str(manifest_payload.get("stage1_checkpoint_sha256") or "").strip()
    if source_stage1_hash and manifest_source_stage1_hash and source_stage1_hash != manifest_source_stage1_hash:
        raise RuntimeError(
            "CE-only Stage2 checkpoint and manifest disagree on source Stage1 hash: "
            f"checkpoint={source_stage1_hash}, manifest={manifest_source_stage1_hash}"
        )
    if not source_stage1_hash:
        source_stage1_hash = manifest_source_stage1_hash
        source_stage1_hash_source = path_str(manifest_path) if source_stage1_hash else ""
    if not source_stage1_hash:
        raise RuntimeError(
            "CE-only Stage2 reference is missing source Stage1 provenance in both checkpoint payload "
            f"and stage2_manifest.json: checkpoint={checkpoint_path}, manifest={manifest_path}"
        )
    if source_stage1_hash != str(current_stage1_hash):
        raise RuntimeError(
            "CE-only guardrail reference was not trained from the fixed Stage1 checkpoint: "
            f"ce_only_source_stage1={source_stage1_hash}, current_stage1={current_stage1_hash}, "
            f"source={source_stage1_hash_source}"
        )
    return {
        "overall_accuracy": _float_value(row, "stage2_eval_accuracy"),
        "overall_macro_f1": _float_value(row, "stage2_eval_macro_f1_label_universe_nanmean"),
        "many_accuracy": _float_value(row, "stage2_eval_many_accuracy"),
        "stage2_checkpoint": path_str(checkpoint_path),
        "stage2_checkpoint_sha256": stored_checkpoint_hash,
        "source_stage1_checkpoint_sha256": source_stage1_hash,
        "source_stage1_checkpoint_sha256_source": source_stage1_hash_source,
    }


def _run_lambda_candidate(
    *,
    args: argparse.Namespace,
    lambda_tail_margin: float,
    lambda_dir: Path,
    run_mode: str,
    paths: Any,
    stage1_payload: Mapping[str, Any],
    gamma_payload: Mapping[str, Any],
    label_buckets: Mapping[int, str],
    label_names: Mapping[int, str],
    num_classes: int,
    device: torch.device,
    stage2: StageSpec,
    reference_root: Path,
) -> dict[str, Any]:
    if run_mode not in {"sweep_candidate", "single"}:
        raise ValueError(f"unsupported tail-repair run mode: {run_mode!r}")
    single_mode = run_mode == "single"
    _seed_everything(int(args.seed))
    lambda_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = lambda_dir / "stage2_checkpoint.pt"
    last_checkpoint_path = lambda_dir / "stage2_last.pt"
    train_log = lambda_dir / "stage2_train_log.csv"
    eval_metrics_csv = lambda_dir / "stage2_eval_metrics.csv"
    stage_log = lambda_dir / "stage_log.jsonl"
    manifest_path = lambda_dir / "stage2_manifest.json"
    export_dir = lambda_dir / "eval_export"

    student, loaded_payload = load_trained_checkpoint(_stage1_checkpoint(args), device=device)
    _require_stage1_checkpoint_payload(loaded_payload, checkpoint_path=_stage1_checkpoint(args), paths=paths)
    optimizer, stage_contract = _configure_stage(student, stage2, weight_decay=float(args.weight_decay))
    train_loader = build_loader(paths, split="train", batch_size=int(args.batch_size), num_workers=int(args.num_workers), seed=int(args.seed))
    eval_loader = build_loader(paths, split="eval", batch_size=int(args.eval_batch_size), num_workers=int(args.num_workers), seed=int(args.seed))
    scheduler = _make_scheduler(
        optimizer,
        total_steps=int(stage2.epochs) * max(1, len(train_loader)),
        warmup_ratio=float(args.warmup_ratio),
    )
    _append_stage_log(
        stage_log,
        {
            "event": "tail_repair_single_start" if single_mode else "tail_repair_candidate_start",
            "run_mode": run_mode,
            "single_mode": bool(single_mode),
            "lambda_tail_margin": float(lambda_tail_margin),
            "gamma_t": float(gamma_payload["gamma_t"]),
            "source_stage1_checkpoint": path_str(_stage1_checkpoint(args)),
            "test_split_touched": False,
            "rng_reset_per_candidate": True,
            "candidate_rng_seed": int(args.seed),
        },
    )

    train_rows: list[dict[str, Any]] = []
    best_eval: dict[str, Any] | None = None
    best_quick_eval: dict[str, Any] | None = None
    best_train_row: dict[str, Any] = {}
    best_quant_audit: dict[str, Any] = {}
    best_stage_epoch = 0
    best_global_epoch = int(stage1_payload.get("epoch", 0))
    best_checkpoint_payload: dict[str, Any] | None = None
    last_quant_audit: dict[str, Any] = {}
    epoch_eval_checkpoint_paths: list[Path] = []

    for stage_epoch in range(1, int(stage2.epochs) + 1):
        global_epoch = int(stage1_payload.get("epoch", 0)) + int(stage_epoch)
        epoch_checkpoint_path = lambda_dir / f"stage2_epoch_{stage_epoch:04d}_checkpoint.pt"
        train_metrics = train_tail_repair_epoch(
            model=student,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_classes=num_classes,
            label_buckets=label_buckets,
            gamma_t=float(gamma_payload["gamma_t"]),
            lambda_tail_margin=float(lambda_tail_margin),
        )
        if float(train_metrics.get("teacher_forward_count", 0.0)) != 0.0:
            raise RuntimeError("Tail repair contract violation: teacher forward was computed")
        last_quant_audit = _quant_audit(student, stage1_payload, context=f"after_tail_repair_lambda_{lambda_tail_margin:g}_epoch_{stage_epoch}")
        eval_metrics = _evaluate_tail_repair(
            model=student,
            loader=eval_loader,
            device=device,
            num_classes=num_classes,
            label_buckets=label_buckets,
            label_names=label_names,
        )
        quick_group_summary = eval_metrics["group_summary"]
        quick_current = float(quick_group_summary.get("tail_macro_f1", float("nan")))
        epoch_checkpoint_payload = _save_tail_repair_checkpoint(
            epoch_checkpoint_path,
            model=student,
            source_payload=stage1_payload,
            source_stage1_checkpoint=_stage1_checkpoint(args),
            stage_epoch=int(stage_epoch),
            global_epoch=int(global_epoch),
            eval_metrics=eval_metrics,
            eval_metrics_source="live_model_training_monitor_quick_eval",
            lambda_tail_margin=float(lambda_tail_margin),
            gamma_payload=gamma_payload,
            stage_contract=stage_contract,
            quant_audit=last_quant_audit,
            best_metric_so_far=quick_current,
        )
        epoch_eval_checkpoint_paths.append(epoch_checkpoint_path)
        selection_eval = _canonical_reload_eval_checkpoint(
            checkpoint_path=epoch_checkpoint_path,
            args=args,
            paths=paths,
            label_buckets=label_buckets,
            label_names=label_names,
            num_classes=num_classes,
            device=device,
            include_logits=False,
        )
        selection_group_summary = selection_eval["group_summary"]
        selection_current = float(selection_group_summary.get("tail_macro_f1", float("nan")))
        shutil.copy2(epoch_checkpoint_path, last_checkpoint_path)
        row = {
            "stage": stage2.name,
            "stage_index": int(stage2.index),
            "stage_epoch": int(stage_epoch),
            "global_epoch": int(global_epoch),
            "objective": stage2.objective,
            "lambda_tail_margin": float(lambda_tail_margin),
            "gamma_t": float(gamma_payload["gamma_t"]),
            "backbone_lr": 0.0,
            "backbone_optimizer_group_exists": bool(stage_contract.get("backbone_optimizer_group_exists", False)),
            "classifier_lr": _optimizer_lr(optimizer, "classifier"),
            "kd_loss_enabled": False,
            "backbone_trainable": False,
            "classifier_trainable": True,
            **{key: float(value) for key, value in train_metrics.items()},
            "eval_loss": float(selection_eval["loss"]),
            "eval_accuracy": float(selection_eval["accuracy"]),
            "eval_macro_f1_label_universe_nanmean": float(selection_eval["eval_breakdown"]["eval_macro_f1_label_universe_nanmean"]),
            "eval_tail_macro_f1": selection_current,
            "eval_tail_accuracy": float(selection_group_summary["tail_accuracy"]),
            "eval_medium_macro_f1": float(selection_group_summary["medium_macro_f1"]),
            "eval_medium_accuracy": float(selection_group_summary["medium_accuracy"]),
            "eval_many_macro_f1": float(selection_group_summary["many_macro_f1"]),
            "eval_many_accuracy": float(selection_group_summary["many_accuracy"]),
            "eval_correct_logit_margin_global": float(selection_eval["correct_logit_margin_global"]),
            "eval_correct_logit_margin_many": float(selection_eval["correct_logit_margin_many"]),
            "eval_correct_logit_margin_tail": float(selection_eval["correct_logit_margin_tail"]),
            "eval_loss_quick": float(eval_metrics["loss"]),
            "eval_accuracy_quick": float(eval_metrics["accuracy"]),
            "eval_macro_f1_label_universe_nanmean_quick": float(eval_metrics["eval_breakdown"]["eval_macro_f1_label_universe_nanmean"]),
            "eval_tail_macro_f1_quick": quick_current,
            "eval_tail_accuracy_quick": float(quick_group_summary["tail_accuracy"]),
            "eval_medium_macro_f1_quick": float(quick_group_summary["medium_macro_f1"]),
            "eval_medium_accuracy_quick": float(quick_group_summary["medium_accuracy"]),
            "eval_many_macro_f1_quick": float(quick_group_summary["many_macro_f1"]),
            "eval_many_accuracy_quick": float(quick_group_summary["many_accuracy"]),
            "eval_correct_logit_margin_global_quick": float(eval_metrics["correct_logit_margin_global"]),
            "eval_correct_logit_margin_many_quick": float(eval_metrics["correct_logit_margin_many"]),
            "eval_correct_logit_margin_tail_quick": float(eval_metrics["correct_logit_margin_tail"]),
            "eval_selection_metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
            "eval_selection_checkpoint": path_str(epoch_checkpoint_path),
            "eval_selection_checkpoint_sha256": file_sha256(epoch_checkpoint_path),
            "tail_margin_hinge_power": 1,
            "teacher_forward_count": 0.0,
            "quant_audit_observer_enabled": bool(last_quant_audit.get("observer_enabled")),
            "quant_audit_fake_quant_enabled": bool(last_quant_audit.get("fake_quant_enabled")),
            "quant_audit_qparams_frozen": bool(last_quant_audit.get("qparams_frozen")),
        }
        train_rows.append(row)
        _write_train_log(train_log, train_rows)
        print(
            f"[tail-repair] lambda={lambda_tail_margin:g} epoch={stage_epoch}/{stage2.epochs} "
            f"train_loss={row['train_loss']:.6f} quick_tail_macro_f1={quick_current:.6f} "
            f"selection_tail_macro_f1={selection_current:.6f}"
        )
        if _is_better_eval(selection_eval, best_eval):
            best_eval = dict(selection_eval)
            best_quick_eval = dict(eval_metrics)
            best_train_row = dict(row)
            best_quant_audit = dict(last_quant_audit)
            best_stage_epoch = int(stage_epoch)
            best_global_epoch = int(global_epoch)
            shutil.copy2(epoch_checkpoint_path, checkpoint_path)
            best_checkpoint_payload = dict(epoch_checkpoint_payload)
            _append_stage_log(
                stage_log,
                {
                    "event": "tail_repair_best_checkpoint",
                    "lambda_tail_margin": float(lambda_tail_margin),
                    "stage_epoch": int(stage_epoch),
                    "global_epoch": int(global_epoch),
                    "selection_metric": SELECTION_METRIC,
                    "selection_metric_value": selection_current,
                    "selection_metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
                    "quick_eval_tail_macro_f1": quick_current,
                    "evaluated_checkpoint": path_str(epoch_checkpoint_path),
                    "evaluated_checkpoint_sha256": file_sha256(epoch_checkpoint_path),
                    "checkpoint": path_str(checkpoint_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                },
            )

    if best_checkpoint_payload is None or best_eval is None or best_quick_eval is None:
        raise RuntimeError(f"Tail repair lambda={lambda_tail_margin:g} produced no eval-best checkpoint")
    selection_sanity = _canonical_reload_sanity_check(
        checkpoint_path=checkpoint_path,
        args=args,
        paths=paths,
        label_buckets=label_buckets,
        label_names=label_names,
        num_classes=num_classes,
        device=device,
    )
    _append_stage_log(
        stage_log,
        {
            "event": "tail_repair_selection_reload_sanity",
            **selection_sanity,
        },
    )
    print(
        "[tail-repair] selection_reload_sanity "
        f"pred_diff={int(selection_sanity['prediction_disagreement_count'])} "
        f"max_abs_logit_diff={float(selection_sanity['max_abs_logit_diff']):.9g} "
        f"max_rel_logit_diff={float(selection_sanity['max_rel_logit_diff']):.9g}"
    )
    token_hook_sanity = _token_hook_sanity_check(
        checkpoint_path=checkpoint_path,
        args=args,
        paths=paths,
        label_buckets=label_buckets,
        label_names=label_names,
        num_classes=num_classes,
        device=device,
    )
    _append_stage_log(
        stage_log,
        {
            "event": "tail_repair_selection_token_hook_sanity",
            **token_hook_sanity,
        },
    )
    token_hook_warning = str(token_hook_sanity.get("warning") or "")
    print(
        "[tail-repair] selection_token_hook_sanity "
        f"pred_diff={int(token_hook_sanity['prediction_disagreement_count'])} "
        f"max_abs_logit_diff={float(token_hook_sanity['max_abs_logit_diff']):.9g} "
        f"max_rel_logit_diff={float(token_hook_sanity['max_rel_logit_diff']):.9g}"
    )
    if token_hook_warning:
        print(f"[tail-repair][warning] {token_hook_warning}: token-hook diagnostic pass changed the canonical path")

    export_result = _export_split(
        checkpoint_path=checkpoint_path,
        split_name="eval",
        split_csv=paths.eval_csv,
        output_dir=export_dir,
        label_mapping_path=paths.label_mapping,
        label_buckets=label_buckets,
        args=args,
        reference_root=reference_root,
    )
    split_result = dict(export_result["split"])
    export_summary = _export_metric_summary(
        split_result=split_result,
        reference_root=reference_root,
        label_buckets=label_buckets,
    )
    export_parity_sanity = _export_parity_sanity_check(
        checkpoint_path=checkpoint_path,
        split_name="eval",
        export_result=export_result,
        args=args,
        paths=paths,
        label_buckets=label_buckets,
        label_names=label_names,
        num_classes=num_classes,
        device=device,
        hard_fail_on_prediction_mismatch=False,
    )
    _append_stage_log(
        stage_log,
        {
            "event": "tail_repair_selection_export_parity_sanity",
            **export_parity_sanity,
        },
    )
    export_parity_warning = str(export_parity_sanity.get("warning") or "")
    print(
        "[tail-repair] selection_export_parity_sanity "
        f"pred_mismatch={int(export_parity_sanity['prediction_mismatch_count'])} "
        f"max_abs_logit_diff={float(export_parity_sanity['max_abs_logit_diff']):.9g} "
        f"max_rel_logit_diff={float(export_parity_sanity['max_rel_logit_diff']):.9g}"
    )
    if export_parity_warning:
        print(f"[tail-repair][warning] {export_parity_warning}: actual export path differs from canonical reload path")
    eval_row = _candidate_eval_metrics_row(
        seed=int(args.seed),
        lambda_tail_margin=float(lambda_tail_margin),
        gamma_payload=gamma_payload,
        stage_epoch=best_stage_epoch,
        global_epoch=best_global_epoch,
        checkpoint_path=checkpoint_path,
        train_row=best_train_row,
        quick_eval=best_quick_eval,
        selection_eval=best_eval,
        selection_sanity=selection_sanity,
        token_hook_sanity=token_hook_sanity,
        export_parity_sanity=export_parity_sanity,
        export_summary=export_summary,
        export_result=export_result,
    )
    _write_csv(eval_metrics_csv, [eval_row])
    manifest = {
        "protocol_version": TAIL_REPAIR_PROTOCOL_VERSION,
        "artifact_type": "tail_repair_single_lambda" if single_mode else "tail_repair_lambda_candidate",
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "run_mode": run_mode,
        "single_mode": bool(single_mode),
        "sweep_mode": not bool(single_mode),
        "selection_enabled": not bool(single_mode),
        "guardrail_selection_enabled": not bool(single_mode),
        "selection_split": "eval",
        "export_splits": ["eval"],
        "test_split_touched": False,
        "rng_reset_per_candidate": True,
        "candidate_rng_seed": int(args.seed),
        "seed": int(args.seed),
        "lambda_tail_margin": float(lambda_tail_margin),
        "gamma_t": float(gamma_payload["gamma_t"]),
        "gamma_t_payload": dict(gamma_payload),
        "source_stage1_checkpoint": path_str(_stage1_checkpoint(args)),
        "source_stage1_checkpoint_sha256": file_sha256(_stage1_checkpoint(args)),
        "source_stage1_run_identity_hash": stage1_payload.get("run_identity_hash"),
        "stage2": stage2.__dict__,
        "stage2_checkpoint": path_str(checkpoint_path),
        "stage2_checkpoint_sha256": file_sha256(checkpoint_path),
        "stage2_last": path_str(last_checkpoint_path),
        "stage2_last_sha256": file_sha256(last_checkpoint_path),
        "stage2_epoch_eval_checkpoints": [path_str(item) for item in epoch_eval_checkpoint_paths],
        "stage2_epoch_eval_checkpoint_sha256": {item.name: file_sha256(item) for item in epoch_eval_checkpoint_paths},
        "stage2_train_log": path_str(train_log),
        "stage2_eval_metrics": path_str(eval_metrics_csv),
        "stage_log": path_str(stage_log),
        "eval_export": export_result,
        "official_classification_metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "diagnostic_export_metric_source": DIAGNOSTIC_EXPORT_METRIC_SOURCE,
        "best_stage_epoch": int(best_stage_epoch),
        "best_global_epoch": int(best_global_epoch),
        "best_eval_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "best_eval": dict(best_eval),
        "best_quick_eval_source": "live_model_training_monitor",
        "best_quick_eval": dict(best_quick_eval),
        "selection_reload_sanity": dict(selection_sanity),
        "selection_token_hook_sanity": dict(token_hook_sanity),
        "selection_export_parity_sanity": dict(export_parity_sanity),
        "stage_contract": dict(stage_contract),
        "best_quant_audit": dict(best_quant_audit),
        "artifact_sha256": {
            "stage2_checkpoint.pt": file_sha256(checkpoint_path),
            "stage2_last.pt": file_sha256(last_checkpoint_path),
            **{item.name: file_sha256(item) for item in epoch_eval_checkpoint_paths},
            "stage2_train_log.csv": file_sha256(train_log),
            "stage2_eval_metrics.csv": file_sha256(eval_metrics_csv),
            **dict(split_result.get("artifact_sha256", {})),
        },
    }
    manifest = {**manifest, "run_manifest_hash": canonical_json_hash(manifest)}
    _write_json(manifest_path, manifest)
    return {
        "lambda_dir": lambda_dir,
        "stage2_checkpoint": checkpoint_path,
        "stage2_last": last_checkpoint_path,
        "stage2_train_log": train_log,
        "stage2_eval_metrics": eval_metrics_csv,
        "stage2_manifest": manifest_path,
        "stage_log": stage_log,
        "manifest": manifest,
        "export_result": export_result,
        "eval_row": eval_row,
    }


def _dry_run_plan(args: argparse.Namespace, *, seed_dir: Path, summary_csv: Path, manifest_path: Path) -> dict[str, Any]:
    if args.single is not None:
        single_value = float(args.single)
        single_dir = _single_lambda_dir(seed_dir, single_value)
        return {
            "protocol_version": TAIL_REPAIR_PROTOCOL_VERSION,
            "baseline": BASELINE_NAME,
            "model_kind": MODEL_KIND,
            "method_variant": METHOD_VARIANT,
            "run_mode": "single",
            "single_mode": True,
            "selection_enabled": False,
            "guardrail_selection_enabled": False,
            "seed": int(args.seed),
            "output_seed_dir": path_str(seed_dir),
            "single_lambda_tail_margin": single_value,
            "single_dir": path_str(single_dir),
            "single_summary_csv": path_str(_single_summary_csv(single_dir)),
            "single_manifest_path": path_str(_single_manifest_path(single_dir)),
            "force_single": bool(args.force_single),
            "stage1_checkpoint": path_str(_stage1_checkpoint(args)),
            "reference_root": path_str(_reference_root(args)),
            "ce_only_stage2_eval_metrics": path_str(_ce_only_metrics(args)),
            "stage2_epochs": int(args.stage2_epochs),
            "gamma_policy": "compute once from Stage1 train tail correct_logit_margin; gamma_T=max(0, median)",
            "single_test_policy": "single lambda validates/exports eval only and does not read test split",
            "single_rng_policy": "reset _seed_everything(seed) before loading Stage1 checkpoint and constructing train/eval loaders",
            "single_guardrail_diagnostic_policy": {
                "enabled": True,
                "used_for_selection": False,
                "delta_overall_accuracy_min": OVERALL_ACC_GUARDRAIL,
                "delta_overall_macro_f1_min": OVERALL_MACRO_F1_GUARDRAIL,
                "delta_many_accuracy_min": MANY_ACC_GUARDRAIL,
            },
        }
    candidates = []
    for value in args.lambda_values:
        lambda_dir = seed_dir / _lambda_dir_name(float(value))
        candidates.append({"lambda_tail_margin": float(value), "lambda_dir": path_str(lambda_dir)})
    return {
        "protocol_version": TAIL_REPAIR_PROTOCOL_VERSION,
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "seed": int(args.seed),
        "output_seed_dir": path_str(seed_dir),
        "summary_csv": path_str(summary_csv),
        "manifest_path": path_str(manifest_path),
        "stage1_checkpoint": path_str(_stage1_checkpoint(args)),
        "reference_root": path_str(_reference_root(args)),
        "ce_only_stage2_eval_metrics": path_str(_ce_only_metrics(args)),
        "lambda_values": [float(item) for item in args.lambda_values],
        "stage2_epochs": int(args.stage2_epochs),
        "gamma_policy": "compute once from Stage1 train tail correct_logit_margin; gamma_T=max(0, median)",
        "sweep_test_policy": "candidate lambdas validate/export eval only and do not read test split",
        "final_test_policy": "export eval+test only for the selected lambda checkpoint",
        "candidate_rng_policy": "reset _seed_everything(seed) at the start of each lambda candidate before loading Stage1/building loaders",
        "selection_policy": {
            "guardrails": {
                "delta_overall_accuracy": OVERALL_ACC_GUARDRAIL,
                "delta_overall_macro_f1": OVERALL_MACRO_F1_GUARDRAIL,
                "delta_many_accuracy": MANY_ACC_GUARDRAIL,
            },
            "geometry_hard_gate": False,
            "primary": "tail_macro_f1",
            "tie_break": ["tail_accuracy", "medium_macro_f1", "medium_accuracy"],
        },
        "candidates": candidates,
        "final_dir_pattern": path_str(seed_dir / "selected_lambda_<value>_final"),
    }


def run_tail_repair_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _reject_amp(args)
    _seed_everything(int(args.seed))
    paths = dataset_paths(args.dataset, None)
    seed_dir = _output_seed_dir(args)
    summary_csv = _summary_csv(args, seed_dir)
    manifest_path = _manifest_path(args, seed_dir)
    if bool(args.dry_run):
        plan = _dry_run_plan(args, seed_dir=seed_dir, summary_csv=summary_csv, manifest_path=manifest_path)
        print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))
        return {"dry_run": True, "plan": plan}

    stage1_checkpoint = _stage1_checkpoint(args)
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {stage1_checkpoint}")
    ce_metrics_path = _ce_only_metrics(args)
    if not ce_metrics_path.exists():
        raise FileNotFoundError(f"CE-only Stage2 eval metrics not found for guardrails: {ce_metrics_path}")
    reference_root = _reference_root(args)
    if not reference_root.exists():
        raise FileNotFoundError(f"FP32 reference artifact root not found: {reference_root}")
    if args.single is not None:
        single_dir_preview = _single_lambda_dir(seed_dir, float(args.single))
        if single_dir_preview.exists() and any(single_dir_preview.iterdir()) and not bool(args.force_single):
            raise FileExistsError(
                f"single-lambda output directory already exists and is not empty: {single_dir_preview}. "
                "Pass --force-single to clear and replace it."
            )

    label_mapping = load_label_mapping(paths.label_mapping)
    label_names = label_names_by_id(label_mapping)
    num_classes = num_predictor_classes(label_mapping)
    validate_split_labels(paths.train_csv, num_classes=num_classes, name="train")
    validate_split_labels(paths.eval_csv, num_classes=num_classes, name="eval")
    label_buckets = build_train_label_buckets(paths.train_csv)
    validate_split_bucket_coverage(paths.eval_csv, label_buckets, split_name="eval")
    ce_only = _ce_only_reference_metrics(ce_metrics_path, seed=int(args.seed), stage1_checkpoint=stage1_checkpoint)

    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    stage1_model, stage1_payload = load_trained_checkpoint(stage1_checkpoint, device=device)
    _require_stage1_checkpoint_payload(stage1_payload, checkpoint_path=stage1_checkpoint, paths=paths)
    _quant_audit(stage1_model, stage1_payload, context="tail_repair_loaded_stage1_for_gamma")
    gamma_loader = build_loader(paths, split="train", batch_size=int(args.eval_batch_size), num_workers=int(args.num_workers), seed=int(args.seed))
    gamma_payload = _compute_gamma_t(
        model=stage1_model,
        loader=gamma_loader,
        device=device,
        num_classes=num_classes,
        label_buckets=label_buckets,
    )
    del stage1_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stage2 = _stage2_spec(args)
    if args.single is not None:
        lambda_value = float(args.single)
        lambda_dir = _single_lambda_dir(seed_dir, lambda_value)
        _prepare_single_output_dir(lambda_dir, force=bool(args.force_single))
        result = _run_lambda_candidate(
            args=args,
            lambda_tail_margin=lambda_value,
            lambda_dir=lambda_dir,
            run_mode="single",
            paths=paths,
            stage1_payload=stage1_payload,
            gamma_payload=gamma_payload,
            label_buckets=label_buckets,
            label_names=label_names,
            num_classes=num_classes,
            device=device,
            stage2=stage2,
            reference_root=reference_root,
        )
        eval_row = _read_single_csv_row(result["stage2_eval_metrics"])
        summary_row = _single_summary_row(
            lambda_tail_margin=lambda_value,
            lambda_dir=lambda_dir,
            checkpoint_path=result["stage2_checkpoint"],
            train_log=result["stage2_train_log"],
            eval_metrics_path=result["stage2_eval_metrics"],
            manifest_path=result["stage2_manifest"],
            eval_row=eval_row,
            ce_only=ce_only,
            export_result=result["export_result"],
        )
        single_summary_csv = _single_summary_csv(lambda_dir)
        single_manifest_path = _single_manifest_path(lambda_dir)
        _write_csv(single_summary_csv, [summary_row], columns=SINGLE_SUMMARY_COLUMNS)
        manifest_identity = {
            "protocol_version": TAIL_REPAIR_PROTOCOL_VERSION,
            "dataset_name": paths.name,
            "seed": int(args.seed),
            "stage1_checkpoint_sha256": file_sha256(stage1_checkpoint),
            "lambda_tail_margin": lambda_value,
            "gamma_t": float(gamma_payload["gamma_t"]),
            "ce_only_metrics_sha256": file_sha256(ce_metrics_path),
            "method_variant": METHOD_VARIANT,
            "run_mode": "single",
        }
        manifest = {
            **manifest_identity,
            "artifact_type": "tail_repair_single_lambda_run",
            "baseline": BASELINE_NAME,
            "model_kind": MODEL_KIND,
            "single_identity_hash": canonical_json_hash(manifest_identity),
            "run_mode": "single",
            "single_mode": True,
            "selection_enabled": False,
            "guardrail_selection_enabled": False,
            "selection_split": "eval",
            "export_splits": ["eval"],
            "final_export_splits": ["eval"],
            "test_split_touched": False,
            "output_seed_dir": path_str(seed_dir),
            "single_dir": path_str(lambda_dir),
            "stage1_checkpoint": path_str(stage1_checkpoint),
            "reference_root": path_str(reference_root),
            "ce_only_stage2_eval_metrics": path_str(ce_metrics_path),
            "ce_only_guardrail_reference": ce_only,
            "lambda_tail_margin": lambda_value,
            "gamma_t_payload": dict(gamma_payload),
            "stage2_checkpoint": path_str(result["stage2_checkpoint"]),
            "stage2_checkpoint_sha256": file_sha256(result["stage2_checkpoint"]),
            "stage2_last": path_str(result["stage2_last"]),
            "stage2_last_sha256": file_sha256(result["stage2_last"]),
            "stage2_train_log": path_str(result["stage2_train_log"]),
            "stage2_eval_metrics": path_str(result["stage2_eval_metrics"]),
            "stage2_manifest": path_str(result["stage2_manifest"]),
            "single_summary_csv": path_str(single_summary_csv),
            "single_summary_csv_sha256": file_sha256(single_summary_csv),
            "eval_export": result["export_result"],
            "summary_row": summary_row,
            "single_policy": {
                "starts_from": "stage1_checkpoint.pt",
                "reuses_existing_lambda_checkpoint": False,
                "test_split_touched": False,
                "selection_enabled": False,
                "ce_only_delta_is_diagnostic_only": True,
            },
            "single_guardrail_diagnostic": {
                "enabled": True,
                "used_for_selection": False,
                "passes_guardrail": bool(summary_row["passes_guardrail"]),
                "thresholds": {
                    "delta_overall_accuracy_min": OVERALL_ACC_GUARDRAIL,
                    "delta_overall_macro_f1_min": OVERALL_MACRO_F1_GUARDRAIL,
                    "delta_many_accuracy_min": MANY_ACC_GUARDRAIL,
                },
                "deltas": {
                    "delta_overall_accuracy_vs_ce_only": float(summary_row["delta_overall_accuracy_vs_ce_only"]),
                    "delta_overall_macro_f1_vs_ce_only": float(summary_row["delta_overall_macro_f1_vs_ce_only"]),
                    "delta_many_accuracy_vs_ce_only": float(summary_row["delta_many_accuracy_vs_ce_only"]),
                },
            },
            "artifact_sha256": {
                "stage2_checkpoint.pt": file_sha256(result["stage2_checkpoint"]),
                "stage2_last.pt": file_sha256(result["stage2_last"]),
                "stage2_train_log.csv": file_sha256(result["stage2_train_log"]),
                "stage2_eval_metrics.csv": file_sha256(result["stage2_eval_metrics"]),
                "stage2_manifest.json": file_sha256(result["stage2_manifest"]),
                "tail_repair_single_eval_summary.csv": file_sha256(single_summary_csv),
            },
        }
        manifest = {**manifest, "run_manifest_hash": canonical_json_hash(manifest)}
        _write_json(single_manifest_path, manifest)
        print(f"[tail-repair] single_lambda={lambda_value:g}")
        print(f"[tail-repair] single_dir={lambda_dir}")
        print(f"[tail-repair] single_summary={single_summary_csv}")
        print(f"[tail-repair] single_manifest={single_manifest_path}")
        return {"manifest": manifest, "summary_csv": single_summary_csv, "single_lambda": lambda_value}

    candidate_rows: list[dict[str, Any]] = []
    candidate_results: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(args.lambda_values, start=1):
        lambda_value = float(value)
        lambda_dir = seed_dir / _lambda_dir_name(lambda_value)
        result = _run_lambda_candidate(
            args=args,
            lambda_tail_margin=lambda_value,
            lambda_dir=lambda_dir,
            run_mode="sweep_candidate",
            paths=paths,
            stage1_payload=stage1_payload,
            gamma_payload=gamma_payload,
            label_buckets=label_buckets,
            label_names=label_names,
            num_classes=num_classes,
            device=device,
            stage2=stage2,
            reference_root=reference_root,
        )
        eval_row = _read_single_csv_row(result["stage2_eval_metrics"])
        summary_row = _candidate_summary_row(
            index=index,
            lambda_tail_margin=lambda_value,
            lambda_dir=lambda_dir,
            checkpoint_path=result["stage2_checkpoint"],
            train_log=result["stage2_train_log"],
            eval_metrics_path=result["stage2_eval_metrics"],
            manifest_path=result["stage2_manifest"],
            eval_row=eval_row,
            ce_only=ce_only,
            export_result=result["export_result"],
        )
        candidate_rows.append(summary_row)
        candidate_results[_lambda_key(lambda_value)] = result
        _write_csv(summary_csv, candidate_rows, columns=SUMMARY_COLUMNS)

    best_row = dict(_select_best_row(candidate_rows))
    best_lambda = float(best_row["lambda_tail_margin"])
    for row in candidate_rows:
        row["selected"] = bool(float(row["lambda_tail_margin"]) == best_lambda)

    best_result = candidate_results[_lambda_key(best_lambda)]
    final_dir = seed_dir / _selected_final_dir_name(best_lambda)
    final_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = final_dir / "stage2_checkpoint.pt"
    shutil.copy2(Path(best_result["stage2_checkpoint"]), final_checkpoint)

    validate_split_labels(paths.test_csv, num_classes=num_classes, name="test")
    validate_split_bucket_coverage(paths.test_csv, label_buckets, split_name="test")

    final_split_results: dict[str, Any] = {}
    final_split_export_results: dict[str, Any] = {}
    final_export_hashes: dict[str, str] = {}
    for split_name, split_csv in (("eval", paths.eval_csv), ("test", paths.test_csv)):
        export_result = _export_split(
            checkpoint_path=final_checkpoint,
            split_name=split_name,
            split_csv=split_csv,
            output_dir=final_dir / f"{split_name}_export",
            label_mapping_path=paths.label_mapping,
            label_buckets=label_buckets,
            args=args,
            reference_root=reference_root,
        )
        final_split_results[split_name] = export_result["split"]
        final_split_export_results[split_name] = export_result
        final_export_hashes[split_name] = str(export_result.get("run_identity_hash") or "")

    final_metrics_csv = final_dir / "stage2_class_metrics.csv"
    final_group_metrics_csv = final_dir / "stage2_group_metrics.csv"
    final_geometry_metrics_csv = final_dir / "stage2_geometry_metrics.csv"
    final_diagnostic_metrics_csv = final_dir / "diagnostic_token_export_class_metrics.csv"
    final_diagnostic_group_metrics_csv = final_dir / "diagnostic_token_export_group_metrics.csv"
    final_diagnostic_geometry_metrics_csv = final_dir / "diagnostic_token_export_geometry_metrics.csv"
    final_summary_csv = final_dir / "tail_repair_selection_summary.csv"
    final_export_manifest = final_dir / "final_export_manifest.json"
    final_export_parity_sanity: dict[str, Any] = {}
    for split_name, export_result in final_split_export_results.items():
        sanity = _export_parity_sanity_check(
            checkpoint_path=final_checkpoint,
            split_name=split_name,
            export_result=export_result,
            args=args,
            paths=paths,
            label_buckets=label_buckets,
            label_names=label_names,
            num_classes=num_classes,
            device=device,
            hard_fail_on_prediction_mismatch=True,
        )
        final_export_parity_sanity[split_name] = sanity
        warning = str(sanity.get("warning") or "")
        print(
            "[tail-repair] final_export_parity_sanity "
            f"split={split_name} pred_mismatch={int(sanity['prediction_mismatch_count'])} "
            f"max_abs_logit_diff={float(sanity['max_abs_logit_diff']):.9g} "
            f"max_rel_logit_diff={float(sanity['max_rel_logit_diff']):.9g}"
        )
        if warning:
            print(f"[tail-repair][warning] final {split_name} {warning}: actual export path logits differ from canonical reload path")
    final_official_evals: dict[str, Any] = {}
    for split_name in ("eval", "test"):
        final_official_evals[split_name] = _canonical_reload_eval_checkpoint(
            checkpoint_path=final_checkpoint,
            split_name=split_name,
            args=args,
            paths=paths,
            label_buckets=label_buckets,
            label_names=label_names,
            num_classes=num_classes,
            device=device,
            token_layer_indices=(),
            prediction_path=OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
            include_logits=False,
        )
    _write_canonical_final_metric_tables(
        split_evals=final_official_evals,
        checkpoint_path=final_checkpoint,
        label_buckets=label_buckets,
        metrics_path=final_metrics_csv,
        group_metrics_path=final_group_metrics_csv,
        geometry_metrics_path=final_geometry_metrics_csv,
    )
    _write_combined_export_table(final_split_results, table_key="metrics", path=final_diagnostic_metrics_csv)
    _write_combined_export_table(final_split_results, table_key="group_metrics", path=final_diagnostic_group_metrics_csv)
    _write_combined_export_table(final_split_results, table_key="geometry_metrics", path=final_diagnostic_geometry_metrics_csv)

    for row in candidate_rows:
        if float(row["lambda_tail_margin"]) != best_lambda:
            continue
        row["final_test_exported"] = True
        row["final_export_splits"] = "eval,test"
        row["final_dir"] = path_str(final_dir)
        row["final_checkpoint"] = path_str(final_checkpoint)
        row["final_metrics_csv"] = path_str(final_metrics_csv)
        row["final_group_metrics_csv"] = path_str(final_group_metrics_csv)
        row["final_geometry_metrics_csv"] = path_str(final_geometry_metrics_csv)
        row["final_export_manifest"] = path_str(final_export_manifest)
        row["final_export_run_identity_hash"] = json.dumps(final_export_hashes, sort_keys=True)
    _write_csv(summary_csv, candidate_rows, columns=SUMMARY_COLUMNS)
    shutil.copy2(summary_csv, final_summary_csv)

    selected_summary = next(row for row in candidate_rows if bool(row.get("selected")))
    final_manifest = {
        "protocol_version": TAIL_REPAIR_PROTOCOL_VERSION,
        "artifact_type": "tail_repair_selected_final_export",
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "sweep_mode": False,
        "selection_split": "eval",
        "export_splits": ["eval", "test"],
        "final_export_splits": ["eval", "test"],
        "test_split_touched": True,
        "seed": int(args.seed),
        "lambda_tail_margin": best_lambda,
        "gamma_t": float(gamma_payload["gamma_t"]),
        "selection_policy": "preservation-constrained tail-first selection",
        "selection_metric": SELECTION_METRIC,
        "selection_metric_value": float(selected_summary["selection_metric_value"]),
        "guardrails": {
            "delta_overall_accuracy": OVERALL_ACC_GUARDRAIL,
            "delta_overall_macro_f1": OVERALL_MACRO_F1_GUARDRAIL,
            "delta_many_accuracy": MANY_ACC_GUARDRAIL,
            "geometry_hard_gate": False,
        },
        "source_candidate_dir": path_str(best_result["lambda_dir"]),
        "source_candidate_checkpoint": path_str(best_result["stage2_checkpoint"]),
        "source_candidate_checkpoint_sha256": file_sha256(best_result["stage2_checkpoint"]),
        "stage2_checkpoint": path_str(final_checkpoint),
        "stage2_checkpoint_sha256": file_sha256(final_checkpoint),
        "stage2_class_metrics": path_str(final_metrics_csv),
        "stage2_group_metrics": path_str(final_group_metrics_csv),
        "stage2_geometry_metrics": path_str(final_geometry_metrics_csv),
        "official_classification_metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
        "diagnostic_export_metric_source": DIAGNOSTIC_EXPORT_METRIC_SOURCE,
        "diagnostic_token_export_class_metrics": path_str(final_diagnostic_metrics_csv),
        "diagnostic_token_export_group_metrics": path_str(final_diagnostic_group_metrics_csv),
        "diagnostic_token_export_geometry_metrics": path_str(final_diagnostic_geometry_metrics_csv),
        "tail_repair_selection_summary": path_str(final_summary_csv),
        "tail_repair_sweep_eval_summary": path_str(summary_csv),
        "official_split_evals": {split: _canonical_eval_manifest_summary(payload) for split, payload in final_official_evals.items()},
        "final_export_parity_sanity": final_export_parity_sanity,
        "split_results": final_split_results,
        "diagnostic_split_results": final_split_results,
        "artifact_sha256": {
            "stage2_checkpoint.pt": file_sha256(final_checkpoint),
            "stage2_class_metrics.csv": file_sha256(final_metrics_csv),
            "stage2_group_metrics.csv": file_sha256(final_group_metrics_csv),
            "stage2_geometry_metrics.csv": file_sha256(final_geometry_metrics_csv),
            "diagnostic_token_export_class_metrics.csv": file_sha256(final_diagnostic_metrics_csv),
            "diagnostic_token_export_group_metrics.csv": file_sha256(final_diagnostic_group_metrics_csv),
            "diagnostic_token_export_geometry_metrics.csv": file_sha256(final_diagnostic_geometry_metrics_csv),
            "tail_repair_selection_summary.csv": file_sha256(final_summary_csv),
        },
    }
    final_manifest = {**final_manifest, "run_manifest_hash": canonical_json_hash(final_manifest)}
    _write_json(final_export_manifest, final_manifest)

    manifest_identity = {
        "protocol_version": TAIL_REPAIR_PROTOCOL_VERSION,
        "dataset_name": paths.name,
        "seed": int(args.seed),
        "stage1_checkpoint_sha256": file_sha256(stage1_checkpoint),
        "lambda_values": [float(item) for item in args.lambda_values],
        "gamma_t": float(gamma_payload["gamma_t"]),
        "ce_only_metrics_sha256": file_sha256(ce_metrics_path),
        "best_lambda_tail_margin": best_lambda,
        "method_variant": METHOD_VARIANT,
    }
    manifest = {
        **manifest_identity,
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "sweep_identity_hash": canonical_json_hash(manifest_identity),
        "output_seed_dir": path_str(seed_dir),
        "stage1_checkpoint": path_str(stage1_checkpoint),
        "reference_root": path_str(reference_root),
        "ce_only_stage2_eval_metrics": path_str(ce_metrics_path),
        "ce_only_guardrail_reference": ce_only,
        "summary_csv": path_str(summary_csv),
        "summary_csv_sha256": file_sha256(summary_csv),
        "best_lambda_tail_margin": best_lambda,
        "best_lambda_dir": path_str(best_result["lambda_dir"]),
        "best_stage2_checkpoint": path_str(best_result["stage2_checkpoint"]),
        "best_stage2_checkpoint_sha256": file_sha256(best_result["stage2_checkpoint"]),
        "final_dir": path_str(final_dir),
        "final_stage2_checkpoint": path_str(final_checkpoint),
        "final_stage2_checkpoint_sha256": file_sha256(final_checkpoint),
        "final_export_manifest": path_str(final_export_manifest),
        "final_export_manifest_sha256": file_sha256(final_export_manifest),
        "selection_policy": {
            "guardrails": {
                "delta_overall_accuracy": OVERALL_ACC_GUARDRAIL,
                "delta_overall_macro_f1": OVERALL_MACRO_F1_GUARDRAIL,
                "delta_many_accuracy": MANY_ACC_GUARDRAIL,
            },
            "geometry_hard_gate": False,
            "primary": "tail_macro_f1",
            "tie_break": ["tail_accuracy", "medium_macro_f1", "medium_accuracy"],
        },
        "candidate_rng_policy": {
            "rng_reset_per_candidate": True,
            "candidate_rng_seed": int(args.seed),
            "reset_point": "before loading Stage1 checkpoint and constructing train/eval loaders for each lambda",
        },
        "candidates": candidate_rows,
        "final_export": {
            "policy": "test split exported once after eval lambda selection",
            "splits": ["eval", "test"],
            "lambda_tail_margin": best_lambda,
            "checkpoint": path_str(final_checkpoint),
            "official_classification_metric_source": OFFICIAL_CLASSIFICATION_METRIC_SOURCE,
            "metrics_csv": path_str(final_metrics_csv),
            "group_metrics_csv": path_str(final_group_metrics_csv),
            "geometry_metrics_csv": path_str(final_geometry_metrics_csv),
            "manifest": path_str(final_export_manifest),
            "official_split_evals": {split: _canonical_eval_manifest_summary(payload) for split, payload in final_official_evals.items()},
            "export_parity_sanity": final_export_parity_sanity,
            "diagnostic_export_metric_source": DIAGNOSTIC_EXPORT_METRIC_SOURCE,
            "diagnostic_metrics_csv": path_str(final_diagnostic_metrics_csv),
            "diagnostic_group_metrics_csv": path_str(final_diagnostic_group_metrics_csv),
            "diagnostic_geometry_metrics_csv": path_str(final_diagnostic_geometry_metrics_csv),
            "split_results": final_split_results,
            "diagnostic_split_results": final_split_results,
        },
        "artifact_sha256": {
            "tail_repair_sweep_eval_summary.csv": file_sha256(summary_csv),
            "best_stage2_checkpoint.pt": file_sha256(best_result["stage2_checkpoint"]),
            "final_stage2_checkpoint.pt": file_sha256(final_checkpoint),
            "final_stage2_class_metrics.csv": file_sha256(final_metrics_csv),
            "final_stage2_group_metrics.csv": file_sha256(final_group_metrics_csv),
            "final_stage2_geometry_metrics.csv": file_sha256(final_geometry_metrics_csv),
            "final_diagnostic_token_export_class_metrics.csv": file_sha256(final_diagnostic_metrics_csv),
            "final_diagnostic_token_export_group_metrics.csv": file_sha256(final_diagnostic_group_metrics_csv),
            "final_diagnostic_token_export_geometry_metrics.csv": file_sha256(final_diagnostic_geometry_metrics_csv),
            "final_tail_repair_selection_summary.csv": file_sha256(final_summary_csv),
            "final_export_manifest.json": file_sha256(final_export_manifest),
        },
    }
    _write_json(manifest_path, manifest)
    print(f"[tail-repair] summary={summary_csv}")
    print(f"[tail-repair] best_lambda={best_lambda:g}")
    print(f"[tail-repair] final_dir={final_dir}")
    print(f"[tail-repair] manifest={manifest_path}")
    return {"manifest": manifest, "summary_csv": summary_csv, "best_lambda": best_lambda}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_tail_repair_sweep(args)


if __name__ == "__main__":
    main()
