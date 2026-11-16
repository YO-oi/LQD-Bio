import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


STAGE1_SELECTION_METRIC = "eval_macro_f1_label_universe_nanmean"

STAGE1_EVAL_METRIC_COLUMNS = [
    "seed",
    "stage",
    "stage_epoch",
    "global_epoch",
    "stage1_checkpoint",
    "stage1_checkpoint_sha256",
    "stage1_selection_metric",
    "stage1_selection_metric_value",
    "stage1_train_total_loss",
    "stage1_train_ce_loss",
    "stage1_train_kd_loss",
    "stage1_train_kl_loss",
    "stage1_train_temperature",
    "stage1_train_alpha",
    "stage1_eval_accuracy",
    "stage1_eval_macro_f1_label_universe_nanmean",
    "stage1_eval_many_accuracy",
    "stage1_eval_medium_accuracy",
    "stage1_eval_few_accuracy",
    "stage1_eval_tail_accuracy",
    "stage1_embedding_distortion_to_fp32",
    "stage1_embedding_cosine_similarity_to_fp32",
    "stage1_correct_logit_margin",
    "stage1_logit_margin_retention_to_fp32",
    "stage1_fisher",
    "stage1_fisher_retention_to_fp32",
    "stage1_geo_margin_main",
    "stage1_geo_margin_retention_to_fp32",
    "stage1_class_trace_variance",
    "stage1_tail_collapse_gap",
    "stage1_backbone_lr",
    "stage1_classifier_lr",
    "stage1_backbone_trainable_param_count",
    "stage1_classifier_trainable_param_count",
    "stage1_teacher_kd_enabled",
    "stage1_qparams_frozen",
    "stage1_fake_quant_enabled",
    "stage1_observer_enabled",
    "stage1_reference_checkpoint_hash",
    "stage1_export_run_identity_hash",
]

_METRIC_SPECS = {
    "stage1_eval_accuracy": ("metrics", "accuracy", "global"),
    "stage1_eval_macro_f1_label_universe_nanmean": ("metrics", "macro_f1_label_universe_nanmean", "global"),
    "stage1_eval_many_accuracy": ("group_metrics", "accuracy", "many"),
    "stage1_eval_medium_accuracy": ("group_metrics", "accuracy", "medium"),
    "stage1_eval_few_accuracy": ("group_metrics", "accuracy", "few"),
    "stage1_eval_tail_accuracy": ("group_metrics", "accuracy", "tail"),
    "stage1_embedding_distortion_to_fp32": ("metrics", "embedding_distortion", "global"),
    "stage1_embedding_cosine_similarity_to_fp32": ("metrics", "embedding_cosine_similarity", "global"),
    "stage1_correct_logit_margin": ("geometry_metrics", "correct_logit_margin_detail", "global"),
    "stage1_logit_margin_retention_to_fp32": ("metrics", "logit_margin_retention", "global"),
    "stage1_fisher": ("geometry_metrics", "fisher", "global"),
    "stage1_fisher_retention_to_fp32": ("metrics", "fisher_retention", "global"),
    "stage1_geo_margin_main": ("geometry_metrics", "geo_margin_main", "global"),
    "stage1_geo_margin_retention_to_fp32": ("metrics", "geo_margin_retention", "global"),
    "stage1_class_trace_variance": ("geometry_metrics", "class_trace_variance", "global"),
    "stage1_tail_collapse_gap": ("metrics", "tail_collapse_gap", "tail_vs_many"),
}


def _read_metric_table(path: str | Path, *, table_name: str) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Stage 1 metric table is missing: {table_name}={target}")
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError(f"Stage 1 metric table is empty: {table_name}={target}")
    return rows


def _as_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if math.isinf(parsed):
        raise ValueError(f"{name} must be finite or NaN, got {value!r}")
    return parsed


def _metric_value(tables: Mapping[str, Sequence[Mapping[str, Any]]], *, table: str, metric_name: str, analysis_group: str) -> float:
    rows = tables.get(table)
    if rows is None:
        raise KeyError(f"Stage 1 export result is missing table {table!r}")
    matches = [
        row
        for row in rows
        if str(row.get("metric_name")) == str(metric_name)
        and str(row.get("analysis_group")) == str(analysis_group)
    ]
    if not matches:
        raise KeyError(
            "Stage 1 metric is missing from export_eval output: "
            f"table={table}, metric_name={metric_name}, analysis_group={analysis_group}"
        )
    return _as_float(matches[0].get("metric_value"), name=f"{table}:{metric_name}:{analysis_group}")


def _trainable_param_count(stage_contract: Mapping[str, Any], group_name: str) -> int:
    status = stage_contract.get("parameter_status")
    if not isinstance(status, Mapping):
        raise KeyError("Stage 1 contract missing parameter_status")
    group = status.get(group_name)
    if not isinstance(group, Mapping):
        raise KeyError(f"Stage 1 contract missing parameter_status.{group_name}")
    return int(group.get("trainable_num_params", 0))


def build_stage1_eval_metric_row(
    *,
    split_result: Mapping[str, Any],
    best_train_row: Mapping[str, Any],
    stage_contract: Mapping[str, Any],
    quant_audit: Mapping[str, Any],
    seed: int,
    stage_epoch: int,
    global_epoch: int,
    checkpoint_path: str | Path,
    checkpoint_sha256: str | None,
    alpha: float,
    temperature: float,
) -> dict[str, Any]:
    paths = split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("Stage 1 export result missing paths")
    tables = {
        "metrics": _read_metric_table(paths["metrics"], table_name="metrics"),
        "group_metrics": _read_metric_table(paths["group_metrics"], table_name="group_metrics"),
        "geometry_metrics": _read_metric_table(paths["geometry_metrics"], table_name="geometry_metrics"),
    }

    row: dict[str, Any] = {
        "seed": int(seed),
        "stage": "stage1_semantic_manifold",
        "stage_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "stage1_checkpoint": str(Path(checkpoint_path).resolve(strict=False)),
        "stage1_checkpoint_sha256": "" if checkpoint_sha256 is None else str(checkpoint_sha256),
        "stage1_selection_metric": STAGE1_SELECTION_METRIC,
        "stage1_train_total_loss": _as_float(best_train_row.get("train_loss"), name="stage1 train_loss"),
        "stage1_train_ce_loss": _as_float(best_train_row.get("train_ce_loss"), name="stage1 train_ce_loss"),
        "stage1_train_kd_loss": _as_float(best_train_row.get("train_kd_term"), name="stage1 train_kd_term"),
        "stage1_train_kl_loss": _as_float(best_train_row.get("train_kl_loss"), name="stage1 train_kl_loss"),
        "stage1_train_temperature": float(temperature),
        "stage1_train_alpha": float(alpha),
        "stage1_backbone_lr": _as_float(stage_contract.get("backbone_lr"), name="stage1 backbone_lr"),
        "stage1_classifier_lr": _as_float(stage_contract.get("classifier_lr"), name="stage1 classifier_lr"),
        "stage1_backbone_trainable_param_count": _trainable_param_count(stage_contract, "backbone"),
        "stage1_classifier_trainable_param_count": _trainable_param_count(stage_contract, "classifier"),
        "stage1_teacher_kd_enabled": bool(stage_contract.get("kd_loss_enabled")),
        "stage1_qparams_frozen": bool(quant_audit.get("qparams_frozen")),
        "stage1_fake_quant_enabled": bool(quant_audit.get("fake_quant_enabled")),
        "stage1_observer_enabled": bool(quant_audit.get("observer_enabled")),
        "stage1_reference_checkpoint_hash": str(split_result.get("reference_checkpoint_hash") or ""),
        "stage1_export_run_identity_hash": str(split_result.get("run_identity_hash") or ""),
    }
    for output_name, (table_name, metric_name, analysis_group) in _METRIC_SPECS.items():
        row[output_name] = _metric_value(tables, table=table_name, metric_name=metric_name, analysis_group=analysis_group)
    row["stage1_selection_metric_value"] = row["stage1_eval_macro_f1_label_universe_nanmean"]

    if row["stage1_backbone_lr"] <= 0.0:
        raise RuntimeError(f"Stage 1 audit violation: backbone lr must be > 0, got {row['stage1_backbone_lr']!r}")
    if not bool(row["stage1_teacher_kd_enabled"]):
        raise RuntimeError("Stage 1 audit violation: KD must be enabled for the best checkpoint row")
    if row["stage1_backbone_trainable_param_count"] <= 0:
        raise RuntimeError("Stage 1 audit violation: backbone must have trainable parameters")
    if row["stage1_classifier_trainable_param_count"] <= 0:
        raise RuntimeError("Stage 1 audit violation: classifier must have trainable parameters")
    if not row["stage1_reference_checkpoint_hash"]:
        raise RuntimeError("Stage 1 FP32 reference alignment metrics were requested, but export_eval did not load a reference bundle")
    return row


def write_stage1_eval_metrics(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(STAGE1_EVAL_METRIC_COLUMNS)
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(str(key))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
