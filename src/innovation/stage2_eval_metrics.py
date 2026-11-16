import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.baseline.baseline_metrics import correct_logit_margin
from src.baseline.export_eval import load_fp32_reference_bundle
from src.metrics.common_metrics import DEFAULT_MARGIN_MIN_ABS, embedding_distortion, margin_retention


STAGE2_SELECTION_METRIC = "eval_macro_f1_label_universe_nanmean"

STAGE2_EVAL_METRIC_COLUMNS = [
    "seed",
    "stage",
    "stage2_epoch",
    "global_epoch",
    "stage2_checkpoint",
    "stage2_checkpoint_sha256",
    "stage2_selection_metric",
    "stage2_selection_metric_value",
    "stage2_tau",
    "stage2_tau_min",
    "stage2_tau_max",
    "stage2_class_prior_hash",
    "stage2_log_prior_min",
    "stage2_log_prior_max",
    "stage2_logit_adjustment_enabled",
    "stage2_train_ce_loss",
    "stage2_train_total_loss",
    "stage2_train_kd_loss",
    "stage2_teacher_kd_enabled",
    "stage2_eval_accuracy",
    "stage2_eval_macro_f1_label_universe_nanmean",
    "stage2_eval_many_accuracy",
    "stage2_eval_medium_accuracy",
    "stage2_eval_few_accuracy",
    "stage2_eval_tail_accuracy",
    "stage2_eval_per_class_recall",
    "stage2_correct_logit_margin",
    "stage2_correct_logit_margin_retention_to_fp32",
    "stage2_many_margin_retention",
    "stage2_medium_margin_retention",
    "stage2_few_margin_retention",
    "stage2_tail_margin_retention",
    "stage2_backbone_lr",
    "stage2_classifier_lr",
    "stage2_backbone_trainable_param_count",
    "stage2_classifier_trainable_param_count",
    "stage2_embedding_drift_from_stage1",
    "stage2_reference_checkpoint_hash",
    "stage2_export_run_identity_hash",
]

_METRIC_SPECS = {
    "stage2_eval_accuracy": ("metrics", "accuracy", "global"),
    "stage2_eval_macro_f1_label_universe_nanmean": ("metrics", "macro_f1_label_universe_nanmean", "global"),
    "stage2_eval_many_accuracy": ("group_metrics", "accuracy", "many"),
    "stage2_eval_medium_accuracy": ("group_metrics", "accuracy", "medium"),
    "stage2_eval_few_accuracy": ("group_metrics", "accuracy", "few"),
    "stage2_eval_tail_accuracy": ("group_metrics", "accuracy", "tail"),
    "stage2_correct_logit_margin": ("geometry_metrics", "correct_logit_margin_detail", "global"),
    "stage2_correct_logit_margin_retention_to_fp32": ("metrics", "logit_margin_retention", "global"),
}


def _read_metric_table(path: str | Path, *, table_name: str) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Stage 2 metric table is missing: {table_name}={target}")
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError(f"Stage 2 metric table is empty: {table_name}={target}")
    return rows


def _as_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if math.isinf(parsed):
        raise ValueError(f"{name} must be finite or NaN, got {value!r}")
    return parsed


def _metric_value(tables: Mapping[str, Sequence[Mapping[str, Any]]], *, table: str, metric_name: str, analysis_group: str) -> float:
    rows = tables.get(table)
    if rows is None:
        raise KeyError(f"Stage 2 export result is missing table {table!r}")
    matches = [
        row
        for row in rows
        if str(row.get("metric_name")) == str(metric_name)
        and str(row.get("analysis_group")) == str(analysis_group)
    ]
    if not matches:
        raise KeyError(
            "Stage 2 metric is missing from export_eval output: "
            f"table={table}, metric_name={metric_name}, analysis_group={analysis_group}"
        )
    return _as_float(matches[0].get("metric_value"), name=f"{table}:{metric_name}:{analysis_group}")


def _trainable_param_count(stage_contract: Mapping[str, Any], group_name: str) -> int:
    status = stage_contract.get("parameter_status")
    if not isinstance(status, Mapping):
        raise KeyError("Stage 2 contract missing parameter_status")
    group = status.get(group_name)
    if not isinstance(group, Mapping):
        raise KeyError(f"Stage 2 contract missing parameter_status.{group_name}")
    return int(group.get("trainable_num_params", 0))


def _load_tensor_payload(path: str | Path) -> Mapping[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"tensor artifact missing: {target}")
    payload = torch.load(target, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"tensor artifact must contain a mapping: {target}")
    return payload


def _payload_tensor(payload: Mapping[str, Any], keys: Sequence[str], *, name: str) -> torch.Tensor:
    for key in keys:
        if key in payload:
            tensor = torch.as_tensor(payload[key]).detach().cpu().float().contiguous()
            if tensor.ndim != 2:
                raise ValueError(f"{name} must be 2D, got shape={tuple(tensor.shape)}")
            return tensor
    raise KeyError(f"payload missing tensor key for {name}: {keys}")


def _payload_labels(payload: Mapping[str, Any]) -> torch.Tensor:
    for key in ("labels", "label_id"):
        if key in payload:
            labels = torch.as_tensor(payload[key]).detach().cpu().long().reshape(-1)
            if int(labels.numel()) <= 0:
                raise ValueError("payload labels are empty")
            return labels
    raise KeyError("payload missing labels/label_id")


def _payload_sample_ids(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("sample_ids", payload.get("sample_id"))
    if isinstance(raw, torch.Tensor):
        return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    raise KeyError("payload missing sample_ids/sample_id")


def _mean_finite(values: torch.Tensor) -> float:
    values = values.detach().cpu().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if int(finite.numel()) <= 0:
        return float("nan")
    return float(finite.mean().item())


def _bucket_mask(labels: torch.Tensor, label_buckets: Mapping[int, str], group_name: str) -> torch.Tensor:
    buckets = [str(label_buckets.get(int(label.item()), "unknown")) for label in labels.reshape(-1)]
    if str(group_name) == "tail":
        mask = [bucket in {"medium", "few"} for bucket in buckets]
    else:
        mask = [bucket == str(group_name) for bucket in buckets]
    return torch.tensor(mask, dtype=torch.bool)


def _per_class_recall_map(per_class_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in per_class_rows:
        if str(row.get("metric_name")) != "per_class_recall":
            continue
        label_id = row.get("label_id")
        if label_id is None or str(label_id) == "":
            group = str(row.get("analysis_group") or "")
            label_id = group.split("class:", 1)[1] if group.startswith("class:") else group
        result[str(label_id)] = _as_float(row.get("metric_value"), name=f"per_class_recall:{label_id}")
    if not result:
        raise RuntimeError("Stage 2 per-class recall rows are missing")
    return result


def _embedding_drift_from_stage1(stage1_split_result: Mapping[str, Any], stage2_split_result: Mapping[str, Any]) -> float:
    stage1_paths = stage1_split_result.get("paths")
    stage2_paths = stage2_split_result.get("paths")
    if not isinstance(stage1_paths, Mapping) or not isinstance(stage2_paths, Mapping):
        raise KeyError("Stage 1/Stage 2 split results must contain paths")
    stage1_payload = _load_tensor_payload(stage1_paths["embeddings"])
    stage2_payload = _load_tensor_payload(stage2_paths["embeddings"])
    if _payload_sample_ids(stage1_payload) != _payload_sample_ids(stage2_payload):
        raise ValueError("Stage 1 and Stage 2 embedding artifacts have different sample order")
    stage1_embedding = _payload_tensor(stage1_payload, ("embeddings", "embedding"), name="stage1 embeddings")
    stage2_embedding = _payload_tensor(stage2_payload, ("embeddings", "embedding"), name="stage2 embeddings")
    drift = embedding_distortion(stage1_embedding, stage2_embedding, reduction="none")
    return _mean_finite(torch.as_tensor(drift))


def _group_margin_retention(
    *,
    stage2_split_result: Mapping[str, Any],
    reference_root: str | Path,
    label_buckets: Mapping[int, str],
) -> dict[str, float]:
    paths = stage2_split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("Stage 2 split result missing paths")
    logits_payload = _load_tensor_payload(paths["logits"])
    labels = _payload_labels(logits_payload)
    logits = _payload_tensor(logits_payload, ("logits",), name="stage2 logits")
    reference = load_fp32_reference_bundle(reference_root, split=str(stage2_split_result.get("split") or "eval"))
    if _payload_sample_ids(logits_payload) != [str(item) for item in reference.sample_ids]:
        raise ValueError("Stage 2 logits and FP32 reference sample order differ")
    if tuple(labels.shape) != tuple(reference.labels.shape) or not bool(torch.equal(labels, reference.labels.long().reshape(-1))):
        raise ValueError("Stage 2 logits labels differ from FP32 reference labels")
    retention = margin_retention(
        reference.logits,
        logits,
        label_id=labels,
        reduction="none",
        min_abs_margin=float(DEFAULT_MARGIN_MIN_ABS),
    )
    retention = torch.as_tensor(retention).detach().cpu().float().reshape(-1)
    result = {"stage2_correct_logit_margin": _mean_finite(correct_logit_margin(logits, labels))}
    for group_name in ("many", "medium", "few", "tail"):
        mask = _bucket_mask(labels, label_buckets, group_name)
        result[f"stage2_{group_name}_margin_retention"] = _mean_finite(retention[mask])
    return result


def build_stage2_eval_metric_row(
    *,
    split_result: Mapping[str, Any],
    stage1_split_result: Mapping[str, Any],
    best_train_row: Mapping[str, Any],
    stage_contract: Mapping[str, Any],
    adjustment_audit: Mapping[str, Any],
    label_buckets: Mapping[int, str],
    reference_root: str | Path,
    seed: int,
    stage_epoch: int,
    global_epoch: int,
    checkpoint_path: str | Path,
    checkpoint_sha256: str | None,
) -> dict[str, Any]:
    paths = split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("Stage 2 export result missing paths")
    tables = {
        "metrics": _read_metric_table(paths["metrics"], table_name="metrics"),
        "group_metrics": _read_metric_table(paths["group_metrics"], table_name="group_metrics"),
        "geometry_metrics": _read_metric_table(paths["geometry_metrics"], table_name="geometry_metrics"),
        "per_class_metrics": _read_metric_table(paths["per_class_metrics"], table_name="per_class_metrics"),
    }
    row: dict[str, Any] = {
        "seed": int(seed),
        "stage": "stage2_decision_boundary",
        "stage2_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "stage2_checkpoint": str(Path(checkpoint_path).resolve(strict=False)),
        "stage2_checkpoint_sha256": "" if checkpoint_sha256 is None else str(checkpoint_sha256),
        "stage2_selection_metric": STAGE2_SELECTION_METRIC,
        "stage2_tau": _as_float(adjustment_audit.get("stage2_tau"), name="stage2_tau"),
        "stage2_tau_min": _as_float(adjustment_audit.get("stage2_tau_min"), name="stage2_tau_min"),
        "stage2_tau_max": _as_float(adjustment_audit.get("stage2_tau_max"), name="stage2_tau_max"),
        "stage2_class_prior_hash": str(adjustment_audit.get("stage2_class_prior_hash") or ""),
        "stage2_log_prior_min": _as_float(adjustment_audit.get("stage2_log_prior_min"), name="stage2_log_prior_min"),
        "stage2_log_prior_max": _as_float(adjustment_audit.get("stage2_log_prior_max"), name="stage2_log_prior_max"),
        "stage2_logit_adjustment_enabled": bool(adjustment_audit.get("stage2_logit_adjustment_enabled")),
        "stage2_train_ce_loss": _as_float(best_train_row.get("train_ce_loss"), name="stage2 train_ce_loss"),
        "stage2_train_total_loss": _as_float(best_train_row.get("train_loss"), name="stage2 train_loss"),
        "stage2_train_kd_loss": 0.0,
        "stage2_teacher_kd_enabled": False,
        "stage2_eval_per_class_recall": json.dumps(
            _per_class_recall_map(tables["per_class_metrics"]),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "stage2_backbone_lr": _as_float(stage_contract.get("backbone_lr"), name="stage2 backbone_lr"),
        "stage2_classifier_lr": _as_float(stage_contract.get("classifier_lr"), name="stage2 classifier_lr"),
        "stage2_backbone_trainable_param_count": _trainable_param_count(stage_contract, "backbone"),
        "stage2_classifier_trainable_param_count": _trainable_param_count(stage_contract, "classifier"),
        "stage2_embedding_drift_from_stage1": _embedding_drift_from_stage1(stage1_split_result, split_result),
        "stage2_reference_checkpoint_hash": str(split_result.get("reference_checkpoint_hash") or ""),
        "stage2_export_run_identity_hash": str(split_result.get("run_identity_hash") or ""),
    }
    for output_name, (table_name, metric_name, analysis_group) in _METRIC_SPECS.items():
        row[output_name] = _metric_value(tables, table=table_name, metric_name=metric_name, analysis_group=analysis_group)
    row.update(
        {
            key: value
            for key, value in _group_margin_retention(
                stage2_split_result=split_result,
                reference_root=reference_root,
                label_buckets=label_buckets,
            ).items()
            if key not in row
        }
    )
    row["stage2_selection_metric_value"] = row["stage2_eval_macro_f1_label_universe_nanmean"]

    if row["stage2_backbone_lr"] != 0.0:
        raise RuntimeError(f"Stage 2 audit violation: backbone lr must be 0, got {row['stage2_backbone_lr']!r}")
    if row["stage2_backbone_trainable_param_count"] != 0:
        raise RuntimeError("Stage 2 audit violation: backbone trainable param count must be 0")
    if bool(row["stage2_teacher_kd_enabled"]):
        raise RuntimeError("Stage 2 audit violation: KD must be disabled")
    if not bool(row["stage2_logit_adjustment_enabled"]):
        raise RuntimeError("Stage 2 audit violation: logit adjustment must be enabled")
    if not row["stage2_class_prior_hash"]:
        raise RuntimeError("Stage 2 audit violation: class prior hash is missing")
    if not row["stage2_reference_checkpoint_hash"]:
        raise RuntimeError("Stage 2 FP32 reference metrics were requested, but export_eval did not load a reference bundle")
    return row


def write_stage2_eval_metrics(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(STAGE2_EVAL_METRIC_COLUMNS)
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(str(key))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
