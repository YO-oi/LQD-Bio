from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.utils.group_utils import bucket_from_train_label_n, group_protocol_metadata, normalize_bucket


PROTOCOL_VERSION = "geometry_task_coupling_audit.v2"
EXPERIMENT_NAME = "geometry_task_coupling_audit"
ANALYSIS_NAME = "Geometry-Task Coupling Audit"
ANALYSIS_SPLIT = "eval"
REFERENCE_MODEL = "vanilla_kd"
DEFAULT_SEED = 2026
MAIN_GEOMETRY_GROUP = "global"
TAIL_COLLAPSE_GROUP = "tail_vs_many"
FULL_PROTOCOL_STATUS = "included_full_protocol"
PARTIAL_PROTOCOL_STATUS = "included_partial_metric_only"
EXCLUDED_PROTOCOL_STATUS = "excluded"
PAIRWISE_CANDIDATE_SCOPE = "metric_pair_valid_candidates_only"

TASK_METRICS = {
    "overall_accuracy": ("accuracy", "global"),
    "macro_f1": ("macro_f1_label_universe_nanmean", "global"),
    "many_accuracy": ("accuracy", "many"),
    "many_macro_f1": ("macro_f1_label_universe_nanmean", "many"),
    "medium_accuracy": ("accuracy", "medium"),
    "medium_macro_f1": ("macro_f1_label_universe_nanmean", "medium"),
    "tail_accuracy": ("accuracy", "tail"),
    "tail_macro_f1": ("macro_f1_label_universe_nanmean", "tail"),
    "few_accuracy": ("accuracy", "few"),
    "few_macro_f1": ("macro_f1_label_universe_nanmean", "few"),
}

TASK_DELTA_COLUMNS = {
    "tail_accuracy": "delta_tail_accuracy",
    "tail_macro_f1": "delta_tail_macro_f1",
    "medium_accuracy": "delta_medium_accuracy",
    "medium_macro_f1": "delta_medium_macro_f1",
    "many_accuracy": "delta_many_accuracy",
    "overall_accuracy": "delta_overall_accuracy",
}

GEOMETRY_METRICS = {
    "embedding_distortion": {
        "names": ("embedding_distortion",),
        "group": MAIN_GEOMETRY_GROUP,
        "direction": "lower",
        "improvement_column": "embedding_distortion_improvement",
    },
    "embedding_cosine": {
        "names": ("embedding_cosine", "embedding_cosine_similarity"),
        "group": MAIN_GEOMETRY_GROUP,
        "direction": "higher",
        "improvement_column": "embedding_cosine_improvement",
    },
    "fisher_retention": {
        "names": ("fisher_retention",),
        "group": MAIN_GEOMETRY_GROUP,
        "direction": "higher",
        "improvement_column": "fisher_improvement",
    },
    "geometry_margin_retention": {
        "names": ("geometry_margin_retention", "geo_margin_retention"),
        "group": MAIN_GEOMETRY_GROUP,
        "direction": "higher",
        "improvement_column": "geometry_margin_improvement",
    },
    "variance_inflation_deviation": {
        "names": ("variance_inflation_deviation",),
        "group": MAIN_GEOMETRY_GROUP,
        "direction": "lower",
        "improvement_column": "variance_improvement",
    },
    "tail_collapse_gap": {
        "names": ("tail_collapse_gap",),
        "group": TAIL_COLLAPSE_GROUP,
        "direction": "lower",
        "improvement_column": "collapse_gap_improvement",
    },
}

SUMMARY_COLUMNS = [
    "candidate_key",
    "method",
    "hyperparameter",
    "candidate_status",
    "protocol_inclusion_status",
    "included_full_protocol",
    "included_partial_metric_only",
    "exclusion_reason",
    "geometry_metric_status",
    "task_metric_status",
    "checkpoint_sha256",
    "metrics_source_files",
    "manifest_path",
    "overall_accuracy",
    "macro_f1",
    "many_accuracy",
    "many_macro_f1",
    "medium_accuracy",
    "medium_macro_f1",
    "tail_accuracy",
    "tail_macro_f1",
    "few_accuracy",
    "few_macro_f1",
    "overall_correct",
    "total",
    "many_correct",
    "many_total",
    "medium_correct",
    "medium_total",
    "tail_correct",
    "tail_total",
    "few_correct",
    "few_total",
    "embedding_distortion",
    "embedding_cosine",
    "fisher_retention",
    "geometry_margin_retention",
    "variance_inflation_deviation",
    "tail_collapse_gap",
    "delta_tail_accuracy",
    "delta_tail_macro_f1",
    "delta_medium_accuracy",
    "delta_medium_macro_f1",
    "delta_many_accuracy",
    "delta_overall_accuracy",
    "delta_embedding_distortion",
    "delta_embedding_cosine",
    "delta_fisher_retention",
    "delta_geometry_margin_retention",
    "delta_variance_inflation_deviation",
    "delta_tail_collapse_gap",
    "embedding_distortion_improvement",
    "embedding_cosine_improvement",
    "fisher_improvement",
    "geometry_margin_improvement",
    "variance_improvement",
    "collapse_gap_improvement",
    "geometry_improvement_direction",
    "missing_task_metrics",
    "missing_geometry_metrics",
    "same_eval_split_as_reference",
    "same_label_mapping_as_reference",
    "same_sample_order_as_reference",
    "same_geometry_definitions_as_reference",
    "same_available_geometry_definitions_as_reference",
    "test_touched",
]

PAIRWISE_COLUMNS = [
    "geometry_metric",
    "task_metric",
    "candidate_scope",
    "n_valid_candidates",
    "pearson",
    "spearman",
    "n_geometry_improved",
    "n_geometry_improved_task_improved",
    "n_geometry_improved_task_worsened",
    "n_geometry_worsened_task_improved",
    "n_geometry_worsened_task_worsened",
    "n_geometry_worsened",
    "n_geometry_neutral",
    "n_task_neutral",
    "direction_consistency_rate",
    "coupling_status",
    "descriptive_only",
]

COUNTEREXAMPLE_COLUMNS = [
    "counterexample_type",
    "geometry_metric",
    "task_metric",
    "candidate_key",
    "method",
    "hyperparameter",
    "geometry_value",
    "reference_geometry_value",
    "geometry_improvement",
    "task_value",
    "reference_task_value",
    "task_delta",
    "note",
]


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    method: str
    hyperparameter: str
    metrics_paths: tuple[tuple[str, ...], ...]
    manifest_path: tuple[str, ...] | None = None
    supplemental: bool = False


@dataclass(frozen=True)
class MetricValue:
    value: float
    row: Mapping[str, Any] | None
    missing_reason: str = ""


def _dataset_root(dataset_root: str | Path | None = None) -> Path:
    if dataset_root is not None:
        return Path(dataset_root)
    current = getattr(project_config, "CURRENT_DATASET", None)
    if isinstance(current, Mapping) and current.get("ROOT"):
        return Path(current["ROOT"])
    active = str(getattr(project_config, "ACTIVE_DATASET", ""))
    datasets = getattr(project_config, "DATASETS", {})
    if isinstance(datasets, Mapping) and active in datasets:
        entry = datasets[active]
        if isinstance(entry, Mapping) and entry.get("ROOT"):
            return Path(entry["ROOT"])
    return Path(getattr(project_config, "BASE_DIR", PROJECT_ROOT)) / "local_data"


def _dataset_path(root: Path, key: str, fallback_parts: Sequence[str]) -> Path:
    current = getattr(project_config, "CURRENT_DATASET", None)
    if isinstance(current, Mapping) and current.get(key) is not None:
        configured_root = current.get("ROOT")
        if configured_root is None or Path(configured_root).resolve(strict=False) == Path(root).resolve(strict=False):
            return Path(current[key])
    return Path(root).joinpath(*fallback_parts)


def _candidate_specs(seed: int, *, include_supplemental: bool) -> list[CandidateSpec]:
    seed_dir = f"seed_{int(seed)}"
    specs = [
        CandidateSpec(
            key="vanilla_kd",
            method="Vanilla KD",
            hyperparameter="reference",
            metrics_paths=(
                ("baseline", "kd", seed_dir, "metrics", "vanilla_kd", "eval_metrics.csv"),
                ("baseline", "kd", seed_dir, "metrics", "vanilla_kd", "eval_group_metrics.csv"),
                ("baseline", "kd", seed_dir, "metrics", "vanilla_kd", "eval_geometry_metrics.csv"),
                ("baseline", "kd", seed_dir, "vanilla_kd_metrics.csv"),
            ),
            manifest_path=("baseline", "kd", seed_dir, "run_manifest.json"),
        ),
        CandidateSpec(
            key="qvf_beta_0p01",
            method="QVF",
            hyperparameter="beta=0.01",
            metrics_paths=(
                ("innovation", "QVF", seed_dir, "beta_0p01", "training_eval_export", "metrics", "vanilla_kd", "eval_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p01", "training_eval_export", "metrics", "vanilla_kd", "eval_group_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p01", "training_eval_export", "metrics", "vanilla_kd", "eval_geometry_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p01", "qvf_metrics.csv"),
            ),
            manifest_path=("innovation", "QVF", seed_dir, "beta_0p01", "qvf_manifest.json"),
        ),
        CandidateSpec(
            key="qvf_beta_0p05",
            method="QVF",
            hyperparameter="beta=0.05",
            metrics_paths=(
                ("innovation", "QVF", seed_dir, "beta_0p05", "training_eval_export", "metrics", "vanilla_kd", "eval_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p05", "training_eval_export", "metrics", "vanilla_kd", "eval_group_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p05", "training_eval_export", "metrics", "vanilla_kd", "eval_geometry_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p05", "qvf_metrics.csv"),
            ),
            manifest_path=("innovation", "QVF", seed_dir, "beta_0p05", "qvf_manifest.json"),
        ),
        CandidateSpec(
            key="qvf_beta_0p10",
            method="QVF",
            hyperparameter="beta=0.10",
            metrics_paths=(
                ("innovation", "QVF", seed_dir, "beta_0p10", "training_eval_export", "metrics", "vanilla_kd", "eval_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p10", "training_eval_export", "metrics", "vanilla_kd", "eval_group_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p10", "training_eval_export", "metrics", "vanilla_kd", "eval_geometry_metrics.csv"),
                ("innovation", "QVF", seed_dir, "beta_0p10", "qvf_metrics.csv"),
            ),
            manifest_path=("innovation", "QVF", seed_dir, "beta_0p10", "qvf_manifest.json"),
        ),
        CandidateSpec(
            key="weighted_qvf_L1_0p25_L9_0p75_beta_0p05",
            method="Weighted QVF",
            hyperparameter="L1=0.25;L9=0.75;beta=0.05",
            metrics_paths=(
                (
                    "innovation",
                    "QVF_L1_0p25_L9_0p75",
                    seed_dir,
                    "beta_0p05",
                    "training_eval_export",
                    "metrics",
                    "vanilla_kd",
                    "eval_metrics.csv",
                ),
                (
                    "innovation",
                    "QVF_L1_0p25_L9_0p75",
                    seed_dir,
                    "beta_0p05",
                    "training_eval_export",
                    "metrics",
                    "vanilla_kd",
                    "eval_group_metrics.csv",
                ),
                (
                    "innovation",
                    "QVF_L1_0p25_L9_0p75",
                    seed_dir,
                    "beta_0p05",
                    "training_eval_export",
                    "metrics",
                    "vanilla_kd",
                    "eval_geometry_metrics.csv",
                ),
                ("innovation", "QVF_L1_0p25_L9_0p75", seed_dir, "beta_0p05", "qvf_metrics.csv"),
            ),
            manifest_path=("innovation", "QVF_L1_0p25_L9_0p75", seed_dir, "beta_0p05", "qvf_manifest.json"),
        ),
        CandidateSpec(
            key="per_lambda_0p001",
            method="PER",
            hyperparameter="lambda=0.001",
            metrics_paths=(("innovation", "PER", seed_dir, "lambada_0p001", "per_metrics.csv"),),
            manifest_path=("innovation", "PER", seed_dir, "lambada_0p001", "per_manifest.json"),
        ),
        CandidateSpec(
            key="per_lambda_0p003",
            method="PER",
            hyperparameter="lambda=0.003",
            metrics_paths=(("innovation", "PER", seed_dir, "lambada_0p003", "per_metrics.csv"),),
            manifest_path=("innovation", "PER", seed_dir, "lambada_0p003", "per_manifest.json"),
        ),
        CandidateSpec(
            key="per_lambda_0p01",
            method="PER",
            hyperparameter="lambda=0.01",
            metrics_paths=(("innovation", "PER", seed_dir, "lambada_0p01", "per_metrics.csv"),),
            manifest_path=("innovation", "PER", seed_dir, "lambada_0p01", "per_manifest.json"),
        ),
    ]
    if include_supplemental:
        specs.extend(
            [
                CandidateSpec(
                    key="two_stage_ce_stage2",
                    method="Two-Stage CE",
                    hyperparameter="stage2",
                    metrics_paths=(
                        ("innovation", "two_stage", seed_dir, "stage2_export", "metrics", "two_stage_decoupling_stage2", "eval_metrics.csv"),
                        ("innovation", "two_stage", seed_dir, "stage2_export", "metrics", "two_stage_decoupling_stage2", "eval_group_metrics.csv"),
                        ("innovation", "two_stage", seed_dir, "stage2_export", "metrics", "two_stage_decoupling_stage2", "eval_geometry_metrics.csv"),
                    ),
                    manifest_path=("innovation", "two_stage", seed_dir, "stage2_manifest.json"),
                    supplemental=True,
                ),
                CandidateSpec(
                    key="tail_margin_tau_0p3",
                    method="Tail Margin",
                    hyperparameter="tau_max=0.3",
                    metrics_paths=(
                        (
                            "innovation",
                            "stage2_class",
                            "tau_sweep",
                            seed_dir,
                            "selected_tau_max_0p3_final",
                            "metrics",
                            "stage2_class",
                            "eval_metrics.csv",
                        ),
                        (
                            "innovation",
                            "stage2_class",
                            "tau_sweep",
                            seed_dir,
                            "selected_tau_max_0p3_final",
                            "metrics",
                            "stage2_class",
                            "eval_group_metrics.csv",
                        ),
                        (
                            "innovation",
                            "stage2_class",
                            "tau_sweep",
                            seed_dir,
                            "selected_tau_max_0p3_final",
                            "metrics",
                            "stage2_class",
                            "eval_geometry_metrics.csv",
                        ),
                    ),
                    manifest_path=("innovation", "stage2_class", "tau_sweep", seed_dir, "selected_tau_max_0p3_final", "final_export_manifest.json"),
                    supplemental=True,
                ),
            ]
        )
    return specs


def _path_from_parts(root: Path, parts: Sequence[str] | None) -> Path | None:
    if parts is None:
        return None
    return root.joinpath(*parts)


def _path_str(path: str | Path | None) -> str:
    if path is None:
        return ""
    return str(Path(path).resolve(strict=False))


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    text = json.dumps(_json_sanitize(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return _path_str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    return value


def _json_dump(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_json_sanitize(dict(payload)), handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    tmp.replace(target)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return value


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
    tmp.replace(target)


def _read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"CSV not found: {target}")
    with target.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise RuntimeError(f"CSV has no header: {target}")
    return rows, fieldnames


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return dict(value) if isinstance(value, Mapping) else {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}


def _as_float(value: Any) -> float:
    if _is_missing(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    number = _as_float(value)
    if not math.isfinite(number):
        return None
    return int(round(number))


def _required_int(value: Any, *, context: str) -> int:
    number = _as_float(value)
    if not math.isfinite(number) or not float(number).is_integer():
        raise ValueError(f"{context} must be an integer, got {value!r}")
    return int(number)


def _select_label_count_column(fieldnames: Sequence[str]) -> str | None:
    for candidate in ("train_label_n", "label_n"):
        if candidate in fieldnames:
            return candidate
    return None


def _load_label_name_to_id(path: Path) -> dict[str, int]:
    payload = _read_json(path)
    result: dict[str, int] = {}
    for key, value in payload.items():
        parsed = _as_int(value)
        if parsed is not None:
            result[str(key)] = int(parsed)
    return result


def _label_id_from_row(row: Mapping[str, Any], *, label_name_to_id: Mapping[str, int], context: str) -> int:
    for key in ("label_id", "class_id", "target"):
        if not _is_missing(row.get(key)):
            return _required_int(row.get(key), context=f"{context}.{key}")
    for key in ("label", "ebird_code", "class_name"):
        value = row.get(key)
        if _is_missing(value):
            continue
        text = str(value).strip()
        if text in label_name_to_id:
            return int(label_name_to_id[text])
    raise KeyError(f"{context} is missing label identity")


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = int(counts.get(key, 0)) + 1
    return dict(sorted(counts.items()))


def _build_bucket_provenance(root: Path) -> dict[str, Any]:
    train_csv = _dataset_path(root, "TRAIN_CSV", ("splits", "train.csv"))
    eval_csv = _dataset_path(root, "EVAL_CSV", ("splits", "eval.csv"))
    label_mapping_path = _dataset_path(root, "LABEL_MAPPING", ("splits", "label_mapping.json"))
    provenance: dict[str, Any] = {
        "bucket_verification_status": "missing_source",
        "bucket_source": "train split label counts via src.utils.group_utils.bucket_from_train_label_n",
        "train_csv": _path_str(train_csv),
        "train_csv_sha256": _file_sha256(train_csv),
        "eval_csv": _path_str(eval_csv),
        "eval_csv_sha256": _file_sha256(eval_csv),
        "label_mapping_path": _path_str(label_mapping_path),
        "label_mapping_sha256": _file_sha256(label_mapping_path),
    }
    if not train_csv.exists() or not eval_csv.exists():
        missing = [str(path) for path in (train_csv, eval_csv) if not path.exists()]
        provenance["missing_sources"] = missing
        return provenance

    label_name_to_id = _load_label_name_to_id(label_mapping_path)
    train_rows, train_fields = _read_csv_rows(train_csv)
    label_count_column = _select_label_count_column(train_fields)
    row_count_by_label: dict[int, int] = {}
    declared_count_by_label: dict[int, int] = {}
    observed_bucket_by_label: dict[int, str] = {}

    for row_number, row in enumerate(train_rows, start=2):
        label_id = _label_id_from_row(row, label_name_to_id=label_name_to_id, context=f"{train_csv}:row={row_number}")
        row_count_by_label[label_id] = int(row_count_by_label.get(label_id, 0)) + 1
        if label_count_column is not None:
            declared_count = _required_int(row.get(label_count_column), context=f"{train_csv}:row={row_number}.{label_count_column}")
            previous = declared_count_by_label.get(label_id)
            if previous is not None and previous != declared_count:
                raise ValueError(
                    f"{train_csv}: inconsistent {label_count_column} for label_id={label_id}: "
                    f"{previous!r} vs {declared_count!r}"
                )
            declared_count_by_label[label_id] = declared_count
        if "bucket" in train_fields and not _is_missing(row.get("bucket")):
            observed_bucket = normalize_bucket(row.get("bucket"), field_name=f"{train_csv}:row={row_number}.bucket")
            previous_bucket = observed_bucket_by_label.get(label_id)
            if previous_bucket is not None and previous_bucket != observed_bucket:
                raise ValueError(
                    f"{train_csv}: inconsistent bucket for label_id={label_id}: "
                    f"{previous_bucket!r} vs {observed_bucket!r}"
                )
            observed_bucket_by_label[label_id] = observed_bucket

    count_by_label = declared_count_by_label if declared_count_by_label else row_count_by_label
    bucket_by_label = {
        int(label_id): bucket_from_train_label_n(int(count))
        for label_id, count in sorted(count_by_label.items())
    }
    for label_id, observed_bucket in observed_bucket_by_label.items():
        expected_bucket = bucket_by_label.get(int(label_id))
        if expected_bucket != observed_bucket:
            raise ValueError(
                f"{train_csv}: bucket mismatch for label_id={label_id}; "
                f"bucket={observed_bucket!r}, train_count_derived={expected_bucket!r}"
            )

    eval_rows, eval_fields = _read_csv_rows(eval_csv)
    eval_identity: list[dict[str, Any]] = []
    eval_buckets: list[str] = []
    for row_number, row in enumerate(eval_rows, start=2):
        label_id = _label_id_from_row(row, label_name_to_id=label_name_to_id, context=f"{eval_csv}:row={row_number}")
        if label_id not in bucket_by_label:
            raise KeyError(f"{eval_csv}:row={row_number} label_id={label_id} is absent from train-derived bucket map")
        expected_bucket = str(bucket_by_label[label_id])
        if "bucket" in eval_fields and not _is_missing(row.get("bucket")):
            observed_bucket = normalize_bucket(row.get("bucket"), field_name=f"{eval_csv}:row={row_number}.bucket")
            if observed_bucket != expected_bucket:
                raise ValueError(
                    f"{eval_csv}:row={row_number} bucket={observed_bucket!r} "
                    f"disagrees with train-derived bucket={expected_bucket!r}"
                )
        sample_id = str(row.get("sample_id", row_number)).strip()
        eval_identity.append({"sample_id": sample_id, "label_id": int(label_id), "bucket": expected_bucket})
        eval_buckets.append(expected_bucket)

    identity_payload = {
        "bucket_source": provenance["bucket_source"],
        "train_label_count_column_used": label_count_column or "computed_from_train_rows_by_label_id",
        "train_label_count_by_label_id": {str(key): int(value) for key, value in sorted(count_by_label.items())},
        "bucket_by_label_id": {str(key): str(value) for key, value in sorted(bucket_by_label.items())},
        "eval_identity": eval_identity,
    }
    provenance.update(
        {
            "bucket_verification_status": "validated",
            "train_label_count_column_used": label_count_column or "computed_from_train_rows_by_label_id",
            "train_label_count_by_label_id": identity_payload["train_label_count_by_label_id"],
            "bucket_by_label_id": identity_payload["bucket_by_label_id"],
            "label_bucket_counts": _count_values(bucket_by_label.values()),
            "eval_sample_count": int(len(eval_identity)),
            "eval_bucket_sample_counts": _count_values(eval_buckets),
            "eval_sample_order_bucket_hash": _json_hash(eval_identity),
            "bucket_protocol_hash": _json_hash(identity_payload),
        }
    )
    return provenance


def _finite(value: Any) -> bool:
    number = _as_float(value)
    return math.isfinite(number)


def _metric_rows_for_candidate(root: Path, spec: CandidateSpec) -> tuple[list[dict[str, str]], list[Path], list[str]]:
    rows: list[dict[str, str]] = []
    source_paths: list[Path] = []
    missing: list[str] = []
    seen: set[Path] = set()
    for parts in spec.metrics_paths:
        path = root.joinpath(*parts)
        if path in seen:
            continue
        seen.add(path)
        if "test" in {part.lower() for part in path.parts}:
            raise RuntimeError(f"refusing to read test path for geometry-task audit: {path}")
        if not path.exists():
            missing.append(_path_str(path))
            continue
        table_rows, _fields = _read_csv_rows(path)
        for row in table_rows:
            split = str(row.get("split", ANALYSIS_SPLIT)).strip().lower()
            if split != ANALYSIS_SPLIT:
                continue
            item = dict(row)
            item["_source_path"] = _path_str(path)
            rows.append(item)
        source_paths.append(path)
    return rows, source_paths, missing


def _lookup_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    analysis_group: str,
) -> MetricValue:
    names = {str(name) for name in metric_names}
    group = str(analysis_group)
    for row in rows:
        split = str(row.get("split", ANALYSIS_SPLIT)).strip().lower()
        if split != ANALYSIS_SPLIT:
            continue
        if str(row.get("metric_name", "")) not in names:
            continue
        if str(row.get("analysis_group", "")) != group:
            continue
        value = _as_float(row.get("metric_value"))
        if not math.isfinite(value):
            return MetricValue(value=value, row=row, missing_reason="nonfinite")
        return MetricValue(value=value, row=row)
    return MetricValue(value=float("nan"), row=None, missing_reason=f"missing:{'/'.join(metric_names)}:{analysis_group}")


def _support_count(metric: MetricValue) -> int | None:
    if metric.row is None:
        return None
    return _as_int(metric.row.get("support_count"))


def _correct_count(accuracy: float, total: int | None) -> int | None:
    if total is None or not math.isfinite(float(accuracy)):
        return None
    return int(round(float(accuracy) * int(total)))


def _manifest_checkpoint_hash(manifest: Mapping[str, Any]) -> str:
    for key in (
        "checkpoint_sha256",
        "stage2_checkpoint_sha256",
        "checkpoint_hash",
        "best_checkpoint_sha256",
        "selected_checkpoint_sha256",
    ):
        value = manifest.get(key)
        if not _is_missing(value):
            return str(value)
    return ""


def _source_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_hashes = sorted({str(row.get("split_hash", "")) for row in rows if not _is_missing(row.get("split_hash"))})
    label_hashes = sorted(
        {str(row.get("label_mapping_hash", "")) for row in rows if not _is_missing(row.get("label_mapping_hash"))}
    )
    sample_hashes = sorted(
        {str(row.get("sample_order_hash", "")) for row in rows if not _is_missing(row.get("sample_order_hash"))}
    )
    run_hashes = sorted({str(row.get("run_identity_hash", "")) for row in rows if not _is_missing(row.get("run_identity_hash"))})
    return {
        "split_hashes": split_hashes,
        "label_mapping_hashes": label_hashes,
        "sample_order_hashes": sample_hashes,
        "run_identity_hashes": run_hashes,
    }


def _geometry_definition_payload(summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for row in summary_rows:
        for metric in GEOMETRY_METRICS:
            detail_key = f"_{metric}_definition_payload"
            detail = row.get(detail_key)
            if isinstance(detail, Mapping):
                payload.setdefault(metric, []).append(dict(detail))
    return payload


def _collect_candidate(root: Path, spec: CandidateSpec) -> dict[str, Any]:
    metric_rows, source_paths, missing_source_paths = _metric_rows_for_candidate(root, spec)
    manifest_path = _path_from_parts(root, spec.manifest_path)
    manifest = _read_json(manifest_path)
    row: dict[str, Any] = {
        "candidate_key": spec.key,
        "method": spec.method,
        "hyperparameter": spec.hyperparameter,
        "checkpoint_sha256": _manifest_checkpoint_hash(manifest),
        "metrics_source_files": ";".join(_path_str(path) for path in source_paths),
        "manifest_path": _path_str(manifest_path) if manifest_path is not None and manifest_path.exists() else "",
        "test_touched": False,
        "_source_identity": _source_identity(metric_rows),
        "_missing_source_paths": missing_source_paths,
        "_source_sha256": {_path_str(path): _file_sha256(path) for path in source_paths},
        "_manifest_sha256": _file_sha256(manifest_path),
        "_supplemental": bool(spec.supplemental),
    }

    missing_task: list[str] = []
    for output_name, (metric_name, group) in TASK_METRICS.items():
        value = _lookup_metric(metric_rows, metric_names=(metric_name, "macro_f1" if "macro_f1" in output_name else metric_name), analysis_group=group)
        row[output_name] = value.value
        if value.missing_reason:
            missing_task.append(output_name)
        if output_name.endswith("_accuracy") or output_name == "overall_accuracy":
            prefix = "overall" if output_name == "overall_accuracy" else output_name.removesuffix("_accuracy")
            total = _support_count(value)
            row["total" if prefix == "overall" else f"{prefix}_total"] = total
            row[f"{prefix}_correct"] = _correct_count(value.value, total)

    missing_geometry: list[str] = []
    for output_name, spec_info in GEOMETRY_METRICS.items():
        value = _lookup_metric(
            metric_rows,
            metric_names=tuple(spec_info["names"]),
            analysis_group=str(spec_info["group"]),
        )
        row[output_name] = value.value
        if value.missing_reason:
            missing_geometry.append(output_name)
        if value.row is not None:
            row[f"_{output_name}_definition_payload"] = {
                "metric_name": value.row.get("metric_name"),
                "analysis_group": value.row.get("analysis_group"),
                "metric_direction": value.row.get("metric_direction"),
                "aggregation_unit": value.row.get("aggregation_unit"),
                "aggregation_fn": value.row.get("aggregation_fn"),
                "definition": value.row.get("definition"),
            }

    row["missing_task_metrics"] = ";".join(missing_task)
    row["missing_geometry_metrics"] = ";".join(missing_geometry)
    row["task_metric_status"] = "ok" if not missing_task else "missing"
    row["geometry_metric_status"] = "ok" if not missing_geometry else "missing"
    exclusion_reasons: list[str] = []
    if not metric_rows:
        exclusion_reasons.append("no_eval_metric_rows")
    if missing_task:
        exclusion_reasons.append("missing_canonical_eval_task_metrics")
    if missing_geometry:
        exclusion_reasons.append("missing_required_geometry_metrics")
    row["exclusion_reason"] = ";".join(exclusion_reasons)
    protocol_status = EXCLUDED_PROTOCOL_STATUS
    if spec.key == REFERENCE_MODEL:
        if not metric_rows or missing_task:
            row["candidate_status"] = "invalid_reference"
        elif missing_geometry:
            row["candidate_status"] = "reference_partial"
            protocol_status = PARTIAL_PROTOCOL_STATUS
        else:
            row["candidate_status"] = "reference"
            protocol_status = FULL_PROTOCOL_STATUS
    elif not metric_rows or missing_task:
        row["candidate_status"] = "excluded"
    elif missing_geometry:
        row["candidate_status"] = "partial"
        protocol_status = PARTIAL_PROTOCOL_STATUS
    else:
        row["candidate_status"] = "included"
        protocol_status = FULL_PROTOCOL_STATUS
    row["protocol_inclusion_status"] = protocol_status
    row["included_full_protocol"] = bool(protocol_status == FULL_PROTOCOL_STATUS)
    row["included_partial_metric_only"] = bool(protocol_status == PARTIAL_PROTOCOL_STATUS)
    return row


def _delta(current: Any, baseline: Any) -> float:
    left = _as_float(current)
    right = _as_float(baseline)
    if not (math.isfinite(left) and math.isfinite(right)):
        return float("nan")
    return float(left - right)


def _improvement(current: Any, baseline: Any, *, direction: str) -> float:
    left = _as_float(current)
    right = _as_float(baseline)
    if not (math.isfinite(left) and math.isfinite(right)):
        return float("nan")
    if direction == "lower":
        return float(right - left)
    if direction == "higher":
        return float(left - right)
    raise ValueError(f"unknown direction={direction!r}")


def _apply_reference_deltas(rows: list[dict[str, Any]]) -> None:
    reference = next((row for row in rows if row.get("candidate_key") == REFERENCE_MODEL), None)
    if reference is None:
        raise RuntimeError("reference candidate vanilla_kd is missing")
    for row in rows:
        for task_metric, delta_column in TASK_DELTA_COLUMNS.items():
            row[delta_column] = _delta(row.get(task_metric), reference.get(task_metric))
        for metric_name, spec_info in GEOMETRY_METRICS.items():
            delta_column = f"delta_{metric_name}"
            row[delta_column] = _delta(row.get(metric_name), reference.get(metric_name))
            row[str(spec_info["improvement_column"])] = _improvement(
                row.get(metric_name),
                reference.get(metric_name),
                direction=str(spec_info["direction"]),
            )
        row["geometry_improvement_direction"] = "positive means improved after metric-direction normalization"

    reference_identity = reference.get("_source_identity", {})
    reference_definitions = {
        metric: reference.get(f"_{metric}_definition_payload")
        for metric in GEOMETRY_METRICS
        if isinstance(reference.get(f"_{metric}_definition_payload"), Mapping)
    }
    for row in rows:
        identity = row.get("_source_identity", {})
        row["same_eval_split_as_reference"] = _same_identity_value(identity, reference_identity, "split_hashes")
        row["same_label_mapping_as_reference"] = _same_identity_value(identity, reference_identity, "label_mapping_hashes")
        row["same_sample_order_as_reference"] = _same_identity_value(identity, reference_identity, "sample_order_hashes")
        row["same_geometry_definitions_as_reference"] = _same_geometry_definitions(row, reference_definitions)
        row["same_available_geometry_definitions_as_reference"] = _same_available_geometry_definitions(row, reference_definitions)


def _same_identity_value(current: Any, reference: Any, key: str) -> bool | str:
    if not isinstance(current, Mapping) or not isinstance(reference, Mapping):
        return "missing"
    current_values = tuple(current.get(key, ()))
    reference_values = tuple(reference.get(key, ()))
    if not current_values or not reference_values:
        return "missing"
    return current_values == reference_values


def _same_geometry_definitions(row: Mapping[str, Any], reference_definitions: Mapping[str, Any]) -> bool | str:
    if len(reference_definitions) < len(GEOMETRY_METRICS):
        return "missing_reference_required_geometry_metrics"
    for metric_name in GEOMETRY_METRICS:
        current = row.get(f"_{metric_name}_definition_payload")
        reference = reference_definitions.get(metric_name)
        if not isinstance(current, Mapping) or not isinstance(reference, Mapping):
            return "missing_required_geometry_metrics"
        if not _geometry_definition_equal(current, reference):
            return False
    return True


def _same_available_geometry_definitions(row: Mapping[str, Any], reference_definitions: Mapping[str, Any]) -> bool | str:
    compared = 0
    for metric_name in GEOMETRY_METRICS:
        current = row.get(f"_{metric_name}_definition_payload")
        reference = reference_definitions.get(metric_name)
        if not isinstance(current, Mapping) or not isinstance(reference, Mapping):
            continue
        compared += 1
        if not _geometry_definition_equal(current, reference):
            return False
    return True if compared else "missing"


def _geometry_definition_equal(current: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    for key in ("metric_name", "analysis_group", "metric_direction", "aggregation_unit", "aggregation_fn", "definition"):
        if str(current.get(key, "")) != str(reference.get(key, "")):
            return False
    return True


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / float(len(values))) if values else float("nan")


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den <= 0.0 or y_den <= 0.0:
        return float("nan")
    return float(numerator / (x_den * y_den))


def _ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _value in indexed[index:end]:
            result[original_index] = rank
        index = end
    return result


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    return _pearson(_ranks(xs), _ranks(ys))


def _coupling_status(n_directional: int, consistency_rate: float) -> str:
    if n_directional < 2 or not math.isfinite(consistency_rate):
        return "insufficient_data"
    if consistency_rate >= 0.8:
        return "strong_directional_coupling"
    if consistency_rate >= 0.6:
        return "weak_directional_coupling"
    if consistency_rate <= 0.4:
        return "decoupled"
    return "mixed"


def _full_protocol_rows(summary_rows: Sequence[Mapping[str, Any]], *, include_reference: bool) -> list[Mapping[str, Any]]:
    return [
        row
        for row in summary_rows
        if (include_reference or row.get("candidate_key") != REFERENCE_MODEL)
        and row.get("protocol_inclusion_status") == FULL_PROTOCOL_STATUS
    ]


def _partial_protocol_rows(summary_rows: Sequence[Mapping[str, Any]], *, include_reference: bool) -> list[Mapping[str, Any]]:
    return [
        row
        for row in summary_rows
        if (include_reference or row.get("candidate_key") != REFERENCE_MODEL)
        and row.get("protocol_inclusion_status") == PARTIAL_PROTOCOL_STATUS
    ]


def _identity_matches_reference(row: Mapping[str, Any]) -> bool:
    return (
        row.get("same_eval_split_as_reference") is True
        and row.get("same_label_mapping_as_reference") is True
        and row.get("same_sample_order_as_reference") is True
    )


def _geometry_definition_matches_reference_for_metric(
    row: Mapping[str, Any],
    reference: Mapping[str, Any],
    geometry_metric: str,
) -> bool:
    current = row.get(f"_{geometry_metric}_definition_payload")
    baseline = reference.get(f"_{geometry_metric}_definition_payload")
    return isinstance(current, Mapping) and isinstance(baseline, Mapping) and _geometry_definition_equal(current, baseline)


def _valid_geometry_metric_candidate(
    row: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    geometry_metric: str,
    improvement_column: str,
) -> bool:
    if row.get("candidate_key") == REFERENCE_MODEL:
        return False
    return (
        _identity_matches_reference(row)
        and _geometry_definition_matches_reference_for_metric(row, reference, geometry_metric)
        and math.isfinite(_as_float(row.get(improvement_column)))
    )


def _valid_task_metric_candidate(row: Mapping[str, Any], *, delta_column: str) -> bool:
    if row.get("candidate_key") == REFERENCE_MODEL:
        return False
    return _identity_matches_reference(row) and math.isfinite(_as_float(row.get(delta_column)))


def _valid_metric_pair_candidate(
    row: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    geometry_metric: str,
    improvement_column: str,
    delta_column: str,
) -> bool:
    return _valid_geometry_metric_candidate(
        row,
        reference,
        geometry_metric=geometry_metric,
        improvement_column=improvement_column,
    ) and math.isfinite(_as_float(row.get(delta_column)))


def build_pairwise_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reference = next((row for row in summary_rows if row.get("candidate_key") == REFERENCE_MODEL), None)
    pairwise_rows: list[dict[str, Any]] = []
    for geometry_metric, geometry_info in GEOMETRY_METRICS.items():
        improvement_column = str(geometry_info["improvement_column"])
        for task_metric, delta_column in TASK_DELTA_COLUMNS.items():
            candidates = [
                row
                for row in summary_rows
                if isinstance(reference, Mapping)
                and _valid_metric_pair_candidate(
                    row,
                    reference,
                    geometry_metric=geometry_metric,
                    improvement_column=improvement_column,
                    delta_column=delta_column,
                )
            ]
            values: list[tuple[float, float]] = []
            counts = {
                "n_geometry_improved": 0,
                "n_geometry_improved_task_improved": 0,
                "n_geometry_improved_task_worsened": 0,
                "n_geometry_worsened_task_improved": 0,
                "n_geometry_worsened_task_worsened": 0,
                "n_geometry_worsened": 0,
                "n_geometry_neutral": 0,
                "n_task_neutral": 0,
            }
            for candidate in candidates:
                geometry_improvement = _as_float(candidate.get(improvement_column))
                task_delta = _as_float(candidate.get(delta_column))
                if not (math.isfinite(geometry_improvement) and math.isfinite(task_delta)):
                    continue
                values.append((geometry_improvement, task_delta))
                if geometry_improvement > 0.0:
                    counts["n_geometry_improved"] += 1
                    if task_delta > 0.0:
                        counts["n_geometry_improved_task_improved"] += 1
                    elif task_delta < 0.0:
                        counts["n_geometry_improved_task_worsened"] += 1
                    else:
                        counts["n_task_neutral"] += 1
                elif geometry_improvement < 0.0:
                    counts["n_geometry_worsened"] += 1
                    if task_delta > 0.0:
                        counts["n_geometry_worsened_task_improved"] += 1
                    elif task_delta < 0.0:
                        counts["n_geometry_worsened_task_worsened"] += 1
                    else:
                        counts["n_task_neutral"] += 1
                else:
                    counts["n_geometry_neutral"] += 1
                    if task_delta == 0.0:
                        counts["n_task_neutral"] += 1
            directional = (
                counts["n_geometry_improved_task_improved"]
                + counts["n_geometry_improved_task_worsened"]
                + counts["n_geometry_worsened_task_improved"]
                + counts["n_geometry_worsened_task_worsened"]
            )
            consistent = counts["n_geometry_improved_task_improved"] + counts["n_geometry_worsened_task_worsened"]
            consistency_rate = float(consistent / directional) if directional else float("nan")
            xs = [item[0] for item in values]
            ys = [item[1] for item in values]
            row = {
                "geometry_metric": geometry_metric,
                "task_metric": task_metric,
                "candidate_scope": PAIRWISE_CANDIDATE_SCOPE,
                "n_valid_candidates": int(len(values)),
                "pearson": _pearson(xs, ys),
                "spearman": _spearman(xs, ys),
                **counts,
                "direction_consistency_rate": consistency_rate,
                "coupling_status": _coupling_status(directional, consistency_rate),
                "descriptive_only": True,
            }
            pairwise_rows.append(row)
    return pairwise_rows


def _max_by(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any] | None:
    finite_rows = [row for row in rows if _finite(row.get(key))]
    if not finite_rows:
        return None
    return max(finite_rows, key=lambda row: _as_float(row.get(key)))


def build_counterexample_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reference = next((row for row in summary_rows if row.get("candidate_key") == REFERENCE_MODEL), None)
    rows: list[dict[str, Any]] = []
    if not isinstance(reference, Mapping):
        return rows
    for geometry_metric, geometry_info in GEOMETRY_METRICS.items():
        improvement_column = str(geometry_info["improvement_column"])
        geometry_candidates = [
            row
            for row in summary_rows
            if _valid_geometry_metric_candidate(
                row,
                reference,
                geometry_metric=geometry_metric,
                improvement_column=improvement_column,
            )
        ]
        best = _max_by(geometry_candidates, improvement_column)
        if best is not None:
            rows.append(
                _counterexample_row(
                    "largest_geometry_improvement",
                    best,
                    reference,
                    geometry_metric=geometry_metric,
                    task_metric="",
                    geometry_improvement_column=improvement_column,
                    task_delta_column="",
                    note="Largest positive direction-normalized geometry improvement among candidates.",
                )
            )
    for task_metric in ("tail_accuracy", "tail_macro_f1"):
        delta_column = TASK_DELTA_COLUMNS[task_metric]
        task_candidates = [
            row for row in summary_rows if _valid_task_metric_candidate(row, delta_column=delta_column)
        ]
        best = _max_by(task_candidates, delta_column)
        if best is not None:
            rows.append(
                _counterexample_row(
                    "largest_tail_task_improvement",
                    best,
                    reference,
                    geometry_metric="",
                    task_metric=task_metric,
                    geometry_improvement_column="",
                    task_delta_column=delta_column,
                    note="Largest positive tail task delta among candidates.",
                )
            )
    for geometry_metric, geometry_info in GEOMETRY_METRICS.items():
        improvement_column = str(geometry_info["improvement_column"])
        for task_metric in ("tail_accuracy", "tail_macro_f1"):
            delta_column = TASK_DELTA_COLUMNS[task_metric]
            candidates = [
                row
                for row in summary_rows
                if _valid_metric_pair_candidate(
                    row,
                    reference,
                    geometry_metric=geometry_metric,
                    improvement_column=improvement_column,
                    delta_column=delta_column,
                )
            ]
            for candidate in candidates:
                geometry_improvement = _as_float(candidate.get(improvement_column))
                task_delta = _as_float(candidate.get(delta_column))
                if not (math.isfinite(geometry_improvement) and math.isfinite(task_delta)):
                    continue
                if geometry_improvement > 0.0 and task_delta < 0.0:
                    rows.append(
                        _counterexample_row(
                            "geometry_improved_task_worsened",
                            candidate,
                            reference,
                            geometry_metric=geometry_metric,
                            task_metric=task_metric,
                            geometry_improvement_column=improvement_column,
                            task_delta_column=delta_column,
                            note="Geometry improved by direction-normalized sign, but tail task delta is negative.",
                        )
                    )
                if task_delta > 0.0 and geometry_improvement < 0.0:
                    rows.append(
                        _counterexample_row(
                            "task_improved_geometry_worsened",
                            candidate,
                            reference,
                            geometry_metric=geometry_metric,
                            task_metric=task_metric,
                            geometry_improvement_column=improvement_column,
                            task_delta_column=delta_column,
                            note="Tail task delta is positive, but geometry worsened by direction-normalized sign.",
                        )
                    )
    return rows


def _counterexample_row(
    counterexample_type: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    geometry_metric: str,
    task_metric: str,
    geometry_improvement_column: str,
    task_delta_column: str,
    note: str,
) -> dict[str, Any]:
    return {
        "counterexample_type": counterexample_type,
        "geometry_metric": geometry_metric,
        "task_metric": task_metric,
        "candidate_key": candidate.get("candidate_key", ""),
        "method": candidate.get("method", ""),
        "hyperparameter": candidate.get("hyperparameter", ""),
        "geometry_value": candidate.get(geometry_metric, "") if geometry_metric else "",
        "reference_geometry_value": reference.get(geometry_metric, "") if geometry_metric else "",
        "geometry_improvement": candidate.get(geometry_improvement_column, "") if geometry_improvement_column else "",
        "task_value": candidate.get(task_metric, "") if task_metric else "",
        "reference_task_value": reference.get(task_metric, "") if task_metric else "",
        "task_delta": candidate.get(task_delta_column, "") if task_delta_column else "",
        "note": note,
    }


def _output_dir(root: Path, seed: int, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    return root / "export" / "geometry_task_coupling" / f"seed_{int(seed)}"


def _metric_source_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        result[str(row["candidate_key"])] = {
            "method": row.get("method"),
            "hyperparameter": row.get("hyperparameter"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "metrics_source_files": str(row.get("metrics_source_files", "")).split(";")
            if row.get("metrics_source_files")
            else [],
            "metrics_source_sha256": row.get("_source_sha256", {}),
            "manifest_path": row.get("manifest_path"),
            "manifest_sha256": row.get("_manifest_sha256"),
            "missing_source_paths": row.get("_missing_source_paths", []),
            "missing_task_metrics": row.get("missing_task_metrics"),
            "missing_geometry_metrics": row.get("missing_geometry_metrics"),
            "candidate_status": row.get("candidate_status"),
            "protocol_inclusion_status": row.get("protocol_inclusion_status"),
            "included_full_protocol": row.get("included_full_protocol"),
            "included_partial_metric_only": row.get("included_partial_metric_only"),
            "exclusion_reason": row.get("exclusion_reason"),
            "source_identity": row.get("_source_identity", {}),
        }
    return result


def _best_pairwise(pairwise_rows: Sequence[Mapping[str, Any]], prefix: str) -> Mapping[str, Any] | None:
    selected = [
        row
        for row in pairwise_rows
        if str(row.get("task_metric", "")).startswith(prefix) and _finite(row.get("direction_consistency_rate"))
    ]
    if not selected:
        return None
    return max(
        selected,
        key=lambda row: (
            _as_float(row.get("direction_consistency_rate")),
            int(_as_int(row.get("n_valid_candidates")) or 0),
        ),
    )


def _overall_evidence_status(pairwise_rows: Sequence[Mapping[str, Any]]) -> str:
    valid = [row for row in pairwise_rows if _finite(row.get("direction_consistency_rate"))]
    if len(valid) < 2:
        return "insufficient evidence"
    statuses = [str(row.get("coupling_status", "")) for row in valid]
    strong_or_weak = sum(1 for status in statuses if status in {"strong_directional_coupling", "weak_directional_coupling"})
    decoupled = sum(1 for status in statuses if status == "decoupled")
    if strong_or_weak >= max(1, math.ceil(0.7 * len(valid))):
        return "geometry-task coupled"
    if decoupled >= max(1, math.ceil(0.5 * len(valid))):
        return "largely decoupled"
    return "partially coupled"


def build_manifest(
    *,
    root: Path,
    seed: int,
    output_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
    counterexample_rows: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Path],
    include_supplemental: bool,
    bucket_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    reference = next((row for row in summary_rows if row.get("candidate_key") == REFERENCE_MODEL), {})
    reference_identity = reference.get("_source_identity", {}) if isinstance(reference, Mapping) else {}
    geometry_definitions = _geometry_definition_payload(summary_rows)
    full_protocol_rows = _full_protocol_rows(summary_rows, include_reference=True)
    partial_protocol_rows = _partial_protocol_rows(summary_rows, include_reference=True)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_name": EXPERIMENT_NAME,
        "analysis_name": ANALYSIS_NAME,
        "analysis_only": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": _path_str(root),
        "seed": int(seed),
        "reference_model": REFERENCE_MODEL,
        "analysis_split": ANALYSIS_SPLIT,
        "geometry_main_group": MAIN_GEOMETRY_GROUP,
        "tail_collapse_gap_group": TAIL_COLLAPSE_GROUP,
        "test_touched": False,
        "include_supplemental_candidates": bool(include_supplemental),
        "candidate_list": [
            {
                "candidate_key": row.get("candidate_key"),
                "method": row.get("method"),
                "hyperparameter": row.get("hyperparameter"),
                "candidate_status": row.get("candidate_status"),
                "protocol_inclusion_status": row.get("protocol_inclusion_status"),
                "included_full_protocol": row.get("included_full_protocol"),
                "included_partial_metric_only": row.get("included_partial_metric_only"),
                "exclusion_reason": row.get("exclusion_reason"),
            }
            for row in summary_rows
        ],
        "included_candidate_policy": (
            "included_candidates contains only full protocol candidates for summary-level completeness; "
            "pairwise coupling independently reselects valid candidates for each geometry_metric x task_metric pair"
        ),
        "pairwise_candidate_policy": (
            "For each geometry_metric x task_metric pair, a candidate is valid only when that geometry improvement "
            "and task delta are finite, eval split / label mapping / sample order match the reference, and that "
            "geometry metric definition matches the reference. Missing unrelated geometry metrics do not exclude "
            "the candidate from other metric pairs."
        ),
        "included_candidates": [
            str(row.get("candidate_key"))
            for row in full_protocol_rows
        ],
        "included_full_protocol_candidates": [
            str(row.get("candidate_key"))
            for row in full_protocol_rows
        ],
        "included_partial_metric_only_candidates": [
            str(row.get("candidate_key"))
            for row in partial_protocol_rows
        ],
        "partial_candidates": [
            {
                "candidate_key": row.get("candidate_key"),
                "reason": row.get("exclusion_reason"),
                "missing_geometry_metrics": row.get("missing_geometry_metrics"),
            }
            for row in partial_protocol_rows
        ],
        "excluded_candidates": [
            {
                "candidate_key": row.get("candidate_key"),
                "reason": row.get("exclusion_reason"),
                "missing_task_metrics": row.get("missing_task_metrics"),
                "missing_geometry_metrics": row.get("missing_geometry_metrics"),
            }
            for row in summary_rows
            if row.get("candidate_status") == "excluded"
        ],
        "candidate_checkpoint_hashes": {
            str(row.get("candidate_key")): row.get("checkpoint_sha256") for row in summary_rows
        },
        "eval_split_hash": _single_or_list(reference_identity.get("split_hashes", [])),
        "label_mapping_hash": _single_or_list(reference_identity.get("label_mapping_hashes", [])),
        "sample_order_hash": _single_or_list(reference_identity.get("sample_order_hashes", [])),
        "geometry_metric_definitions_hash": _json_hash(geometry_definitions),
        "geometry_metric_definitions": geometry_definitions,
        "group_protocol": group_protocol_metadata(
            tail_rule="task tail is medium + few; geometry metrics are read only from their existing metric-specific analysis_group",
            canonical_group_field="bucket",
            legacy_group_alias="shot_group",
            main_comparison_group="tail",
            supplementary_extreme_group="few",
        ),
        "bucket_provenance": dict(bucket_provenance),
        "metric_sources": _metric_source_manifest(summary_rows),
        "validity_checks": {
            "same_eval_split_checked": True,
            "same_label_mapping_checked": True,
            "same_bucket_mapping_checked_via_metric_sample_order_and_group_protocol": False,
            "same_bucket_mapping_checked_via_train_split_label_counts": bucket_provenance.get("bucket_verification_status") == "validated",
            "bucket_protocol_hash_recorded": not _is_missing(bucket_provenance.get("bucket_protocol_hash")),
            "same_geometry_definitions_checked": "strict_required_metrics_and_available_metric_subset",
            "canonical_checkpoint_hashes_recorded": True,
            "no_test_split_read": True,
            "finite_metrics_required_for_pairwise": True,
            "pairwise_uses_metric_pair_valid_candidates_only": True,
            "missing_geometry_metric_only_invalidates_its_own_pairs": True,
        },
        "analysis_questions": {
            "best_tail_direction_consistency": _best_pairwise(pairwise_rows, "tail"),
            "best_medium_direction_consistency": _best_pairwise(pairwise_rows, "medium"),
            "evidence_status": _overall_evidence_status(pairwise_rows),
            "descriptive_not_causal": True,
        },
        "outputs": {name: _path_str(path) for name, path in outputs.items()},
        "output_dir": _path_str(output_dir),
        "row_counts": {
            "summary": int(len(summary_rows)),
            "pairwise": int(len(pairwise_rows)),
            "counterexamples": int(len(counterexample_rows)),
            "included_full_protocol_candidates": int(len(full_protocol_rows)),
            "included_partial_metric_only_candidates": int(len(partial_protocol_rows)),
        },
        "correlation_interpretation": "descriptive_only",
        "causal_claim_allowed": False,
        "quantization_context": {
            "qmin": -7,
            "qmax": 7,
            "source": "project Quant Injection effective signed INT4 codebook; recorded only as provenance",
        },
    }


def _single_or_list(values: Any) -> Any:
    if isinstance(values, (list, tuple)) and len(values) == 1:
        return values[0]
    return values


def run_geometry_task_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = _dataset_root(args.dataset_root)
    specs = _candidate_specs(int(args.seed), include_supplemental=bool(args.include_supplemental))
    bucket_provenance = _build_bucket_provenance(root)
    summary_rows = [_collect_candidate(root, spec) for spec in specs]
    _apply_reference_deltas(summary_rows)
    pairwise_rows = build_pairwise_rows(summary_rows)
    counterexample_rows = build_counterexample_rows(summary_rows)

    output_dir = _output_dir(root, int(args.seed), args.output_dir)
    outputs = {
        "summary_csv": output_dir / "geometry_task_coupling_summary.csv",
        "pairwise_csv": output_dir / "geometry_task_pairwise_coupling.csv",
        "counterexamples_csv": output_dir / "geometry_task_counterexamples.csv",
        "manifest_json": output_dir / "geometry_task_coupling_manifest.json",
    }
    manifest = build_manifest(
        root=root,
        seed=int(args.seed),
        output_dir=output_dir,
        summary_rows=summary_rows,
        pairwise_rows=pairwise_rows,
        counterexample_rows=counterexample_rows,
        outputs=outputs,
        include_supplemental=bool(args.include_supplemental),
        bucket_provenance=bucket_provenance,
    )
    if not bool(args.dry_run):
        _write_csv(outputs["summary_csv"], summary_rows, SUMMARY_COLUMNS)
        _write_csv(outputs["pairwise_csv"], pairwise_rows, PAIRWISE_COLUMNS)
        _write_csv(outputs["counterexamples_csv"], counterexample_rows, COUNTEREXAMPLE_COLUMNS)
        manifest["artifact_sha256"] = {
            "geometry_task_coupling_summary.csv": _file_sha256(outputs["summary_csv"]),
            "geometry_task_pairwise_coupling.csv": _file_sha256(outputs["pairwise_csv"]),
            "geometry_task_counterexamples.csv": _file_sha256(outputs["counterexamples_csv"]),
        }
        manifest["run_manifest_hash"] = _json_hash(manifest)
        _json_dump(outputs["manifest_json"], manifest)
    return {
        "output_dir": output_dir,
        "outputs": outputs,
        "summary_rows": summary_rows,
        "pairwise_rows": pairwise_rows,
        "counterexample_rows": counterexample_rows,
        "manifest": manifest,
        "bucket_provenance": bucket_provenance,
        "dry_run": bool(args.dry_run),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ANALYSIS_NAME)
    parser.add_argument("--dataset-root", default=None, help="Default: config.CURRENT_DATASET['ROOT'] or local_data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-supplemental", action="store_true", help="Also audit optional Two-Stage CE / Tail Margin candidates.")
    parser.add_argument("--dry-run", action="store_true", help="Read sources and build tables in memory without writing output files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_geometry_task_audit(args)
    mode = "dry-run" if result["dry_run"] else "write"
    print(f"[geometry-task] mode={mode}")
    print(f"[geometry-task] output_dir={result['output_dir']}")
    print(f"[geometry-task] summary_rows={len(result['summary_rows'])}")
    print(f"[geometry-task] pairwise_rows={len(result['pairwise_rows'])}")
    print(f"[geometry-task] counterexample_rows={len(result['counterexample_rows'])}")
    full_protocol = [
        str(row.get("candidate_key"))
        for row in result["summary_rows"]
        if row.get("protocol_inclusion_status") == FULL_PROTOCOL_STATUS
    ]
    partial_metric_only = [
        str(row.get("candidate_key"))
        for row in result["summary_rows"]
        if row.get("protocol_inclusion_status") == PARTIAL_PROTOCOL_STATUS
    ]
    excluded = [
        f"{row.get('candidate_key')}:{row.get('exclusion_reason')}"
        for row in result["summary_rows"]
        if row.get("protocol_inclusion_status") == EXCLUDED_PROTOCOL_STATUS
    ]
    print(f"[geometry-task] included_full_protocol={';'.join(full_protocol)}")
    print(f"[geometry-task] included_partial_metric_only={';'.join(partial_metric_only)}")
    print(f"[geometry-task] excluded={';'.join(excluded)}")
    if not result["dry_run"]:
        for name, path in result["outputs"].items():
            print(f"[geometry-task] {name}={path}")
    return result


if __name__ == "__main__":
    main()
