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
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.utils.group_utils import TAIL_BUCKETS, bucket_from_train_label_n, group_protocol_metadata, normalize_bucket


PROTOCOL_VERSION = "vanilla_kd_residual_tail_attribution.v1"
EXPERIMENT_NAME = "vanilla_kd_residual_tail_attribution"
ANALYSIS_NAME = "Vanilla KD Residual Tail Attribution"
ANALYSIS_SPLIT = "eval"
DEFAULT_SEED = 2026
SOFTMAX_TEMPERATURE = 1.0
PARITY_ATOL = 1e-6
PARITY_RTOL = 1e-6

TEACHER_KEY = "fp32_teacher"
INT4_KEY = "int4_init"
KD_KEY = "vanilla_kd"
MODEL_KEYS = (TEACHER_KEY, INT4_KEY, KD_KEY)
BASE_BUCKETS = ("many", "medium", "few")
TAIL_BUCKET_SET = set(TAIL_BUCKETS)

SUMMARY_FILENAME = "vanilla_kd_residual_tail_summary.csv"
SAMPLES_FILENAME = "vanilla_kd_residual_tail_samples.csv"
COHORTS_FILENAME = "vanilla_kd_residual_tail_cohorts.csv"
PARITY_FILENAME = "vanilla_kd_residual_tail_audit_parity.csv"
MANIFEST_FILENAME = "vanilla_kd_residual_tail_manifest.json"

SAMPLE_COLUMNS = [
    "sample_index",
    "analysis_group",
    "sample_id",
    "audio_path",
    "true_label",
    "true_label_id",
    "true_bucket",
    "teacher_pred",
    "teacher_pred_label",
    "teacher_correct",
    "teacher_margin",
    "teacher_p_true",
    "teacher_P_many",
    "teacher_P_tail",
    "teacher_competition",
    "int4_pred",
    "int4_pred_label",
    "int4_correct",
    "int4_margin",
    "int4_P_many",
    "int4_P_tail",
    "int4_competition",
    "kd_pred",
    "kd_pred_label",
    "kd_correct",
    "kd_margin",
    "kd_P_many",
    "kd_P_tail",
    "kd_competition",
    "kd_pred_bucket",
    "delta_competition_int4",
    "delta_competition_kd_repair",
    "residual_competition",
    "margin_drop_int4",
    "margin_residual_kd",
    "kl_teacher_int4",
    "kl_teacher_kd",
    "cohort",
]

COHORT_COLUMNS = [
    "analysis_group",
    "cohort",
    "description",
    "sample_count",
    "fraction_of_all_tail",
    "fraction_of_kd_tail_errors",
    "mean_teacher_margin",
    "mean_int4_margin",
    "mean_kd_margin",
    "mean_teacher_p_true",
    "mean_teacher_P_many",
    "mean_int4_P_many",
    "mean_kd_P_many",
    "mean_teacher_P_tail",
    "mean_int4_P_tail",
    "mean_kd_P_tail",
    "mean_delta_competition_int4",
    "mean_delta_competition_kd_repair",
    "mean_residual_competition",
    "mean_margin_drop_int4",
    "mean_margin_residual_kd",
    "mean_kl_teacher_int4",
    "mean_kl_teacher_kd",
    "tail_to_many_count",
    "tail_to_many_rate",
    "predicted_many_label",
    "predicted_many_label_count",
    "predicted_many_label_counts",
]

SUMMARY_COLUMNS = [
    "analysis_group",
    "tail_total",
    "teacher_correct_tail",
    "teacher_wrong_tail",
    "teacher_correct_int4_wrong",
    "teacher_correct_int4_wrong_kd_correct",
    "teacher_correct_int4_wrong_kd_wrong",
    "teacher_correct_int4_correct_kd_wrong",
    "teacher_wrong_kd_wrong",
    "teacher_wrong_kd_correct",
    "int4_error_rate_on_teacher_correct",
    "kd_repair_rate",
    "quant_assoc_residual_rate",
    "kd_induced_degradation_rate",
    "kd_tail_error_total",
    "C3_tail_to_many_count",
    "C3_tail_to_many_rate",
    "C5_tail_to_many_count",
    "C5_tail_to_many_rate",
    "mean_teacher_competition_tail",
    "mean_int4_competition_tail",
    "mean_kd_competition_tail",
    "mean_delta_competition_int4_tail",
    "mean_delta_competition_kd_repair_tail",
    "mean_residual_competition_tail",
    "mean_teacher_margin_Tcorrect_Kcorrect",
    "mean_teacher_margin_Tcorrect_Kwrong",
    "mean_teacher_p_true_Tcorrect_Kcorrect",
    "mean_teacher_p_true_Tcorrect_Kwrong",
    "audit_parity",
    "test_touched",
    "protocol_status",
]

PARITY_COLUMNS = ["metric", "old_value", "new_value", "abs_diff", "parity_status", "note"]

COHORT_DESCRIPTIONS = {
    "C1_Tcorrect_Icorrect_Kcorrect": "Teacher correct, INT4 correct, Vanilla KD correct.",
    "C2_Tcorrect_Iwrong_Kcorrect": "Teacher correct -> INT4 wrong -> Vanilla KD repaired.",
    "C3_Tcorrect_Iwrong_Kwrong": "Quantization-associated residual candidate: Teacher correct -> INT4 wrong -> KD still wrong.",
    "C4_Tcorrect_Icorrect_Kwrong": "KD-induced degradation candidate: Teacher correct -> INT4 correct -> KD wrong.",
    "C5_Twrong_Kwrong": "Teacher-existing / teacher-reliability-associated residual: Teacher wrong -> KD wrong.",
    "C6_Twrong_Kcorrect": "Student exceeds teacher on this sample: Teacher wrong -> KD correct.",
}
COHORT_ORDER = tuple(COHORT_DESCRIPTIONS)
OUTPUT_ANALYSIS_GROUPS = ("tail", "medium", "few")


@dataclass(frozen=True)
class LogitArtifact:
    key: str
    path: Path
    artifact_sha256: str
    payload: Mapping[str, Any]
    sample_ids: list[str]
    labels: Tensor
    predictions: Tensor
    logits: Tensor
    buckets: list[str]
    audio_paths: list[str]
    split_hash: str
    label_mapping_hash: str
    sample_order_hash: str
    checkpoint_path: str
    checkpoint_sha256: str


def _path_str(path: str | Path | None) -> str:
    return "" if path is None else str(Path(path).resolve(strict=False))


def _file_sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"file is required for sha256: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_sanitize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _sample_order_hash(sample_ids: Sequence[Any]) -> str:
    return _json_hash([str(item) for item in sample_ids])


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


def _as_int_or_none(value: Any) -> int | None:
    number = _as_float(value)
    if not math.isfinite(number) or not float(number).is_integer():
        return None
    return int(number)


def _read_csv_rows(path: str | Path, *, context: str) -> tuple[list[dict[str, str]], list[str]]:
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"{context}: CSV not found: {target}")
    if "test" in {part.lower() for part in target.parts}:
        raise RuntimeError(f"{context}: refusing to read test CSV: {target}")
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError(f"{context}: CSV has no header: {target}")
    return rows, fields


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"JSON not found: {target}")
    if "test" in {part.lower() for part in target.parts}:
        raise RuntimeError(f"refusing to read test JSON: {target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {target}")
    return dict(payload)


def _atomic_csv_write(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in fieldnames})
    tmp.replace(target)


def _atomic_json_write(path: str | Path, payload: Mapping[str, Any]) -> None:
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
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return _json_dumps(value)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return value


def _dataset_root(dataset_root: str | Path | None) -> Path:
    if dataset_root is not None:
        return Path(dataset_root)
    current = getattr(project_config, "CURRENT_DATASET", None)
    if isinstance(current, Mapping) and current.get("ROOT"):
        return Path(current["ROOT"])
    active = str(getattr(project_config, "ACTIVE_DATASET", "local_data"))
    datasets = getattr(project_config, "DATASETS", {})
    if isinstance(datasets, Mapping) and active in datasets:
        entry = datasets[active]
        if isinstance(entry, Mapping) and entry.get("ROOT"):
            return Path(entry["ROOT"])
    return Path(getattr(project_config, "BASE_DIR", PROJECT_ROOT)) / "local_data"


def _dataset_path(root: Path, config_key: str, fallback: Sequence[str]) -> Path:
    current = getattr(project_config, "CURRENT_DATASET", None)
    if isinstance(current, Mapping) and current.get(config_key):
        configured_root = current.get("ROOT")
        if configured_root is None or Path(configured_root).resolve(strict=False) == root.resolve(strict=False):
            return Path(current[config_key])
    return root.joinpath(*fallback)


def _output_dir(root: Path, seed: int, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    return root / "export" / EXPERIMENT_NAME / f"seed_{int(seed)}"


def _logit_paths(root: Path, seed: int) -> dict[str, Path]:
    seed_dir = f"seed_{int(seed)}"
    return {
        TEACHER_KEY: root / "baseline" / "teacher" / seed_dir / "logits" / "fp32_teacher" / "eval_logits.pt",
        INT4_KEY: root / "baseline" / "student" / seed_dir / "logits" / "int4_student" / "eval_logits.pt",
        KD_KEY: root / "baseline" / "kd" / seed_dir / "logits" / "vanilla_kd" / "eval_logits.pt",
    }


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_1d(value: Any, *, name: str, dtype: torch.dtype) -> Tensor:
    if value is None:
        raise KeyError(f"{name} is missing")
    return torch.as_tensor(value, dtype=dtype).detach().cpu().reshape(-1).contiguous()


def _tensor_2d_float(value: Any, *, name: str) -> Tensor:
    if value is None:
        raise KeyError(f"{name} is missing")
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be [N,C], got shape={tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{name} contains NaN/Inf")
    return tensor.contiguous()


def _payload_list(payload: Mapping[str, Any], keys: Sequence[str], *, name: str) -> list[Any]:
    for key in keys:
        if key in payload:
            value = payload[key]
            if isinstance(value, (str, bytes)):
                raise RuntimeError(f"{name} must be a sequence, not a scalar string")
            return list(value)
    raise KeyError(f"artifact missing {name}; tried keys={tuple(keys)}")


def _load_label_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    raw = _read_json(path)
    label_to_id: dict[str, int] = {}
    id_to_label: dict[int, str] = {}
    for label, raw_id in raw.items():
        label_id = _as_int_or_none(raw_id)
        if label_id is None:
            raise RuntimeError(f"label_mapping has non-integer id: {label!r} -> {raw_id!r}")
        if label_id in id_to_label:
            raise RuntimeError(f"label_mapping has duplicate label_id={label_id}")
        label_to_id[str(label)] = int(label_id)
        id_to_label[int(label_id)] = str(label)
    expected = list(range(len(id_to_label)))
    if sorted(id_to_label) != expected:
        raise RuntimeError(f"label ids must be dense 0..C-1; got {sorted(id_to_label)}")
    return label_to_id, id_to_label


def _label_id_from_row(row: Mapping[str, Any], *, label_to_id: Mapping[str, int], context: str) -> int:
    for column in ("label_id", "class_id", "target"):
        label_id = _as_int_or_none(row.get(column))
        if label_id is not None:
            return int(label_id)
    for column in ("label", "ebird_code", "primary_label", "class_name"):
        value = row.get(column)
        if _is_missing(value):
            continue
        label = str(value).strip()
        if label in label_to_id:
            return int(label_to_id[label])
    raise RuntimeError(f"{context}: cannot resolve label id")


def _canonical_label_buckets(root: Path, label_to_id: Mapping[str, int]) -> tuple[dict[int, str], dict[str, Any]]:
    train_csv = _dataset_path(root, "TRAIN_CSV", ("splits", "train.csv"))
    rows, fields = _read_csv_rows(train_csv, context="train split")
    declared_column = "train_label_n" if "train_label_n" in fields else "label_n" if "label_n" in fields else ""
    if not declared_column:
        raise RuntimeError(
            f"{train_csv}: strict bucket protocol requires an existing train-derived class-count column "
            "named 'train_label_n' or 'label_n'"
        )
    row_counts: dict[int, int] = {}
    declared_counts: dict[int, set[int]] = {}
    for row_number, row in enumerate(rows, start=2):
        label_id = _label_id_from_row(row, label_to_id=label_to_id, context=f"{train_csv}:row={row_number}")
        row_counts[label_id] = int(row_counts.get(label_id, 0)) + 1
        if declared_column:
            declared = _as_int_or_none(row.get(declared_column))
            if declared is None:
                raise RuntimeError(f"{train_csv}:row={row_number}.{declared_column} is not an integer")
            declared_counts.setdefault(label_id, set()).add(int(declared))

    train_count_by_label: dict[int, int] = {}
    for label_id in sorted(row_counts):
        declared_values = declared_counts.get(label_id, set())
        if len(declared_values) > 1:
            raise RuntimeError(f"{train_csv}: inconsistent {declared_column} for label_id={label_id}: {sorted(declared_values)}")
        if not declared_values:
            raise RuntimeError(f"{train_csv}: missing {declared_column} for label_id={label_id}")
        train_count_by_label[int(label_id)] = int(next(iter(declared_values)))
    bucket_by_label = {
        int(label_id): bucket_from_train_label_n(int(count))
        for label_id, count in sorted(train_count_by_label.items())
    }
    return bucket_by_label, {
        "status": "validated",
        "canonical_bucket_source": "train split canonical label count via src.utils.group_utils.bucket_from_train_label_n",
        "train_csv": _path_str(train_csv),
        "train_csv_sha256": _file_sha256(train_csv),
        "strict_train_label_count_required": True,
        "train_label_count_column_used": declared_column,
        "train_label_count_by_label_id": {str(key): int(value) for key, value in sorted(train_count_by_label.items())},
        "train_physical_row_count_by_label_id": {str(key): int(value) for key, value in sorted(row_counts.items())},
        "bucket_by_label_id": {str(key): str(value) for key, value in sorted(bucket_by_label.items())},
        "bucket_protocol_hash": _json_hash(
            {
                "train_label_count_by_label_id": {str(key): int(value) for key, value in sorted(train_count_by_label.items())},
                "bucket_by_label_id": {str(key): str(value) for key, value in sorted(bucket_by_label.items())},
            }
        ),
    }


def _load_logit_artifact(key: str, path: Path) -> LogitArtifact:
    if "test" in {part.lower() for part in path.parts}:
        raise RuntimeError(f"{key}: refusing to read test logits: {path}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{key}: eval full-logit artifact not found: {path}")
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{key}: logit artifact must be a mapping")
    split = str(payload.get("split", "")).strip().lower()
    if split != ANALYSIS_SPLIT:
        raise RuntimeError(f"{key}: expected split={ANALYSIS_SPLIT!r}, got {split!r}")

    logits = _tensor_2d_float(payload.get("logits"), name=f"{key}.logits")
    labels = _tensor_1d(payload.get("label_id", payload.get("labels")), name=f"{key}.label_id", dtype=torch.long)
    predictions = logits.argmax(dim=1).detach().cpu().long()
    sample_ids = [str(item) for item in _payload_list(payload, ("sample_id", "sample_ids"), name=f"{key}.sample_id")]
    buckets = [normalize_bucket(item, field_name=f"{key}.bucket") for item in _payload_list(payload, ("bucket",), name=f"{key}.bucket")]
    audio_paths = [str(item) for item in payload.get("audio_path", [""] * len(sample_ids))]
    if len(audio_paths) != len(sample_ids):
        audio_paths = [""] * len(sample_ids)

    if len(sample_ids) != int(logits.shape[0]) or int(labels.numel()) != int(logits.shape[0]) or len(buckets) != int(logits.shape[0]):
        raise RuntimeError(f"{key}: sample_id/label/bucket/logits length mismatch")
    for pred_key in ("prediction", "predictions"):
        if pred_key not in payload:
            continue
        exported = _tensor_1d(payload[pred_key], name=f"{key}.{pred_key}", dtype=torch.long)
        if tuple(exported.shape) != tuple(predictions.shape) or not bool(torch.equal(exported, predictions)):
            mismatch = torch.nonzero(exported != predictions, as_tuple=False).reshape(-1)[:8].tolist()
            raise RuntimeError(f"{key}: exported {pred_key} differs from logits.argmax; mismatch indices={mismatch}")

    sample_hash = _sample_order_hash(sample_ids)
    recorded_sample_hash = str(payload.get("sample_order_hash", "")).strip()
    if recorded_sample_hash and recorded_sample_hash != sample_hash:
        raise RuntimeError(f"{key}: recorded sample_order_hash does not match sample_id order")

    return LogitArtifact(
        key=key,
        path=path,
        artifact_sha256=_file_sha256(path),
        payload=payload,
        sample_ids=sample_ids,
        labels=labels,
        predictions=predictions,
        logits=logits,
        buckets=buckets,
        audio_paths=audio_paths,
        split_hash=str(payload.get("split_hash", "")).strip(),
        label_mapping_hash=str(payload.get("label_mapping_hash", "")).strip(),
        sample_order_hash=sample_hash,
        checkpoint_path=str(payload.get("checkpoint_path", "")).strip(),
        checkpoint_sha256=str(payload.get("checkpoint_sha256", "")).strip(),
    )


def _validate_identity(artifacts: Mapping[str, LogitArtifact], *, label_mapping_hash: str) -> dict[str, Any]:
    reference = artifacts[TEACHER_KEY]
    for key in MODEL_KEYS:
        artifact = artifacts[key]
        if artifact.sample_ids != reference.sample_ids:
            raise RuntimeError(f"{key}: sample_id/order mismatch against {TEACHER_KEY}")
        if not bool(torch.equal(artifact.labels, reference.labels)):
            raise RuntimeError(f"{key}: true label mismatch against {TEACHER_KEY}")
        if artifact.buckets != reference.buckets:
            raise RuntimeError(f"{key}: bucket sequence mismatch against {TEACHER_KEY}")
        if tuple(artifact.logits.shape) != tuple(reference.logits.shape):
            raise RuntimeError(f"{key}: class dimension/logits shape mismatch against {TEACHER_KEY}")
        if artifact.sample_order_hash != reference.sample_order_hash:
            raise RuntimeError(f"{key}: sample_order_hash mismatch against {TEACHER_KEY}")
        if artifact.label_mapping_hash != reference.label_mapping_hash:
            raise RuntimeError(f"{key}: label_mapping_hash mismatch against {TEACHER_KEY}")
        if artifact.split_hash != reference.split_hash:
            raise RuntimeError(f"{key}: eval split_hash mismatch against {TEACHER_KEY}")
        if artifact.label_mapping_hash and artifact.label_mapping_hash != label_mapping_hash:
            raise RuntimeError(f"{key}: label_mapping_hash does not match current label_mapping.json")
    return {
        "eval_split_hash": reference.split_hash,
        "label_mapping_hash": reference.label_mapping_hash,
        "sample_order_hash": reference.sample_order_hash,
        "sample_count": int(len(reference.sample_ids)),
        "num_classes": int(reference.logits.shape[1]),
    }


def _validate_eval_split(
    root: Path,
    reference: LogitArtifact,
    *,
    label_to_id: Mapping[str, int],
    label_buckets: Mapping[int, str],
) -> dict[str, Any]:
    eval_csv = _dataset_path(root, "EVAL_CSV", ("splits", "eval.csv"))
    rows, fields = _read_csv_rows(eval_csv, context="eval split")
    eval_sha = _file_sha256(eval_csv)
    if reference.split_hash and reference.split_hash != eval_sha:
        raise RuntimeError(f"{reference.key}: artifact split_hash does not match eval.csv sha256")
    if "sample_id" not in fields:
        raise RuntimeError(f"eval split missing sample_id column: {eval_csv}")
    eval_sample_ids = [str(row.get("sample_id")) for row in rows]
    if eval_sample_ids != reference.sample_ids:
        raise RuntimeError("artifact sample_id/order does not match eval.csv")
    assignments: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = eval_sample_ids[idx]
        label_id = _label_id_from_row(row, label_to_id=label_to_id, context=f"{eval_csv}:row={idx + 2}")
        if label_id != int(reference.labels[idx].item()):
            raise RuntimeError(f"true label mismatch for sample_id={sample_id}")
        if label_id not in label_buckets:
            raise RuntimeError(f"label_id={label_id} is missing from train-derived bucket map")
        expected_bucket = str(label_buckets[label_id])
        if normalize_bucket(reference.buckets[idx], field_name=f"artifact.bucket[{idx}]") != expected_bucket:
            raise RuntimeError(f"artifact bucket mismatch for sample_id={sample_id}")
        if "bucket" in fields and not _is_missing(row.get("bucket")):
            observed_bucket = normalize_bucket(row.get("bucket"), field_name=f"eval.bucket[{idx}]")
            if observed_bucket != expected_bucket:
                raise RuntimeError(f"eval bucket mismatch for sample_id={sample_id}")
        assignments.append({"sample_id": sample_id, "label_id": int(label_id), "bucket": expected_bucket})
    return {
        "eval_csv": _path_str(eval_csv),
        "eval_csv_sha256": eval_sha,
        "eval_sample_count": int(len(rows)),
        "eval_sample_order_hash": _sample_order_hash(eval_sample_ids),
        "eval_bucket_assignment_hash": _json_hash(assignments),
    }


def _label_name(label_id: int, id_to_label: Mapping[int, str]) -> str:
    return str(id_to_label.get(int(label_id), str(label_id)))


def _class_ids_by_bucket(label_buckets: Mapping[int, str], *, num_classes: int) -> dict[str, list[int]]:
    missing = [label_id for label_id in range(num_classes) if label_id not in label_buckets]
    if missing:
        raise RuntimeError(f"bucket map missing class ids: {missing[:8]}")
    result = {bucket: [] for bucket in BASE_BUCKETS}
    result["tail"] = []
    for label_id in range(num_classes):
        bucket = normalize_bucket(label_buckets[label_id], field_name=f"label_buckets[{label_id}]")
        result[bucket].append(label_id)
        if bucket in TAIL_BUCKET_SET:
            result["tail"].append(label_id)
    return result


def _true_margin(logits: Tensor, labels: Tensor) -> Tensor:
    labels = labels.to(dtype=torch.long).reshape(-1)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    rivals = logits.clone()
    rivals.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - rivals.max(dim=1).values


def _prob_mass(probs: Tensor, class_ids: Sequence[int]) -> Tensor:
    if not class_ids:
        return probs.new_zeros((int(probs.shape[0]),))
    return probs.index_select(1, torch.as_tensor(list(class_ids), dtype=torch.long)).sum(dim=1)


def _kl(teacher_logits: Tensor, other_logits: Tensor) -> Tensor:
    teacher_log_prob = torch.log_softmax(teacher_logits / SOFTMAX_TEMPERATURE, dim=1)
    other_log_prob = torch.log_softmax(other_logits / SOFTMAX_TEMPERATURE, dim=1)
    teacher_prob = teacher_log_prob.exp()
    return (teacher_prob * (teacher_log_prob - other_log_prob)).sum(dim=1)


def _cohort(*, teacher_correct: bool, int4_correct: bool, kd_correct: bool) -> str:
    if teacher_correct and int4_correct and kd_correct:
        return "C1_Tcorrect_Icorrect_Kcorrect"
    if teacher_correct and (not int4_correct) and kd_correct:
        return "C2_Tcorrect_Iwrong_Kcorrect"
    if teacher_correct and (not int4_correct) and (not kd_correct):
        return "C3_Tcorrect_Iwrong_Kwrong"
    if teacher_correct and int4_correct and (not kd_correct):
        return "C4_Tcorrect_Icorrect_Kwrong"
    if (not teacher_correct) and (not kd_correct):
        return "C5_Twrong_Kwrong"
    if (not teacher_correct) and kd_correct:
        return "C6_Twrong_Kcorrect"
    raise RuntimeError("unreachable cohort state")


def _build_tail_samples(
    artifacts: Mapping[str, LogitArtifact],
    *,
    label_buckets: Mapping[int, str],
    id_to_label: Mapping[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher = artifacts[TEACHER_KEY]
    int4 = artifacts[INT4_KEY]
    kd = artifacts[KD_KEY]
    labels = teacher.labels
    num_classes = int(teacher.logits.shape[1])
    class_ids = _class_ids_by_bucket(label_buckets, num_classes=num_classes)

    teacher_probs = torch.softmax(teacher.logits / SOFTMAX_TEMPERATURE, dim=1)
    int4_probs = torch.softmax(int4.logits / SOFTMAX_TEMPERATURE, dim=1)
    kd_probs = torch.softmax(kd.logits / SOFTMAX_TEMPERATURE, dim=1)
    probability_sum_max_abs_error = max(
        float(torch.max(torch.abs(teacher_probs.sum(dim=1) - 1.0)).item()),
        float(torch.max(torch.abs(int4_probs.sum(dim=1) - 1.0)).item()),
        float(torch.max(torch.abs(kd_probs.sum(dim=1) - 1.0)).item()),
    )

    many_ids = class_ids["many"]
    tail_ids = class_ids["tail"]
    teacher_p_many = _prob_mass(teacher_probs, many_ids)
    int4_p_many = _prob_mass(int4_probs, many_ids)
    kd_p_many = _prob_mass(kd_probs, many_ids)
    teacher_p_tail = _prob_mass(teacher_probs, tail_ids)
    int4_p_tail = _prob_mass(int4_probs, tail_ids)
    kd_p_tail = _prob_mass(kd_probs, tail_ids)
    many_tail_sum_max_abs_error = max(
        float(torch.max(torch.abs((teacher_p_many + teacher_p_tail) - 1.0)).item()),
        float(torch.max(torch.abs((int4_p_many + int4_p_tail) - 1.0)).item()),
        float(torch.max(torch.abs((kd_p_many + kd_p_tail) - 1.0)).item()),
    )

    teacher_comp = teacher_p_many - teacher_p_tail
    int4_comp = int4_p_many - int4_p_tail
    kd_comp = kd_p_many - kd_p_tail
    teacher_margin = _true_margin(teacher.logits, labels)
    int4_margin = _true_margin(int4.logits, labels)
    kd_margin = _true_margin(kd.logits, labels)
    teacher_p_true = teacher_probs.gather(1, labels[:, None]).squeeze(1)
    teacher_correct = teacher.predictions == labels
    int4_correct = int4.predictions == labels
    kd_correct = kd.predictions == labels
    kl_teacher_int4 = _kl(teacher.logits, int4.logits)
    kl_teacher_kd = _kl(teacher.logits, kd.logits)

    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(teacher.sample_ids):
        true_bucket = normalize_bucket(teacher.buckets[idx], field_name=f"true_bucket[{idx}]")
        if true_bucket not in TAIL_BUCKET_SET:
            continue
        true_label_id = int(labels[idx].item())
        teacher_pred = int(teacher.predictions[idx].item())
        int4_pred = int(int4.predictions[idx].item())
        kd_pred = int(kd.predictions[idx].item())
        t_ok = bool(teacher_correct[idx].item())
        i_ok = bool(int4_correct[idx].item())
        k_ok = bool(kd_correct[idx].item())
        delta_int4 = float((int4_comp[idx] - teacher_comp[idx]).item())
        delta_kd = float((kd_comp[idx] - int4_comp[idx]).item())
        residual = float((kd_comp[idx] - teacher_comp[idx]).item())
        margin_drop_int4 = float((int4_margin[idx] - teacher_margin[idx]).item())
        margin_residual_kd = float((kd_margin[idx] - teacher_margin[idx]).item())
        rows.append(
            {
                "sample_index": int(idx),
                "analysis_group": true_bucket,
                "sample_id": str(sample_id),
                "audio_path": teacher.audio_paths[idx] if idx < len(teacher.audio_paths) else "",
                "true_label": _label_name(true_label_id, id_to_label),
                "true_label_id": true_label_id,
                "true_bucket": true_bucket,
                "teacher_pred": teacher_pred,
                "teacher_pred_label": _label_name(teacher_pred, id_to_label),
                "teacher_correct": t_ok,
                "teacher_margin": float(teacher_margin[idx].item()),
                "teacher_p_true": float(teacher_p_true[idx].item()),
                "teacher_P_many": float(teacher_p_many[idx].item()),
                "teacher_P_tail": float(teacher_p_tail[idx].item()),
                "teacher_competition": float(teacher_comp[idx].item()),
                "int4_pred": int4_pred,
                "int4_pred_label": _label_name(int4_pred, id_to_label),
                "int4_correct": i_ok,
                "int4_margin": float(int4_margin[idx].item()),
                "int4_P_many": float(int4_p_many[idx].item()),
                "int4_P_tail": float(int4_p_tail[idx].item()),
                "int4_competition": float(int4_comp[idx].item()),
                "kd_pred": kd_pred,
                "kd_pred_label": _label_name(kd_pred, id_to_label),
                "kd_correct": k_ok,
                "kd_margin": float(kd_margin[idx].item()),
                "kd_P_many": float(kd_p_many[idx].item()),
                "kd_P_tail": float(kd_p_tail[idx].item()),
                "kd_competition": float(kd_comp[idx].item()),
                "kd_pred_bucket": normalize_bucket(label_buckets[kd_pred], field_name=f"kd_pred_bucket[{idx}]"),
                "delta_competition_int4": delta_int4,
                "delta_competition_kd_repair": delta_kd,
                "residual_competition": residual,
                "margin_drop_int4": margin_drop_int4,
                "margin_residual_kd": margin_residual_kd,
                "kl_teacher_int4": float(kl_teacher_int4[idx].item()),
                "kl_teacher_kd": float(kl_teacher_kd[idx].item()),
                "cohort": _cohort(teacher_correct=t_ok, int4_correct=i_ok, kd_correct=k_ok),
            }
        )

    diagnostics = {
        "class_ids_by_bucket": {key: [int(item) for item in value] for key, value in class_ids.items()},
        "probability_sum_max_abs_error": probability_sum_max_abs_error,
        "many_tail_sum_max_abs_error": many_tail_sum_max_abs_error,
        "full_logits_available": True,
    }
    return rows, diagnostics


def _subset(rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> list[Mapping[str, Any]]:
    return [row for row in rows if predicate(row)]


def _finite_values(rows: Sequence[Mapping[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _as_float(row.get(column))
        if math.isfinite(value):
            values.append(float(value))
    return values


def _mean(rows: Sequence[Mapping[str, Any]], column: str) -> float:
    values = _finite_values(rows, column)
    return float(sum(values) / len(values)) if values else float("nan")


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _samples_for_analysis_group(samples: Sequence[Mapping[str, Any]], analysis_group: str) -> list[Mapping[str, Any]]:
    group = str(analysis_group).strip().lower()
    if group == "tail":
        return list(samples)
    if group in TAIL_BUCKET_SET:
        return _subset(samples, lambda row, group=group: str(row.get("true_bucket")) == group)
    raise ValueError(f"unsupported analysis_group={analysis_group!r}; expected one of {OUTPUT_ANALYSIS_GROUPS}")


def _cohort_rows_for_group(samples: Sequence[Mapping[str, Any]], analysis_group: str) -> list[dict[str, Any]]:
    n_tail = len(samples)
    kd_error_total = sum(1 for row in samples if not bool(row.get("kd_correct")))
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        selected = _subset(samples, lambda row, cohort=cohort: str(row.get("cohort")) == cohort)
        is_kd_wrong_cohort = cohort in {"C3_Tcorrect_Iwrong_Kwrong", "C4_Tcorrect_Icorrect_Kwrong", "C5_Twrong_Kwrong"}
        many_error_rows = _subset(selected, lambda row: is_kd_wrong_cohort and str(row.get("kd_pred_bucket")) == "many")
        label_counts: dict[str, int] = {}
        for row in many_error_rows:
            label = str(row.get("kd_pred_label", row.get("kd_pred", "")))
            label_counts[label] = int(label_counts.get(label, 0)) + 1
        sorted_labels = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        rows.append(
            {
                "analysis_group": str(analysis_group),
                "cohort": cohort,
                "description": COHORT_DESCRIPTIONS[cohort],
                "sample_count": int(len(selected)),
                "fraction_of_all_tail": _rate(len(selected), n_tail),
                "fraction_of_kd_tail_errors": _rate(len(selected), kd_error_total) if is_kd_wrong_cohort else float("nan"),
                "mean_teacher_margin": _mean(selected, "teacher_margin"),
                "mean_int4_margin": _mean(selected, "int4_margin"),
                "mean_kd_margin": _mean(selected, "kd_margin"),
                "mean_teacher_p_true": _mean(selected, "teacher_p_true"),
                "mean_teacher_P_many": _mean(selected, "teacher_P_many"),
                "mean_int4_P_many": _mean(selected, "int4_P_many"),
                "mean_kd_P_many": _mean(selected, "kd_P_many"),
                "mean_teacher_P_tail": _mean(selected, "teacher_P_tail"),
                "mean_int4_P_tail": _mean(selected, "int4_P_tail"),
                "mean_kd_P_tail": _mean(selected, "kd_P_tail"),
                "mean_delta_competition_int4": _mean(selected, "delta_competition_int4"),
                "mean_delta_competition_kd_repair": _mean(selected, "delta_competition_kd_repair"),
                "mean_residual_competition": _mean(selected, "residual_competition"),
                "mean_margin_drop_int4": _mean(selected, "margin_drop_int4"),
                "mean_margin_residual_kd": _mean(selected, "margin_residual_kd"),
                "mean_kl_teacher_int4": _mean(selected, "kl_teacher_int4"),
                "mean_kl_teacher_kd": _mean(selected, "kl_teacher_kd"),
                "tail_to_many_count": int(len(many_error_rows)) if is_kd_wrong_cohort else float("nan"),
                "tail_to_many_rate": _rate(len(many_error_rows), len(selected)) if is_kd_wrong_cohort else float("nan"),
                "predicted_many_label": sorted_labels[0][0] if sorted_labels else "",
                "predicted_many_label_count": int(sorted_labels[0][1]) if sorted_labels else float("nan"),
                "predicted_many_label_counts": ";".join(f"{label}:{count}" for label, count in sorted_labels),
            }
        )
    return rows


def _cohort_rows(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis_group in OUTPUT_ANALYSIS_GROUPS:
        rows.extend(_cohort_rows_for_group(_samples_for_analysis_group(samples, analysis_group), analysis_group))
    return rows


def _count(samples: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> int:
    return sum(1 for row in samples if predicate(row))


def _summary_row(
    samples: Sequence[Mapping[str, Any]],
    *,
    analysis_group: str,
    audit_parity: str,
    protocol_status: str,
) -> dict[str, Any]:
    n_tail = len(samples)
    teacher_correct = _count(samples, lambda row: bool(row.get("teacher_correct")))
    teacher_wrong = n_tail - teacher_correct
    teacher_correct_int4_wrong = _count(samples, lambda row: bool(row.get("teacher_correct")) and not bool(row.get("int4_correct")))
    c2 = _count(samples, lambda row: row.get("cohort") == "C2_Tcorrect_Iwrong_Kcorrect")
    c3 = _count(samples, lambda row: row.get("cohort") == "C3_Tcorrect_Iwrong_Kwrong")
    c4 = _count(samples, lambda row: row.get("cohort") == "C4_Tcorrect_Icorrect_Kwrong")
    c5 = _count(samples, lambda row: row.get("cohort") == "C5_Twrong_Kwrong")
    c6 = _count(samples, lambda row: row.get("cohort") == "C6_Twrong_Kcorrect")
    kd_error_total = _count(samples, lambda row: not bool(row.get("kd_correct")))
    c3_rows = _subset(samples, lambda row: row.get("cohort") == "C3_Tcorrect_Iwrong_Kwrong")
    c5_rows = _subset(samples, lambda row: row.get("cohort") == "C5_Twrong_Kwrong")
    tcorrect_kcorrect = _subset(samples, lambda row: bool(row.get("teacher_correct")) and bool(row.get("kd_correct")))
    tcorrect_kwrong = _subset(samples, lambda row: bool(row.get("teacher_correct")) and not bool(row.get("kd_correct")))
    return {
        "analysis_group": str(analysis_group),
        "tail_total": int(n_tail),
        "teacher_correct_tail": int(teacher_correct),
        "teacher_wrong_tail": int(teacher_wrong),
        "teacher_correct_int4_wrong": int(teacher_correct_int4_wrong),
        "teacher_correct_int4_wrong_kd_correct": int(c2),
        "teacher_correct_int4_wrong_kd_wrong": int(c3),
        "teacher_correct_int4_correct_kd_wrong": int(c4),
        "teacher_wrong_kd_wrong": int(c5),
        "teacher_wrong_kd_correct": int(c6),
        "int4_error_rate_on_teacher_correct": _rate(teacher_correct_int4_wrong, teacher_correct),
        "kd_repair_rate": _rate(c2, teacher_correct_int4_wrong),
        "quant_assoc_residual_rate": _rate(c3, teacher_correct),
        "kd_induced_degradation_rate": _rate(c4, teacher_correct),
        "kd_tail_error_total": int(kd_error_total),
        "C3_tail_to_many_count": _count(c3_rows, lambda row: str(row.get("kd_pred_bucket")) == "many"),
        "C3_tail_to_many_rate": _rate(_count(c3_rows, lambda row: str(row.get("kd_pred_bucket")) == "many"), len(c3_rows)),
        "C5_tail_to_many_count": _count(c5_rows, lambda row: str(row.get("kd_pred_bucket")) == "many"),
        "C5_tail_to_many_rate": _rate(_count(c5_rows, lambda row: str(row.get("kd_pred_bucket")) == "many"), len(c5_rows)),
        "mean_teacher_competition_tail": _mean(samples, "teacher_competition"),
        "mean_int4_competition_tail": _mean(samples, "int4_competition"),
        "mean_kd_competition_tail": _mean(samples, "kd_competition"),
        "mean_delta_competition_int4_tail": _mean(samples, "delta_competition_int4"),
        "mean_delta_competition_kd_repair_tail": _mean(samples, "delta_competition_kd_repair"),
        "mean_residual_competition_tail": _mean(samples, "residual_competition"),
        "mean_teacher_margin_Tcorrect_Kcorrect": _mean(tcorrect_kcorrect, "teacher_margin"),
        "mean_teacher_margin_Tcorrect_Kwrong": _mean(tcorrect_kwrong, "teacher_margin"),
        "mean_teacher_p_true_Tcorrect_Kcorrect": _mean(tcorrect_kcorrect, "teacher_p_true"),
        "mean_teacher_p_true_Tcorrect_Kwrong": _mean(tcorrect_kwrong, "teacher_p_true"),
        "audit_parity": audit_parity,
        "test_touched": False,
        "protocol_status": protocol_status,
    }


def _summary_rows(
    samples: Sequence[Mapping[str, Any]],
    *,
    audit_parity: str,
    protocol_status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis_group in OUTPUT_ANALYSIS_GROUPS:
        group_samples = _samples_for_analysis_group(samples, analysis_group)
        rows.append(
            _summary_row(
                group_samples,
                analysis_group=analysis_group,
                audit_parity=audit_parity if analysis_group == "tail" else "not_applicable",
                protocol_status=protocol_status,
            )
        )
    return rows


def _previous_audit_paths(root: Path, seed: int) -> dict[str, Path]:
    base = root / "export" / "residual_tail_many_competition" / f"seed_{int(seed)}"
    return {
        "base": base,
        "summary": base / "tail_many_competition_summary.csv",
        "samples": base / "tail_many_competition_samples.csv",
        "manifest": base / "residual_tail_many_competition_manifest.json",
    }


def _old_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parity_row(metric: str, old_value: Any, new_value: Any, *, comparable: bool, note: str = "") -> dict[str, Any]:
    old_float = _as_float(old_value)
    new_float = _as_float(new_value)
    if not comparable:
        return {
            "metric": metric,
            "old_value": old_value,
            "new_value": new_value,
            "abs_diff": float("nan"),
            "parity_status": "not_comparable_due_to_definition_difference",
            "note": note,
        }
    if not (math.isfinite(old_float) and math.isfinite(new_float)):
        status = "pass" if _is_missing(old_value) and _is_missing(new_value) else "fail"
        diff = float("nan")
    else:
        diff = abs(old_float - new_float)
        status = "pass" if diff <= (PARITY_ATOL + PARITY_RTOL * abs(old_float)) else "fail"
    return {
        "metric": metric,
        "old_value": old_value,
        "new_value": new_value,
        "abs_diff": diff,
        "parity_status": status,
        "note": note,
    }


def _build_parity_rows(root: Path, seed: int, samples: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _previous_audit_paths(root, seed)
    if not paths["summary"].exists() or not paths["samples"].exists() or not paths["manifest"].exists():
        return (
            [
                {
                    "metric": "previous_audit",
                    "old_value": "",
                    "new_value": "",
                    "abs_diff": float("nan"),
                    "parity_status": "previous_audit_not_found",
                    "note": "No residual_tail_many_competition audit was found for this dataset/seed.",
                }
            ],
            {"previous_audit_found": False, "previous_audit_path": "", "audit_parity": "not_found"},
        )

    old_manifest = _read_json(paths["manifest"])
    comparable = (
        old_manifest.get("protocol_version") == "residual_tail_many_competition.v1"
        and old_manifest.get("competition_protocol", {}).get("C_definition") == "P_M - P_T"
        and old_manifest.get("split_protocol", {}).get("test_split_touched") is False
    )
    summary_rows, _summary_fields = _read_csv_rows(paths["summary"], context="previous residual tail many competition summary")
    old_samples, _sample_fields = _read_csv_rows(paths["samples"], context="previous residual tail many competition samples")
    old_tail_row = next((row for row in summary_rows if row.get("scope") == "tail_all"), None)
    if old_tail_row is None:
        comparable = False
        old_tail_row = {}

    old_teacher_correct = _count(old_samples, lambda row: _old_bool(row.get("teacher_correct")))
    old_teacher_wrong = len(old_samples) - old_teacher_correct
    old_int4_induced = _count(old_samples, lambda row: _old_bool(row.get("teacher_correct")) and not _old_bool(row.get("int4_init_correct")))
    old_kd_repaired = _count(
        old_samples,
        lambda row: _old_bool(row.get("teacher_correct"))
        and not _old_bool(row.get("int4_init_correct"))
        and _old_bool(row.get("vanilla_kd_correct")),
    )
    old_kd_errors = _count(old_samples, lambda row: not _old_bool(row.get("vanilla_kd_correct")))
    old_tail_to_many = _count(
        old_samples,
        lambda row: (not _old_bool(row.get("vanilla_kd_correct"))) and str(row.get("vanilla_kd_prediction_bucket")) == "many",
    )

    new_values = {
        "N_tail": len(samples),
        "teacher_correct_tail": _count(samples, lambda row: bool(row.get("teacher_correct"))),
        "teacher_wrong_tail": _count(samples, lambda row: not bool(row.get("teacher_correct"))),
        "INT4_induced_error_count": _count(samples, lambda row: bool(row.get("teacher_correct")) and not bool(row.get("int4_correct"))),
        "KD_repaired_count": _count(
            samples,
            lambda row: bool(row.get("teacher_correct")) and not bool(row.get("int4_correct")) and bool(row.get("kd_correct")),
        ),
        "KD_final_tail_error_count": _count(samples, lambda row: not bool(row.get("kd_correct"))),
        "tail_to_many_count": _count(
            samples,
            lambda row: (not bool(row.get("kd_correct"))) and str(row.get("kd_pred_bucket")) == "many",
        ),
        "fp32_teacher_mean_P_many": _mean(samples, "teacher_P_many"),
        "fp32_teacher_mean_P_tail": _mean(samples, "teacher_P_tail"),
        "fp32_teacher_mean_competition": _mean(samples, "teacher_competition"),
        "int4_init_mean_P_many": _mean(samples, "int4_P_many"),
        "int4_init_mean_P_tail": _mean(samples, "int4_P_tail"),
        "int4_init_mean_competition": _mean(samples, "int4_competition"),
        "vanilla_kd_mean_P_many": _mean(samples, "kd_P_many"),
        "vanilla_kd_mean_P_tail": _mean(samples, "kd_P_tail"),
        "vanilla_kd_mean_competition": _mean(samples, "kd_competition"),
    }
    old_values = {
        "N_tail": old_tail_row.get("n_tail_samples", len(old_samples)),
        "teacher_correct_tail": old_tail_row.get("n_teacher_correct", old_teacher_correct),
        "teacher_wrong_tail": old_tail_row.get("n_teacher_wrong", old_teacher_wrong),
        "INT4_induced_error_count": old_int4_induced,
        "KD_repaired_count": old_kd_repaired,
        "KD_final_tail_error_count": old_kd_errors,
        "tail_to_many_count": old_tail_to_many,
        "fp32_teacher_mean_P_many": old_tail_row.get("fp32_teacher_mean_P_M"),
        "fp32_teacher_mean_P_tail": old_tail_row.get("fp32_teacher_mean_P_T"),
        "fp32_teacher_mean_competition": old_tail_row.get("fp32_teacher_mean_C"),
        "int4_init_mean_P_many": old_tail_row.get("int4_init_mean_P_M"),
        "int4_init_mean_P_tail": old_tail_row.get("int4_init_mean_P_T"),
        "int4_init_mean_competition": old_tail_row.get("int4_init_mean_C"),
        "vanilla_kd_mean_P_many": old_tail_row.get("vanilla_kd_mean_P_M"),
        "vanilla_kd_mean_P_tail": old_tail_row.get("vanilla_kd_mean_P_T"),
        "vanilla_kd_mean_competition": old_tail_row.get("vanilla_kd_mean_C"),
    }
    rows = [
        _parity_row(metric, old_values[metric], new_values[metric], comparable=comparable)
        for metric in old_values
    ]
    if not comparable:
        status = "not_comparable_due_to_definition_difference"
    elif any(row.get("parity_status") == "fail" for row in rows):
        status = "fail"
    else:
        status = "pass"
    return rows, {
        "previous_audit_found": True,
        "previous_audit_path": _path_str(paths["base"]),
        "audit_parity": status,
        "previous_audit_manifest": _path_str(paths["manifest"]),
    }


def _core_fields_finite(samples: Sequence[Mapping[str, Any]]) -> bool:
    fields = [
        "teacher_margin",
        "teacher_p_true",
        "teacher_P_many",
        "teacher_P_tail",
        "teacher_competition",
        "int4_margin",
        "int4_P_many",
        "int4_P_tail",
        "int4_competition",
        "kd_margin",
        "kd_P_many",
        "kd_P_tail",
        "kd_competition",
        "delta_competition_int4",
        "delta_competition_kd_repair",
        "residual_competition",
        "margin_drop_int4",
        "margin_residual_kd",
        "kl_teacher_int4",
        "kl_teacher_kd",
    ]
    for row in samples:
        for field in fields:
            if not math.isfinite(_as_float(row.get(field))):
                return False
    return True


def _protocol_status(
    samples: Sequence[Mapping[str, Any]],
    *,
    diagnostics: Mapping[str, Any],
    parity_info: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    n_tail = len(samples)
    cohort_count_sum = sum(_count(samples, lambda row, cohort=cohort: row.get("cohort") == cohort) for cohort in COHORT_ORDER)
    cohort_valid = cohort_count_sum == n_tail
    probability_valid = _as_float(diagnostics.get("probability_sum_max_abs_error")) <= 1e-5
    many_tail_sum_valid = _as_float(diagnostics.get("many_tail_sum_max_abs_error")) <= 1e-5
    parity_status = str(parity_info.get("audit_parity", "not_found"))
    parity_valid = parity_status in {"pass", "not_found", "not_comparable_due_to_definition_difference"}
    checks = {
        "identity_order_labels_buckets_consistent": True,
        "test_touched": False,
        "cohort_mutually_exclusive_and_cover_tail": cohort_valid,
        "cohort_count_sum": int(cohort_count_sum),
        "tail_total": int(n_tail),
        "probability_sum_valid": probability_valid,
        "many_tail_probability_sum_valid": many_tail_sum_valid,
        "audit_parity_valid": parity_valid,
        "core_fields_finite": _core_fields_finite(samples),
    }
    status = "PROTOCOL_PASS" if all(bool(value) for key, value in checks.items() if key not in {"test_touched"}) and checks["test_touched"] is False else "PROTOCOL_FAIL"
    return status, checks


def _build_manifest(
    *,
    root: Path,
    seed: int,
    output_dir: Path,
    outputs: Mapping[str, Path],
    artifacts: Mapping[str, LogitArtifact],
    identity: Mapping[str, Any],
    eval_metadata: Mapping[str, Any],
    bucket_provenance: Mapping[str, Any],
    label_mapping_path: Path,
    samples: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    parity_info: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    tail_summary: Mapping[str, Any],
    protocol_checks: Mapping[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "analysis_name": ANALYSIS_NAME,
        "analysis_only": True,
        "formal_export": True,
        "formal_paper_result": True,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": _path_str(root),
        "seed": int(seed),
        "models": [TEACHER_KEY, INT4_KEY, KD_KEY],
        "analysis_split": ANALYSIS_SPLIT,
        "tail_definition": "medium + few",
        "output_analysis_groups": list(OUTPUT_ANALYSIS_GROUPS),
        "main_analysis_groups": ["tail", "medium"],
        "diagnostic_analysis_groups": ["few"],
        "sample_rows_analysis_group": "actual true bucket for each true-tail sample",
        "summary_and_cohort_rows_analysis_group": "tail aggregates medium+few; medium/few rows are bucket-specific repeats",
        "test_touched": False,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "teacher_checkpoint_hash": artifacts[TEACHER_KEY].checkpoint_sha256,
        "int4_checkpoint_hash": artifacts[INT4_KEY].checkpoint_sha256,
        "vanilla_kd_checkpoint_hash": artifacts[KD_KEY].checkpoint_sha256,
        "eval_split_hash": identity.get("eval_split_hash", ""),
        "label_mapping_hash": identity.get("label_mapping_hash", ""),
        "sample_order_hash": identity.get("sample_order_hash", ""),
        "bucket_mapping_source": bucket_provenance.get("canonical_bucket_source", ""),
        "full_logits_available": bool(diagnostics.get("full_logits_available")),
        "previous_audit_found": bool(parity_info.get("previous_audit_found")),
        "previous_audit_path": parity_info.get("previous_audit_path", ""),
        "audit_parity": parity_info.get("audit_parity", ""),
        "causal_claim_allowed": False,
        "research_direction_decision_allowed": False,
        "metric_interpretation": "descriptive diagnostics only; no causal or method-selection claim is allowed",
        "label_mapping_path": _path_str(label_mapping_path),
        "label_mapping_sha256": _file_sha256(label_mapping_path),
        "eval_metadata": dict(eval_metadata),
        "bucket_provenance": dict(bucket_provenance),
        "group_protocol": group_protocol_metadata(
            tail_rule="tail is recomputed from eval records whose train-count-derived bucket is medium or few",
            canonical_group_field="bucket",
            legacy_group_alias="shot_group",
            main_comparison_group="tail",
            supplementary_extreme_group="few",
        ),
        "identity_checks": {
            "sample_id_order_equal": True,
            "true_label_equal": True,
            "label_mapping_equal": True,
            "bucket_mapping_equal": True,
            "eval_split_hash_equal": True,
            "class_dimension_equal": True,
            "eval_csv_order_matches_artifacts": True,
        },
        "source_artifacts": {
            key: {
                "path": _path_str(artifact.path),
                "sha256": artifact.artifact_sha256,
                "checkpoint_path": artifact.checkpoint_path,
                "checkpoint_sha256": artifact.checkpoint_sha256,
                "split_hash": artifact.split_hash,
                "label_mapping_hash": artifact.label_mapping_hash,
                "sample_order_hash": artifact.sample_order_hash,
                "logits_shape": list(artifact.logits.shape),
            }
            for key, artifact in artifacts.items()
        },
        "metric_definitions": {
            "probability": "softmax(logits / 1.0, dim=-1)",
            "P_many": "sum probability over train-count-derived many classes",
            "P_tail": "sum probability over train-count-derived medium + few classes",
            "competition_score": "P_many - P_tail",
            "delta_competition_int4": "competition_int4 - competition_teacher",
            "delta_competition_kd_repair": "competition_kd - competition_int4",
            "residual_competition": "competition_kd - competition_teacher",
            "margin": "z_true - max_{j != true} z_j",
            "margin_drop_int4": "int4_margin - teacher_margin",
            "margin_residual_kd": "kd_margin - teacher_margin",
            "KL": "KL(teacher probability || student probability), descriptive diagnostic only",
        },
        "row_counts": {
            "tail_samples": int(len(samples)),
            "cohort_rows": int(len(cohort_rows)),
            "parity_rows": int(len(parity_rows)),
            "summary_rows": int(len(summary_rows)),
        },
        "protocol_checks": dict(protocol_checks),
        "tail_summary": dict(tail_summary),
        "summary_rows": [dict(row) for row in summary_rows],
        "outputs": {key: _path_str(path) for key, path in outputs.items()},
        "output_dir": _path_str(output_dir),
        "dry_run": bool(dry_run),
    }


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    root = _dataset_root(args.dataset_root)
    label_mapping_path = _dataset_path(root, "LABEL_MAPPING", ("splits", "label_mapping.json"))
    label_to_id, id_to_label = _load_label_mapping(label_mapping_path)
    label_mapping_hash = _file_sha256(label_mapping_path)
    label_buckets, bucket_provenance = _canonical_label_buckets(root, label_to_id)
    artifacts = {key: _load_logit_artifact(key, path) for key, path in _logit_paths(root, int(args.seed)).items()}
    identity = _validate_identity(artifacts, label_mapping_hash=label_mapping_hash)
    eval_metadata = _validate_eval_split(root, artifacts[TEACHER_KEY], label_to_id=label_to_id, label_buckets=label_buckets)
    samples, diagnostics = _build_tail_samples(artifacts, label_buckets=label_buckets, id_to_label=id_to_label)
    cohort_rows = _cohort_rows(samples)
    parity_rows, parity_info = _build_parity_rows(root, int(args.seed), samples)
    protocol_status, protocol_checks = _protocol_status(samples, diagnostics=diagnostics, parity_info=parity_info)
    summary_rows = _summary_rows(
        samples,
        audit_parity=str(parity_info.get("audit_parity", "")),
        protocol_status=protocol_status,
    )
    tail_summary = next(row for row in summary_rows if row.get("analysis_group") == "tail")

    output_dir = _output_dir(root, int(args.seed), args.output_dir)
    outputs = {
        "summary_csv": output_dir / SUMMARY_FILENAME,
        "samples_csv": output_dir / SAMPLES_FILENAME,
        "cohorts_csv": output_dir / COHORTS_FILENAME,
        "parity_csv": output_dir / PARITY_FILENAME,
        "manifest_json": output_dir / MANIFEST_FILENAME,
    }
    manifest = _build_manifest(
        root=root,
        seed=int(args.seed),
        output_dir=output_dir,
        outputs=outputs,
        artifacts=artifacts,
        identity=identity,
        eval_metadata=eval_metadata,
        bucket_provenance=bucket_provenance,
        label_mapping_path=label_mapping_path,
        samples=samples,
        cohort_rows=cohort_rows,
        parity_rows=parity_rows,
        parity_info=parity_info,
        diagnostics=diagnostics,
        summary_rows=summary_rows,
        tail_summary=tail_summary,
        protocol_checks=protocol_checks,
        dry_run=bool(args.dry_run),
    )

    if not bool(args.dry_run):
        _atomic_csv_write(outputs["summary_csv"], summary_rows, SUMMARY_COLUMNS)
        _atomic_csv_write(outputs["samples_csv"], samples, SAMPLE_COLUMNS)
        _atomic_csv_write(outputs["cohorts_csv"], cohort_rows, COHORT_COLUMNS)
        _atomic_csv_write(outputs["parity_csv"], parity_rows, PARITY_COLUMNS)
        manifest["artifact_sha256"] = {
            SUMMARY_FILENAME: _file_sha256(outputs["summary_csv"]),
            SAMPLES_FILENAME: _file_sha256(outputs["samples_csv"]),
            COHORTS_FILENAME: _file_sha256(outputs["cohorts_csv"]),
            PARITY_FILENAME: _file_sha256(outputs["parity_csv"]),
        }
        manifest["run_manifest_hash"] = _json_hash(
            {key: value for key, value in manifest.items() if key not in {"run_manifest_hash", "run_manifest_hash_algorithm"}}
        )
        manifest["run_manifest_hash_algorithm"] = "sha256-canonical-json-before-run_manifest_hash"
        _atomic_json_write(outputs["manifest_json"], manifest)

    return {
        "root": root,
        "output_dir": output_dir,
        "outputs": outputs,
        "samples": samples,
        "cohort_rows": cohort_rows,
        "parity_rows": parity_rows,
        "parity_info": parity_info,
        "summary": tail_summary,
        "summary_rows": summary_rows,
        "manifest": manifest,
        "protocol_status": protocol_status,
        "dry_run": bool(args.dry_run),
    }


def _format_rate(value: Any) -> str:
    number = _as_float(value)
    return "nan" if not math.isfinite(number) else f"{number:.6g}"


def print_console_report(result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    print("[Vanilla KD Residual Tail Attribution]")
    print()
    print(f"Tail samples: {summary['tail_total']}")
    print(f"Teacher correct: {summary['teacher_correct_tail']}")
    print(f"Teacher wrong: {summary['teacher_wrong_tail']}")
    print()
    print(f"Teacher correct -> INT4 wrong: {summary['teacher_correct_int4_wrong']}")
    print(f"Teacher correct -> INT4 wrong -> KD correct: {summary['teacher_correct_int4_wrong_kd_correct']}")
    print(f"Teacher correct -> INT4 wrong -> KD wrong: {summary['teacher_correct_int4_wrong_kd_wrong']}")
    print(f"Teacher correct -> INT4 correct -> KD wrong: {summary['teacher_correct_int4_correct_kd_wrong']}")
    print()
    print(f"Teacher wrong -> KD wrong: {summary['teacher_wrong_kd_wrong']}")
    print(f"Teacher wrong -> KD correct: {summary['teacher_wrong_kd_correct']}")
    print()
    print(f"KD repair rate: {_format_rate(summary['kd_repair_rate'])}")
    print(f"Quantization-associated residual rate: {_format_rate(summary['quant_assoc_residual_rate'])}")
    print(f"KD-induced degradation rate: {_format_rate(summary['kd_induced_degradation_rate'])}")
    print()
    print(f"C3 tail-to-many: {summary['C3_tail_to_many_count']} / {_format_rate(summary['C3_tail_to_many_rate'])}")
    print(f"C5 tail-to-many: {summary['C5_tail_to_many_count']} / {_format_rate(summary['C5_tail_to_many_rate'])}")
    print()
    print(f"Audit parity: {summary['audit_parity']}")
    print(f"Test touched: {str(summary['test_touched']).lower()}")
    print()
    print(f"Protocol status: {summary['protocol_status']}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ANALYSIS_NAME)
    parser.add_argument("--dataset-root", default=None, help="Default: config.CURRENT_DATASET['ROOT'] or local_data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=None, help="Default: <dataset_root>/export/<experiment_name>/seed_<seed>.")
    parser.add_argument("--dry-run", action="store_true", help="Build tables in memory without writing output files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    result = run_export(parse_args(argv))
    print_console_report(result)
    return result


if __name__ == "__main__":
    main()
