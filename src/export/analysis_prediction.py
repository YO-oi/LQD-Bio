from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.utils.group_utils import bucket_from_train_label_n, group_protocol_metadata, tail_included, validate_bucket_values


PROTOCOL_VERSION = "analysis_prediction.v1"
ANALYSIS_NAME = "Medium/Tail Prediction Absorption Analysis"
DEFAULT_SEED = 2026
TAIL_BUCKETS = {"medium", "few"}
MANY_BUCKET = "many"


@dataclass(frozen=True)
class ModelSpec:
    prefix: str
    display_name: str
    default_relative_path: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec(
        prefix="fp32_teacher",
        display_name="FP32 teacher",
        default_relative_path=("baseline", "teacher", "seed_{seed}", "predictions", "fp32_teacher", "eval_predictions.csv"),
    ),
    ModelSpec(
        prefix="int4_init",
        display_name="INT4 init",
        default_relative_path=("baseline", "student", "seed_{seed}", "predictions", "int4_student", "eval_predictions.csv"),
    ),
    ModelSpec(
        prefix="vanilla_kd",
        display_name="Vanilla KD best",
        default_relative_path=("baseline", "kd", "seed_{seed}", "predictions", "vanilla_kd", "eval_predictions.csv"),
    ),
    ModelSpec(
        prefix="two_stage_ce_only",
        display_name="Two-Stage CE-only",
        default_relative_path=(
            "innovation",
            "two_stage",
            "seed_{seed}",
            "stage2_export",
            "predictions",
            "two_stage_decoupling_stage2",
            "eval_predictions.csv",
        ),
    ),
)


REQUIRED_PREDICTION_COLUMNS = (
    "sample_index",
    "sample_id",
    "label_id",
    "prediction",
    "correct",
    "bucket",
    "tail_included",
    "correct_logit_margin",
)

BASE_SUMMARY_COLUMNS = [
    "dataset_name",
    "seed",
    "source_true_bucket",
    "source_tail_included",
    "true_label_id",
    "true_label_name",
    "true_eval_support",
    "absorbed_many_label_id",
    "absorbed_many_label_name",
    "absorbed_many_bucket",
    "models_with_absorption",
]

PER_MODEL_SUMMARY_COLUMNS = [
    "absorbed_count",
    "absorbed_rate",
    "correct_logit_margin_mean",
    "correct_logit_margin_valid_count",
    "fp32_margin_on_same_samples_mean",
    "correct_logit_margin_delta_to_fp32_mean",
    "correct_logit_margin_delta_to_fp32_valid_count",
]

BASE_SAMPLE_COLUMNS = [
    "dataset_name",
    "seed",
    "sample_index",
    "sample_id",
    "audio_path",
    "true_label_id",
    "true_label_name",
    "true_bucket",
    "tail_included",
    "absorbing_many_labels_union",
]

PER_MODEL_SAMPLE_COLUMNS = [
    "prediction",
    "prediction_label_name",
    "prediction_bucket",
    "correct",
    "absorbed_into_many",
    "correct_logit_margin",
    "correct_logit_margin_delta_to_fp32",
]


@dataclass(frozen=True)
class PredictionRecord:
    sample_index: int
    sample_id: str
    audio_path: str
    label_id: int
    prediction: int
    correct: bool
    bucket: str
    tail_included: bool
    correct_logit_margin: float
    top1_top2_gap: float


def _file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _path_str(path: str | Path) -> str:
    return str(Path(path).resolve(strict=False))


def _json_dump(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    tmp.replace(target)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(target)


def _dataset_entry(dataset_name: str | None) -> tuple[str, Mapping[str, Any]]:
    name = dataset_name or str(getattr(project_config, "ACTIVE_DATASET", ""))
    datasets = getattr(project_config, "DATASETS", {})
    if not isinstance(datasets, Mapping) or name not in datasets:
        raise ValueError(f"Unknown dataset={name!r}; choose one of {tuple(datasets)}")
    entry = datasets[name]
    if not isinstance(entry, Mapping):
        raise TypeError(f"config.DATASETS[{name!r}] must be a mapping")
    return name, entry


def _path_from_dataset(entry: Mapping[str, Any], key: str, override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    if key not in entry:
        raise KeyError(f"dataset config missing {key}")
    return Path(entry[key])


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


def _load_label_mapping(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"label mapping not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"label mapping must be a non-empty object: {target}")
    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    for label_name, raw_id in raw.items():
        label_id = int(raw_id)
        if label_id in id_to_name:
            raise ValueError(f"duplicate label_id={label_id} in label mapping")
        name = str(label_name)
        name_to_id[name] = label_id
        id_to_name[label_id] = name
    return name_to_id, id_to_name


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "<na>"}


def _parse_int(value: Any, *, context: str) -> int:
    if _is_missing(value) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer, got {value!r}")
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"{context} must be an integer, got {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{context} must be an integer, got {value!r}")
    return int(numeric)


def _parse_bool(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{context} must be a boolean, got {value!r}")


def _as_float_or_nan(value: Any, *, context: str) -> float:
    if _is_missing(value):
        return float("nan")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{context} must be a float or NaN, got {value!r}") from exc


def _select_label_count_column(fieldnames: Sequence[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise KeyError(f"requested train label count column {explicit!r} is missing")
        return str(explicit)
    for candidate in ("train_label_n", "label_n"):
        if candidate in fieldnames:
            return candidate
    raise KeyError("train CSV must contain train_label_n or label_n to derive class buckets")


def _label_id_from_metadata_row(
    row: Mapping[str, Any],
    *,
    label_name_to_id: Mapping[str, int],
    context: str,
) -> int:
    for key in ("label_id", "class_id", "target"):
        if not _is_missing(row.get(key)):
            return _parse_int(row.get(key), context=f"{context}.{key}")
    for key in ("label", "ebird_code", "class_name"):
        value = row.get(key)
        if _is_missing(value):
            continue
        text = str(value).strip()
        if text in label_name_to_id:
            return int(label_name_to_id[text])
    raise KeyError(f"{context} is missing label identity: expected label_id or label name from label mapping")


def _canonical_bucket(value: Any, *, context: str) -> str:
    return validate_bucket_values([value], context=context)[0]


def _load_train_label_buckets(
    train_csv: Path,
    *,
    label_name_to_id: Mapping[str, int],
    label_count_column: str | None,
) -> tuple[dict[int, str], str, dict[int, int]]:
    rows, fieldnames = _read_csv_rows(train_csv)
    count_column = _select_label_count_column(fieldnames, label_count_column)
    count_by_label: dict[int, int] = {}
    bucket_by_label: dict[int, str] = {}
    for row_number, row in enumerate(rows, start=2):
        label_id = _label_id_from_metadata_row(row, label_name_to_id=label_name_to_id, context=f"{train_csv}:row={row_number}")
        train_count = _parse_int(row.get(count_column), context=f"{train_csv}:row={row_number}.{count_column}")
        expected_bucket = bucket_from_train_label_n(train_count)
        previous_count = count_by_label.get(label_id)
        if previous_count is not None and previous_count != train_count:
            raise ValueError(
                f"{train_csv}: inconsistent {count_column} for label_id={label_id}: "
                f"{previous_count!r} vs {train_count!r}"
            )
        previous_bucket = bucket_by_label.get(label_id)
        if previous_bucket is not None and previous_bucket != expected_bucket:
            raise ValueError(
                f"{train_csv}: inconsistent derived bucket for label_id={label_id}: "
                f"{previous_bucket!r} vs {expected_bucket!r}"
            )
        if "bucket" in fieldnames and not _is_missing(row.get("bucket")):
            observed_bucket = _canonical_bucket(row.get("bucket"), context=f"{train_csv}:row={row_number}")
            if observed_bucket != expected_bucket:
                raise ValueError(
                    f"{train_csv}: bucket mismatch for label_id={label_id}; "
                    f"bucket={observed_bucket!r}, {count_column}-derived={expected_bucket!r}"
                )
        count_by_label[label_id] = train_count
        bucket_by_label[label_id] = expected_bucket
    if not bucket_by_label:
        raise RuntimeError(f"train CSV has no label bucket rows: {train_csv}")
    return bucket_by_label, count_column, count_by_label


def _read_eval_identity(
    eval_csv: Path,
    *,
    label_name_to_id: Mapping[str, int],
    label_buckets: Mapping[int, str],
) -> list[tuple[str, int, str]]:
    rows, fieldnames = _read_csv_rows(eval_csv)
    if "sample_id" not in fieldnames:
        raise KeyError(f"eval CSV must contain sample_id: {eval_csv}")
    result: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{eval_csv}:row={row_number} has blank sample_id")
        if sample_id in seen:
            raise ValueError(f"{eval_csv}: duplicate sample_id={sample_id!r}")
        seen.add(sample_id)
        label_id = _label_id_from_metadata_row(row, label_name_to_id=label_name_to_id, context=f"{eval_csv}:row={row_number}")
        if label_id not in label_buckets:
            raise KeyError(f"{eval_csv}:row={row_number} label_id={label_id} is absent from train bucket map")
        result.append((sample_id, label_id, str(label_buckets[label_id])))
    if not result:
        raise RuntimeError(f"eval CSV has no rows: {eval_csv}")
    return result


def _read_prediction_csv(path: str | Path, *, model_name: str, label_buckets: Mapping[int, str]) -> list[PredictionRecord]:
    rows, fieldnames = _read_csv_rows(path)
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in fieldnames]
    if missing:
        raise KeyError(f"{model_name} prediction CSV missing required columns {missing}: {path}")
    records: list[PredictionRecord] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{model_name}:{path}:row={row_number} has blank sample_id")
        if sample_id in seen:
            raise ValueError(f"{model_name}:{path}: duplicate sample_id={sample_id!r}")
        seen.add(sample_id)
        label_id = _parse_int(row.get("label_id"), context=f"{model_name}:{path}:row={row_number}.label_id")
        prediction = _parse_int(row.get("prediction"), context=f"{model_name}:{path}:row={row_number}.prediction")
        correct = _parse_bool(row.get("correct"), context=f"{model_name}:{path}:row={row_number}.correct")
        if correct != (prediction == label_id):
            raise ValueError(
                f"{model_name}:{path}:row={row_number} correct={correct!r} disagrees with prediction==label_id"
            )
        if label_id not in label_buckets:
            raise KeyError(f"{model_name}:{path}:row={row_number} label_id={label_id} is absent from train bucket map")
        if prediction not in label_buckets:
            raise KeyError(f"{model_name}:{path}:row={row_number} prediction={prediction} is absent from train bucket map")
        bucket = _canonical_bucket(row.get("bucket"), context=f"{model_name}:{path}:row={row_number}")
        expected_bucket = str(label_buckets[label_id])
        if bucket != expected_bucket:
            raise ValueError(
                f"{model_name}:{path}:row={row_number} bucket={bucket!r} disagrees with train-derived "
                f"label_id={label_id} bucket={expected_bucket!r}"
            )
        tail_flag = _parse_bool(row.get("tail_included"), context=f"{model_name}:{path}:row={row_number}.tail_included")
        expected_tail = bool(tail_included([bucket])[0])
        if tail_flag != expected_tail:
            raise ValueError(
                f"{model_name}:{path}:row={row_number} tail_included={tail_flag!r} disagrees with bucket={bucket!r}"
            )
        records.append(
            PredictionRecord(
                sample_index=_parse_int(row.get("sample_index"), context=f"{model_name}:{path}:row={row_number}.sample_index"),
                sample_id=sample_id,
                audio_path=str(row.get("audio_path", "")),
                label_id=label_id,
                prediction=prediction,
                correct=correct,
                bucket=bucket,
                tail_included=tail_flag,
                correct_logit_margin=_as_float_or_nan(
                    row.get("correct_logit_margin"),
                    context=f"{model_name}:{path}:row={row_number}.correct_logit_margin",
                ),
                top1_top2_gap=_as_float_or_nan(
                    row.get("top1_top2_gap"),
                    context=f"{model_name}:{path}:row={row_number}.top1_top2_gap",
                ),
            )
        )
    if not records:
        raise RuntimeError(f"{model_name} prediction CSV has no rows: {path}")
    return records


def _sample_order_hash(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _identity_sequence(records: Sequence[PredictionRecord]) -> list[tuple[str, int, str]]:
    return [(record.sample_id, int(record.label_id), str(record.bucket)) for record in records]


def _first_mismatch(
    left: Sequence[tuple[str, int, str]],
    right: Sequence[tuple[str, int, str]],
) -> dict[str, Any]:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {"index": index, "left": left[index], "right": right[index]}
    if len(left) != len(right):
        return {"index": limit, "left_len": len(left), "right_len": len(right)}
    return {}


def _validate_prediction_identity(
    predictions: Mapping[str, Sequence[PredictionRecord]],
    *,
    eval_identity: Sequence[tuple[str, int, str]],
) -> None:
    reference_prefix = MODEL_SPECS[0].prefix
    reference = list(_identity_sequence(predictions[reference_prefix]))
    if list(eval_identity) != reference:
        mismatch = _first_mismatch(list(eval_identity), reference)
        raise RuntimeError(f"eval.csv and FP32 prediction identity/order differ: {mismatch}")
    for spec in MODEL_SPECS[1:]:
        current = list(_identity_sequence(predictions[spec.prefix]))
        if current != reference:
            mismatch = _first_mismatch(reference, current)
            raise RuntimeError(
                f"prediction identity/order differs between {reference_prefix} and {spec.prefix}: {mismatch}"
            )


def _default_prediction_path(dataset_root: Path, seed: int, spec: ModelSpec) -> Path:
    parts = [part.format(seed=int(seed)) for part in spec.default_relative_path]
    return dataset_root.joinpath(*parts)


def _prediction_paths(args: argparse.Namespace, dataset_root: Path) -> dict[str, Path]:
    overrides = {
        "fp32_teacher": args.fp32_predictions,
        "int4_init": args.int4_predictions,
        "vanilla_kd": args.vanilla_kd_predictions,
        "two_stage_ce_only": args.two_stage_ce_predictions,
    }
    return {
        spec.prefix: Path(overrides[spec.prefix])
        if overrides[spec.prefix]
        else _default_prediction_path(dataset_root, int(args.seed), spec)
        for spec in MODEL_SPECS
    }


def _output_dir(dataset_root: Path, seed: int, override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    return dataset_root / "export" / "analysis_prediction" / f"seed_{int(seed)}"


def _label_name(label_id: int, id_to_name: Mapping[int, str]) -> str:
    return str(id_to_name.get(int(label_id), str(label_id)))


def _mean_finite(values: Sequence[float]) -> tuple[float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), 0
    return float(sum(finite) / float(len(finite))), int(len(finite))


def _is_absorbed_into_many(record: PredictionRecord, *, label_buckets: Mapping[int, str]) -> bool:
    true_bucket = str(label_buckets[int(record.label_id)])
    pred_bucket = str(label_buckets[int(record.prediction)])
    return true_bucket in TAIL_BUCKETS and pred_bucket == MANY_BUCKET and int(record.prediction) != int(record.label_id)


def _absorbed_pairs(
    predictions: Mapping[str, Sequence[PredictionRecord]],
    *,
    label_buckets: Mapping[int, str],
) -> list[tuple[str, int, int]]:
    pairs: set[tuple[str, int, int]] = set()
    for records in predictions.values():
        for record in records:
            if not _is_absorbed_into_many(record, label_buckets=label_buckets):
                continue
            pairs.add((str(label_buckets[int(record.label_id)]), int(record.label_id), int(record.prediction)))
    bucket_order = {"medium": 0, "few": 1}
    return sorted(pairs, key=lambda item: (bucket_order.get(item[0], 99), item[1], item[2]))


def _support_by_label(records: Sequence[PredictionRecord]) -> Counter[int]:
    return Counter(int(record.label_id) for record in records)


def _records_by_sample(records: Sequence[PredictionRecord]) -> dict[str, PredictionRecord]:
    return {str(record.sample_id): record for record in records}


def _summarize_pair_for_model(
    *,
    records: Sequence[PredictionRecord],
    fp32_by_sample: Mapping[str, PredictionRecord],
    true_label_id: int,
    absorbed_many_label_id: int,
    true_support: int,
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if int(record.label_id) == int(true_label_id)
        and int(record.prediction) == int(absorbed_many_label_id)
        and int(record.prediction) != int(record.label_id)
    ]
    current_mean, current_valid = _mean_finite([record.correct_logit_margin for record in selected])
    fp32_same_values: list[float] = []
    delta_values: list[float] = []
    for record in selected:
        fp32_record = fp32_by_sample.get(str(record.sample_id))
        if fp32_record is None:
            raise RuntimeError(f"missing FP32 prediction row for sample_id={record.sample_id!r}")
        fp32_margin = float(fp32_record.correct_logit_margin)
        current_margin = float(record.correct_logit_margin)
        fp32_same_values.append(fp32_margin)
        if math.isfinite(fp32_margin) and math.isfinite(current_margin):
            delta_values.append(current_margin - fp32_margin)
    fp32_same_mean, _fp32_same_valid = _mean_finite(fp32_same_values)
    delta_mean, delta_valid = _mean_finite(delta_values)
    return {
        "absorbed_count": int(len(selected)),
        "absorbed_rate": float(len(selected) / true_support) if int(true_support) > 0 else float("nan"),
        "correct_logit_margin_mean": current_mean,
        "correct_logit_margin_valid_count": current_valid,
        "fp32_margin_on_same_samples_mean": fp32_same_mean,
        "correct_logit_margin_delta_to_fp32_mean": delta_mean,
        "correct_logit_margin_delta_to_fp32_valid_count": delta_valid,
    }


def _build_summary_rows(
    *,
    dataset_name: str,
    seed: int,
    predictions: Mapping[str, Sequence[PredictionRecord]],
    label_buckets: Mapping[int, str],
    id_to_name: Mapping[int, str],
) -> list[dict[str, Any]]:
    fp32_records = predictions["fp32_teacher"]
    fp32_by_sample = _records_by_sample(fp32_records)
    support = _support_by_label(fp32_records)
    rows: list[dict[str, Any]] = []
    for true_bucket, true_label_id, absorbed_many_label_id in _absorbed_pairs(predictions, label_buckets=label_buckets):
        true_support = int(support[int(true_label_id)])
        row: dict[str, Any] = {
            "dataset_name": str(dataset_name),
            "seed": int(seed),
            "source_true_bucket": str(true_bucket),
            "source_tail_included": bool(str(true_bucket) in TAIL_BUCKETS),
            "true_label_id": int(true_label_id),
            "true_label_name": _label_name(true_label_id, id_to_name),
            "true_eval_support": true_support,
            "absorbed_many_label_id": int(absorbed_many_label_id),
            "absorbed_many_label_name": _label_name(absorbed_many_label_id, id_to_name),
            "absorbed_many_bucket": str(label_buckets[int(absorbed_many_label_id)]),
        }
        models_with_absorption: list[str] = []
        for spec in MODEL_SPECS:
            values = _summarize_pair_for_model(
                records=predictions[spec.prefix],
                fp32_by_sample=fp32_by_sample,
                true_label_id=int(true_label_id),
                absorbed_many_label_id=int(absorbed_many_label_id),
                true_support=true_support,
            )
            if int(values["absorbed_count"]) > 0:
                models_with_absorption.append(spec.prefix)
            for key, value in values.items():
                row[f"{spec.prefix}_{key}"] = value
        row["models_with_absorption"] = ";".join(models_with_absorption)
        rows.append(row)
    return rows


def _sample_absorbing_many_labels(
    records_by_model: Mapping[str, PredictionRecord],
    *,
    label_buckets: Mapping[int, str],
    id_to_name: Mapping[int, str],
) -> str:
    labels: list[str] = []
    seen: set[int] = set()
    for record in records_by_model.values():
        if not _is_absorbed_into_many(record, label_buckets=label_buckets):
            continue
        label_id = int(record.prediction)
        if label_id in seen:
            continue
        seen.add(label_id)
        labels.append(f"{label_id}:{_label_name(label_id, id_to_name)}")
    return ";".join(labels)


def _build_sample_rows(
    *,
    dataset_name: str,
    seed: int,
    predictions: Mapping[str, Sequence[PredictionRecord]],
    label_buckets: Mapping[int, str],
    id_to_name: Mapping[int, str],
) -> list[dict[str, Any]]:
    fp32_records = predictions["fp32_teacher"]
    fp32_by_sample = _records_by_sample(fp32_records)
    records_by_prefix_and_sample = {
        prefix: _records_by_sample(records)
        for prefix, records in predictions.items()
    }
    rows: list[dict[str, Any]] = []
    for base_record in fp32_records:
        true_bucket = str(label_buckets[int(base_record.label_id)])
        if true_bucket not in TAIL_BUCKETS:
            continue
        sample_records = {
            prefix: by_sample[str(base_record.sample_id)]
            for prefix, by_sample in records_by_prefix_and_sample.items()
        }
        if not any(_is_absorbed_into_many(record, label_buckets=label_buckets) for record in sample_records.values()):
            continue
        row: dict[str, Any] = {
            "dataset_name": str(dataset_name),
            "seed": int(seed),
            "sample_index": int(base_record.sample_index),
            "sample_id": str(base_record.sample_id),
            "audio_path": str(base_record.audio_path),
            "true_label_id": int(base_record.label_id),
            "true_label_name": _label_name(int(base_record.label_id), id_to_name),
            "true_bucket": true_bucket,
            "tail_included": bool(true_bucket in TAIL_BUCKETS),
            "absorbing_many_labels_union": _sample_absorbing_many_labels(
                sample_records,
                label_buckets=label_buckets,
                id_to_name=id_to_name,
            ),
        }
        fp32_margin = float(fp32_by_sample[str(base_record.sample_id)].correct_logit_margin)
        for spec in MODEL_SPECS:
            record = sample_records[spec.prefix]
            pred_bucket = str(label_buckets[int(record.prediction)])
            margin = float(record.correct_logit_margin)
            row[f"{spec.prefix}_prediction"] = int(record.prediction)
            row[f"{spec.prefix}_prediction_label_name"] = _label_name(int(record.prediction), id_to_name)
            row[f"{spec.prefix}_prediction_bucket"] = pred_bucket
            row[f"{spec.prefix}_correct"] = bool(record.correct)
            row[f"{spec.prefix}_absorbed_into_many"] = bool(
                _is_absorbed_into_many(record, label_buckets=label_buckets)
            )
            row[f"{spec.prefix}_correct_logit_margin"] = margin
            row[f"{spec.prefix}_correct_logit_margin_delta_to_fp32"] = (
                margin - fp32_margin if math.isfinite(margin) and math.isfinite(fp32_margin) else float("nan")
            )
        rows.append(row)
    return rows


def _summary_columns() -> list[str]:
    columns = list(BASE_SUMMARY_COLUMNS)
    for spec in MODEL_SPECS:
        columns.extend(f"{spec.prefix}_{column}" for column in PER_MODEL_SUMMARY_COLUMNS)
    return columns


def _sample_columns() -> list[str]:
    columns = list(BASE_SAMPLE_COLUMNS)
    for spec in MODEL_SPECS:
        columns.extend(f"{spec.prefix}_{column}" for column in PER_MODEL_SAMPLE_COLUMNS)
    return columns


def _prediction_source_manifest(
    *,
    paths: Mapping[str, Path],
    predictions: Mapping[str, Sequence[PredictionRecord]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        path = paths[spec.prefix]
        records = predictions[spec.prefix]
        result[spec.prefix] = {
            "display_name": spec.display_name,
            "path": _path_str(path),
            "sha256": _file_sha256(path),
            "row_count": int(len(records)),
            "sample_order_hash": _sample_order_hash(record.sample_id for record in records),
        }
    return result


def _build_absorption_counts(predictions: Mapping[str, Sequence[PredictionRecord]], *, label_buckets: Mapping[int, str]) -> dict[str, int]:
    return {
        spec.prefix: sum(1 for record in predictions[spec.prefix] if _is_absorbed_into_many(record, label_buckets=label_buckets))
        for spec in MODEL_SPECS
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ANALYSIS_NAME)
    parser.add_argument("--dataset", default=None, help="Default: config.ACTIVE_DATASET.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--label-mapping-path", default=None)
    parser.add_argument("--train-label-count-column", default=None)
    parser.add_argument("--fp32-predictions", default=None)
    parser.add_argument("--int4-predictions", default=None)
    parser.add_argument("--vanilla-kd-predictions", default=None)
    parser.add_argument("--two-stage-ce-predictions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-sample-rows", action="store_true", help="Only write the summary table and manifest.")
    return parser.parse_args(argv)


def run_analysis_prediction(args: argparse.Namespace) -> dict[str, Any]:
    dataset_name, dataset_entry = _dataset_entry(args.dataset)
    dataset_root = Path(args.dataset_root) if args.dataset_root else _path_from_dataset(dataset_entry, "ROOT")
    train_csv = _path_from_dataset(dataset_entry, "TRAIN_CSV", args.train_csv)
    eval_csv = _path_from_dataset(dataset_entry, "EVAL_CSV", args.eval_csv)
    label_mapping_path = _path_from_dataset(dataset_entry, "LABEL_MAPPING", args.label_mapping_path)
    output_dir = _output_dir(dataset_root, int(args.seed), args.output_dir)
    prediction_paths = _prediction_paths(args, dataset_root)

    label_name_to_id, id_to_name = _load_label_mapping(label_mapping_path)
    label_buckets, label_count_column, train_label_counts = _load_train_label_buckets(
        train_csv,
        label_name_to_id=label_name_to_id,
        label_count_column=args.train_label_count_column,
    )
    eval_identity = _read_eval_identity(
        eval_csv,
        label_name_to_id=label_name_to_id,
        label_buckets=label_buckets,
    )
    predictions = {
        spec.prefix: _read_prediction_csv(
            prediction_paths[spec.prefix],
            model_name=spec.prefix,
            label_buckets=label_buckets,
        )
        for spec in MODEL_SPECS
    }
    _validate_prediction_identity(predictions, eval_identity=eval_identity)

    summary_rows = _build_summary_rows(
        dataset_name=dataset_name,
        seed=int(args.seed),
        predictions=predictions,
        label_buckets=label_buckets,
        id_to_name=id_to_name,
    )
    sample_rows = [] if bool(args.no_sample_rows) else _build_sample_rows(
        dataset_name=dataset_name,
        seed=int(args.seed),
        predictions=predictions,
        label_buckets=label_buckets,
        id_to_name=id_to_name,
    )

    summary_path = output_dir / "prediction_absorption_summary.csv"
    sample_path = output_dir / "prediction_absorption_samples.csv"
    manifest_path = output_dir / "analysis_prediction_manifest.json"
    _write_csv(summary_path, summary_rows, _summary_columns())
    output_paths = {
        "prediction_absorption_summary_csv": _path_str(summary_path),
        "manifest_json": _path_str(manifest_path),
    }
    if not bool(args.no_sample_rows):
        _write_csv(sample_path, sample_rows, _sample_columns())
        output_paths["prediction_absorption_samples_csv"] = _path_str(sample_path)

    manifest_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_name": dataset_name,
        "seed": int(args.seed),
        "train_csv_sha256": _file_sha256(train_csv),
        "eval_csv_sha256": _file_sha256(eval_csv),
        "label_mapping_sha256": _file_sha256(label_mapping_path),
        "prediction_sources": {
            prefix: _file_sha256(path)
            for prefix, path in prediction_paths.items()
        },
    }
    manifest = {
        **manifest_identity,
        "analysis_name": ANALYSIS_NAME,
        "artifact_type": "prediction_absorption_analysis",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_identity_hash": _json_hash(manifest_identity),
        "dataset_root": _path_str(dataset_root),
        "output_dir": _path_str(output_dir),
        "outputs": dict(output_paths),
        "split_protocol": {
            "analysis_split": "eval",
            "train_csv_for_bucket_counts": _path_str(train_csv),
            "train_label_count_column_used": label_count_column,
            "eval_csv": _path_str(eval_csv),
            "eval_sample_count": len(eval_identity),
            "eval_sample_order_hash": _sample_order_hash(sample_id for sample_id, _label_id, _bucket in eval_identity),
            "label_mapping_path": _path_str(label_mapping_path),
            "bucket_source": "train split class count via src.utils.group_utils.bucket_from_train_label_n",
            "test_split_touched": False,
        },
        "prediction_sources": _prediction_source_manifest(paths=prediction_paths, predictions=predictions),
        "group_protocol": group_protocol_metadata(),
        "absorption_protocol": {
            "true_class_scope": "bucket in {medium, few}; tail_included is derived from bucket in {medium, few}",
            "absorbed_many_definition": "prediction != label_id and predicted label bucket == many",
            "predicted_bucket_source": "train_csv train_label_n/label_n-derived label_id bucket map",
            "pair_key": ["source_true_bucket", "true_label_id", "absorbed_many_label_id"],
            "summary_pair_source": "union of absorption pairs observed in all four eval prediction tables",
            "margin_delta_definition": "current correct_logit_margin - fp32_teacher correct_logit_margin on the same absorbed samples",
            "sample_rows_written": not bool(args.no_sample_rows),
        },
        "consistency_checks": {
            "four_prediction_tables_same_sample_id_label_id_bucket_sequence": True,
            "eval_csv_matches_prediction_sample_id_label_id_bucket_sequence": True,
            "prediction_bucket_matches_train_count_bucket": True,
            "tail_included_matches_bucket": True,
            "correct_column_matches_prediction_equals_label_id": True,
        },
        "counts": {
            "summary_rows": int(len(summary_rows)),
            "sample_rows": int(len(sample_rows)),
            "absorption_sample_counts_by_model": _build_absorption_counts(predictions, label_buckets=label_buckets),
            "label_bucket_counts": Counter(str(value) for value in label_buckets.values()),
            "train_label_count_by_label_id": {str(key): int(value) for key, value in sorted(train_label_counts.items())},
        },
        "artifact_sha256": {
            "prediction_absorption_summary.csv": _file_sha256(summary_path),
            "prediction_absorption_samples.csv": _file_sha256(sample_path) if not bool(args.no_sample_rows) else None,
        },
    }
    manifest["run_manifest_hash"] = _json_hash(manifest)
    _json_dump(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "outputs": output_paths,
        "summary_rows": len(summary_rows),
        "sample_rows": len(sample_rows),
        "manifest": manifest,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_analysis_prediction(args)
    print(f"[analysis-prediction] output_dir={result['output_dir']}")
    print(f"[analysis-prediction] summary_rows={result['summary_rows']}")
    print(f"[analysis-prediction] sample_rows={result['sample_rows']}")
    for name, path in result["outputs"].items():
        print(f"[analysis-prediction] {name}={path}")
    return result


if __name__ == "__main__":
    main()
