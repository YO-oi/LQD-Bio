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

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.utils.group_utils import bucket_from_train_label_n, group_protocol_metadata


PROTOCOL_VERSION = "residual_tail_many_competition.v1"
ANALYSIS_NAME = "Residual Tail-to-Many Competition Analysis"
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
        default_relative_path=("baseline", "teacher", "seed_{seed}", "logits", "fp32_teacher", "eval_logits.pt"),
    ),
    ModelSpec(
        prefix="int4_init",
        display_name="INT4 init",
        default_relative_path=("baseline", "student", "seed_{seed}", "logits", "int4_student", "eval_logits.pt"),
    ),
    ModelSpec(
        prefix="vanilla_kd",
        display_name="Vanilla KD best",
        default_relative_path=("baseline", "kd", "seed_{seed}", "logits", "vanilla_kd", "eval_logits.pt"),
    ),
)


SUMMARY_BASE_COLUMNS = [
    "dataset_name",
    "seed",
    "scope",
    "teacher_correct_filter",
    "n_tail_samples",
    "n_teacher_correct",
    "n_teacher_wrong",
]

PER_MODEL_SUMMARY_COLUMNS = [
    "mean_P_M",
    "mean_P_T",
    "mean_C",
    "median_C",
    "c_gt_0_rate",
    "c_gt_0_count",
]

DELTA_SUMMARY_COLUMNS = [
    "delta_C_INT4_mean",
    "delta_C_INT4_median",
    "delta_C_INT4_gt_0_rate",
    "delta_C_INT4_gt_0_count",
    "delta_C_KD_mean",
    "delta_C_KD_median",
    "delta_C_KD_lt_0_rate",
    "delta_C_KD_lt_0_count",
    "residual_C_KD_minus_FP32_mean",
    "residual_C_KD_minus_FP32_median",
    "residual_C_KD_minus_FP32_gt_0_rate",
    "residual_C_KD_minus_FP32_gt_0_count",
    "int4_increases_C_by_mean",
    "kd_pulls_back_by_mean",
    "residual_many_attraction_by_mean",
]


SAMPLE_BASE_COLUMNS = [
    "dataset_name",
    "seed",
    "sample_index",
    "sample_id",
    "true_label_id",
    "true_label_name",
    "true_bucket",
    "teacher_correct",
]

PER_MODEL_SAMPLE_COLUMNS = [
    "P_M",
    "P_T",
    "C",
    "prediction",
    "prediction_label_name",
    "prediction_bucket",
    "correct",
]

SAMPLE_DELTA_COLUMNS = [
    "delta_C_INT4",
    "delta_C_KD",
    "residual_C_KD_minus_FP32",
]


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


def _dataset_root(entry: Mapping[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    if "ROOT" not in entry:
        raise KeyError("dataset config missing ROOT")
    return Path(entry["ROOT"])


def _path_from_dataset_root(
    entry: Mapping[str, Any],
    key: str,
    *,
    dataset_root: Path,
    dataset_root_overridden: bool,
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return Path(override)
    if dataset_root_overridden:
        filename_by_key = {
            "TRAIN_CSV": "train.csv",
            "EVAL_CSV": "eval.csv",
            "LABEL_MAPPING": "label_mapping.json",
        }
        if key not in filename_by_key:
            raise KeyError(f"no dataset-root fallback for key={key!r}")
        return Path(dataset_root) / "splits" / filename_by_key[key]
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


def _select_label_count_column(fieldnames: Sequence[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise KeyError(f"requested train label count column {explicit!r} is missing")
        return str(explicit)
    if "train_label_n" in fieldnames:
        return "train_label_n"
    if "label_n" in fieldnames:
        return "label_n"
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
        previous_count = count_by_label.get(label_id)
        if previous_count is not None and previous_count != train_count:
            raise ValueError(
                f"{train_csv}: inconsistent {count_column} for label_id={label_id}: "
                f"{previous_count!r} vs {train_count!r}"
            )
        previous_bucket = bucket_by_label.get(label_id)
        expected_bucket = bucket_from_train_label_n(train_count)
        if previous_bucket is not None and previous_bucket != expected_bucket:
            raise ValueError(
                f"{train_csv}: inconsistent derived bucket for label_id={label_id}: "
                f"{previous_bucket!r} vs {expected_bucket!r}"
            )
        count_by_label[label_id] = train_count
        bucket_by_label[label_id] = expected_bucket
    if not bucket_by_label:
        raise RuntimeError(f"train CSV has no label bucket rows: {train_csv}")
    return bucket_by_label, count_column, count_by_label


def _assert_label_mapping_matches_logit_columns(id_to_name: Mapping[int, str], *, path: Path) -> None:
    label_ids = sorted(int(label_id) for label_id in id_to_name)
    expected = list(range(len(label_ids)))
    if label_ids != expected:
        raise ValueError(
            f"{path}: label IDs must be contiguous logits column indices 0..{len(label_ids) - 1}; "
            f"got first_ids={label_ids[:8]}"
        )


def _assert_bucket_map_covers_label_mapping(
    label_buckets: Mapping[int, str],
    id_to_name: Mapping[int, str],
    *,
    train_csv: Path,
    label_mapping_path: Path,
) -> None:
    missing = sorted(int(label_id) for label_id in id_to_name if int(label_id) not in label_buckets)
    if missing:
        raise RuntimeError(
            f"{train_csv}: train-derived bucket map is missing {len(missing)} label_mapping classes from "
            f"{label_mapping_path}; missing_label_ids={missing[:16]}"
        )


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


def _sample_order_hash(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_torch_payload(path: str | Path) -> Mapping[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"logits artifact not found: {target}")
    try:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(target, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"logits artifact must be a mapping, got {type(payload)!r}: {target}")
    return payload


def _payload_sample_ids(payload: Mapping[str, Any], *, path: Path) -> list[str]:
    value = payload.get("sample_ids", payload.get("sample_id"))
    if value is None:
        raise KeyError(f"logits payload missing sample_ids/sample_id: {path}")
    return [str(item) for item in list(value)]


def _payload_labels(payload: Mapping[str, Any], *, path: Path) -> torch.Tensor:
    value = payload.get("labels", payload.get("label_id"))
    if value is None:
        raise KeyError(f"logits payload missing labels/label_id: {path}")
    labels = torch.as_tensor(value, dtype=torch.long).cpu().reshape(-1)
    if labels.numel() <= 0:
        raise ValueError(f"logits payload has no labels: {path}")
    return labels


def _payload_logits(payload: Mapping[str, Any], *, path: Path) -> torch.Tensor:
    value = payload.get("logits")
    if value is None:
        raise KeyError(f"logits payload missing logits tensor: {path}")
    logits = torch.as_tensor(value, dtype=torch.float32).cpu()
    if logits.ndim != 2:
        raise ValueError(f"logits must be [N,C], got shape={tuple(logits.shape)}: {path}")
    if int(logits.shape[0]) <= 0 or int(logits.shape[1]) <= 1:
        raise ValueError(f"logits must have positive N and at least 2 classes, got shape={tuple(logits.shape)}: {path}")
    if not torch.isfinite(logits).all().item():
        raise ValueError(f"logits contain NaN/Inf: {path}")
    return logits.contiguous()


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


@dataclass(frozen=True)
class LogitBundle:
    spec: ModelSpec
    path: Path
    logits: torch.Tensor
    sample_ids: tuple[str, ...]
    labels: torch.Tensor
    sample_order_hash: str
    split: str
    checkpoint_sha256: str
    payload_sha256: str | None


def _load_logit_bundle(
    spec: ModelSpec,
    path: Path,
    *,
    eval_identity: Sequence[tuple[str, int, str]],
    label_buckets: Mapping[int, str],
    expected_num_classes: int,
) -> LogitBundle:
    payload = _load_torch_payload(path)
    split = str(payload.get("split", ""))
    if split and split != "eval":
        raise ValueError(f"{spec.prefix} logits split must be eval, got {split!r}: {path}")
    sample_ids = tuple(_payload_sample_ids(payload, path=path))
    labels = _payload_labels(payload, path=path)
    logits = _payload_logits(payload, path=path)
    if int(logits.shape[0]) != len(sample_ids) or int(labels.numel()) != len(sample_ids):
        raise ValueError(
            f"{spec.prefix} logits/sample_ids/labels length mismatch: "
            f"logits={tuple(logits.shape)}, sample_ids={len(sample_ids)}, labels={int(labels.numel())}"
        )
    if int(logits.shape[1]) != int(expected_num_classes):
        raise ValueError(
            f"{spec.prefix} logit dimension={int(logits.shape[1])} differs from label_mapping classes={int(expected_num_classes)}"
        )
    identity = []
    for index, sample_id in enumerate(sample_ids):
        label_id = int(labels[index].item())
        if label_id not in label_buckets:
            raise KeyError(f"{spec.prefix} logits row={index} label_id={label_id} absent from train bucket map")
        identity.append((str(sample_id), label_id, str(label_buckets[label_id])))
    if list(eval_identity) != identity:
        mismatch = _first_mismatch(list(eval_identity), identity)
        raise RuntimeError(f"eval.csv and {spec.prefix} logits identity/order differ: {mismatch}")
    return LogitBundle(
        spec=spec,
        path=path,
        logits=logits,
        sample_ids=sample_ids,
        labels=labels,
        sample_order_hash=_sample_order_hash(sample_ids),
        split=split or "eval",
        checkpoint_sha256=str(payload.get("checkpoint_sha256") or ""),
        payload_sha256=_file_sha256(path),
    )


def _default_logit_path(dataset_root: Path, seed: int, spec: ModelSpec) -> Path:
    parts = [part.format(seed=int(seed)) for part in spec.default_relative_path]
    return dataset_root.joinpath(*parts)


def _logit_paths(args: argparse.Namespace, dataset_root: Path) -> dict[str, Path]:
    overrides = {
        "fp32_teacher": args.fp32_logits,
        "int4_init": args.int4_logits,
        "vanilla_kd": args.vanilla_kd_logits,
    }
    return {
        spec.prefix: Path(overrides[spec.prefix])
        if overrides[spec.prefix]
        else _default_logit_path(dataset_root, int(args.seed), spec)
        for spec in MODEL_SPECS
    }


def _output_dir(dataset_root: Path, seed: int, override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    return dataset_root / "export" / "residual_tail_many_competition" / f"seed_{int(seed)}"


def _label_name(label_id: int, id_to_name: Mapping[int, str]) -> str:
    return str(id_to_name.get(int(label_id), str(label_id)))


def _finite_values(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean(values: Sequence[float]) -> float:
    finite = _finite_values(values)
    if not finite:
        return float("nan")
    return float(sum(finite) / float(len(finite)))


def _median(values: Sequence[float]) -> float:
    finite = sorted(_finite_values(values))
    if not finite:
        return float("nan")
    middle = len(finite) // 2
    if len(finite) % 2:
        return float(finite[middle])
    return float((finite[middle - 1] + finite[middle]) / 2.0)


def _rate(values: Sequence[bool]) -> float:
    if not values:
        return float("nan")
    return float(sum(1 for value in values if bool(value)) / float(len(values)))


@dataclass(frozen=True)
class CompetitionBundle:
    p_many: torch.Tensor
    p_tail: torch.Tensor
    c_score: torch.Tensor
    predictions: torch.Tensor
    correct: torch.Tensor


def _competition_bundle(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    many_label_ids: Sequence[int],
    tail_label_ids: Sequence[int],
) -> CompetitionBundle:
    if not many_label_ids:
        raise RuntimeError("many class set is empty; cannot compute P_M")
    if not tail_label_ids:
        raise RuntimeError("tail class set is empty; cannot compute P_T")
    probs = torch.softmax(logits.float(), dim=1)
    many_index = torch.as_tensor(list(many_label_ids), dtype=torch.long)
    tail_index = torch.as_tensor(list(tail_label_ids), dtype=torch.long)
    p_many = probs.index_select(1, many_index).sum(dim=1).cpu().float()
    p_tail = probs.index_select(1, tail_index).sum(dim=1).cpu().float()
    predictions = logits.argmax(dim=1).cpu().long()
    labels = labels.cpu().long().reshape(-1)
    return CompetitionBundle(
        p_many=p_many,
        p_tail=p_tail,
        c_score=(p_many - p_tail).cpu().float(),
        predictions=predictions,
        correct=(predictions == labels).cpu().bool(),
    )


def _tensor_values(tensor: torch.Tensor, indices: Sequence[int]) -> list[float]:
    return [float(tensor[int(index)].item()) for index in indices]


def _tensor_bool_values(tensor: torch.Tensor, indices: Sequence[int]) -> list[bool]:
    return [bool(tensor[int(index)].item()) for index in indices]


def _summarize_competition(bundle: CompetitionBundle, indices: Sequence[int]) -> dict[str, Any]:
    c_values = _tensor_values(bundle.c_score, indices)
    return {
        "mean_P_M": _mean(_tensor_values(bundle.p_many, indices)),
        "mean_P_T": _mean(_tensor_values(bundle.p_tail, indices)),
        "mean_C": _mean(c_values),
        "median_C": _median(c_values),
        "c_gt_0_rate": _rate([value > 0.0 for value in c_values]),
        "c_gt_0_count": int(sum(1 for value in c_values if value > 0.0)),
    }


def _delta_values(left: torch.Tensor, right: torch.Tensor, indices: Sequence[int]) -> list[float]:
    return [float((left[int(index)] - right[int(index)]).item()) for index in indices]


def _summarize_delta(values: Sequence[float], *, positive_direction: str) -> dict[str, Any]:
    if positive_direction == "gt_zero":
        flags = [value > 0.0 for value in values]
    elif positive_direction == "lt_zero":
        flags = [value < 0.0 for value in values]
    else:
        raise ValueError(f"unknown positive_direction={positive_direction!r}")
    return {
        "mean": _mean(values),
        "median": _median(values),
        "rate": _rate(flags),
        "count": int(sum(1 for flag in flags if bool(flag))),
    }


def _summary_columns() -> list[str]:
    columns = list(SUMMARY_BASE_COLUMNS)
    for spec in MODEL_SPECS:
        columns.extend(f"{spec.prefix}_{column}" for column in PER_MODEL_SUMMARY_COLUMNS)
    columns.extend(DELTA_SUMMARY_COLUMNS)
    return columns


def _sample_columns() -> list[str]:
    columns = list(SAMPLE_BASE_COLUMNS)
    for spec in MODEL_SPECS:
        columns.extend(f"{spec.prefix}_{column}" for column in PER_MODEL_SAMPLE_COLUMNS)
    columns.extend(SAMPLE_DELTA_COLUMNS)
    return columns


def _build_summary_rows(
    *,
    dataset_name: str,
    seed: int,
    labels: torch.Tensor,
    label_buckets: Mapping[int, str],
    competition: Mapping[str, CompetitionBundle],
) -> list[dict[str, Any]]:
    true_tail_indices = [
        int(index)
        for index, label in enumerate(labels.cpu().long().tolist())
        if str(label_buckets[int(label)]) in TAIL_BUCKETS
    ]
    if not true_tail_indices:
        raise RuntimeError("eval split contains no true tail samples under train_label_n-derived buckets")
    teacher_correct = competition["fp32_teacher"].correct
    scopes = [
        ("tail_all", "all", true_tail_indices),
        ("teacher_correct", "teacher_correct", [index for index in true_tail_indices if bool(teacher_correct[index].item())]),
        ("teacher_wrong", "teacher_wrong", [index for index in true_tail_indices if not bool(teacher_correct[index].item())]),
    ]
    rows: list[dict[str, Any]] = []
    fp32_c = competition["fp32_teacher"].c_score
    int4_c = competition["int4_init"].c_score
    kd_c = competition["vanilla_kd"].c_score
    for scope, teacher_filter, indices in scopes:
        row: dict[str, Any] = {
            "dataset_name": str(dataset_name),
            "seed": int(seed),
            "scope": str(scope),
            "teacher_correct_filter": str(teacher_filter),
            "n_tail_samples": int(len(indices)),
            "n_teacher_correct": int(sum(1 for index in indices if bool(teacher_correct[index].item()))),
            "n_teacher_wrong": int(sum(1 for index in indices if not bool(teacher_correct[index].item()))),
        }
        for spec in MODEL_SPECS:
            values = _summarize_competition(competition[spec.prefix], indices)
            for key, value in values.items():
                row[f"{spec.prefix}_{key}"] = value
        delta_int4 = _summarize_delta(_delta_values(int4_c, fp32_c, indices), positive_direction="gt_zero")
        delta_kd = _summarize_delta(_delta_values(kd_c, int4_c, indices), positive_direction="lt_zero")
        residual = _summarize_delta(_delta_values(kd_c, fp32_c, indices), positive_direction="gt_zero")
        row.update(
            {
                "delta_C_INT4_mean": delta_int4["mean"],
                "delta_C_INT4_median": delta_int4["median"],
                "delta_C_INT4_gt_0_rate": delta_int4["rate"],
                "delta_C_INT4_gt_0_count": delta_int4["count"],
                "delta_C_KD_mean": delta_kd["mean"],
                "delta_C_KD_median": delta_kd["median"],
                "delta_C_KD_lt_0_rate": delta_kd["rate"],
                "delta_C_KD_lt_0_count": delta_kd["count"],
                "residual_C_KD_minus_FP32_mean": residual["mean"],
                "residual_C_KD_minus_FP32_median": residual["median"],
                "residual_C_KD_minus_FP32_gt_0_rate": residual["rate"],
                "residual_C_KD_minus_FP32_gt_0_count": residual["count"],
                "int4_increases_C_by_mean": bool(math.isfinite(delta_int4["mean"]) and float(delta_int4["mean"]) > 0.0),
                "kd_pulls_back_by_mean": bool(math.isfinite(delta_kd["mean"]) and float(delta_kd["mean"]) < 0.0),
                "residual_many_attraction_by_mean": bool(math.isfinite(residual["mean"]) and float(residual["mean"]) > 0.0),
            }
        )
        rows.append(row)
    return rows


def _build_sample_rows(
    *,
    dataset_name: str,
    seed: int,
    bundles: Mapping[str, LogitBundle],
    label_buckets: Mapping[int, str],
    id_to_name: Mapping[int, str],
    competition: Mapping[str, CompetitionBundle],
) -> list[dict[str, Any]]:
    fp32 = bundles["fp32_teacher"]
    labels = fp32.labels.cpu().long().reshape(-1)
    true_tail_indices = [
        int(index)
        for index, label in enumerate(labels.tolist())
        if str(label_buckets[int(label)]) in TAIL_BUCKETS
    ]
    rows: list[dict[str, Any]] = []
    for index in true_tail_indices:
        label_id = int(labels[index].item())
        teacher_correct = bool(competition["fp32_teacher"].correct[index].item())
        row: dict[str, Any] = {
            "dataset_name": str(dataset_name),
            "seed": int(seed),
            "sample_index": int(index),
            "sample_id": str(fp32.sample_ids[index]),
            "true_label_id": label_id,
            "true_label_name": _label_name(label_id, id_to_name),
            "true_bucket": str(label_buckets[label_id]),
            "teacher_correct": teacher_correct,
        }
        for spec in MODEL_SPECS:
            comp = competition[spec.prefix]
            pred = int(comp.predictions[index].item())
            row[f"{spec.prefix}_P_M"] = float(comp.p_many[index].item())
            row[f"{spec.prefix}_P_T"] = float(comp.p_tail[index].item())
            row[f"{spec.prefix}_C"] = float(comp.c_score[index].item())
            row[f"{spec.prefix}_prediction"] = pred
            row[f"{spec.prefix}_prediction_label_name"] = _label_name(pred, id_to_name)
            row[f"{spec.prefix}_prediction_bucket"] = str(label_buckets[pred])
            row[f"{spec.prefix}_correct"] = bool(comp.correct[index].item())
        row["delta_C_INT4"] = float((competition["int4_init"].c_score[index] - competition["fp32_teacher"].c_score[index]).item())
        row["delta_C_KD"] = float((competition["vanilla_kd"].c_score[index] - competition["int4_init"].c_score[index]).item())
        row["residual_C_KD_minus_FP32"] = float((competition["vanilla_kd"].c_score[index] - competition["fp32_teacher"].c_score[index]).item())
        rows.append(row)
    return rows


def _source_manifest(paths: Mapping[str, Path], bundles: Mapping[str, LogitBundle]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in MODEL_SPECS:
        bundle = bundles[spec.prefix]
        result[spec.prefix] = {
            "display_name": spec.display_name,
            "path": _path_str(paths[spec.prefix]),
            "sha256": bundle.payload_sha256,
            "split": bundle.split,
            "row_count": int(bundle.logits.shape[0]),
            "num_classes": int(bundle.logits.shape[1]),
            "sample_order_hash": bundle.sample_order_hash,
            "checkpoint_sha256": bundle.checkpoint_sha256,
        }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=ANALYSIS_NAME)
    parser.add_argument("--dataset", default=None, help="Default: config.ACTIVE_DATASET.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--label-mapping-path", default=None)
    parser.add_argument("--train-label-count-column", default=None)
    parser.add_argument("--fp32-logits", default=None)
    parser.add_argument("--int4-logits", default=None)
    parser.add_argument("--vanilla-kd-logits", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-sample-rows", action="store_true", help="Only write the summary table and manifest.")
    return parser.parse_args(argv)


def run_residual_tail_many_competition(args: argparse.Namespace) -> dict[str, Any]:
    dataset_name, entry = _dataset_entry(args.dataset)
    dataset_root_overridden = args.dataset_root is not None
    dataset_root = _dataset_root(entry, args.dataset_root)
    train_csv = _path_from_dataset_root(
        entry,
        "TRAIN_CSV",
        dataset_root=dataset_root,
        dataset_root_overridden=dataset_root_overridden,
        override=args.train_csv,
    )
    eval_csv = _path_from_dataset_root(
        entry,
        "EVAL_CSV",
        dataset_root=dataset_root,
        dataset_root_overridden=dataset_root_overridden,
        override=args.eval_csv,
    )
    label_mapping_path = _path_from_dataset_root(
        entry,
        "LABEL_MAPPING",
        dataset_root=dataset_root,
        dataset_root_overridden=dataset_root_overridden,
        override=args.label_mapping_path,
    )
    output_dir = _output_dir(dataset_root, int(args.seed), args.output_dir)
    logit_paths = _logit_paths(args, dataset_root)

    label_name_to_id, id_to_name = _load_label_mapping(label_mapping_path)
    _assert_label_mapping_matches_logit_columns(id_to_name, path=label_mapping_path)
    label_buckets, label_count_column, train_label_counts = _load_train_label_buckets(
        train_csv,
        label_name_to_id=label_name_to_id,
        label_count_column=args.train_label_count_column,
    )
    _assert_bucket_map_covers_label_mapping(
        label_buckets,
        id_to_name,
        train_csv=train_csv,
        label_mapping_path=label_mapping_path,
    )
    eval_identity = _read_eval_identity(eval_csv, label_name_to_id=label_name_to_id, label_buckets=label_buckets)
    expected_num_classes = len(id_to_name)
    bundles = {
        spec.prefix: _load_logit_bundle(
            spec,
            logit_paths[spec.prefix],
            eval_identity=eval_identity,
            label_buckets=label_buckets,
            expected_num_classes=expected_num_classes,
        )
        for spec in MODEL_SPECS
    }

    labels = bundles["fp32_teacher"].labels.cpu().long()
    many_label_ids = sorted(int(label_id) for label_id, bucket in label_buckets.items() if str(bucket) == MANY_BUCKET)
    tail_label_ids = sorted(int(label_id) for label_id, bucket in label_buckets.items() if str(bucket) in TAIL_BUCKETS)
    competition = {
        spec.prefix: _competition_bundle(
            bundles[spec.prefix].logits,
            bundles[spec.prefix].labels,
            many_label_ids=many_label_ids,
            tail_label_ids=tail_label_ids,
        )
        for spec in MODEL_SPECS
    }

    summary_rows = _build_summary_rows(
        dataset_name=dataset_name,
        seed=int(args.seed),
        labels=labels,
        label_buckets=label_buckets,
        competition=competition,
    )
    sample_rows = [] if bool(args.no_sample_rows) else _build_sample_rows(
        dataset_name=dataset_name,
        seed=int(args.seed),
        bundles=bundles,
        label_buckets=label_buckets,
        id_to_name=id_to_name,
        competition=competition,
    )

    summary_path = output_dir / "tail_many_competition_summary.csv"
    sample_path = output_dir / "tail_many_competition_samples.csv"
    manifest_path = output_dir / "residual_tail_many_competition_manifest.json"
    _write_csv(summary_path, summary_rows, _summary_columns())
    outputs = {
        "tail_many_competition_summary_csv": _path_str(summary_path),
        "manifest_json": _path_str(manifest_path),
    }
    if not bool(args.no_sample_rows):
        _write_csv(sample_path, sample_rows, _sample_columns())
        outputs["tail_many_competition_samples_csv"] = _path_str(sample_path)

    tail_eval_count = int(len(sample_rows)) if not bool(args.no_sample_rows) else int(summary_rows[0]["n_tail_samples"])
    manifest_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_name": dataset_name,
        "seed": int(args.seed),
        "train_csv_sha256": _file_sha256(train_csv),
        "eval_csv_sha256": _file_sha256(eval_csv),
        "label_mapping_sha256": _file_sha256(label_mapping_path),
        "logit_sources": {prefix: _file_sha256(path) for prefix, path in logit_paths.items()},
    }
    manifest = {
        **manifest_identity,
        "analysis_name": ANALYSIS_NAME,
        "artifact_type": "residual_tail_many_competition_analysis",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_identity_hash": _json_hash(manifest_identity),
        "dataset_root": _path_str(dataset_root),
        "output_dir": _path_str(output_dir),
        "outputs": dict(outputs),
        "split_protocol": {
            "analysis_split": "eval",
            "train_csv_for_bucket_counts": _path_str(train_csv),
            "train_label_count_column_used": label_count_column,
            "eval_csv": _path_str(eval_csv),
            "eval_sample_count": len(eval_identity),
            "eval_sample_order_hash": _sample_order_hash(sample_id for sample_id, _label_id, _bucket in eval_identity),
            "label_mapping_path": _path_str(label_mapping_path),
            "bucket_source": "train split class-count column only",
            "test_split_touched": False,
        },
        "competition_protocol": {
            "true_sample_scope": "eval samples whose train_label_n-derived true bucket is medium or few",
            "bucket_thresholds": {
                "many": "train_label_n >= 100",
                "medium": "20 <= train_label_n <= 99",
                "few": "train_label_n <= 19",
                "tail": "medium + few",
            },
            "probability_source": "torch.softmax over complete eval logits for every class",
            "P_M_definition": "sum softmax probability over all many classes",
            "P_T_definition": "sum softmax probability over all medium/few classes",
            "C_definition": "P_M - P_T",
            "delta_C_INT4": "C_INT4 - C_FP32",
            "delta_C_KD": "C_KD - C_INT4",
            "residual": "C_KD - C_FP32",
            "teacher_groups": ["tail_all", "teacher_correct", "teacher_wrong"],
            "top1_prediction_not_used_for_probability_mass": True,
            "label_id_logit_column_alignment": "required: label_id == logits class index",
        },
        "logit_sources": _source_manifest(logit_paths, bundles),
        "group_protocol": group_protocol_metadata(),
        "consistency_checks": {
            "three_logit_payloads_same_sample_id_label_id_bucket_sequence_as_eval_csv": True,
            "all_logit_payloads_split_eval": True,
            "all_logit_payloads_complete_class_logits": True,
            "bucket_mapping_uses_train_split_class_count_only": True,
            "bucket_map_covers_every_label_mapping_class": True,
            "label_mapping_ids_are_contiguous_logit_columns": True,
            "test_split_not_read": True,
        },
        "counts": {
            "tail_eval_sample_count": tail_eval_count,
            "many_class_count": int(len(many_label_ids)),
            "tail_class_count": int(len(tail_label_ids)),
            "medium_class_count": int(sum(1 for bucket in label_buckets.values() if str(bucket) == "medium")),
            "few_class_count": int(sum(1 for bucket in label_buckets.values() if str(bucket) == "few")),
            "summary_rows": int(len(summary_rows)),
            "sample_rows": int(len(sample_rows)),
            "train_label_count_by_label_id": {str(key): int(value) for key, value in sorted(train_label_counts.items())},
        },
        "artifact_sha256": {
            "tail_many_competition_summary.csv": _file_sha256(summary_path),
            "tail_many_competition_samples.csv": _file_sha256(sample_path) if not bool(args.no_sample_rows) else None,
        },
    }
    manifest["run_manifest_hash"] = _json_hash(manifest)
    _json_dump(manifest_path, manifest)
    return {
        "output_dir": output_dir,
        "outputs": outputs,
        "summary_rows": len(summary_rows),
        "sample_rows": len(sample_rows),
        "manifest": manifest,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_residual_tail_many_competition(args)
    print(f"[tail-many-competition] output_dir={result['output_dir']}")
    print(f"[tail-many-competition] summary_rows={result['summary_rows']}")
    print(f"[tail-many-competition] sample_rows={result['sample_rows']}")
    for name, path in result["outputs"].items():
        print(f"[tail-many-competition] {name}={path}")
    return result


if __name__ == "__main__":
    main()
