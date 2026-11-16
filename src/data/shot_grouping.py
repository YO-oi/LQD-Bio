from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

try:
    from src.utils.group_utils import (
        BASE_BUCKETS,
        GroupValidationError,
        bucket_from_train_label_n,
        normalize_bucket,
        read_canonical_bucket,
    )
except ImportError:  # pragma: no cover
    from ..utils.group_utils import (
        BASE_BUCKETS,
        GroupValidationError,
        bucket_from_train_label_n,
        normalize_bucket,
        read_canonical_bucket,
    )


DEFAULT_GROUP_ORDER = BASE_BUCKETS
GROUP_COLUMNS = ("bucket", "shot_group")
COUNT_COLUMNS = ("train_label_n", "label_n", "label_count", "count")
LABEL_ID_COLUMNS = ("label_id", "class_id", "target")
SAMPLE_ID_COLUMNS = ("sample_id", "audio_id", "file_id")


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}


def group_order(cfg: Any | None = None) -> tuple[str, ...]:
    del cfg
    return DEFAULT_GROUP_ORDER


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "<na>"}


def normalize_group_name(value: Any, cfg: Any | None = None) -> str | None:
    del cfg
    if _is_missing(value):
        return None
    return normalize_bucket(value, field_name="bucket")


def parse_optional_int(value: Any) -> int | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = int(value)
    elif isinstance(value, float):
        if not (math.isfinite(value) and value.is_integer()):
            return None
        parsed = int(value)
    else:
        text = str(value).strip()
        try:
            numeric = float(text)
        except ValueError:
            return None
        if not math.isfinite(numeric) or not numeric.is_integer():
            return None
        parsed = int(numeric)
    if parsed < 0:
        raise GroupValidationError(f"train_label_n={value!r} must be non-negative")
    return parsed


def group_from_train_count(count: int, cfg: Any | None = None) -> str:
    del cfg
    return bucket_from_train_label_n(count)


def label_id_from_row(row: Mapping[str, Any]) -> int | None:
    for key in LABEL_ID_COLUMNS:
        parsed = parse_optional_int(row.get(key))
        if parsed is not None:
            return parsed
    return None


def sample_id_from_row(row: Mapping[str, Any]) -> str | None:
    for key in SAMPLE_ID_COLUMNS:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _optional_sequence_value(row: Mapping[str, Any], key: str) -> list[Any] | None:
    if key not in row or _is_missing(row.get(key)):
        return None
    return [row.get(key)]


def group_name_from_row(row: Mapping[str, Any], cfg: Any | None = None) -> str | None:
    del cfg
    bucket = _optional_sequence_value(row, "bucket")
    shot_group = _optional_sequence_value(row, "shot_group")
    train_label_n = _optional_sequence_value(row, "train_label_n")
    if bucket is None:
        if shot_group is not None:
            raise GroupValidationError("sample-level shot_group requires an explicit bucket column")
        return None
    result = read_canonical_bucket(
        bucket=bucket,
        shot_group=shot_group,
        train_label_n=train_label_n,
        context="metadata row",
        allow_shot_group_fallback=False,
    )
    return result.bucket[0]


def split_csv_path(cfg: Any, name: str) -> Path | None:
    key = f"{name.upper()}_CSV"
    for source in (cfg_get(cfg, "FFN_LAYER_HEATMAP", {}), cfg_get(cfg, "FFN_FP32_REFERENCE_CACHE", {}), cfg):
        if isinstance(source, Mapping) and source.get(key) is not None:
            return Path(source[key])
        if not isinstance(source, Mapping):
            value = cfg_get(source, key, None)
            if value is not None:
                return Path(value)
    dataset = dataset_cfg(cfg)
    value = nested_get(dataset, key, None)
    return Path(value) if value is not None else None


def full_dataset_csv_path(cfg: Any) -> Path | None:
    for key in ("FULL_CSV", "FULL_DATASET_CSV"):
        for source in (cfg_get(cfg, "FFN_LAYER_HEATMAP", {}), cfg_get(cfg, "FFN_FP32_REFERENCE_CACHE", {}), cfg):
            if isinstance(source, Mapping) and source.get(key) is not None:
                return Path(source[key])
            if not isinstance(source, Mapping):
                value = cfg_get(source, key, None)
                if value is not None:
                    return Path(value)
        dataset = dataset_cfg(cfg)
        value = nested_get(dataset, key, None)
        if value is not None:
            return Path(value)
    split_dir = nested_get(dataset_cfg(cfg), "SPLIT_DIR", None)
    return Path(split_dir) / "full_dataset.csv" if split_dir is not None else None


def metadata_csv_paths(cfg: Any, *, split_names: Sequence[str] = ("eval",), include_full: bool = True) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for name in split_names:
        path = split_csv_path(cfg, name)
        if path is not None:
            key = str(Path(path))
            if key not in seen:
                paths.append(Path(path))
                seen.add(key)
    if include_full:
        path = full_dataset_csv_path(cfg)
        if path is not None:
            key = str(Path(path))
            if key not in seen:
                paths.append(Path(path))
                seen.add(key)
    return paths


def _set_once(mapping: dict[Any, str], key: Any, value: str, *, context: str) -> None:
    previous = mapping.get(key)
    if previous is not None and previous != value:
        raise GroupValidationError(f"{context} has conflicting bucket values: {previous!r} vs {value!r}")
    mapping[key] = value


def _read_group_tables(
    paths: Sequence[Path],
    cfg: Any,
    *,
    allowed_labels: set[int] | None = None,
) -> tuple[dict[str, str], dict[int, str]]:
    del cfg
    sample_to_group: dict[str, str] = {}
    label_to_group: dict[int, str] = {}

    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader, start=2):
                label_id = label_id_from_row(row)
                if allowed_labels is not None and label_id is not None and int(label_id) not in allowed_labels:
                    continue
                group = group_name_from_row(row, None)
                if group is None:
                    continue
                sample_id = sample_id_from_row(row)
                if sample_id is not None:
                    _set_once(sample_to_group, sample_id, group, context=f"{path}:{row_idx} sample_id={sample_id!r}")
                if label_id is not None:
                    _set_once(label_to_group, int(label_id), group, context=f"{path}:{row_idx} label_id={label_id!r}")

    return sample_to_group, label_to_group


def label_groups_from_metadata(
    labels: Any,
    cfg: Any,
    *,
    required_groups: Sequence[str] = ("many", "few"),
    paths: Sequence[Path] | None = None,
    strict: bool = True,
) -> dict[str, Tensor]:
    label_tensor = torch.as_tensor(labels).detach().cpu().long().reshape(-1)
    allowed_labels = {int(item) for item in torch.unique(label_tensor, sorted=True).tolist()}
    source_paths = list(paths) if paths is not None else metadata_csv_paths(cfg, split_names=("eval",), include_full=True)
    _sample_to_group, label_to_group = _read_group_tables(source_paths, cfg, allowed_labels=allowed_labels)

    groups: dict[str, list[int]] = {group: [] for group in group_order(cfg)}
    for label_id in sorted(allowed_labels):
        group = label_to_group.get(int(label_id))
        if group in groups:
            groups[group].append(int(label_id))

    missing = [group for group in required_groups if not groups.get(str(group), [])]
    if missing and strict:
        raise ValueError(
            "Missing bucket labels for required groups "
            f"{missing}; provide strict bucket metadata in eval/full_dataset metadata."
        )
    return {group: torch.tensor(values, dtype=torch.long) for group, values in groups.items()}


def resolve_sample_shot_groups(
    sample_ids: Sequence[str],
    labels: Any,
    cfg: Any,
    *,
    paths: Sequence[Path] | None = None,
    strict: bool = True,
) -> list[str]:
    label_list = [int(item) for item in torch.as_tensor(labels).detach().cpu().long().reshape(-1).tolist()]
    if len(sample_ids) != len(label_list):
        raise ValueError(f"sample_ids length={len(sample_ids)} must match labels length={len(label_list)}")
    allowed_labels = set(label_list)
    source_paths = list(paths) if paths is not None else metadata_csv_paths(cfg, split_names=("eval",), include_full=True)
    sample_to_group, label_to_group = _read_group_tables(source_paths, cfg, allowed_labels=allowed_labels)

    result: list[str] = []
    missing: list[str] = []
    for sample_id, label_id in zip(sample_ids, label_list):
        group = sample_to_group.get(str(sample_id)) or label_to_group.get(int(label_id))
        if group is None:
            missing.append(str(sample_id))
            group = "unknown"
        result.append(str(group))

    if missing and strict:
        raise ValueError(
            "Missing bucket metadata for samples "
            f"{missing[:8]}; sample-level bucket is required."
        )
    return result


def validate_required_groups(groups: Sequence[str], *, required_groups: Sequence[str] = ("many", "medium", "few")) -> None:
    counts = Counter(str(item) for item in groups)
    missing = [group for group in required_groups if int(counts.get(group, 0)) <= 0]
    if missing:
        raise ValueError(f"Missing required shot groups {missing}; observed_counts={dict(counts)}")


__all__ = [
    "COUNT_COLUMNS",
    "DEFAULT_GROUP_ORDER",
    "GROUP_COLUMNS",
    "LABEL_ID_COLUMNS",
    "SAMPLE_ID_COLUMNS",
    "cfg_get",
    "dataset_cfg",
    "full_dataset_csv_path",
    "group_from_train_count",
    "group_name_from_row",
    "group_order",
    "label_groups_from_metadata",
    "label_id_from_row",
    "metadata_csv_paths",
    "normalize_group_name",
    "parse_optional_int",
    "resolve_sample_shot_groups",
    "sample_id_from_row",
    "split_csv_path",
    "validate_required_groups",
]
