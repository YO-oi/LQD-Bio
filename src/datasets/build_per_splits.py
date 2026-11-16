"""
PER
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.datasets.build_calibration import (  # noqa: E402
    REQUIRED_OUTPUT_COLUMNS as CALIBRATION_OUTPUT_COLUMNS,
    bucket_thresholds_metadata,
    build_coverage_report,
    sha256_file,
    sha256_text_lines,
    standardize_train_csv,
    validate_train_only_frame,
)
from src.utils.group_utils import (  # noqa: E402
    BASE_BUCKETS,
    TRAIN_LABEL_MANY_MIN,
    TRAIN_LABEL_MEDIUM_MIN,
    bucket_from_train_label_n,
)


DEFAULT_METADATA_DIR = PROJECT_ROOT / "PER_audio" / "metadata"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "PER_audio" / "splits"
REQUIRED_METADATA_COLUMNS = ("audio_path", "ebird_code")
OUTPUT_COLUMNS = (
    "sample_id",
    "file_name",
    "file_hash",
    "rel_path",
    "ebird_code",
    "label_id",
    "split",
    "source_split",
    "split_seed",
    "bucket",
    "train_label_n",
    "orig_index",
)
BUCKET_ORDER = BASE_BUCKETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PER train/test/eval/full split CSVs.")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-seed", type=int, default=int(getattr(config, "SPLIT_SEED", 42)))
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--many-threshold", type=int, default=TRAIN_LABEL_MANY_MIN)
    parser.add_argument("--medium-threshold", type=int, default=TRAIN_LABEL_MEDIUM_MIN)
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Upgrade existing calibration.csv and its manifest without rebuilding splits or resampling.",
    )
    return parser.parse_args()


def _read_metadata(path: Path, *, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} metadata does not exist: {path}")
    frame = pd.read_csv(path).reset_index(drop=True)
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{split_name} metadata missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{split_name} metadata is empty: {path}")
    for column in REQUIRED_METADATA_COLUMNS:
        if frame[column].isna().any():
            bad = frame.index[frame[column].isna()].tolist()[:8]
            raise ValueError(f"{split_name} metadata column {column!r} has missing values at rows: {bad}")
    return frame


def _normalize_ebird_code(value: Any) -> str:
    if value is None or pd.isna(value):
        raise ValueError("ebird_code is missing")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"invalid ebird_code value: {value!r}")
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        raise ValueError(f"invalid ebird_code value: {value!r}")
    return text


def extract_file_name(audio_path: Any) -> str:
    if audio_path is None or pd.isna(audio_path):
        raise ValueError("audio_path is missing")
    parts = [part for part in re.split(r"[\\/]+", str(audio_path).strip()) if part]
    if not parts:
        raise ValueError(f"cannot extract file_name from audio_path={audio_path!r}")
    return parts[-1]


def _stable_hex(value: str, *, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode("utf-8"))
    return hasher.hexdigest()


def _sample_id_from_rel_path(rel_path: str) -> str:
    return f"per_{_stable_hex(rel_path, algorithm='sha1')[:16]}"


def _bucket_from_train_count(
    train_label_n: int,
    *,
    many_threshold: int,
    medium_threshold: int,
) -> str:
    _validate_bucket_thresholds(many_threshold, medium_threshold)
    return bucket_from_train_label_n(train_label_n)


def _validate_bucket_thresholds(many_threshold: int, medium_threshold: int) -> None:
    if int(many_threshold) != TRAIN_LABEL_MANY_MIN or int(medium_threshold) != TRAIN_LABEL_MEDIUM_MIN:
        raise ValueError(
            "Bucket thresholds are locked by src.utils.group_utils: "
            f"expected many_threshold={TRAIN_LABEL_MANY_MIN}, "
            f"medium_threshold={TRAIN_LABEL_MEDIUM_MIN}; "
            f"got {many_threshold}, {medium_threshold}."
        )


def _build_label_mapping(train_metadata: pd.DataFrame) -> dict[str, int]:
    codes = [
        _normalize_ebird_code(value)
        for value in train_metadata["ebird_code"].tolist()
    ]

    def sort_key(code: str) -> tuple[int, int | str]:
        return (0, int(code)) if code.isdigit() else (1, code)

    unique_codes = sorted(set(codes), key=sort_key)

    # ebird_code 如果已经是连续分类头索引，就直接保留原始数字语义。
    if all(code.isdigit() for code in unique_codes):
        numeric_codes = sorted(int(code) for code in unique_codes)
        expected_codes = list(range(len(numeric_codes)))
        if numeric_codes == expected_codes:
            return {str(code): int(code) for code in numeric_codes}

    # 否则压缩成稳定、连续、从 0 开始的 label_id。
    return {
        str(code): label_id
        for label_id, code in enumerate(unique_codes)
    }


def _check_known_eval_classes(test_metadata: pd.DataFrame, label_mapping: dict[str, int]) -> None:
    test_codes = {_normalize_ebird_code(value) for value in test_metadata["ebird_code"].tolist()}
    missing = sorted(test_codes - set(label_mapping))
    if missing:
        raise ValueError(
            "test/eval metadata contains ebird_code values unseen in train metadata: "
            f"{missing[:20]}"
        )


def _test_eval_assignments(test_metadata: pd.DataFrame, *, split_seed: int, eval_ratio: float) -> pd.Series:
    if not (0.0 < float(eval_ratio) < 1.0):
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}")

    work = test_metadata.copy()
    work["_ebird_code_norm"] = [_normalize_ebird_code(value) for value in work["ebird_code"].tolist()]
    work["_file_name"] = [extract_file_name(value) for value in work["audio_path"].tolist()]
    assignments = pd.Series("test", index=work.index, dtype="object")

    for code, group in work.groupby("_ebird_code_norm", sort=True):
        n = len(group)
        if n <= 2:
            continue
        if n >= 5:
            n_eval = max(1, int(round(n * float(eval_ratio))))
        else:
            n_eval = 1
        n_eval = min(n_eval, n - 1)
        if n_eval <= 0:
            continue

        tie_breaker = "_orig_index" if "_orig_index" in group.columns else "_file_name"
        keyed = group.assign(
            _split_key=[
                _stable_hex(f"{split_seed}|{code}|{int(idx)}|{file_name}", algorithm="sha256")
                for idx, file_name in zip(group.index, group["_file_name"])
            ]
        ).sort_values(["_split_key", "_file_name", tie_breaker])
        eval_indices = keyed.index[:n_eval]
        assignments.loc[eval_indices] = "eval"

    return assignments


def _prepare_source_frame(
    metadata: pd.DataFrame,
    *,
    output_split: str,
    source_split: str,
    label_mapping: dict[str, int],
    train_label_counts: dict[str, int],
    split_seed: int,
    many_threshold: int,
    medium_threshold: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    physical_split = "test" if output_split == "eval" else output_split

    for fallback_index, row in metadata.reset_index(drop=True).iterrows():
        orig_index = int(row["_orig_index"]) if "_orig_index" in row and not pd.isna(row["_orig_index"]) else int(fallback_index)
        ebird_code = _normalize_ebird_code(row["ebird_code"])
        if ebird_code not in label_mapping:
            raise ValueError(f"{output_split} row {orig_index} has ebird_code not found in train: {ebird_code}")
        file_name = extract_file_name(row["audio_path"])
        rel_path = f"{physical_split}/{file_name}"
        train_label_n = int(train_label_counts.get(ebird_code, 0))
        bucket = _bucket_from_train_count(
            train_label_n,
            many_threshold=many_threshold,
            medium_threshold=medium_threshold,
        )
        file_hash = _stable_hex(rel_path, algorithm="sha256")
        rows.append(
            {
                "sample_id": _sample_id_from_rel_path(rel_path),
                "file_name": file_name,
                "file_hash": file_hash,
                "rel_path": rel_path,
                "ebird_code": ebird_code,
                "label_id": int(label_mapping[ebird_code]),
                "split": output_split,
                "source_split": source_split,
                "split_seed": int(split_seed),
                "bucket": bucket,
                "train_label_n": train_label_n,
                "orig_index": orig_index,
            }
        )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if result["rel_path"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{output_split} split contains empty rel_path")
    return result


def build_per_splits(
    *,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    split_seed: int = int(getattr(config, "SPLIT_SEED", 42)),
    eval_ratio: float = 0.2,
    many_threshold: int = TRAIN_LABEL_MANY_MIN,
    medium_threshold: int = TRAIN_LABEL_MEDIUM_MIN,
) -> dict[str, pd.DataFrame]:
    _validate_bucket_thresholds(many_threshold, medium_threshold)

    train_metadata = _read_metadata(metadata_dir / "train_metadata.csv", split_name="train")
    test_metadata = _read_metadata(metadata_dir / "test_metadata.csv", split_name="test")
    train_metadata["_orig_index"] = np.arange(len(train_metadata), dtype=np.int64)
    test_metadata["_orig_index"] = np.arange(len(test_metadata), dtype=np.int64)
    label_mapping = _build_label_mapping(train_metadata)
    _check_known_eval_classes(test_metadata, label_mapping)

    train_codes = [_normalize_ebird_code(value) for value in train_metadata["ebird_code"].tolist()]
    train_label_counts = pd.Series(train_codes).value_counts().astype(int).to_dict()

    train_split = _prepare_source_frame(
        train_metadata,
        output_split="train",
        source_split="train",
        label_mapping=label_mapping,
        train_label_counts=train_label_counts,
        split_seed=split_seed,
        many_threshold=many_threshold,
        medium_threshold=medium_threshold,
    )

    assignments = _test_eval_assignments(test_metadata, split_seed=split_seed, eval_ratio=eval_ratio)
    test_split = _prepare_source_frame(
        test_metadata.loc[assignments.eq("test")].reset_index(drop=True),
        output_split="test",
        source_split="test",
        label_mapping=label_mapping,
        train_label_counts=train_label_counts,
        split_seed=split_seed,
        many_threshold=many_threshold,
        medium_threshold=medium_threshold,
    )
    eval_split = _prepare_source_frame(
        test_metadata.loc[assignments.eq("eval")].reset_index(drop=True),
        output_split="eval",
        source_split="test",
        label_mapping=label_mapping,
        train_label_counts=train_label_counts,
        split_seed=split_seed,
        many_threshold=many_threshold,
        medium_threshold=medium_threshold,
    )
    full_dataset = pd.concat([train_split, test_split, eval_split], ignore_index=True)

    _validate_split_outputs(
        {
            "train": train_split,
            "test": test_split,
            "eval": eval_split,
            "full_dataset": full_dataset,
        }
    )
    return {
        "train": train_split,
        "test": test_split,
        "eval": eval_split,
        "full_dataset": full_dataset,
    }


def _validate_split_outputs(splits: dict[str, pd.DataFrame]) -> None:
    full = splits["full_dataset"]
    for name, frame in splits.items():
        missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
        if missing:
            raise KeyError(f"{name}.csv missing output columns: {missing}")
        empty_rel = frame["rel_path"].astype(str).str.strip().eq("")
        if empty_rel.any():
            raise ValueError(f"{name}.csv contains empty rel_path rows: {frame.index[empty_rel].tolist()[:8]}")
    for column in ("sample_id", "file_hash", "rel_path"):
        if full[column].duplicated().any():
            values = full.loc[full[column].duplicated(), column].head(8).tolist()
            raise ValueError(f"full_dataset has duplicated {column}, which would break sample alignment: {values}")


def _duplicate_report(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    duplicated = frame[column].astype(str).duplicated(keep=False)
    values = sorted(frame.loc[duplicated, column].astype(str).unique().tolist())
    return {
        "has_duplicates": bool(len(values) > 0),
        "num_duplicate_values": int(len(values)),
        "examples": values[:20],
    }


def _bucket_sample_count_report(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["bucket"].astype(str).value_counts()
    return {bucket: int(counts.get(bucket, 0)) for bucket in BUCKET_ORDER}


def _bucket_class_count_report(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bucket in BUCKET_ORDER:
        part = frame[frame["bucket"].astype(str).eq(bucket)]
        counts[bucket] = int(part["ebird_code"].astype(str).nunique())
    return counts


def build_split_report(
    splits: dict[str, pd.DataFrame],
    *,
    split_seed: int,
    eval_ratio_requested: float,
    train_class_set: set[str],
    many_threshold: int,
    medium_threshold: int,
) -> dict[str, Any]:
    full = splits["full_dataset"]
    test_eval_total = len(splits["test"]) + len(splits["eval"])
    test_eval_codes = set(pd.concat([splits["test"], splits["eval"]], ignore_index=True)["ebird_code"].astype(str))
    unseen = sorted(test_eval_codes - set(train_class_set))

    class_counts_by_split = {
        name: int(frame["ebird_code"].astype(str).nunique())
        for name, frame in splits.items()
        if name != "full_dataset"
    }
    bucket_sample_counts = {
        name: _bucket_sample_count_report(frame)
        for name, frame in splits.items()
    }
    bucket_class_counts = {
        name: _bucket_class_count_report(frame)
        for name, frame in splits.items()
    }

    return {
        "split_seed": int(split_seed),
        "eval_ratio": {
            "requested": float(eval_ratio_requested),
            "actual": float(len(splits["eval"]) / test_eval_total) if test_eval_total > 0 else float("nan"),
        },
        "bucket_order": list(BUCKET_ORDER),
        "bucket_thresholds": {
            "many": {"train_label_n_min": int(many_threshold)},
            "medium": {"train_label_n_min": int(medium_threshold), "train_label_n_max": int(many_threshold - 1)},
            "few": {"train_label_n_max": int(medium_threshold - 1)},
        },
        "num_samples": {
            "train": int(len(splits["train"])),
            "test": int(len(splits["test"])),
            "eval": int(len(splits["eval"])),
            "full_dataset": int(len(full)),
        },
        "num_classes": int(len(train_class_set)),
        "num_classes_by_split": class_counts_by_split,
        "test_eval_contains_unseen_train_classes": {
            "exists": bool(unseen),
            "values": unseen,
        },
        "file_name_duplicates": _duplicate_report(full, "file_name"),
        "file_hash_duplicates": _duplicate_report(full, "file_hash"),
        "sample_id_duplicates": _duplicate_report(full, "sample_id"),
        "rel_path_empty": {
            "exists": bool(full["rel_path"].astype(str).str.strip().eq("").any()),
            "count": int(full["rel_path"].astype(str).str.strip().eq("").sum()),
        },
        "bucket_sample_counts": bucket_sample_counts,
        "bucket_class_counts": bucket_class_counts,
    }


def _label_mapping_from_train_split(train_split: pd.DataFrame) -> dict[str, int]:
    mapping = (
        train_split[["ebird_code", "label_id"]]
        .drop_duplicates()
        .sort_values("label_id")
    )
    return {str(row["ebird_code"]): int(row["label_id"]) for _, row in mapping.iterrows()}


def upgrade_calibration_frame(calibration: pd.DataFrame, train_split: pd.DataFrame) -> pd.DataFrame:
    """Fill the current schema from train rows while retaining calibration selection and order."""

    for name, frame in (("train.csv", train_split), ("calibration.csv", calibration)):
        validate_train_only_frame(frame, source_name=name)
        if frame.empty:
            raise ValueError(f"{name} is empty")
        if "sample_id" not in frame.columns or frame["sample_id"].isna().any():
            raise ValueError(f"{name} must contain non-missing sample_id values")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicated sample_id values")
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in train_split.columns]
    if missing_columns:
        raise ValueError(f"train.csv missing columns: {missing_columns}")

    train = standardize_train_csv(train_split).set_index("sample_id", drop=False)
    train_counts = train_split.groupby("label_id").size()
    if not train["train_label_n"].eq(train["label_id"].map(train_counts)).all():
        raise ValueError("train.csv train_label_n does not match actual train label counts")
    sample_ids = calibration["sample_id"].tolist()
    missing_ids = [sample_id for sample_id in sample_ids if sample_id not in train.index]
    if missing_ids:
        raise ValueError(f"calibration.csv contains samples outside train.csv: {missing_ids[:8]}")

    selected = train.loc[sample_ids].reset_index(drop=True)
    result = calibration.reset_index(drop=True).copy()
    for column in ("rel_path", "ebird_code", "label_id"):
        if column not in result:
            continue
        if column == "rel_path":
            observed = result[column].astype(str).str.replace("\\", "/", regex=False)
            expected = selected[column].astype(str).str.replace("\\", "/", regex=False)
        else:
            observed = result[column].map(_normalize_ebird_code)
            expected = selected[column].map(_normalize_ebird_code)
        if not observed.eq(expected).all():
            raise ValueError(f"calibration.csv {column} does not match train.csv sample identity")

    for column in CALIBRATION_OUTPUT_COLUMNS:
        if column != "calibration_index":
            result[column] = selected[column]
    if "calibration_index" not in result:
        result["calibration_index"] = np.arange(len(result), dtype=np.int64)
    columns = list(CALIBRATION_OUTPUT_COLUMNS)
    columns.extend(column for column in result if column not in columns)
    return result[columns]


def upgrade_existing_calibration(
    output_dir: Path,
    *,
    train_split: pd.DataFrame | None = None,
) -> dict[str, Path]:
    calibration_path = output_dir / "calibration.csv"
    if not calibration_path.exists():
        return {}
    train_path = output_dir / "train.csv"
    if train_split is None:
        train_split = pd.read_csv(train_path)
    calibration = upgrade_calibration_frame(pd.read_csv(calibration_path), train_split)
    manifest_path = calibration_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        raise ValueError(f"calibration manifest must be a JSON object: {manifest_path}")

    # Retain sampling provenance; this operation only refreshes schema and identities.
    manifest.update(
        {
            "dataset": "PER_audio",
            "source_train_csv": str(train_path),
            "source_train_csv_sha256": sha256_file(train_path),
            "output_calibration_csv": str(calibration_path),
            "bucket_order": list(BUCKET_ORDER),
            "bucket_thresholds": bucket_thresholds_metadata(),
            "bucket_aliases": dict(config.BUCKET_ALIASES),
            "num_samples": len(calibration),
            "group_counts": calibration["bucket"].value_counts().reindex(BUCKET_ORDER, fill_value=0).astype(int).to_dict(),
            "num_labels": int(calibration["label_id"].nunique()),
            "label_counts": calibration["label"].astype(str).value_counts().sort_index().astype(int).to_dict(),
            "sample_ids": calibration["sample_id"].astype(str).tolist(),
            "sample_id_sha256": sha256_text_lines(calibration["sample_id"].astype(str).tolist()),
            "file_hash_sha256": sha256_text_lines(calibration["file_hash"].astype(str).tolist()),
            "calibration_index_sha256": sha256_text_lines(calibration["calibration_index"].astype(str).tolist()),
            "coverage": build_coverage_report(
                train_df=standardize_train_csv(train_split),
                calibration_df=calibration,
            ),
        }
    )
    temporary_csv = calibration_path.with_name(calibration_path.name + ".tmp")
    calibration.to_csv(temporary_csv, index=False, encoding="utf-8")
    manifest["calibration_csv_sha256"] = sha256_file(temporary_csv)
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_csv.replace(calibration_path)
    temporary_manifest.replace(manifest_path)
    return {"calibration": calibration_path, "calibration_manifest": manifest_path}


def save_outputs(
    splits: dict[str, pd.DataFrame],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    split_seed: int,
    eval_ratio_requested: float,
    many_threshold: int = TRAIN_LABEL_MANY_MIN,
    medium_threshold: int = TRAIN_LABEL_MEDIUM_MIN,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for name in ("train", "test", "eval", "full_dataset"):
        output_path = output_dir / f"{name}.csv"
        splits[name].to_csv(output_path, index=False)
        output_paths[name] = output_path

    train_class_set = set(splits["train"]["ebird_code"].astype(str).unique().tolist())
    report = build_split_report(
        splits,
        split_seed=split_seed,
        eval_ratio_requested=eval_ratio_requested,
        train_class_set=train_class_set,
        many_threshold=many_threshold,
        medium_threshold=medium_threshold,
    )
    report_path = output_dir / "split_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    output_paths["report"] = report_path

    label_mapping_path = output_dir / "label_mapping.json"
    with label_mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(_label_mapping_from_train_split(splits["train"]), handle, ensure_ascii=False, indent=2)
    output_paths["label_mapping"] = label_mapping_path
    output_paths.update(upgrade_existing_calibration(output_dir, train_split=splits["train"]))
    return output_paths


def main() -> None:
    args = parse_args()
    if args.calibration_only:
        paths = upgrade_existing_calibration(args.output_dir)
        if not paths:
            raise FileNotFoundError(f"No existing calibration.csv to upgrade under {args.output_dir}")
        for name, path in paths.items():
            print(f"Saved {name}: {path}")
        return
    splits = build_per_splits(
        metadata_dir=args.metadata_dir,
        split_seed=args.split_seed,
        eval_ratio=args.eval_ratio,
        many_threshold=args.many_threshold,
        medium_threshold=args.medium_threshold,
    )
    paths = save_outputs(
        splits,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        eval_ratio_requested=args.eval_ratio,
        many_threshold=args.many_threshold,
        medium_threshold=args.medium_threshold,
    )
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
