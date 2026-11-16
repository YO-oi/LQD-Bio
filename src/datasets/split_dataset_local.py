"""
local data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.utils.group_utils import (  # noqa: E402
    BASE_BUCKETS,
    TRAIN_LABEL_MANY_MIN,
    TRAIN_LABEL_MEDIUM_MIN,
    bucket_from_train_label_n,
)


LOCAL_DATA_ROOT = PROJECT_ROOT / "local_data"
DEFAULT_DATA_DIR = LOCAL_DATA_ROOT / "data"
DEFAULT_OUTPUT_DIR = LOCAL_DATA_ROOT / "splits"
AUDIO_SUFFIXES = (".ogg", ".wav", ".flac", ".mp3")
BUCKET_ORDER = BASE_BUCKETS
EVAL_SPLITS = ("eval",)
OUTPUT_COLUMNS = (
    "sample_id",
    "file_name",
    "file_hash",
    "rel_path",
    "filepath",
    "file_id",
    "ebird_code",
    "label",
    "label_id",
    "split",
    "source_split",
    "split_seed",
    "bucket",
    "label_n",
    "train_label_n",
    "orig_index",
    "file_size",
)
TRAIN_RATIO = 0.80
VAL_RATIO =  0.10
TEST_RATIO = 0.10


def _stable_hex(value: str, *, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode("utf-8"))
    return hasher.hexdigest()


def _sample_id_from_rel_path(rel_path: str) -> str:
    return f"local_{_stable_hex(rel_path, algorithm='sha1')[:16]}"


def generate_file_hash(filepath: Path) -> str:
    """Compute a content hash when explicitly requested."""
    sha256_hash = hashlib.sha256()
    with filepath.open("rb") as file_obj:
        for byte_block in iter(lambda: file_obj.read(1024 * 1024), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def shot_bucket(n: int) -> str:
    return bucket_from_train_label_n(n)


def _build_label_mapping(class_folders: list[Path]) -> dict[str, int]:
    return {folder.name: idx for idx, folder in enumerate(sorted(class_folders, key=lambda path: path.name))}


def load_dataset_index(data_dir: Path = DEFAULT_DATA_DIR, *, hash_files: bool = False) -> pd.DataFrame:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    class_folders = sorted((path for path in data_dir.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not class_folders:
        raise ValueError(f"No class folders found under {data_dir}")

    label_mapping = _build_label_mapping(class_folders)
    records: list[dict[str, Any]] = []
    for folder in class_folders:
        audio_files = sorted(
            (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES),
            key=lambda path: path.name,
        )
        if not audio_files:
            raise ValueError(f"Class folder contains no supported audio files: {folder}")

        for file_path in audio_files:
            rel_path = f"data/{folder.name}/{file_path.name}"
            project_rel_path = f"local_data/{rel_path}"
            records.append(
                {
                    "sample_id": _sample_id_from_rel_path(rel_path),
                    "file_name": file_path.name,
                    "file_hash": generate_file_hash(file_path) if hash_files else _stable_hex(rel_path),
                    "rel_path": rel_path,
                    "filepath": project_rel_path,
                    "file_id": project_rel_path,
                    "ebird_code": folder.name,
                    "label": folder.name,
                    "label_id": int(label_mapping[folder.name]),
                    "source_split": "local_data",
                    "orig_index": len(records),
                    "file_size": int(file_path.stat().st_size),
                }
            )

    dataset = pd.DataFrame.from_records(records)
    if dataset.empty:
        raise ValueError(f"No supported audio files found under {data_dir}")
    _validate_dataset_index(dataset)
    return dataset


def _validate_dataset_index(dataset: pd.DataFrame) -> None:
    for column in ("sample_id", "rel_path", "filepath", "file_hash", "ebird_code", "label_id"):
        if dataset[column].isna().any():
            raise ValueError(f"Dataset index contains missing values in {column}")
    for column in ("sample_id", "rel_path", "filepath", "file_id"):
        if dataset[column].duplicated().any():
            examples = dataset.loc[dataset[column].duplicated(), column].head(8).tolist()
            raise ValueError(f"Dataset index contains duplicated {column}: {examples}")
    expected_label_ids = list(range(int(dataset["label_id"].nunique())))
    observed_label_ids = sorted(int(value) for value in dataset["label_id"].unique())
    if observed_label_ids != expected_label_ids:
        raise ValueError(f"label_id must be continuous from 0: {observed_label_ids[:12]}")


def summarize_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, label_df in df.groupby("label", sort=True):
        n = len(label_df)
        rows.append(
            {
                "ebird_code": str(label),
                "label": str(label),
                "label_id": int(label_df["label_id"].iloc[0]),
                "label_n": int(n),
                "bucket": shot_bucket(n),
            }
        )
    return pd.DataFrame(rows).sort_values("label_id").reset_index(drop=True)


def append_split(
    parts: list[pd.DataFrame],
    rows: pd.DataFrame,
    *,
    split_seed: int,
    split_name: str,
    label_n: int,
) -> None:
    if rows.empty:
        return
    part = rows.copy()
    part["split_seed"] = int(split_seed)
    part["split"] = split_name
    part["label_n"] = int(label_n)
    parts.append(part)


def _apply_train_label_buckets(split_df: pd.DataFrame) -> pd.DataFrame:
    train_rows = split_df[split_df["split"].astype(str).eq("train")]
    train_counts = train_rows.groupby("label_id", sort=True).size().astype(int).to_dict()
    label_ids = sorted(int(value) for value in split_df["label_id"].unique())
    missing = [label_id for label_id in label_ids if label_id not in train_counts]
    if missing:
        raise ValueError(f"Missing train rows for label_id values: {missing[:8]}")

    result = split_df.copy()
    result["train_label_n"] = [int(train_counts[int(label_id)]) for label_id in result["label_id"].tolist()]
    result["label_n"] = result["train_label_n"].astype(int)
    result["bucket"] = [shot_bucket(int(count)) for count in result["train_label_n"].tolist()]
    return result


def apply_split_policy(df: pd.DataFrame, split_seed: int = int(config.SPLIT_SEED)) -> pd.DataFrame:
    """Keep the original local split policy and derive buckets from train counts."""
    parts: list[pd.DataFrame] = []

    for _, group in df.groupby("label", sort=True):
        n = len(group)
        group = group.sample(frac=1, random_state=int(split_seed)).reset_index(drop=True)

        if n >= TRAIN_LABEL_MANY_MIN:
            n_train = int(n * TRAIN_RATIO)
            n_val = int(n * VAL_RATIO)
            append_split(parts, group.iloc[:n_train], split_seed=split_seed, split_name="train", label_n=n)
            append_split(
                parts,
                group.iloc[n_train : n_train + n_val],
                split_seed=split_seed,
                split_name="eval",
                label_n=n,
            )
            append_split(
                parts,
                group.iloc[n_train + n_val :],
                split_seed=split_seed,
                split_name="test",
                label_n=n,
            )
        elif n >= TRAIN_LABEL_MEDIUM_MIN:
            n_train = int(n * TRAIN_RATIO)
            n_val = max(1, int(n * VAL_RATIO))
            append_split(parts, group.iloc[:n_train], split_seed=split_seed, split_name="train", label_n=n)
            append_split(
                parts,
                group.iloc[n_train : n_train + n_val],
                split_seed=split_seed,
                split_name="eval",
                label_n=n,
            )
            append_split(
                parts,
                group.iloc[n_train + n_val :],
                split_seed=split_seed,
                split_name="test",
                label_n=n,
            )
        elif n >= 10:
            n_train = int(n * TRAIN_RATIO)
            append_split(parts, group.iloc[:n_train], split_seed=split_seed, split_name="train", label_n=n)
            append_split(
                parts,
                group.iloc[n_train:],
                split_seed=split_seed,
                split_name="eval",
                label_n=n,
            )
        else:
            if n < 3:
                raise ValueError(
                    f"Few class {group['label'].iloc[0]!r} has only {n} samples; "
                    "cannot keep at least one train, one eval, and one test sample."
                )
            append_split(parts, group.iloc[:-2], split_seed=split_seed, split_name="train", label_n=n)
            append_split(parts, group.iloc[-2:-1], split_seed=split_seed, split_name="eval", label_n=n)
            append_split(parts, group.iloc[-1:], split_seed=split_seed, split_name="test", label_n=n)

    if not parts:
        raise ValueError("Split policy produced no rows")
    split_df = pd.concat(parts, ignore_index=True)
    split_df = _apply_train_label_buckets(split_df)
    return split_df.sort_values(["split", "label", "file_name"]).reset_index(drop=True)


def build_output_splits(split_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": split_df[split_df["split"].eq("train")].reset_index(drop=True),
        "test": split_df[split_df["split"].eq("test")].reset_index(drop=True),
        "eval": split_df[split_df["split"].isin(EVAL_SPLITS)].reset_index(drop=True),
    }


def _label_mapping_from_split(split_df: pd.DataFrame) -> dict[str, int]:
    mapping = (
        split_df[["ebird_code", "label_id"]]
        .drop_duplicates()
        .sort_values("label_id")
    )
    return {str(row["ebird_code"]): int(row["label_id"]) for _, row in mapping.iterrows()}


def _bucket_sample_count_report(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["bucket"].astype(str).value_counts()
    return {bucket: int(counts.get(bucket, 0)) for bucket in BUCKET_ORDER}


def _bucket_class_count_report(frame: pd.DataFrame) -> dict[str, int]:
    return {
        bucket: int(frame.loc[frame["bucket"].astype(str).eq(bucket), "ebird_code"].astype(str).nunique())
        for bucket in BUCKET_ORDER
    }


def _duplicate_report(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    duplicated = frame[column].astype(str).duplicated(keep=False)
    values = sorted(frame.loc[duplicated, column].astype(str).unique().tolist())
    return {
        "has_duplicates": bool(values),
        "num_duplicate_values": int(len(values)),
        "examples": values[:20],
    }


def build_split_report(split_df: pd.DataFrame, output_splits: dict[str, pd.DataFrame], *, split_seed: int) -> dict[str, Any]:
    return {
        "split_seed": int(split_seed),
        "data_layout": "local_data/data/<ebird_code>/<audio_file>",
        "path_columns": {
            "rel_path": "relative to local_data root",
            "filepath": "relative to project root",
        },
        "eval_splits": list(EVAL_SPLITS),
        "bucket_order": list(BUCKET_ORDER),
        "bucket_source": "train split label_id counts stored in train_label_n",
        "bucket_thresholds": {
            "many": {"train_label_n_min": TRAIN_LABEL_MANY_MIN},
            "medium": {"train_label_n_min": TRAIN_LABEL_MEDIUM_MIN, "train_label_n_max": TRAIN_LABEL_MANY_MIN - 1},
            "few": {"train_label_n_max": TRAIN_LABEL_MEDIUM_MIN - 1},
        },
        "num_samples": {
            "train": int(len(output_splits["train"])),
            "test": int(len(output_splits["test"])),
            "eval": int(len(output_splits["eval"])),
            "all_assignments": int(len(split_df)),
        },
        "num_classes": int(split_df["ebird_code"].astype(str).nunique()),
        "num_classes_by_output": {
            name: int(frame["ebird_code"].astype(str).nunique())
            for name, frame in output_splits.items()
        },
        "num_classes_by_internal_split": {
            str(split): int(part["ebird_code"].astype(str).nunique())
            for split, part in split_df.groupby("split", sort=True)
        },
        "internal_split_sample_counts": {
            str(split): int(count)
            for split, count in split_df["split"].value_counts().sort_index().items()
        },
        "bucket_sample_counts": {
            name: _bucket_sample_count_report(frame)
            for name, frame in output_splits.items()
        },
        "bucket_class_counts": {
            name: _bucket_class_count_report(frame)
            for name, frame in output_splits.items()
        },
        "rel_path_empty": {
            "exists": bool(split_df["rel_path"].astype(str).str.strip().eq("").any()),
            "count": int(split_df["rel_path"].astype(str).str.strip().eq("").sum()),
        },
        "file_name_duplicates": _duplicate_report(split_df, "file_name"),
        "rel_path_duplicates": _duplicate_report(split_df, "rel_path"),
        "file_hash_duplicates": _duplicate_report(split_df, "file_hash"),
        "sample_id_duplicates": _duplicate_report(split_df, "sample_id"),
    }


def _validate_split_outputs(output_splits: dict[str, pd.DataFrame]) -> None:
    for name, frame in output_splits.items():
        missing = [column for column in OUTPUT_COLUMNS if column not in frame.columns]
        if missing:
            raise KeyError(f"{name}.csv missing columns: {missing}")
        if frame["rel_path"].astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}.csv contains empty rel_path")
        invalid_bucket = sorted(set(frame["bucket"].astype(str)) - set(BUCKET_ORDER))
        if invalid_bucket:
            raise ValueError(f"{name}.csv contains invalid bucket values: {invalid_bucket}")
        expected_bucket = [shot_bucket(value) for value in frame["train_label_n"].tolist()]
        mismatched = [
            idx for idx, (observed, expected) in enumerate(zip(frame["bucket"].astype(str).tolist(), expected_bucket))
            if observed != expected
        ]
        if mismatched:
            preview = ", ".join(str(idx) for idx in mismatched[:8])
            raise ValueError(f"{name}.csv bucket does not match train_label_n at rows: {preview}")
        if not frame["label_n"].astype(int).equals(frame["train_label_n"].astype(int)):
            raise ValueError(f"{name}.csv label_n must equal train_label_n for legacy compatibility")


def save_split_csvs(
    split_df: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    split_seed: int = int(config.SPLIT_SEED),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_splits = build_output_splits(split_df)
    _validate_split_outputs(output_splits)

    output_paths: dict[str, Path] = {}
    for name in ("train", "test", "eval"):
        output_path = output_dir / f"{name}.csv"
        output_splits[name][list(OUTPUT_COLUMNS)].to_csv(output_path, index=False)
        output_paths[name] = output_path

    full_dataset = pd.concat(
        [output_splits["train"], output_splits["test"], output_splits["eval"]],
        ignore_index=True,
    )
    full_dataset_path = output_dir / "full_dataset.csv"
    full_dataset[list(OUTPUT_COLUMNS)].to_csv(full_dataset_path, index=False)
    output_paths["full_dataset"] = full_dataset_path

    label_mapping_path = output_dir / "label_mapping.json"
    with label_mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(_label_mapping_from_split(split_df), handle, ensure_ascii=False, indent=2)
    output_paths["label_mapping"] = label_mapping_path

    report_path = output_dir / "split_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(build_split_report(split_df, output_splits, split_seed=split_seed), handle, ensure_ascii=False, indent=2)
    output_paths["split_report"] = report_path
    return output_paths


def print_summary(split_df: pd.DataFrame, output_paths: dict[str, Path]) -> None:
    print("\nLocal dataset split summary")
    print("===========================")
    print("\nInternal split counts")
    print(split_df["split"].value_counts().sort_index().to_string())
    print("\nBucket counts")
    print(
        split_df.drop_duplicates(["label", "bucket"])
        .groupby("bucket", as_index=False)
        .agg(labels=("label", "nunique"))
        .sort_values("bucket")
        .to_string(index=False)
    )
    print("\nCSV outputs")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create fixed train/eval/test CSVs for local_data.")
    parser.add_argument("--split-seed", type=int, default=int(config.SPLIT_SEED))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="Use content sha256 as file_hash. By default file_hash is derived from portable rel_path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = load_dataset_index(args.data_dir, hash_files=bool(args.hash_files))
    split_df = apply_split_policy(dataset, split_seed=int(args.split_seed))
    output_paths = save_split_csvs(split_df, args.output_dir, split_seed=int(args.split_seed))
    print_summary(split_df, output_paths)


if __name__ == "__main__":
    main()
