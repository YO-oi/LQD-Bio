# build_calibration.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import config

try:
    from src.utils.group_utils import (
        BASE_BUCKETS,
        TRAIN_LABEL_MANY_MIN,
        TRAIN_LABEL_MEDIUM_MIN,
        GroupValidationError,
        bucket_from_train_label_n,
        normalize_bucket as _normalize_sample_bucket,
        parse_train_label_n,
        read_canonical_bucket,
    )
except ImportError:  # pragma: no cover - package-relative debugging
    from ..utils.group_utils import (
        BASE_BUCKETS,
        TRAIN_LABEL_MANY_MIN,
        TRAIN_LABEL_MEDIUM_MIN,
        GroupValidationError,
        bucket_from_train_label_n,
        normalize_bucket as _normalize_sample_bucket,
        parse_train_label_n,
        read_canonical_bucket,
    )


REQUIRED_OUTPUT_COLUMNS = [
    "calibration_index",
    "sample_id",
    "file_name",
    "file_hash",
    "rel_path",
    "filepath",
    "file_id",
    "label",
    "ebird_code",
    "label_id",
    "split",
    "source_split",
    "split_seed",
    "shot_group",
    "bucket",
    "label_n",
    "train_label_n",
    "orig_index",
    "file_size",
]


UINT32_MAX = 2**32 - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed calibration.csv from train.csv."
    )
    parser.add_argument(
        "--dataset",
        default=config.ACTIVE_DATASET,
        choices=tuple(config.DATASETS.keys()),
        help="Dataset name defined in config.DATASETS.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(config.SPLIT_SEED),
        help="Fixed random seed for calibration sampling.",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=128,
        help="Maximum samples per many / medium / few group.",
    )
    parser.add_argument(
        "--min-samples-per-group",
        type=int,
        default=1,
        help="Minimum required samples per group if the group exists.",
    )
    parser.add_argument(
        "--many-max-samples-per-label",
        type=int,
        default=32,
        help=(
            "Per-label upper bound inside the many group. "
            "Use <=0 to disable the cap, which is not recommended."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="calibration.csv",
        help="Output CSV name saved under the dataset split directory.",
    )
    parser.add_argument(
        "--allow-missing-groups",
        action="store_true",
        help="Allow a group to be absent instead of raising an error.",
    )
    return parser.parse_args()


def normalize_path_string(value: Any) -> str:
    text = str(value).strip()
    return text.replace("\\", "/")


def stable_uint32_seed(*parts: Any) -> int:
    """Derive a stable pandas.sample random_state from seed and metadata."""

    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % UINT32_MAX


def sha256_text_lines(values: list[str]) -> str:
    """Compute a stable sha256 over ordered string lines for sample_id hash."""

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def get_file_name(row: pd.Series) -> str:
    if "file_name" in row and pd.notna(row["file_name"]) and str(row["file_name"]).strip():
        return Path(normalize_path_string(row["file_name"])).name

    for col in ("rel_path", "filepath", "audio_path", "path", "absolute_path"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return Path(normalize_path_string(row[col])).name

    raise ValueError("Cannot infer file_name: missing file_name / rel_path / audio_path.")


def get_rel_path(row: pd.Series) -> str:
    if "rel_path" in row and pd.notna(row["rel_path"]) and str(row["rel_path"]).strip():
        return normalize_path_string(row["rel_path"])

    if "split" in row and pd.notna(row["split"]):
        split = str(row["split"]).strip()
    else:
        split = "train"

    file_name = get_file_name(row)
    return f"{split}/{file_name}"


def get_label(row: pd.Series) -> str:
    for col in ("label", "ebird_code", "primary_label", "class_name", "species"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    raise ValueError("Cannot infer label: missing label / ebird_code / primary_label.")


def normalize_bucket(value: Any) -> str:
    raw = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        str(k).lower().replace(" ", "_").replace("-", "_"): str(v)
        for k, v in config.BUCKET_ALIASES.items()
    }
    normalized = aliases.get(raw, raw)
    try:
        return _normalize_sample_bucket(normalized, field_name="bucket")
    except GroupValidationError as exc:
        raise ValueError(str(exc)) from exc

def bucket_order() -> tuple[str, ...]:
    try:
        configured = tuple(
            _normalize_sample_bucket(item, field_name="config.BUCKET_ORDER")
            for item in getattr(config, "BUCKET_ORDER", BASE_BUCKETS)
        )
    except GroupValidationError as exc:
        raise ValueError(str(exc)) from exc
    expected = tuple(BASE_BUCKETS)
    if configured != expected:
        raise ValueError(
            "config.BUCKET_ORDER must match src.utils.group_utils.BASE_BUCKETS exactly: "
            f"expected {expected}, got {configured}."
        )
    return expected


def bucket_thresholds_metadata() -> dict[str, dict[str, int]]:
    return {
        "many": {"min": TRAIN_LABEL_MANY_MIN},
        "medium": {"min": TRAIN_LABEL_MEDIUM_MIN, "max": TRAIN_LABEL_MANY_MIN - 1},
        "few": {"max": TRAIN_LABEL_MEDIUM_MIN - 1},
    }


def bucket_from_count(label_n: int) -> str:
    return bucket_from_train_label_n(label_n)

def ensure_label_count(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve the canonical label_n source."""

    frame = df.copy()

    if "label" not in frame.columns:
        frame["label"] = frame.apply(get_label, axis=1)

    if "ebird_code" not in frame.columns:
        frame["ebird_code"] = frame["label"].astype(str)

    if "train_label_n" in frame.columns:
        frame["label_n"] = pd.to_numeric(
            frame["train_label_n"], errors="raise"
        ).astype(int)
    elif "label_n" in frame.columns and not frame["label_n"].isna().any():
        frame["label_n"] = pd.to_numeric(frame["label_n"], errors="raise").astype(int)
    else:
        counts = frame["label"].astype(str).value_counts().to_dict()
        frame["label_n"] = frame["label"].astype(str).map(counts).astype(int)

    frame["label_n"] = [parse_train_label_n(value) for value in frame["label_n"].tolist()]
    # Preserve train_label_n so downstream scripts do not reinterpret label_n.
    if "train_label_n" not in frame.columns:
        frame["train_label_n"] = frame["label_n"].astype(int)
    else:
        frame["train_label_n"] = frame["label_n"].astype(int)

    return frame


def ensure_shot_group(df: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_label_count(df)
    train_values = frame["train_label_n"].tolist() if "train_label_n" in frame.columns else None

    if "bucket" in frame.columns:
        try:
            result = read_canonical_bucket(
                bucket=frame["bucket"].tolist(),
                shot_group=frame["shot_group"].tolist() if "shot_group" in frame.columns else None,
                train_label_n=train_values,
                context="calibration source table",
                allow_shot_group_fallback=False,
            )
        except GroupValidationError as exc:
            raise ValueError(str(exc)) from exc
        frame["bucket"] = list(result.bucket)
    else:
        derived_bucket = [bucket_from_count(value) for value in frame["label_n"].tolist()]
        if "shot_group" in frame.columns:
            try:
                result = read_canonical_bucket(
                    bucket=derived_bucket,
                    shot_group=frame["shot_group"].tolist(),
                    train_label_n=train_values,
                    context="calibration source table",
                    allow_shot_group_fallback=False,
                )
            except GroupValidationError as exc:
                raise ValueError(str(exc)) from exc
            frame["bucket"] = list(result.bucket)
        else:
            frame["bucket"] = derived_bucket

    frame["shot_group"] = frame["bucket"].astype(str)
    return frame

def ensure_file_hash(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "file_hash" not in result.columns:
        result["file_hash"] = ""
    else:
        result["file_hash"] = result["file_hash"].fillna("").astype(str)

    empty_mask = result["file_hash"].str.strip() == ""
    if empty_mask.any():
        result.loc[empty_mask, "file_hash"] = result.loc[empty_mask, "rel_path"].map(
            lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()
        )

    return result




def stable_sample_id_from_rel_path(rel_path: Any) -> str:
    normalized = normalize_path_string(rel_path)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sid_{digest[:20]}"

def ensure_sample_id(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "sample_id" not in result.columns:
        result["sample_id"] = result["rel_path"].map(stable_sample_id_from_rel_path)
    else:
        result["sample_id"] = result["sample_id"].fillna("").astype(str)
        empty_mask = result["sample_id"].str.strip() == ""
        result.loc[empty_mask, "sample_id"] = result.loc[
            empty_mask, "rel_path"
        ].map(stable_sample_id_from_rel_path)

    dup_mask = result["sample_id"].duplicated(keep=False)
    if dup_mask.any():
        result.loc[dup_mask, "sample_id"] = result.loc[
            dup_mask, "rel_path"
        ].map(stable_sample_id_from_rel_path)

    if result["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id still exists after rel_path repair.")

    return result

def validate_train_only_frame(
    df: pd.DataFrame,
    *,
    source_name: str,
    require_marker: bool = True,
) -> None:
    """Check only whether the split is train."""

    if "split" not in df.columns:
        if require_marker:
            raise ValueError(
                f"Cannot verify that {source_name} is train-only: missing 'split' column."
            )
        return

    values = df["split"].fillna("").astype(str).str.strip().str.lower()
    bad_mask = values != "train"

    if bad_mask.any():
        bad_values = sorted(values.loc[bad_mask].unique().tolist())
        examples = df.loc[bad_mask].head(10).to_dict(orient="records")
        raise ValueError(
            f"Calibration source must be train-only, but column 'split' in "
            f"{source_name} contains non-train values: {bad_values}. "
            f"Examples: {examples}"
        )










def standardize_train_csv(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    frame["file_name"] = frame.apply(get_file_name, axis=1)
    frame["rel_path"] = frame.apply(get_rel_path, axis=1)
    frame["label"] = frame.apply(get_label, axis=1)

    if "ebird_code" not in frame.columns:
        frame["ebird_code"] = frame["label"]

    frame = ensure_shot_group(frame)

    if "split" not in frame.columns:
        frame["split"] = "train"
    else:
        frame["split"] = frame["split"].fillna("train").astype(str)

    if "source_split" not in frame.columns:
        frame["source_split"] = pd.NA

    if "orig_index" not in frame.columns:
        frame["orig_index"] = frame.index.astype(int)

    for optional_col in ("filepath", "file_id", "label_id", "split_seed", "file_size"):
        if optional_col not in frame.columns:
            frame[optional_col] = pd.NA

    if "train_label_n" not in frame.columns:
        frame["train_label_n"] = frame["label_n"]

    frame = ensure_sample_id(frame)
    frame = ensure_file_hash(frame)

    return frame

def sample_group_balanced(
    group_df: pd.DataFrame,
    *,
    n: int,
    seed: int,
    group_name: str,
    max_samples_per_label: Optional[int] = None,
) -> pd.DataFrame:
    """Balanced sampling within one bucket."""

    if n <= 0:
        raise ValueError(f"samples_per_group must be positive, got {n}")

    if group_df.empty:
        return group_df.copy()

    part = group_df.copy()
    part["label"] = part["label"].astype(str)

    if len(part) <= n and max_samples_per_label is None:
        return part.copy()

    labels = sorted(part["label"].unique().tolist())
    group_seed = stable_uint32_seed(seed, "group", group_name)

    if len(labels) > n:
        # If the budget is smaller than the label count, choose anchored labels first.
        selected_labels = (
            pd.Series(labels)
            .sample(n=n, random_state=stable_uint32_seed(group_seed, "label_subset"))
            .astype(str)
            .tolist()
        )
    else:
        selected_labels = labels

    selected_indices: list[Any] = []

    for label in selected_labels:
        label_rows = part[part["label"] == label]
        one = label_rows.sample(
            n=1,
            random_state=stable_uint32_seed(group_seed, "anchor", label),
        )
        selected_indices.extend(one.index.tolist())

    selected = part.loc[selected_indices].copy()
    remaining_slots = int(n) - len(selected)

    if remaining_slots <= 0:
        return selected.reset_index(drop=True)

    remaining = part.drop(index=selected_indices)

    # many per-label cap: one anchor is already selected, so the fill pool gets cap - 1.
    if max_samples_per_label is not None:
        if max_samples_per_label <= 0:
            raise ValueError(
                f"max_samples_per_label must be positive or None, got {max_samples_per_label}"
            )

        capped_parts = []
        selected_label_counts = selected["label"].value_counts().to_dict()

        for label, label_rows in remaining.groupby("label", sort=True):
            already = int(selected_label_counts.get(str(label), 0))
            allowance = int(max_samples_per_label) - already
            if allowance <= 0:
                continue
            if len(label_rows) <= allowance:
                capped_parts.append(label_rows)
            else:
                capped_parts.append(
                    label_rows.sample(
                        n=allowance,
                        random_state=stable_uint32_seed(group_seed, "cap", label),
                    )
                )

        if capped_parts:
            remaining = pd.concat(capped_parts, axis=0)
        else:
            remaining = remaining.iloc[0:0]

    if remaining.empty:
        return selected.reset_index(drop=True)

    fill_n = min(remaining_slots, len(remaining))
    fill = remaining.sample(
        n=fill_n,
        random_state=stable_uint32_seed(group_seed, "fill"),
    )

    return pd.concat([selected, fill], axis=0, ignore_index=True)


def build_calibration_table(
    train_df: pd.DataFrame,
    *,
    seed: int,
    samples_per_group: int,
    min_samples_per_group: int,
    allow_missing_groups: bool,
    many_max_samples_per_label: Optional[int],
) -> pd.DataFrame:
    validate_train_only_frame(
        train_df,
        source_name="train_df",
        require_marker=True,
    )
    frame = standardize_train_csv(train_df)
    validate_train_only_frame(
        frame,
        source_name="standardized train_df",
        require_marker=True,
    )

    rows = []
    for group in bucket_order():
        part = frame[frame["shot_group"] == group].copy()

        if part.empty:
            if allow_missing_groups:
                continue
            raise ValueError(
                f"train.csv has no samples for shot_group={group!r}. "
                "If this is expected, pass --allow-missing-groups."
            )

        if len(part) < min_samples_per_group:
            raise ValueError(
                f"shot_group={group!r} has only {len(part)} samples, "
                f"less than min_samples_per_group={min_samples_per_group}."
            )

        per_label_cap = None
        if group == "many" and many_max_samples_per_label is not None:
            per_label_cap = int(many_max_samples_per_label)

        group_seed = stable_uint32_seed(seed, "group", group)
        sampled = sample_group_balanced(
            part,
            n=int(samples_per_group),
            seed=int(seed),
            group_name=str(group),
            max_samples_per_label=per_label_cap,
        )
        sampled["calibration_group_seed"] = int(group_seed)
        sampled["calibration_group_budget"] = int(samples_per_group)
        sampled["calibration_many_max_samples_per_label"] = (
            int(many_max_samples_per_label)
            if group == "many" and many_max_samples_per_label is not None
            else pd.NA
        )
        rows.append(sampled)

    if not rows:
        raise RuntimeError("No calibration samples selected.")

    calibration = pd.concat(rows, axis=0, ignore_index=True)

    calibration = calibration.reset_index(drop=True)
    calibration.insert(0, "calibration_index", range(len(calibration)))

    calibration["calibration_seed"] = int(seed)
    calibration["calibration_role"] = "quant_calibration"

    extra_cols = [
        "calibration_seed",
        "calibration_role",
    ]

    output_cols = REQUIRED_OUTPUT_COLUMNS + extra_cols

    for col in output_cols:
        if col not in calibration.columns:
            calibration[col] = pd.NA

    calibration = calibration[output_cols].reset_index(drop=True)
    return calibration


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def build_coverage_report(
    *,
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> Dict[str, Any]:
    train_labels = set(train_df["label"].astype(str).unique().tolist())
    calibration_labels = set(calibration_df["label"].astype(str).unique().tolist())

    train_total_labels = len(train_labels)
    covered_total_labels = len(calibration_labels)

    train_group_label_counts = (
        train_df.groupby("shot_group")["label"]
        .nunique()
        .reindex(bucket_order(), fill_value=0)
        .astype(int)
        .to_dict()
    )
    calibration_group_label_counts = (
        calibration_df.groupby("shot_group")["label"]
        .nunique()
        .reindex(bucket_order(), fill_value=0)
        .astype(int)
        .to_dict()
    )
    calibration_group_sample_counts = (
        calibration_df["shot_group"]
        .value_counts()
        .reindex(bucket_order(), fill_value=0)
        .astype(int)
        .to_dict()
    )
    train_group_sample_counts = (
        train_df["shot_group"]
        .value_counts()
        .reindex(bucket_order(), fill_value=0)
        .astype(int)
        .to_dict()
    )

    group_label_coverage_ratio_within_group = {
        group: _safe_ratio(
            int(calibration_group_label_counts.get(group, 0)),
            int(train_group_label_counts.get(group, 0)),
        )
        for group in bucket_order()
    }

    group_label_coverage_ratio_of_train_total = {
        group: _safe_ratio(
            int(calibration_group_label_counts.get(group, 0)),
            int(train_total_labels),
        )
        for group in bucket_order()
    }

    covered_labels_by_group = {
        group: sorted(
            calibration_df.loc[
                calibration_df["shot_group"] == group,
                "label",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        for group in bucket_order()
    }

    return {
        "calibration_num_samples": int(len(calibration_df)),
        "calibration_group_sample_counts": calibration_group_sample_counts,
        "train_group_sample_counts": train_group_sample_counts,
        "train_num_labels": int(train_total_labels),
        "calibration_covered_num_labels": int(covered_total_labels),
        "calibration_label_coverage_ratio_of_train": _safe_ratio(
            covered_total_labels,
            train_total_labels,
        ),
        "train_group_label_counts": train_group_label_counts,
        "calibration_group_label_counts": calibration_group_label_counts,
        "group_label_coverage_ratio_within_group": group_label_coverage_ratio_within_group,
        "group_label_coverage_ratio_of_train_total": group_label_coverage_ratio_of_train_total,
        "covered_labels_by_group": covered_labels_by_group,
    }


def write_manifest(
    *,
    dataset_name: str,
    train_csv: Path,
    calibration_csv: Path,
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    seed: int,
    samples_per_group: int,
    many_max_samples_per_label: Optional[int],
) -> Path:
    manifest_path = calibration_csv.with_suffix(".manifest.json")

    label_counts = (
        calibration_df["label"]
        .astype(str)
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    sample_ids = calibration_df["sample_id"].astype(str).tolist()
    file_hashes = calibration_df["file_hash"].astype(str).tolist()
    calibration_indices = calibration_df["calibration_index"].astype(str).tolist()

    group_seeds = {
        group: int(stable_uint32_seed(seed, "group", group))
        for group in bucket_order()
    }

    coverage = build_coverage_report(
        train_df=train_df,
        calibration_df=calibration_df,
    )

    payload = {
        "dataset": dataset_name,
        "source_train_csv": str(train_csv),
        "source_train_csv_sha256": sha256_file(train_csv),
        "output_calibration_csv": str(calibration_csv),
        "calibration_csv_sha256": sha256_file(calibration_csv),
        "seed": int(seed),
        "group_seeds": group_seeds,
        "samples_per_group": int(samples_per_group),
        "many_max_samples_per_label": (
            int(many_max_samples_per_label)
            if many_max_samples_per_label is not None
            else None
        ),
        "bucket_order": list(bucket_order()),
        "bucket_thresholds": bucket_thresholds_metadata(),
        "bucket_aliases": config.BUCKET_ALIASES,
        "coverage": coverage,
        "label_counts": label_counts,
        "sample_id_sha256": sha256_text_lines(sample_ids),
        "file_hash_sha256": sha256_text_lines(file_hashes),
        "calibration_index_sha256": sha256_text_lines(calibration_indices),
        "sample_ids": sample_ids,
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return manifest_path


def main() -> None:
    args = parse_args()

    dataset_cfg = config.DATASETS[args.dataset]
    train_csv = Path(dataset_cfg["TRAIN_CSV"])
    split_dir = Path(dataset_cfg["SPLIT_DIR"])
    output_csv = split_dir / args.output_name

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    if int(args.samples_per_group) <= 0:
        raise ValueError("--samples-per-group must be positive")
    if int(args.min_samples_per_group) <= 0:
        raise ValueError("--min-samples-per-group must be positive")

    many_max_samples_per_label = int(args.many_max_samples_per_label)
    if many_max_samples_per_label <= 0:
        many_max_samples_per_label_or_none: Optional[int] = None
    else:
        many_max_samples_per_label_or_none = many_max_samples_per_label

    split_dir.mkdir(parents=True, exist_ok=True)

    raw_train_df = pd.read_csv(train_csv)
    validate_train_only_frame(
        raw_train_df,
        source_name=str(train_csv),
        require_marker=True,
    )
    standardized_train_df = standardize_train_csv(raw_train_df)
    validate_train_only_frame(
        standardized_train_df,
        source_name=f"standardized {train_csv}",
        require_marker=True,
    )

    calibration_df = build_calibration_table(
        standardized_train_df,
        seed=int(args.seed),
        samples_per_group=int(args.samples_per_group),
        min_samples_per_group=int(args.min_samples_per_group),
        allow_missing_groups=bool(args.allow_missing_groups),
        many_max_samples_per_label=many_max_samples_per_label_or_none,
    )

    calibration_df.to_csv(output_csv, index=False, encoding="utf-8")
    manifest_path = write_manifest(
        dataset_name=args.dataset,
        train_csv=train_csv,
        calibration_csv=output_csv,
        train_df=standardized_train_df,
        calibration_df=calibration_df,
        seed=int(args.seed),
        samples_per_group=int(args.samples_per_group),
        many_max_samples_per_label=many_max_samples_per_label_or_none,
    )

    coverage = build_coverage_report(
        train_df=standardized_train_df,
        calibration_df=calibration_df,
    )

   # print(f"[OK] dataset={args.dataset}")
   # print(f"[OK] source={train_csv}")
    print(f"[OK] saved calibration={output_csv}")
    print(f"[OK] saved manifest={manifest_path}")
    #print("[OK] calibration group sample counts:")
    #print(calibration_df["shot_group"].value_counts().reindex(bucket_order(), fill_value=0))
    #print("[OK] label coverage:")
    #print(f"covered={coverage['calibration_covered_num_labels']} / "f"train={coverage['train_num_labels']} "f"ratio={coverage['calibration_label_coverage_ratio_of_train']:.6f}")


if __name__ == "__main__":
    main()
