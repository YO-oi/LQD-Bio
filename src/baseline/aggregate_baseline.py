from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline.manifest import canonical_json_hash, file_sha256
from src.train.train_beats_classifier_head import dataset_paths
from src.utils.group_utils import AGGREGATION_FNS


FORMAL_SEEDS = (42, 3407, 2026)
BASELINES = ("fp32_teacher", "int4_student", "vanilla_kd")
BASELINE_MODEL_KINDS = {
    "fp32_teacher": "fp32_teacher",
    "int4_student": "int4_student_init",
    "vanilla_kd": "vanilla_kd",
}
TABLE_KINDS = ("main", "group", "geometry")
TABLE_OUTPUT_FILES = {
    "main": "baseline_main_mean_std.csv",
    "group": "baseline_group_mean_std.csv",
    "geometry": "baseline_geometry_mean_std.csv",
}
METRIC_SCHEMA_REQUIRED_FIELDS = (
    "model_kind",
    "split",
    "seed",
    "metric_name",
    "metric_value",
    "analysis_group",
    "metric_direction",
    "aggregation_unit",
    "aggregation_fn",
    "metric_role",
    "run_identity_hash",
    "label_mapping_hash",
    "split_hash",
    "pooling",
)
TOKEN_INDEX_REQUIRED_FIELDS = (
    "model_name",
    "split",
    "seed",
    "run_identity_hash",
    "run_manifest_path",
    "artifact_kind",
    "layer",
    "shard_id",
    "shape",
    "file_path",
    "file_sha256",
)
TOKEN_SHARD_ARTIFACT_KIND = "token_feature_shard"


def _parse_seeds(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return FORMAL_SEEDS
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Baseline Pipeline per-seed CSVs into mean/std tables.")
    parser.add_argument("--dataset", default=None, help="Dataset key from config.DATASETS; default: config.ACTIVE_DATASET.")
    parser.add_argument("--teacher-root", default=None, help="FP32 teacher baseline root; default: <dataset_root>/baseline/teacher.")
    parser.add_argument("--student-root", default=None, help="INT4 student baseline root; default: <dataset_root>/baseline/student.")
    parser.add_argument("--kd-root", default=None, help="Vanilla KD baseline root; default: <dataset_root>/baseline/kd.")
    parser.add_argument("--seeds", default=",".join(str(item) for item in FORMAL_SEEDS))
    parser.add_argument("--output-dir", default=None, help="Default: <dataset_root>/baseline/aggregate.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing seed/baseline inputs instead of failing.")
    return parser.parse_args(argv)


def _dataset_root(dataset_name: str | None) -> Path:
    return dataset_paths(dataset_name, None).root


def _stage_seed_dir(root: str | Path, seed: int) -> Path:
    path = Path(root)
    seed_name = f"seed_{int(seed)}"
    return path if path.name == seed_name else path / seed_name


def _baseline_stage_root(args: argparse.Namespace, baseline: str) -> Path:
    dataset_baseline_root = _dataset_root(args.dataset) / "baseline"
    if baseline == "fp32_teacher":
        return Path(args.teacher_root) if args.teacher_root else dataset_baseline_root / "teacher"
    if baseline == "int4_student":
        return Path(args.student_root) if args.student_root else dataset_baseline_root / "student"
    if baseline == "vanilla_kd":
        return Path(args.kd_root) if args.kd_root else dataset_baseline_root / "kd"
    raise ValueError(f"unknown baseline: {baseline!r}")


def _seed_dir_for_baseline(args: argparse.Namespace, *, seed: int, baseline: str) -> Path:
    return _stage_seed_dir(_baseline_stage_root(args, baseline), int(seed))


def _seed_dirs_for_baselines(args: argparse.Namespace, seed: int) -> dict[str, Path]:
    return {baseline: _seed_dir_for_baseline(args, seed=int(seed), baseline=baseline) for baseline in BASELINES}


def _default_output_dir(args: argparse.Namespace) -> Path:
    return _dataset_root(args.dataset) / "baseline" / "aggregate"


def _float_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _read_csv_with_schema(path: Path, *, required_fields: Sequence[str]) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(str(item) for item in (reader.fieldnames or ()))
        if not fieldnames:
            raise RuntimeError(f"metric CSV has no header and is not eligible for formal aggregate: {path}")
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            raise RuntimeError(f"metric CSV schema violation: {path} missing required field(s): {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError(f"metric CSV has no rows and is not eligible for formal aggregate: {path}")
    return fieldnames, rows


def _required_value(row: Mapping[str, Any], field: str, *, path: Path, row_number: int) -> str:
    value = row.get(field)
    text = "" if value is None else str(value).strip()
    if not text:
        raise RuntimeError(f"{path}:row={row_number} missing required value for {field!r}")
    return text


def _metric_table_paths(metrics_dir: Path, table_kind: str) -> list[Path]:
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        return []

    def is_main(path: Path) -> bool:
        name = path.name
        if name == "metrics.csv":
            return True
        if not name.endswith("_metrics.csv"):
            return False
        excluded_suffixes = ("_group_metrics.csv", "_geometry_metrics.csv", "_per_class_metrics.csv", "_retention_metrics.csv")
        return not any(name.endswith(suffix) for suffix in excluded_suffixes)

    def is_group(path: Path) -> bool:
        return path.name == "group_metrics.csv" or path.name.endswith("_group_metrics.csv")

    def is_geometry(path: Path) -> bool:
        return path.name == "geometry_metrics.csv" or path.name.endswith("_geometry_metrics.csv")

    predicates = {"main": is_main, "group": is_group, "geometry": is_geometry}
    if table_kind not in predicates:
        raise ValueError(f"unknown metric table kind: {table_kind!r}")
    return sorted(path for path in metrics_dir.glob("*.csv") if predicates[table_kind](path))


def _load_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _load_required_manifest(path: Path) -> Mapping[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"canonical run manifest is missing or invalid: {path}")
    return payload


def _canonical_manifest_path(seed_dir: Path, baseline: str) -> Path:
    return seed_dir / "manifests" / baseline / "run_manifest.json"


def _manifest_model_name(payload: Mapping[str, Any]) -> str:
    return str(payload.get("model_name") or "")


def _manifest_model_kind(payload: Mapping[str, Any]) -> str:
    return str(payload.get("model_kind") or "")


def _index_model_kind(payload: Mapping[str, Any], baseline: str) -> str:
    return _manifest_model_kind(payload).strip()


def _manifest_checkpoint_path(payload: Mapping[str, Any]) -> str:
    value = payload.get("checkpoint")
    return "" if value is None else str(value).strip()


def _manifest_checkpoint_hash(payload: Mapping[str, Any]) -> str:
    value = payload.get("checkpoint_sha256")
    return "" if value is None else str(value).strip()


def _canonical_pooling(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text == "mask_aware_mean" or text.startswith("mask_aware_mean("):
        return "mask_aware_mean"
    return text


def _manifest_allows_split(payload: Mapping[str, Any], split: str) -> bool:
    direct = payload.get("split")
    if direct is not None and str(direct).strip():
        return str(direct).strip() == split
    split_csv = payload.get("split_csv")
    if isinstance(split_csv, Mapping) and split in {str(key) for key in split_csv.keys()}:
        return True
    split_name = payload.get("split_name")
    if split_name is not None:
        names = {item.strip() for item in str(split_name).split(",") if item.strip()}
        if split in names:
            return True
    return False


def _manifest_split_hash(payload: Mapping[str, Any], split: str) -> str:
    split_hashes = payload.get("split_hashes")
    if not isinstance(split_hashes, Mapping):
        raise RuntimeError("canonical manifest missing split_hashes mapping")
    value = split_hashes.get(split)
    if value is None and len(split_hashes) == 1:
        value = next(iter(split_hashes.values()))
    text = "" if value is None else str(value).strip()
    if not text:
        raise RuntimeError(f"canonical manifest missing split hash for split={split!r}")
    return text


def _validate_manifest_payload(*, manifest_path: Path, payload: Mapping[str, Any], seed: int, baseline: str) -> None:
    manifest_baseline = _manifest_model_name(payload)
    if manifest_baseline != baseline:
        raise RuntimeError(f"{manifest_path}: manifest baseline/model_name={manifest_baseline!r} does not match expected {baseline!r}")
    manifest_model_kind = _manifest_model_kind(payload)
    expected_model_kind = BASELINE_MODEL_KINDS[str(baseline)]
    if manifest_model_kind != expected_model_kind:
        raise RuntimeError(f"{manifest_path}: manifest model_kind={manifest_model_kind!r} does not match expected {expected_model_kind!r}")
    manifest_seed = payload.get("seed")
    if manifest_seed is None or str(manifest_seed).strip() == "":
        raise RuntimeError(f"{manifest_path}: manifest missing seed")
    if int(manifest_seed) != int(seed):
        raise RuntimeError(f"{manifest_path}: seed={manifest_seed!r} does not match seed directory seed={seed}")
    for field in ("run_identity_hash", "label_mapping_hash", "pooling"):
        if not str(payload.get(field) or "").strip():
            raise RuntimeError(f"{manifest_path}: manifest missing required identity field {field!r}")
    if not isinstance(payload.get("split_hashes"), Mapping):
        raise RuntimeError(f"{manifest_path}: manifest missing split_hashes mapping")
    checkpoint_path = _manifest_checkpoint_path(payload)
    checkpoint_hash = _manifest_checkpoint_hash(payload)
    if not checkpoint_path or not checkpoint_hash:
        raise RuntimeError(f"{manifest_path}: manifest missing checkpoint/checkpoint_sha256 identity")
    current_hash = file_sha256(checkpoint_path)
    if not current_hash:
        raise RuntimeError(f"{manifest_path}: checkpoint artifact missing or unreadable: {checkpoint_path}")
    if str(current_hash) != checkpoint_hash:
        raise RuntimeError(f"{manifest_path}: checkpoint hash mismatch current={current_hash!r} manifest={checkpoint_hash!r}")


def _load_canonical_manifest(seed_dir: Path, seed: int, baseline: str) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = _canonical_manifest_path(seed_dir, baseline)
    payload = _load_required_manifest(manifest_path)
    _validate_manifest_payload(manifest_path=manifest_path, payload=payload, seed=seed, baseline=baseline)
    return manifest_path, payload


def _validate_metric_row_identity(
    *,
    row: Mapping[str, Any],
    path: Path,
    row_number: int,
    seed: int,
    baseline: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    for field in METRIC_SCHEMA_REQUIRED_FIELDS:
        _required_value(row, field, path=path, row_number=row_number)
    row_seed = int(_required_value(row, "seed", path=path, row_number=row_number))
    if row_seed != int(seed):
        raise RuntimeError(f"{path}:row={row_number} seed={row_seed} does not match seed directory seed={seed}")
    split = _required_value(row, "split", path=path, row_number=row_number)
    if not _manifest_allows_split(manifest, split):
        raise RuntimeError(f"{path}:row={row_number} split={split!r} is not represented by canonical manifest {manifest_path}")
    manifest_run_identity = str(manifest.get("run_identity_hash") or "")
    if _required_value(row, "run_identity_hash", path=path, row_number=row_number) != manifest_run_identity:
        raise RuntimeError(f"{path}:row={row_number} run_identity_hash does not match canonical manifest {manifest_path}")
    manifest_model_kind = _manifest_model_kind(manifest)
    if manifest_model_kind and _required_value(row, "model_kind", path=path, row_number=row_number) != manifest_model_kind:
        raise RuntimeError(f"{path}:row={row_number} model_kind does not match canonical manifest model_kind={manifest_model_kind!r}")
    if _required_value(row, "label_mapping_hash", path=path, row_number=row_number) != str(manifest.get("label_mapping_hash") or ""):
        raise RuntimeError(f"{path}:row={row_number} label_mapping_hash does not match canonical manifest")
    if _required_value(row, "split_hash", path=path, row_number=row_number) != _manifest_split_hash(manifest, split):
        raise RuntimeError(f"{path}:row={row_number} split_hash does not match canonical manifest split_hashes[{split!r}]")
    if _canonical_pooling(_required_value(row, "pooling", path=path, row_number=row_number)) != _canonical_pooling(manifest.get("pooling")):
        raise RuntimeError(f"{path}:row={row_number} pooling does not match canonical manifest")
    aggregation_fn = _required_value(row, "aggregation_fn", path=path, row_number=row_number)
    if aggregation_fn not in AGGREGATION_FNS:
        raise RuntimeError(f"{path}:row={row_number} aggregation_fn={aggregation_fn!r} is not registered in group_utils.AGGREGATION_FNS")
    _required_value(row, "metric_value", path=path, row_number=row_number)
    if not baseline:
        raise RuntimeError("baseline must be non-empty")


def _metric_row_duplicate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["seed"]),
        str(row["baseline"]),
        str(row["table_kind"]),
        str(row["split"]),
        str(row["metric_name"]),
        str(row["analysis_group"]),
        str(row["metric_direction"]),
        str(row["aggregation_unit"]),
        str(row["aggregation_fn"]),
        str(row["metric_role"]),
        str(row.get("reference_checkpoint_hash", "")),
        str(row.get("label_id", "")),
        str(row.get("layer_idx", "")),
        str(row.get("module_name", "")),
    )


def _aggregate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["baseline"]),
        str(row["split"]),
        str(row["analysis_group"]),
        str(row["metric_name"]),
        str(row["metric_direction"]),
        str(row["aggregation_unit"]),
        str(row["aggregation_fn"]),
        str(row["metric_role"]),
        str(row.get("reference_checkpoint_hash", "")),
        str(row.get("definition", "")),
    )


def _validate_no_duplicate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = _metric_row_duplicate_key(row)
        existing = seen.get(key)
        if existing is not None:
            raise RuntimeError(
                "duplicate metric row for the same seed/key; "
                f"first={existing.get('source_csv')} duplicate={row.get('source_csv')} key={key}"
            )
        seen[key] = row


def _validate_cross_seed_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    identities: dict[tuple[str, str], tuple[str, str, str]] = {}
    for row in rows:
        key = (str(row["baseline"]), str(row["split"]))
        value = (str(row["label_mapping_hash"]), str(row["split_hash"]), str(row["pooling"]))
        existing = identities.get(key)
        if existing is None:
            identities[key] = value
            continue
        if existing != value:
            raise RuntimeError(f"cross-seed metric identity mismatch for {key}: existing={existing}, current={value}")



def _add_kd_oer_gain_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    int4_rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    kd_rows: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("table_kind")) != "main" or str(row.get("metric_name")) != "oer_retention":
            continue
        key = (
            int(row["seed"]),
            str(row["split"]),
            str(row["analysis_group"]),
            str(row.get("reference_checkpoint_hash", "")),
        )
        baseline = str(row.get("baseline"))
        if baseline == "int4_student":
            if key in int4_rows:
                raise RuntimeError(f"duplicate INT4 OER row while computing kd_oer_gain_over_int4: key={key}")
            int4_rows[key] = row
        elif baseline == "vanilla_kd":
            if key in kd_rows:
                raise RuntimeError(f"duplicate KD OER row while computing kd_oer_gain_over_int4: key={key}")
            kd_rows[key] = row
    present_main_baselines = {str(row.get("baseline")) for row in rows if str(row.get("table_kind")) == "main"}
    if "int4_student" in present_main_baselines and not int4_rows:
        raise RuntimeError("formal aggregate requires int4_student oer_retention rows")
    if "vanilla_kd" in present_main_baselines and not kd_rows:
        raise RuntimeError("formal aggregate requires vanilla_kd oer_retention rows")
    for key, kd_row in sorted(kd_rows.items(), key=lambda item: item[0]):
        int4_row = int4_rows.get(key)
        if int4_row is None:
            raise RuntimeError(f"missing INT4 OER row for kd_oer_gain_over_int4 key={key}")
        for field in ("label_mapping_hash", "split_hash", "pooling", "reference_checkpoint_hash", "analysis_group"):
            if str(kd_row.get(field, "")) != str(int4_row.get(field, "")):
                raise RuntimeError(f"kd_oer_gain_over_int4 identity mismatch for {field}: kd={kd_row.get(field)!r} int4={int4_row.get(field)!r}")
        kd_value = _float_or_nan(kd_row.get("metric_value"))
        int4_value = _float_or_nan(int4_row.get("metric_value"))
        gain = kd_value - int4_value if math.isfinite(kd_value) and math.isfinite(int4_value) else float("nan")
        row = dict(kd_row)
        row["metric_name"] = "kd_oer_gain_over_int4"
        row["metric_value"] = gain
        row["metric_direction"] = "higher_is_better"
        row["aggregation_unit"] = "global_derived"
        row["aggregation_fn"] = "group_gap"
        row["metric_role"] = "main"
        row["definition"] = "vanilla_kd oer_retention minus int4_student oer_retention against the same FP32 teacher OER oracle"
        row["source_csv"] = f"{kd_row.get('source_csv', '')}|{int4_row.get('source_csv', '')}"
        row["derived_from_metric_name"] = "oer_retention"
        row["derived_from_baselines"] = "vanilla_kd,int4_student"
        result.append(row)
    return result

def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    seeds: dict[tuple[Any, ...], set[int]] = {}
    for row in rows:
        key = _aggregate_key(row)
        value = _float_or_nan(row["metric_value"])
        if not math.isfinite(value):
            continue
        grouped.setdefault(key, []).append(value)
        seeds.setdefault(key, set()).add(int(row["seed"]))
    out: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: item[0]):
        baseline, split, analysis_group, metric_name, metric_direction, aggregation_unit, aggregation_fn, metric_role, reference_checkpoint_hash, definition = key
        out.append(
            {
                "baseline": baseline,
                "split": split,
                "analysis_group": analysis_group,
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "aggregation_unit": aggregation_unit,
                "aggregation_fn": aggregation_fn,
                "metric_role": metric_role,
                "reference_checkpoint_hash": reference_checkpoint_hash,
                "definition": definition,
                "num_seeds": int(len(seeds.get(key, set()))),
                "num_values": int(len(values)),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
                "min": min(values),
                "max": max(values),
            }
        )
    return out


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty formal aggregate table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _metric_index_row(
    *,
    seed: int,
    baseline: str,
    table_kind: str,
    source_csv: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    splits = sorted({str(row["split"]) for row in rows})
    metric_names = sorted({str(row["metric_name"]) for row in rows})
    source_split_hashes = {str(row.get("split_hash") or "").strip() for row in rows}
    source_split_hashes.discard("")
    run_split_hashes = {_manifest_split_hash(manifest, split) for split in splits}
    return {
        "artifact_type": "metric_table",
        "seed": int(seed),
        "baseline": baseline,
        "table_kind": table_kind,
        "split": ",".join(splits),
        "source_csv": str(source_csv.resolve(strict=False)),
        "source_csv_sha256": file_sha256(source_csv),
        "row_count": int(len(rows)),
        "schema_columns": json.dumps(list(fieldnames), separators=(",", ":")),
        "schema_hash": canonical_json_hash(list(fieldnames)),
        "metric_names": json.dumps(metric_names, separators=(",", ":")),
        "run_identity_hash": manifest.get("run_identity_hash", ""),
        "run_manifest_hash": manifest.get("run_manifest_hash", ""),
        "manifest_path": str(manifest_path.resolve(strict=False)),
        "checkpoint": _manifest_checkpoint_path(manifest),
        "checkpoint_sha256": _manifest_checkpoint_hash(manifest),
        "label_mapping_hash": manifest.get("label_mapping_hash", ""),
        "source_split_hash": ",".join(sorted(source_split_hashes)),
        "run_split_hash": ",".join(sorted(run_split_hashes)),
        "pooling": manifest.get("pooling", ""),
    }


def read_seed_metric_rows(
    seed: int,
    *,
    seed_dirs: Mapping[str, Path],
    allow_missing: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str], tuple[Path, Mapping[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    metric_index: list[dict[str, Any]] = []
    manifest_cache: dict[tuple[int, str], tuple[Path, Mapping[str, Any]]] = {}
    missing_inputs: list[Path] = []
    for baseline in BASELINES:
        seed_dir = Path(seed_dirs[baseline])
        if not seed_dir.exists():
            if allow_missing:
                continue
            missing_inputs.append(seed_dir)
            continue
        metrics_dir = seed_dir / "metrics" / baseline
        if not metrics_dir.exists():
            if allow_missing:
                continue
            missing_inputs.append(metrics_dir)
            continue
        manifest_path, manifest = _load_canonical_manifest(seed_dir, seed, baseline)
        manifest_cache[(int(seed), baseline)] = (manifest_path, manifest)
        for table_kind in TABLE_KINDS:
            table_paths = _metric_table_paths(metrics_dir, table_kind)
            if not table_paths:
                missing_inputs.append(metrics_dir / f"*_{table_kind}_metrics.csv")
                continue
            for source_csv in table_paths:
                fieldnames, csv_rows = _read_csv_with_schema(source_csv, required_fields=METRIC_SCHEMA_REQUIRED_FIELDS)
                normalized_rows: list[dict[str, Any]] = []
                for row_number, row in enumerate(csv_rows, start=2):
                    _validate_metric_row_identity(
                        row=row,
                        path=source_csv,
                        row_number=row_number,
                        seed=int(seed),
                        baseline=baseline,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )
                    normalized = dict(row)
                    normalized["baseline"] = baseline
                    normalized["table_kind"] = table_kind
                    normalized["source_csv"] = str(source_csv.resolve(strict=False))
                    normalized_rows.append(normalized)
                rows.extend(normalized_rows)
                metric_index.append(
                    _metric_index_row(
                        seed=int(seed),
                        baseline=baseline,
                        table_kind=table_kind,
                        source_csv=source_csv,
                        fieldnames=fieldnames,
                        rows=normalized_rows,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )
                )
    if missing_inputs and not allow_missing:
        names = ", ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"missing formal baseline aggregate input(s): {names}")
    return rows, metric_index, manifest_cache


def _validate_token_row_manifest(
    *,
    token_index: Path,
    row: Mapping[str, Any],
    row_number: int,
    seed: int,
    baseline: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    for field in TOKEN_INDEX_REQUIRED_FIELDS:
        _required_value(row, field, path=token_index, row_number=row_number)
    row_seed = int(_required_value(row, "seed", path=token_index, row_number=row_number))
    if row_seed != int(seed):
        raise RuntimeError(f"{token_index}:row={row_number} seed={row_seed} does not match seed directory seed={seed}")
    artifact_kind = _required_value(row, "artifact_kind", path=token_index, row_number=row_number)
    if artifact_kind != TOKEN_SHARD_ARTIFACT_KIND:
        raise RuntimeError(f"{token_index}:row={row_number} artifact_kind={artifact_kind!r} is not {TOKEN_SHARD_ARTIFACT_KIND!r}")
    model_name = _required_value(row, "model_name", path=token_index, row_number=row_number)
    manifest_model_name = _manifest_model_name(manifest)
    if model_name != manifest_model_name or model_name != baseline:
        raise RuntimeError(f"{token_index}:row={row_number} model_name={model_name!r} does not match canonical manifest/baseline {manifest_model_name!r}/{baseline!r}")
    declared_manifest_path = Path(_required_value(row, "run_manifest_path", path=token_index, row_number=row_number)).resolve(strict=False)
    canonical_manifest_path = manifest_path.resolve(strict=False)
    if declared_manifest_path != canonical_manifest_path:
        raise RuntimeError(f"{token_index}:row={row_number} run_manifest_path={declared_manifest_path} does not match canonical {canonical_manifest_path}")
    if _required_value(row, "run_identity_hash", path=token_index, row_number=row_number) != str(manifest.get("run_identity_hash") or ""):
        raise RuntimeError(f"{token_index}:row={row_number} run_identity_hash does not match canonical manifest")
    split = _required_value(row, "split", path=token_index, row_number=row_number)
    if not _manifest_allows_split(manifest, split):
        raise RuntimeError(f"{token_index}:row={row_number} split={split!r} is not represented by canonical manifest")


def manifest_index_rows(
    manifest_cache: Mapping[tuple[int, str], tuple[Path, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (seed, baseline), (manifest_path, payload) in sorted(manifest_cache.items(), key=lambda item: item[0]):
        rows.append(
            {
                "artifact_type": "manifest",
                "seed": int(seed),
                "manifest_path": str(manifest_path.resolve(strict=False)),
                "stage": payload.get("stage", ""),
                "baseline": baseline,
                "model_name": _manifest_model_name(payload),
                "model_kind": _index_model_kind(payload, baseline),
                "run_identity_hash": payload.get("run_identity_hash", ""),
                "run_manifest_hash": payload.get("run_manifest_hash", ""),
                "checkpoint": _manifest_checkpoint_path(payload),
                "checkpoint_sha256": _manifest_checkpoint_hash(payload),
                "split_hash": payload.get("split_hash", ""),
                "label_mapping_hash": payload.get("label_mapping_hash", ""),
                "calibration_hash": payload.get("calibration_hash", ""),
                "qparams_csv": payload.get("qparams_csv", ""),
                "git_commit": payload.get("git_commit", ""),
            }
        )
        seed_dir = manifest_path.parents[2]
        features_dir = seed_dir / "features" / baseline
        if not features_dir.exists():
            continue
        for token_index in sorted(features_dir.rglob("token_features.index.csv")):
            fieldnames, token_rows = _read_csv_with_schema(token_index, required_fields=TOKEN_INDEX_REQUIRED_FIELDS)
            for row_number, token_row in enumerate(token_rows, start=2):
                _validate_token_row_manifest(
                    token_index=token_index,
                    row=token_row,
                    row_number=row_number,
                    seed=int(seed),
                    baseline=baseline,
                    manifest_path=manifest_path,
                    manifest=payload,
                )
                rows.append(
                    {
                        "artifact_type": "token_shard",
                        "artifact_kind": token_row.get("artifact_kind", ""),
                        "seed": int(seed),
                        "baseline": baseline,
                        "manifest_path": str(manifest_path.resolve(strict=False)),
                        "run_manifest_path": token_row.get("run_manifest_path", ""),
                        "token_index_path": str(token_index.resolve(strict=False)),
                        "token_index_schema_hash": canonical_json_hash(list(fieldnames)),
                        "stage": payload.get("stage", ""),
                        "model_name": token_row.get("model_name", ""),
                        "model_kind": _index_model_kind(payload, baseline),
                        "run_identity_hash": token_row.get("run_identity_hash", ""),
                        "run_manifest_hash": payload.get("run_manifest_hash", ""),
                        "split": token_row.get("split", ""),
                        "layer": token_row.get("layer", ""),
                        "shard_id": token_row.get("shard_id", ""),
                        "start_row": token_row.get("start_row", ""),
                        "end_row": token_row.get("end_row", ""),
                        "sample_id_start": token_row.get("sample_id_start", ""),
                        "sample_id_end": token_row.get("sample_id_end", ""),
                        "shape": token_row.get("shape", ""),
                        "dtype": token_row.get("dtype", ""),
                        "file_path": token_row.get("file_path", ""),
                        "file_sha256": token_row.get("file_sha256", ""),
                    }
                )
    return rows


def run_aggregate(args: argparse.Namespace) -> dict[str, Path]:
    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args)
    all_rows: list[dict[str, Any]] = []
    metric_index: list[dict[str, Any]] = []
    manifest_cache: dict[tuple[int, str], tuple[Path, Mapping[str, Any]]] = {}
    for seed in seeds:
        seed_dirs = _seed_dirs_for_baselines(args, int(seed))
        if not any(path.exists() for path in seed_dirs.values()):
            if bool(args.allow_missing):
                continue
            raise FileNotFoundError(f"missing seed directories for seed={int(seed)}: {seed_dirs}")
        seed_rows, seed_metric_index, seed_manifest_cache = read_seed_metric_rows(int(seed), seed_dirs=seed_dirs, allow_missing=bool(args.allow_missing))
        all_rows.extend(seed_rows)
        metric_index.extend(seed_metric_index)
        manifest_cache.update(seed_manifest_cache)
    if not all_rows:
        raise RuntimeError("no formal baseline metric rows found for aggregation")
    if not metric_index:
        raise RuntimeError("no formal baseline metric table index rows found")
    all_rows = _add_kd_oer_gain_rows(all_rows)
    _validate_no_duplicate_seed_rows(all_rows)
    _validate_cross_seed_identity(all_rows)

    output_paths = {
        "main": output_dir / TABLE_OUTPUT_FILES["main"],
        "group": output_dir / TABLE_OUTPUT_FILES["group"],
        "geometry": output_dir / TABLE_OUTPUT_FILES["geometry"],
        "manifest_index": output_dir / "baseline_manifest_index.csv",
        "metric_index": output_dir / "baseline_metric_index.csv",
    }
    for table_kind in TABLE_KINDS:
        rows = [row for row in all_rows if row["table_kind"] == table_kind]
        if not rows:
            raise RuntimeError(f"no rows found for required aggregate table kind {table_kind!r}")
        write_csv(output_paths[table_kind], aggregate_rows(rows))
    write_csv(output_paths["manifest_index"], manifest_index_rows(manifest_cache))
    write_csv(output_paths["metric_index"], metric_index)
    print(f"[baseline-aggregate] main={output_paths['main']}")
    print(f"[baseline-aggregate] group={output_paths['group']}")
    print(f"[baseline-aggregate] geometry={output_paths['geometry']}")
    print(f"[baseline-aggregate] manifest_index={output_paths['manifest_index']}")
    print(f"[baseline-aggregate] metric_index={output_paths['metric_index']}")
    return output_paths


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_aggregate(args)


if __name__ == "__main__":
    main()
