from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline.manifest import canonical_json_hash, file_sha256, path_str
from src.innovation.stage2_class_prior_adjustment import (
    BASELINE_NAME,
    INNOVATION_NAME,
    METHOD_VARIANT,
    MODEL_KIND,
    STAGE2_CLASS_PRIOR_PROTOCOL,
    build_train_label_buckets,
    export_stage2_class_prior_checkpoint,
    parse_args as parse_stage2_args,
    run_stage2_class_prior_adjustment,
    validate_split_bucket_coverage,
)
from src.innovation.stage2_eval_metrics import STAGE2_SELECTION_METRIC
from src.train.train_beats_classifier_head import dataset_paths


SWEEP_PROTOCOL_VERSION = "innovation_stage2_class_prior_tau_sweep.v1"
DEFAULT_TAU_MAX_VALUES = (0.3, 0.4, 0.5)
SUMMARY_COLUMNS = [
    "candidate_index",
    "selected",
    "tau_min",
    "tau_max",
    "seed",
    "tau_dir",
    "stage2_checkpoint",
    "stage2_checkpoint_sha256",
    "stage2_eval_metrics",
    "stage2_manifest",
    "selection_metric",
    "selection_metric_value",
    "eval_accuracy",
    "stage2_eval_many_accuracy",
    "stage2_eval_tail_accuracy",
    "best_stage_epoch",
    "best_global_epoch",
    "run_manifest_hash",
    "near_tie_note",
    "final_test_exported",
    "final_export_splits",
    "final_dir",
    "final_checkpoint",
    "final_metrics_csv",
    "final_group_metrics_csv",
    "final_geometry_metrics_csv",
    "final_export_manifest",
    "final_export_run_identity_hash",
]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(f"must be a non-negative finite float, got {value!r}")
    return parsed


def _parse_tau_values(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("tau list must contain at least one value")
    result: list[float] = []
    for part in parts:
        parsed = float(part)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise argparse.ArgumentTypeError(f"tau_max must be positive and finite, got {part!r}")
        result.append(float(parsed))
    return result


def _tau_text(value: float) -> str:
    return format(float(value), "g")


def _tau_dir_name(value: float) -> str:
    text = _tau_text(value).replace("-", "m").replace(".", "p")
    return f"tau_max_{text}"


def _selected_final_dir_name(value: float) -> str:
    return f"selected_{_tau_dir_name(value)}_final"


def _tau_key(value: float) -> str:
    return _tau_text(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an eval-only tau_max sweep for Stage 2 class-prior adjustment, "
            "then export eval+test only for the eval-selected tau."
        )
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stage1-checkpoint", default=None, help="Default: <dataset_root>/innovation/two_stage/seed_<seed>/stage1_checkpoint.pt.")
    parser.add_argument("--baseline-root", default=None, help="Forwarded to stage2_class_prior_adjustment.py.")
    parser.add_argument("--two-stage-root", default=None, help="Forwarded to stage2_class_prior_adjustment.py.")
    parser.add_argument("--reference-root", default=None, help="Default: <dataset_root>/baseline/teacher/seed_<seed>.")
    parser.add_argument("--sweep-root", default=None, help="Default: <dataset_root>/innovation/stage2_class/tau_sweep.")
    parser.add_argument("--tau-max-values", type=_parse_tau_values, default=list(DEFAULT_TAU_MAX_VALUES))
    parser.add_argument("--tau-min", type=_nonnegative_float, default=0.0)
    parser.add_argument("--stage2-epochs", type=_positive_int, default=10)
    parser.add_argument("--stage2-classifier-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--export-name", default=INNOVATION_NAME)
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=None)
    parser.add_argument("--token-layers", default=None)
    parser.add_argument("--export-all-token-layers", action="store_true")
    parser.add_argument("--max-export-payload-gib", type=float, default=None)
    parser.add_argument("--selection-metric", default=STAGE2_SELECTION_METRIC)
    parser.add_argument("--near-tie-delta", type=_nonnegative_float, default=1e-4)
    parser.add_argument("--summary-csv", default=None, help="Default: <sweep_seed_dir>/tau_sweep_eval_summary.csv.")
    parser.add_argument("--manifest-path", default=None, help="Default: <sweep_seed_dir>/tau_sweep_manifest.json.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _sweep_seed_dir(args: argparse.Namespace) -> Path:
    paths = dataset_paths(args.dataset, None)
    sweep_root = Path(args.sweep_root) if args.sweep_root else paths.root / "innovation" / INNOVATION_NAME / "tau_sweep"
    return sweep_root / f"seed_{int(args.seed)}"


def _summary_csv(args: argparse.Namespace, sweep_seed_dir: Path) -> Path:
    return Path(args.summary_csv) if args.summary_csv else sweep_seed_dir / "tau_sweep_eval_summary.csv"


def _manifest_path(args: argparse.Namespace, sweep_seed_dir: Path) -> Path:
    return Path(args.manifest_path) if args.manifest_path else sweep_seed_dir / "tau_sweep_manifest.json"


def _fixed_stage1_checkpoint(args: argparse.Namespace) -> Path:
    root = dataset_paths(args.dataset, None).root
    two_stage_root = Path(args.two_stage_root) if args.two_stage_root else root / "innovation" / "two_stage"
    return Path(args.stage1_checkpoint) if args.stage1_checkpoint else two_stage_root / f"seed_{int(args.seed)}" / "stage1_checkpoint.pt"


def _fixed_reference_root(args: argparse.Namespace) -> Path:
    root = dataset_paths(args.dataset, None).root
    baseline_root = Path(args.baseline_root) if args.baseline_root else root / "baseline"
    return Path(args.reference_root) if args.reference_root else baseline_root / "teacher" / f"seed_{int(args.seed)}"


def _append_optional(argv: list[str], flag: str, value: Any) -> None:
    if value is not None and str(value).strip() != "":
        argv.extend([flag, str(value)])


def _stage2_argv(args: argparse.Namespace, *, tau_max: float, tau_dir: Path) -> list[str]:
    argv: list[str] = [
        "--seed",
        str(int(args.seed)),
        "--seed-dir",
        str(tau_dir),
        "--tau-min",
        _tau_text(float(args.tau_min)),
        "--tau-max",
        _tau_text(float(tau_max)),
        "--stage2-epochs",
        str(int(args.stage2_epochs)),
        "--export-name",
        str(args.export_name),
        "--skip-test-during-sweep",
    ]
    _append_optional(argv, "--dataset", args.dataset)
    _append_optional(argv, "--stage1-checkpoint", args.stage1_checkpoint)
    _append_optional(argv, "--baseline-root", args.baseline_root)
    _append_optional(argv, "--two-stage-root", args.two_stage_root)
    _append_optional(argv, "--reference-root", args.reference_root)
    _append_optional(argv, "--stage2-classifier-lr", args.stage2_classifier_lr)
    _append_optional(argv, "--weight-decay", args.weight_decay)
    _append_optional(argv, "--warmup-ratio", args.warmup_ratio)
    _append_optional(argv, "--batch-size", args.batch_size)
    _append_optional(argv, "--eval-batch-size", args.eval_batch_size)
    _append_optional(argv, "--num-workers", args.num_workers)
    _append_optional(argv, "--device", args.device)
    _append_optional(argv, "--token-export", args.token_export)
    _append_optional(argv, "--token-layers", args.token_layers)
    _append_optional(argv, "--max-export-payload-gib", args.max_export_payload_gib)
    if bool(args.export_all_token_layers):
        argv.append("--export-all-token-layers")
    return argv


def _read_single_csv_row(path: str | Path) -> dict[str, str]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"expected CSV is missing: {target}")
    with target.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one CSV row in {target}, got {len(rows)}")
    return rows[0]


def _float_value(row: Mapping[str, Any], key: str, *, default: float = float("nan")) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return float(default)
    parsed = float(value)
    if math.isinf(parsed):
        raise ValueError(f"{key} must be finite or NaN, got {value!r}")
    return parsed


def _selection_metric_value(row: Mapping[str, Any], selection_metric: str) -> float:
    candidates = [
        "stage2_selection_metric_value",
        selection_metric,
        f"stage2_{selection_metric}",
    ]
    for key in candidates:
        if key in row and str(row.get(key)).strip() != "":
            return _float_value(row, key)
    raise KeyError(f"stage2 eval metrics row missing selection metric {selection_metric!r}")


def _candidate_summary_row(
    *,
    index: int,
    tau_min: float,
    tau_max: float,
    tau_dir: Path,
    result: Mapping[str, Any],
    eval_row: Mapping[str, Any],
    selection_metric: str,
) -> dict[str, Any]:
    resolved = result.get("resolved")
    manifest = result.get("manifest")
    if not isinstance(resolved, Mapping):
        raise KeyError("stage2 runner result missing resolved paths")
    if not isinstance(manifest, Mapping):
        raise KeyError("stage2 runner result missing manifest")
    stage2_checkpoint = Path(resolved["stage2_checkpoint"])
    return {
        "candidate_index": int(index),
        "selected": False,
        "tau_min": float(tau_min),
        "tau_max": float(tau_max),
        "seed": int(eval_row.get("seed") or 0),
        "tau_dir": path_str(tau_dir),
        "stage2_checkpoint": path_str(stage2_checkpoint),
        "stage2_checkpoint_sha256": file_sha256(stage2_checkpoint),
        "stage2_eval_metrics": path_str(resolved["stage2_eval_metrics"]),
        "stage2_manifest": path_str(resolved["stage2_manifest"]),
        "selection_metric": str(selection_metric),
        "selection_metric_value": _selection_metric_value(eval_row, selection_metric),
        "eval_accuracy": _float_value(eval_row, "stage2_eval_accuracy"),
        "stage2_eval_many_accuracy": _float_value(eval_row, "stage2_eval_many_accuracy"),
        "stage2_eval_tail_accuracy": _float_value(eval_row, "stage2_eval_tail_accuracy"),
        "best_stage_epoch": int(float(eval_row.get("stage2_epoch") or 0)),
        "best_global_epoch": int(float(eval_row.get("global_epoch") or 0)),
        "run_manifest_hash": str(manifest.get("run_manifest_hash") or ""),
        "near_tie_note": "",
        "final_test_exported": False,
        "final_export_splits": "",
        "final_dir": "",
        "final_checkpoint": "",
        "final_metrics_csv": "",
        "final_group_metrics_csv": "",
        "final_geometry_metrics_csv": "",
        "final_export_manifest": "",
        "final_export_run_identity_hash": "",
    }


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    fieldnames = list(SUMMARY_COLUMNS)
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _write_combined_export_table(split_results: Mapping[str, Any], *, table_key: str, path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for split, result in split_results.items():
        paths = result.get("paths")
        if not isinstance(paths, Mapping) or table_key not in paths:
            raise KeyError(f"final export split result missing {table_key!r} path for split={split!r}")
        source_csv = Path(paths[table_key])
        with source_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(str(name))
            for row in reader:
                out = dict(row)
                out.setdefault("split", str(split))
                rows.append(out)
    if not rows:
        raise RuntimeError(f"final export produced no rows for table={table_key!r}")
    if "split" not in fieldnames:
        fieldnames.insert(0, "split")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _select_best_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    finite = [
        row
        for row in rows
        if math.isfinite(float(row["selection_metric_value"]))
    ]
    if not finite:
        raise RuntimeError("tau sweep produced no finite eval selection metric")
    return max(
        finite,
        key=lambda row: (
            float(row["selection_metric_value"]),
            -int(row["candidate_index"]),
        ),
    )


def _near_tie_note(rows: Sequence[Mapping[str, Any]], *, best_row: Mapping[str, Any], delta: float) -> str:
    finite = sorted(
        (row for row in rows if math.isfinite(float(row["selection_metric_value"]))),
        key=lambda row: float(row["selection_metric_value"]),
        reverse=True,
    )
    if len(finite) < 2:
        return ""
    best_value = float(best_row["selection_metric_value"])
    second = finite[1]
    second_value = float(second["selection_metric_value"])
    if best_value - second_value > float(delta):
        return ""
    return (
        f"near_tie_delta<={float(delta):g}; "
        f"second_tau_max={second['tau_max']}; "
        f"best_tail={best_row.get('stage2_eval_tail_accuracy', '')}; "
        f"best_many={best_row.get('stage2_eval_many_accuracy', '')}; "
        f"second_tail={second.get('stage2_eval_tail_accuracy', '')}; "
        f"second_many={second.get('stage2_eval_many_accuracy', '')}"
    )


def _dry_run_plan(args: argparse.Namespace, *, sweep_seed_dir: Path, summary_csv: Path, manifest_path: Path) -> dict[str, Any]:
    candidates = []
    for tau_max in args.tau_max_values:
        tau_dir = sweep_seed_dir / _tau_dir_name(float(tau_max))
        candidates.append(
            {
                "tau_max": float(tau_max),
                "tau_dir": path_str(tau_dir),
                "stage2_argv": _stage2_argv(args, tau_max=float(tau_max), tau_dir=tau_dir),
            }
        )
    return {
        "protocol_version": SWEEP_PROTOCOL_VERSION,
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "seed": int(args.seed),
        "tau_min": float(args.tau_min),
        "tau_max_values": [float(item) for item in args.tau_max_values],
        "stage2_epochs": int(args.stage2_epochs),
        "selection_metric": str(args.selection_metric),
        "sweep_seed_dir": path_str(sweep_seed_dir),
        "summary_csv": path_str(summary_csv),
        "manifest_path": path_str(manifest_path),
        "fixed_inputs": {
            "stage1_checkpoint": path_str(_fixed_stage1_checkpoint(args)),
            "reference_root": path_str(_fixed_reference_root(args)),
        },
        "candidates": candidates,
        "final_dir_pattern": path_str(sweep_seed_dir / "selected_tau_max_<value>_final"),
        "final_test_policy": "export eval+test only for the eval-selected tau_max checkpoint",
    }


def run_stage2_tau_sweep(args: argparse.Namespace) -> dict[str, Any]:
    if float(args.tau_min) < 0.0:
        raise ValueError("tau_min must be non-negative")
    for tau_max in args.tau_max_values:
        if float(tau_max) < float(args.tau_min):
            raise ValueError(f"tau_max must be >= tau_min, got tau_min={args.tau_min}, tau_max={tau_max}")

    dataset = dataset_paths(args.dataset, None)
    sweep_seed_dir = _sweep_seed_dir(args)
    summary_csv = _summary_csv(args, sweep_seed_dir)
    manifest_path = _manifest_path(args, sweep_seed_dir)
    if bool(args.dry_run):
        plan = _dry_run_plan(args, sweep_seed_dir=sweep_seed_dir, summary_csv=summary_csv, manifest_path=manifest_path)
        print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))
        return {"dry_run": True, "plan": plan}

    candidate_rows: list[dict[str, Any]] = []
    candidate_results: dict[str, dict[str, Any]] = {}
    for index, tau_max in enumerate(args.tau_max_values, start=1):
        tau_max = float(tau_max)
        tau_dir = sweep_seed_dir / _tau_dir_name(tau_max)
        stage2_args = parse_stage2_args(_stage2_argv(args, tau_max=tau_max, tau_dir=tau_dir))
        result = run_stage2_class_prior_adjustment(stage2_args)
        eval_row = _read_single_csv_row(result["resolved"]["stage2_eval_metrics"])
        summary_row = _candidate_summary_row(
            index=index,
            tau_min=float(args.tau_min),
            tau_max=tau_max,
            tau_dir=tau_dir,
            result=result,
            eval_row=eval_row,
            selection_metric=str(args.selection_metric),
        )
        candidate_rows.append(summary_row)
        candidate_results[_tau_key(tau_max)] = {
            "args": stage2_args,
            "result": result,
            "tau_dir": tau_dir,
        }

    best_row = dict(_select_best_row(candidate_rows))
    best_tau_max = float(best_row["tau_max"])
    near_tie_note = _near_tie_note(candidate_rows, best_row=best_row, delta=float(args.near_tie_delta))
    for row in candidate_rows:
        selected = float(row["tau_max"]) == best_tau_max
        row["selected"] = bool(selected)
        row["near_tie_note"] = near_tie_note if selected else ""
    _write_csv(summary_csv, candidate_rows)

    best_entry = candidate_results[_tau_key(best_tau_max)]
    stage2_args = best_entry["args"]
    result = best_entry["result"]
    resolved = result["resolved"]
    tau_dir = Path(best_entry["tau_dir"])
    final_dir = sweep_seed_dir / _selected_final_dir_name(best_tau_max)
    final_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = final_dir / "stage2_checkpoint.pt"
    shutil.copy2(Path(resolved["stage2_checkpoint"]), final_checkpoint)
    final_metrics_csv = final_dir / "stage2_class_metrics.csv"
    final_group_metrics_csv = final_dir / "stage2_group_metrics.csv"
    final_geometry_metrics_csv = final_dir / "stage2_geometry_metrics.csv"
    final_summary_csv = final_dir / "tau_selection_summary.csv"
    final_export_manifest = final_dir / "final_export_manifest.json"
    label_buckets = build_train_label_buckets(dataset.train_csv)
    validate_split_bucket_coverage(dataset.eval_csv, label_buckets, split_name="eval")
    validate_split_bucket_coverage(dataset.test_csv, label_buckets, split_name="test")
    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    final_resolved = dict(resolved)
    final_resolved["export_root"] = final_dir
    final_export = export_stage2_class_prior_checkpoint(
        args=stage2_args,
        resolved=final_resolved,
        checkpoint_path=final_checkpoint,
        calibration_hash=None,
        label_buckets=label_buckets,
        device=device,
        splits=("eval", "test"),
        export_root=final_dir,
        export_name=str(stage2_args.export_name),
        metrics_csv=final_metrics_csv,
    )
    _write_combined_export_table(final_export["split_results"], table_key="group_metrics", path=final_group_metrics_csv)
    _write_combined_export_table(final_export["split_results"], table_key="geometry_metrics", path=final_geometry_metrics_csv)
    for row in candidate_rows:
        if float(row["tau_max"]) != best_tau_max:
            continue
        row["final_test_exported"] = True
        row["final_export_splits"] = "eval,test"
        row["final_dir"] = path_str(final_dir)
        row["final_checkpoint"] = path_str(final_checkpoint)
        row["final_metrics_csv"] = path_str(final_metrics_csv)
        row["final_group_metrics_csv"] = path_str(final_group_metrics_csv)
        row["final_geometry_metrics_csv"] = path_str(final_geometry_metrics_csv)
        row["final_export_manifest"] = path_str(final_export_manifest)
        row["final_export_run_identity_hash"] = str(final_export["export_run_identity_hash"])
    _write_csv(summary_csv, candidate_rows)
    shutil.copy2(summary_csv, final_summary_csv)

    selected_summary = next(row for row in candidate_rows if float(row["tau_max"]) == best_tau_max)
    final_manifest = {
        "protocol_version": SWEEP_PROTOCOL_VERSION,
        "stage2_protocol_version": STAGE2_CLASS_PRIOR_PROTOCOL,
        "artifact_type": "stage2_class_prior_tau_final_export",
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "sweep_mode": False,
        "selection_split": "eval",
        "export_splits": ["eval", "test"],
        "final_export_splits": ["eval", "test"],
        "test_split_touched": True,
        "seed": int(args.seed),
        "tau_min": float(args.tau_min),
        "tau_max": best_tau_max,
        "selection_metric": str(args.selection_metric),
        "selection_metric_value": float(selected_summary["selection_metric_value"]),
        "source_candidate_dir": path_str(tau_dir),
        "source_candidate_checkpoint": path_str(resolved["stage2_checkpoint"]),
        "source_candidate_checkpoint_sha256": file_sha256(resolved["stage2_checkpoint"]),
        "stage2_checkpoint": path_str(final_checkpoint),
        "stage2_checkpoint_sha256": file_sha256(final_checkpoint),
        "stage2_class_metrics": path_str(final_metrics_csv),
        "stage2_group_metrics": path_str(final_group_metrics_csv),
        "stage2_geometry_metrics": path_str(final_geometry_metrics_csv),
        "tau_selection_summary": path_str(final_summary_csv),
        "tau_sweep_eval_summary": path_str(summary_csv),
        "export_run_identity_hash": final_export["export_run_identity_hash"],
        "split_results": final_export["split_results"],
        "artifact_sha256": {
            "stage2_checkpoint.pt": file_sha256(final_checkpoint),
            "stage2_class_metrics.csv": file_sha256(final_metrics_csv),
            "stage2_group_metrics.csv": file_sha256(final_group_metrics_csv),
            "stage2_geometry_metrics.csv": file_sha256(final_geometry_metrics_csv),
            "tau_selection_summary.csv": file_sha256(final_summary_csv),
            **{
                key: value
                for split_result in final_export["split_results"].values()
                for key, value in dict(split_result.get("artifact_sha256", {})).items()
            },
        },
    }
    _write_json(final_export_manifest, final_manifest)

    fixed_start_checkpoint = Path(resolved["stage1_checkpoint"])
    manifest_identity = {
        "protocol_version": SWEEP_PROTOCOL_VERSION,
        "dataset_name": dataset.name,
        "seed": int(args.seed),
        "stage1_checkpoint_sha256": file_sha256(fixed_start_checkpoint),
        "tau_min": float(args.tau_min),
        "tau_max_values": [float(item) for item in args.tau_max_values],
        "stage2_epochs": int(args.stage2_epochs),
        "selection_metric": str(args.selection_metric),
        "best_tau_max": best_tau_max,
        "method_variant": METHOD_VARIANT,
    }
    manifest = {
        **manifest_identity,
        "sweep_protocol_version": SWEEP_PROTOCOL_VERSION,
        "stage2_protocol_version": STAGE2_CLASS_PRIOR_PROTOCOL,
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "sweep_identity_hash": canonical_json_hash(manifest_identity),
        "sweep_seed_dir": path_str(sweep_seed_dir),
        "stage1_checkpoint": path_str(fixed_start_checkpoint),
        "stage1_checkpoint_sha256": file_sha256(fixed_start_checkpoint),
        "reference_root": path_str(_fixed_reference_root(args)),
        "summary_csv": path_str(summary_csv),
        "summary_csv_sha256": file_sha256(summary_csv),
        "best_tau_max": best_tau_max,
        "best_tau_dir": path_str(tau_dir),
        "best_stage2_checkpoint": path_str(resolved["stage2_checkpoint"]),
        "best_stage2_checkpoint_sha256": file_sha256(resolved["stage2_checkpoint"]),
        "final_dir": path_str(final_dir),
        "final_stage2_checkpoint": path_str(final_checkpoint),
        "final_stage2_checkpoint_sha256": file_sha256(final_checkpoint),
        "final_export_manifest": path_str(final_export_manifest),
        "final_export_manifest_sha256": file_sha256(final_export_manifest),
        "near_tie_note": near_tie_note,
        "candidates": candidate_rows,
        "final_export": {
            "policy": "test split exported once after eval tau selection",
            "splits": ["eval", "test"],
            "tau_max": best_tau_max,
            "checkpoint": path_str(final_checkpoint),
            "metrics_csv": path_str(final_metrics_csv),
            "group_metrics_csv": path_str(final_group_metrics_csv),
            "geometry_metrics_csv": path_str(final_geometry_metrics_csv),
            "manifest": path_str(final_export_manifest),
            "export_run_identity_hash": final_export["export_run_identity_hash"],
            "split_results": final_export["split_results"],
        },
        "artifact_sha256": {
            "tau_sweep_eval_summary.csv": file_sha256(summary_csv),
            "best_stage2_checkpoint.pt": file_sha256(resolved["stage2_checkpoint"]),
            "final_stage2_checkpoint.pt": file_sha256(final_checkpoint),
            "final_stage2_class_metrics.csv": file_sha256(final_metrics_csv),
            "final_stage2_group_metrics.csv": file_sha256(final_group_metrics_csv),
            "final_stage2_geometry_metrics.csv": file_sha256(final_geometry_metrics_csv),
            "final_tau_selection_summary.csv": file_sha256(final_summary_csv),
            "final_export_manifest.json": file_sha256(final_export_manifest),
            **{
                key: value
                for split_result in final_export["split_results"].values()
                for key, value in dict(split_result.get("artifact_sha256", {})).items()
            },
        },
    }
    _write_json(manifest_path, manifest)
    print(f"[tau-sweep] summary={summary_csv}")
    print(f"[tau-sweep] best_tau_max={best_tau_max:g}")
    print(f"[tau-sweep] final_dir={final_dir}")
    print(f"[tau-sweep] final_metrics={final_metrics_csv}")
    print(f"[tau-sweep] manifest={manifest_path}")
    return {"manifest": manifest, "summary_csv": summary_csv, "best_tau_max": best_tau_max}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_stage2_tau_sweep(args)


if __name__ == "__main__":
    main()
