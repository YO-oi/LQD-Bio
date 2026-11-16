from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import run_vanilla_kd as vanilla_kd


DEFAULT_SEED = 2026
SPLIT_NAME = "eval"
OVERALL_ACC_GUARDRAIL = -0.005
OVERALL_MACRO_F1_GUARDRAIL = -0.005
MANY_ACC_GUARDRAIL = -0.010
SELECTION_METRIC = "tail_macro_f1"
TIE_BREAK_METRICS = ("tail_accuracy", "medium_macro_f1", "medium_accuracy")
METHOD_STATUS_VALUES = ("task_positive", "task_neutral", "task_negative", "mechanism_only", "invalid")


def _owned_flag_present(argv: Sequence[str], names: Sequence[str]) -> str | None:
    names = tuple(str(item) for item in names)
    for item in argv:
        text = str(item)
        if text in names:
            return text
        if any(text.startswith(f"{name}=") for name in names):
            return text
    return None


def _optional_extend(argv: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _default_output_root(dataset: str | None) -> Path:
    return vanilla_kd._dataset_root(dataset) / "innovation" / "PER"


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Projected Energy Retention Stage1 trainer/probe. PER is fixed to L9.fc1 and reuses Vanilla KD/QVF capture infrastructure.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-lambda", type=float, default=None, help="Required for training; ignored by loss-scale probe unless supplied.")
    parser.add_argument("--output-root", default=None, help="Default: <dataset_root>/innovation/PER.")
    parser.add_argument("--loss-scale-probe", action="store_true", help="Forward train batches only, write per_loss_scale_probe.json, and do not update weights.")
    parser.add_argument("--per-probe-batches", type=int, default=8)
    parser.add_argument("--per-probe-output", default=None)
    parser.add_argument("--per-epsilon", type=float, default=vanilla_kd.PER_DEFAULT_EPS)
    parser.add_argument("--formal-lambda-set", default=None, help="Comma-separated three formal lambdas chosen after the probe.")
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    blocked = _owned_flag_present(
        passthrough,
        (
            "--baseline-root",
            "--use-qvf-token-fidelity",
            "--qvf-beta",
            "--qvf-layer-weights",
            "--qvf-output-root",
            "--qvf-method-variant",
            "--qvf-protocol-version",
            "--qvf-weighted-variant",
            "--use-projected-energy-retention",
            "--per-lambda",
            "--per-epsilon",
            "--per-output-root",
            "--per-loss-scale-probe",
            "--per-probe-batches",
            "--per-probe-output",
            "--per-formal-lambda-set",
            "--per-method-variant",
            "--per-protocol-version",
            "--export-name",
            "--seed",
            "--dataset",
            "--dry-run",
        ),
    )
    if blocked:
        parser.error(f"{blocked} is owned by run_projected_energy_retention.py")
    if (
        not bool(args.loss_scale_probe)
        and args.per_lambda is None
        and not str(args.formal_lambda_set or "").strip()
        and not bool(args.dry_run)
    ):
        parser.error("--per-lambda is required for PER training; run --loss-scale-probe first to choose it")
    return args, list(passthrough)


def _baseline_argv(args: argparse.Namespace, passthrough: Sequence[str], *, per_lambda_override: float | None = None) -> list[str]:
    per_lambda = 0.0 if per_lambda_override is None and args.per_lambda is None else float(args.per_lambda if per_lambda_override is None else per_lambda_override)
    if per_lambda < 0.0:
        raise argparse.ArgumentTypeError("--per-lambda must be non-negative")
    if float(args.per_epsilon) <= 0.0:
        raise argparse.ArgumentTypeError("--per-epsilon must be positive")
    output_root = Path(str(args.output_root)) if str(args.output_root or "").strip() else _default_output_root(args.dataset)
    argv = list(passthrough)
    _optional_extend(argv, "--dataset", args.dataset)
    if per_lambda == 0.0 and not bool(args.loss_scale_probe):
        argv.extend(["--seed", str(int(args.seed))])
        if bool(args.dry_run):
            argv.append("--dry-run")
        return argv
    argv.extend(
        [
            "--seed",
            str(int(args.seed)),
            "--use-projected-energy-retention",
            "--per-lambda",
            f"{per_lambda:.10g}",
            "--per-epsilon",
            f"{float(args.per_epsilon):.10g}",
            "--per-output-root",
            str(output_root),
            "--per-method-variant",
            vanilla_kd.PER_METHOD_VARIANT,
            "--per-protocol-version",
            vanilla_kd.PER_PROTOCOL_VERSION,
            "--export-name",
            vanilla_kd.PER_METHOD_NAME,
        ]
    )
    if bool(args.loss_scale_probe):
        argv.append("--per-loss-scale-probe")
        argv.extend(["--per-probe-batches", str(int(args.per_probe_batches))])
        if args.per_probe_output:
            argv.extend(["--per-probe-output", str(args.per_probe_output)])
        elif args.output_root:
            probe_output = output_root / f"seed_{int(args.seed)}" / "per_loss_scale_probe.json"
            argv.extend(["--per-probe-output", str(probe_output)])
    if args.formal_lambda_set:
        argv.extend(["--per-formal-lambda-set", str(args.formal_lambda_set)])
    if bool(args.dry_run):
        argv.append("--dry-run")
    return argv


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"metrics CSV missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _portable_path_parts(path_value: str | Path) -> tuple[str, ...]:
    raw = str(path_value or "").strip()
    if not raw:
        return ()
    return tuple(part for part in raw.replace("\\", "/").split("/") if part and part != ".")


def _cross_platform_path_name(path_value: str | Path) -> str:
    parts = _portable_path_parts(path_value)
    return parts[-1] if parts else ""


def _resolve_uploaded_artifact_path(path_value: Any, *, manifest_path: Path, dataset_root: Path) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        raise RuntimeError(f"empty artifact path in manifest: {manifest_path}")
    recorded = Path(raw)
    if recorded.exists() and recorded.is_file():
        return recorded.resolve(strict=False)
    candidates: list[Path] = []
    if not recorded.is_absolute():
        candidates.append(manifest_path.parent / recorded)
    parts = _portable_path_parts(raw)
    for anchor in ("baseline", "innovation"):
        if anchor in parts:
            suffix = parts[parts.index(anchor) :]
            candidates.append(dataset_root.joinpath(*suffix))
    filename = _cross_platform_path_name(raw)
    if filename:
        candidates.append(manifest_path.parent / filename)
    seen: set[str] = set()
    for candidate in candidates:
        marker = str(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve(strict=False)
    checked = ", ".join(str(item) for item in candidates[:8])
    raise FileNotFoundError(f"artifact path from manifest is missing after migration: recorded={recorded}; checked={checked}")


def _read_metric_tables_from_paths(paths: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in ("metrics", "group_metrics", "geometry_metrics", "retention_metrics"):
        value = paths.get(key)
        if not value:
            continue
        rows.extend(_read_csv_rows(Path(str(value))))
    if not rows:
        raise RuntimeError(f"export result contains no metric tables: {paths}")
    return rows


def _rows_from_run_manifest(manifest_path: Path, *, dataset_root: Path) -> list[dict[str, str]]:
    if not manifest_path.exists() or not manifest_path.is_file():
        return []
    manifest = _load_json(manifest_path)
    splits = manifest.get("splits")
    split_result: Any = None
    if isinstance(splits, Mapping):
        split_result = splits.get(SPLIT_NAME)
    if not isinstance(split_result, Mapping):
        outputs = manifest.get("outputs")
        if isinstance(outputs, Mapping):
            split_result = outputs.get(SPLIT_NAME)
    if not isinstance(split_result, Mapping):
        return []
    raw_paths = split_result.get("paths") if "paths" in split_result else split_result
    if not isinstance(raw_paths, Mapping):
        return []
    resolved_paths: dict[str, str] = {}
    for key in ("metrics", "group_metrics", "geometry_metrics", "retention_metrics"):
        value = raw_paths.get(key)
        if not value:
            continue
        resolved_paths[key] = str(
            _resolve_uploaded_artifact_path(value, manifest_path=manifest_path, dataset_root=dataset_root)
        )
    if "metrics" not in resolved_paths or "group_metrics" not in resolved_paths:
        return []
    return _read_metric_tables_from_paths(resolved_paths)


def _metric_rows_for_run(*, metrics_csv: Path, manifest_path: Path | None, dataset_root: Path) -> tuple[list[dict[str, str]], str]:
    if manifest_path is not None:
        rows = _rows_from_run_manifest(manifest_path, dataset_root=dataset_root)
        if rows:
            return rows, str(manifest_path.resolve(strict=False))
    rows = [row for row in _read_csv_rows(metrics_csv) if str(row.get("split")) == SPLIT_NAME]
    if not rows:
        raise RuntimeError(f"metrics table contains no eval rows: {metrics_csv}")
    return rows, str(metrics_csv.resolve(strict=False))


def _metric_value(rows: Sequence[Mapping[str, Any]], *, metric_name: str, analysis_group: str, split: str = "eval") -> float:
    matches = [
        row
        for row in rows
        if str(row.get("split", split)) == str(split)
        and str(row.get("metric_name")) == str(metric_name)
        and str(row.get("analysis_group")) == str(analysis_group)
    ]
    if not matches:
        raise KeyError(f"missing metric: split={split}, metric_name={metric_name}, analysis_group={analysis_group}")
    return float(matches[-1]["metric_value"])


def _classification_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "overall_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="global"),
        "macro_f1_label_universe_nanmean": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="global"),
        "many_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="many"),
        "many_macro_f1": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="many"),
        "medium_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="medium"),
        "medium_macro_f1": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="medium"),
        "tail_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="tail"),
        "tail_macro_f1": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="tail"),
        "few_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="few"),
        "few_f1": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="few"),
    }


def _optional_metric_value(rows: Sequence[Mapping[str, Any]], *, metric_name: str, analysis_group: str, split: str = "eval") -> float:
    try:
        return _metric_value(rows, metric_name=metric_name, analysis_group=analysis_group, split=split)
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nested_mapping(value: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(str(key), {})
    return current if isinstance(current, Mapping) else {}


def _nested_float(value: Mapping[str, Any], key: str) -> float:
    try:
        return float(value.get(str(key)))
    except (TypeError, ValueError):
        return float("nan")


def _candidate_row(
    *,
    index: int,
    lambda_per: float,
    result: Mapping[str, Any],
    baseline: Mapping[str, float],
    baseline_metric_rows: Sequence[Mapping[str, Any]],
    dataset_root: Path,
) -> dict[str, Any]:
    resolved = result.get("resolved")
    if not isinstance(resolved, Mapping):
        raise RuntimeError("PER candidate result missing resolved paths")
    metrics_path = Path(resolved["metrics_csv"])
    manifest_path = Path(resolved["manifest_path"])
    candidate_metric_rows, candidate_metric_source = _metric_rows_for_run(
        metrics_csv=metrics_path,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    metrics = _classification_payload(candidate_metric_rows)
    delta_overall = float(metrics["overall_accuracy"] - baseline["overall_accuracy"])
    delta_macro = float(metrics["macro_f1_label_universe_nanmean"] - baseline["macro_f1_label_universe_nanmean"])
    delta_many = float(metrics["many_accuracy"] - baseline["many_accuracy"])
    delta_medium_accuracy = float(metrics["medium_accuracy"] - baseline["medium_accuracy"])
    delta_medium_macro = float(metrics["medium_macro_f1"] - baseline["medium_macro_f1"])
    delta_tail_accuracy = float(metrics["tail_accuracy"] - baseline["tail_accuracy"])
    delta_tail_macro = float(metrics["tail_macro_f1"] - baseline["tail_macro_f1"])
    failures: list[str] = []
    if delta_overall < OVERALL_ACC_GUARDRAIL:
        failures.append("delta_overall_accuracy")
    if delta_macro < OVERALL_MACRO_F1_GUARDRAIL:
        failures.append("delta_macro_f1_label_universe_nanmean")
    if delta_many < MANY_ACC_GUARDRAIL:
        failures.append("delta_many_accuracy")
    manifest = _load_json(Path(resolved["manifest_path"]))
    per_metrics = _nested_mapping(manifest, ("per_diagnostic_metrics", "actual_eval_output", "metrics"))
    projected_tail_mean = _nested_float(per_metrics, "projected_retention_mean:tail")
    projected_tail_abs_error = _nested_float(per_metrics, "projected_retention_abs_error:tail")
    projected_medium_mean = _nested_float(per_metrics, "projected_retention_mean:medium")
    projected_medium_abs_error = _nested_float(per_metrics, "projected_retention_abs_error:medium")
    baseline_oer_tail = _optional_metric_value(baseline_metric_rows, metric_name="oer_retention", analysis_group="tail")
    candidate_oer_tail = _optional_metric_value(candidate_metric_rows, metric_name="oer_retention", analysis_group="tail")
    baseline_oer_medium = _optional_metric_value(baseline_metric_rows, metric_name="oer_retention", analysis_group="medium")
    candidate_oer_medium = _optional_metric_value(candidate_metric_rows, metric_name="oer_retention", analysis_group="medium")
    baseline_dist_tail = _optional_metric_value(baseline_metric_rows, metric_name="teacher_topk_relative_distortion", analysis_group="tail")
    candidate_dist_tail = _optional_metric_value(candidate_metric_rows, metric_name="teacher_topk_relative_distortion", analysis_group="tail")
    baseline_dist_medium = _optional_metric_value(baseline_metric_rows, metric_name="teacher_topk_relative_distortion", analysis_group="medium")
    candidate_dist_medium = _optional_metric_value(candidate_metric_rows, metric_name="teacher_topk_relative_distortion", analysis_group="medium")
    delta_oer_tail = float(candidate_oer_tail - baseline_oer_tail) if _finite(candidate_oer_tail) and _finite(baseline_oer_tail) else float("nan")
    delta_oer_medium = float(candidate_oer_medium - baseline_oer_medium) if _finite(candidate_oer_medium) and _finite(baseline_oer_medium) else float("nan")
    delta_dist_tail = float(candidate_dist_tail - baseline_dist_tail) if _finite(candidate_dist_tail) and _finite(baseline_dist_tail) else float("nan")
    delta_dist_medium = float(candidate_dist_medium - baseline_dist_medium) if _finite(candidate_dist_medium) and _finite(baseline_dist_medium) else float("nan")
    projected_retention_positive = _finite(projected_tail_mean) and _finite(projected_tail_abs_error)
    oer_positive = (_finite(delta_oer_tail) and delta_oer_tail > 0.0) or (_finite(delta_oer_medium) and delta_oer_medium > 0.0)
    distortion_positive = (_finite(delta_dist_tail) and delta_dist_tail < 0.0) or (_finite(delta_dist_medium) and delta_dist_medium < 0.0)
    mechanism_positive = bool(projected_retention_positive and (oer_positive or distortion_positive))
    medium_tail_task_positive = any(
        value > 0.0
        for value in (delta_medium_accuracy, delta_medium_macro, delta_tail_accuracy, delta_tail_macro)
        if _finite(value)
    )
    medium_tail_task_not_better = all(
        _finite(value) and value <= 0.0
        for value in (delta_medium_accuracy, delta_medium_macro, delta_tail_accuracy, delta_tail_macro)
    )
    medium_tail_task_negative = any(
        value < 0.0
        for value in (delta_medium_accuracy, delta_medium_macro, delta_tail_accuracy, delta_tail_macro)
        if _finite(value)
    )
    if not all(
        _finite(value)
        for value in (
            metrics["overall_accuracy"],
            metrics["macro_f1_label_universe_nanmean"],
            metrics["medium_macro_f1"],
            metrics["tail_macro_f1"],
        )
    ):
        candidate_method_status = "invalid"
    elif mechanism_positive and not medium_tail_task_positive:
        candidate_method_status = "mechanism_only"
    elif not failures and medium_tail_task_positive:
        candidate_method_status = "task_positive"
    elif failures or medium_tail_task_negative:
        candidate_method_status = "task_negative"
    else:
        candidate_method_status = "task_neutral"
    return {
        "candidate_index": int(index),
        "selected": False,
        "passes_guardrail": not failures,
        "guardrail_failure_reasons": ";".join(failures),
        "candidate_method_status": candidate_method_status,
        "candidate_method_success": candidate_method_status == "task_positive",
        "mechanism_positive": bool(mechanism_positive),
        "projected_retention_positive": bool(projected_retention_positive),
        "oer_positive": bool(oer_positive),
        "distortion_positive": bool(distortion_positive),
        "medium_tail_task_positive": bool(medium_tail_task_positive),
        "medium_tail_task_not_better": bool(medium_tail_task_not_better),
        "per_lambda": float(lambda_per),
        "candidate_dir": str(Path(resolved["seed_dir"]).resolve(strict=False)),
        "manifest": str(Path(resolved["manifest_path"]).resolve(strict=False)),
        "checkpoint": str(Path(resolved["best_checkpoint"]).resolve(strict=False)),
        "metrics_csv": str(metrics_path.resolve(strict=False)),
        "metrics_source": candidate_metric_source,
        **metrics,
        "delta_overall_accuracy_vs_vanilla_kd": delta_overall,
        "delta_macro_f1_label_universe_nanmean_vs_vanilla_kd": delta_macro,
        "delta_many_accuracy_vs_vanilla_kd": delta_many,
        "delta_medium_accuracy_vs_vanilla_kd": delta_medium_accuracy,
        "delta_medium_macro_f1_vs_vanilla_kd": delta_medium_macro,
        "delta_tail_accuracy_vs_vanilla_kd": delta_tail_accuracy,
        "delta_tail_macro_f1_vs_vanilla_kd": delta_tail_macro,
        "projected_retention_tail_mean": projected_tail_mean,
        "projected_retention_tail_abs_error": projected_tail_abs_error,
        "projected_retention_medium_mean": projected_medium_mean,
        "projected_retention_medium_abs_error": projected_medium_abs_error,
        "delta_oer_tail_vs_vanilla_kd": delta_oer_tail,
        "delta_oer_medium_vs_vanilla_kd": delta_oer_medium,
        "delta_teacher_topk_relative_distortion_tail_vs_vanilla_kd": delta_dist_tail,
        "delta_teacher_topk_relative_distortion_medium_vs_vanilla_kd": delta_dist_medium,
    }


def _select_best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    feasible = [row for row in rows if bool(row.get("passes_guardrail"))]
    if not feasible:
        return None

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row[SELECTION_METRIC]),
            float(row["tail_accuracy"]),
            float(row["medium_macro_f1"]),
            float(row["medium_accuracy"]),
            -float(row["per_lambda"]),
        )

    return max(feasible, key=key)


def _predefined_stop_status(rows: Sequence[Mapping[str, Any]]) -> str:
    if len(rows) != 3:
        return "not_applicable"
    triggered = all(bool(row.get("mechanism_positive")) and bool(row.get("medium_tail_task_not_better")) for row in rows)
    return "mechanism_positive_task_negative" if triggered else "not_triggered"


def _formal_method_status(rows: Sequence[Mapping[str, Any]], *, best: Mapping[str, Any] | None, predefined_stop_status: str) -> str:
    if not rows:
        return "invalid"
    candidate_statuses = [str(row.get("candidate_method_status", "invalid")) for row in rows]
    if any(status not in METHOD_STATUS_VALUES for status in candidate_statuses):
        return "invalid"
    if str(predefined_stop_status) == "mechanism_positive_task_negative":
        return "mechanism_only"
    if best is None:
        if any(status == "mechanism_only" for status in candidate_statuses):
            return "mechanism_only"
        return "task_negative"
    best_status = str(best.get("candidate_method_status", "invalid"))
    if best_status in METHOD_STATUS_VALUES:
        return best_status
    return "invalid"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    vanilla_kd._write_csv_rows_atomic(path, fieldnames, rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    vanilla_kd._write_json_atomic(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"manifest must be a JSON object: {path}")
    return payload


def _update_candidate_manifests(rows: Sequence[Mapping[str, Any]], *, best: Mapping[str, Any] | None, summary: Mapping[str, Any]) -> None:
    selected_manifest = "" if best is None else str(best.get("manifest", ""))
    for row in rows:
        manifest_path = Path(str(row["manifest"]))
        manifest = _load_json(manifest_path)
        manifest.update(
            {
                "method_status": str(row.get("candidate_method_status", "invalid")),
                "method_status_reason": str(row.get("guardrail_failure_reasons", "")),
                "method_success": bool(row.get("candidate_method_success", False)),
                "selected_candidate": str(row.get("manifest", "")) == selected_manifest,
                "formal_selection": dict(summary),
            }
        )
        payload = vanilla_kd.attach_run_manifest_hash(manifest)
        _write_json(manifest_path, payload)


def _run_formal_lambda_sweep(args: argparse.Namespace, passthrough: Sequence[str]) -> dict[str, object]:
    lambdas = vanilla_kd._parse_per_formal_lambda_set(args.formal_lambda_set)
    if any(float(item) <= 0.0 for item in lambdas):
        raise RuntimeError("formal PER lambda set must contain three positive PER candidates; use plain Vanilla KD as the lambda=0 baseline")
    if args.per_lambda is not None and float(args.per_lambda) not in set(float(item) for item in lambdas):
        raise RuntimeError("--per-lambda must be one of --formal-lambda-set when both are supplied")
    output_root = Path(str(args.output_root)) if str(args.output_root or "").strip() else _default_output_root(args.dataset)
    seed_root = output_root / f"seed_{int(args.seed)}"
    if bool(args.dry_run):
        return {
            "status": "dry_run",
            "formal_lambda_set": [float(item) for item in lambdas],
            "candidate_argv": [_baseline_argv(args, passthrough, per_lambda_override=float(item)) for item in lambdas],
            "summary_dir": str(seed_root.resolve(strict=False)),
        }
    dataset_root = vanilla_kd._dataset_root(args.dataset)
    baseline_seed_root = vanilla_kd._default_baseline_root(args.dataset) / f"seed_{int(args.seed)}"
    baseline_metrics = baseline_seed_root / "vanilla_kd_metrics.csv"
    baseline_manifest = baseline_seed_root / "manifests" / "vanilla_kd" / "run_manifest.json"
    baseline_metric_rows, baseline_metric_source = _metric_rows_for_run(
        metrics_csv=baseline_metrics,
        manifest_path=baseline_manifest,
        dataset_root=dataset_root,
    )
    baseline_payload = _classification_payload(baseline_metric_rows)
    candidate_results: list[Mapping[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, lambda_per in enumerate(lambdas, start=1):
        baseline_args = vanilla_kd.parse_args(_baseline_argv(args, passthrough, per_lambda_override=float(lambda_per)))
        result = vanilla_kd.run_vanilla_kd(baseline_args)
        candidate_results.append(result)
        rows.append(
            _candidate_row(
                index=index,
                lambda_per=float(lambda_per),
                result=result,
                baseline=baseline_payload,
                baseline_metric_rows=baseline_metric_rows,
                dataset_root=dataset_root,
            )
        )
    best = _select_best(rows)
    if best is not None:
        selected_manifest = str(best["manifest"])
        for row in rows:
            row["selected"] = str(row["manifest"]) == selected_manifest
    predefined_stop = _predefined_stop_status(rows)
    method_status = _formal_method_status(rows, best=best, predefined_stop_status=predefined_stop)
    summary_csv = seed_root / "per_formal_selection.csv"
    _write_csv(summary_csv, rows)
    summary = {
        "protocol_version": "per_formal_selection.v1",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "formal_lambda_set": [float(item) for item in lambdas],
        "vanilla_kd_metrics_csv": str(baseline_metrics.resolve(strict=False)),
        "vanilla_kd_metric_source": baseline_metric_source,
        "summary_csv": str(summary_csv.resolve(strict=False)),
        "summary_csv_sha256": vanilla_kd.file_sha256(summary_csv),
        "method_status": method_status,
        "method_status_values": list(METHOD_STATUS_VALUES),
        "method_success": method_status == "task_positive",
        "selected_candidate": None if best is None else dict(best),
        "predefined_stop_status": predefined_stop,
        "predefined_stop_triggered": predefined_stop == "mechanism_positive_task_negative",
        "predefined_stop_criterion": {
            "status": "mechanism_positive_task_negative",
            "condition": (
                "all three formal lambdas have projected retention sanity plus OER/distortion improvement, "
                "while medium/tail task metrics are not better than Vanilla KD"
            ),
        },
        "guardrails": {
            "delta_overall_accuracy_min": OVERALL_ACC_GUARDRAIL,
            "delta_macro_f1_label_universe_nanmean_min": OVERALL_MACRO_F1_GUARDRAIL,
            "delta_many_accuracy_min": MANY_ACC_GUARDRAIL,
        },
        "primary": SELECTION_METRIC,
        "tie_break": list(TIE_BREAK_METRICS),
        "few_used_for_selection": False,
    }
    summary_json = seed_root / "per_formal_selection_manifest.json"
    summary["summary_manifest"] = str(summary_json.resolve(strict=False))
    _write_json(summary_json, vanilla_kd.attach_run_manifest_hash(summary))
    summary["summary_manifest_sha256"] = vanilla_kd.file_sha256(summary_json)
    _update_candidate_manifests(rows, best=best, summary=summary)
    return {"summary": summary, "rows": rows, "candidate_results": candidate_results}


def _cli_result_payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if "candidate_results" not in result:
        return result
    payload = {str(key): value for key, value in result.items() if str(key) != "candidate_results"}
    payload["candidate_result_count"] = len(result.get("candidate_results", [])) if isinstance(result.get("candidate_results"), Sequence) else 0
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args, passthrough = parse_args(argv)
    if str(args.formal_lambda_set or "").strip() and not bool(args.loss_scale_probe):
        return _run_formal_lambda_sweep(args, passthrough)
    baseline_args = vanilla_kd.parse_args(_baseline_argv(args, passthrough))
    return vanilla_kd.run_vanilla_kd(baseline_args)


if __name__ == "__main__":
    cli_result = main()
    if isinstance(cli_result, Mapping) and (
        str(cli_result.get("status", "")) == "dry_run" or "summary" in cli_result
    ):
        print(json.dumps(_cli_result_payload(cli_result), ensure_ascii=False, indent=2, default=str))
