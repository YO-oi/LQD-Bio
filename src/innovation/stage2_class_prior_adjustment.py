from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline.export_eval import DEFAULT_TOKEN_EXPORT, DEFAULT_TOKEN_LAYERS, audit_quant_state, load_trained_checkpoint, move_batch_to_device
from src.baseline.manifest import (
    attach_run_manifest_hash,
    baseline_common_manifest,
    canonical_json_hash,
    file_sha256,
    path_str,
    write_run_manifests,
)
from src.baseline.run_vanilla_kd import _validate_label_tensor
from src.innovation.run_two_stage_decoupling import (
    FORMAL_DEFAULTS,
    StageSpec,
    _append_stage_log,
    _configure_stage,
    _export_dirs,
    _make_scheduler,
    _optimizer_lr,
    _reject_amp,
    _run_export_eval,
    _seed_everything,
    _write_json,
    _write_train_log,
    forward_pooled_logits,
)
from src.innovation.stage2_eval_metrics import (
    STAGE2_SELECTION_METRIC,
    build_stage2_eval_metric_row,
    write_stage2_eval_metrics,
)
from src.train.train_beats_classifier_head import (
    build_loader,
    compute_eval_breakdown,
    dataset_paths,
    infer_label_buckets,
    label_names_by_id,
    load_label_mapping,
    num_predictor_classes,
    validate_split_labels,
)
from src.utils.group_utils import bucket_from_train_label_n, read_canonical_bucket


STAGE2_CLASS_PRIOR_PROTOCOL = "innovation_stage2_class_prior_adjustment.v1"
INNOVATION_NAME = "stage2_class"
MODEL_NAME = "stage2_class_prior_adjustment"
BASELINE_NAME = INNOVATION_NAME
MODEL_KIND = "two_stage_class_prior_tau"
METHOD_VARIANT = "stage2_class_prior_tau"
_EXPORT_METRIC_TABLE_KEYS = ("metrics", "group_metrics", "per_class_metrics", "geometry_metrics")


def _rewrite_metric_table_identity(path: str | Path, *, baseline: str, model_kind: str, method_variant: str) -> None:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"stage2 class-prior export metric table missing during identity rewrite: {target}")
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError(f"stage2 class-prior export metric table is empty during identity rewrite: {target}")
    for name in reversed(("baseline", "model_kind", "method_variant")):
        if name not in fieldnames:
            fieldnames.insert(0, name)
    for row in rows:
        row["baseline"] = str(baseline)
        row["model_kind"] = str(model_kind)
        row["method_variant"] = str(method_variant)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _rewrite_export_metric_identity(
    split_result: dict[str, Any],
    *,
    baseline: str = BASELINE_NAME,
    model_kind: str = MODEL_KIND,
    method_variant: str = METHOD_VARIANT,
) -> None:
    paths = split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("stage2 class-prior export split result missing paths for identity rewrite")
    artifact_sha256 = dict(split_result.get("artifact_sha256") or {})
    for key in _EXPORT_METRIC_TABLE_KEYS:
        value = paths.get(key)
        if value is None or not str(value).strip():
            continue
        target = Path(value)
        _rewrite_metric_table_identity(
            target,
            baseline=baseline,
            model_kind=model_kind,
            method_variant=method_variant,
        )
        artifact_sha256[str(target.resolve(strict=False))] = file_sha256(target)
    split_result["baseline"] = str(baseline)
    split_result["model_kind"] = str(model_kind)
    split_result["method_variant"] = str(method_variant)
    split_result["artifact_sha256"] = artifact_sha256


def _rewrite_export_results_identity(split_results: Mapping[str, Any]) -> None:
    for split_result in split_results.values():
        if not isinstance(split_result, dict):
            raise TypeError(f"stage2 class-prior split result must be mutable dict, got {type(split_result)!r}")
        _rewrite_export_metric_identity(split_result)


def _write_combined_metrics(split_results: Mapping[str, Any], path: Path, *, seed: int) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for split, result in split_results.items():
        metrics_path = Path(result["paths"]["metrics"])
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in ["baseline", "seed", "model_kind", "method_variant", *reader.fieldnames]:
                    if name not in fieldnames:
                        fieldnames.append(name)
            for row in reader:
                out = dict(row)
                out.setdefault("split", str(split))
                out["baseline"] = BASELINE_NAME
                out["model_kind"] = MODEL_KIND
                out["method_variant"] = METHOD_VARIANT
                out["seed"] = int(seed)
                rows.append(out)
    if not rows:
        raise RuntimeError("no stage2 class-prior metrics rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _missing_csv_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "<na>"}


def _csv_label_id(value: Any, *, path: Path, row_number: int) -> int:
    if _missing_csv_value(value):
        raise ValueError(f"{path}:row={row_number} missing label_id")
    numeric = float(str(value).strip())
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{path}:row={row_number} label_id must be an integer, got {value!r}")
    return int(numeric)


def _bucket_from_train_row(row: Mapping[str, Any], *, path: Path, row_number: int) -> str | None:
    bucket_value = row.get("bucket")
    shot_group_value = row.get("shot_group")
    train_label_n_value = row.get("train_label_n")
    if _missing_csv_value(train_label_n_value):
        train_label_n_value = row.get("label_n")
    has_bucket = not _missing_csv_value(bucket_value)
    has_shot_group = not _missing_csv_value(shot_group_value)
    has_train_label_n = not _missing_csv_value(train_label_n_value)
    context = f"{path}:row={row_number}"
    if has_bucket or has_shot_group:
        result = read_canonical_bucket(
            bucket=[bucket_value] if has_bucket else None,
            shot_group=[shot_group_value] if has_shot_group else None,
            train_label_n=[train_label_n_value] if has_train_label_n else None,
            context=context,
            allow_shot_group_fallback=False,
        )
        return result.bucket[0]
    if has_train_label_n:
        return bucket_from_train_label_n(train_label_n_value)
    return None


def build_train_label_buckets(train_csv: str | Path) -> dict[int, str]:
    target = Path(train_csv)
    if not target.exists():
        raise FileNotFoundError(f"train split CSV not found for label bucket map: {target}")
    counts: Counter[int] = Counter()
    buckets_by_label: dict[int, set[str]] = {}
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "label_id" not in reader.fieldnames:
            raise KeyError(f"train split missing label_id column for label buckets: {target}")
        for row_number, row in enumerate(reader, start=2):
            label_id = _csv_label_id(row.get("label_id"), path=target, row_number=row_number)
            counts[int(label_id)] += 1
            bucket = _bucket_from_train_row(row, path=target, row_number=row_number)
            if bucket is not None:
                buckets_by_label.setdefault(int(label_id), set()).add(str(bucket))
    conflicts = {
        label_id: sorted(values)
        for label_id, values in buckets_by_label.items()
        if len(values) > 1
    }
    if conflicts:
        preview = "; ".join(f"label_id={label_id}: buckets={values}" for label_id, values in sorted(conflicts.items())[:8])
        raise ValueError(f"{target}: inconsistent train bucket metadata; {preview}")
    result = {int(label_id): sorted(values)[0] for label_id, values in buckets_by_label.items()}
    for label_id, count in counts.items():
        result.setdefault(int(label_id), bucket_from_train_label_n(int(count)))
    if not result:
        raise RuntimeError(f"train split produced no label bucket map: {target}")
    return result


def validate_split_bucket_coverage(split_csv: str | Path, label_buckets: Mapping[int, str], *, split_name: str) -> None:
    target = Path(split_csv)
    if not target.exists():
        raise FileNotFoundError(f"{split_name} split CSV not found for label bucket coverage: {target}")
    missing: set[int] = set()
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "label_id" not in reader.fieldnames:
            raise KeyError(f"{split_name} split missing label_id column for label bucket coverage: {target}")
        for row_number, row in enumerate(reader, start=2):
            label_id = _csv_label_id(row.get("label_id"), path=target, row_number=row_number)
            if int(label_id) not in label_buckets:
                missing.add(int(label_id))
    if missing:
        raise ValueError(f"{split_name} split contains labels missing from train-derived bucket map: {sorted(missing)[:8]}")


def _strip_test_split_manifest_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    split_csv = payload.get("split_csv")
    if isinstance(split_csv, Mapping):
        split_csv_payload = dict(split_csv)
        split_csv_payload.pop("test", None)
        payload["split_csv"] = split_csv_payload
    split_hashes = payload.get("split_hashes")
    if isinstance(split_hashes, Mapping):
        split_hash_payload = dict(split_hashes)
        split_hash_payload.pop("test", None)
        payload["split_hashes"] = split_hash_payload
        payload["split_hash"] = canonical_json_hash(split_hash_payload)
    return payload


def export_stage2_class_prior_checkpoint(
    *,
    args: argparse.Namespace,
    resolved: Mapping[str, Path],
    checkpoint_path: str | Path,
    calibration_hash: str | None,
    label_buckets: Mapping[int, str],
    device: torch.device,
    splits: Sequence[str],
    export_root: str | Path,
    export_name: str,
    metrics_csv: str | Path | None = None,
) -> dict[str, Any]:
    split_results, export_checkpoint, token_layer_indices, token_export, export_run_identity_hash, export_quant_audit = _run_export_eval(
        args=args,
        resolved=resolved,
        checkpoint_path=Path(checkpoint_path),
        calibration_hash=calibration_hash,
        label_buckets=label_buckets,
        device=device,
        splits=tuple(str(item) for item in splits),
        export_root=Path(export_root),
        export_name=str(export_name),
    )
    _rewrite_export_results_identity(split_results)
    if metrics_csv is not None:
        _write_combined_metrics(split_results, Path(metrics_csv), seed=int(args.seed))
    return {
        "split_results": split_results,
        "export_checkpoint": export_checkpoint,
        "token_layer_indices": [int(item) for item in token_layer_indices],
        "token_export": token_export,
        "export_run_identity_hash": export_run_identity_hash,
        "export_quant_audit": export_quant_audit,
    }


def build_class_prior_payload(train_csv: str | Path, *, num_classes: int) -> dict[str, Any]:
    target = Path(train_csv)
    if not target.exists():
        raise FileNotFoundError(f"train split CSV not found for class prior: {target}")
    counts = [0 for _ in range(int(num_classes))]
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "label_id" not in reader.fieldnames:
            raise KeyError(f"train split missing label_id column for class prior: {target}")
        for row_index, row in enumerate(reader, start=2):
            label_id = int(row["label_id"])
            if label_id < 0 or label_id >= int(num_classes):
                raise ValueError(f"train split label_id out of range at row {row_index}: {label_id}")
            counts[label_id] += 1
    total = int(sum(counts))
    if total <= 0:
        raise RuntimeError(f"train split has no samples for class prior: {target}")
    zero_classes = [index for index, count in enumerate(counts) if int(count) <= 0]
    if zero_classes:
        raise RuntimeError(
            "class prior correction refuses zero-count classes; "
            f"missing train samples for label_id={zero_classes[:16]}"
        )
    prior = [float(count) / float(total) for count in counts]
    log_prior = [math.log(value) for value in prior]
    hash_payload = {
        "protocol_version": STAGE2_CLASS_PRIOR_PROTOCOL,
        "num_classes": int(num_classes),
        "total_count": int(total),
        "class_count": [int(value) for value in counts],
        "class_prior": [float(value) for value in prior],
        "log_class_prior": [float(value) for value in log_prior],
    }
    class_prior_hash = canonical_json_hash(hash_payload)
    return {
        **hash_payload,
        "class_prior_hash": class_prior_hash,
        "train_csv": str(target.resolve(strict=False)),
        "train_csv_sha256": file_sha256(target),
        "log_prior_min": float(min(log_prior)),
        "log_prior_max": float(max(log_prior)),
    }


def write_class_prior_payload(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def log_prior_tensor(payload: Mapping[str, Any], *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    values = torch.as_tensor(payload.get("log_class_prior"), dtype=dtype, device=device)
    if values.ndim != 1 or int(values.numel()) <= 0:
        raise ValueError("class prior payload must contain non-empty 1D log_class_prior")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("class prior log_class_prior contains non-finite values")
    return values.contiguous()


def cosine_rising_tau(epoch: int, total_epochs: int, *, tau_min: float, tau_max: float) -> float:
    epoch = int(epoch)
    total_epochs = int(total_epochs)
    tau_min = float(tau_min)
    tau_max = float(tau_max)
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be positive, got {total_epochs}")
    if epoch < 1 or epoch > total_epochs:
        raise ValueError(f"epoch must be in [1, {total_epochs}], got {epoch}")
    if not math.isfinite(tau_min) or not math.isfinite(tau_max):
        raise ValueError("tau_min/tau_max must be finite")
    if tau_min < 0.0 or tau_max < 0.0:
        raise ValueError("tau_min/tau_max must be non-negative")
    if tau_max < tau_min:
        raise ValueError(f"tau_max must be >= tau_min, got tau_min={tau_min}, tau_max={tau_max}")
    progress = 0.0 if total_epochs <= 1 else float(epoch - 1) / float(total_epochs - 1)
    weight = 0.5 * (1.0 - math.cos(math.pi * progress))
    return float(tau_min + (tau_max - tau_min) * weight)


def classifier_bias_adjustment(log_prior: Tensor, *, tau: float, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Tensor:
    target = log_prior.detach()
    if device is not None or dtype is not None:
        target = target.to(device=device if device is not None else target.device, dtype=dtype if dtype is not None else target.dtype)
    return (-float(tau) * target.float()).to(device=target.device, dtype=target.dtype).contiguous()


def apply_logit_adjustment(logits: Tensor, log_prior: Tensor, *, tau: float) -> Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logit adjustment expects logits [N,C], got shape={tuple(logits.shape)}")
    if log_prior.ndim != 1 or int(log_prior.numel()) != int(logits.shape[1]):
        raise ValueError(
            "log prior must be [C] and match logits class dimension: "
            f"log_prior={tuple(log_prior.shape)}, logits={tuple(logits.shape)}"
        )
    adjustment = classifier_bias_adjustment(log_prior, tau=float(tau), device=logits.device, dtype=logits.dtype)
    return (logits.float() + adjustment.float().unsqueeze(0)).contiguous()


def state_dict_with_folded_logit_adjustment(model: torch.nn.Module, log_prior: Tensor, *, tau: float) -> dict[str, Tensor]:
    state = {str(key): value.detach().clone() for key, value in model.state_dict().items()}
    bias_key = "predictor.bias"
    if bias_key not in state:
        raise KeyError("cannot fold Stage 2 logit adjustment into checkpoint: predictor.bias missing")
    bias = state[bias_key]
    if bias.ndim != 1:
        raise ValueError(f"predictor.bias must be 1D, got shape={tuple(bias.shape)}")
    adjustment = classifier_bias_adjustment(log_prior, tau=float(tau), device=bias.device, dtype=bias.dtype)
    if int(adjustment.numel()) != int(bias.numel()):
        raise ValueError(
            "class prior dimension must match predictor.bias: "
            f"adjustment={int(adjustment.numel())}, bias={int(bias.numel())}"
        )
    state[bias_key] = (bias + adjustment).contiguous()
    return state


def stage2_adjustment_audit(
    *,
    class_prior_payload: Mapping[str, Any],
    epoch: int,
    tau: float,
    tau_min: float,
    tau_max: float,
    enabled: bool,
    folded_into_classifier: bool,
) -> dict[str, Any]:
    return {
        "stage2_epoch": int(epoch),
        "stage2_tau": float(tau),
        "stage2_tau_min": float(tau_min),
        "stage2_tau_max": float(tau_max),
        "stage2_class_prior_hash": str(class_prior_payload.get("class_prior_hash") or ""),
        "stage2_log_prior_min": float(class_prior_payload.get("log_prior_min")),
        "stage2_log_prior_max": float(class_prior_payload.get("log_prior_max")),
        "stage2_logit_adjustment_enabled": bool(enabled),
        "stage2_logit_adjustment_folded_into_classifier": bool(folded_into_classifier),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"must be a positive finite float, got {value!r}")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(f"must be a non-negative finite float, got {value!r}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 2 class-prior adjustment from an existing "
            "run_two_stage_decoupling.py stage1_checkpoint.pt."
        )
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--baseline-root", default=None, help="Default: <dataset_root>/baseline.")
    parser.add_argument("--two-stage-root", default=None, help="Default: <dataset_root>/innovation/two_stage.")
    parser.add_argument("--innovation-root", default=None, help="Default: <dataset_root>/innovation/stage2_class.")
    parser.add_argument("--seed-dir", default=None, help="Override the resolved seed directory for sweep/variant runs.")
    parser.add_argument("--stage1-checkpoint", default=None, help="Default: <dataset_root>/innovation/two_stage/seed_<seed>/stage1_checkpoint.pt.")
    parser.add_argument("--reference-root", default=None, help="Default: <dataset_root>/baseline/teacher/seed_<seed>.")
    parser.add_argument("--stage2-epochs", type=_positive_int, default=int(FORMAL_DEFAULTS["stage2_epochs"]))
    parser.add_argument("--stage2-classifier-lr", type=_positive_float, default=float(FORMAL_DEFAULTS["stage2_classifier_lr"]))
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=float(FORMAL_DEFAULTS["warmup_ratio"]))
    parser.add_argument("--tau-min", type=_nonnegative_float, default=0.0)
    parser.add_argument("--tau-max", type=_nonnegative_float, default=1.0)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=8)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage2-checkpoint", default=None, help="Default: <seed_dir>/stage2_checkpoint.pt.")
    parser.add_argument("--stage2-last-checkpoint", "--stage2-last", dest="stage2_last_checkpoint", default=None, help="Default: <seed_dir>/stage2_last.pt.")
    parser.add_argument("--final-checkpoint", default=None, help="Default: <seed_dir>/final_checkpoint.pt.")
    parser.add_argument("--class-prior-json", default=None, help="Default: <seed_dir>/class_prior.json.")
    parser.add_argument("--stage2-train-log", default=None, help="Default: <seed_dir>/stage2_train_log.csv.")
    parser.add_argument("--stage2-eval-metrics", default=None, help="Default: <seed_dir>/stage2_eval_metrics.csv.")
    parser.add_argument("--metrics-csv", default=None, help="Default: <seed_dir>/stage2_class_metrics.csv.")
    parser.add_argument("--stage2-manifest", default=None, help="Default: <seed_dir>/stage2_manifest.json.")
    parser.add_argument("--stage-log", default=None, help="Default: <seed_dir>/stage_log.jsonl.")
    parser.add_argument("--manifest-path", default=None, help="Default: <seed_dir>/manifests/stage2_class/run_manifest.json.")
    parser.add_argument("--export-root", default=None, help="Default: <seed_dir>.")
    parser.add_argument("--stage1-export-root", default=None, help="Default: <seed_dir>/stage1_export.")
    parser.add_argument("--stage2-export-root", default=None, help="Default: <seed_dir>/stage2_export.")
    parser.add_argument("--export-name", default=INNOVATION_NAME)
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT)
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS)
    parser.add_argument("--export-all-token-layers", action="store_true")
    parser.add_argument("--max-export-payload-gib", type=float, default=16.0)
    parser.add_argument("--skip-test-during-sweep", action="store_true", help="Export eval only; tau sweep will run test once after selecting the best tau.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolved_paths(args: argparse.Namespace, *, dataset_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(dataset_root) if dataset_root is not None else dataset_paths(args.dataset, None).root
    seed = int(args.seed)
    two_stage_root = Path(args.two_stage_root) if args.two_stage_root else root / "innovation" / "two_stage"
    output_root = Path(args.innovation_root) if args.innovation_root else root / "innovation" / INNOVATION_NAME
    baseline_root = Path(args.baseline_root) if args.baseline_root else root / "baseline"
    seed_dir = Path(args.seed_dir) if args.seed_dir else output_root / f"seed_{seed}"
    teacher_seed_dir = baseline_root / "teacher" / f"seed_{seed}"
    two_stage_seed_dir = two_stage_root / f"seed_{seed}"
    return {
        "seed_dir": seed_dir,
        "stage1_checkpoint": Path(args.stage1_checkpoint) if args.stage1_checkpoint else two_stage_seed_dir / "stage1_checkpoint.pt",
        "reference_root": Path(args.reference_root) if args.reference_root else teacher_seed_dir,
        "stage2_checkpoint": Path(args.stage2_checkpoint) if args.stage2_checkpoint else seed_dir / "stage2_checkpoint.pt",
        "stage2_last_checkpoint": Path(args.stage2_last_checkpoint) if args.stage2_last_checkpoint else seed_dir / "stage2_last.pt",
        "final_checkpoint": Path(args.final_checkpoint) if args.final_checkpoint else seed_dir / "final_checkpoint.pt",
        "class_prior_json": Path(args.class_prior_json) if args.class_prior_json else seed_dir / "class_prior.json",
        "stage2_train_log": Path(args.stage2_train_log) if args.stage2_train_log else seed_dir / "stage2_train_log.csv",
        "stage2_eval_metrics": Path(args.stage2_eval_metrics) if args.stage2_eval_metrics else seed_dir / "stage2_eval_metrics.csv",
        "stage2_manifest": Path(args.stage2_manifest) if args.stage2_manifest else seed_dir / "stage2_manifest.json",
        "stage_log": Path(args.stage_log) if args.stage_log else seed_dir / "stage_log.jsonl",
        "manifest_path": Path(args.manifest_path) if args.manifest_path else seed_dir / "manifests" / INNOVATION_NAME / "run_manifest.json",
        "export_root": Path(args.export_root) if args.export_root else seed_dir,
        "stage1_export_root": Path(args.stage1_export_root) if args.stage1_export_root else seed_dir / "stage1_export",
        "stage2_export_root": Path(args.stage2_export_root) if args.stage2_export_root else seed_dir / "stage2_export",
        "metrics_csv": Path(args.metrics_csv) if args.metrics_csv else seed_dir / f"{INNOVATION_NAME}_metrics.csv",
    }


def _stage2_spec(args: argparse.Namespace) -> StageSpec:
    return StageSpec(
        name=MODEL_NAME,
        index=2,
        objective="reshape_decision_boundary_with_class_prior_adjustment",
        epochs=int(args.stage2_epochs),
        backbone_lr=0.0,
        classifier_lr=float(args.stage2_classifier_lr),
        kd_enabled=False,
        train_backbone=False,
        train_classifier=True,
    )


def _require_stage1_checkpoint_payload(payload: Mapping[str, Any], *, checkpoint_path: Path, paths: Any) -> None:
    if int(payload.get("stage_index", -1)) != 1:
        raise RuntimeError(f"stage2_class requires a Stage 1 checkpoint, got stage_index={payload.get('stage_index')!r}: {checkpoint_path}")
    if str(payload.get("stage_name") or "") != "stage1_semantic_manifold":
        raise RuntimeError(f"stage2_class requires stage1_semantic_manifold, got {payload.get('stage_name')!r}: {checkpoint_path}")
    if str(payload.get("selection_metric") or "") != STAGE2_SELECTION_METRIC:
        raise RuntimeError(f"Stage 1 checkpoint selection metric mismatch: {payload.get('selection_metric')!r}")
    if str(payload.get("label_mapping_hash") or "") != str(file_sha256(paths.label_mapping)):
        raise RuntimeError("Stage 1 checkpoint label_mapping_hash differs from the selected dataset")
    if not payload.get("run_identity_hash"):
        raise RuntimeError(f"Stage 1 checkpoint missing run_identity_hash: {checkpoint_path}")
    if not isinstance(payload.get("eval_metrics"), Mapping):
        raise RuntimeError(f"Stage 1 checkpoint missing eval_metrics: {checkpoint_path}")


def _quant_audit(model: torch.nn.Module, payload: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    controller = getattr(model, "_baseline_quant_controller", None)
    if controller is None:
        raise RuntimeError(f"{context}: loaded Stage 1/Stage 2 model is missing _baseline_quant_controller")
    audit = audit_quant_state(
        model,
        checkpoint=payload,
        quant_controller=controller,
        expected_quantized_layers=[dict(row) for row in payload.get("quantized_layers", [])],
        expected_qparams_hash=str(payload.get("qparams_hash") or ""),
    )
    audit["context"] = context
    return audit


def train_stage2_class_prior_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    log_prior: Tensor,
    tau: float,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="stage2_class.train.label_id")
        optimizer.zero_grad(set_to_none=True)
        logits = forward_pooled_logits(model, batch)
        adjusted_logits = apply_logit_adjustment(logits, log_prior, tau=float(tau))
        loss = F.cross_entropy(adjusted_logits.float(), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        predictions = adjusted_logits.detach().argmax(dim=1)
        batch_size = int(labels.numel())
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
    if total_samples <= 0:
        raise RuntimeError("Stage 2 class-prior train loader produced no samples")
    return {
        "train_loss": total_loss / float(total_samples),
        "train_ce_loss": total_loss / float(total_samples),
        "train_kd_loss": 0.0,
        "train_accuracy": total_correct / float(total_samples),
        "train_num_samples": float(total_samples),
        "teacher_forward_count": 0.0,
    }


def evaluate_stage2_class_prior(
    *,
    model: torch.nn.Module,
    loader: Any,
    log_prior: Tensor,
    tau: float,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    labels_all: list[int] = []
    preds_all: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="stage2_class.eval.label_id")
            logits = forward_pooled_logits(model, batch)
            adjusted_logits = apply_logit_adjustment(logits, log_prior, tau=float(tau))
            loss = F.cross_entropy(adjusted_logits.float(), labels)
            predictions = adjusted_logits.detach().argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size
            labels_all.extend(int(item) for item in labels.detach().cpu().tolist())
            preds_all.extend(int(item) for item in predictions.detach().cpu().tolist())
    if total_samples <= 0:
        raise RuntimeError("Stage 2 class-prior eval loader produced no samples")
    return {
        "loss": total_loss / float(total_samples),
        "accuracy": total_correct / float(total_samples),
        "num_samples": float(total_samples),
        "labels": labels_all,
        "predictions": preds_all,
    }


def _save_stage2_class_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    source_payload: Mapping[str, Any],
    source_stage1_checkpoint: Path,
    stage_epoch: int,
    global_epoch: int,
    eval_metrics: Mapping[str, Any],
    class_prior_payload: Mapping[str, Any],
    tau: float,
    tau_min: float,
    tau_max: float,
    stage_contract: Mapping[str, Any],
    quant_audit: Mapping[str, Any],
    best_metric_so_far: float,
) -> dict[str, Any]:
    log_prior = log_prior_tensor(class_prior_payload, device=next(model.parameters()).device)
    folded_state = state_dict_with_folded_logit_adjustment(model, log_prior, tau=float(tau))
    adjustment = stage2_adjustment_audit(
        class_prior_payload=class_prior_payload,
        epoch=int(stage_epoch),
        tau=float(tau),
        tau_min=float(tau_min),
        tau_max=float(tau_max),
        enabled=True,
        folded_into_classifier=True,
    )
    identity_payload = {
        "protocol_version": STAGE2_CLASS_PRIOR_PROTOCOL,
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "model_name": MODEL_NAME,
        "source_stage1_checkpoint_sha256": file_sha256(source_stage1_checkpoint),
        "source_stage1_run_identity_hash": source_payload.get("run_identity_hash"),
        "teacher_checkpoint_sha256": source_payload.get("teacher_checkpoint_sha256"),
        "qparams_hash": source_payload.get("qparams_hash"),
        "quantized_layers_hash": source_payload.get("quantized_layers_hash"),
        "label_mapping_hash": source_payload.get("label_mapping_hash"),
        "class_prior_hash": class_prior_payload.get("class_prior_hash"),
        "stage_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "tau": float(tau),
        "seed": int(source_payload.get("seed", 0)),
    }
    run_identity_hash = canonical_json_hash(identity_payload)
    payload = {str(key): value for key, value in dict(source_payload).items() if not str(key).startswith("_")}
    payload.update(
        {
            "protocol_version": STAGE2_CLASS_PRIOR_PROTOCOL,
            "artifact_type": "innovation_stage2_class_prior_adjustment_checkpoint",
            "baseline": BASELINE_NAME,
            "model_kind": MODEL_KIND,
            "method_variant": METHOD_VARIANT,
            "model_name": MODEL_NAME,
            "innovation_name": INNOVATION_NAME,
            "stage_name": MODEL_NAME,
            "stage_index": 2,
            "stage_objective": "reshape_decision_boundary_with_class_prior_adjustment",
            "epoch": int(global_epoch),
            "stage_epoch": int(stage_epoch),
            "selection_metric": STAGE2_SELECTION_METRIC,
            "best_metric_so_far": float(best_metric_so_far),
            "eval_metrics": dict(eval_metrics),
            "model_state_dict": folded_state,
            "source_stage1_checkpoint": path_str(source_stage1_checkpoint),
            "source_stage1_checkpoint_sha256": file_sha256(source_stage1_checkpoint),
            "source_stage1_run_identity_hash": source_payload.get("run_identity_hash"),
            "class_prior": dict(class_prior_payload),
            "class_prior_hash": class_prior_payload.get("class_prior_hash"),
            "stage2_tau": float(tau),
            "stage2_tau_min": float(tau_min),
            "stage2_tau_max": float(tau_max),
            "stage2_adjustment_audit": adjustment,
            "stage2_contract": dict(stage_contract),
            "training_quant_audit": dict(quant_audit),
            "run_identity_hash": run_identity_hash,
            "run_identity_payload": identity_payload,
            "load_note": "Load through src.baseline.export_eval.load_trained_checkpoint; classifier bias already includes the selected class-prior logit adjustment.",
        }
    )
    for parameter in model.parameters():
        parameter.grad = None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def run_stage2_class_prior_adjustment(args: argparse.Namespace) -> dict[str, Any]:
    _reject_amp(args)
    _seed_everything(int(args.seed))
    paths = dataset_paths(args.dataset, None)
    resolved = _resolved_paths(args, dataset_root=paths.root)
    stage2 = _stage2_spec(args)
    if bool(args.dry_run):
        print(json.dumps({"resolved": {key: path_str(value) for key, value in resolved.items()}, "stage2": stage2.__dict__}, indent=2, ensure_ascii=False))
        return {"dry_run": True, "resolved": resolved}

    if not resolved["stage1_checkpoint"].exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {resolved['stage1_checkpoint']}; run run_two_stage_decoupling.py first")
    sweep_mode = bool(args.skip_test_during_sweep)
    label_mapping = load_label_mapping(paths.label_mapping)
    num_classes = num_predictor_classes(label_mapping)
    validate_split_labels(paths.train_csv, num_classes=num_classes, name="train")
    validate_split_labels(paths.eval_csv, num_classes=num_classes, name="eval")
    if sweep_mode:
        label_buckets = build_train_label_buckets(paths.train_csv)
        validate_split_bucket_coverage(paths.eval_csv, label_buckets, split_name="eval")
    else:
        validate_split_labels(paths.test_csv, num_classes=num_classes, name="test")
        label_buckets = infer_label_buckets(paths)
    label_names = label_names_by_id(label_mapping)

    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    student, stage1_payload = load_trained_checkpoint(resolved["stage1_checkpoint"], device=device)
    _require_stage1_checkpoint_payload(stage1_payload, checkpoint_path=resolved["stage1_checkpoint"], paths=paths)
    initial_quant_audit = _quant_audit(student, stage1_payload, context="stage2_class_loaded_stage1_best")
    class_prior_payload = build_class_prior_payload(paths.train_csv, num_classes=num_classes)
    write_class_prior_payload(resolved["class_prior_json"], class_prior_payload)
    log_prior = log_prior_tensor(class_prior_payload, device=device)

    train_loader = build_loader(paths, split="train", batch_size=int(args.batch_size), num_workers=int(args.num_workers), seed=int(args.seed))
    eval_loader = build_loader(paths, split="eval", batch_size=int(args.eval_batch_size), num_workers=int(args.num_workers), seed=int(args.seed))

    stage1_split_results, _stage1_export_checkpoint, _stage1_token_layers, _stage1_token_export, stage1_export_run_identity_hash, _stage1_export_quant_audit = _run_export_eval(
        args=args,
        resolved=resolved,
        checkpoint_path=resolved["stage1_checkpoint"],
        calibration_hash=str(stage1_payload.get("calibration_hash") or ""),
        label_buckets=label_buckets,
        device=device,
        splits=("eval",),
        export_root=resolved["stage1_export_root"],
        export_name=f"{INNOVATION_NAME}_stage1_source",
    )
    _rewrite_export_results_identity(stage1_split_results)

    optimizer, stage_contract = _configure_stage(student, stage2, weight_decay=float(args.weight_decay))
    scheduler = _make_scheduler(
        optimizer,
        total_steps=int(stage2.epochs) * max(1, len(train_loader)),
        warmup_ratio=float(args.warmup_ratio),
    )
    stage_protocol = {
        "protocol_version": STAGE2_CLASS_PRIOR_PROTOCOL,
        "baseline": BASELINE_NAME,
        "model_kind": MODEL_KIND,
        "method_variant": METHOD_VARIANT,
        "sweep_mode": bool(sweep_mode),
        "selection_split": "eval",
        "export_splits": ["eval"] if sweep_mode else ["eval", "test"],
        "test_split_touched": False if sweep_mode else True,
        "source_stage1_checkpoint": path_str(resolved["stage1_checkpoint"]),
        "source_stage1_checkpoint_sha256": file_sha256(resolved["stage1_checkpoint"]),
        "source_stage1_run_identity_hash": stage1_payload.get("run_identity_hash"),
        "stage2": stage2.__dict__,
        "tau_schedule": "cosine_rising_tau",
        "tau_min": float(args.tau_min),
        "tau_max": float(args.tau_max),
        "class_prior_hash": class_prior_payload.get("class_prior_hash"),
        "stage_contract": dict(stage_contract),
    }
    _append_stage_log(
        resolved["stage_log"],
        {
            "event": "stage2_class_start",
            "stage": stage2.name,
            "stage_index": int(stage2.index),
            "source_stage1_checkpoint": path_str(resolved["stage1_checkpoint"]),
            "class_prior_hash": class_prior_payload.get("class_prior_hash"),
            "contract": dict(stage_contract),
        },
    )

    train_rows: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_train_row: dict[str, Any] = {}
    best_eval_payload: dict[str, Any] = {}
    best_quant_audit: dict[str, Any] = {}
    best_adjustment_audit: dict[str, Any] = {}
    best_stage_epoch = 0
    best_global_epoch = int(stage1_payload.get("epoch", 0))
    best_checkpoint_payload: dict[str, Any] | None = None
    last_eval_payload: dict[str, Any] = {}
    last_quant_audit: dict[str, Any] = {}
    last_tau = float(args.tau_min)

    for stage_epoch in range(1, int(stage2.epochs) + 1):
        global_epoch = int(stage1_payload.get("epoch", 0)) + int(stage_epoch)
        tau = cosine_rising_tau(stage_epoch, int(stage2.epochs), tau_min=float(args.tau_min), tau_max=float(args.tau_max))
        last_tau = float(tau)
        train_metrics = train_stage2_class_prior_epoch(
            model=student,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            log_prior=log_prior,
            tau=float(tau),
            device=device,
            num_classes=num_classes,
        )
        if float(train_metrics.get("teacher_forward_count", 0.0)) != 0.0:
            raise RuntimeError("Stage 2 class-prior contract violation: teacher forward was computed")
        last_quant_audit = _quant_audit(student, stage1_payload, context=f"after_stage2_class_epoch_{stage_epoch}")
        eval_metrics = evaluate_stage2_class_prior(
            model=student,
            loader=eval_loader,
            log_prior=log_prior,
            tau=float(tau),
            device=device,
            num_classes=num_classes,
        )
        eval_breakdown, _per_class_rows = compute_eval_breakdown(eval_metrics, label_names=label_names, label_buckets=label_buckets)
        current = float(eval_breakdown.get(STAGE2_SELECTION_METRIC, float("nan")))
        adjustment_audit = stage2_adjustment_audit(
            class_prior_payload=class_prior_payload,
            epoch=int(stage_epoch),
            tau=float(tau),
            tau_min=float(args.tau_min),
            tau_max=float(args.tau_max),
            enabled=True,
            folded_into_classifier=False,
        )
        row = {
            "stage": stage2.name,
            "stage_index": int(stage2.index),
            "stage_epoch": int(stage_epoch),
            "global_epoch": int(global_epoch),
            "objective": stage2.objective,
            "backbone_lr": 0.0,
            "backbone_optimizer_group_exists": bool(stage_contract.get("backbone_optimizer_group_exists", False)),
            "classifier_lr": _optimizer_lr(optimizer, "classifier"),
            "kd_loss_enabled": False,
            "backbone_trainable": False,
            "classifier_trainable": True,
            **{key: float(value) for key, value in train_metrics.items()},
            "eval_loss": float(eval_metrics["loss"]),
            "eval_accuracy": float(eval_metrics["accuracy"]),
            "eval_macro_f1_label_universe_nanmean": current,
            "stage2_tau": float(tau),
            "stage2_tau_min": float(args.tau_min),
            "stage2_tau_max": float(args.tau_max),
            "stage2_class_prior_hash": class_prior_payload.get("class_prior_hash"),
            "stage2_logit_adjustment_enabled": True,
            "stage2_train_ce_loss": float(train_metrics["train_ce_loss"]),
            "stage2_train_total_loss": float(train_metrics["train_loss"]),
            "stage2_train_kd_loss": 0.0,
            "stage2_teacher_kd_enabled": False,
            "stage2_eval_accuracy": float(eval_metrics["accuracy"]),
            "stage2_eval_macro_f1_label_universe_nanmean": current,
            "stage2_backbone_lr": 0.0,
            "stage2_backbone_optimizer_group_exists": bool(stage_contract.get("backbone_optimizer_group_exists", False)),
            "stage2_classifier_lr": _optimizer_lr(optimizer, "classifier"),
            "stage2_backbone_trainable_param_count": 0,
            "stage2_classifier_trainable_param_count": int(stage_contract["parameter_status"]["classifier"]["trainable_num_params"]),
            "quant_audit_observer_enabled": bool(last_quant_audit.get("observer_enabled")),
            "quant_audit_fake_quant_enabled": bool(last_quant_audit.get("fake_quant_enabled")),
            "quant_audit_qparams_frozen": bool(last_quant_audit.get("qparams_frozen")),
        }
        train_rows.append(row)
        _write_train_log(resolved["stage2_train_log"], train_rows)
        print(
            f"[stage2-class] epoch={stage_epoch}/{stage2.epochs} tau={tau:.6f} "
            f"train_loss={row['train_loss']:.6f} eval_macro_f1_label_universe_nanmean={current:.6f}"
        )
        last_eval_payload = {**eval_metrics, **eval_breakdown}
        if math.isfinite(current) and current > best_metric:
            best_metric = current
            best_train_row = dict(row)
            best_eval_payload = dict(last_eval_payload)
            best_quant_audit = dict(last_quant_audit)
            best_adjustment_audit = dict(adjustment_audit)
            best_stage_epoch = int(stage_epoch)
            best_global_epoch = int(global_epoch)
            best_checkpoint_payload = _save_stage2_class_checkpoint(
                resolved["stage2_checkpoint"],
                model=student,
                source_payload=stage1_payload,
                source_stage1_checkpoint=resolved["stage1_checkpoint"],
                stage_epoch=int(stage_epoch),
                global_epoch=int(global_epoch),
                eval_metrics=best_eval_payload,
                class_prior_payload=class_prior_payload,
                tau=float(tau),
                tau_min=float(args.tau_min),
                tau_max=float(args.tau_max),
                stage_contract=stage_contract,
                quant_audit=last_quant_audit,
                best_metric_so_far=best_metric,
            )
            _append_stage_log(
                resolved["stage_log"],
                {
                    "event": "stage2_class_best_checkpoint",
                    "stage_epoch": int(stage_epoch),
                    "global_epoch": int(global_epoch),
                    "selection_metric": STAGE2_SELECTION_METRIC,
                    "selection_metric_value": float(current),
                    "tau": float(tau),
                    "checkpoint": path_str(resolved["stage2_checkpoint"]),
                    "checkpoint_sha256": file_sha256(resolved["stage2_checkpoint"]),
                },
            )

    if best_checkpoint_payload is None or not best_eval_payload:
        raise RuntimeError(f"Stage 2 class-prior produced no finite {STAGE2_SELECTION_METRIC} checkpoint")
    _save_stage2_class_checkpoint(
        resolved["stage2_last_checkpoint"],
        model=student,
        source_payload=stage1_payload,
        source_stage1_checkpoint=resolved["stage1_checkpoint"],
        stage_epoch=int(stage2.epochs),
        global_epoch=int(stage1_payload.get("epoch", 0)) + int(stage2.epochs),
        eval_metrics=last_eval_payload,
        class_prior_payload=class_prior_payload,
        tau=float(last_tau),
        tau_min=float(args.tau_min),
        tau_max=float(args.tau_max),
        stage_contract=stage_contract,
        quant_audit=last_quant_audit,
        best_metric_so_far=best_metric,
    )

    best_model, best_loaded_payload = load_trained_checkpoint(resolved["stage2_checkpoint"], device=device)
    final_quant_audit = _quant_audit(best_model, best_loaded_payload, context="stage2_class_best_loaded_for_final_export")
    final_payload = dict(best_loaded_payload)
    final_payload["artifact_type"] = "innovation_stage2_class_prior_adjustment_final_checkpoint"
    final_payload["baseline"] = BASELINE_NAME
    final_payload["model_kind"] = MODEL_KIND
    final_payload["method_variant"] = METHOD_VARIANT
    final_payload["checkpoint_usage"] = "final_selected_stage2_best"
    resolved["final_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value for key, value in final_payload.items() if not str(key).startswith("_")}, resolved["final_checkpoint"])

    stage2_split_results, export_checkpoint, token_layer_indices, token_export, export_run_identity_hash, export_quant_audit = _run_export_eval(
        args=args,
        resolved=resolved,
        checkpoint_path=resolved["stage2_checkpoint"],
        calibration_hash=str(stage1_payload.get("calibration_hash") or ""),
        label_buckets=label_buckets,
        device=device,
        splits=("eval",),
        export_root=resolved["stage2_export_root"],
        export_name=INNOVATION_NAME,
    )
    _rewrite_export_results_identity(stage2_split_results)
    stage2_eval_metric_row = build_stage2_eval_metric_row(
        split_result=stage2_split_results["eval"],
        stage1_split_result=stage1_split_results["eval"],
        best_train_row=best_train_row,
        stage_contract=stage_contract,
        adjustment_audit=best_adjustment_audit,
        label_buckets=label_buckets,
        reference_root=resolved["reference_root"],
        seed=int(args.seed),
        stage_epoch=best_stage_epoch,
        global_epoch=best_global_epoch,
        checkpoint_path=resolved["stage2_checkpoint"],
        checkpoint_sha256=file_sha256(resolved["stage2_checkpoint"]),
    )
    stage2_eval_metric_row.update(
        {
            "baseline": BASELINE_NAME,
            "model_kind": MODEL_KIND,
            "method_variant": METHOD_VARIANT,
        }
    )
    write_stage2_eval_metrics(resolved["stage2_eval_metrics"], stage2_eval_metric_row)

    final_export_splits = ("eval",) if sweep_mode else ("eval", "test")
    final_export_payload = export_stage2_class_prior_checkpoint(
        args=args,
        resolved=resolved,
        checkpoint_path=resolved["final_checkpoint"],
        calibration_hash=str(stage1_payload.get("calibration_hash") or ""),
        label_buckets=label_buckets,
        device=device,
        splits=final_export_splits,
        export_root=resolved["export_root"],
        export_name=str(args.export_name),
        metrics_csv=resolved["metrics_csv"],
    )
    final_split_results = final_export_payload["split_results"]
    final_export_checkpoint = final_export_payload["export_checkpoint"]
    final_export_run_identity_hash = final_export_payload["export_run_identity_hash"]
    final_export_quant_audit = final_export_payload["export_quant_audit"]

    audio_cfg = project_config.AUDIO if isinstance(getattr(project_config, "AUDIO", None), Mapping) else {}
    manifest = baseline_common_manifest(
        stage="innovation_stage2_class_prior_adjustment",
        dataset_name=paths.name,
        checkpoint=resolved["final_checkpoint"],
        train_csv=paths.train_csv,
        eval_csv=paths.eval_csv,
        test_csv=None if sweep_mode else paths.test_csv,
        label_mapping=paths.label_mapping,
        calibration_csv=stage1_payload.get("calibration_csv"),
        calibration_hash=stage1_payload.get("calibration_hash"),
        quantized_layers=[dict(row) for row in stage1_payload.get("quantized_layers", [])],
        bitwidth={"weight": 4, "activation": 4},
        input_length_samples=int(audio_cfg.get("TARGET_NUM_SAMPLES", 0) or 0) or None,
        input_length_seconds=float(audio_cfg.get("TARGET_SECONDS", 0.0) or 0.0) or None,
        sample_rate=int(audio_cfg.get("TARGET_SR", 0) or 0) or None,
        seed=int(args.seed),
        extra={
            "protocol_version": STAGE2_CLASS_PRIOR_PROTOCOL,
            "baseline": BASELINE_NAME,
            "innovation_name": INNOVATION_NAME,
            "method_variant": METHOD_VARIANT,
            "model_name": MODEL_NAME,
            "model_kind": MODEL_KIND,
            "sweep_mode": bool(sweep_mode),
            "selection_split": "eval",
            "export_splits": list(final_export_splits),
            "test_split_touched": False if sweep_mode else True,
            "skip_test_during_sweep": bool(sweep_mode),
            "final_export_splits": list(final_export_splits),
            "source_stage1_checkpoint": path_str(resolved["stage1_checkpoint"]),
            "source_stage1_checkpoint_sha256": file_sha256(resolved["stage1_checkpoint"]),
            "source_stage1_run_identity_hash": stage1_payload.get("run_identity_hash"),
            "stage1_export": {"run_identity_hash": stage1_export_run_identity_hash, "split_results": stage1_split_results},
            "stage2_checkpoint": path_str(resolved["stage2_checkpoint"]),
            "stage2_checkpoint_sha256": file_sha256(resolved["stage2_checkpoint"]),
            "stage2_checkpoint_usage": "eval_best_folded_class_prior",
            "stage2_last": path_str(resolved["stage2_last_checkpoint"]),
            "stage2_last_sha256": file_sha256(resolved["stage2_last_checkpoint"]),
            "stage2_last_usage": "audit_only",
            "final_checkpoint": path_str(resolved["final_checkpoint"]),
            "final_checkpoint_sha256": file_sha256(resolved["final_checkpoint"]),
            "class_prior_json": path_str(resolved["class_prior_json"]),
            "class_prior_json_sha256": file_sha256(resolved["class_prior_json"]),
            "class_prior": class_prior_payload,
            "tau_schedule": "cosine_rising_tau",
            "tau_min": float(args.tau_min),
            "tau_max": float(args.tau_max),
            "best_tau": float(best_adjustment_audit.get("stage2_tau")),
            "selection_metric": STAGE2_SELECTION_METRIC,
            "best_eval_macro_f1_label_universe_nanmean": float(best_metric),
            "best_stage_epoch": int(best_stage_epoch),
            "best_global_epoch": int(best_global_epoch),
            "best_eval": best_eval_payload,
            "stage2_train_log": path_str(resolved["stage2_train_log"]),
            "stage2_eval_metrics": path_str(resolved["stage2_eval_metrics"]),
            "stage2_eval_metric_row": stage2_eval_metric_row,
            "stage2_manifest": path_str(resolved["stage2_manifest"]),
            "stage_log": path_str(resolved["stage_log"]),
            "metrics_csv": path_str(resolved["metrics_csv"]),
            "stage2_protocol": stage_protocol,
            "token_layers": [int(item) for item in token_layer_indices],
            "token_export": token_export,
            "run_identity_hash": final_export_run_identity_hash,
            "checkpoint_run_identity_hash": final_export_checkpoint.get("run_identity_hash"),
            "train_quant_audit_initial": initial_quant_audit,
            "train_quant_audit_best": best_quant_audit,
            "train_quant_audit_final": final_quant_audit,
            "export_quant_audit": export_quant_audit,
            "final_export_quant_audit": final_export_quant_audit,
            "artifact_sha256": {
                "source_stage1_checkpoint.pt": file_sha256(resolved["stage1_checkpoint"]),
                "stage2_checkpoint.pt": file_sha256(resolved["stage2_checkpoint"]),
                "stage2_last.pt": file_sha256(resolved["stage2_last_checkpoint"]),
                "final_checkpoint.pt": file_sha256(resolved["final_checkpoint"]),
                "class_prior.json": file_sha256(resolved["class_prior_json"]),
                "stage2_train_log.csv": file_sha256(resolved["stage2_train_log"]),
                "stage2_eval_metrics.csv": file_sha256(resolved["stage2_eval_metrics"]),
                **{key: value for split_result in final_split_results.values() for key, value in dict(split_result.get("artifact_sha256", {})).items()},
            },
            "splits": final_split_results,
            "stage2_export": {"run_identity_hash": export_run_identity_hash, "split_results": stage2_split_results},
        },
    )
    if sweep_mode:
        manifest = _strip_test_split_manifest_fields(manifest)
    payload = attach_run_manifest_hash(manifest)
    _write_json(resolved["stage2_manifest"], payload)
    _write_json(resolved["manifest_path"], payload)
    write_run_manifests({"seed": resolved["seed_dir"], **_export_dirs(resolved["export_root"], str(args.export_name))}, payload)
    print(f"[stage2-class] source_stage1_checkpoint={resolved['stage1_checkpoint']}")
    print(f"[stage2-class] stage2_checkpoint={resolved['stage2_checkpoint']}")
    print(f"[stage2-class] final_checkpoint={resolved['final_checkpoint']}")
    print(f"[stage2-class] run_manifest_hash={payload['run_manifest_hash']}")
    return {"manifest": payload, "resolved": resolved, "splits": final_split_results}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_stage2_class_prior_adjustment(args)


if __name__ == "__main__":
    main()
