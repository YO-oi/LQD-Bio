from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline.export_eval import DEFAULT_TOKEN_EXPORT, DEFAULT_TOKEN_LAYERS, run_export_eval
from src.train.train_beats_classifier_head import dataset_paths, infer_label_buckets


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
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value!r}")
    return parsed


def _default_baseline_root(dataset_name: str | None) -> Path:
    return dataset_paths(dataset_name, None).root / "baseline" / "teacher"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline FP32 teacher runner: train the classifier head, select best eval_macro_f1_label_universe_nanmean, then export eval/test artifacts.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset key from config.DATASETS; default: config.ACTIVE_DATASET.")
    parser.add_argument("--checkpoint", default=None, help="Source BEATs checkpoint; default is delegated to train_beats_classifier_head.py.")
    parser.add_argument("--baseline-root", default=None, help="Root for FP32 teacher per-seed outputs; default: <dataset_root>/baseline/teacher.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=_positive_int, default=20)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--patience", type=_nonnegative_int, default=5)
    parser.add_argument("--device", default=None, help="Device string; default is delegated to train_beats_classifier_head.py.")
    parser.add_argument("--output", default=None, help="Teacher checkpoint path; default: <dataset_root>/baseline/teacher/seed_<seed>/checkpoints/fp32_teacher_seed_<seed>.pt.")
    parser.add_argument("--log-dir", default=None, help="Epoch log dir; default: <dataset_root>/baseline/teacher/seed_<seed>/logs.")
    parser.add_argument("--export-root", default=None, help="Export root; default: <dataset_root>/baseline/teacher/seed_<seed>.")
    parser.add_argument("--export-name", default="fp32_teacher")
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT)
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS)
    parser.add_argument("--export-all-token-layers", action="store_true")
    parser.add_argument("--max-export-payload-gib", type=_positive_float, default=16.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolved_seed_paths(args: argparse.Namespace) -> dict[str, Path]:
    baseline_root = Path(args.baseline_root) if args.baseline_root else _default_baseline_root(args.dataset)
    seed_dir = baseline_root / f"seed_{int(args.seed)}"
    return {
        "seed_dir": seed_dir,
        "output": Path(args.output) if args.output else seed_dir / "checkpoints" / f"fp32_teacher_seed_{int(args.seed)}.pt",
        "log_dir": Path(args.log_dir) if args.log_dir else seed_dir / "logs",
        "export_root": Path(args.export_root) if args.export_root else seed_dir,
        "metrics_csv": seed_dir / "fp32_teacher_metrics.csv",
    }


def write_combined_metrics(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    rows: list[dict[str, object]] = []
    fieldnames: list[str] = []
    metrics_dir = paths["export_root"] / "metrics" / str(args.export_name)
    for split in ("eval", "test"):
        source = metrics_dir / f"{split}_metrics.csv"
        if not source.exists():
            raise FileNotFoundError(f"export_eval metrics missing: {source}")
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in ["baseline", "seed", *reader.fieldnames]:
                    if name not in fieldnames:
                        fieldnames.append(name)
            for row in reader:
                out = dict(row)
                out.setdefault("split", split)
                out["baseline"] = "fp32_teacher"
                out["seed"] = int(args.seed)
                rows.append(out)
    paths["metrics_csv"].parent.mkdir(parents=True, exist_ok=True)
    with paths["metrics_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"[baseline-fp32-teacher] metrics={paths['metrics_csv']}")


def build_train_command(args: argparse.Namespace) -> list[str]:
    paths = resolved_seed_paths(args)
    output = paths["output"]
    log_dir = paths["log_dir"]

    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train" / "train_beats_classifier_head.py"),
        "--output",
        str(output),
        "--log-dir",
        str(log_dir),
        "--selection-metric",
        "eval_macro_f1_label_universe_nanmean",
        "--skip-export-eval",
        "--seed",
        str(int(args.seed)),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--lr",
        str(float(args.lr)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--label-smoothing",
        str(float(args.label_smoothing)),
        "--num-workers",
        str(int(args.num_workers)),
        "--patience",
        str(int(args.patience)),
    ]
    if args.dataset:
        command.extend(["--dataset", str(args.dataset)])
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint)])
    if args.eval_batch_size is not None:
        command.extend(["--eval-batch-size", str(int(args.eval_batch_size))])
    if args.device:
        command.extend(["--device", str(args.device)])
    if bool(args.overwrite):
        command.append("--overwrite")
    return command


def _load_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"FP32 teacher checkpoint must be a mapping, got {type(payload)!r}: {path}")
    return payload


def _best_export_metadata(checkpoint: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    training = checkpoint.get("classifier_head_training")
    if not isinstance(training, Mapping):
        raise KeyError("FP32 teacher checkpoint missing classifier_head_training metadata required for formal export")
    best_epoch = int(training.get("epoch", -1))
    if best_epoch < 0:
        raise RuntimeError("FP32 teacher training did not produce a best checkpoint; refusing formal export")
    best_metrics = {
        "loss": float(training.get("best_eval_loss", float("nan"))),
        "accuracy": float(training.get("best_eval_accuracy", float("nan"))),
        "eval_macro_f1_label_universe_nanmean": float(training.get("best_eval_macro_f1_label_universe_nanmean", float("nan"))),
        "eval_macro_f1_observed_classes": float(training.get("best_eval_macro_f1_observed_classes", float("nan"))),
        "eval_macro_f1_legacy_zero_absent_diagnostic": float(training.get("best_eval_macro_f1_legacy_zero_absent_diagnostic", float("nan"))),
    }
    return best_epoch, best_metrics


def _export_args(args: argparse.Namespace, paths: dict[str, Path]) -> argparse.Namespace:
    return argparse.Namespace(
        eval_batch_size=int(args.eval_batch_size or args.batch_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        amp=False,
        token_export=str(args.token_export),
        token_layers=str(args.token_layers),
        export_all_token_layers=bool(args.export_all_token_layers),
        max_export_payload_gib=float(args.max_export_payload_gib),
        export_name=str(args.export_name),
        export_root=str(paths["export_root"]),
        selection_metric="eval_macro_f1_label_universe_nanmean",
    )


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"run manifest must contain a JSON object: {path}")
    return payload


def _rewrite_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _annotate_token_indexes(args: argparse.Namespace, paths: dict[str, Path], manifest_path: Path) -> None:
    if str(args.token_export) == "none":
        return
    manifest = _load_json_mapping(manifest_path)
    run_identity_hash = str(manifest.get("run_identity_hash") or "")
    if not run_identity_hash:
        raise RuntimeError(f"formal FP32 manifest missing run_identity_hash: {manifest_path}")
    manifest_path_text = str(manifest_path.resolve(strict=False))
    features_root = paths["export_root"] / "features" / str(args.export_name)
    token_indexes = sorted(features_root.rglob("token_features.index.csv"))
    if not token_indexes:
        raise FileNotFoundError(f"formal token export expected token_features.index.csv under {features_root}")
    required = ("seed", "run_identity_hash", "run_manifest_path", "artifact_kind")
    for token_index in token_indexes:
        with token_index.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"token index has no header: {token_index}")
            fieldnames = list(reader.fieldnames)
            rows = [dict(row) for row in reader]
        if not rows:
            raise RuntimeError(f"token index has no rows: {token_index}")
        for field in required:
            if field not in fieldnames:
                fieldnames.append(field)
        for row_number, row in enumerate(rows, start=2):
            model_name = str(row.get("model_name") or "")
            if model_name != str(args.export_name):
                raise RuntimeError(f"{token_index}:row={row_number} model_name={model_name!r} does not match export_name={args.export_name!r}")
            existing_run_identity = str(row.get("run_identity_hash") or "")
            if existing_run_identity and existing_run_identity != run_identity_hash:
                raise RuntimeError(f"{token_index}:row={row_number} run_identity_hash mismatch against manifest")
            existing_manifest = str(row.get("run_manifest_path") or "")
            if existing_manifest and Path(existing_manifest).resolve(strict=False) != manifest_path.resolve(strict=False):
                raise RuntimeError(f"{token_index}:row={row_number} run_manifest_path mismatch against canonical manifest")
            existing_kind = str(row.get("artifact_kind") or "")
            if existing_kind and existing_kind != "token_feature_shard":
                raise RuntimeError(f"{token_index}:row={row_number} artifact_kind={existing_kind!r} is not token_feature_shard")
            existing_seed = str(row.get("seed") or "")
            if existing_seed and int(existing_seed) != int(args.seed):
                raise RuntimeError(f"{token_index}:row={row_number} seed mismatch against formal seed={args.seed}")
            row["seed"] = int(args.seed)
            row["run_identity_hash"] = run_identity_hash
            row["run_manifest_path"] = manifest_path_text
            row["artifact_kind"] = "token_feature_shard"
        _rewrite_csv(token_index, rows, fieldnames)
        print(f"[baseline-fp32-teacher] annotated token index={token_index}")


def _assert_formal_outputs(args: argparse.Namespace, paths: dict[str, Path], manifest_path: Path) -> None:
    export_root = paths["export_root"]
    export_name = str(args.export_name)
    required = [paths["output"], manifest_path]
    for split in ("eval", "test"):
        required.extend(
            [
                export_root / "logits" / export_name / f"{split}_logits.pt",
                export_root / "embeddings" / export_name / f"{split}_embeddings.pt",
                export_root / "predictions" / export_name / f"{split}_predictions.csv",
                export_root / "metrics" / export_name / f"{split}_metrics.csv",
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"formal FP32 teacher export is incomplete; missing: {names}")


def export_fp32_teacher(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    checkpoint = _load_checkpoint_payload(paths["output"])
    best_epoch, best_metrics = _best_export_metadata(checkpoint)
    train_paths = dataset_paths(args.dataset, str(paths["output"]))
    label_buckets = infer_label_buckets(train_paths)
    export_args = _export_args(args, paths)

    import torch

    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    summary = run_export_eval(
        paths=train_paths,
        args=export_args,
        label_buckets=label_buckets,
        device=device,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
    )
    manifest_path = paths["export_root"] / "manifests" / str(args.export_name) / "run_manifest.json"
    _annotate_token_indexes(args, paths, manifest_path)
    _assert_formal_outputs(args, paths, manifest_path)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = resolved_seed_paths(args)
    command = build_train_command(args)
    print("[baseline-fp32-teacher] running train-only command:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    export_fp32_teacher(args, paths)
    write_combined_metrics(args, paths)


if __name__ == "__main__":
    main()
