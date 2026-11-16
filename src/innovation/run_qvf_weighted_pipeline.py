from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.innovation import export_QVF_weighted, export_qvf_weighted_token_survival, run_qvf_weighted
from src.innovation.run_qvf_weighted import DEFAULT_BETA, DEFAULT_LAYER_WEIGHTS, DEFAULT_SEED


def _optional_extend(argv: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _split_extra(value: str | None) -> list[str]:
    text = str(value or "").strip()
    return [] if not text else shlex.split(text)


def _reject_owned_extra_args(stage: str, argv: Sequence[str], owned_flags: Sequence[str]) -> None:
    owned = tuple(str(flag) for flag in owned_flags)
    for item in argv:
        text = str(item)
        if text in owned or any(text.startswith(f"{flag}=") for flag in owned):
            raise ValueError(f"{stage} extra args must not override pipeline-owned argument {text!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run weighted QVF end to end: train, token_survival, then formal export.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qvf-beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--qvf-layer-weights", default=DEFAULT_LAYER_WEIGHTS)
    parser.add_argument("--output-root", default=None, help="Default: <dataset_root>/innovation/QVF_<weight_variant>.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--train-eval-batch-size", type=int, default=None)
    parser.add_argument("--mechanism-eval-batch-size", type=int, default=None)
    parser.add_argument("--mechanism-calibration-batch-size", type=int, default=None)
    parser.add_argument("--export-batch-size", type=int, default=None)
    parser.add_argument("--force-export", action="store_true", help="Forward to final export_QVF_weighted.")
    parser.add_argument("--no-resume", action="store_true", help="Forward to training and final export stages.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-token-survival", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--train-extra-args", default="", help="Extra quoted args forwarded only to run_qvf_weighted.py.")
    parser.add_argument("--token-survival-extra-args", default="", help="Extra quoted args forwarded only to export_qvf_weighted_token_survival.py.")
    parser.add_argument("--export-extra-args", default="", help="Extra quoted args forwarded only to export_QVF_weighted.py.")
    return parser


def _common_args(args: argparse.Namespace) -> list[str]:
    argv: list[str] = [
        "--seed",
        str(int(args.seed)),
        "--qvf-beta",
        f"{float(args.qvf_beta):.10g}",
        "--qvf-layer-weights",
        str(args.qvf_layer_weights),
    ]
    _optional_extend(argv, "--dataset", args.dataset)
    _optional_extend(argv, "--output-root", args.output_root)
    return argv


def _train_args(args: argparse.Namespace) -> list[str]:
    extra = _split_extra(args.train_extra_args)
    _reject_owned_extra_args(
        "train",
        extra,
        (
            "--dataset",
            "--seed",
            "--qvf-beta",
            "--qvf-layer-weights",
            "--qvf-output-root",
            "--qvf-method-variant",
            "--qvf-protocol-version",
            "--qvf-weighted-variant",
            "--device",
            "--num-workers",
            "--batch-size",
            "--eval-batch-size",
            "--resume",
            "--no-resume",
            "--dry-run",
        ),
    )
    argv = [
        "--seed",
        str(int(args.seed)),
        "--qvf-beta",
        f"{float(args.qvf_beta):.10g}",
        "--qvf-layer-weights",
        str(args.qvf_layer_weights),
    ]
    _optional_extend(argv, "--dataset", args.dataset)
    _optional_extend(argv, "--qvf-output-root", args.output_root)
    _optional_extend(argv, "--device", args.device)
    _optional_extend(argv, "--num-workers", args.num_workers)
    _optional_extend(argv, "--batch-size", args.train_batch_size)
    _optional_extend(argv, "--eval-batch-size", args.train_eval_batch_size)
    if bool(args.no_resume):
        argv.append("--no-resume")
    if bool(args.dry_run):
        argv.append("--dry-run")
    argv.extend(extra)
    return argv


def _token_survival_args(args: argparse.Namespace) -> list[str]:
    extra = _split_extra(args.token_survival_extra_args)
    _reject_owned_extra_args(
        "token_survival",
        extra,
        (
            "--dataset",
            "--seed",
            "--qvf-beta",
            "--qvf-layer-weights",
            "--output-root",
            "--qvf-root",
            "--betas",
            "--expected-method-variant",
            "--expected-protocol-version",
            "--expected-layer-weights",
            "--expected-qvf-weighted-variant",
            "--vanilla-token-survival-dir",
            "--device",
            "--num-workers",
            "--eval-batch-size",
            "--calibration-batch-size",
            "--dry-run",
        ),
    )
    argv = _common_args(args)
    _optional_extend(argv, "--device", args.device)
    _optional_extend(argv, "--num-workers", args.num_workers)
    _optional_extend(argv, "--eval-batch-size", args.mechanism_eval_batch_size)
    _optional_extend(argv, "--calibration-batch-size", args.mechanism_calibration_batch_size)
    if bool(args.dry_run):
        argv.append("--dry-run")
    argv.extend(extra)
    return argv


def _export_args(args: argparse.Namespace) -> list[str]:
    extra = _split_extra(args.export_extra_args)
    _reject_owned_extra_args(
        "export",
        extra,
        (
            "--dataset",
            "--seed",
            "--qvf-beta",
            "--qvf-layer-weights",
            "--output-root",
            "--betas",
            "--formal-betas",
            "--expected-method-variant",
            "--expected-protocol-version",
            "--expected-layer-weights",
            "--expected-qvf-weighted-variant",
            "--device",
            "--num-workers",
            "--batch-size",
            "--resume",
            "--no-resume",
            "--force-export",
            "--dry-run",
        ),
    )
    argv = _common_args(args)
    _optional_extend(argv, "--device", args.device)
    _optional_extend(argv, "--num-workers", args.num_workers)
    _optional_extend(argv, "--batch-size", args.export_batch_size)
    if bool(args.force_export):
        argv.append("--force-export")
    if bool(args.no_resume):
        argv.append("--no-resume")
    if bool(args.dry_run):
        argv.append("--dry-run")
    argv.extend(extra)
    return argv


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    stages: dict[str, object] = {}
    if not bool(args.skip_train):
        train_argv = _train_args(args)
        print(f"[qvf-weighted-pipeline] stage=train argv={train_argv}")
        run_qvf_weighted.main(train_argv)
        stages["train"] = {"status": "completed", "argv": train_argv}
    else:
        stages["train"] = {"status": "skipped"}

    if not bool(args.skip_token_survival):
        token_argv = _token_survival_args(args)
        print(f"[qvf-weighted-pipeline] stage=token_survival argv={token_argv}")
        export_qvf_weighted_token_survival.main(token_argv)
        stages["token_survival"] = {"status": "completed", "argv": token_argv}
    else:
        stages["token_survival"] = {"status": "skipped"}

    if not bool(args.skip_export):
        export_argv = _export_args(args)
        print(f"[qvf-weighted-pipeline] stage=export argv={export_argv}")
        result = export_QVF_weighted.main(export_argv)
        stages["export"] = {"status": "completed", "argv": export_argv, "result": result}
    else:
        stages["export"] = {"status": "skipped"}

    return {
        "status": "completed",
        "seed": int(args.seed),
        "qvf_beta": float(args.qvf_beta),
        "qvf_layer_weights": str(args.qvf_layer_weights),
        "stages": stages,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    result = run_pipeline(args)
    print(json.dumps({"status": result["status"], "stages": {k: v["status"] for k, v in result["stages"].items()}}, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
