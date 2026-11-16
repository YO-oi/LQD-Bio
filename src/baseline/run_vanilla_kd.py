from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline.export_eval import (
    DEFAULT_POOLING,
    DEFAULT_TOKEN_EXPORT,
    DEFAULT_TOKEN_LAYERS,
    _export_dirs,
    _formal_export_args,
    _resolve_export_token_layers,
    _split_hash_payload,
    _token_export_mode_from_layers,
    audio_config_from_project,
    audit_quant_state,
    build_run_identity,
    export_eval_split,
    load_trained_checkpoint,
    move_batch_to_device,
    pooled_logits_from_features,
    prepare_encoder_input_for_export,
)
from src.baseline import quant_checkpoint as qck
from src.baseline.fp32_teacher_staging import stage_fp32_teacher_checkpoint
from src.baseline.manifest import attach_run_manifest_hash, baseline_common_manifest, canonical_json_hash, file_sha256, path_str, write_run_manifests
from src.baseline.run_int4_student import BASELINE_NAME as INT4_BASELINE_NAME
from src.metrics.metric_config import DEFAULT_EPS
from src.metrics.token_survival_metrics import select_fp32_topk_token_ids
from src.quant.ffn_quant_injector import get_ffn_module
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
from src.utils.group_utils import bucket_from_train_label_n
from src.utils.layout_utils import canonicalize_activation_btc


PROTOCOL_VERSION = "baseline_vanilla_kd.v1"
QVF_PROTOCOL_VERSION = "qvf_stage1_token_fidelity.v1"
QVF_WEIGHTED_PROTOCOL_VERSION = "qvf_stage1_token_fidelity_weighted.v1"
BASELINE_NAME = "vanilla_kd"
QVF_METHOD_VARIANT = "stage1_vanilla_kd_plus_qvf_token_fidelity"
QVF_WEIGHTED_METHOD_VARIANT_PREFIX = "stage1_vanilla_kd_plus_qvf_weighted_token_fidelity"
QVF_SENSITIVE_LAYERS = (1, 9)
QVF_MODULE_NAME = "fc1"
QVF_TOPK_RATIO = float(getattr(project_config, "TOKEN_TOPK_RATIO", 0.05))
PER_PROTOCOL_VERSION = "per_stage1_projected_energy_retention.v1"
PER_METHOD_VARIANT = "stage1_vanilla_kd_plus_projected_energy_retention"
PER_METHOD_NAME = "projected_energy_retention"
PER_LAYERS = (9,)
PER_MODULE_NAME = QVF_MODULE_NAME
PER_TOPK_RATIO = QVF_TOPK_RATIO
PER_DEFAULT_EPS = 1e-12
PER_LOSS_SCALE_PROBE_PROTOCOL_VERSION = "per_loss_scale_probe.v1"
_RUN_MANIFEST_HASH_EXCLUDE_KEYS = {"run_manifest_hash", "run_manifest_hash_algorithm", "manifest_path"}
FORMAL_DEFAULTS = {
    "alpha": 0.5,
    "temperature": 4.0,
    "epochs": 30,
    "backbone_lr": 1e-5,
    "classifier_lr": 1e-3,
    "warmup_ratio": 0.05,
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
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be positive finite float, got {value!r}")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"must be non-negative finite float, got {value!r}")
    return parsed


def _dataset_root(dataset_name: str | None) -> Path:
    return dataset_paths(dataset_name, None).root


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _train_label_bucket_counts(train_csv: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    declared_counts: dict[int, set[int]] = {}
    with Path(train_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_id = _optional_int(row.get("label_id"))
            if label_id is None:
                continue
            counts[label_id] = int(counts.get(label_id, 0)) + 1
            train_label_n = _optional_int(row.get("train_label_n"))
            if train_label_n is None:
                train_label_n = _optional_int(row.get("label_n"))
            if train_label_n is not None:
                declared_counts.setdefault(label_id, set()).add(int(train_label_n))
    result: dict[int, int] = {}
    for label_id, count in counts.items():
        declared = declared_counts.get(label_id, set())
        if len(declared) > 1:
            raise ValueError(f"{train_csv}: inconsistent train_label_n values for label_id={label_id}: {sorted(declared)}")
        result[label_id] = int(next(iter(declared))) if declared else int(count)
    return result


def _eval_label_ids(eval_csv: Path) -> set[int]:
    label_ids: set[int] = set()
    with Path(eval_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_id = _optional_int(row.get("label_id"))
            if label_id is not None:
                label_ids.add(int(label_id))
    return label_ids


def infer_label_buckets_without_full_csv(paths: Any, *, require_eval_coverage: bool = True) -> dict[int, str]:
    train_counts = _train_label_bucket_counts(Path(paths.train_csv))
    result = {int(label_id): bucket_from_train_label_n(int(count)) for label_id, count in train_counts.items()}
    if require_eval_coverage:
        missing = sorted(_eval_label_ids(Path(paths.eval_csv)) - set(result))
        if missing:
            raise RuntimeError(
                "QVF/PER bucket inference must not read full_csv/test-derived metadata, "
                f"but eval labels are missing from train_csv: {missing[:16]}"
            )
    return result


def _default_baseline_root(dataset_name: str | None) -> Path:
    return _dataset_root(dataset_name) / "baseline" / "kd"


def _teacher_seed_dir(dataset_name: str | None, seed: int) -> Path:
    return _dataset_root(dataset_name) / "baseline" / "teacher" / f"seed_{int(seed)}"


def _student_seed_dir(dataset_name: str | None, seed: int) -> Path:
    return _dataset_root(dataset_name) / "baseline" / "student" / f"seed_{int(seed)}"


def _teacher_checkpoint_from_seed_dir(seed_dir: Path, seed: int) -> Path:
    return Path(seed_dir) / "checkpoints" / f"fp32_teacher_seed_{int(seed)}.pt"


def _qparams_csv_from_seed_dir(seed_dir: Path) -> Path:
    return Path(seed_dir) / "qparams" / "int4_qparams.csv"


def _int4_manifest_from_seed_dir(seed_dir: Path) -> Path:
    return Path(seed_dir) / "manifests" / "int4_student" / "run_manifest.json"


def _int4_checkpoint_from_seed_dir(seed_dir: Path, seed: int) -> Path:
    return Path(seed_dir) / "checkpoints" / f"int4_student_calibrated_seed_{int(seed)}.pt"


def _qvf_beta_stem(beta: float) -> str:
    value = float(beta)
    if math.isclose(value, 0.01, rel_tol=0.0, abs_tol=1e-12):
        return "0p01"
    if math.isclose(value, 0.05, rel_tol=0.0, abs_tol=1e-12):
        return "0p05"
    if math.isclose(value, 0.10, rel_tol=0.0, abs_tol=1e-12):
        return "0p10"
    return f"{value:.10g}".replace("+", "").replace("-", "m").replace(".", "p")


def _qvf_float_stem(value: float) -> str:
    text = f"{float(value):.10g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def qvf_weighted_variant(layer_weights: Mapping[int, float]) -> str:
    return "_".join(f"L{int(layer)}_{_qvf_float_stem(float(weight))}" for layer, weight in sorted(layer_weights.items()))


def _parse_qvf_layer_weights(value: str | None) -> dict[int, float] | None:
    text = str(value or "").strip()
    if not text:
        return None
    result: dict[int, float] = {}
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"qvf layer weight must be layer:weight, got {item!r}")
        raw_layer, raw_weight = item.split(":", 1)
        layer_text = raw_layer.strip().lower()
        if layer_text.startswith("l"):
            layer_text = layer_text[1:]
        if "." in layer_text:
            layer_text = layer_text.split(".", 1)[0]
        layer_idx = int(layer_text)
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise argparse.ArgumentTypeError(f"qvf layer weight must be finite and non-negative, got {raw_weight!r}")
        result[int(layer_idx)] = float(weight)
    expected = tuple(int(item) for item in QVF_SENSITIVE_LAYERS)
    if tuple(sorted(result)) != tuple(sorted(expected)):
        raise argparse.ArgumentTypeError(f"qvf layer weights must cover exactly {expected}, got {sorted(result)}")
    total = sum(float(item) for item in result.values())
    if not math.isfinite(total) or total <= 0.0:
        raise argparse.ArgumentTypeError("qvf layer weights must sum to a positive finite value")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise argparse.ArgumentTypeError(f"qvf layer weights must sum to 1.0, got {total}")
    return {int(layer): float(result[int(layer)]) for layer in expected}


def _qvf_layer_weights_from_args(args: argparse.Namespace) -> dict[int, float] | None:
    return _parse_qvf_layer_weights(getattr(args, "qvf_layer_weights", None))


def _qvf_layer_weight_payload(layer_weights: Mapping[int, float]) -> dict[str, float]:
    return {f"L{int(layer)}.{QVF_MODULE_NAME}": float(weight) for layer, weight in sorted(layer_weights.items())}


def _qvf_weighted_variant_from_args(args: argparse.Namespace) -> str:
    configured = str(getattr(args, "qvf_weighted_variant", "") or "").strip()
    if configured:
        return configured
    layer_weights = _qvf_layer_weights_from_args(args)
    return "" if not layer_weights else qvf_weighted_variant(layer_weights)


def _qvf_method_variant_from_args(args: argparse.Namespace) -> str:
    configured = str(getattr(args, "qvf_method_variant", "") or "").strip()
    if configured:
        return configured
    variant = _qvf_weighted_variant_from_args(args)
    return QVF_METHOD_VARIANT if not variant else f"{QVF_WEIGHTED_METHOD_VARIANT_PREFIX}_{variant}"


def _qvf_protocol_version_from_args(args: argparse.Namespace) -> str:
    configured = str(getattr(args, "qvf_protocol_version", "") or "").strip()
    if configured:
        return configured
    return QVF_WEIGHTED_PROTOCOL_VERSION if _qvf_layer_weights_from_args(args) else QVF_PROTOCOL_VERSION


def _qvf_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_qvf_token_fidelity", False))


def _per_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "use_projected_energy_retention", False))


def _per_active(args: argparse.Namespace) -> bool:
    return _per_enabled(args) and float(getattr(args, "per_lambda", 0.0)) > 0.0


def _qvf_candidate_dir(dataset_name: str | None, seed: int, beta: float) -> Path:
    return _dataset_root(dataset_name) / "innovation" / "QVF" / f"seed_{int(seed)}" / f"beta_{_qvf_beta_stem(float(beta))}"


def _per_lambda_stem(value: float) -> str:
    return _qvf_float_stem(float(value))


def _per_candidate_dir(dataset_name: str | None, seed: int, lambda_per: float) -> Path:
    return _dataset_root(dataset_name) / "innovation" / "PER" / f"seed_{int(seed)}" / f"lambda_{_per_lambda_stem(float(lambda_per))}"


def _seed_dir(args: argparse.Namespace) -> Path:
    if _qvf_enabled(args) and not args.baseline_root:
        qvf_output_root = str(getattr(args, "qvf_output_root", "") or "").strip()
        if qvf_output_root:
            return Path(qvf_output_root) / f"seed_{int(args.seed)}" / f"beta_{_qvf_beta_stem(float(args.qvf_beta))}"
        return _qvf_candidate_dir(args.dataset, int(args.seed), float(args.qvf_beta))
    if _per_enabled(args) and not args.baseline_root:
        per_output_root = str(getattr(args, "per_output_root", "") or "").strip()
        if per_output_root:
            return Path(per_output_root) / f"seed_{int(args.seed)}" / f"lambda_{_per_lambda_stem(float(args.per_lambda))}"
        return _per_candidate_dir(args.dataset, int(args.seed), float(args.per_lambda))
    baseline_root = Path(args.baseline_root) if args.baseline_root else _default_baseline_root(args.dataset)
    return baseline_root / f"seed_{int(args.seed)}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline vanilla KD runner: INT4 student with fixed qparams trained by CE + alpha*T^2*KL.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--baseline-root", default=None, help="Root for vanilla KD per-seed outputs; default: <dataset_root>/baseline/kd.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--teacher-checkpoint", default=None, help="Optional FP32 teacher checkpoint. Default: <dataset_root>/baseline/teacher/seed_<seed>/checkpoints/fp32_teacher_seed_<seed>.pt.")
    parser.add_argument("--force-stage-fp32-teacher", action="store_true", help="Overwrite the per-seed staged FP32 teacher checkpoint when its hash differs from the source.")
    parser.add_argument("--int4-checkpoint", default=None, help="Optional INT4 calibrated student checkpoint. Default: <dataset_root>/baseline/student/seed_<seed>/checkpoints/int4_student_calibrated_seed_<seed>.pt.")
    parser.add_argument("--qparams-csv", default=None, help="Optional INT4 qparams CSV. Default: <dataset_root>/baseline/student/seed_<seed>/qparams/int4_qparams.csv.")
    parser.add_argument("--int4-manifest", default=None, help="Optional INT4 student canonical manifest. Default: <dataset_root>/baseline/student/seed_<seed>/manifests/int4_student/run_manifest.json.")
    parser.add_argument("--epochs", type=_positive_int, default=int(FORMAL_DEFAULTS["epochs"]))
    parser.add_argument("--alpha", type=float, default=float(FORMAL_DEFAULTS["alpha"]))
    parser.add_argument("--temperature", type=_positive_float, default=float(FORMAL_DEFAULTS["temperature"]))
    parser.add_argument("--backbone-lr", type=_positive_float, default=float(FORMAL_DEFAULTS["backbone_lr"]))
    parser.add_argument("--classifier-lr", type=_positive_float, default=float(FORMAL_DEFAULTS["classifier_lr"]))
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=float(FORMAL_DEFAULTS["warmup_ratio"]))
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=8)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--best-checkpoint", default=None, help="Default: <dataset_root>/baseline/kd/seed_<seed>/checkpoints/vanilla_kd_best_seed_<seed>.pt.")
    parser.add_argument("--last-checkpoint", default=None, help="Default: <dataset_root>/baseline/kd/seed_<seed>/checkpoints/vanilla_kd_last_seed_<seed>.pt.")
    parser.add_argument("--train-log", default=None, help="Default: <dataset_root>/baseline/kd/seed_<seed>/logs/vanilla_kd_train_log.csv.")
    parser.add_argument("--manifest-path", default=None, help="Default: <dataset_root>/baseline/kd/seed_<seed>/manifests/vanilla_kd/run_manifest.json.")
    parser.add_argument("--export-root", default=None, help="Default: <dataset_root>/baseline/kd/seed_<seed>.")
    parser.add_argument("--export-name", default=BASELINE_NAME)
    parser.add_argument("--reference-root", default=None, help="Seed export root containing fp32_teacher reference artifacts. Default follows the selected FP32 teacher input root.")
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT)
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS)
    parser.add_argument("--export-all-token-layers", action="store_true")
    parser.add_argument("--max-export-payload-gib", type=float, default=16.0)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from last checkpoint when it exists.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore any existing last checkpoint and restart from the INT4 init.")
    parser.add_argument("--use-qvf-token-fidelity", action="store_true", help="Enable Stage1 QVF token fidelity on top of Vanilla KD.")
    parser.add_argument("--qvf-beta", type=_nonnegative_float, default=0.0, help="QVF loss coefficient. beta=0 is only for regression/debug runs.")
    parser.add_argument("--qvf-layer-weights", default=None, help="Optional weighted QVF sensitive layer coefficients, e.g. 1:0.25,9:0.75. Default keeps original equal layer mean.")
    parser.add_argument("--qvf-output-root", default=None, help="Optional QVF family root. Used as <root>/seed_<seed>/beta_<beta> when QVF is enabled.")
    parser.add_argument("--qvf-method-variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--qvf-protocol-version", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--qvf-weighted-variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--use-projected-energy-retention", action="store_true", help="Enable Stage1 PER on L9.fc1 on top of Vanilla KD.")
    parser.add_argument("--per-lambda", type=_nonnegative_float, default=0.0, help="PER loss coefficient. lambda=0 keeps the Vanilla KD loss active only.")
    parser.add_argument("--per-epsilon", type=_positive_float, default=PER_DEFAULT_EPS, help="PER epsilon for teacher norm denominators.")
    parser.add_argument("--per-output-root", default=None, help="Optional PER family root. Used as <root>/seed_<seed>/lambda_<lambda> when PER is enabled.")
    parser.add_argument("--per-loss-scale-probe", action="store_true", help="Run a train-split PER loss-scale probe only; no optimizer update and no eval/test.")
    parser.add_argument("--per-probe-batches", type=_positive_int, default=8, help="Maximum train batches for --per-loss-scale-probe.")
    parser.add_argument("--per-probe-output", default=None, help="Default: <PER seed dir>/per_loss_scale_probe.json.")
    parser.add_argument("--per-formal-lambda-set", default=None, help="Comma-separated formal PER lambdas selected by the user after the probe.")
    parser.add_argument("--per-method-variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--per-protocol-version", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow-artifact-path-migration", action="store_true", help="Allow same-hash INT4 manifest artifact paths to resolve to the current dataset root. QVF/PER default is strict.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    seed_dir = _seed_dir(args)
    qvf_mode = _qvf_enabled(args)
    per_mode = _per_enabled(args)
    teacher_seed_dir = _teacher_seed_dir(args.dataset, int(args.seed))
    student_seed_dir = _student_seed_dir(args.dataset, int(args.seed))
    teacher_checkpoint = (
        Path(args.teacher_checkpoint)
        if args.teacher_checkpoint
        else _teacher_checkpoint_from_seed_dir(teacher_seed_dir, int(args.seed))
    )
    reference_root = Path(args.reference_root) if args.reference_root else teacher_seed_dir
    qparams_csv = (
        Path(args.qparams_csv)
        if args.qparams_csv
        else _qparams_csv_from_seed_dir(student_seed_dir)
    )
    int4_manifest = (
        Path(args.int4_manifest)
        if args.int4_manifest
        else _int4_manifest_from_seed_dir(student_seed_dir)
    )
    int4_checkpoint = (
        Path(args.int4_checkpoint)
        if args.int4_checkpoint
        else _int4_checkpoint_from_seed_dir(student_seed_dir, int(args.seed))
    )
    return {
        "seed_dir": seed_dir,
        "teacher_seed_dir": teacher_seed_dir,
        "student_seed_dir": student_seed_dir,
        "teacher_checkpoint": teacher_checkpoint,
        "requested_teacher_checkpoint": Path(args.teacher_checkpoint) if args.teacher_checkpoint else teacher_checkpoint,
        "reference_root": reference_root,
        "int4_checkpoint": int4_checkpoint,
        "qparams_csv": qparams_csv,
        "int4_manifest": int4_manifest,
        "best_checkpoint": Path(args.best_checkpoint) if args.best_checkpoint else (
            seed_dir / "qvf_stage1_checkpoint.pt" if qvf_mode else (
                seed_dir / "per_stage1_checkpoint.pt" if per_mode else seed_dir / "checkpoints" / f"vanilla_kd_best_seed_{int(args.seed)}.pt"
            )
        ),
        "last_checkpoint": Path(args.last_checkpoint) if args.last_checkpoint else (
            seed_dir / "qvf_stage1_last.pt" if qvf_mode else (
                seed_dir / "per_stage1_last.pt" if per_mode else seed_dir / "checkpoints" / f"vanilla_kd_last_seed_{int(args.seed)}.pt"
            )
        ),
        "train_log": Path(args.train_log) if args.train_log else (
            seed_dir / "qvf_train_log.csv" if qvf_mode else (
                seed_dir / "per_train_log.csv" if per_mode else seed_dir / "logs" / "vanilla_kd_train_log.csv"
            )
        ),
        "manifest_path": Path(args.manifest_path) if args.manifest_path else (
            seed_dir / "qvf_manifest.json" if qvf_mode else (
                seed_dir / "per_manifest.json" if per_mode else seed_dir / "manifests" / BASELINE_NAME / "run_manifest.json"
            )
        ),
        "export_root": Path(args.export_root) if args.export_root else (
            seed_dir / "training_eval_export" if (qvf_mode or per_mode) else seed_dir
        ),
        "metrics_csv": seed_dir / ("qvf_metrics.csv" if qvf_mode else ("per_metrics.csv" if per_mode else "vanilla_kd_metrics.csv")),
        "per_probe_output": Path(args.per_probe_output) if getattr(args, "per_probe_output", None) else seed_dir / "per_loss_scale_probe.json",
    }


def _seed_everything(seed: int) -> None:
    seed_fn = getattr(project_config, "seed_everything", None)
    if callable(seed_fn):
        seed_fn(int(seed))
        return
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _rng_state_payload(device: torch.device) -> dict[str, Any]:
    capture_cuda = bool(device.type == "cuda" and torch.cuda.is_available())
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if capture_cuda else [],
        "torch_cuda_device_count": torch.cuda.device_count() if capture_cuda else 0,
        "resume_granularity": "epoch_boundary",
        "dataloader_shuffle_state_semantic": "main_process_rng_state_before_next_epoch_iterator",
    }


def _restore_rng_state(payload: Mapping[str, Any], *, device: torch.device) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("resume checkpoint missing rng_state mapping")
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    if "numpy_random_state" in payload:
        np.random.set_state(payload["numpy_random_state"])
    torch_cpu_state = payload.get("torch_cpu_rng_state")
    if torch_cpu_state is None:
        raise RuntimeError("resume checkpoint missing torch_cpu_rng_state")
    torch.set_rng_state(torch_cpu_state)
    cuda_states = list(payload.get("torch_cuda_rng_state_all") or [])
    expected_cuda_count = int(payload.get("torch_cuda_device_count") or 0)
    if cuda_states:
        if not torch.cuda.is_available() or device.type != "cuda":
            raise RuntimeError("resume checkpoint contains CUDA RNG state but current run is not CUDA")
        if len(cuda_states) != torch.cuda.device_count() or expected_cuda_count != torch.cuda.device_count():
            raise RuntimeError(
                "resume CUDA device count mismatch: "
                f"checkpoint={expected_cuda_count}, states={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _reject_amp(args: argparse.Namespace) -> None:
    if bool(getattr(args, "amp", False)):
        raise RuntimeError("Baseline vanilla KD formal training forbids AMP; remove --amp and train with FP32 logits/loss.")


def _validate_qvf_args(args: argparse.Namespace) -> None:
    enabled = _qvf_enabled(args)
    beta = float(getattr(args, "qvf_beta", 0.0))
    if not enabled and beta != 0.0:
        raise RuntimeError("qvf_beta is only valid when --use-qvf-token-fidelity is enabled")
    if not enabled and str(getattr(args, "qvf_layer_weights", "") or "").strip():
        raise RuntimeError("qvf_layer_weights is only valid when --use-qvf-token-fidelity is enabled")
    if enabled:
        _qvf_layer_weights_from_args(args)
    if enabled and not (0.0 < float(QVF_TOPK_RATIO) <= 1.0):
        raise RuntimeError(f"QVF_TOPK_RATIO must be in (0, 1], got {QVF_TOPK_RATIO!r}")


def _parse_per_formal_lambda_set(value: str | None) -> tuple[float, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    result: list[float] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        parsed = float(item)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise argparse.ArgumentTypeError(f"formal PER lambda must be finite and non-negative, got {item!r}")
        result.append(float(parsed))
    if len(result) != 3:
        raise argparse.ArgumentTypeError(f"formal PER lambda set must contain exactly 3 values, got {len(result)}")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"formal PER lambda set contains duplicates: {result!r}")
    return tuple(result)


def _per_formal_lambda_set_from_args(args: argparse.Namespace) -> tuple[float, ...]:
    return _parse_per_formal_lambda_set(getattr(args, "per_formal_lambda_set", None))


def _per_method_variant_from_args(args: argparse.Namespace) -> str:
    configured = str(getattr(args, "per_method_variant", "") or "").strip()
    return configured or PER_METHOD_VARIANT


def _per_protocol_version_from_args(args: argparse.Namespace) -> str:
    configured = str(getattr(args, "per_protocol_version", "") or "").strip()
    return configured or PER_PROTOCOL_VERSION


def _validate_per_args(args: argparse.Namespace) -> None:
    enabled = _per_enabled(args)
    lambda_per = float(getattr(args, "per_lambda", 0.0))
    probe_enabled = bool(getattr(args, "per_loss_scale_probe", False))
    if enabled and _qvf_enabled(args):
        raise RuntimeError("PER and QVF are mutually exclusive for this Stage1 runner")
    if not enabled and lambda_per != 0.0:
        raise RuntimeError("per_lambda is only valid when --use-projected-energy-retention is enabled")
    if probe_enabled and not enabled:
        raise RuntimeError("--per-loss-scale-probe requires --use-projected-energy-retention")
    if not enabled and str(getattr(args, "per_output_root", "") or "").strip():
        raise RuntimeError("per_output_root is only valid when --use-projected-energy-retention is enabled")
    _per_formal_lambda_set_from_args(args)
    eps = float(getattr(args, "per_epsilon", PER_DEFAULT_EPS))
    if not math.isfinite(eps) or eps <= 0.0:
        raise RuntimeError(f"per_epsilon must be positive finite, got {eps!r}")
    if enabled and not (0.0 < float(PER_TOPK_RATIO) <= 1.0):
        raise RuntimeError(f"PER_TOPK_RATIO must be in (0, 1], got {PER_TOPK_RATIO!r}")


def _require_equal(name: str, existing: Any, expected: Any) -> None:
    if existing is None or expected is None or str(existing) != str(expected):
        raise RuntimeError(f"KD provenance mismatch for {name}: existing={existing!r}, expected={expected!r}")


def _require_qvf_weight_payload(name: str, actual: Any, expected: Mapping[str, float]) -> None:
    if not isinstance(actual, Mapping):
        raise RuntimeError(f"{name} must be a mapping, got {type(actual)!r}")
    actual_keys = set(str(key) for key in actual.keys())
    expected_keys = set(str(key) for key in expected.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(f"{name} keys mismatch: actual={sorted(actual_keys)!r}, expected={sorted(expected_keys)!r}")
    for key, expected_value in expected.items():
        _require_float_close(f"{name}.{key}", actual.get(key), float(expected_value))


def _require_qvf_protocol_matches_args(context: str, args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    expected = _qvf_protocol_for_args(args)
    _require_equal(f"{context}.method_variant", payload.get("method_variant"), expected.get("method_variant"))
    protocol = payload.get("qvf_protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError(f"{context}.qvf_protocol must be a mapping")
    _require_equal(f"{context}.qvf_protocol.protocol_version", protocol.get("protocol_version"), expected.get("protocol_version"))
    _require_equal(f"{context}.qvf_protocol.method_variant", protocol.get("method_variant"), expected.get("method_variant"))
    _require_float_close(f"{context}.qvf_protocol.qvf_beta", protocol.get("qvf_beta"), float(args.qvf_beta))
    _require_float_close(f"{context}.qvf_protocol.topk_ratio", protocol.get("topk_ratio"), float(QVF_TOPK_RATIO))
    expected_weights = expected.get("sensitive_layer_weights")
    if expected_weights or "layer_weight_policy" in protocol:
        _require_equal(f"{context}.qvf_protocol.layer_weight_policy", protocol.get("layer_weight_policy"), expected.get("layer_weight_policy"))
    if expected_weights:
        _require_equal(f"{context}.qvf_weighted_variant", payload.get("qvf_weighted_variant"), expected.get("qvf_weighted_variant"))
        _require_qvf_weight_payload(f"{context}.sensitive_layer_weights", payload.get("sensitive_layer_weights"), expected_weights)
        _require_qvf_weight_payload(f"{context}.qvf_layer_weights", payload.get("qvf_layer_weights"), expected_weights)
        _require_equal(f"{context}.qvf_protocol.qvf_weighted_variant", protocol.get("qvf_weighted_variant"), expected.get("qvf_weighted_variant"))
        _require_qvf_weight_payload(f"{context}.qvf_protocol.sensitive_layer_weights", protocol.get("sensitive_layer_weights"), expected_weights)
        _require_qvf_weight_payload(f"{context}.qvf_protocol.qvf_layer_weights", protocol.get("qvf_layer_weights"), expected_weights)


def _require_per_protocol_matches_args(context: str, args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    expected = _per_protocol_for_args(args)
    _require_equal(f"{context}.method_variant", payload.get("method_variant"), expected.get("method_variant"))
    protocol = payload.get("per_protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError(f"{context}.per_protocol must be a mapping")
    _require_equal(f"{context}.per_protocol.protocol_version", protocol.get("protocol_version"), expected.get("protocol_version"))
    _require_equal(f"{context}.per_protocol.method_variant", protocol.get("method_variant"), expected.get("method_variant"))
    _require_float_close(f"{context}.per_protocol.per_lambda", protocol.get("per_lambda"), float(args.per_lambda))
    _require_float_close(f"{context}.per_protocol.topk_ratio", protocol.get("topk_ratio"), float(PER_TOPK_RATIO))
    _require_float_close(f"{context}.per_protocol.epsilon", protocol.get("epsilon"), float(args.per_epsilon))
    _require_equal(f"{context}.per_protocol.target_layers", protocol.get("target_layers"), expected.get("target_layers"))
    _require_equal(f"{context}.per_protocol.module_paths", protocol.get("module_paths"), expected.get("module_paths"))


def _drop_legacy_hash_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("source_beats_checkpoint_hash", "teacher_checkpoint_hash", "qparams_csv_hash"):
        payload.pop(key, None)
    return payload


def _teacher_source_beats_checkpoint(payload: Mapping[str, Any], *, context: str) -> Any:
    value = payload.get("source_beats_checkpoint")
    identity_payload = payload.get("run_identity_payload")
    if (value is None or not str(value).strip()) and isinstance(identity_payload, Mapping):
        value = identity_payload.get("source_beats_checkpoint")
    if value is None or not str(value).strip():
        raise RuntimeError(f"{context} missing canonical source_beats_checkpoint field")
    return value


def _load_required_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"required {name} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"required {name} must contain a JSON object: {path}")
    return dict(payload)


def _manifest_path_value(manifest_path: Path, manifest: Mapping[str, Any], field: str) -> Path:
    value = manifest.get(field)
    if value is None or not str(value).strip():
        raise RuntimeError(f"INT4 manifest missing required path field {field!r}: {manifest_path}")
    target = Path(str(value))
    if not target.is_absolute():
        target = manifest_path.parent / target
    return target


def _dataset_config_path(dataset_name: str | None, key: str, fallback: Path) -> Path:
    datasets = getattr(project_config, "DATASETS", {})
    if isinstance(datasets, Mapping) and dataset_name in datasets:
        value = datasets[str(dataset_name)].get(key)
        if value is not None:
            return Path(value)
    return Path(fallback)


def _dataset_calibration_csv(paths: Any) -> Path:
    return _dataset_config_path(
        getattr(paths, "name", None),
        "CALIBRATION_CSV",
        Path(getattr(paths, "root")) / "splits" / "calibration.csv",
    )


def _dataset_calibration_manifest(paths: Any) -> Path:
    return _dataset_config_path(
        getattr(paths, "name", None),
        "CALIBRATION_MANIFEST",
        Path(getattr(paths, "root")) / "splits" / "calibration.manifest.json",
    )


def _resolve_migrated_artifact_path(
    *,
    recorded_path: Path,
    current_path: Path,
    expected_sha256: Any,
    name: str,
    allow_path_migration: bool,
) -> tuple[Path, str, bool]:
    expected = None if expected_sha256 is None else str(expected_sha256)
    recorded_hash = file_sha256(recorded_path)
    if recorded_hash and (expected is None or str(recorded_hash) == expected):
        return recorded_path, str(recorded_hash), False
    current_hash = file_sha256(current_path)
    if current_hash and (expected is None or str(current_hash) == expected):
        migrated = recorded_path.resolve(strict=False) != current_path.resolve(strict=False)
        if migrated and not allow_path_migration:
            raise RuntimeError(
                f"{name} path migration is disabled for this run: "
                f"recorded={recorded_path.resolve(strict=False)} current={current_path.resolve(strict=False)} "
                f"sha256={current_hash}"
            )
        return current_path, str(current_hash), migrated
    raise FileNotFoundError(
        f"{name} could not be resolved with expected sha256={expected!r}: "
        f"recorded={recorded_path.resolve(strict=False)} recorded_sha256={recorded_hash!r}; "
        f"current={current_path.resolve(strict=False)} current_sha256={current_hash!r}"
    )


def _assert_fp32_teacher_payload(payload: Mapping[str, Any], *, context: str) -> None:
    model_kind = str(payload.get("model_kind") or "")
    if model_kind != "fp32_teacher":
        raise RuntimeError(f"{context} must load an FP32 teacher checkpoint, got model_kind={model_kind!r}")


def _validate_label_tensor(value: Tensor, *, num_classes: int, context: str) -> Tensor:
    labels = value.detach()
    if labels.dtype == torch.bool:
        raise TypeError(f"{context}: label_id tensor must not be bool")
    if torch.is_complex(labels):
        raise TypeError(f"{context}: label_id tensor must not be complex")
    if torch.is_floating_point(labels):
        if not bool(torch.isfinite(labels).all().item()):
            raise ValueError(f"{context}: label_id tensor contains NaN/Inf")
        if not bool(torch.equal(labels, labels.round())):
            raise ValueError(f"{context}: label_id tensor contains non-integer values")
    labels_long = labels.to(dtype=torch.long)
    if labels_long.ndim != 1:
        raise ValueError(f"{context}: label_id tensor must be 1D [B], got shape={tuple(labels_long.shape)}")
    if int(labels_long.numel()) <= 0:
        raise ValueError(f"{context}: label_id tensor is empty")
    if bool((labels_long < 0).any().item()) or bool((labels_long >= int(num_classes)).any().item()):
        raise ValueError(f"{context}: label_id out of range [0,{int(num_classes)})")
    return labels_long


def _expected_split_hashes(paths: Any, calibration_csv: Path, *, include_test: bool = True) -> dict[str, str | None]:
    payload = {
        "train": file_sha256(paths.train_csv),
        "eval": file_sha256(paths.eval_csv),
        "calibration": file_sha256(calibration_csv),
    }
    if include_test:
        payload["test"] = file_sha256(paths.test_csv)
    return payload


def _validate_int4_student_provenance(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    teacher_checkpoint: Path,
    int4_checkpoint: Path | None = None,
    qparams_csv: Path,
    paths: Any,
    validate_test_split: bool = True,
    allow_path_migration: bool = False,
) -> dict[str, Any]:
    _require_equal("int4.model_kind", manifest.get("model_kind"), "int4_student_init")
    _require_equal("int4.teacher_checkpoint_sha256", manifest.get("teacher_checkpoint_sha256"), file_sha256(teacher_checkpoint))
    _require_equal("int4.label_mapping_hash", manifest.get("label_mapping_hash"), file_sha256(paths.label_mapping))

    checkpoint_sha256 = None
    manifest_checkpoint = None
    if int4_checkpoint is not None:
        manifest_checkpoint = _manifest_path_value(manifest_path, manifest, "checkpoint")
        checkpoint_sha256 = file_sha256(int4_checkpoint)
        _require_equal("int4.checkpoint_sha256", manifest.get("checkpoint_sha256"), checkpoint_sha256)
        _require_equal("int4.int4_calibrated_checkpoint_sha256", manifest.get("int4_calibrated_checkpoint_sha256"), checkpoint_sha256)
        if manifest_checkpoint.resolve(strict=False) != int4_checkpoint.resolve(strict=False) and not allow_path_migration:
            raise RuntimeError(
                "INT4 checkpoint path migration is disabled for this run: "
                f"recorded={manifest_checkpoint.resolve(strict=False)} current={int4_checkpoint.resolve(strict=False)}"
            )

    manifest_qparams_csv = _manifest_path_value(manifest_path, manifest, "qparams_csv")
    qparams_csv_sha256 = file_sha256(qparams_csv)
    _require_equal("int4.qparams_csv_sha256", manifest.get("qparams_csv_sha256"), qparams_csv_sha256)
    if manifest_qparams_csv.resolve(strict=False) != qparams_csv.resolve(strict=False) and not allow_path_migration:
        raise RuntimeError(
            "INT4 qparams_csv path migration is disabled for this run: "
            f"recorded={manifest_qparams_csv.resolve(strict=False)} current={qparams_csv.resolve(strict=False)}"
        )
    qparams = qck.load_qparams_csv(qparams_csv)
    qparams_hash = qck.qparams_hash(qparams)
    _require_equal("int4.qparams_hash", manifest.get("qparams_hash"), qparams_hash)

    recorded_calibration_csv = _manifest_path_value(manifest_path, manifest, "calibration_csv")
    calibration_csv, calibration_csv_hash, calibration_csv_path_migrated = _resolve_migrated_artifact_path(
        recorded_path=recorded_calibration_csv,
        current_path=_dataset_calibration_csv(paths),
        expected_sha256=manifest.get("calibration_csv_hash"),
        name="INT4 calibration_csv",
        allow_path_migration=bool(allow_path_migration),
    )
    _require_equal("int4.calibration_csv_hash", manifest.get("calibration_csv_hash"), calibration_csv_hash)
    calibration_manifest_hash = None
    calibration_manifest = None
    calibration_manifest_path_migrated = False
    calibration_manifest_value = manifest.get("calibration_manifest")
    if calibration_manifest_value is not None and str(calibration_manifest_value).strip():
        recorded_calibration_manifest = Path(str(calibration_manifest_value))
        if not recorded_calibration_manifest.is_absolute():
            recorded_calibration_manifest = manifest_path.parent / recorded_calibration_manifest
        calibration_manifest, calibration_manifest_hash, calibration_manifest_path_migrated = _resolve_migrated_artifact_path(
            recorded_path=recorded_calibration_manifest,
            current_path=_dataset_calibration_manifest(paths),
            expected_sha256=manifest.get("calibration_manifest_hash"),
            name="INT4 calibration_manifest",
            allow_path_migration=bool(allow_path_migration),
        )
        _require_equal("int4.calibration_manifest_hash", manifest.get("calibration_manifest_hash"), calibration_manifest_hash)
    _require_equal("int4.calibration_hash", manifest.get("calibration_hash"), calibration_manifest_hash or calibration_csv_hash)

    qparams_metadata = qck.qparams_csv_metadata(qparams_csv)
    metadata_calibration_hash = qparams_metadata.get("calibration_hash")
    if metadata_calibration_hash and str(metadata_calibration_hash) != str(calibration_csv_hash):
        raise RuntimeError(
            "KD qparams_csv calibration hash does not match INT4 calibration CSV: "
            f"csv={metadata_calibration_hash}, calibration_csv={calibration_csv_hash}"
        )

    existing_split_hashes = manifest.get("split_hashes")
    if not isinstance(existing_split_hashes, Mapping):
        raise RuntimeError(f"INT4 manifest missing split_hashes mapping: {manifest_path}")
    expected_split_hashes = _expected_split_hashes(paths, calibration_csv, include_test=bool(validate_test_split))
    for key, value in expected_split_hashes.items():
        _require_equal(f"int4.split_hashes.{key}", existing_split_hashes.get(key), value)
    if bool(validate_test_split):
        _require_equal("int4.split_hash", manifest.get("split_hash"), canonical_json_hash(expected_split_hashes))

    raw_layers = manifest.get("quantized_layers")
    if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes)) or not raw_layers:
        raise RuntimeError(f"INT4 manifest missing non-empty quantized_layers: {manifest_path}")
    layers = [dict(row) for row in raw_layers]
    quantized_layers_hash = canonical_json_hash(layers)
    _require_equal("int4.quantized_layers_hash", manifest.get("quantized_layers_hash"), quantized_layers_hash)
    return {
        "manifest_path": path_str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_run_identity_hash": manifest.get("run_identity_hash"),
        "manifest_hash": manifest.get("run_manifest_hash"),
        "checkpoint": path_str(int4_checkpoint) if int4_checkpoint is not None else path_str(manifest.get("checkpoint")),
        "manifest_checkpoint": path_str(manifest_checkpoint),
        "checkpoint_path_migrated": bool(
            int4_checkpoint is not None
            and manifest_checkpoint is not None
            and manifest_checkpoint.resolve(strict=False) != int4_checkpoint.resolve(strict=False)
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_csv": calibration_csv,
        "manifest_calibration_csv": path_str(recorded_calibration_csv),
        "calibration_csv_path_migrated": bool(calibration_csv_path_migrated),
        "calibration_manifest": calibration_manifest,
        "calibration_manifest_path_migrated": bool(calibration_manifest_path_migrated),
        "calibration_csv_hash": calibration_csv_hash,
        "calibration_manifest_hash": calibration_manifest_hash,
        "calibration_hash": str(manifest.get("calibration_hash")),
        "qparams_hash": qparams_hash,
        "qparams_csv_sha256": qparams_csv_sha256,
        "qparams_csv": qparams_csv,
        "manifest_qparams_csv": path_str(manifest_qparams_csv),
        "qparams_csv_path_migrated": bool(manifest_qparams_csv.resolve(strict=False) != qparams_csv.resolve(strict=False)),
        "artifact_path_migration_allowed": bool(allow_path_migration),
        "quantized_layers": layers,
        "quantized_layers_hash": quantized_layers_hash,
        "validated_split_hash_keys": sorted(expected_split_hashes),
        "full_split_hash_recomputed": bool(validate_test_split),
    }


def _validate_loaded_int4_checkpoint_payload(
    *,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    int4_provenance: Mapping[str, Any],
    teacher_checkpoint: Path,
    paths: Any,
) -> None:
    checkpoint_kind = str(checkpoint.get("model_kind") or "")
    _require_equal("int4_checkpoint.model_kind", checkpoint_kind, "int4_student_init")
    expected_checkpoint_sha256 = str(int4_provenance.get("checkpoint_sha256") or "")
    if not expected_checkpoint_sha256:
        raise RuntimeError("INT4 provenance missing checkpoint_sha256 before KD training")
    _require_equal("int4_checkpoint.file_sha256", file_sha256(checkpoint_path), expected_checkpoint_sha256)
    _require_equal("int4_checkpoint.teacher_checkpoint_sha256", checkpoint.get("teacher_checkpoint_sha256"), file_sha256(teacher_checkpoint))
    _require_equal("int4_checkpoint.label_mapping_hash", checkpoint.get("label_mapping_hash"), file_sha256(paths.label_mapping))
    _require_equal("int4_checkpoint.qparams_hash", checkpoint.get("qparams_hash"), int4_provenance.get("qparams_hash"))
    _require_equal("int4_checkpoint.qparams_csv_sha256", checkpoint.get("qparams_csv_sha256"), int4_provenance.get("qparams_csv_sha256"))
    _require_equal("int4_checkpoint.calibration_csv_hash", checkpoint.get("calibration_csv_hash"), int4_provenance.get("calibration_csv_hash"))
    _require_equal("int4_checkpoint.calibration_hash", checkpoint.get("calibration_hash"), int4_provenance.get("calibration_hash"))

    raw_layers = checkpoint.get("quantized_layers")
    if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes)) or not raw_layers:
        raise RuntimeError(f"INT4 checkpoint missing non-empty quantized_layers: {checkpoint_path}")
    checkpoint_layers_hash = canonical_json_hash([dict(row) for row in raw_layers])
    _require_equal("int4_checkpoint.quantized_layers_hash", checkpoint.get("quantized_layers_hash"), int4_provenance.get("quantized_layers_hash"))
    _require_equal("int4_checkpoint.quantized_layers_semantic_hash", checkpoint_layers_hash, int4_provenance.get("quantized_layers_hash"))


def _assert_quant_training_state(
    model: torch.nn.Module,
    controller: Any,
    int4_provenance: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    audit = audit_quant_state(
        model,
        quant_controller=controller,
        expected_quantized_layers=int4_provenance["quantized_layers"],
        expected_qparams_hash=str(int4_provenance["qparams_hash"]),
    )
    audit["context"] = context
    return audit


def _qvf_protocol(
    beta: float,
    *,
    layer_weights: Mapping[int, float] | None = None,
    method_variant: str | None = None,
    protocol_version: str | None = None,
    weighted_variant: str | None = None,
) -> dict[str, Any]:
    weighted = bool(layer_weights)
    method = str(method_variant or (QVF_METHOD_VARIANT if not weighted else f"{QVF_WEIGHTED_METHOD_VARIANT_PREFIX}_{weighted_variant or qvf_weighted_variant(layer_weights or {})}"))
    version = str(protocol_version or (QVF_PROTOCOL_VERSION if not weighted else QVF_WEIGHTED_PROTOCOL_VERSION))
    payload = {
        "protocol_version": version,
        "method_variant": method,
        "use_qvf_token_fidelity": True,
        "qvf_beta": float(beta),
        "qvf_loss_definition": (
            "mean over samples of weighted sensitive-layer teacher-energy normalized relative distortion on FP32 teacher Top-K tokens"
            if weighted
            else "mean over samples of mean sensitive-layer teacher-energy normalized relative distortion on FP32 teacher Top-K tokens"
        ),
        "loss_composition": "L_CE + alpha * T^2 * L_KD + beta * L_QVF",
        "sensitive_layer_list": [f"L{layer}.{QVF_MODULE_NAME}" for layer in QVF_SENSITIVE_LAYERS],
        "exact_module_paths": [f"encoder.layers.{layer}.{QVF_MODULE_NAME}" for layer in QVF_SENSITIVE_LAYERS],
        "layer_weight_policy": "weighted_sensitive_layer_mean" if weighted else "equal_sensitive_layer_mean",
        "activation_hook_semantic": "FFN fc1 input activation",
        "student_activation_semantic": "post_fake_quant_dequant_input_activation_with_autograd_graph",
        "teacher_activation_semantic": "fp32_teacher_fc1_input_activation_detached",
        "student_layerdrop_disabled_during_qvf_capture": True,
        "teacher_layerdrop_disabled_during_qvf_capture": True,
        "dropout_training_state_preserved": True,
        "teacher_token_selection_source": "fp32_token_energy",
        "topk_unit": "token",
        "topk_source": "fp32_token_energy",
        "topk_selected_on": "fp32",
        "topk_ratio": float(QVF_TOPK_RATIO),
        "topk_k_policy": "ceil(valid_token_count * topk_ratio), clipped to valid_token_count",
        "topk_policy": "per_sample_teacher_token_l2_norm_descending",
        "padding_policy": "padding tokens excluded before Top-K and loss",
        "teacher_no_grad": True,
        "topk_index_no_grad": True,
        "qvf_loss_backpropagates_to_student_activation": True,
        "epsilon": float(DEFAULT_EPS),
        "qparams_frozen_required": True,
        "fake_quant_enabled_required": True,
        "observer_enabled_required": False,
        "test_isolation": {
            "test_split_touched": False,
            "test_csv_read": False,
            "full_csv_read": False,
            "test_loader_built": False,
            "test_forward_run": False,
            "test_metrics_exported": False,
            "test_predictions_exported": False,
            "test_path_recorded": False,
            "label_bucket_source": "train_csv.train_label_n_or_train_counts",
        },
    }
    if weighted:
        weights_payload = _qvf_layer_weight_payload(layer_weights or {})
        payload["qvf_weighted_variant"] = str(weighted_variant or qvf_weighted_variant(layer_weights or {}))
        payload["sensitive_layer_weights"] = weights_payload
        payload["qvf_layer_weights"] = weights_payload
        payload["layer_weight_sum"] = float(sum(float(item) for item in (layer_weights or {}).values()))
    return payload


def _qvf_protocol_for_args(args: argparse.Namespace) -> dict[str, Any]:
    layer_weights = _qvf_layer_weights_from_args(args)
    return _qvf_protocol(
        float(args.qvf_beta),
        layer_weights=layer_weights,
        method_variant=_qvf_method_variant_from_args(args),
        protocol_version=_qvf_protocol_version_from_args(args),
        weighted_variant=_qvf_weighted_variant_from_args(args),
    )


def _per_protocol(
    lambda_per: float,
    *,
    epsilon: float,
    method_variant: str | None = None,
    protocol_version: str | None = None,
    formal_lambda_set: Sequence[float] | None = None,
) -> dict[str, Any]:
    formal_lambdas = [float(item) for item in tuple(formal_lambda_set or ())]
    return {
        "protocol_version": str(protocol_version or PER_PROTOCOL_VERSION),
        "method_name": PER_METHOD_NAME,
        "method_variant": str(method_variant or PER_METHOD_VARIANT),
        "stage": "Stage1 Vanilla KD",
        "loss_definition": "L_CE + alpha * T^2 * L_KD + lambda_per * L_PER",
        "loss_composition": "L_CE + alpha * T^2 * L_KD + lambda_per * L_PER",
        "use_projected_energy_retention": True,
        "per_lambda": float(lambda_per),
        "target_layers": [f"L{layer}.{PER_MODULE_NAME}" for layer in PER_LAYERS],
        "module_paths": [f"encoder.layers.{layer}.{PER_MODULE_NAME}" for layer in PER_LAYERS],
        "per_layers": [f"L{layer}.{PER_MODULE_NAME}" for layer in PER_LAYERS],
        "activation_hook_semantic": "FFN fc1 input activation",
        "student_activation_semantic": "post_fake_quant_with_autograd_graph",
        "teacher_activation_semantic": "FP32 teacher fc1 input",
        "teacher_token_selection": "teacher-only FP32 token energy Top-K",
        "teacher_token_selection_source": "fp32_token_energy",
        "topk_unit": "token",
        "topk_source": "fp32_token_energy",
        "topk_selected_on": "fp32",
        "topk_ratio": float(PER_TOPK_RATIO),
        "effective_topk_policy": "ceil(valid_token_count * topk_ratio), clipped to valid_token_count",
        "topk_k_policy": "ceil(valid_token_count * topk_ratio), clipped to valid_token_count",
        "topk_policy": "per_sample_teacher_token_l2_norm_descending",
        "padding_policy": "padding tokens excluded before Top-K and loss",
        "epsilon": float(epsilon),
        "sample_weighting": "sample_equal",
        "tail_weighting": False,
        "geometry_in_loss": False,
        "layer_weighting": False,
        "teacher_no_grad": True,
        "teacher_activation_detached": True,
        "topk_index_no_grad": True,
        "student_layerdrop_disabled_during_per_capture": True,
        "teacher_layerdrop_disabled_during_per_capture": True,
        "dropout_training_state_preserved": True,
        "student_activation_requires_grad_sanity": "checked_once_on_first_valid_PER_batch",
        "per_loss_backpropagates_to_student_activation": True,
        "qparams_frozen_required": True,
        "fake_quant_enabled_required": True,
        "observer_enabled_required": False,
        "formal_lambda_set": formal_lambdas,
        "formal_lambda_policy": "exactly_three_lambdas_after_manual_loss_scale_probe" if formal_lambdas else "not_declared_for_debug_or_probe",
        "selection_rule": {
            "selection_split": "eval",
            "guardrail_first": {
                "delta_overall_accuracy_min": -0.005,
                "delta_macro_f1_min": -0.005,
                "delta_many_accuracy_min": -0.01,
            },
            "primary": "tail_macro_f1",
            "tie_break": ["tail_accuracy", "medium_macro_f1", "medium_accuracy"],
            "few_used_for_selection": False,
            "geometry_required": "finite_only",
            "selected_candidate_and_method_success_are_separate": True,
        },
        "predefined_stop_criterion": {
            "status": "mechanism_positive_task_negative",
            "condition": (
                "all three reasonable lambdas improve projected retention and OER/distortion, "
                "but medium/tail task metrics do not improve over Vanilla KD"
            ),
            "auto_continue_to_new_losses": False,
        },
        "test_isolation": {
            "test_split_touched": False,
            "test_csv_read": False,
            "full_csv_read": False,
            "test_loader_built": False,
            "test_forward_run": False,
            "test_metrics_exported": False,
            "test_predictions_exported": False,
            "test_path_recorded": False,
            "label_bucket_source": "train_csv.train_label_n_or_train_counts",
        },
    }


def _per_protocol_for_args(args: argparse.Namespace) -> dict[str, Any]:
    return _per_protocol(
        float(getattr(args, "per_lambda", 0.0)),
        epsilon=float(getattr(args, "per_epsilon", PER_DEFAULT_EPS)),
        method_variant=_per_method_variant_from_args(args),
        protocol_version=_per_protocol_version_from_args(args),
        formal_lambda_set=_per_formal_lambda_set_from_args(args),
    )


class _TeacherQvfCapture:
    def __init__(self, model: torch.nn.Module, *, layer_indices: Sequence[int]) -> None:
        self.model = model
        self.layer_indices = tuple(int(item) for item in layer_indices)
        self.handles: list[Any] = []
        self.padding_mask: Tensor | None = None
        self.records: dict[int, Tensor] = {}
        self.valid_masks: dict[int, Tensor] = {}

    def set_batch_context(self, *, padding_mask: Tensor) -> None:
        self.padding_mask = padding_mask.detach().to(dtype=torch.bool)
        self.records.clear()
        self.valid_masks.clear()

    def register_hooks(self) -> None:
        for layer_idx in self.layer_indices:
            module = get_ffn_module(self.model, int(layer_idx), QVF_MODULE_NAME)

            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], *, layer_idx: int = int(layer_idx)) -> None:
                if self.padding_mask is None:
                    raise RuntimeError("QVF teacher capture missing token padding mask")
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise RuntimeError(f"QVF teacher capture expected tensor input at L{layer_idx}.{QVF_MODULE_NAME}")
                canonical = canonicalize_activation_btc(
                    inputs[0].detach().float(),
                    padding_mask=self.padding_mask,
                    context=f"qvf.teacher.L{layer_idx}.{QVF_MODULE_NAME}.input",
                )
                if canonical.padding_mask is None:
                    raise RuntimeError(f"QVF teacher capture missing aligned padding mask for L{layer_idx}.{QVF_MODULE_NAME}")
                self.records[int(layer_idx)] = canonical.tensor.detach()
                self.valid_masks[int(layer_idx)] = (~canonical.padding_mask).detach().to(dtype=torch.bool)

            self.handles.append(module.register_forward_pre_hook(hook))

    def pop(self) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        missing = [layer for layer in self.layer_indices if int(layer) not in self.records]
        if missing:
            raise RuntimeError(f"QVF teacher capture missing layer(s): {missing}")
        records = {int(layer): self.records[int(layer)] for layer in self.layer_indices}
        valid_masks = {int(layer): self.valid_masks[int(layer)] for layer in self.layer_indices}
        self.records.clear()
        self.valid_masks.clear()
        return records, valid_masks

    def clear_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.padding_mask = None
        self.records.clear()
        self.valid_masks.clear()


def _qvf_modules(model: torch.nn.Module, *, layer_indices: Sequence[int]) -> dict[int, torch.nn.Module]:
    return {
        int(layer_idx): get_ffn_module(model, int(layer_idx), QVF_MODULE_NAME)
        for layer_idx in layer_indices
    }


@contextmanager
def _disable_encoder_layerdrop_for_qvf(model: torch.nn.Module) -> Any:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "layerdrop"):
        yield
        return
    previous = getattr(encoder, "layerdrop")
    setattr(encoder, "layerdrop", 0.0)
    try:
        yield
    finally:
        setattr(encoder, "layerdrop", previous)


def _set_student_qvf_capture(modules: Mapping[int, torch.nn.Module], *, enabled: bool) -> None:
    for layer_idx, module in modules.items():
        setter = getattr(module, "set_qvf_activation_capture", None)
        if not callable(setter):
            raise RuntimeError(f"QVF requires QuantLinear capture support at L{int(layer_idx)}.{QVF_MODULE_NAME}")
        setter(bool(enabled))


def _collect_student_qvf_capture(
    modules: Mapping[int, torch.nn.Module],
    *,
    token_padding_mask: Tensor,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    expected_valid = (~token_padding_mask.detach().bool()).cpu()
    activations: dict[int, Tensor] = {}
    valid_masks: dict[int, Tensor] = {}
    for layer_idx, module in modules.items():
        getter = getattr(module, "get_qvf_activation_tensor", None)
        if not callable(getter):
            raise RuntimeError(f"QVF requires QuantLinear get_qvf_activation_tensor() at L{int(layer_idx)}.{QVF_MODULE_NAME}")
        activation, valid_mask = getter()
        if activation is None:
            raise RuntimeError(f"QVF student capture missing post-fake-quant activation at L{int(layer_idx)}.{QVF_MODULE_NAME}")
        if valid_mask is None:
            raise RuntimeError(f"QVF student capture missing valid token mask at L{int(layer_idx)}.{QVF_MODULE_NAME}")
        if tuple(valid_mask.shape) != tuple(token_padding_mask.shape):
            raise RuntimeError(
                f"QVF valid mask shape mismatch at L{int(layer_idx)}.{QVF_MODULE_NAME}: "
                f"{tuple(valid_mask.shape)} vs {tuple(token_padding_mask.shape)}"
            )
        if not bool(torch.equal(valid_mask.detach().cpu().bool(), expected_valid)):
            raise RuntimeError(f"QVF student valid mask differs from encoder token padding mask at L{int(layer_idx)}.{QVF_MODULE_NAME}")
        activations[int(layer_idx)] = activation
        valid_masks[int(layer_idx)] = valid_mask.detach().to(dtype=torch.bool)
    return activations, valid_masks


def forward_pooled_logits_and_padding(model: torch.nn.Module, batch: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
        model,
        batch["waveform"],
        batch.get("padding_mask"),
    )
    features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
    return pooled_logits_from_features(model, features, token_padding_mask), token_padding_mask


def forward_pooled_logits(model: torch.nn.Module, batch: Mapping[str, Any]) -> Tensor:
    logits, _token_padding_mask = forward_pooled_logits_and_padding(model, batch)
    return logits


def forward_teacher_qvf(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    layer_indices: Sequence[int],
) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, Tensor]]:
    encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
        model,
        batch["waveform"],
        batch.get("padding_mask"),
    )
    manager = _TeacherQvfCapture(model, layer_indices=layer_indices)
    manager.set_batch_context(padding_mask=token_padding_mask)
    manager.register_hooks()
    try:
        with _disable_encoder_layerdrop_for_qvf(model):
            features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
        teacher_activations, teacher_valid_masks = manager.pop()
    finally:
        manager.clear_hooks()
    logits = pooled_logits_from_features(model, features, token_padding_mask)
    return logits, token_padding_mask, teacher_activations, teacher_valid_masks


def forward_student_qvf(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    layer_indices: Sequence[int],
) -> tuple[Tensor, Tensor, dict[int, Tensor], dict[int, Tensor]]:
    modules = _qvf_modules(model, layer_indices=layer_indices)
    _set_student_qvf_capture(modules, enabled=True)
    try:
        with _disable_encoder_layerdrop_for_qvf(model):
            logits, token_padding_mask = forward_pooled_logits_and_padding(model, batch)
        student_activations, student_valid_masks = _collect_student_qvf_capture(
            modules,
            token_padding_mask=token_padding_mask,
        )
    finally:
        _set_student_qvf_capture(modules, enabled=False)
    return logits, token_padding_mask, student_activations, student_valid_masks


def qvf_token_fidelity_loss(
    *,
    teacher_activations: Mapping[int, Tensor],
    student_activations: Mapping[int, Tensor],
    teacher_valid_masks: Mapping[int, Tensor],
    student_valid_masks: Mapping[int, Tensor],
    layer_indices: Sequence[int],
    topk_ratio: float,
    eps: float,
    layer_weights: Mapping[int, float] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    first_student = next(iter(student_activations.values()), None)
    if first_student is None:
        raise RuntimeError("QVF loss received no student activations")
    batch_size = int(first_student.shape[0])
    sample_losses: list[dict[int, Tensor]] = [dict() for _ in range(batch_size)]
    layer_means: dict[int, float] = {}
    teacher_topk_token_count = 0
    weight_map = {int(layer): float(weight) for layer, weight in dict(layer_weights or {}).items()}

    for layer_idx in tuple(int(item) for item in layer_indices):
        if layer_idx not in teacher_activations or layer_idx not in student_activations:
            raise RuntimeError(f"QVF loss missing activation for L{layer_idx}.{QVF_MODULE_NAME}")
        teacher = teacher_activations[layer_idx].detach().float()
        student = student_activations[layer_idx].float()
        if tuple(teacher.shape) != tuple(student.shape):
            raise RuntimeError(f"QVF activation shape mismatch at L{layer_idx}.{QVF_MODULE_NAME}: {tuple(teacher.shape)} vs {tuple(student.shape)}")
        teacher_valid = teacher_valid_masks[layer_idx].detach().to(device=teacher.device, dtype=torch.bool)
        student_valid = student_valid_masks[layer_idx].detach().to(device=teacher.device, dtype=torch.bool)
        if tuple(teacher_valid.shape) != tuple(student_valid.shape) or not bool(torch.equal(teacher_valid, student_valid)):
            raise RuntimeError(f"QVF teacher/student valid masks differ at L{layer_idx}.{QVF_MODULE_NAME}")
        padding_mask = (~teacher_valid).to(device=teacher.device, dtype=torch.bool)
        topk_ids = select_fp32_topk_token_ids(
            teacher,
            padding_mask,
            topk=None,
            topk_ratio=float(topk_ratio),
            layout="BTC",
        )
        layer_losses: list[Tensor] = []
        for sample_idx, ids_cpu in enumerate(topk_ids):
            ids = ids_cpu.to(device=student.device, dtype=torch.long).reshape(-1)
            if int(ids.numel()) <= 0:
                continue
            teacher_tokens = teacher[int(sample_idx)].to(device=student.device).index_select(0, ids)
            student_tokens = student[int(sample_idx)].index_select(0, ids)
            denom = teacher_tokens.square().sum(dim=-1).clamp_min(float(eps))
            per_token = (student_tokens - teacher_tokens).square().sum(dim=-1) / denom
            sample_loss = per_token.mean()
            sample_losses[int(sample_idx)][int(layer_idx)] = sample_loss
            layer_losses.append(sample_loss.detach())
            teacher_topk_token_count += int(ids.numel())
        if layer_losses:
            layer_means[layer_idx] = float(torch.stack(layer_losses).mean().detach().cpu().item())
        else:
            layer_means[layer_idx] = float("nan")

    per_sample_losses: list[Tensor] = []
    for losses in sample_losses:
        if not losses:
            continue
        if not weight_map:
            per_sample_losses.append(torch.stack(list(losses.values())).mean())
            continue
        weighted_terms: list[Tensor] = []
        available_weight_sum = 0.0
        for layer_idx, layer_loss in sorted(losses.items()):
            weight = float(weight_map.get(int(layer_idx), 0.0))
            if weight <= 0.0:
                continue
            weighted_terms.append(layer_loss * weight)
            available_weight_sum += weight
        if not weighted_terms or available_weight_sum <= 0.0:
            continue
        per_sample_losses.append(torch.stack(weighted_terms).sum() / float(available_weight_sum))
    valid_sample_count = int(len(per_sample_losses))
    if valid_sample_count <= 0:
        raise RuntimeError("QVF loss found no valid non-padding teacher Top-K tokens in batch")
    loss = torch.stack(per_sample_losses).mean()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("QVF loss is non-finite")
    parts = {
        "qvf_loss": float(loss.detach().cpu().item()),
        "qvf_valid_sample_count": float(valid_sample_count),
        "qvf_teacher_topk_token_count": float(teacher_topk_token_count),
    }
    for layer_idx, value in layer_means.items():
        parts[f"qvf_L{layer_idx}_{QVF_MODULE_NAME}_relative_distortion_mean"] = float(value)
    for layer_idx, weight in sorted(weight_map.items()):
        parts[f"qvf_layer_weight_L{int(layer_idx)}_{QVF_MODULE_NAME}"] = float(weight)
    return loss, parts


def projected_energy_retention_loss(
    *,
    teacher_activations: Mapping[int, Tensor],
    student_activations: Mapping[int, Tensor],
    teacher_valid_masks: Mapping[int, Tensor],
    student_valid_masks: Mapping[int, Tensor],
    layer_indices: Sequence[int],
    topk_ratio: float,
    eps: float,
) -> tuple[Tensor, dict[str, float]]:
    layer_tuple = tuple(int(item) for item in layer_indices)
    if layer_tuple != tuple(PER_LAYERS):
        raise RuntimeError(f"PER first version only allows L9.{PER_MODULE_NAME}, got {layer_tuple!r}")
    first_student = next(iter(student_activations.values()), None)
    if first_student is None:
        raise RuntimeError("PER loss received no student activations")
    batch_size = int(first_student.shape[0])
    zero_loss = first_student.float().sum() * 0.0
    sample_losses: list[Tensor] = []
    layer_sample_losses: dict[int, list[Tensor]] = {int(layer): [] for layer in layer_tuple}
    retention_values: dict[int, list[Tensor]] = {int(layer): [] for layer in layer_tuple}
    valid_token_count = 0

    for layer_idx in layer_tuple:
        if layer_idx not in teacher_activations or layer_idx not in student_activations:
            raise RuntimeError(f"PER loss missing activation for L{layer_idx}.{PER_MODULE_NAME}")
        teacher = teacher_activations[layer_idx].detach().float()
        student = student_activations[layer_idx].float()
        if tuple(teacher.shape) != tuple(student.shape):
            raise RuntimeError(f"PER activation shape mismatch at L{layer_idx}.{PER_MODULE_NAME}: {tuple(teacher.shape)} vs {tuple(student.shape)}")
        teacher_valid = teacher_valid_masks[layer_idx].detach().to(device=teacher.device, dtype=torch.bool)
        student_valid = student_valid_masks[layer_idx].detach().to(device=teacher.device, dtype=torch.bool)
        if tuple(teacher_valid.shape) != tuple(student_valid.shape) or not bool(torch.equal(teacher_valid, student_valid)):
            raise RuntimeError(f"PER teacher/student valid masks differ at L{layer_idx}.{PER_MODULE_NAME}")
        padding_mask = (~teacher_valid).to(device=teacher.device, dtype=torch.bool)

        for sample_idx in range(batch_size):
            if int(teacher_valid[int(sample_idx)].sum().item()) <= 0:
                continue
            ids_cpu = select_fp32_topk_token_ids(
                teacher[int(sample_idx) : int(sample_idx) + 1],
                padding_mask[int(sample_idx) : int(sample_idx) + 1],
                topk=None,
                topk_ratio=float(topk_ratio),
                layout="BTC",
            )[0]
            ids = ids_cpu.to(device=student.device, dtype=torch.long).reshape(-1)
            if int(ids.numel()) <= 0:
                continue
            teacher_tokens = teacher[int(sample_idx)].to(device=student.device).index_select(0, ids)
            student_tokens = student[int(sample_idx)].index_select(0, ids)
            teacher_norm = torch.linalg.vector_norm(teacher_tokens, ord=2, dim=-1)
            teacher_unit = teacher_tokens / (teacher_norm.unsqueeze(-1) + float(eps))
            projected = (student_tokens * teacher_unit).sum(dim=-1)
            retention = projected / (teacher_norm + float(eps))
            per_token_loss = (retention - 1.0).square()
            sample_loss = per_token_loss.mean()
            sample_losses.append(sample_loss)
            layer_sample_losses[int(layer_idx)].append(sample_loss.detach())
            retention_values[int(layer_idx)].append(retention.detach())
            valid_token_count += int(ids.numel())

    valid_sample_count = int(len(sample_losses))
    if valid_sample_count <= 0:
        parts = {
            "per_loss": 0.0,
            "per_valid_sample_count": 0.0,
            "per_valid_token_count": 0.0,
            f"per_l{PER_LAYERS[0]}_mean_projected_retention": float("nan"),
            f"per_l{PER_LAYERS[0]}_mean_abs_retention_error": float("nan"),
            f"per_l{PER_LAYERS[0]}_mean_squared_retention_error": float("nan"),
        }
        return zero_loss, parts

    loss = torch.stack(sample_losses).mean()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("PER loss is non-finite")
    parts = {
        "per_loss": float(loss.detach().cpu().item()),
        "per_valid_sample_count": float(valid_sample_count),
        "per_valid_token_count": float(valid_token_count),
    }
    for layer_idx, values in retention_values.items():
        if values:
            layer_retention = torch.cat([item.reshape(-1) for item in values]).float()
            abs_error = (layer_retention - 1.0).abs()
            squared_error = (layer_retention - 1.0).square()
            parts[f"per_l{layer_idx}_mean_projected_retention"] = float(layer_retention.mean().cpu().item())
            parts[f"per_l{layer_idx}_mean_abs_retention_error"] = float(abs_error.mean().cpu().item())
            parts[f"per_l{layer_idx}_mean_squared_retention_error"] = float(squared_error.mean().cpu().item())
        else:
            parts[f"per_l{layer_idx}_mean_projected_retention"] = float("nan")
            parts[f"per_l{layer_idx}_mean_abs_retention_error"] = float("nan")
            parts[f"per_l{layer_idx}_mean_squared_retention_error"] = float("nan")
    for layer_idx, losses in layer_sample_losses.items():
        parts[f"per_L{layer_idx}_{PER_MODULE_NAME}_loss_mean"] = (
            float(torch.stack(losses).mean().cpu().item()) if losses else float("nan")
        )
    return loss, parts


def vanilla_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> tuple[Tensor, dict[str, float]]:
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    ce = F.cross_entropy(student_logits, labels)
    temp = float(temperature)
    kl = F.kl_div(
        F.log_softmax(student_logits / temp, dim=-1),
        F.softmax(teacher_logits.detach() / temp, dim=-1),
        reduction="batchmean",
    )
    kd = float(alpha) * (temp ** 2) * kl
    loss = ce + kd
    return loss, {"ce_loss": float(ce.detach().item()), "kl_loss": float(kl.detach().item()), "kd_term": float(kd.detach().item())}

def _make_optimizer(model: torch.nn.Module, *, backbone_lr: float, classifier_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    classifier_params: list[torch.nn.Parameter] = []
    backbone_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = True
        if name.startswith("predictor."):
            classifier_params.append(parameter)
        else:
            backbone_params.append(parameter)
    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(backbone_lr), "name": "backbone"})
    if classifier_params:
        groups.append({"params": classifier_params, "lr": float(classifier_lr), "name": "classifier"})
    if not groups:
        raise RuntimeError("student has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=float(weight_decay))


def _make_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_ratio: float) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(round(float(total_steps) * float(warmup_ratio)))
    warmup_steps = max(1, min(warmup_steps, int(total_steps))) if total_steps > 0 else 1

    def lr_lambda(step: int) -> float:
        current = int(step) + 1
        if current <= warmup_steps:
            return float(current) / float(warmup_steps)
        denom = max(1, int(total_steps) - warmup_steps)
        progress = min(1.0, float(current - warmup_steps) / float(denom))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



def train_one_epoch(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    amp: bool,
    alpha: float,
    temperature: float,
    num_classes: int,
    qvf_config: Mapping[str, Any] | None = None,
    per_config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    if bool(amp):
        raise RuntimeError("Baseline vanilla KD formal training forbids AMP")
    student.train()
    teacher.eval()
    qvf_enabled = bool(qvf_config and qvf_config.get("enabled"))
    qvf_beta = float((qvf_config or {}).get("beta", 0.0))
    qvf_layers = tuple(int(item) for item in (qvf_config or {}).get("layers", QVF_SENSITIVE_LAYERS))
    qvf_topk_ratio = float((qvf_config or {}).get("topk_ratio", QVF_TOPK_RATIO))
    qvf_layer_weights = (qvf_config or {}).get("layer_weights")
    qvf_layer_weights = {int(layer): float(weight) for layer, weight in dict(qvf_layer_weights or {}).items()} or None
    per_requested = bool(per_config and per_config.get("enabled"))
    per_lambda = float((per_config or {}).get("lambda", 0.0))
    per_enabled = bool(per_requested and per_lambda > 0.0)
    if qvf_enabled and per_requested:
        raise RuntimeError("QVF and PER cannot be enabled in the same Stage1 training epoch")
    per_layers = tuple(int(item) for item in (per_config or {}).get("layers", PER_LAYERS))
    per_topk_ratio = float((per_config or {}).get("topk_ratio", PER_TOPK_RATIO))
    per_epsilon = float((per_config or {}).get("epsilon", PER_DEFAULT_EPS))
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_kd = 0.0
    total_qvf = 0.0
    total_qvf_valid_samples = 0.0
    total_qvf_teacher_topk_tokens = 0.0
    qvf_layer_totals: dict[str, float] = {}
    total_per = 0.0
    total_per_valid_samples = 0.0
    total_per_valid_tokens = 0.0
    total_per_retention_weight = 0.0
    total_per_projected_retention = 0.0
    total_per_abs_retention_error = 0.0
    total_per_squared_retention_error = 0.0
    per_layer_totals: dict[str, float] = {}
    per_grad_sanity_checked = False
    total_correct = 0
    total_samples = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="kd.train.label_id")
        optimizer.zero_grad(set_to_none=True)
        if qvf_enabled or per_enabled:
            capture_layers = qvf_layers if qvf_enabled else per_layers
            student_logits, student_padding_mask, student_capture, student_capture_valid = forward_student_qvf(
                student,
                batch,
                layer_indices=capture_layers,
            )
            with torch.no_grad():
                teacher_logits, teacher_padding_mask, teacher_capture, teacher_capture_valid = forward_teacher_qvf(
                    teacher,
                    batch,
                    layer_indices=capture_layers,
                )
            if tuple(student_padding_mask.shape) != tuple(teacher_padding_mask.shape) or not bool(torch.equal(student_padding_mask, teacher_padding_mask)):
                raise RuntimeError("student and teacher token padding masks differ during QVF/PER capture")
        else:
            student_logits = forward_pooled_logits(student, batch)
            with torch.no_grad():
                teacher_logits = forward_pooled_logits(teacher, batch)
        loss, parts = vanilla_kd_loss(student_logits, teacher_logits, labels, alpha=alpha, temperature=temperature)
        qvf_parts: dict[str, float] = {}
        if qvf_enabled:
            qvf_loss, qvf_parts = qvf_token_fidelity_loss(
                teacher_activations=teacher_capture,
                student_activations=student_capture,
                teacher_valid_masks=teacher_capture_valid,
                student_valid_masks=student_capture_valid,
                layer_indices=qvf_layers,
                topk_ratio=qvf_topk_ratio,
                eps=float(DEFAULT_EPS),
                layer_weights=qvf_layer_weights,
            )
            loss = loss + (float(qvf_beta) * qvf_loss)
        per_parts: dict[str, float] = {}
        if per_enabled:
            per_loss, per_parts = projected_energy_retention_loss(
                teacher_activations=teacher_capture,
                student_activations=student_capture,
                teacher_valid_masks=teacher_capture_valid,
                student_valid_masks=student_capture_valid,
                layer_indices=per_layers,
                topk_ratio=per_topk_ratio,
                eps=per_epsilon,
            )
            if int(per_parts.get("per_valid_sample_count", 0.0)) > 0 and not per_grad_sanity_checked:
                if not bool(per_loss.requires_grad):
                    raise RuntimeError("PER sanity failed: L_PER.requires_grad is False on the first valid PER batch")
                if any(bool(value.requires_grad) for value in teacher_capture.values()):
                    raise RuntimeError("PER sanity failed: teacher activations unexpectedly require grad")
                if not any(bool(value.requires_grad) for value in student_capture.values()):
                    raise RuntimeError("PER sanity failed: student post-fake-quant activations do not require grad")
                per_grad_sanity_checked = True
            loss = loss + (per_lambda * per_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        predictions = student_logits.detach().argmax(dim=1)
        batch_size = int(labels.numel())
        total_samples += batch_size
        total_correct += int((predictions == labels).sum().item())
        total_loss += float(loss.detach().item()) * batch_size
        total_ce += parts["ce_loss"] * batch_size
        total_kl += parts["kl_loss"] * batch_size
        total_kd += parts["kd_term"] * batch_size
        if qvf_enabled:
            total_qvf += float(qvf_parts["qvf_loss"]) * batch_size
            total_qvf_valid_samples += float(qvf_parts["qvf_valid_sample_count"])
            total_qvf_teacher_topk_tokens += float(qvf_parts["qvf_teacher_topk_token_count"])
            for key, value in qvf_parts.items():
                if key.startswith("qvf_L") and math.isfinite(float(value)):
                    qvf_layer_totals[key] = qvf_layer_totals.get(key, 0.0) + (float(value) * batch_size)
        if per_enabled:
            per_valid_samples = float(per_parts["per_valid_sample_count"])
            per_valid_tokens = float(per_parts["per_valid_token_count"])
            total_per += float(per_parts["per_loss"]) * per_valid_samples
            total_per_valid_samples += per_valid_samples
            total_per_valid_tokens += per_valid_tokens
            if per_valid_tokens > 0.0:
                retention_mean = float(per_parts.get(f"per_l{PER_LAYERS[0]}_mean_projected_retention", float("nan")))
                abs_error_mean = float(per_parts.get(f"per_l{PER_LAYERS[0]}_mean_abs_retention_error", float("nan")))
                squared_error_mean = float(per_parts.get(f"per_l{PER_LAYERS[0]}_mean_squared_retention_error", float("nan")))
                if math.isfinite(retention_mean):
                    total_per_projected_retention += retention_mean * per_valid_tokens
                if math.isfinite(abs_error_mean):
                    total_per_abs_retention_error += abs_error_mean * per_valid_tokens
                if math.isfinite(squared_error_mean):
                    total_per_squared_retention_error += squared_error_mean * per_valid_tokens
                total_per_retention_weight += per_valid_tokens
            for key, value in per_parts.items():
                if key.startswith("per_L") and math.isfinite(float(value)):
                    per_layer_totals[key] = per_layer_totals.get(key, 0.0) + (float(value) * max(1.0, per_valid_samples))
    if total_samples <= 0:
        raise RuntimeError("KD train loader produced no samples")
    result = {
        "train_loss": total_loss / float(total_samples),
        "train_ce_loss": total_ce / float(total_samples),
        "train_kl_loss": total_kl / float(total_samples),
        "train_kd_term": total_kd / float(total_samples),
        "train_accuracy": total_correct / float(total_samples),
        "train_num_samples": float(total_samples),
    }
    if qvf_enabled:
        result.update(
            {
                "train_total_loss": total_loss / float(total_samples),
                "train_kd_loss": total_kd / float(total_samples),
                "train_qvf_loss": total_qvf / float(total_samples),
                "qvf_beta": float(qvf_beta),
                "qvf_topk_ratio": float(qvf_topk_ratio),
                "qvf_valid_sample_count": float(total_qvf_valid_samples),
                "qvf_teacher_topk_token_count": float(total_qvf_teacher_topk_tokens),
            }
        )
        if qvf_layer_weights:
            for layer_idx, weight in sorted(qvf_layer_weights.items()):
                result[f"qvf_layer_weight_L{int(layer_idx)}_{QVF_MODULE_NAME}"] = float(weight)
        for key, value in sorted(qvf_layer_totals.items()):
            result[key] = float(value) / float(total_samples)
    if per_enabled:
        retention_denom = float(total_per_retention_weight)
        valid_sample_denom = max(1.0, float(total_per_valid_samples))
        result.update(
            {
                "train_total_loss": total_loss / float(total_samples),
                "train_kd_loss": total_kd / float(total_samples),
                "train_per_loss": total_per / valid_sample_denom,
                "per_lambda": float(per_lambda),
                "per_topk_ratio": float(per_topk_ratio),
                "per_epsilon": float(per_epsilon),
                "per_valid_sample_count": float(total_per_valid_samples),
                "per_valid_token_count": float(total_per_valid_tokens),
                f"per_l{PER_LAYERS[0]}_mean_projected_retention": (
                    total_per_projected_retention / retention_denom if retention_denom > 0.0 else float("nan")
                ),
                f"per_l{PER_LAYERS[0]}_mean_abs_retention_error": (
                    total_per_abs_retention_error / retention_denom if retention_denom > 0.0 else float("nan")
                ),
                f"per_l{PER_LAYERS[0]}_mean_squared_retention_error": (
                    total_per_squared_retention_error / retention_denom if retention_denom > 0.0 else float("nan")
                ),
                "per_grad_sanity_checked": bool(per_grad_sanity_checked),
            }
        )
        for key, value in sorted(per_layer_totals.items()):
            result[key] = float(value) / valid_sample_denom
    return result


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan"), "count": 0}
    count = len(finite)
    mid = count // 2
    median = finite[mid] if count % 2 == 1 else (finite[mid - 1] + finite[mid]) / 2.0
    return {
        "mean": float(sum(finite) / float(count)),
        "median": float(median),
        "min": float(finite[0]),
        "max": float(finite[-1]),
        "count": int(count),
    }


def per_loss_scale_probe(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Any,
    device: torch.device,
    alpha: float,
    temperature: float,
    num_classes: int,
    max_batches: int,
    topk_ratio: float,
    eps: float,
) -> dict[str, Any]:
    if int(max_batches) <= 0:
        raise ValueError("max_batches must be positive")
    student.train()
    teacher.eval()
    ce_values: list[float] = []
    kd_values: list[float] = []
    per_values: list[float] = []
    qvf_l9_values: list[float] = []
    valid_batch_count = 0
    valid_sample_count = 0.0
    valid_token_count = 0.0
    retention_values: list[float] = []
    abs_error_values: list[float] = []
    squared_error_values: list[float] = []
    batch_count = 0
    per_grad_sanity_checked = False

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > int(max_batches):
            break
        batch_count += 1
        batch = move_batch_to_device(batch, device)
        labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="per.probe.label_id")
        student.zero_grad(set_to_none=True)
        student_logits, student_padding_mask, student_capture, student_capture_valid = forward_student_qvf(
            student,
            batch,
            layer_indices=PER_LAYERS,
        )
        with torch.no_grad():
            teacher_logits, teacher_padding_mask, teacher_capture, teacher_capture_valid = forward_teacher_qvf(
                teacher,
                batch,
                layer_indices=PER_LAYERS,
            )
        if tuple(student_padding_mask.shape) != tuple(teacher_padding_mask.shape) or not bool(torch.equal(student_padding_mask, teacher_padding_mask)):
            raise RuntimeError("PER probe student and teacher token padding masks differ")
        _loss, parts = vanilla_kd_loss(student_logits, teacher_logits, labels, alpha=alpha, temperature=temperature)
        per_loss, per_parts = projected_energy_retention_loss(
            teacher_activations=teacher_capture,
            student_activations=student_capture,
            teacher_valid_masks=teacher_capture_valid,
            student_valid_masks=student_capture_valid,
            layer_indices=PER_LAYERS,
            topk_ratio=float(topk_ratio),
            eps=float(eps),
        )
        ce_values.append(float(parts["ce_loss"]))
        kd_values.append(float(parts["kd_term"]))
        per_values.append(float(per_parts["per_loss"]))
        if int(per_parts.get("per_valid_sample_count", 0.0)) > 0:
            qvf_l9_loss, qvf_l9_parts = qvf_token_fidelity_loss(
                teacher_activations=teacher_capture,
                student_activations=student_capture,
                teacher_valid_masks=teacher_capture_valid,
                student_valid_masks=student_capture_valid,
                layer_indices=PER_LAYERS,
                topk_ratio=float(topk_ratio),
                eps=float(DEFAULT_EPS),
                layer_weights=None,
            )
            qvf_l9_values.append(float(qvf_l9_parts["qvf_loss"]))
            valid_batch_count += 1
            valid_sample_count += float(per_parts["per_valid_sample_count"])
            valid_token_count += float(per_parts["per_valid_token_count"])
            retention_values.append(float(per_parts[f"per_l{PER_LAYERS[0]}_mean_projected_retention"]))
            abs_error_values.append(float(per_parts[f"per_l{PER_LAYERS[0]}_mean_abs_retention_error"]))
            squared_error_values.append(float(per_parts[f"per_l{PER_LAYERS[0]}_mean_squared_retention_error"]))
            if not per_grad_sanity_checked:
                if not bool(per_loss.requires_grad):
                    raise RuntimeError("PER probe sanity failed: L_PER.requires_grad is False")
                if any(bool(value.requires_grad) for value in teacher_capture.values()):
                    raise RuntimeError("PER probe sanity failed: teacher activations unexpectedly require grad")
                if not any(bool(value.requires_grad) for value in student_capture.values()):
                    raise RuntimeError("PER probe sanity failed: student post-fake-quant activations do not require grad")
                per_grad_sanity_checked = True
            del qvf_l9_loss
        del _loss, per_loss

    return {
        "protocol_version": PER_LOSS_SCALE_PROBE_PROTOCOL_VERSION,
        "split": "train",
        "train_forward_only": True,
        "optimizer_step": False,
        "backward_run": False,
        "batch_count": int(batch_count),
        "max_batches": int(max_batches),
        "valid_batch_count": int(valid_batch_count),
        "valid_sample_count": float(valid_sample_count),
        "valid_token_count": float(valid_token_count),
        "ce_loss": _numeric_summary(ce_values),
        "alpha_kd_loss": _numeric_summary(kd_values),
        "raw_per_loss": _numeric_summary(per_values),
        "raw_qvf_l9_loss_reference_only": _numeric_summary(qvf_l9_values),
        "projected_retention_mean": _numeric_summary(retention_values),
        "projected_retention_abs_error": _numeric_summary(abs_error_values),
        "projected_retention_squared_error": _numeric_summary(squared_error_values),
        "per_grad_sanity_checked": bool(per_grad_sanity_checked),
        "student_activation_semantic": "post_fake_quant_with_autograd_graph",
        "teacher_activation_semantic": "FP32 teacher fc1 input detached/no_grad",
        "topk_source": "fp32_token_energy",
        "topk_ratio": float(topk_ratio),
        "target_layers": [f"L{layer}.{PER_MODULE_NAME}" for layer in PER_LAYERS],
        "test_isolation": {
            "test_split_touched": False,
            "test_csv_read": False,
            "full_csv_read": False,
            "test_loader_built": False,
            "test_forward_run": False,
            "test_metrics_exported": False,
            "test_predictions_exported": False,
            "label_bucket_source": "not_used_train_split_probe",
        },
    }

def evaluate_student(*, model: torch.nn.Module, loader: Any, device: torch.device, amp: bool, num_classes: int) -> dict[str, Any]:
    if bool(amp):
        raise RuntimeError("Baseline vanilla KD formal eval forbids AMP")
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    labels_all: list[int] = []
    preds_all: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="kd.eval.label_id")
            logits = forward_pooled_logits(model, batch).float()
            loss = F.cross_entropy(logits, labels)
            predictions = logits.detach().argmax(dim=1)
            batch_size = int(labels.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size
            labels_all.extend(int(item) for item in labels.detach().cpu().tolist())
            preds_all.extend(int(item) for item in predictions.detach().cpu().tolist())
    if total_samples <= 0:
        raise RuntimeError("eval loader produced no samples")
    return {
        "loss": total_loss / float(total_samples),
        "accuracy": total_correct / float(total_samples),
        "num_samples": float(total_samples),
        "labels": labels_all,
        "predictions": preds_all,
    }


def _nanmean_float_values(values: Sequence[float]) -> tuple[float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), 0
    return float(sum(finite) / float(len(finite))), int(len(finite))


def _per_groups_for_bucket(bucket: str) -> tuple[str, ...]:
    bucket = str(bucket)
    groups = ["global"]
    if bucket in {"many", "medium", "few"}:
        groups.append(bucket)
    if bucket in {"medium", "few"}:
        groups.append("tail")
    return tuple(groups)


def _per_eval_sample_records(
    *,
    teacher_activations: Mapping[int, Tensor],
    student_activations: Mapping[int, Tensor],
    teacher_valid_masks: Mapping[int, Tensor],
    student_valid_masks: Mapping[int, Tensor],
    labels: Tensor,
    label_buckets: Mapping[int, str],
    topk_ratio: float,
    eps: float,
) -> list[dict[str, Any]]:
    labels_cpu = labels.detach().cpu().long().reshape(-1)
    records: list[dict[str, Any]] = []
    for layer_idx in tuple(int(item) for item in PER_LAYERS):
        if layer_idx not in teacher_activations or layer_idx not in student_activations:
            raise RuntimeError(f"PER eval diagnostic missing activation for L{layer_idx}.{PER_MODULE_NAME}")
        teacher = teacher_activations[layer_idx].detach().float()
        student = student_activations[layer_idx].detach().float()
        if tuple(teacher.shape) != tuple(student.shape):
            raise RuntimeError(
                f"PER eval diagnostic activation shape mismatch at L{layer_idx}.{PER_MODULE_NAME}: "
                f"{tuple(teacher.shape)} vs {tuple(student.shape)}"
            )
        teacher_valid = teacher_valid_masks[layer_idx].detach().to(device=teacher.device, dtype=torch.bool)
        student_valid = student_valid_masks[layer_idx].detach().to(device=teacher.device, dtype=torch.bool)
        if tuple(teacher_valid.shape) != tuple(student_valid.shape) or not bool(torch.equal(teacher_valid, student_valid)):
            raise RuntimeError(f"PER eval diagnostic valid masks differ at L{layer_idx}.{PER_MODULE_NAME}")
        padding_mask = (~teacher_valid).to(device=teacher.device, dtype=torch.bool)
        batch_size = int(teacher.shape[0])
        for sample_idx in range(batch_size):
            label_id = int(labels_cpu[int(sample_idx)].item())
            bucket = str(label_buckets.get(label_id, ""))
            if bucket not in {"many", "medium", "few"}:
                raise RuntimeError(f"PER eval diagnostic missing train-derived bucket for label_id={label_id}")
            record: dict[str, Any] = {
                "layer": f"L{layer_idx}.{PER_MODULE_NAME}",
                "label_id": int(label_id),
                "bucket": bucket,
                "valid_token_count": 0,
                "projected_retention_mean": float("nan"),
                "projected_retention_abs_error": float("nan"),
                "projected_retention_squared_error": float("nan"),
            }
            if int(teacher_valid[int(sample_idx)].sum().item()) <= 0:
                records.append(record)
                continue
            ids_cpu = select_fp32_topk_token_ids(
                teacher[int(sample_idx) : int(sample_idx) + 1],
                padding_mask[int(sample_idx) : int(sample_idx) + 1],
                topk=None,
                topk_ratio=float(topk_ratio),
                layout="BTC",
            )[0]
            ids = ids_cpu.to(device=student.device, dtype=torch.long).reshape(-1)
            if int(ids.numel()) <= 0:
                records.append(record)
                continue
            teacher_tokens = teacher[int(sample_idx)].to(device=student.device).index_select(0, ids)
            student_tokens = student[int(sample_idx)].to(device=student.device).index_select(0, ids)
            teacher_norm = torch.linalg.vector_norm(teacher_tokens, ord=2, dim=-1)
            teacher_unit = teacher_tokens / (teacher_norm.unsqueeze(-1) + float(eps))
            projected = (student_tokens * teacher_unit).sum(dim=-1)
            retention = (projected / (teacher_norm + float(eps))).detach().cpu().float().reshape(-1)
            finite = torch.isfinite(retention)
            if bool(finite.any().item()):
                valid_values = retention[finite]
                record.update(
                    {
                        "valid_token_count": int(valid_values.numel()),
                        "projected_retention_mean": float(valid_values.mean().item()),
                        "projected_retention_abs_error": float((valid_values - 1.0).abs().mean().item()),
                        "projected_retention_squared_error": float((valid_values - 1.0).square().mean().item()),
                    }
                )
            records.append(record)
    return records


def _per_eval_diagnostic_rows(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Any,
    device: torch.device,
    num_classes: int,
    label_buckets: Mapping[int, str],
    topk_ratio: float,
    eps: float,
) -> list[dict[str, Any]]:
    student.eval()
    teacher.eval()
    group_values: dict[str, dict[str, list[float]]] = {
        group: {
            "projected_retention_mean": [],
            "projected_retention_abs_error": [],
            "projected_retention_squared_error": [],
        }
        for group in ("global", "many", "medium", "few", "tail")
    }
    group_total = {group: 0 for group in group_values}
    layer_name = f"L{PER_LAYERS[0]}.{PER_MODULE_NAME}"
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="per.eval.label_id")
            _student_logits, student_padding_mask, student_capture, student_capture_valid = forward_student_qvf(
                student,
                batch,
                layer_indices=PER_LAYERS,
            )
            _teacher_logits, teacher_padding_mask, teacher_capture, teacher_capture_valid = forward_teacher_qvf(
                teacher,
                batch,
                layer_indices=PER_LAYERS,
            )
            if tuple(student_padding_mask.shape) != tuple(teacher_padding_mask.shape) or not bool(torch.equal(student_padding_mask, teacher_padding_mask)):
                raise RuntimeError("student and teacher token padding masks differ during PER eval diagnostics")
            for record in _per_eval_sample_records(
                teacher_activations=teacher_capture,
                student_activations=student_capture,
                teacher_valid_masks=teacher_capture_valid,
                student_valid_masks=student_capture_valid,
                labels=labels,
                label_buckets=label_buckets,
                topk_ratio=float(topk_ratio),
                eps=float(eps),
            ):
                for group in _per_groups_for_bucket(str(record["bucket"])):
                    group_total[group] += 1
                    for metric_name in group_values[group]:
                        group_values[group][metric_name].append(float(record[metric_name]))
    rows: list[dict[str, Any]] = []
    for group in ("global", "many", "medium", "few", "tail"):
        for metric_name, values in group_values[group].items():
            value, n_valid = _nanmean_float_values(values)
            rows.append(
                {
                    "split": "eval",
                    "layer": layer_name,
                    "metric_name": metric_name,
                    "analysis_group": group,
                    "metric_value": float(value),
                    "n_total": int(group_total[group]),
                    "n_valid": int(n_valid),
                    "metric_direction": "higher_is_better" if metric_name == "projected_retention_mean" else "lower_is_better",
                    "aggregation_unit": "sample",
                    "aggregation_fn": "nanmean_valid_only",
                    "metric_role": "diagnostic",
                }
            )
    return rows


def _read_csv_dict_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metric_value_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
    analysis_group: str,
    split: str = "eval",
    required: bool = True,
) -> float:
    matches = [
        row
        for row in rows
        if str(row.get("split", split)) == str(split)
        and str(row.get("metric_name")) == str(metric_name)
        and str(row.get("analysis_group")) == str(analysis_group)
    ]
    if not matches:
        if required:
            raise KeyError(f"missing metric: split={split}, metric_name={metric_name}, analysis_group={analysis_group}")
        return float("nan")
    value = matches[-1].get("metric_value")
    try:
        return float(value)
    except (TypeError, ValueError):
        if required:
            raise
        return float("nan")


def _per_mechanism_gap_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    specs = (
        ("oer_gap_many_minus_tail", "oer_retention", "many", "tail", "many_minus_tail"),
        ("distortion_gap_tail_minus_many", "teacher_topk_relative_distortion", "tail", "many", "tail_minus_many"),
        ("topk_overlap_gap_many_minus_tail", "topk_overlap", "many", "tail", "many_minus_tail"),
    )
    rows: list[dict[str, Any]] = []
    for field_name, source_metric, left_group, right_group, gap_direction in specs:
        left = _metric_value_from_rows(metric_rows, metric_name=source_metric, analysis_group=left_group, required=False)
        right = _metric_value_from_rows(metric_rows, metric_name=source_metric, analysis_group=right_group, required=False)
        available = math.isfinite(left) and math.isfinite(right)
        rows.append(
            {
                "split": "eval",
                "field_name": field_name,
                "source_metric_name": source_metric,
                "left_group": left_group,
                "right_group": right_group,
                "gap_direction": gap_direction,
                "left_value": float(left),
                "right_value": float(right),
                "gap_value": float(left - right) if available else float("nan"),
                "available": bool(available),
                "reason": "" if available else "source_metric_not_available_from_current_export_metrics",
            }
        )
    return rows


def _rows_by_name_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        metric_name = str(row.get("metric_name", ""))
        group = str(row.get("analysis_group", ""))
        if not metric_name or not group:
            continue
        try:
            result[f"{metric_name}:{group}"] = float(row.get("metric_value"))
        except (TypeError, ValueError):
            result[f"{metric_name}:{group}"] = float("nan")
    return result


def _gap_rows_by_field(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        field_name = str(row.get("field_name", ""))
        if not field_name:
            continue
        try:
            result[field_name] = float(row.get("gap_value"))
        except (TypeError, ValueError):
            result[field_name] = float("nan")
    return result



def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    eval_metrics: Mapping[str, Any],
    teacher_checkpoint: Path,
    teacher_payload: Mapping[str, Any],
    qparams_csv: Path,
    controller: Any,
    layers: Sequence[Mapping[str, Any]],
    paths: Any,
    label_mapping: Mapping[str, int],
    args: argparse.Namespace,
    calibration_csv: str | Path | None,
    calibration_hash: str | None,
    int4_provenance: Mapping[str, Any],
    quant_audit: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    training_resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _assert_fp32_teacher_payload(teacher_payload, context="KD checkpoint save teacher payload")
    cfg = dict(teacher_payload.get("cfg") or {})
    if not cfg:
        raise KeyError("FP32 teacher payload missing cfg; cannot create reproducible vanilla KD checkpoint")
    qparams = qck.load_qparams_csv(qparams_csv)
    qparams_hash_value = qck.qparams_hash(qparams)
    expected_qparams_hash = str(int4_provenance.get("qparams_hash") or "")
    if qparams_hash_value != expected_qparams_hash:
        raise RuntimeError(
            "KD qparams semantic hash differs from INT4 provenance before checkpoint save: "
            f"current={qparams_hash_value}, int4={expected_qparams_hash}"
        )
    qparams_csv_sha256 = file_sha256(qparams_csv)
    if str(qparams_csv_sha256) != str(int4_provenance.get("qparams_csv_sha256")):
        raise RuntimeError(
            "KD qparams CSV artifact hash differs from INT4 provenance before checkpoint save: "
            f"current={qparams_csv_sha256}, int4={int4_provenance.get('qparams_csv_sha256')}"
        )
    controller_qparams_hash = qck.qparams_hash(qck.collect_qparams_from_controller(controller))
    if controller_qparams_hash != qparams_hash_value:
        raise RuntimeError(
            "controller qparams and qparams_csv disagree before saving KD checkpoint: "
            f"controller={controller_qparams_hash}, csv={qparams_hash_value}"
        )
    layers_payload = [dict(row) for row in layers]
    quantized_layers_hash = canonical_json_hash(layers_payload)
    if str(quantized_layers_hash) != str(int4_provenance.get("quantized_layers_hash")):
        raise RuntimeError(
            "KD quantized_layers hash differs from INT4 provenance before checkpoint save: "
            f"current={quantized_layers_hash}, int4={int4_provenance.get('quantized_layers_hash')}"
        )
    qparams_metadata = qck.qparams_csv_metadata(qparams_csv)
    metadata_calibration_hash = qparams_metadata.get("calibration_hash")
    calibration_csv_hash = file_sha256(calibration_csv)
    if calibration_csv_hash and str(calibration_csv_hash) != str(int4_provenance.get("calibration_csv_hash")):
        raise RuntimeError(
            "KD calibration CSV hash differs from INT4 provenance before checkpoint save: "
            f"current={calibration_csv_hash}, int4={int4_provenance.get('calibration_csv_hash')}"
        )
    if calibration_csv_hash and metadata_calibration_hash and str(metadata_calibration_hash) != str(calibration_csv_hash):
        raise RuntimeError(
            "qparams_csv calibration hash does not match KD checkpoint calibration CSV identity: "
            f"csv={metadata_calibration_hash}, calibration_csv={calibration_csv_hash}"
        )
    calibration_hash = str(calibration_hash or metadata_calibration_hash or calibration_csv_hash or "") or None
    if calibration_hash and str(calibration_hash) != str(int4_provenance.get("calibration_hash")):
        raise RuntimeError(
            "KD calibration_hash differs from INT4 provenance before checkpoint save: "
            f"current={calibration_hash}, int4={int4_provenance.get('calibration_hash')}"
        )
    qvf_mode = _qvf_enabled(args)
    per_mode = _per_enabled(args)
    method_variant = _qvf_method_variant_from_args(args) if qvf_mode else (_per_method_variant_from_args(args) if per_mode else BASELINE_NAME)
    qvf_layer_weights = _qvf_layer_weights_from_args(args) if qvf_mode else None
    qvf_protocol_payload = _qvf_protocol_for_args(args) if qvf_mode else {
        "use_qvf_token_fidelity": False,
        "qvf_beta": 0.0,
        "activation_capture_registered": False,
        "teacher_token_feature_computed": False,
    }
    per_protocol_payload = _per_protocol_for_args(args) if per_mode else {
        "use_projected_energy_retention": False,
        "per_lambda": 0.0,
        "activation_capture_registered": False,
        "teacher_token_feature_computed": False,
    }
    qvf_loss_name = "weighted QVF" if qvf_mode and qvf_layer_weights else "QVF"
    loss_name = (
        f"CE + alpha * T^2 * KL + beta * {qvf_loss_name}"
        if qvf_mode
        else ("CE + alpha * T^2 * KL + lambda_per * L_PER" if per_mode else "CE + alpha * T^2 * KL")
    )
    kd_config = {
        "loss": loss_name,
        "alpha": float(args.alpha),
        "temperature": float(args.temperature),
        "use_qvf_token_fidelity": bool(qvf_mode),
        "qvf_beta": float(args.qvf_beta) if qvf_mode else 0.0,
        "qvf_protocol": qvf_protocol_payload,
        "use_projected_energy_retention": bool(per_mode),
        "per_lambda": float(args.per_lambda) if per_mode else 0.0,
        "per_protocol": per_protocol_payload,
        "optimizer": "AdamW",
        "backbone_lr": float(args.backbone_lr),
        "classifier_lr": float(args.classifier_lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": "warmup + cosine annealing",
        "warmup_ratio": float(args.warmup_ratio),
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "class_weight": None,
        "sampler": "normal shuffle train loader",
        "teacher_frozen": True,
        "qparams_frozen": True,
        "student_backbone_trainable": True,
        "amp": False,
        "training_dtype": "float32",
        "loss_logits_dtype": "float32",
    }
    first_layer = dict(layers_payload[0]) if layers_payload else {}
    source_beats_checkpoint = _teacher_source_beats_checkpoint(teacher_payload, context="KD teacher")
    int4_provenance_payload = {
        "manifest_path": str(int4_provenance.get("manifest_path") or ""),
        "manifest_sha256": int4_provenance.get("manifest_sha256"),
        "run_manifest_hash": int4_provenance.get("manifest_hash"),
        "manifest_run_identity_hash": int4_provenance.get("manifest_run_identity_hash"),
        "checkpoint": int4_provenance.get("checkpoint"),
        "checkpoint_sha256": int4_provenance.get("checkpoint_sha256"),
        "calibration_csv": path_str(int4_provenance.get("calibration_csv")),
        "calibration_csv_hash": int4_provenance.get("calibration_csv_hash"),
        "calibration_manifest_hash": int4_provenance.get("calibration_manifest_hash"),
        "calibration_hash": int4_provenance.get("calibration_hash"),
        "qparams_hash": int4_provenance.get("qparams_hash"),
        "qparams_csv_sha256": int4_provenance.get("qparams_csv_sha256"),
        "quantized_layers_hash": int4_provenance.get("quantized_layers_hash"),
        "quantized_layers": layers_payload,
        "artifact_path_migration_allowed": bool(int4_provenance.get("artifact_path_migration_allowed", False)),
        "checkpoint_path_migrated": bool(int4_provenance.get("checkpoint_path_migrated", False)),
        "qparams_csv_path_migrated": bool(int4_provenance.get("qparams_csv_path_migrated", False)),
        "calibration_csv_path_migrated": bool(int4_provenance.get("calibration_csv_path_migrated", False)),
        "calibration_manifest_path_migrated": bool(int4_provenance.get("calibration_manifest_path_migrated", False)),
    }
    identity_payload = _drop_legacy_hash_aliases(qck.checkpoint_identity_payload(
        model_kind="vanilla_kd",
        cfg=cfg,
        label_mapping=label_mapping,
        label_mapping_hash=file_sha256(paths.label_mapping),
        source_beats_checkpoint=source_beats_checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        quantized_layers=layers_payload,
        qparams=qparams,
        qparams_csv=qparams_csv,
        calibration_csv=calibration_csv,
        calibration_hash=calibration_hash,
        pooling=DEFAULT_POOLING,
        seed=int(args.seed),
    ))
    identity_payload["kd_config"] = kd_config
    identity_payload["method_variant"] = method_variant
    identity_payload["qvf_protocol"] = qvf_protocol_payload
    identity_payload["per_protocol"] = per_protocol_payload
    identity_payload["int4_student_checkpoint_sha256"] = int4_provenance_payload["checkpoint_sha256"]
    identity_payload["int4_student_manifest_sha256"] = int4_provenance_payload["manifest_sha256"]
    identity_payload["int4_student_run_identity_hash"] = int4_provenance_payload["manifest_run_identity_hash"]
    identity_payload["int4_student_quantized_layers_hash"] = int4_provenance_payload["quantized_layers_hash"]
    run_identity_hash = canonical_json_hash(identity_payload)
    source_beats_checkpoint_sha256 = identity_payload.get("source_beats_checkpoint_sha256")
    teacher_checkpoint_sha256 = file_sha256(teacher_checkpoint)
    audio_input_py_sha256 = identity_payload.get("audio_input_py_sha256") or identity_payload.get("audio_input_hash")
    group_utils_py_sha256 = identity_payload.get("group_utils_py_sha256") or identity_payload.get("group_utils_hash")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact_type": "baseline_vanilla_kd_quantized_student_checkpoint",
        "model_kind": "vanilla_kd",
        "method_variant": method_variant,
        "use_qvf_token_fidelity": bool(qvf_mode),
        "qvf_beta": float(args.qvf_beta) if qvf_mode else 0.0,
        "qvf_loss_definition": qvf_protocol_payload.get("qvf_loss_definition", ""),
        "qvf_protocol": qvf_protocol_payload,
        "use_projected_energy_retention": bool(per_mode),
        "per_lambda": float(args.per_lambda) if per_mode else 0.0,
        "per_loss_definition": per_protocol_payload.get("loss_definition", ""),
        "per_protocol": per_protocol_payload,
        "epoch": int(epoch),
        "selection_metric": "eval_macro_f1_label_universe_nanmean",
        "eval_metrics": dict(eval_metrics),
        "cfg": cfg,
        "model_state_dict": model.state_dict(),
        "label_mapping": dict(label_mapping),
        "label_mapping_hash": file_sha256(paths.label_mapping),
        "source_beats_checkpoint": path_str(source_beats_checkpoint),
        "source_beats_checkpoint_sha256": source_beats_checkpoint_sha256,
        "teacher_checkpoint": path_str(teacher_checkpoint),
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "int4_calibrated_student_checkpoint": int4_provenance_payload["checkpoint"],
        "int4_calibrated_student_checkpoint_sha256": int4_provenance_payload["checkpoint_sha256"],
        "source_int4_init_hash": int4_provenance_payload["checkpoint_sha256"],
        "int4_student_manifest": int4_provenance_payload["manifest_path"],
        "int4_student_manifest_sha256": int4_provenance_payload["manifest_sha256"],
        "int4_student_run_manifest_hash": int4_provenance_payload["run_manifest_hash"],
        "int4_student_run_identity_hash": int4_provenance_payload["manifest_run_identity_hash"],
        "int4_student_provenance": int4_provenance_payload,
        "quantized_layers": layers_payload,
        "quantized_layers_hash": quantized_layers_hash,
        "bitwidth": 4,
        "bitwidth_detail": {"weight": int(first_layer.get("weight_bit", 4)), "activation": int(first_layer.get("activation_bit", 4))},
        "qmin": int(first_layer.get("qmin", -7)),
        "qmax": int(first_layer.get("qmax", 7)),
        "symmetric": bool(first_layer.get("symmetric", True)),
        "weight_granularity": str(first_layer.get("weight_granularity", "per_channel")),
        "activation_granularity": str(first_layer.get("activation_granularity", "per_tensor")),
        "qparams": qck.qparams_to_jsonable(qparams),
        "qparams_hash": qparams_hash_value,
        "qparams_csv": path_str(qparams_csv),
        "qparams_csv_sha256": qparams_csv_sha256,
        "qparams_csv_semantic_hash": qparams_hash_value,
        "qparams_csv_metadata": qparams_metadata,
        "calibration_csv": path_str(calibration_csv),
        "calibration_csv_hash": calibration_csv_hash,
        "calibration_hash": calibration_hash,
        "audio_input_py_sha256": audio_input_py_sha256,
        "audio_input_hash": audio_input_py_sha256,
        "group_utils_py_sha256": group_utils_py_sha256,
        "group_utils_hash": group_utils_py_sha256,
        "pooling": DEFAULT_POOLING,
        "seed": int(args.seed),
        "git_commit": identity_payload.get("git_commit"),
        "run_identity_hash": run_identity_hash,
        "run_identity_payload": identity_payload,
        "kd_config": kd_config,
        "teacher_frozen": True,
        "qparams_frozen": True,
        "fake_quant_enabled": True,
        "observer_enabled": False,
        "student_backbone_trainable": True,
        "test_split_touched": False if (qvf_mode or per_mode) else None,
        "controller_state_summary": controller.state_summary(),
        "training_quant_audit": dict(quant_audit or {}),
        "load_note": "Load through src.baseline.export_eval.load_trained_checkpoint; it injects INT4, applies qparams, freezes qparams, disables observer, enables fake quant, strict-loads state_dict, and audits quant state.",
    }
    if qvf_mode and qvf_layer_weights:
        payload["qvf_weighted_variant"] = qvf_protocol_payload.get("qvf_weighted_variant")
        payload["qvf_layer_weight_policy"] = qvf_protocol_payload.get("layer_weight_policy")
        payload["sensitive_layer_weights"] = qvf_protocol_payload.get("sensitive_layer_weights")
        payload["qvf_layer_weights"] = qvf_protocol_payload.get("qvf_layer_weights")
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if training_resume_state is not None:
        payload["training_resume_protocol_version"] = "baseline_kd_training_resume.v1"
        payload["training_resume_state"] = dict(training_resume_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return payload


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding=encoding, newline="") as handle:
        handle.write(text)
    tmp_path.replace(path)


def _write_train_log(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    _atomic_write_text(path, buffer.getvalue())


def _write_csv_rows_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    _atomic_write_text(path, buffer.getvalue())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n"
    _atomic_write_text(path, text)


def _write_run_manifests_atomic(output_dirs: Mapping[str, str | Path], manifest: Mapping[str, Any]) -> dict[str, str]:
    written: dict[str, str] = {}
    for name, directory in output_dirs.items():
        manifest_path = Path(directory) / "run_manifest.json"
        payload = attach_run_manifest_hash(manifest)
        _write_json_atomic(manifest_path, payload)
        written[str(name)] = path_str(manifest_path) or str(manifest_path)
    return written


def _manifest_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in manifest.items() if str(key) not in _RUN_MANIFEST_HASH_EXCLUDE_KEYS}


def _assert_manifest_hash_valid(path: Path, manifest: Mapping[str, Any]) -> None:
    stored = manifest.get("run_manifest_hash")
    if not stored:
        raise RuntimeError(f"completed run manifest missing run_manifest_hash: {path}")
    recomputed = canonical_json_hash(_manifest_hash_payload(manifest))
    if str(stored) != str(recomputed):
        raise RuntimeError(f"completed run manifest hash mismatch: existing={stored!r}, recomputed={recomputed!r}, path={path}")


def _assert_csv_readable(path: Path, *, name: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{name} has no CSV header: {path}")
        for _row in reader:
            break


def _try_load_completed_run_manifest(
    *,
    args: argparse.Namespace,
    resolved: Mapping[str, Path],
    qvf_mode: bool,
    per_mode: bool = False,
) -> dict[str, Any] | None:
    manifest_path = Path(resolved["manifest_path"])
    if not manifest_path.exists():
        return None
    manifest = _load_required_json(manifest_path, name="completed run manifest")
    _assert_manifest_hash_valid(manifest_path, manifest)
    errors: list[str] = []
    expected_stage = "qvf_stage1_vanilla_kd" if qvf_mode else ("per_stage1_vanilla_kd" if per_mode else "baseline_vanilla_kd")
    if str(manifest.get("stage") or "") != expected_stage:
        errors.append(f"stage={manifest.get('stage')!r} expected={expected_stage!r}")
    if int(manifest.get("seed", -1)) != int(args.seed):
        errors.append(f"seed={manifest.get('seed')!r} expected={int(args.seed)!r}")
    if qvf_mode:
        if not bool(manifest.get("use_qvf_token_fidelity")):
            errors.append("use_qvf_token_fidelity is not true")
        try:
            _require_float_close("completed_manifest.qvf_beta", manifest.get("qvf_beta"), float(args.qvf_beta))
            _require_qvf_protocol_matches_args("completed_manifest", args, manifest)
        except RuntimeError as exc:
            errors.append(str(exc))
    if per_mode:
        if not bool(manifest.get("use_projected_energy_retention")):
            errors.append("use_projected_energy_retention is not true")
        try:
            _require_float_close("completed_manifest.per_lambda", manifest.get("per_lambda"), float(args.per_lambda))
            _require_per_protocol_matches_args("completed_manifest", args, manifest)
        except RuntimeError as exc:
            errors.append(str(exc))
    checkpoint_hash = file_sha256(resolved["best_checkpoint"])
    if not checkpoint_hash or str(manifest.get("checkpoint_sha256")) != str(checkpoint_hash):
        errors.append(f"checkpoint_sha256={manifest.get('checkpoint_sha256')!r} expected={checkpoint_hash!r}")
    last_hash = file_sha256(resolved["last_checkpoint"])
    if not last_hash or str(manifest.get("last_checkpoint_sha256")) != str(last_hash):
        errors.append(f"last_checkpoint_sha256={manifest.get('last_checkpoint_sha256')!r} expected={last_hash!r}")
    train_log_hash = manifest.get("train_log_sha256")
    if train_log_hash is not None and str(train_log_hash) != str(file_sha256(resolved["train_log"])):
        errors.append(f"train_log_sha256={train_log_hash!r} expected={file_sha256(resolved['train_log'])!r}")
    if not Path(resolved["metrics_csv"]).exists():
        errors.append(f"metrics_csv missing: {resolved['metrics_csv']}")
    else:
        try:
            _assert_csv_readable(Path(resolved["metrics_csv"]), name="completed metrics_csv")
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(
            "completed training manifest exists but failed completion audit; refusing to overwrite/export again: "
            + "; ".join(errors)
        )
    return manifest


def _load_torch_checkpoint(path: Path, *, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(payload)!r}: {path}")
    return payload


def _require_float_close(name: str, actual: Any, expected: Any, *, abs_tol: float = 1e-12) -> None:
    if actual is None or expected is None:
        if actual != expected:
            raise RuntimeError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")
        return
    actual_float = float(actual)
    expected_float = float(expected)
    if not math.isfinite(actual_float) or not math.isfinite(expected_float):
        if str(actual) != str(expected):
            raise RuntimeError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")
        return
    if not math.isclose(actual_float, expected_float, rel_tol=0.0, abs_tol=float(abs_tol)):
        raise RuntimeError(f"{name} mismatch: actual={actual_float!r}, expected={expected_float!r}")


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_train_rows_for_resume(path: Path, resume_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows = resume_state.get("train_rows")
    if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
        for row in raw_rows:
            if isinstance(row, Mapping):
                rows.append(dict(row))
    if not rows and path.exists() and path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            file_rows = [dict(row) for row in reader]
        if file_rows:
            rows = file_rows
    return rows


def _training_resume_state(
    *,
    epoch: int,
    best_epoch: int,
    best_eval_macro_f1_label_universe_nanmean: float,
    best_eval: Mapping[str, Any],
    best_quant_audit: Mapping[str, Any],
    train_rows: Sequence[Mapping[str, Any]],
    best_checkpoint: Path,
    last_checkpoint: Path,
    rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "best_epoch": int(best_epoch),
        "best_eval_macro_f1_label_universe_nanmean": float(best_eval_macro_f1_label_universe_nanmean),
        "best_eval": dict(best_eval),
        "best_quant_audit": dict(best_quant_audit),
        "train_rows": [dict(row) for row in train_rows],
        "train_log_row_count": int(len(train_rows)),
        "best_checkpoint": path_str(best_checkpoint),
        "best_checkpoint_sha256": file_sha256(best_checkpoint),
        "last_checkpoint": path_str(last_checkpoint),
        "rng_state": dict(rng_state),
        "resume_granularity": "epoch_boundary",
        "bitwise_resume_scope": "matches uninterrupted run from completed epoch boundary when command, device count, data, and dependencies are unchanged",
    }


def _validate_resume_checkpoint(
    *,
    payload: Mapping[str, Any],
    path: Path,
    args: argparse.Namespace,
    paths: Any,
    teacher_checkpoint: Path,
    int4_checkpoint: Path,
    qparams_csv: Path,
    calibration_hash: str | None,
    int4_provenance: Mapping[str, Any],
    total_epochs: int,
) -> Mapping[str, Any]:
    qvf_mode = _qvf_enabled(args)
    per_mode = _per_enabled(args)
    _require_equal("resume.model_kind", payload.get("model_kind"), "vanilla_kd")
    _require_equal("resume.seed", payload.get("seed"), int(args.seed))
    _require_equal("resume.use_qvf_token_fidelity", bool(payload.get("use_qvf_token_fidelity")), bool(qvf_mode))
    _require_float_close("resume.qvf_beta", payload.get("qvf_beta", 0.0), float(args.qvf_beta) if qvf_mode else 0.0)
    _require_equal("resume.use_projected_energy_retention", bool(payload.get("use_projected_energy_retention")), bool(per_mode))
    _require_float_close("resume.per_lambda", payload.get("per_lambda", 0.0), float(args.per_lambda) if per_mode else 0.0)
    _require_equal("resume.teacher_checkpoint_sha256", payload.get("teacher_checkpoint_sha256"), file_sha256(teacher_checkpoint))
    _require_equal("resume.int4_calibrated_student_checkpoint_sha256", payload.get("int4_calibrated_student_checkpoint_sha256"), file_sha256(int4_checkpoint))
    _require_equal("resume.source_int4_init_hash", payload.get("source_int4_init_hash"), int4_provenance.get("checkpoint_sha256"))
    _require_equal("resume.qparams_csv_sha256", payload.get("qparams_csv_sha256"), file_sha256(qparams_csv))
    _require_equal("resume.qparams_hash", payload.get("qparams_hash"), int4_provenance.get("qparams_hash"))
    _require_equal("resume.label_mapping_hash", payload.get("label_mapping_hash"), file_sha256(paths.label_mapping))
    if calibration_hash is not None:
        _require_equal("resume.calibration_hash", payload.get("calibration_hash"), calibration_hash)
    kd_config = payload.get("kd_config")
    if not isinstance(kd_config, Mapping):
        raise RuntimeError(f"resume checkpoint missing kd_config mapping: {path}")
    _require_float_close("resume.kd_config.alpha", kd_config.get("alpha"), float(args.alpha))
    _require_float_close("resume.kd_config.temperature", kd_config.get("temperature"), float(args.temperature))
    _require_float_close("resume.kd_config.backbone_lr", kd_config.get("backbone_lr"), float(args.backbone_lr))
    _require_float_close("resume.kd_config.classifier_lr", kd_config.get("classifier_lr"), float(args.classifier_lr))
    _require_float_close("resume.kd_config.weight_decay", kd_config.get("weight_decay"), float(args.weight_decay))
    _require_float_close("resume.kd_config.warmup_ratio", kd_config.get("warmup_ratio"), float(args.warmup_ratio))
    _require_equal("resume.kd_config.epochs_requested", kd_config.get("epochs_requested"), int(args.epochs))
    _require_equal("resume.kd_config.batch_size", kd_config.get("batch_size"), int(args.batch_size))
    _require_equal("resume.kd_config.eval_batch_size", kd_config.get("eval_batch_size"), int(args.eval_batch_size))
    _require_equal("resume.kd_config.num_workers", kd_config.get("num_workers"), int(args.num_workers))
    qvf_protocol = payload.get("qvf_protocol")
    if qvf_mode:
        if not isinstance(qvf_protocol, Mapping):
            raise RuntimeError(f"resume checkpoint missing qvf_protocol mapping: {path}")
        _require_qvf_protocol_matches_args("resume", args, payload)
        _require_float_close("resume.qvf_protocol.topk_ratio", qvf_protocol.get("topk_ratio"), float(QVF_TOPK_RATIO))
    per_protocol = payload.get("per_protocol")
    if per_mode:
        if not isinstance(per_protocol, Mapping):
            raise RuntimeError(f"resume checkpoint missing per_protocol mapping: {path}")
        _require_per_protocol_matches_args("resume", args, payload)
        _require_float_close("resume.per_protocol.topk_ratio", per_protocol.get("topk_ratio"), float(PER_TOPK_RATIO))
    epoch = int(payload.get("epoch", 0))
    if epoch < 1:
        raise RuntimeError(f"resume checkpoint has invalid epoch={epoch}: {path}")
    if epoch > int(total_epochs):
        raise RuntimeError(f"resume checkpoint epoch={epoch} exceeds requested epochs={total_epochs}: {path}")
    if "model_state_dict" not in payload:
        raise RuntimeError(f"resume checkpoint missing model_state_dict: {path}")
    if "optimizer_state_dict" not in payload:
        raise RuntimeError(f"resume checkpoint missing optimizer_state_dict; use --no-resume to restart: {path}")
    if "scheduler_state_dict" not in payload:
        raise RuntimeError(f"resume checkpoint missing scheduler_state_dict; use --no-resume to restart: {path}")
    resume_state = payload.get("training_resume_state")
    if not isinstance(resume_state, Mapping):
        raise RuntimeError(f"resume checkpoint missing training_resume_state; use --no-resume to restart: {path}")
    if int(resume_state.get("epoch", epoch)) != epoch:
        raise RuntimeError(f"resume training_resume_state epoch mismatch: checkpoint={epoch}, state={resume_state.get('epoch')}")
    if not isinstance(resume_state.get("rng_state"), Mapping):
        raise RuntimeError(f"resume checkpoint missing rng_state; use --no-resume to restart: {path}")
    best_epoch = int(resume_state.get("best_epoch", 0))
    if best_epoch < 0:
        raise RuntimeError(f"resume checkpoint has invalid best_epoch={best_epoch}: {path}")
    best_metric = float(resume_state.get("best_eval_macro_f1_label_universe_nanmean", float("nan")))
    if best_epoch > 0 and not math.isfinite(best_metric):
        raise RuntimeError(f"resume checkpoint missing finite best eval metric: {path}")
    return resume_state


def _try_resume_training(
    *,
    args: argparse.Namespace,
    resolved: Mapping[str, Path],
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    paths: Any,
    teacher_checkpoint: Path,
    int4_checkpoint: Path,
    qparams_csv: Path,
    calibration_hash: str | None,
    int4_provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "resume", True)):
        if Path(resolved["last_checkpoint"]).exists():
            print(f"[baseline-kd:resume] --no-resume set; restarting from INT4 init and overwriting outputs under {resolved['seed_dir']}")
        return None
    last_checkpoint = Path(resolved["last_checkpoint"])
    if not last_checkpoint.exists():
        return None
    payload = _load_torch_checkpoint(last_checkpoint, map_location="cpu")
    resume_state = _validate_resume_checkpoint(
        payload=payload,
        path=last_checkpoint,
        args=args,
        paths=paths,
        teacher_checkpoint=teacher_checkpoint,
        int4_checkpoint=int4_checkpoint,
        qparams_csv=qparams_csv,
        calibration_hash=calibration_hash,
        int4_provenance=int4_provenance,
        total_epochs=int(args.epochs),
    )
    student.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    _optimizer_state_to_device(optimizer, device)
    _restore_rng_state(resume_state["rng_state"], device=device)
    epoch = int(payload["epoch"])
    best_epoch = int(resume_state.get("best_epoch", 0))
    if best_epoch > 0:
        best_checkpoint = Path(resolved["best_checkpoint"])
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"resume best checkpoint missing: {best_checkpoint}")
        expected_best_hash = resume_state.get("best_checkpoint_sha256")
        if expected_best_hash is not None:
            _require_equal("resume.best_checkpoint_sha256", file_sha256(best_checkpoint), expected_best_hash)
    train_rows = _load_train_rows_for_resume(Path(resolved["train_log"]), resume_state)
    print(f"[baseline-kd:resume] resumed {last_checkpoint} at epoch={epoch}; next_epoch={epoch + 1}")
    return {
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "best_epoch": best_epoch,
        "best_eval_macro_f1_label_universe_nanmean": float(resume_state["best_eval_macro_f1_label_universe_nanmean"]),
        "best_eval": dict(resume_state.get("best_eval") or {}),
        "best_quant_audit": dict(resume_state.get("best_quant_audit") or {}),
        "train_rows": train_rows,
        "payload": payload,
    }


def _write_combined_metrics(split_results: Mapping[str, Any], path: Path, *, seed: int) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for split, result in split_results.items():
        metrics_path = Path(result["paths"]["metrics"])
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in ["baseline", "seed", *reader.fieldnames]:
                    if name not in fieldnames:
                        fieldnames.append(name)
            for row in reader:
                row = dict(row)
                row.setdefault("split", str(split))
                row["baseline"] = BASELINE_NAME
                row["seed"] = int(seed)
                rows.append(row)
    _write_csv_rows_atomic(path, fieldnames, rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json_atomic(path, payload)


def run_vanilla_kd(args: argparse.Namespace) -> dict[str, Any]:
    _reject_amp(args)
    _validate_qvf_args(args)
    _validate_per_args(args)
    _seed_everything(int(args.seed))
    resolved = _resolved_paths(args)
    qvf_mode = _qvf_enabled(args)
    per_mode = _per_enabled(args)
    probe_only = bool(getattr(args, "per_loss_scale_probe", False))
    artifact_path_migration_allowed = bool(
        getattr(args, "allow_artifact_path_migration", False) or not (qvf_mode or per_mode or probe_only)
    )
    fp32_teacher_stage = stage_fp32_teacher_checkpoint(
        dataset_name=args.dataset,
        seed=int(args.seed),
        baseline_root=resolved["teacher_checkpoint"].parent.parent,
        explicit_teacher_checkpoint=resolved.get("requested_teacher_checkpoint"),
        staged_checkpoint=resolved["teacher_checkpoint"],
        force_restage=bool(args.force_stage_fp32_teacher),
        dry_run=bool(args.dry_run),
    )
    resolved["teacher_checkpoint"] = Path(str(fp32_teacher_stage["staged_checkpoint"]))
    teacher_checkpoint = resolved["teacher_checkpoint"]
    int4_checkpoint = resolved["int4_checkpoint"]
    qparams_csv = resolved["qparams_csv"]
    if not bool(args.dry_run) and not teacher_checkpoint.exists():
        raise FileNotFoundError(f"FP32 teacher checkpoint not found after staging: {teacher_checkpoint}")
    if not int4_checkpoint.exists():
        raise FileNotFoundError(f"INT4 calibrated student checkpoint not found: {int4_checkpoint}; run run_int4_student.py first")
    if not qparams_csv.exists():
        raise FileNotFoundError(f"INT4 qparams CSV not found: {qparams_csv}; run run_int4_student.py first")
    if not resolved["int4_manifest"].exists():
        raise FileNotFoundError(f"INT4 student canonical manifest not found: {resolved['int4_manifest']}; run run_int4_student.py first")
    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "resolved": {k: path_str(v) for k, v in resolved.items()},
                    "fp32_teacher_stage": fp32_teacher_stage,
                    "int4_checkpoint": path_str(int4_checkpoint),
                    "qparams_csv": path_str(qparams_csv),
                    "int4_manifest": path_str(resolved["int4_manifest"]),
                    "qvf_mode": bool(qvf_mode),
                    "qvf_protocol": _qvf_protocol_for_args(args) if qvf_mode else {"use_qvf_token_fidelity": False},
                    "per_mode": bool(per_mode),
                    "per_protocol": _per_protocol_for_args(args) if per_mode else {"use_projected_energy_retention": False},
                    "per_loss_scale_probe": bool(probe_only),
                    "artifact_path_migration_allowed": bool(artifact_path_migration_allowed),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return {"dry_run": True, "resolved": resolved, "fp32_teacher_stage": fp32_teacher_stage}

    paths = dataset_paths(args.dataset, str(teacher_checkpoint))
    label_mapping = load_label_mapping(paths.label_mapping)
    num_classes = num_predictor_classes(label_mapping)
    validate_split_labels(paths.train_csv, num_classes=num_classes, name="train")
    if not probe_only:
        validate_split_labels(paths.eval_csv, num_classes=num_classes, name="eval")
    if not (qvf_mode or per_mode or probe_only):
        validate_split_labels(paths.test_csv, num_classes=num_classes, name="test")
    if probe_only:
        label_buckets = {}
    elif qvf_mode or per_mode:
        label_buckets = infer_label_buckets_without_full_csv(paths)
    else:
        label_buckets = infer_label_buckets(paths)
    label_names = label_names_by_id(label_mapping)

    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    teacher, teacher_payload = load_trained_checkpoint(teacher_checkpoint, device=device)
    _assert_fp32_teacher_payload(teacher_payload, context="KD teacher")
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()

    student, student_init_payload = load_trained_checkpoint(
        int4_checkpoint,
        device=device,
        qparams_csv_override=qparams_csv,
    )
    student_kind = str(student_init_payload.get("model_kind") or "")
    if student_kind != "int4_student_init":
        raise RuntimeError(f"KD student init must be loaded from INT4 calibrated checkpoint, got model_kind={student_kind!r}")

    int4_manifest = _load_required_json(resolved["int4_manifest"], name="INT4 student manifest")
    int4_provenance = _validate_int4_student_provenance(
        manifest_path=resolved["int4_manifest"],
        manifest=int4_manifest,
        teacher_checkpoint=teacher_checkpoint,
        int4_checkpoint=int4_checkpoint,
        qparams_csv=qparams_csv,
        paths=paths,
        validate_test_split=not (qvf_mode or per_mode or probe_only),
        allow_path_migration=bool(artifact_path_migration_allowed),
    )
    _validate_loaded_int4_checkpoint_payload(
        checkpoint_path=int4_checkpoint,
        checkpoint=student_init_payload,
        int4_provenance=int4_provenance,
        teacher_checkpoint=teacher_checkpoint,
        paths=paths,
    )
    calibration_csv = int4_provenance["calibration_csv"]
    calibration_hash = str(int4_provenance["calibration_hash"])
    layers = [dict(row) for row in int4_provenance["quantized_layers"]]

    controller = getattr(student, "_baseline_quant_controller", None)
    if controller is None:
        raise RuntimeError("loaded INT4 calibrated student checkpoint is missing _baseline_quant_controller")
    initial_quant_audit = _assert_quant_training_state(student, controller, int4_provenance, context="before_kd_training")

    train_loader = build_loader(paths, split="train", batch_size=int(args.batch_size), num_workers=int(args.num_workers), seed=int(args.seed))
    if bool(getattr(args, "per_loss_scale_probe", False)):
        probe_payload = per_loss_scale_probe(
            student=student,
            teacher=teacher,
            loader=train_loader,
            device=device,
            alpha=float(args.alpha),
            temperature=float(args.temperature),
            num_classes=num_classes,
            max_batches=int(args.per_probe_batches),
            topk_ratio=float(PER_TOPK_RATIO),
            eps=float(args.per_epsilon),
        )
        probe_manifest = {
            **probe_payload,
            "dataset_name": paths.name,
            "seed": int(args.seed),
            "output_path": path_str(resolved["per_probe_output"]),
            "per_protocol": _per_protocol_for_args(args),
            "teacher_checkpoint": path_str(teacher_checkpoint),
            "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
            "int4_calibrated_student_checkpoint": path_str(resolved["int4_checkpoint"]),
            "int4_calibrated_student_checkpoint_sha256": file_sha256(resolved["int4_checkpoint"]),
            "qparams_csv": path_str(qparams_csv),
            "qparams_csv_sha256": file_sha256(qparams_csv),
            "qparams_hash": int4_provenance.get("qparams_hash"),
            "train_csv": path_str(paths.train_csv),
            "train_split_hash": file_sha256(paths.train_csv),
            "eval_csv_recorded_not_read": path_str(paths.eval_csv),
            "test_csv_recorded_not_read": False,
            "full_csv_read": False,
            "artifact_path_migration_allowed": bool(artifact_path_migration_allowed),
            "calibration_csv": path_str(calibration_csv),
            "calibration_hash": calibration_hash,
            "label_mapping": path_str(paths.label_mapping),
            "label_mapping_hash": file_sha256(paths.label_mapping),
            "quant_audit": dict(initial_quant_audit),
        }
        probe_manifest = attach_run_manifest_hash(probe_manifest)
        _write_json(resolved["per_probe_output"], probe_manifest)
        print(f"[baseline-kd:per-probe] output={resolved['per_probe_output']}")
        print(f"[baseline-kd:per-probe] valid_batches={probe_manifest['valid_batch_count']} valid_samples={probe_manifest['valid_sample_count']}")
        return {"probe": probe_manifest, "resolved": resolved}
    eval_loader = build_loader(paths, split="eval", batch_size=int(args.eval_batch_size), num_workers=int(args.num_workers), seed=int(args.seed))
    optimizer = _make_optimizer(student, backbone_lr=float(args.backbone_lr), classifier_lr=float(args.classifier_lr), weight_decay=float(args.weight_decay))
    total_steps = int(args.epochs) * max(1, len(train_loader))
    scheduler = _make_scheduler(optimizer, total_steps=total_steps, warmup_ratio=float(args.warmup_ratio))

    best_eval_macro_f1_label_universe_nanmean = float("-inf")
    best_epoch = 0
    best_eval: dict[str, Any] = {}
    best_quant_audit: dict[str, Any] = {}
    train_rows: list[dict[str, Any]] = []
    start_epoch = 1
    resume_state = _try_resume_training(
        args=args,
        resolved=resolved,
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        paths=paths,
        teacher_checkpoint=teacher_checkpoint,
        int4_checkpoint=int4_checkpoint,
        qparams_csv=qparams_csv,
        calibration_hash=calibration_hash,
        int4_provenance=int4_provenance,
    )
    if resume_state is not None:
        start_epoch = int(resume_state["next_epoch"])
        best_eval_macro_f1_label_universe_nanmean = float(resume_state["best_eval_macro_f1_label_universe_nanmean"])
        best_epoch = int(resume_state["best_epoch"])
        best_eval = dict(resume_state.get("best_eval") or {})
        best_quant_audit = dict(resume_state.get("best_quant_audit") or {})
        train_rows = [dict(row) for row in resume_state.get("train_rows", [])]
    if start_epoch > int(args.epochs):
        print(f"[baseline-kd:resume] training already complete at epoch={start_epoch - 1}; skipping train loop")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_dataset = getattr(train_loader, "dataset", None)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp=False,
            alpha=float(args.alpha),
            temperature=float(args.temperature),
            num_classes=num_classes,
            qvf_config={
                "enabled": True,
                "beta": float(args.qvf_beta),
                "layers": QVF_SENSITIVE_LAYERS,
                "topk_ratio": float(QVF_TOPK_RATIO),
                "layer_weights": _qvf_layer_weights_from_args(args),
            } if qvf_mode else None,
            per_config={
                "enabled": True,
                "lambda": float(args.per_lambda),
                "layers": PER_LAYERS,
                "topk_ratio": float(PER_TOPK_RATIO),
                "epsilon": float(args.per_epsilon),
            } if per_mode else None,
        )
        epoch_quant_audit = _assert_quant_training_state(student, controller, int4_provenance, context=f"after_epoch_{epoch}")
        eval_metrics = evaluate_student(model=student, loader=eval_loader, device=device, amp=False, num_classes=num_classes)
        eval_breakdown, _per_class_rows = compute_eval_breakdown(
            eval_metrics,
            label_names=label_names,
            label_buckets=label_buckets,
        )
        current = float(eval_breakdown.get("eval_macro_f1_label_universe_nanmean", float("nan")))
        row = {
            "epoch": int(epoch),
            **{key: float(value) for key, value in train_metrics.items()},
            "eval_loss": float(eval_metrics["loss"]),
            "eval_accuracy": float(eval_metrics["accuracy"]),
            "eval_macro_f1_label_universe_nanmean": current,
            "eval_macro_f1_observed_classes": float(eval_breakdown.get("eval_macro_f1_observed_classes", float("nan"))),
            "eval_macro_f1_legacy_zero_absent_diagnostic": float(eval_breakdown.get("eval_macro_f1_legacy_zero_absent_diagnostic", float("nan"))),
            "quant_audit_observer_enabled": bool(epoch_quant_audit.get("observer_enabled")),
            "quant_audit_fake_quant_enabled": bool(epoch_quant_audit.get("fake_quant_enabled")),
            "quant_audit_qparams_frozen": bool(epoch_quant_audit.get("qparams_frozen")),
        }
        if qvf_mode:
            qvf_protocol_payload = _qvf_protocol_for_args(args)
            row.update(
                {
                    "use_qvf_token_fidelity": True,
                    "qvf_beta": float(args.qvf_beta),
                    "qvf_protocol_version": qvf_protocol_payload["protocol_version"],
                    "qvf_method_variant": qvf_protocol_payload["method_variant"],
                    "qvf_layer_weight_policy": qvf_protocol_payload["layer_weight_policy"],
                    "qvf_topk_ratio": float(QVF_TOPK_RATIO),
                    "qvf_sensitive_layers": ",".join(f"L{layer}.{QVF_MODULE_NAME}" for layer in QVF_SENSITIVE_LAYERS),
                    "qvf_student_activation_semantic": "post_fake_quant_dequant_input_activation_with_autograd_graph",
                    "qvf_student_layerdrop_disabled_during_capture": True,
                    "qvf_teacher_topk_source": "fp32_token_energy",
                    "qvf_test_split_touched": False,
                }
            )
            if qvf_protocol_payload.get("sensitive_layer_weights"):
                row["qvf_weighted_variant"] = qvf_protocol_payload.get("qvf_weighted_variant")
                for layer_name, weight in dict(qvf_protocol_payload["sensitive_layer_weights"]).items():
                    row[f"qvf_layer_weight_{str(layer_name).replace('.', '_')}"] = float(weight)
        if per_mode:
            per_protocol_payload = _per_protocol_for_args(args)
            row.update(
                {
                    "use_projected_energy_retention": True,
                    "per_lambda": float(args.per_lambda),
                    "per_protocol_version": per_protocol_payload["protocol_version"],
                    "per_method_variant": per_protocol_payload["method_variant"],
                    "per_loss_active": bool(float(args.per_lambda) > 0.0),
                    "per_topk_ratio": float(PER_TOPK_RATIO),
                    "per_epsilon": float(args.per_epsilon),
                    "per_sensitive_layers": ",".join(f"L{layer}.{PER_MODULE_NAME}" for layer in PER_LAYERS),
                    "per_student_activation_semantic": "post_fake_quant_with_autograd_graph",
                    "per_teacher_topk_source": "fp32_token_energy",
                    "per_test_split_touched": False,
                }
            )
            row.setdefault("train_per_loss", 0.0)
            row.setdefault("per_valid_sample_count", 0.0)
            row.setdefault("per_valid_token_count", 0.0)
        train_rows.append(row)
        _write_train_log(resolved["train_log"], train_rows)
        print(f"[baseline-kd] epoch={epoch} train_loss={row['train_loss']:.6f} eval_macro_f1_label_universe_nanmean={current:.6f}")
        if math.isfinite(current) and current > best_eval_macro_f1_label_universe_nanmean:
            best_eval_macro_f1_label_universe_nanmean = current
            best_epoch = int(epoch)
            best_eval = {**eval_metrics, **eval_breakdown}
            best_quant_audit = dict(epoch_quant_audit)
            _save_checkpoint(
                resolved["best_checkpoint"],
                model=student,
                epoch=epoch,
                eval_metrics=best_eval,
                teacher_checkpoint=teacher_checkpoint,
                teacher_payload=teacher_payload,
                qparams_csv=qparams_csv,
                controller=controller,
                layers=layers,
                paths=paths,
                label_mapping=label_mapping,
                args=args,
                calibration_csv=calibration_csv,
                calibration_hash=calibration_hash,
                int4_provenance=int4_provenance,
                quant_audit=epoch_quant_audit,
            )
        last_eval = {**eval_metrics, **eval_breakdown}
        _save_checkpoint(
            resolved["last_checkpoint"],
            model=student,
            epoch=epoch,
            eval_metrics=last_eval,
            teacher_checkpoint=teacher_checkpoint,
            teacher_payload=teacher_payload,
            qparams_csv=qparams_csv,
            controller=controller,
            layers=layers,
            paths=paths,
            label_mapping=label_mapping,
            args=args,
            calibration_csv=calibration_csv,
            calibration_hash=calibration_hash,
            int4_provenance=int4_provenance,
            quant_audit=epoch_quant_audit,
            optimizer=optimizer,
            scheduler=scheduler,
            training_resume_state=_training_resume_state(
                epoch=epoch,
                best_epoch=best_epoch,
                best_eval_macro_f1_label_universe_nanmean=best_eval_macro_f1_label_universe_nanmean,
                best_eval=best_eval,
                best_quant_audit=best_quant_audit,
                train_rows=train_rows,
                best_checkpoint=resolved["best_checkpoint"],
                last_checkpoint=resolved["last_checkpoint"],
                rng_state=_rng_state_payload(device),
            ),
        )

    if start_epoch > int(args.epochs):
        if best_epoch <= 0:
            raise RuntimeError("Vanilla KD completed checkpoint has no finite best eval_macro_f1_label_universe_nanmean checkpoint")
        completed_manifest = _try_load_completed_run_manifest(args=args, resolved=resolved, qvf_mode=qvf_mode, per_mode=per_mode)
        if completed_manifest is not None:
            print(f"[baseline-kd:resume] completed manifest found; skipping final eval/export: {resolved['manifest_path']}")
            return {"manifest": completed_manifest, "resolved": resolved, "splits": completed_manifest.get("splits", {})}

    final_eval = evaluate_student(model=student, loader=eval_loader, device=device, amp=False, num_classes=num_classes)
    final_quant_audit = _assert_quant_training_state(student, controller, int4_provenance, context="after_final_eval")
    _save_checkpoint(
        resolved["last_checkpoint"],
        model=student,
        epoch=int(args.epochs),
        eval_metrics=final_eval,
        teacher_checkpoint=teacher_checkpoint,
        teacher_payload=teacher_payload,
        qparams_csv=qparams_csv,
        controller=controller,
        layers=layers,
        paths=paths,
        label_mapping=label_mapping,
        args=args,
        calibration_csv=calibration_csv,
        calibration_hash=calibration_hash,
        int4_provenance=int4_provenance,
        quant_audit=final_quant_audit,
        optimizer=optimizer,
        scheduler=scheduler,
        training_resume_state=_training_resume_state(
            epoch=int(args.epochs),
            best_epoch=best_epoch,
            best_eval_macro_f1_label_universe_nanmean=best_eval_macro_f1_label_universe_nanmean,
            best_eval=best_eval,
            best_quant_audit=best_quant_audit,
            train_rows=train_rows,
            best_checkpoint=resolved["best_checkpoint"],
            last_checkpoint=resolved["last_checkpoint"],
            rng_state=_rng_state_payload(device),
        ),
    )
    if best_epoch <= 0:
        raise RuntimeError("Vanilla KD never produced a finite eval_macro_f1_label_universe_nanmean checkpoint")

    export_model, export_checkpoint = load_trained_checkpoint(
        resolved["best_checkpoint"],
        device=device,
        qparams_csv_override=qparams_csv,
    )
    export_paths = dataset_paths(args.dataset, str(resolved["best_checkpoint"]))
    token_layer_indices = _resolve_export_token_layers(export_model, args)
    token_export = _token_export_mode_from_layers(export_model, token_layer_indices)
    export_split_hashes = _split_hash_payload(export_paths.eval_csv) if (qvf_mode or per_mode) else _split_hash_payload(paths=export_paths)
    export_run_identity_hash = build_run_identity(
        model_name=str(args.export_name),
        checkpoint_path=resolved["best_checkpoint"],
        checkpoint=export_checkpoint,
        split_hashes=export_split_hashes,
        label_mapping_hash=file_sha256(export_paths.label_mapping) or export_checkpoint.get("label_mapping_hash"),
        calibration_hash=export_checkpoint.get("calibration_hash") or calibration_hash,
        pooling=DEFAULT_POOLING,
        token_export=token_export,
        token_layers=token_layer_indices,
        seed=int(args.seed),
    )
    export_args = _formal_export_args(args, run_identity_hash=export_run_identity_hash, token_export=token_export)
    export_args.reference_root = str(resolved["reference_root"])
    output_dirs = _export_dirs(resolved["export_root"], str(args.export_name))
    split_results: dict[str, Any] = {}
    export_splits = ("eval",) if (qvf_mode or per_mode) else ("eval", "test")
    for split in export_splits:
        split_results[split] = export_eval_split(
            model=export_model,
            paths=export_paths,
            split=split,
            args=export_args,
            label_buckets=label_buckets,
            token_layer_indices=token_layer_indices,
            output_dirs=output_dirs,
            device=device,
            checkpoint_path=resolved["best_checkpoint"],
            quant_controller=getattr(export_model, "_baseline_quant_controller", None),
            quant_audit=getattr(export_model, "_baseline_quant_audit", None),
            run_identity_hash=export_run_identity_hash,
        )
    _write_combined_metrics(split_results, resolved["metrics_csv"], seed=int(args.seed))
    per_eval_diagnostic_path: Path | None = None
    per_mechanism_gap_path: Path | None = None
    per_eval_diagnostic_rows: list[dict[str, Any]] = []
    per_mechanism_gap_rows: list[dict[str, Any]] = []
    if per_mode:
        per_eval_loader = build_loader(
            export_paths,
            split="eval",
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )
        per_eval_diagnostic_rows = _per_eval_diagnostic_rows(
            student=export_model,
            teacher=teacher,
            loader=per_eval_loader,
            device=device,
            num_classes=num_classes,
            label_buckets=label_buckets,
            topk_ratio=float(PER_TOPK_RATIO),
            eps=float(args.per_epsilon),
        )
        per_eval_diagnostic_path = Path(resolved["seed_dir"]) / "per_eval_diagnostics.csv"
        _write_csv_rows_atomic(
            per_eval_diagnostic_path,
            [
                "split",
                "layer",
                "metric_name",
                "analysis_group",
                "metric_value",
                "n_total",
                "n_valid",
                "metric_direction",
                "aggregation_unit",
                "aggregation_fn",
                "metric_role",
            ],
            per_eval_diagnostic_rows,
        )
        per_mechanism_gap_rows = _per_mechanism_gap_rows(_read_csv_dict_rows(Path(resolved["metrics_csv"])))
        per_mechanism_gap_path = Path(resolved["seed_dir"]) / "per_mechanism_gaps.csv"
        _write_csv_rows_atomic(
            per_mechanism_gap_path,
            [
                "split",
                "field_name",
                "source_metric_name",
                "left_group",
                "right_group",
                "gap_direction",
                "left_value",
                "right_value",
                "gap_value",
                "available",
                "reason",
            ],
            per_mechanism_gap_rows,
        )
    oer_protocol_by_split = {str(split): result.get("oer_protocol") for split, result in split_results.items()}
    first_oer_protocol = next((value for value in oer_protocol_by_split.values() if isinstance(value, Mapping)), {})

    audio_cfg = audio_config_from_project()
    int4_provenance_payload = {
        "manifest_path": int4_provenance.get("manifest_path"),
        "manifest_sha256": int4_provenance.get("manifest_sha256"),
        "run_manifest_hash": int4_provenance.get("manifest_hash"),
        "manifest_run_identity_hash": int4_provenance.get("manifest_run_identity_hash"),
        "checkpoint": int4_provenance.get("checkpoint"),
        "checkpoint_sha256": int4_provenance.get("checkpoint_sha256"),
        "calibration_csv": path_str(int4_provenance.get("calibration_csv")),
        "calibration_csv_hash": int4_provenance.get("calibration_csv_hash"),
        "calibration_manifest_hash": int4_provenance.get("calibration_manifest_hash"),
        "calibration_hash": int4_provenance.get("calibration_hash"),
        "qparams_hash": int4_provenance.get("qparams_hash"),
        "qparams_csv_sha256": int4_provenance.get("qparams_csv_sha256"),
        "quantized_layers_hash": int4_provenance.get("quantized_layers_hash"),
        "quantized_layers": layers,
        "artifact_path_migration_allowed": bool(int4_provenance.get("artifact_path_migration_allowed", False)),
        "checkpoint_path_migrated": bool(int4_provenance.get("checkpoint_path_migrated", False)),
        "qparams_csv_path_migrated": bool(int4_provenance.get("qparams_csv_path_migrated", False)),
        "calibration_csv_path_migrated": bool(int4_provenance.get("calibration_csv_path_migrated", False)),
        "calibration_manifest_path_migrated": bool(int4_provenance.get("calibration_manifest_path_migrated", False)),
    }
    if qvf_mode:
        qvf_protocol_payload = _qvf_protocol_for_args(args)
        qvf_method_variant = _qvf_method_variant_from_args(args)
        qvf_loss_name = "weighted QVF" if qvf_protocol_payload.get("sensitive_layer_weights") else "QVF"
        qvf_split_hashes = {
            "train": file_sha256(paths.train_csv),
            "eval": file_sha256(paths.eval_csv),
            "calibration": int4_provenance.get("calibration_csv_hash"),
        }
        qvf_artifact_sha256 = {
            "qvf_stage1_checkpoint.pt": file_sha256(resolved["best_checkpoint"]),
            "qvf_stage1_last.pt": file_sha256(resolved["last_checkpoint"]),
            "qvf_train_log.csv": file_sha256(resolved["train_log"]),
            "int4_calibrated_student_checkpoint.pt": file_sha256(resolved["int4_checkpoint"]),
            "qparams.csv": file_sha256(qparams_csv),
            "int4_student_run_manifest.json": file_sha256(resolved["int4_manifest"]),
            **{key: value for split_result in split_results.values() for key, value in dict(split_result.get("artifact_sha256", {})).items()},
        }
        manifest = {
            "manifest_version": 2,
            "stage": "qvf_stage1_vanilla_kd",
            "protocol_version": qvf_protocol_payload["protocol_version"],
            "model_name": str(args.export_name),
            "model_kind": "vanilla_kd",
            "method_variant": qvf_method_variant,
            "dataset_name": paths.name,
            "seed": int(args.seed),
            "checkpoint": path_str(resolved["best_checkpoint"]),
            "checkpoint_sha256": file_sha256(resolved["best_checkpoint"]),
            "last_checkpoint": path_str(resolved["last_checkpoint"]),
            "last_checkpoint_sha256": file_sha256(resolved["last_checkpoint"]),
            "train_log": path_str(resolved["train_log"]),
            "train_log_sha256": file_sha256(resolved["train_log"]),
            "metrics_csv": path_str(resolved["metrics_csv"]),
            "teacher_checkpoint": path_str(teacher_checkpoint),
            "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
            "source_int4_init_hash": int4_provenance_payload["checkpoint_sha256"],
            "int4_calibrated_student_checkpoint": path_str(resolved["int4_checkpoint"]),
            "int4_calibrated_student_checkpoint_sha256": file_sha256(resolved["int4_checkpoint"]),
            "int4_student_manifest": int4_provenance_payload["manifest_path"],
            "int4_student_manifest_sha256": int4_provenance_payload["manifest_sha256"],
            "int4_student_run_manifest_hash": int4_provenance_payload["run_manifest_hash"],
            "int4_student_run_identity_hash": int4_provenance_payload["manifest_run_identity_hash"],
            "int4_student_provenance": int4_provenance_payload,
            "artifact_path_migration_allowed": bool(artifact_path_migration_allowed),
            "qparams_csv": path_str(qparams_csv),
            "qparams_csv_sha256": file_sha256(qparams_csv),
            "qparams_hash": export_checkpoint.get("qparams_hash"),
            "calibration_csv": path_str(calibration_csv),
            "calibration_csv_hash": int4_provenance.get("calibration_csv_hash"),
            "calibration_hash": calibration_hash,
            "label_mapping": path_str(paths.label_mapping),
            "label_mapping_hash": file_sha256(paths.label_mapping),
            "train_csv": path_str(paths.train_csv),
            "eval_csv": path_str(paths.eval_csv),
            "split_hashes": qvf_split_hashes,
            "split_hash": canonical_json_hash(qvf_split_hashes),
            "label_bucket_source": "train_csv.train_label_n_or_train_counts",
            "selection_metric": "eval_macro_f1_label_universe_nanmean",
            "best_epoch": int(best_epoch),
            "best_eval_macro_f1_label_universe_nanmean": float(best_eval_macro_f1_label_universe_nanmean),
            "training_resume": {
                "resume_enabled": bool(args.resume),
                "resumed_from_checkpoint": bool(resume_state is not None),
                "resume_checkpoint": path_str(resolved["last_checkpoint"]) if resume_state is not None else None,
                "resume_start_epoch": int(start_epoch),
                "last_checkpoint_save_policy": "after_every_epoch_and_after_final_eval",
                "last_checkpoint_contains_optimizer_state": True,
                "last_checkpoint_contains_scheduler_state": True,
                "checkpoint_write_policy": "atomic_tmp_replace",
            },
            "checkpoint_run_identity_hash": export_checkpoint.get("run_identity_hash"),
            "run_identity_hash": export_run_identity_hash,
            "use_qvf_token_fidelity": True,
            "qvf_beta": float(args.qvf_beta),
            "qvf_loss_definition": qvf_protocol_payload["qvf_loss_definition"],
            "qvf_protocol": qvf_protocol_payload,
            "qvf_layer_weight_policy": qvf_protocol_payload["layer_weight_policy"],
            "sensitive_layer_list": qvf_protocol_payload["sensitive_layer_list"],
            "exact_module_paths": qvf_protocol_payload["exact_module_paths"],
            "activation_hook_semantic": qvf_protocol_payload["activation_hook_semantic"],
            "student_activation_semantic": qvf_protocol_payload["student_activation_semantic"],
            "teacher_token_selection_source": qvf_protocol_payload["teacher_token_selection_source"],
            "topk_unit": qvf_protocol_payload["topk_unit"],
            "topk_source": qvf_protocol_payload["topk_source"],
            "topk_selected_on": qvf_protocol_payload["topk_selected_on"],
            "topk_ratio": qvf_protocol_payload["topk_ratio"],
            "topk_k_policy": qvf_protocol_payload["topk_k_policy"],
            "topk_policy": qvf_protocol_payload["topk_policy"],
            "padding_policy": qvf_protocol_payload["padding_policy"],
            "kd_protocol": {
                "loss": f"CE + alpha * T^2 * KL + beta * {qvf_loss_name}",
                "alpha": float(args.alpha),
                "temperature": float(args.temperature),
                "epochs": int(args.epochs),
                "optimizer": "AdamW",
                "backbone_lr": float(args.backbone_lr),
                "classifier_lr": float(args.classifier_lr),
                "scheduler": "cosine annealing",
                "warmup_ratio": float(args.warmup_ratio),
                "ce": "plain CrossEntropyLoss",
                "sampler": "normal shuffle train loader",
                "class_weight": None,
                "balanced_sampler": False,
                "long_tail_prior": False,
                "amp": False,
                "training_dtype": "float32",
                "loss_logits_dtype": "float32",
                "teacher_frozen": True,
                "qparams_frozen": True,
                "student_backbone_trainable": True,
            },
            "qparams_frozen": True,
            "fake_quant_enabled": True,
            "observer_enabled": False,
            "train_quant_audit_initial": initial_quant_audit,
            "train_quant_audit_best": best_quant_audit,
            "train_quant_audit_final": final_quant_audit,
            "export_quant_audit": getattr(export_model, "_baseline_quant_audit", None),
            "token_layers": [int(item) for item in token_layer_indices],
            "token_export": token_export,
            "export_all_token_layers": token_export == "all",
            "outputs": {str(split): dict(result.get("paths", {})) for split, result in split_results.items()},
            "artifact_sha256": qvf_artifact_sha256,
            "oer_protocol": oer_protocol_by_split,
            "test_split_touched": False,
            "test_isolation": {
                "test_split_touched": False,
                "test_csv_read": False,
                "full_csv_read": False,
                "test_loader_built": False,
                "test_forward_run": False,
                "test_metrics_exported": False,
                "test_predictions_exported": False,
                "test_path_recorded": False,
                "label_bucket_source": "train_csv.train_label_n_or_train_counts",
            },
        }
        if qvf_protocol_payload.get("sensitive_layer_weights"):
            manifest["qvf_weighted_variant"] = qvf_protocol_payload.get("qvf_weighted_variant")
            manifest["sensitive_layer_weights"] = qvf_protocol_payload.get("sensitive_layer_weights")
            manifest["qvf_layer_weights"] = qvf_protocol_payload.get("qvf_layer_weights")
        payload = attach_run_manifest_hash(manifest)
        _write_json(resolved["manifest_path"], payload)
        print(f"[baseline-kd:qvf] best_checkpoint={resolved['best_checkpoint']}")
        print(f"[baseline-kd:qvf] last_checkpoint={resolved['last_checkpoint']}")
        print(f"[baseline-kd:qvf] run_manifest_hash={payload['run_manifest_hash']}")
        return {"manifest": payload, "resolved": resolved, "splits": split_results}

    if per_mode:
        per_protocol_payload = _per_protocol_for_args(args)
        per_split_hashes = {
            "train": file_sha256(paths.train_csv),
            "eval": file_sha256(paths.eval_csv),
            "calibration": int4_provenance.get("calibration_csv_hash"),
        }
        per_artifact_sha256 = {
            "per_stage1_checkpoint.pt": file_sha256(resolved["best_checkpoint"]),
            "per_stage1_last.pt": file_sha256(resolved["last_checkpoint"]),
            "per_train_log.csv": file_sha256(resolved["train_log"]),
            "int4_calibrated_student_checkpoint.pt": file_sha256(resolved["int4_checkpoint"]),
            "qparams.csv": file_sha256(qparams_csv),
            "int4_student_run_manifest.json": file_sha256(resolved["int4_manifest"]),
            **{key: value for split_result in split_results.values() for key, value in dict(split_result.get("artifact_sha256", {})).items()},
        }
        if per_eval_diagnostic_path is not None:
            per_artifact_sha256["per_eval_diagnostics.csv"] = file_sha256(per_eval_diagnostic_path)
        if per_mechanism_gap_path is not None:
            per_artifact_sha256["per_mechanism_gaps.csv"] = file_sha256(per_mechanism_gap_path)
        per_eval_diagnostic_summary = {
            "path": path_str(per_eval_diagnostic_path) if per_eval_diagnostic_path is not None else None,
            "sha256": file_sha256(per_eval_diagnostic_path) if per_eval_diagnostic_path is not None else None,
            "metrics": _rows_by_name_group(per_eval_diagnostic_rows),
            "row_count": int(len(per_eval_diagnostic_rows)),
            "computed_from": "canonical_reload_best_checkpoint_eval_forward",
        }
        per_mechanism_gap_summary = {
            "path": path_str(per_mechanism_gap_path) if per_mechanism_gap_path is not None else None,
            "sha256": file_sha256(per_mechanism_gap_path) if per_mechanism_gap_path is not None else None,
            "gaps": _gap_rows_by_field(per_mechanism_gap_rows),
            "row_count": int(len(per_mechanism_gap_rows)),
            "computed_from": "canonical_export_eval_metrics",
        }
        run_identity_payload = export_checkpoint.get("run_identity_payload")
        if not isinstance(run_identity_payload, Mapping):
            run_identity_payload = {}
        manifest = {
            "manifest_version": 2,
            "stage": "per_stage1_vanilla_kd",
            "protocol_version": per_protocol_payload["protocol_version"],
            "model_name": str(args.export_name),
            "model_kind": "vanilla_kd",
            "method_name": PER_METHOD_NAME,
            "method_variant": per_protocol_payload["method_variant"],
            "method_status": "invalid",
            "method_status_reason": "formal_lambda_selection_not_run",
            "method_success": None,
            "selected_candidate": False,
            "dataset_name": paths.name,
            "seed": int(args.seed),
            "checkpoint": path_str(resolved["best_checkpoint"]),
            "checkpoint_sha256": file_sha256(resolved["best_checkpoint"]),
            "last_checkpoint": path_str(resolved["last_checkpoint"]),
            "last_checkpoint_sha256": file_sha256(resolved["last_checkpoint"]),
            "train_log": path_str(resolved["train_log"]),
            "train_log_sha256": file_sha256(resolved["train_log"]),
            "metrics_csv": path_str(resolved["metrics_csv"]),
            "teacher_checkpoint": path_str(teacher_checkpoint),
            "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
            "student_init_checkpoint_hash": int4_provenance_payload["checkpoint_sha256"],
            "source_int4_init_hash": int4_provenance_payload["checkpoint_sha256"],
            "int4_calibrated_student_checkpoint": path_str(resolved["int4_checkpoint"]),
            "int4_calibrated_student_checkpoint_sha256": file_sha256(resolved["int4_checkpoint"]),
            "int4_student_manifest": int4_provenance_payload["manifest_path"],
            "int4_student_manifest_sha256": int4_provenance_payload["manifest_sha256"],
            "int4_student_run_manifest_hash": int4_provenance_payload["run_manifest_hash"],
            "int4_student_run_identity_hash": int4_provenance_payload["manifest_run_identity_hash"],
            "int4_student_provenance": int4_provenance_payload,
            "artifact_path_migration_allowed": bool(artifact_path_migration_allowed),
            "qparams_csv": path_str(qparams_csv),
            "qparams_csv_sha256": file_sha256(qparams_csv),
            "qparams_hash": export_checkpoint.get("qparams_hash"),
            "calibration_csv": path_str(calibration_csv),
            "calibration_csv_hash": int4_provenance.get("calibration_csv_hash"),
            "calibration_hash": calibration_hash,
            "label_mapping": path_str(paths.label_mapping),
            "label_mapping_hash": file_sha256(paths.label_mapping),
            "train_csv": path_str(paths.train_csv),
            "eval_csv": path_str(paths.eval_csv),
            "split_hashes": per_split_hashes,
            "split_hash": canonical_json_hash(per_split_hashes),
            "label_bucket_source": "train_csv.train_label_n_or_train_counts",
            "train_split_hash": file_sha256(paths.train_csv),
            "eval_split_hash": file_sha256(paths.eval_csv),
            "checkpoint_run_identity_hash": export_checkpoint.get("run_identity_hash"),
            "run_identity_hash": export_run_identity_hash,
            "git_commit": export_checkpoint.get("git_commit") or run_identity_payload.get("git_commit"),
            "selection_metric": "eval_macro_f1_label_universe_nanmean",
            "best_epoch": int(best_epoch),
            "best_eval_macro_f1_label_universe_nanmean": float(best_eval_macro_f1_label_universe_nanmean),
            "training_resume": {
                "resume_enabled": bool(args.resume),
                "resumed_from_checkpoint": bool(resume_state is not None),
                "resume_checkpoint": path_str(resolved["last_checkpoint"]) if resume_state is not None else None,
                "resume_start_epoch": int(start_epoch),
                "last_checkpoint_save_policy": "after_every_epoch_and_after_final_eval",
                "last_checkpoint_contains_optimizer_state": True,
                "last_checkpoint_contains_scheduler_state": True,
                "checkpoint_write_policy": "atomic_tmp_replace",
            },
            "use_projected_energy_retention": True,
            "per_lambda": float(args.per_lambda),
            "per_loss_definition": per_protocol_payload["loss_definition"],
            "loss_definition": per_protocol_payload["loss_definition"],
            "per_protocol": per_protocol_payload,
            "target_layers": per_protocol_payload["target_layers"],
            "module_paths": per_protocol_payload["module_paths"],
            "activation_hook_semantic": per_protocol_payload["activation_hook_semantic"],
            "student_activation_semantic": per_protocol_payload["student_activation_semantic"],
            "teacher_activation_semantic": per_protocol_payload["teacher_activation_semantic"],
            "teacher_token_selection": per_protocol_payload["teacher_token_selection"],
            "topk_source": per_protocol_payload["topk_source"],
            "topk_ratio": per_protocol_payload["topk_ratio"],
            "effective_topk_policy": per_protocol_payload["effective_topk_policy"],
            "padding_policy": per_protocol_payload["padding_policy"],
            "epsilon": per_protocol_payload["epsilon"],
            "sample_weighting": per_protocol_payload["sample_weighting"],
            "tail_weighting": per_protocol_payload["tail_weighting"],
            "geometry_in_loss": per_protocol_payload["geometry_in_loss"],
            "formal_lambda_set": per_protocol_payload["formal_lambda_set"],
            "formal_lambda_policy": per_protocol_payload["formal_lambda_policy"],
            "kd_protocol": {
                "loss": "CE + alpha * T^2 * KL + lambda_per * L_PER",
                "alpha": float(args.alpha),
                "temperature": float(args.temperature),
                "epochs": int(args.epochs),
                "optimizer": "AdamW",
                "backbone_lr": float(args.backbone_lr),
                "classifier_lr": float(args.classifier_lr),
                "scheduler": "cosine annealing",
                "warmup_ratio": float(args.warmup_ratio),
                "ce": "plain CrossEntropyLoss",
                "sampler": "normal shuffle train loader",
                "class_weight": None,
                "balanced_sampler": False,
                "long_tail_prior": False,
                "amp": False,
                "training_dtype": "float32",
                "loss_logits_dtype": "float32",
                "teacher_frozen": True,
                "qparams_frozen": True,
                "student_backbone_trainable": True,
            },
            "per_diagnostic_metrics": {
                "training_log": [
                    "train_per_loss",
                    "per_valid_sample_count",
                    "per_valid_token_count",
                    "per_l9_mean_projected_retention",
                    "per_l9_mean_abs_retention_error",
                    "per_l9_mean_squared_retention_error",
                ],
                "eval_bucket_metrics": [
                    "projected_retention_mean",
                    "projected_retention_abs_error",
                    "projected_retention_squared_error",
                ],
                "bucket_source": "train_csv.train_label_n_or_train_counts",
                "analysis_groups": ["many", "medium", "few", "tail"],
                "tail_definition": "medium + few",
                "strict_few_diagnostic_only": True,
                "actual_eval_output": per_eval_diagnostic_summary,
            },
            "mechanism_eval_diagnostics": {
                "training_target_layers": ["L9.fc1"],
                "suggested_probe_layers": ["L1.fc1", "L2.fc1", "L3.fc1", "L4.fc1", "L9.fc1"],
                "primary_compare": "Vanilla KD vs PER",
                "metrics": [
                    "oer_retention",
                    "mean_norm_retention",
                    "teacher_topk_relative_distortion",
                    "topk_overlap",
                    "high_energy_scalar_exact_zero_rate",
                    "high_energy_scalar_strict_overflow_rate",
                ],
                "many_tail_gap_fields": [
                    "oer_gap_many_minus_tail",
                    "distortion_gap_tail_minus_many",
                    "topk_overlap_gap_many_minus_tail",
                ],
                "actual_gap_output": per_mechanism_gap_summary,
                "diagnostic_only_not_in_loss": True,
            },
            "candidate_selection": per_protocol_payload["selection_rule"],
            "predefined_stop_criterion": per_protocol_payload["predefined_stop_criterion"],
            "qparams_frozen": True,
            "fake_quant_enabled": True,
            "observer_enabled": False,
            "train_quant_audit_initial": initial_quant_audit,
            "train_quant_audit_best": best_quant_audit,
            "train_quant_audit_final": final_quant_audit,
            "export_quant_audit": getattr(export_model, "_baseline_quant_audit", None),
            "token_layers": [int(item) for item in token_layer_indices],
            "token_export": token_export,
            "export_all_token_layers": token_export == "all",
            "outputs": {str(split): dict(result.get("paths", {})) for split, result in split_results.items()},
            "artifact_sha256": per_artifact_sha256,
            "oer_protocol": oer_protocol_by_split,
            "test_split_touched": False,
            "test_isolation": per_protocol_payload["test_isolation"],
        }
        payload = attach_run_manifest_hash(manifest)
        _write_json(resolved["manifest_path"], payload)
        print(f"[baseline-kd:per] best_checkpoint={resolved['best_checkpoint']}")
        print(f"[baseline-kd:per] last_checkpoint={resolved['last_checkpoint']}")
        print(f"[baseline-kd:per] run_manifest_hash={payload['run_manifest_hash']}")
        return {"manifest": payload, "resolved": resolved, "splits": split_results}

    manifest = baseline_common_manifest(
        stage="baseline_vanilla_kd",
        dataset_name=paths.name,
        checkpoint=resolved["best_checkpoint"],
        train_csv=paths.train_csv,
        eval_csv=paths.eval_csv,
        test_csv=paths.test_csv,
        label_mapping=paths.label_mapping,
        calibration_csv=calibration_csv,
        calibration_hash=calibration_hash,
        quantized_layers=layers,
        bitwidth={"weight": 4, "activation": 4},
        input_length_samples=int(audio_cfg.target_num_samples),
        input_length_seconds=float(audio_cfg.target_seconds),
        sample_rate=int(audio_cfg.sample_rate),
        seed=int(args.seed),
        extra={
            "protocol_version": PROTOCOL_VERSION,
            "model_name": BASELINE_NAME,
            "teacher_checkpoint": path_str(teacher_checkpoint),
            "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
            "fp32_teacher_stage": dict(fp32_teacher_stage),
            "student_init": f"{INT4_BASELINE_NAME}_checkpoint_seed_{int(args.seed)}",
            "int4_calibrated_student_checkpoint": path_str(resolved["int4_checkpoint"]),
            "int4_calibrated_student_checkpoint_sha256": file_sha256(resolved["int4_checkpoint"]),
            "int4_student_manifest": int4_provenance_payload["manifest_path"],
            "int4_student_manifest_sha256": int4_provenance_payload["manifest_sha256"],
            "int4_student_run_manifest_hash": int4_provenance_payload["run_manifest_hash"],
            "int4_student_run_identity_hash": int4_provenance_payload["manifest_run_identity_hash"],
            "int4_student_provenance": int4_provenance_payload,
            "qparams_csv": path_str(qparams_csv),
            "qparams_csv_sha256": file_sha256(qparams_csv),
            "qparams_hash": export_checkpoint.get("qparams_hash"),
            "checkpoint_run_identity_hash": export_checkpoint.get("run_identity_hash"),
            "run_identity_hash": export_run_identity_hash,
            "model_kind": "vanilla_kd",
            "last_checkpoint": path_str(resolved["last_checkpoint"]),
            "last_checkpoint_sha256": file_sha256(resolved["last_checkpoint"]),
            "selection_metric": "eval_macro_f1_label_universe_nanmean",
            "best_epoch": int(best_epoch),
            "best_eval_macro_f1_label_universe_nanmean": float(best_eval_macro_f1_label_universe_nanmean),
            "training_resume": {
                "resume_enabled": bool(args.resume),
                "resumed_from_checkpoint": bool(resume_state is not None),
                "resume_checkpoint": path_str(resolved["last_checkpoint"]) if resume_state is not None else None,
                "resume_start_epoch": int(start_epoch),
                "last_checkpoint_save_policy": "after_every_epoch_and_after_final_eval",
                "last_checkpoint_contains_optimizer_state": True,
                "last_checkpoint_contains_scheduler_state": True,
                "checkpoint_write_policy": "atomic_tmp_replace",
            },
            "train_log": path_str(resolved["train_log"]),
            "metrics_csv": path_str(resolved["metrics_csv"]),
            "kd_protocol": {
                "loss": "CE + alpha * T^2 * KL",
                "alpha": float(args.alpha),
                "temperature": float(args.temperature),
                "epochs": int(args.epochs),
                "optimizer": "AdamW",
                "backbone_lr": float(args.backbone_lr),
                "classifier_lr": float(args.classifier_lr),
                "scheduler": "cosine annealing",
                "warmup_ratio": float(args.warmup_ratio),
                "ce": "plain CrossEntropyLoss",
                "sampler": "normal shuffle train loader",
                "class_weight": None,
                "balanced_sampler": False,
                "long_tail_prior": False,
                "amp": False,
                "training_dtype": "float32",
                "loss_logits_dtype": "float32",
                "teacher_frozen": True,
                "qparams_frozen": True,
                "student_backbone_trainable": True,
                "test_usage_discipline": "test is exported once for final reporting only; no checkpoint, hyperparameter, temperature, or alpha selection",
            },
            "token_layers": [int(item) for item in token_layer_indices],
            "token_export": token_export,
            "export_all_token_layers": token_export == "all",
            "teacher_frozen": True,
            "qparams_frozen": True,
            "student_backbone_trainable": True,
            "train_quant_audit_initial": initial_quant_audit,
            "train_quant_audit_best": best_quant_audit,
            "train_quant_audit_final": final_quant_audit,
            "export_quant_audit": getattr(export_model, "_baseline_quant_audit", None),
            "artifact_sha256": {
                "checkpoint.pt": file_sha256(resolved["best_checkpoint"]),
                "last_checkpoint.pt": file_sha256(resolved["last_checkpoint"]),
                "int4_calibrated_student_checkpoint.pt": file_sha256(resolved["int4_checkpoint"]),
                "qparams.csv": file_sha256(qparams_csv),
                "int4_student_run_manifest.json": file_sha256(resolved["int4_manifest"]),
                **{key: value for split_result in split_results.values() for key, value in dict(split_result.get("artifact_sha256", {})).items()},
            },
            "oer_protocol": oer_protocol_by_split,
            "oer_activation_position_used": first_oer_protocol.get("oer_activation_position_used"),
            "oer_hook_list_used": first_oer_protocol.get("oer_hook_list_used"),
            "oer_hook_list_used_hash": first_oer_protocol.get("oer_hook_list_used_hash"),
            "splits": split_results,
        },
    )
    payload = attach_run_manifest_hash(manifest)
    _write_json(resolved["manifest_path"], payload)
    _write_run_manifests_atomic({"seed": resolved["seed_dir"], **output_dirs}, payload)
    print(f"[baseline-kd] best_checkpoint={resolved['best_checkpoint']}")
    print(f"[baseline-kd] last_checkpoint={resolved['last_checkpoint']}")
    print(f"[baseline-kd] run_manifest_hash={payload['run_manifest_hash']}")
    return {"manifest": payload, "resolved": resolved, "splits": split_results}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_vanilla_kd(args)


if __name__ == "__main__":
    main()
