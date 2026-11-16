from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader


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
    build_run_identity,
    export_eval_split,
    load_trained_checkpoint,
    move_batch_to_device,
    prepare_encoder_input_for_export,
)
from src.baseline import quant_checkpoint as qck
from src.baseline.fp32_teacher_staging import stage_fp32_teacher_checkpoint
from src.baseline.manifest import (
    attach_run_manifest_hash,
    baseline_common_manifest,
    canonical_json_hash,
    file_sha256,
    path_str,
    qparams_reuse_allowed,
    write_run_manifests,
)
from src.data.audio_input import AudioInputDataset, audio_input_collate_fn
from src.quant.ffn_quant_injector import apply_fixed_qparams_to_quant_module
from src.quant.quant_core import prepare_beats_ffn_int4_quantization_from_project_config
from src.train.train_beats_classifier_head import (
    dataset_paths,
    infer_label_buckets,
    load_label_mapping,
    num_predictor_classes,
    validate_split_labels,
)


PROTOCOL_VERSION = "baseline_int4_student_ptq.v1"
BASELINE_NAME = "int4_student"
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


def _dataset_cfg(dataset_name: str | None) -> Mapping[str, Any]:
    name = dataset_name or str(getattr(project_config, "ACTIVE_DATASET"))
    datasets = getattr(project_config, "DATASETS")
    if name not in datasets:
        raise ValueError(f"Unknown dataset={name!r}; available={tuple(datasets)}")
    return datasets[name]


def _dataset_root(dataset_name: str | None) -> Path:
    return Path(_dataset_cfg(dataset_name)["ROOT"])


def _default_baseline_root(dataset_name: str | None) -> Path:
    return _dataset_root(dataset_name) / "baseline" / "student"


def _default_teacher_baseline_root(dataset_name: str | None) -> Path:
    return _dataset_root(dataset_name) / "baseline" / "teacher"


def _default_teacher_seed_dir(dataset_name: str | None, seed: int) -> Path:
    return _default_teacher_baseline_root(dataset_name) / f"seed_{int(seed)}"


def _default_teacher_checkpoint(dataset_name: str | None, seed: int) -> Path:
    return _default_teacher_seed_dir(dataset_name, seed) / "checkpoints" / f"fp32_teacher_seed_{int(seed)}.pt"


def _default_calibration_csv(dataset_name: str | None) -> Path:
    cfg = _dataset_cfg(dataset_name)
    value = cfg.get("CALIBRATION_CSV")
    if value is None:
        value = Path(cfg["ROOT"]) / "splits" / "calibration.csv"
    return Path(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline INT4 student runner: FP32 teacher direct PTQ, calibration, frozen qparams, eval only.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--baseline-root", default=None, help="Root for INT4 student per-seed outputs; default: <dataset_root>/baseline/student.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--teacher-checkpoint", default=None, help="Optional FP32 teacher source for staging; default uses the same-seed baseline fp32_teacher checkpoint.")
    parser.add_argument("--force-stage-fp32-teacher", action="store_true", help="Overwrite the per-seed staged FP32 teacher checkpoint when its hash differs from the source.")
    parser.add_argument("--calibration-csv", default=None, help="Train-derived calibration CSV; default: config.DATASETS[dataset]['CALIBRATION_CSV'].")
    parser.add_argument("--calibration-manifest", default=None)
    parser.add_argument("--qparams-csv", default=None, help="Default: <dataset_root>/baseline/student/seed_<seed>/qparams/int4_qparams.csv.")
    parser.add_argument("--quantized-layers-json", default=None, help="Default: <dataset_root>/baseline/student/seed_<seed>/qparams/quantized_layers.json.")
    parser.add_argument("--manifest-path", default=None, help="Default: <dataset_root>/baseline/student/seed_<seed>/manifests/int4_student_manifest.json.")
    parser.add_argument("--int4-checkpoint", default=None, help="Default: <dataset_root>/baseline/student/seed_<seed>/checkpoints/int4_student_calibrated_seed_<seed>.pt.")
    parser.add_argument("--export-root", default=None, help="Default: <dataset_root>/baseline/student/seed_<seed>.")
    parser.add_argument("--export-name", default=BASELINE_NAME)
    parser.add_argument("--reference-root", default=None, help="Seed export root containing fp32_teacher reference artifacts; default: <dataset_root>/baseline/teacher/seed_<seed>.")
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT)
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS)
    parser.add_argument("--export-all-token-layers", action="store_true")
    parser.add_argument("--max-export-payload-gib", type=float, default=16.0)
    parser.add_argument("--calibration-batch-size", type=_positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=8)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reuse-matching-qparams", action="store_true", help="Reuse qparams only when the stored manifest fully matches this run protocol.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and protocol without running PTQ/export.")
    return parser.parse_args(argv)

def _reject_amp(args: argparse.Namespace) -> None:
    if bool(getattr(args, "amp", False)):
        raise RuntimeError("Baseline INT4 student formal calibration forbids AMP; remove --amp and regenerate FP32 qparams.")

def _resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    baseline_root = Path(args.baseline_root) if args.baseline_root else _default_baseline_root(args.dataset)
    teacher_seed_dir = _default_teacher_seed_dir(args.dataset, int(args.seed))
    seed_dir = baseline_root / f"seed_{int(args.seed)}"
    return {
        "seed_dir": seed_dir,
        "teacher_baseline_root": teacher_seed_dir.parent,
        "teacher_checkpoint": _default_teacher_checkpoint(args.dataset, int(args.seed)),
        "requested_teacher_checkpoint": Path(args.teacher_checkpoint) if args.teacher_checkpoint else None,
        "reference_root": Path(args.reference_root) if args.reference_root else teacher_seed_dir,
        "calibration_csv": Path(args.calibration_csv) if args.calibration_csv else _default_calibration_csv(args.dataset),
        "calibration_manifest": Path(args.calibration_manifest) if args.calibration_manifest else Path(_dataset_cfg(args.dataset).get("CALIBRATION_MANIFEST", "")),
        "qparams_csv": Path(args.qparams_csv) if args.qparams_csv else seed_dir / "qparams" / "int4_qparams.csv",
        "quantized_layers_json": Path(args.quantized_layers_json) if args.quantized_layers_json else seed_dir / "qparams" / "quantized_layers.json",
        "manifest_path": Path(args.manifest_path) if args.manifest_path else seed_dir / "manifests" / "int4_student_manifest.json",
        "int4_checkpoint": Path(args.int4_checkpoint) if args.int4_checkpoint else seed_dir / "checkpoints" / f"int4_student_calibrated_seed_{int(args.seed)}.pt",
        "export_root": Path(args.export_root) if args.export_root else seed_dir,
        "metrics_csv": seed_dir / "int4_student_metrics.csv",
    }


def _teacher_stage_source(resolved: Mapping[str, Path | None]) -> Path:
    requested = resolved.get("requested_teacher_checkpoint")
    if requested is not None:
        return Path(requested)
    return Path(resolved["teacher_checkpoint"])


def _calibration_manifest_path(path: Path) -> Path | None:
    text = str(path)
    if not text:
        return None
    return path



def build_calibration_loader(paths: Any, calibration_csv: Path, *, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    dataset = AudioInputDataset(
        calibration_csv,
        mode="calibration",
        dataset_root=Path(paths.root),
        project_root=PROJECT_ROOT,
        audio_config=audio_config_from_project(),
        global_seed=int(seed),
        epoch=0,
        return_dict=True,
        require_label_id=True,
        include_audio_stats=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(audio_input_collate_fn, include_audio_stats=False),
    )


def _run_calibration_forward(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = move_batch_to_device(batch, device)
            encoder_input, encoder_padding_mask, _token_padding_mask = prepare_encoder_input_for_export(
                model,
                moved["waveform"],
                moved.get("padding_mask"),
            )
            model.encoder(encoder_input.float(), padding_mask=encoder_padding_mask)

def run_two_pass_calibration(controller: Any, model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> None:
    """Consume a fixed calibration loader and regenerate qparams; never creates or mutates calibration.csv."""

    if controller.activation_quant_enabled:
        controller.prepare_calibration_range(reset_observers=True)
        _run_calibration_forward(model, loader, device=device)
        controller.prepare_calibration_histogram()
        _run_calibration_forward(model, loader, device=device)
        controller.freeze_qparams()
    controller.qparams_frozen = True
    controller.prepare_quant_forward()

def _calibration_protocol_identity(args: argparse.Namespace) -> dict[str, Any]:
    quant_cfg = getattr(project_config, "QUANT", {})
    observer_cfg = getattr(project_config, "OBSERVER", {})
    weight_cfg = dict(quant_cfg.get("WEIGHT", {})) if isinstance(quant_cfg, Mapping) else {}
    activation_cfg = dict(quant_cfg.get("ACTIVATION", {})) if isinstance(quant_cfg, Mapping) else {}
    activation_observer = dict(observer_cfg.get("ACTIVATION", {})) if isinstance(observer_cfg, Mapping) else {}
    weight_observer = dict(observer_cfg.get("WEIGHT", {})) if isinstance(observer_cfg, Mapping) else {}
    return {
        "calibration_amp": False,
        "calibration_forward_dtype": "float32",
        "calibration_num_passes": 2,
        "calibration_passes": "range,histogram",
        "calibration_runner": "src.baseline.run_int4_student.run_two_pass_calibration",
        "calibration_csv_role": "fixed_input_only",
        "calibration_csv_created_by_runner": False,
        "calibration_batch_size": int(args.calibration_batch_size),
        "calibration_num_workers": int(args.num_workers),
        "activation_calibration_method": str(activation_cfg.get("CALIBRATION_METHOD", "percentile")),
        "activation_percentile_mode": str(activation_cfg.get("PERCENTILE_MODE", "retained_mass")),
        "activation_percentile": float(activation_cfg.get("PERCENTILE", 99.9)),
        "activation_observer_bins": int(activation_cfg.get("OBSERVER_BINS", activation_observer.get("HISTOGRAM_BINS", 2048))),
        "activation_observer_backend": str(activation_observer.get("PERCENTILE_BACKEND", "two_pass_histogram")),
        "activation_observer_exclude_padding": bool(activation_observer.get("EXCLUDE_PADDING", True)),
        "activation_eps": float(activation_cfg.get("EPS", activation_observer.get("EPS", 1e-8))),
        "weight_observer_backend": str(weight_observer.get("PERCENTILE_BACKEND", "two_pass_histogram")),
        "weight_eps": float(weight_cfg.get("EPS", weight_observer.get("EPS", 1e-8))),
        "qparams_csv_io": "src.baseline.quant_checkpoint",
    }


def _attach_calibration_protocol_identity(
    layers: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    protocol = _calibration_protocol_identity(args)
    return [{**dict(row), **protocol} for row in layers]

def apply_qparams_to_controller(controller: Any, qparams: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    missing = sorted(set(controller.modules) - set(qparams))
    extra = sorted(set(qparams) - set(controller.modules))
    if missing or extra:
        raise RuntimeError(f"qparams target mismatch: missing={missing}, extra={extra}")
    for module_path, module in sorted(controller.modules.items()):
        module_qparams = qparams[module_path]
        apply_fixed_qparams_to_quant_module(
            module,
            mode="w4a4",
            weight_qparams=module_qparams["weight"],
            activation_qparams=module_qparams["activation"],
            require_qparams=True,
        )
    controller.qparams_frozen = True
    controller.prepare_quant_forward()


def freeze_runtime_qparams(controller: Any, qparams_csv: Path) -> None:
    apply_qparams_to_controller(controller, qck.load_qparams_csv(qparams_csv))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _write_combined_metrics(split_results: Mapping[str, Any], path: Path, *, baseline_name: str, seed: int) -> None:
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
                row["baseline"] = baseline_name
                row["seed"] = int(seed)
                rows.append(row)
    if not rows:
        raise RuntimeError("no INT4 metrics rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _save_int4_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    teacher_checkpoint: Path,
    teacher_payload: Mapping[str, Any],
    qparams_csv: Path,
    layers_json: Path,
    layers: Sequence[Mapping[str, Any]],
    controller: Any,
    paths: Any,
    label_mapping: Mapping[str, int],
    calibration_csv: Path,
    calibration_hash: str | None,
    seed: int,
) -> dict[str, Any]:
    cfg = dict(teacher_payload.get("cfg") or {})
    if not cfg:
        raise KeyError("FP32 teacher payload missing cfg; cannot create reproducible INT4 checkpoint")
    qparams = qck.load_qparams_csv(qparams_csv)
    qparams_hash_value = qck.qparams_hash(qparams)
    controller_qparams_hash = qck.qparams_hash(qck.collect_qparams_from_controller(controller))
    if controller_qparams_hash != qparams_hash_value:
        raise RuntimeError(
            "controller qparams and qparams_csv disagree before saving INT4 checkpoint: "
            f"controller={controller_qparams_hash}, csv={qparams_hash_value}"
        )
    qparams_metadata = qck.qparams_csv_metadata(qparams_csv)
    if calibration_hash and qparams_metadata.get("calibration_hash") and str(qparams_metadata["calibration_hash"]) != str(calibration_hash):
        raise RuntimeError(
            "qparams_csv calibration hash does not match INT4 calibration source: "
            f"csv={qparams_metadata.get('calibration_hash')}, calibration={calibration_hash}"
        )
    first_layer = dict(layers[0]) if layers else {}
    source_beats_checkpoint = teacher_payload.get("source_beats_checkpoint") or teacher_payload.get("source_checkpoint")
    identity_payload = qck.checkpoint_identity_payload(
        model_kind="int4_student_init",
        cfg=cfg,
        label_mapping=label_mapping,
        label_mapping_hash=file_sha256(paths.label_mapping),
        source_beats_checkpoint=source_beats_checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        quantized_layers=layers,
        qparams=qparams,
        qparams_csv=qparams_csv,
        calibration_csv=calibration_csv,
        calibration_hash=calibration_hash,
        pooling=DEFAULT_POOLING,
        seed=int(seed),
    )
    if not identity_payload.get("source_beats_checkpoint_hash"):
        identity_payload["source_beats_checkpoint_hash"] = teacher_payload.get("source_beats_checkpoint_hash") or teacher_payload.get("source_checkpoint_sha256")
    run_identity_hash = canonical_json_hash(identity_payload)
    source_beats_checkpoint_sha256 = identity_payload.get("source_beats_checkpoint_sha256") or identity_payload.get("source_beats_checkpoint_hash")
    teacher_checkpoint_sha256 = file_sha256(teacher_checkpoint)
    qparams_csv_sha256 = file_sha256(qparams_csv)
    audio_input_py_sha256 = identity_payload.get("audio_input_py_sha256") or identity_payload.get("audio_input_hash")
    group_utils_py_sha256 = identity_payload.get("group_utils_py_sha256") or identity_payload.get("group_utils_hash")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact_type": "baseline_int4_calibrated_student_checkpoint",
        "model_kind": "int4_student_init",
        "cfg": cfg,
        "model_state_dict": model.state_dict(),
        "label_mapping": dict(label_mapping),
        "label_mapping_hash": file_sha256(paths.label_mapping),
        "source_beats_checkpoint": path_str(source_beats_checkpoint),
        "source_beats_checkpoint_sha256": source_beats_checkpoint_sha256,
        "source_beats_checkpoint_hash": source_beats_checkpoint_sha256,
        "teacher_checkpoint": path_str(teacher_checkpoint),
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
        "teacher_checkpoint_hash": teacher_checkpoint_sha256,
        "quantized_layers": [dict(row) for row in layers],
        "quantized_layers_hash": identity_payload["quantized_layers_hash"],
        "quantized_layers_json": path_str(layers_json),
        "quantized_layers_json_hash": file_sha256(layers_json),
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
        "qparams_csv_hash": qparams_csv_sha256,
        "qparams_csv_semantic_hash": qparams_hash_value,
        "qparams_csv_metadata": qparams_metadata,
        "calibration_csv": path_str(calibration_csv),
        "calibration_hash": calibration_hash,
        "audio_input_py_sha256": audio_input_py_sha256,
        "audio_input_hash": audio_input_py_sha256,
        "group_utils_py_sha256": group_utils_py_sha256,
        "group_utils_hash": group_utils_py_sha256,
        "pooling": DEFAULT_POOLING,
        "seed": int(seed),
        "git_commit": identity_payload.get("git_commit"),
        "run_identity_hash": run_identity_hash,
        "run_identity_payload": identity_payload,
        "controller_state_summary": controller.state_summary(),
        "load_note": "Load through src.baseline.export_eval.load_trained_checkpoint; it injects INT4, applies qparams, freezes qparams, disables observer, enables fake quant, strict-loads state_dict, and audits quant state.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _qparams_manifest_fields(qparams_csv: Path) -> dict[str, Any]:
    csv_sha256 = file_sha256(qparams_csv)
    fields: dict[str, Any] = {
        "qparams_csv": path_str(qparams_csv),
        "qparams_csv_hash": csv_sha256,
        "qparams_csv_sha256": csv_sha256,
        "qparams_hash": None,
    }
    if qparams_csv.exists():
        fields["qparams_hash"] = qck.qparams_hash(qck.load_qparams_csv(qparams_csv))
    return fields

def _expected_manifest(
    args: argparse.Namespace,
    resolved: Mapping[str, Path],
    paths: Any,
    layers: Sequence[Mapping[str, Any]],
    *,
    fp32_teacher_stage: Mapping[str, Any],
) -> dict[str, Any]:
    audio_cfg = audio_config_from_project()
    calibration_manifest = _calibration_manifest_path(resolved["calibration_manifest"])
    return baseline_common_manifest(
        stage="baseline_int4_student_ptq",
        dataset_name=paths.name,
        checkpoint=resolved["teacher_checkpoint"],
        train_csv=paths.train_csv,
        eval_csv=paths.eval_csv,
        test_csv=paths.test_csv,
        label_mapping=paths.label_mapping,
        calibration_csv=resolved["calibration_csv"],
        calibration_manifest=calibration_manifest,
        quantized_layers=layers,
        bitwidth={"weight": 4, "activation": 4},
        input_length_samples=int(audio_cfg.target_num_samples),
        input_length_seconds=float(audio_cfg.target_seconds),
        sample_rate=int(audio_cfg.sample_rate),
        seed=int(args.seed),
        extra={
            "protocol_version": PROTOCOL_VERSION,
            "baseline_name": BASELINE_NAME,
            "teacher_checkpoint": path_str(resolved["teacher_checkpoint"]),
            "teacher_checkpoint_sha256": file_sha256(resolved["teacher_checkpoint"]),
            "fp32_teacher_stage": dict(fp32_teacher_stage),
            **_qparams_manifest_fields(resolved["qparams_csv"]),
            "quantized_layers_json": path_str(resolved["quantized_layers_json"]),
            "int4_calibrated_checkpoint": path_str(resolved["int4_checkpoint"]),
            "export_root": path_str(resolved["export_root"]),
            "export_name": str(args.export_name),
            "quant_protocol": {
                "source": "FP32 teacher direct PTQ",
                "target_scope": "BEATs Transformer FFN fc1/fc2 only",
                "calibration_source": "train-derived calibration.csv",
                "calibration_identity": _calibration_protocol_identity(args),
                "qparams_policy": "regenerate unless --reuse-matching-qparams and manifest fully matches",
                "finetune": False,
                "qat": False,
                "kd": False,
                "test_usage_discipline": "test is evaluated once for final reporting only; no qparams or hyperparameter selection",
            },
        },
    )


def run_int4_student(args: argparse.Namespace) -> dict[str, Any]:
    _reject_amp(args)
    resolved = _resolved_paths(args)
    fp32_teacher_stage = stage_fp32_teacher_checkpoint(
        dataset_name=args.dataset,
        seed=int(args.seed),
        baseline_root=resolved["teacher_baseline_root"],
        explicit_teacher_checkpoint=_teacher_stage_source(resolved),
        staged_checkpoint=resolved["teacher_checkpoint"],
        force_restage=bool(args.force_stage_fp32_teacher),
        dry_run=bool(args.dry_run),
    )
    resolved["teacher_checkpoint"] = Path(str(fp32_teacher_stage["staged_checkpoint"]))
    teacher_checkpoint = resolved["teacher_checkpoint"]
    if not bool(args.dry_run) and not teacher_checkpoint.exists():
        raise FileNotFoundError(f"FP32 teacher checkpoint not found after staging: {teacher_checkpoint}")
    if not resolved["calibration_csv"].exists():
        raise FileNotFoundError(f"calibration CSV not found: {resolved['calibration_csv']}")
    if bool(args.dry_run):
        print(json.dumps({"resolved": {k: path_str(v) for k, v in resolved.items()}, "fp32_teacher_stage": fp32_teacher_stage}, indent=2, ensure_ascii=False))
        return {"dry_run": True, "resolved": resolved, "fp32_teacher_stage": fp32_teacher_stage}

    paths = dataset_paths(args.dataset, str(teacher_checkpoint))
    label_mapping = load_label_mapping(paths.label_mapping)
    num_classes = num_predictor_classes(label_mapping)
    validate_split_labels(paths.eval_csv, num_classes=num_classes, name="eval")
    validate_split_labels(paths.test_csv, num_classes=num_classes, name="test")
    validate_split_labels(resolved["calibration_csv"], num_classes=num_classes, name="calibration")
    label_buckets = infer_label_buckets(paths)

    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    model, teacher_payload = load_trained_checkpoint(teacher_checkpoint, device=device)
    controller = prepare_beats_ffn_int4_quantization_from_project_config(model)
    calibration_csv_hash = file_sha256(resolved["calibration_csv"])
    layers = _attach_calibration_protocol_identity(qck.collect_quantized_layers(controller, calibration_csv_hash=calibration_csv_hash), args=args)
    expected_manifest = _expected_manifest(args, resolved, paths, layers, fp32_teacher_stage=fp32_teacher_stage)

    reused = False
    if bool(args.reuse_matching_qparams) and resolved["qparams_csv"].exists() and resolved["manifest_path"].exists():
        expected_manifest.update(_qparams_manifest_fields(resolved["qparams_csv"]))
        ok, mismatches = qparams_reuse_allowed(
            resolved["manifest_path"],
            expected_manifest,
            qparams_csv_path=resolved["qparams_csv"],
            qparams_hash_value=expected_manifest.get("qparams_hash"),
        )
        if ok:
            print(f"[baseline-int4] reuse matching qparams: {resolved['qparams_csv']}")
            freeze_runtime_qparams(controller, resolved["qparams_csv"])
            reused = True
        else:
            print("[baseline-int4] existing qparams manifest mismatch; recalibrating")
            for item in mismatches[:8]:
                print(f"[baseline-int4] mismatch: {item}")

    if not reused:
        calibration_loader = build_calibration_loader(
            paths,
            resolved["calibration_csv"],
            batch_size=int(args.calibration_batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )
        run_two_pass_calibration(controller, model, calibration_loader, device=device)
        qck.write_qparams_csv(resolved["qparams_csv"], controller, calibration_csv_hash=calibration_csv_hash)
        freeze_runtime_qparams(controller, resolved["qparams_csv"])
    qck.write_quantized_layers_json(resolved["quantized_layers_json"], layers)

    int4_checkpoint_payload = _save_int4_checkpoint(
        resolved["int4_checkpoint"],
        model=model,
        teacher_checkpoint=teacher_checkpoint,
        teacher_payload=teacher_payload,
        qparams_csv=resolved["qparams_csv"],
        layers_json=resolved["quantized_layers_json"],
        layers=layers,
        controller=controller,
        paths=paths,
        label_mapping=label_mapping,
        calibration_csv=resolved["calibration_csv"],
        calibration_hash=calibration_csv_hash,
        seed=int(args.seed),
    )

    export_model, export_checkpoint = load_trained_checkpoint(resolved["int4_checkpoint"], device=device)
    export_paths = dataset_paths(args.dataset, str(resolved["int4_checkpoint"]))
    token_layer_indices = _resolve_export_token_layers(export_model, args)
    token_export = _token_export_mode_from_layers(export_model, token_layer_indices)
    export_run_identity_hash = build_run_identity(
        model_name=str(args.export_name),
        checkpoint_path=resolved["int4_checkpoint"],
        checkpoint=export_checkpoint,
        split_hashes=_split_hash_payload(paths=export_paths),
        label_mapping_hash=file_sha256(export_paths.label_mapping) or export_checkpoint.get("label_mapping_hash"),
        calibration_hash=export_checkpoint.get("calibration_hash") or calibration_csv_hash,
        pooling=DEFAULT_POOLING,
        token_export=token_export,
        token_layers=token_layer_indices,
        seed=int(args.seed),
    )
    export_args = _formal_export_args(args, run_identity_hash=export_run_identity_hash, token_export=token_export)
    export_args.reference_root = str(resolved["reference_root"])
    output_dirs = _export_dirs(resolved["export_root"], str(args.export_name))
    split_results: dict[str, Any] = {}
    for split in ("eval", "test"):
        split_results[split] = export_eval_split(
            model=export_model,
            paths=export_paths,
            split=split,
            args=export_args,
            label_buckets=label_buckets,
            token_layer_indices=token_layer_indices,
            output_dirs=output_dirs,
            device=device,
            checkpoint_path=resolved["int4_checkpoint"],
            quant_controller=getattr(export_model, "_baseline_quant_controller", None),
            quant_audit=getattr(export_model, "_baseline_quant_audit", None),
            run_identity_hash=export_run_identity_hash,
        )
    _write_combined_metrics(split_results, resolved["metrics_csv"], baseline_name=BASELINE_NAME, seed=int(args.seed))
    oer_protocol_by_split = {str(split): result.get("oer_protocol") for split, result in split_results.items()}
    first_oer_protocol = next((value for value in oer_protocol_by_split.values() if isinstance(value, Mapping)), {})

    final_manifest = dict(expected_manifest)
    final_manifest.update(
        {
            "created_qparams": not reused,
            "reused_qparams": bool(reused),
            **_qparams_manifest_fields(resolved["qparams_csv"]),
            "quantized_layers_json_hash": file_sha256(resolved["quantized_layers_json"]),
            "checkpoint": path_str(resolved["int4_checkpoint"]),
            "checkpoint_sha256": file_sha256(resolved["int4_checkpoint"]),
            "teacher_checkpoint": path_str(teacher_checkpoint),
            "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
            "int4_calibrated_checkpoint_sha256": file_sha256(resolved["int4_checkpoint"]),
            "checkpoint_run_identity_hash": int4_checkpoint_payload.get("run_identity_hash"),
            "run_identity_hash": export_run_identity_hash,
            "model_kind": "int4_student_init",
            "token_layers": [int(item) for item in token_layer_indices],
            "token_export": token_export,
            "export_all_token_layers": token_export == "all",
            "controller_state_summary": controller.state_summary(),
            "export_quant_audit": getattr(export_model, "_baseline_quant_audit", None),
            "artifact_sha256": {
                "checkpoint.pt": file_sha256(resolved["int4_checkpoint"]),
                "qparams.csv": file_sha256(resolved["qparams_csv"]),
                "quantized_layers.json": file_sha256(resolved["quantized_layers_json"]),
                **{key: value for split_result in split_results.values() for key, value in dict(split_result.get("artifact_sha256", {})).items()},
            },
            "oer_protocol": oer_protocol_by_split,
            "oer_activation_position_used": first_oer_protocol.get("oer_activation_position_used"),
            "oer_hook_list_used": first_oer_protocol.get("oer_hook_list_used"),
            "oer_hook_list_used_hash": first_oer_protocol.get("oer_hook_list_used_hash"),
            "splits": split_results,            "metrics_csv": path_str(resolved["metrics_csv"]),
        }
    )
    payload = attach_run_manifest_hash(final_manifest)
    _write_json(resolved["manifest_path"], payload)
    write_run_manifests(
        {
            "qparams": resolved["qparams_csv"].parent,
            "seed": resolved["seed_dir"],
            **output_dirs,
        },
        payload,
    )
    print(f"[baseline-int4] qparams={resolved['qparams_csv']}")
    print(f"[baseline-int4] manifest={resolved['manifest_path']}")
    print(f"[baseline-int4] run_manifest_hash={payload['run_manifest_hash']}")
    return {"manifest": payload, "resolved": resolved, "splits": split_results}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_int4_student(args)


if __name__ == "__main__":
    main()
