from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline import quant_checkpoint as qck
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
    pooled_logits_from_features,
    prepare_encoder_input_for_export,
)
from src.baseline.fp32_teacher_staging import stage_fp32_teacher_checkpoint
from src.baseline.manifest import (
    attach_run_manifest_hash,
    baseline_common_manifest,
    canonical_json_hash,
    file_sha256,
    path_str,
    write_run_manifests,
)
from src.baseline.run_int4_student import BASELINE_NAME as INT4_BASELINE_NAME
from src.baseline.run_vanilla_kd import (
    _assert_fp32_teacher_payload,
    _assert_quant_training_state,
    _drop_legacy_hash_aliases,
    _load_required_json,
    _teacher_source_beats_checkpoint,
    _validate_int4_student_provenance,
    _validate_loaded_int4_checkpoint_payload,
    _validate_label_tensor,
    evaluate_student,
    vanilla_kd_loss,
)
from src.innovation.stage1_eval_metrics import (
    STAGE1_SELECTION_METRIC,
    build_stage1_eval_metric_row,
    write_stage1_eval_metrics,
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


PROTOCOL_VERSION = "innovation_two_stage_decoupling.v1"
INNOVATION_NAME = "two_stage_decoupling"
STAGE2_SELECTION_METRIC = "eval_macro_f1_label_universe_nanmean"
FORMAL_DEFAULTS = {
    "stage1_epochs": 30,
    "stage2_epochs": 10,
    "stage1_backbone_lr": 1e-5,
    "stage1_classifier_lr": 1e-3,
    "stage2_classifier_lr": 1e-3,
    "alpha": 0.5,
    "temperature": 4.0,
    "warmup_ratio": 0.05,
}


@dataclass(frozen=True)
class StageSpec:
    name: str
    index: int
    objective: str
    epochs: int
    backbone_lr: float
    classifier_lr: float
    kd_enabled: bool
    train_backbone: bool
    train_classifier: bool


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


def _default_innovation_root(dataset_root: str | Path | None = None) -> Path:
    if dataset_root is None:
        return Path(getattr(project_config, "BASE_RESULT_DIR", PROJECT_ROOT / "outputs")) / "innovation" / "two_stage"
    return Path(dataset_root) / "innovation" / "two_stage"


def _dataset_root(dataset_name: str | None, dataset_root: str | Path | None = None) -> Path:
    return Path(dataset_root) if dataset_root is not None else dataset_paths(dataset_name, None).root


def _canonical_baseline_root(dataset_name: str | None, dataset_root: str | Path | None = None) -> Path:
    return _dataset_root(dataset_name, dataset_root) / "baseline"


def _teacher_seed_dir(baseline_root: str | Path, seed: int) -> Path:
    return Path(baseline_root) / "teacher" / f"seed_{int(seed)}"


def _student_seed_dir(baseline_root: str | Path, seed: int) -> Path:
    return Path(baseline_root) / "student" / f"seed_{int(seed)}"


def _default_teacher_checkpoint(seed_dir: str | Path, seed: int) -> Path:
    return Path(seed_dir) / "checkpoints" / f"fp32_teacher_seed_{int(seed)}.pt"


def _default_int4_qparams_csv(seed_dir: str | Path) -> Path:
    return Path(seed_dir) / "qparams" / "int4_qparams.csv"


def _default_int4_manifest(seed_dir: str | Path) -> Path:
    return Path(seed_dir) / "manifests" / "int4_student" / "run_manifest.json"


def _default_int4_checkpoint(seed_dir: str | Path, seed: int) -> Path:
    return Path(seed_dir) / "checkpoints" / f"int4_student_calibrated_seed_{int(seed)}.pt"


def _seed_dir(args: argparse.Namespace, *, dataset_root: str | Path | None = None) -> Path:
    root = Path(args.innovation_root) if getattr(args, "innovation_root", None) else _default_innovation_root(dataset_root)
    return root / f"seed_{int(args.seed)}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Innovation runner: two-stage INT4 decoupling. Stage 1 repairs the semantic manifold "
            "with FP32 teacher KD; Stage 2 freezes the INT4 backbone and repairs only the classifier boundary."
        ),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--baseline-root",
        default=None,
        help="Canonical baseline root; default: <dataset_root>/baseline.",
    )
    parser.add_argument("--innovation-root", default=None, help="Default: <dataset_root>/innovation/two_stage.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--teacher-checkpoint", default=None, help="FP32 teacher source for staging; default: <dataset_root>/baseline/teacher/seed_<seed>/checkpoints/fp32_teacher_seed_<seed>.pt.")
    parser.add_argument("--force-stage-fp32-teacher", action="store_true")
    parser.add_argument("--qparams-csv", default=None, help="Default: <dataset_root>/baseline/student/seed_<seed>/qparams/int4_qparams.csv.")
    parser.add_argument("--int4-manifest", default=None, help="Default: <dataset_root>/baseline/student/seed_<seed>/manifests/int4_student/run_manifest.json.")
    parser.add_argument("--int4-checkpoint", default=None, help="Default: <dataset_root>/baseline/student/seed_<seed>/checkpoints/int4_student_calibrated_seed_<seed>.pt.")
    parser.add_argument("--stage1-epochs", type=_positive_int, default=int(FORMAL_DEFAULTS["stage1_epochs"]))
    parser.add_argument("--stage2-epochs", type=_positive_int, default=int(FORMAL_DEFAULTS["stage2_epochs"]))
    parser.add_argument("--stage1-backbone-lr", type=_positive_float, default=float(FORMAL_DEFAULTS["stage1_backbone_lr"]))
    parser.add_argument("--stage1-classifier-lr", type=_positive_float, default=float(FORMAL_DEFAULTS["stage1_classifier_lr"]))
    parser.add_argument("--stage2-classifier-lr", type=_positive_float, default=float(FORMAL_DEFAULTS["stage2_classifier_lr"]))
    parser.add_argument("--stage2-backbone-lr", type=_nonnegative_float, default=0.0, help="Must remain exactly 0.0 for the two-stage decoupling protocol.")
    parser.add_argument("--alpha", type=float, default=float(FORMAL_DEFAULTS["alpha"]))
    parser.add_argument("--temperature", type=_positive_float, default=float(FORMAL_DEFAULTS["temperature"]))
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=float(FORMAL_DEFAULTS["warmup_ratio"]))
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=_positive_int, default=8)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage1-checkpoint", default=None, help="Default: <seed_dir>/stage1_checkpoint.pt; saved from the best Stage 1 eval macro F1.")
    parser.add_argument("--stage1-last-checkpoint", "--stage1-last", dest="stage1_last_checkpoint", default=None, help="Default: <seed_dir>/stage1_last.pt; audit-only last Stage 1 checkpoint.")
    parser.add_argument("--resume-from-stage1-best", action="store_true", help="Skip Stage 1 training by validating and reusing the existing Stage 1 eval-best checkpoint, then continue with Stage 1 export and Stage 2.")
    parser.add_argument("--stage2-checkpoint", default=None, help="Default: <seed_dir>/stage2_checkpoint.pt; saved from the best Stage 2 eval macro F1.")
    parser.add_argument("--stage2-last-checkpoint", "--stage2-last", dest="stage2_last_checkpoint", default=None, help="Default: <seed_dir>/stage2_last.pt; audit-only last Stage 2 checkpoint.")
    parser.add_argument("--final-checkpoint", default=None, help="Default: <seed_dir>/final_checkpoint.pt.")
    parser.add_argument("--stage1-train-log", "--train-log", dest="stage1_train_log", default=None, help="Default: <seed_dir>/stage1_train_log.csv.")
    parser.add_argument("--stage1-eval-metrics", default=None, help="Default: <seed_dir>/stage1_eval_metrics.csv.")
    parser.add_argument("--stage1-manifest", default=None, help="Default: <seed_dir>/stage1_manifest.json.")
    parser.add_argument("--stage2-train-log", default=None, help="Default: <seed_dir>/stage2_train_log.csv.")
    parser.add_argument("--stage2-eval-metrics", default=None, help="Default: <seed_dir>/stage2_eval_metrics.csv.")
    parser.add_argument("--stage2-manifest", default=None, help="Default: <seed_dir>/stage2_manifest.json.")
    parser.add_argument("--stage-log", default=None, help="Default: <seed_dir>/stage_log.jsonl.")
    parser.add_argument("--manifest-path", default=None, help="Default: <seed_dir>/manifests/two_stage_decoupling/run_manifest.json.")
    parser.add_argument("--export-root", default=None, help="Default: <seed_dir>.")
    parser.add_argument("--export-name", default=INNOVATION_NAME)
    parser.add_argument("--reference-root", default=None, help="Seed export root containing fp32_teacher reference artifacts; default follows the selected FP32 teacher baseline seed directory.")
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT)
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS)
    parser.add_argument("--export-all-token-layers", action="store_true")
    parser.add_argument("--max-export-payload-gib", type=float, default=16.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _reject_amp(args: argparse.Namespace) -> None:
    if bool(getattr(args, "amp", False)):
        raise RuntimeError("Two-stage decoupling formal training forbids AMP; remove --amp and train with FP32 logits/loss.")


def _seed_everything(seed: int) -> None:
    seed_fn = getattr(project_config, "seed_everything", None)
    if callable(seed_fn):
        seed_fn(int(seed))
        return
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _resolved_paths(args: argparse.Namespace, *, dataset_root: str | Path | None = None) -> dict[str, Path]:
    root = _dataset_root(args.dataset, dataset_root)
    seed = int(args.seed)
    seed_dir = _seed_dir(args, dataset_root=root)
    canonical_root = Path(args.baseline_root) if args.baseline_root else _canonical_baseline_root(args.dataset, root)
    teacher_seed_dir = _teacher_seed_dir(canonical_root, seed)
    int4_student_seed_dir = _student_seed_dir(canonical_root, seed)
    teacher_checkpoint = _default_teacher_checkpoint(teacher_seed_dir, seed)
    reference_root = Path(args.reference_root) if args.reference_root else teacher_seed_dir
    return {
        "seed_dir": seed_dir,
        "teacher_checkpoint": teacher_checkpoint,
        "requested_teacher_checkpoint": Path(args.teacher_checkpoint) if args.teacher_checkpoint else teacher_checkpoint,
        "reference_root": reference_root,
        "qparams_csv": Path(args.qparams_csv) if args.qparams_csv else _default_int4_qparams_csv(int4_student_seed_dir),
        "int4_manifest": Path(args.int4_manifest) if args.int4_manifest else _default_int4_manifest(int4_student_seed_dir),
        "int4_checkpoint": Path(args.int4_checkpoint) if args.int4_checkpoint else _default_int4_checkpoint(int4_student_seed_dir, int(args.seed)),
        "stage1_checkpoint": Path(args.stage1_checkpoint) if args.stage1_checkpoint else seed_dir / "stage1_checkpoint.pt",
        "stage1_last_checkpoint": Path(args.stage1_last_checkpoint) if args.stage1_last_checkpoint else seed_dir / "stage1_last.pt",
        "stage2_checkpoint": Path(args.stage2_checkpoint) if args.stage2_checkpoint else seed_dir / "stage2_checkpoint.pt",
        "stage2_last_checkpoint": Path(args.stage2_last_checkpoint) if args.stage2_last_checkpoint else seed_dir / "stage2_last.pt",
        "final_checkpoint": Path(args.final_checkpoint) if args.final_checkpoint else seed_dir / "final_checkpoint.pt",
        "stage1_train_log": Path(args.stage1_train_log) if args.stage1_train_log else seed_dir / "stage1_train_log.csv",
        "stage1_eval_metrics": Path(args.stage1_eval_metrics) if args.stage1_eval_metrics else seed_dir / "stage1_eval_metrics.csv",
        "stage1_manifest": Path(args.stage1_manifest) if args.stage1_manifest else seed_dir / "stage1_manifest.json",
        "stage2_train_log": Path(args.stage2_train_log) if args.stage2_train_log else seed_dir / "stage2_train_log.csv",
        "stage2_eval_metrics": Path(args.stage2_eval_metrics) if args.stage2_eval_metrics else seed_dir / "stage2_eval_metrics.csv",
        "stage2_manifest": Path(args.stage2_manifest) if args.stage2_manifest else seed_dir / "stage2_manifest.json",
        "stage_log": Path(args.stage_log) if args.stage_log else seed_dir / "stage_log.jsonl",
        "manifest_path": Path(args.manifest_path) if args.manifest_path else seed_dir / "manifests" / INNOVATION_NAME / "run_manifest.json",
        "export_root": Path(args.export_root) if args.export_root else seed_dir,
        "metrics_csv": seed_dir / f"{INNOVATION_NAME}_metrics.csv",
    }


def _stage_specs(args: argparse.Namespace) -> tuple[StageSpec, StageSpec]:
    return (
        StageSpec(
            name="stage1_semantic_manifold",
            index=1,
            objective="fit_fp32_teacher_semantic_manifold",
            epochs=int(args.stage1_epochs),
            backbone_lr=float(args.stage1_backbone_lr),
            classifier_lr=float(args.stage1_classifier_lr),
            kd_enabled=True,
            train_backbone=True,
            train_classifier=True,
        ),
        StageSpec(
            name="stage2_decision_boundary",
            index=2,
            objective="reshape_decision_boundary_with_clean_cross_entropy",
            epochs=int(args.stage2_epochs),
            backbone_lr=float(args.stage2_backbone_lr),
            classifier_lr=float(args.stage2_classifier_lr),
            kd_enabled=False,
            train_backbone=False,
            train_classifier=True,
        ),
    )


def _is_classifier_parameter(name: str) -> bool:
    return str(name).startswith("predictor.")


def _split_named_parameters(model: torch.nn.Module) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    backbone: list[tuple[str, torch.nn.Parameter]] = []
    classifier: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        if _is_classifier_parameter(name):
            classifier.append((name, parameter))
        else:
            backbone.append((name, parameter))
    if not classifier:
        raise RuntimeError("student model has no predictor.* classifier parameters")
    if not backbone:
        raise RuntimeError("student model has no backbone parameters")
    return backbone, classifier


def _freeze_backbone(model: torch.nn.Module) -> None:
    backbone_params, _classifier_params = _split_named_parameters(model)
    for _name, parameter in backbone_params:
        parameter.requires_grad = False


def _unfreeze_backbone(model: torch.nn.Module) -> None:
    backbone_params, _classifier_params = _split_named_parameters(model)
    for _name, parameter in backbone_params:
        parameter.requires_grad = True


def _freeze_classifier(model: torch.nn.Module) -> None:
    _backbone_params, classifier_params = _split_named_parameters(model)
    for _name, parameter in classifier_params:
        parameter.requires_grad = False


def _unfreeze_classifier(model: torch.nn.Module) -> None:
    _backbone_params, classifier_params = _split_named_parameters(model)
    for _name, parameter in classifier_params:
        parameter.requires_grad = True


def _set_requires_grad(model: torch.nn.Module, *, backbone: bool, classifier: bool) -> None:
    if bool(backbone):
        _unfreeze_backbone(model)
    else:
        _freeze_backbone(model)
    if bool(classifier):
        _unfreeze_classifier(model)
    else:
        _freeze_classifier(model)


def _parameter_status(model: torch.nn.Module) -> dict[str, Any]:
    backbone_params, classifier_params = _split_named_parameters(model)

    def status(named_params: Sequence[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
        trainable = [(name, parameter) for name, parameter in named_params if parameter.requires_grad]
        frozen = [(name, parameter) for name, parameter in named_params if not parameter.requires_grad]
        return {
            "num_params": int(sum(parameter.numel() for _name, parameter in named_params)),
            "num_tensors": int(len(named_params)),
            "trainable_num_params": int(sum(parameter.numel() for _name, parameter in trainable)),
            "trainable_num_tensors": int(len(trainable)),
            "frozen_num_tensors": int(len(frozen)),
            "all_requires_grad": bool(trainable and len(trainable) == len(named_params)),
            "none_requires_grad": bool(len(trainable) == 0),
        }

    return {
        "backbone": status(backbone_params),
        "classifier": status(classifier_params),
    }


def _make_stage_optimizer(
    model: torch.nn.Module,
    stage: StageSpec,
    *,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params, classifier_params = _split_named_parameters(model)
    groups: list[dict[str, Any]] = []
    backbone_trainable = [parameter for _name, parameter in backbone_params if parameter.requires_grad]
    classifier_trainable = [parameter for _name, parameter in classifier_params if parameter.requires_grad]
    if int(stage.index) == 1:
        groups.append(
            {
                "params": backbone_trainable,
                "lr": float(stage.backbone_lr),
                "name": "backbone",
            }
        )
    elif int(stage.index) != 2:
        raise ValueError(f"unknown stage index: {stage.index}")
    groups.append(
        {
            "params": classifier_trainable,
            "lr": float(stage.classifier_lr),
            "name": "classifier",
        }
    )
    if not groups or not any(parameter.requires_grad for group in groups for parameter in group["params"]):
        raise RuntimeError(f"{stage.name} produced no trainable optimizer parameters")
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


def _optimizer_lr(optimizer: torch.optim.Optimizer, group_name: str) -> float | None:
    for group in optimizer.param_groups:
        if str(group.get("name")) == str(group_name):
            return float(group.get("lr", 0.0))
    return None


def _optimizer_group_status(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for group in optimizer.param_groups:
        group_name = str(group.get("name", "unnamed"))
        params = list(group.get("params") or [])
        trainable = [parameter for parameter in params if parameter.requires_grad]
        status[group_name] = {
            "lr": float(group.get("lr", 0.0)),
            "num_params": int(sum(parameter.numel() for parameter in params)),
            "num_tensors": int(len(params)),
            "trainable_num_params": int(sum(parameter.numel() for parameter in trainable)),
            "trainable_num_tensors": int(len(trainable)),
        }
    return status


def _assert_stage_contract(model: torch.nn.Module, optimizer: torch.optim.Optimizer, stage: StageSpec) -> dict[str, Any]:
    status = _parameter_status(model)
    optimizer_status = _optimizer_group_status(optimizer)
    optimizer_group_names = [str(group.get("name", "unnamed")) for group in optimizer.param_groups]
    backbone_optimizer_group_exists = any("backbone" in name for name in optimizer_group_names)
    backbone_lr = _optimizer_lr(optimizer, "backbone") if int(stage.index) == 1 else None
    classifier_lr = _optimizer_lr(optimizer, "classifier")
    backbone_optimizer_status = optimizer_status.get("backbone", {})
    classifier_optimizer_status = optimizer_status.get("classifier", {})
    if classifier_lr is None or classifier_lr <= 0.0:
        raise RuntimeError(f"{stage.name}: classifier lr must be > 0, got {classifier_lr!r}")
    if int(classifier_optimizer_status.get("trainable_num_params", 0)) <= 0:
        raise RuntimeError(f"{stage.name}: classifier optimizer group must contain trainable parameters")

    if stage.index == 1:
        if not stage.kd_enabled:
            raise RuntimeError("Stage 1 contract violation: kd_loss must be enabled")
        if backbone_lr is None or backbone_lr <= 0.0:
            raise RuntimeError(f"Stage 1 contract violation: backbone_lr must be > 0, got {backbone_lr!r}")
        if int(backbone_optimizer_status.get("trainable_num_params", 0)) <= 0:
            raise RuntimeError("Stage 1 contract violation: backbone optimizer group must contain trainable parameters")
        if not bool(status["backbone"]["all_requires_grad"]):
            raise RuntimeError("Stage 1 contract violation: backbone requires_grad must be True")
        if not bool(status["classifier"]["all_requires_grad"]):
            raise RuntimeError("Stage 1 contract violation: classifier requires_grad must be True")
    elif stage.index == 2:
        if stage.kd_enabled:
            raise RuntimeError("Stage 2 contract violation: kd_loss must be disabled")
        if float(stage.backbone_lr) != 0.0:
            raise RuntimeError(f"Stage 2 contract violation: protocol backbone_lr must be exactly 0, got {stage.backbone_lr!r}")
        if backbone_optimizer_group_exists:
            raise RuntimeError("Stage 2 contract violation: backbone must not appear in optimizer param group names")
        if not bool(status["backbone"]["none_requires_grad"]):
            raise RuntimeError("Stage 2 contract violation: backbone requires_grad must be False")
        if not bool(status["classifier"]["all_requires_grad"]):
            raise RuntimeError("Stage 2 contract violation: classifier requires_grad must be True")
        backbone_lr = 0.0
    else:
        raise ValueError(f"unknown stage index: {stage.index}")

    return {
        "stage": stage.name,
        "stage_index": int(stage.index),
        "backbone_lr": backbone_lr,
        "classifier_lr": classifier_lr,
        "kd_loss_enabled": bool(stage.kd_enabled),
        "parameter_status": status,
        "optimizer_status": optimizer_status,
        "optimizer_group_names": optimizer_group_names,
        "backbone_optimizer_group_exists": bool(backbone_optimizer_group_exists),
    }


def _configure_stage(model: torch.nn.Module, stage: StageSpec, *, weight_decay: float) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    _set_requires_grad(model, backbone=bool(stage.train_backbone), classifier=bool(stage.train_classifier))
    optimizer = _make_stage_optimizer(model, stage, weight_decay=weight_decay)
    contract = _assert_stage_contract(model, optimizer, stage)
    return optimizer, contract


def forward_pooled_logits(model: torch.nn.Module, batch: Mapping[str, Any]) -> Tensor:
    encoder_input, encoder_padding_mask, token_padding_mask = prepare_encoder_input_for_export(
        model,
        batch["waveform"],
        batch.get("padding_mask"),
    )
    features, _layer_results = model.encoder(encoder_input, padding_mask=encoder_padding_mask)
    return pooled_logits_from_features(model, features, token_padding_mask)


def train_stage_one_epoch(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    alpha: float,
    temperature: float,
    num_classes: int,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_kd = 0.0
    total_correct = 0
    total_samples = 0
    teacher_forward_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="two_stage.stage1.label_id")
        optimizer.zero_grad(set_to_none=True)
        student_logits = forward_pooled_logits(student, batch)
        with torch.no_grad():
            teacher_logits = forward_pooled_logits(teacher, batch)
        teacher_forward_count += 1
        loss, parts = vanilla_kd_loss(student_logits, teacher_logits, labels, alpha=alpha, temperature=temperature)
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
    if total_samples <= 0:
        raise RuntimeError("Stage 1 train loader produced no samples")
    if teacher_forward_count <= 0:
        raise RuntimeError("Stage 1 contract violation: teacher distillation forward was never executed")
    return {
        "train_loss": total_loss / float(total_samples),
        "train_ce_loss": total_ce / float(total_samples),
        "train_kl_loss": total_kl / float(total_samples),
        "train_kd_term": total_kd / float(total_samples),
        "train_accuracy": total_correct / float(total_samples),
        "train_num_samples": float(total_samples),
        "teacher_forward_count": float(teacher_forward_count),
    }


def train_stage_two_epoch(
    *,
    student: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    student.eval()
    if hasattr(student, "predictor"):
        student.predictor.train()
    if hasattr(student, "predictor_dropout"):
        student.predictor_dropout.train()
    total_loss = 0.0
    total_ce = 0.0
    total_correct = 0
    total_samples = 0
    teacher_forward_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = _validate_label_tensor(batch["label_id"], num_classes=num_classes, context="two_stage.stage2.label_id")
        optimizer.zero_grad(set_to_none=True)
        student_logits = forward_pooled_logits(student, batch)
        loss_logits = student_logits.float()
        loss = F.cross_entropy(loss_logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        predictions = loss_logits.detach().argmax(dim=1)
        batch_size = int(labels.numel())
        total_samples += batch_size
        total_correct += int((predictions == labels).sum().item())
        total_loss += float(loss.detach().item()) * batch_size
        total_ce += float(loss.detach().item()) * batch_size
    if total_samples <= 0:
        raise RuntimeError("Stage 2 train loader produced no samples")
    if teacher_forward_count != 0:
        raise RuntimeError("Stage 2 contract violation: teacher distillation forward was executed")
    return {
        "train_loss": total_loss / float(total_samples),
        "train_ce_loss": total_ce / float(total_samples),
        "train_kl_loss": 0.0,
        "train_kd_term": 0.0,
        "train_accuracy": total_correct / float(total_samples),
        "train_num_samples": float(total_samples),
        "teacher_forward_count": float(teacher_forward_count),
    }


def _write_train_log(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_stage_log(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, default=str)
        handle.write("\n")


def _write_combined_metrics(split_results: Mapping[str, Any], path: Path, *, seed: int) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for split, result in split_results.items():
        metrics_path = Path(result["paths"]["metrics"])
        with metrics_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in ["baseline", "seed", "model_kind", *reader.fieldnames]:
                    if name not in fieldnames:
                        fieldnames.append(name)
            for row in reader:
                out = dict(row)
                out.setdefault("split", str(split))
                out["baseline"] = INNOVATION_NAME
                out["model_kind"] = INNOVATION_NAME
                out["seed"] = int(seed)
                rows.append(out)
    if not rows:
        raise RuntimeError("no two-stage decoupling metrics rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


_EXPORT_METRIC_TABLE_KEYS = ("metrics", "group_metrics", "per_class_metrics", "geometry_metrics")


def _rewrite_metric_table_model_kind(path: str | Path, *, model_kind: str) -> None:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"two-stage export metric table missing during model_kind rewrite: {target}")
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError(f"two-stage export metric table is empty during model_kind rewrite: {target}")
    if "model_kind" not in fieldnames:
        fieldnames.insert(0, "model_kind")
    for row in rows:
        row["model_kind"] = str(model_kind)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _rewrite_export_metric_model_kind(split_result: dict[str, Any], *, model_kind: str) -> None:
    paths = split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("two-stage export split result missing paths for model_kind rewrite")
    artifact_sha256 = dict(split_result.get("artifact_sha256") or {})
    for key in _EXPORT_METRIC_TABLE_KEYS:
        value = paths.get(key)
        if value is None or not str(value).strip():
            continue
        target = Path(value)
        _rewrite_metric_table_model_kind(target, model_kind=model_kind)
        artifact_sha256[str(target.resolve(strict=False))] = file_sha256(target)
    split_result["artifact_sha256"] = artifact_sha256


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


STAGE2_EVAL_METRIC_COLUMNS = [
    "seed",
    "stage",
    "stage2_epoch",
    "global_epoch",
    "stage2_checkpoint",
    "stage2_checkpoint_sha256",
    "stage2_selection_metric",
    "stage2_selection_metric_value",
    "stage2_train_ce_loss",
    "stage2_train_total_loss",
    "stage2_train_kd_loss",
    "stage2_teacher_kd_enabled",
    "stage2_eval_accuracy",
    "stage2_eval_macro_f1_label_universe_nanmean",
    "stage2_eval_many_accuracy",
    "stage2_eval_medium_accuracy",
    "stage2_eval_few_accuracy",
    "stage2_eval_tail_accuracy",
    "stage2_backbone_lr",
    "stage2_backbone_optimizer_group_exists",
    "stage2_classifier_lr",
    "stage2_backbone_trainable_param_count",
    "stage2_classifier_trainable_param_count",
    "stage2_reference_checkpoint_hash",
    "stage2_export_run_identity_hash",
]


_STAGE2_EXPORT_METRIC_SPECS = {
    "stage2_eval_accuracy": ("metrics", "accuracy", "global"),
    "stage2_eval_macro_f1_label_universe_nanmean": ("metrics", "macro_f1_label_universe_nanmean", "global"),
    "stage2_eval_many_accuracy": ("group_metrics", "accuracy", "many"),
    "stage2_eval_medium_accuracy": ("group_metrics", "accuracy", "medium"),
    "stage2_eval_few_accuracy": ("group_metrics", "accuracy", "few"),
    "stage2_eval_tail_accuracy": ("group_metrics", "accuracy", "tail"),
}


def _read_export_metric_table(path: str | Path, *, table_name: str) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Stage 2 metric table is missing: {table_name}={target}")
    with target.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise RuntimeError(f"Stage 2 metric table is empty: {table_name}={target}")
    return rows


def _as_finite_or_nan(value: Any, *, name: str) -> float:
    parsed = float(value)
    if math.isinf(parsed):
        raise ValueError(f"{name} must be finite or NaN, got {value!r}")
    return parsed


def _export_metric_value(tables: Mapping[str, Sequence[Mapping[str, Any]]], *, table: str, metric_name: str, analysis_group: str) -> float:
    rows = tables.get(table)
    if rows is None:
        raise KeyError(f"Stage 2 export result is missing table {table!r}")
    matches = [
        row
        for row in rows
        if str(row.get("metric_name")) == str(metric_name)
        and str(row.get("analysis_group")) == str(analysis_group)
    ]
    if not matches:
        raise KeyError(
            "Stage 2 metric is missing from export_eval output: "
            f"table={table}, metric_name={metric_name}, analysis_group={analysis_group}"
        )
    return _as_finite_or_nan(matches[0].get("metric_value"), name=f"{table}:{metric_name}:{analysis_group}")


def _stage_trainable_param_count(stage_contract: Mapping[str, Any], group_name: str) -> int:
    status = stage_contract.get("parameter_status")
    if not isinstance(status, Mapping):
        raise KeyError("Stage contract missing parameter_status")
    group = status.get(group_name)
    if not isinstance(group, Mapping):
        raise KeyError(f"Stage contract missing parameter_status.{group_name}")
    return int(group.get("trainable_num_params", 0))


def build_stage2_clean_eval_metric_row(
    *,
    split_result: Mapping[str, Any],
    best_train_row: Mapping[str, Any],
    stage_contract: Mapping[str, Any],
    seed: int,
    stage_epoch: int,
    global_epoch: int,
    checkpoint_path: str | Path,
    checkpoint_sha256: str | None,
) -> dict[str, Any]:
    paths = split_result.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("Stage 2 export result missing paths")
    tables = {
        "metrics": _read_export_metric_table(paths["metrics"], table_name="metrics"),
        "group_metrics": _read_export_metric_table(paths["group_metrics"], table_name="group_metrics"),
    }
    row: dict[str, Any] = {
        "seed": int(seed),
        "stage": "stage2_decision_boundary",
        "stage2_epoch": int(stage_epoch),
        "global_epoch": int(global_epoch),
        "stage2_checkpoint": str(Path(checkpoint_path).resolve(strict=False)),
        "stage2_checkpoint_sha256": "" if checkpoint_sha256 is None else str(checkpoint_sha256),
        "stage2_selection_metric": STAGE2_SELECTION_METRIC,
        "stage2_train_ce_loss": _as_finite_or_nan(best_train_row.get("train_ce_loss"), name="stage2 train_ce_loss"),
        "stage2_train_total_loss": _as_finite_or_nan(best_train_row.get("train_loss"), name="stage2 train_loss"),
        "stage2_train_kd_loss": 0.0,
        "stage2_teacher_kd_enabled": False,
        "stage2_backbone_lr": 0.0,
        "stage2_backbone_optimizer_group_exists": bool(stage_contract.get("backbone_optimizer_group_exists", False)),
        "stage2_classifier_lr": _as_finite_or_nan(stage_contract.get("classifier_lr"), name="stage2 classifier_lr"),
        "stage2_backbone_trainable_param_count": 0,
        "stage2_classifier_trainable_param_count": _stage_trainable_param_count(stage_contract, "classifier"),
        "stage2_reference_checkpoint_hash": str(split_result.get("reference_checkpoint_hash") or ""),
        "stage2_export_run_identity_hash": str(split_result.get("run_identity_hash") or ""),
    }
    for output_name, (table_name, metric_name, analysis_group) in _STAGE2_EXPORT_METRIC_SPECS.items():
        row[output_name] = _export_metric_value(tables, table=table_name, metric_name=metric_name, analysis_group=analysis_group)
    row["stage2_selection_metric_value"] = row["stage2_eval_macro_f1_label_universe_nanmean"]

    if row["stage2_backbone_lr"] != 0.0:
        raise RuntimeError(f"Stage 2 audit violation: backbone lr must be 0, got {row['stage2_backbone_lr']!r}")
    if bool(row["stage2_backbone_optimizer_group_exists"]):
        raise RuntimeError("Stage 2 audit violation: backbone optimizer group must be absent")
    if row["stage2_backbone_trainable_param_count"] != 0:
        raise RuntimeError("Stage 2 audit violation: backbone trainable param count must be 0")
    if bool(row["stage2_teacher_kd_enabled"]):
        raise RuntimeError("Stage 2 audit violation: KD must be disabled")
    if not row["stage2_reference_checkpoint_hash"]:
        raise RuntimeError("Stage 2 FP32 reference metrics were requested, but export_eval did not load a reference bundle")
    return row


def write_stage2_eval_metrics(path: str | Path, row: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(STAGE2_EVAL_METRIC_COLUMNS)
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(str(key))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _stage_protocol_payload(args: argparse.Namespace, stage_contracts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "strategy": "first repair features, then repair boundary",
        "stage_order": ["stage1_semantic_manifold", "stage2_decision_boundary"],
        "stage1_forbidden_modules_disabled": {
            "class_prior": True,
            "logit_adjustment": True,
            "tau_schedule": True,
            "tail_reweighting": True,
        },
        "stage1": {
            "objective": "fit the FP32 teacher semantic manifold",
            "student": "INT4 calibrated student",
            "train_backbone": True,
            "train_classifier": True,
            "kd_loss_enabled": True,
            "loss": "CrossEntropyLoss + alpha * T^2 * KL(student, fp32_teacher)",
            "alpha": float(args.alpha),
            "temperature": float(args.temperature),
            "backbone_lr": float(args.stage1_backbone_lr),
            "classifier_lr": float(args.stage1_classifier_lr),
            "epochs": int(args.stage1_epochs),
        },
        "stage2": {
            "objective": "reshape the classifier decision boundary",
            "student": "Stage 1 trained INT4 student",
            "train_backbone": False,
            "train_classifier": True,
            "kd_loss_enabled": False,
            "loss": "CrossEntropyLoss on raw student logits",
            "backbone_lr": float(args.stage2_backbone_lr),
            "classifier_lr": float(args.stage2_classifier_lr),
            "epochs": int(args.stage2_epochs),
            "class_correction": False,
            "class_prior": False,
            "long_tail_reweight": False,
        },
        "optimizer": "AdamW",
        "scheduler": "warmup + cosine annealing per stage",
        "warmup_ratio": float(args.warmup_ratio),
        "weight_decay": float(args.weight_decay),
        "sampler": "normal shuffle train loader",
        "amp": False,
        "training_dtype": "float32",
        "loss_logits_dtype": "float32",
        "test_usage_discipline": "test is exported once for final reporting only; no checkpoint, hyperparameter, temperature, alpha, or stage selection",
        "stage_contracts": [dict(item) for item in stage_contracts],
    }


def _checkpoint_identity_payload(
    *,
    model_kind: str,
    cfg: Mapping[str, Any],
    label_mapping: Mapping[str, int],
    paths: Any,
    source_beats_checkpoint: Any,
    teacher_checkpoint: Path,
    layers: Sequence[Mapping[str, Any]],
    qparams: Mapping[str, Any],
    qparams_csv: Path,
    calibration_csv: str | Path | None,
    calibration_hash: str | None,
    args: argparse.Namespace,
    stage_name: str,
    stage_index: int,
    stage_protocol: Mapping[str, Any],
    int4_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    identity_payload = qck.checkpoint_identity_payload(
        model_kind=model_kind,
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
        seed=int(args.seed),
    )
    identity_payload["innovation_protocol_version"] = PROTOCOL_VERSION
    identity_payload["innovation_name"] = INNOVATION_NAME
    identity_payload["stage_name"] = str(stage_name)
    identity_payload["stage_index"] = int(stage_index)
    identity_payload["stage_protocol"] = dict(stage_protocol)
    identity_payload["int4_student_manifest_sha256"] = int4_provenance.get("manifest_sha256")
    identity_payload["int4_student_run_identity_hash"] = int4_provenance.get("manifest_run_identity_hash")
    identity_payload["int4_student_quantized_layers_hash"] = int4_provenance.get("quantized_layers_hash")
    return identity_payload


def _save_stage_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    stage: StageSpec,
    global_epoch: int,
    stage_epoch: int,
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
    quant_audit: Mapping[str, Any] | None,
    stage_protocol: Mapping[str, Any],
    best_metric_so_far: float,
    model_state_dict_override: Mapping[str, Tensor] | None = None,
) -> dict[str, Any]:
    _assert_fp32_teacher_payload(teacher_payload, context="two-stage checkpoint save teacher payload")
    cfg = dict(teacher_payload.get("cfg") or {})
    if not cfg:
        raise KeyError("FP32 teacher payload missing cfg; cannot create reproducible two-stage checkpoint")

    qparams = qck.load_qparams_csv(qparams_csv)
    qparams_hash_value = qck.qparams_hash(qparams)
    expected_qparams_hash = str(int4_provenance.get("qparams_hash") or "")
    if qparams_hash_value != expected_qparams_hash:
        raise RuntimeError(
            "two-stage qparams semantic hash differs from INT4 provenance before checkpoint save: "
            f"current={qparams_hash_value}, int4={expected_qparams_hash}"
        )
    qparams_csv_sha256 = file_sha256(qparams_csv)
    if str(qparams_csv_sha256) != str(int4_provenance.get("qparams_csv_sha256")):
        raise RuntimeError(
            "two-stage qparams CSV artifact hash differs from INT4 provenance before checkpoint save: "
            f"current={qparams_csv_sha256}, int4={int4_provenance.get('qparams_csv_sha256')}"
        )
    controller_qparams_hash = qck.qparams_hash(qck.collect_qparams_from_controller(controller))
    if controller_qparams_hash != qparams_hash_value:
        raise RuntimeError(
            "controller qparams and qparams_csv disagree before saving two-stage checkpoint: "
            f"controller={controller_qparams_hash}, csv={qparams_hash_value}"
        )

    layers_payload = [dict(row) for row in layers]
    quantized_layers_hash = canonical_json_hash(layers_payload)
    if str(quantized_layers_hash) != str(int4_provenance.get("quantized_layers_hash")):
        raise RuntimeError(
            "two-stage quantized_layers hash differs from INT4 provenance before checkpoint save: "
            f"current={quantized_layers_hash}, int4={int4_provenance.get('quantized_layers_hash')}"
        )
    qparams_metadata = qck.qparams_csv_metadata(qparams_csv)
    metadata_calibration_hash = qparams_metadata.get("calibration_hash")
    calibration_csv_hash = file_sha256(calibration_csv)
    if calibration_csv_hash and str(calibration_csv_hash) != str(int4_provenance.get("calibration_csv_hash")):
        raise RuntimeError(
            "two-stage calibration CSV hash differs from INT4 provenance before checkpoint save: "
            f"current={calibration_csv_hash}, int4={int4_provenance.get('calibration_csv_hash')}"
        )
    if calibration_csv_hash and metadata_calibration_hash and str(metadata_calibration_hash) != str(calibration_csv_hash):
        raise RuntimeError(
            "qparams_csv calibration hash does not match two-stage checkpoint calibration CSV identity: "
            f"csv={metadata_calibration_hash}, calibration_csv={calibration_csv_hash}"
        )
    resolved_calibration_hash = str(calibration_hash or metadata_calibration_hash or calibration_csv_hash or "") or None
    if resolved_calibration_hash and str(resolved_calibration_hash) != str(int4_provenance.get("calibration_hash")):
        raise RuntimeError(
            "two-stage calibration_hash differs from INT4 provenance before checkpoint save: "
            f"current={resolved_calibration_hash}, int4={int4_provenance.get('calibration_hash')}"
        )

    first_layer = dict(layers_payload[0]) if layers_payload else {}
    source_beats_checkpoint = _teacher_source_beats_checkpoint(teacher_payload, context="two-stage teacher")
    int4_provenance_payload = {
        "manifest_path": str(int4_provenance.get("manifest_path") or ""),
        "manifest_sha256": int4_provenance.get("manifest_sha256"),
        "run_manifest_hash": int4_provenance.get("manifest_hash"),
        "manifest_run_identity_hash": int4_provenance.get("manifest_run_identity_hash"),
        "calibration_csv": path_str(int4_provenance.get("calibration_csv")),
        "calibration_csv_hash": int4_provenance.get("calibration_csv_hash"),
        "calibration_manifest_hash": int4_provenance.get("calibration_manifest_hash"),
        "calibration_hash": int4_provenance.get("calibration_hash"),
        "qparams_hash": int4_provenance.get("qparams_hash"),
        "qparams_csv_sha256": int4_provenance.get("qparams_csv_sha256"),
        "quantized_layers_hash": int4_provenance.get("quantized_layers_hash"),
        "quantized_layers": layers_payload,
    }
    model_kind = INNOVATION_NAME
    identity_payload = _drop_legacy_hash_aliases(_checkpoint_identity_payload(
        model_kind=model_kind,
        cfg=cfg,
        label_mapping=label_mapping,
        paths=paths,
        source_beats_checkpoint=source_beats_checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        layers=layers_payload,
        qparams=qparams,
        qparams_csv=qparams_csv,
        calibration_csv=calibration_csv,
        calibration_hash=resolved_calibration_hash,
        args=args,
        stage_name=stage.name,
        stage_index=stage.index,
        stage_protocol=stage_protocol,
        int4_provenance=int4_provenance,
    ))
    run_identity_hash = canonical_json_hash(identity_payload)
    source_beats_checkpoint_sha256 = identity_payload.get("source_beats_checkpoint_sha256")
    teacher_checkpoint_sha256 = file_sha256(teacher_checkpoint)
    audio_input_py_sha256 = identity_payload.get("audio_input_py_sha256") or identity_payload.get("audio_input_hash")
    group_utils_py_sha256 = identity_payload.get("group_utils_py_sha256") or identity_payload.get("group_utils_hash")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "artifact_type": "innovation_two_stage_decoupling_quantized_student_checkpoint",
        "model_kind": model_kind,
        "innovation_name": INNOVATION_NAME,
        "stage_name": stage.name,
        "stage_index": int(stage.index),
        "stage_objective": stage.objective,
        "epoch": int(global_epoch),
        "stage_epoch": int(stage_epoch),
        "selection_metric": "eval_macro_f1_label_universe_nanmean",
        "best_metric_so_far": float(best_metric_so_far),
        "eval_metrics": dict(eval_metrics),
        "cfg": cfg,
        "model_state_dict": dict(model_state_dict_override) if model_state_dict_override is not None else model.state_dict(),
        "label_mapping": dict(label_mapping),
        "label_mapping_hash": file_sha256(paths.label_mapping),
        "source_beats_checkpoint": path_str(source_beats_checkpoint),
        "source_beats_checkpoint_sha256": source_beats_checkpoint_sha256,
        "teacher_checkpoint": path_str(teacher_checkpoint),
        "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
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
        "calibration_hash": resolved_calibration_hash,
        "audio_input_py_sha256": audio_input_py_sha256,
        "audio_input_hash": audio_input_py_sha256,
        "group_utils_py_sha256": group_utils_py_sha256,
        "group_utils_hash": group_utils_py_sha256,
        "pooling": DEFAULT_POOLING,
        "seed": int(args.seed),
        "git_commit": identity_payload.get("git_commit"),
        "run_identity_hash": run_identity_hash,
        "run_identity_payload": identity_payload,
        "kd_config": dict(stage_protocol),
        "two_stage_protocol": dict(stage_protocol),
        "teacher_frozen": True,
        "qparams_frozen": True,
        "student_backbone_trainable": bool(stage.train_backbone),
        "controller_state_summary": controller.state_summary(),
        "training_quant_audit": dict(quant_audit or {}),
        "load_note": "Load through src.baseline.export_eval.load_trained_checkpoint; it injects INT4, applies qparams, freezes qparams, disables observer, enables fake quant, strict-loads state_dict, and audits quant state.",
    }
    for parameter in model.parameters():
        parameter.grad = None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _run_export_eval(
    *,
    args: argparse.Namespace,
    resolved: Mapping[str, Path],
    checkpoint_path: Path,
    calibration_hash: str | None,
    label_buckets: Mapping[int, str],
    device: torch.device,
    splits: Sequence[str] = ("eval", "test"),
    export_root: Path | None = None,
    export_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Sequence[int], str, str, Mapping[str, Any] | None]:
    export_model, export_checkpoint = load_trained_checkpoint(checkpoint_path, device=device)
    export_paths = dataset_paths(args.dataset, str(checkpoint_path))
    export_name_value = str(export_name or args.export_name)
    export_root_value = Path(export_root) if export_root is not None else Path(resolved["export_root"])
    export_args_base = argparse.Namespace(**vars(args))
    export_args_base.export_name = export_name_value
    export_args_base.export_root = str(export_root_value)
    token_layer_indices = _resolve_export_token_layers(export_model, export_args_base)
    token_export = _token_export_mode_from_layers(export_model, token_layer_indices)
    export_run_identity_hash = build_run_identity(
        model_name=export_name_value,
        checkpoint_path=checkpoint_path,
        checkpoint=export_checkpoint,
        split_hashes=_split_hash_payload(paths=export_paths),
        label_mapping_hash=file_sha256(export_paths.label_mapping) or export_checkpoint.get("label_mapping_hash"),
        calibration_hash=export_checkpoint.get("calibration_hash") or calibration_hash,
        pooling=DEFAULT_POOLING,
        token_export=token_export,
        token_layers=token_layer_indices,
        seed=int(args.seed),
    )
    export_args = _formal_export_args(export_args_base, run_identity_hash=export_run_identity_hash, token_export=token_export)
    export_args.reference_root = str(resolved["reference_root"])
    output_dirs = _export_dirs(export_root_value, export_name_value)
    split_results: dict[str, Any] = {}
    for split in splits:
        split_result = export_eval_split(
            model=export_model,
            paths=export_paths,
            split=split,
            args=export_args,
            label_buckets=label_buckets,
            token_layer_indices=token_layer_indices,
            output_dirs=output_dirs,
            device=device,
            checkpoint_path=checkpoint_path,
            quant_controller=getattr(export_model, "_baseline_quant_controller", None),
            quant_audit=getattr(export_model, "_baseline_quant_audit", None),
            run_identity_hash=export_run_identity_hash,
        )
        _rewrite_export_metric_model_kind(split_result, model_kind=INNOVATION_NAME)
        split_results[split] = split_result
    return (
        split_results,
        export_checkpoint,
        token_layer_indices,
        token_export,
        export_run_identity_hash,
        getattr(export_model, "_baseline_quant_audit", None),
    )


def _require_loaded_checkpoint_identity(
    *,
    context: str,
    loaded_payload: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
    expected_stage_index: int,
    expected_stage_epoch: int,
    expected_global_epoch: int,
) -> None:
    expected_run_identity_hash = expected_payload.get("run_identity_hash")
    loaded_run_identity_hash = loaded_payload.get("run_identity_hash")
    if not expected_run_identity_hash or loaded_run_identity_hash != expected_run_identity_hash:
        raise RuntimeError(
            f"{context} checkpoint run_identity_hash mismatch: "
            f"loaded={loaded_run_identity_hash!r}, expected={expected_run_identity_hash!r}"
        )
    if loaded_payload.get("model_kind") != INNOVATION_NAME:
        raise RuntimeError(f"{context} checkpoint model_kind mismatch: loaded={loaded_payload.get('model_kind')!r}, expected={INNOVATION_NAME!r}")
    numeric_expectations = {
        "stage_index": int(expected_stage_index),
        "stage_epoch": int(expected_stage_epoch),
        "epoch": int(expected_global_epoch),
    }
    for field, expected in numeric_expectations.items():
        loaded = loaded_payload.get(field)
        try:
            loaded_int = int(loaded)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{context} checkpoint {field} is not an integer: {loaded!r}") from exc
        if loaded_int != expected:
            raise RuntimeError(f"{context} checkpoint {field} mismatch: loaded={loaded_int!r}, expected={expected!r}")


def _payload_int(payload: Mapping[str, Any], field: str, *, context: str) -> int:
    value = payload.get(field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} {field} is not an integer: {value!r}") from exc


def _require_payload_equal(context: str, field: str, existing: Any, expected: Any) -> None:
    if existing != expected:
        raise RuntimeError(f"{context} mismatch for {field}: existing={existing!r}, expected={expected!r}")


def _read_train_log_rows(path: Path, *, stage_index: int, context: str) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{context} requires existing Stage {stage_index} train log: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader if str(row.get("stage_index")) == str(int(stage_index))]
    if not rows:
        raise RuntimeError(f"{context} found no Stage {stage_index} rows in train log: {path}")
    return rows


def _train_log_int(row: Mapping[str, Any], field: str, *, context: str) -> int:
    value = row.get(field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} train log field {field} is not an integer: {value!r}") from exc


def _train_log_bool(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _find_train_log_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage_epoch: int,
    global_epoch: int,
    context: str,
) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in rows
        if _train_log_int(row, "stage_epoch", context=context) == int(stage_epoch)
        and _train_log_int(row, "global_epoch", context=context) == int(global_epoch)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{context} expected exactly one train-log row for stage_epoch={stage_epoch}, "
            f"global_epoch={global_epoch}; found {len(matches)}"
        )
    return matches[0]


def _last_train_log_row(rows: Sequence[Mapping[str, Any]], *, context: str) -> dict[str, Any]:
    return max(
        (dict(row) for row in rows),
        key=lambda row: (
            _train_log_int(row, "stage_epoch", context=context),
            _train_log_int(row, "global_epoch", context=context),
        ),
    )


def _quant_audit_from_train_log_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observer_enabled": _train_log_bool(row, "quant_audit_observer_enabled"),
        "fake_quant_enabled": _train_log_bool(row, "quant_audit_fake_quant_enabled"),
        "qparams_frozen": _train_log_bool(row, "quant_audit_qparams_frozen"),
    }


def _validate_stage1_resume_checkpoint_payload(
    *,
    payload: Mapping[str, Any],
    stage: StageSpec,
    args: argparse.Namespace,
    teacher_checkpoint: Path,
    qparams_csv: Path,
    paths: Any,
    int4_provenance: Mapping[str, Any],
) -> None:
    context = "resume_from_stage1_best"
    _require_payload_equal(context, "model_kind", payload.get("model_kind"), INNOVATION_NAME)
    _require_payload_equal(context, "stage_name", payload.get("stage_name"), stage.name)
    _require_payload_equal(context, "stage_index", _payload_int(payload, "stage_index", context=context), 1)
    _require_payload_equal(context, "seed", _payload_int(payload, "seed", context=context), int(args.seed))
    _require_payload_equal(context, "teacher_checkpoint_sha256", payload.get("teacher_checkpoint_sha256"), file_sha256(teacher_checkpoint))
    _require_payload_equal(context, "qparams_csv_sha256", payload.get("qparams_csv_sha256"), file_sha256(qparams_csv))
    _require_payload_equal(context, "qparams_hash", payload.get("qparams_hash"), int4_provenance.get("qparams_hash"))
    _require_payload_equal(context, "quantized_layers_hash", payload.get("quantized_layers_hash"), int4_provenance.get("quantized_layers_hash"))
    _require_payload_equal(context, "calibration_csv_hash", payload.get("calibration_csv_hash"), int4_provenance.get("calibration_csv_hash"))
    _require_payload_equal(context, "calibration_hash", payload.get("calibration_hash"), int4_provenance.get("calibration_hash"))
    _require_payload_equal(context, "label_mapping_hash", payload.get("label_mapping_hash"), file_sha256(paths.label_mapping))
    _require_payload_equal(context, "int4_student_manifest_sha256", payload.get("int4_student_manifest_sha256"), int4_provenance.get("manifest_sha256"))
    _require_payload_equal(context, "int4_student_run_manifest_hash", payload.get("int4_student_run_manifest_hash"), int4_provenance.get("manifest_hash"))
    _require_payload_equal(context, "int4_student_run_identity_hash", payload.get("int4_student_run_identity_hash"), int4_provenance.get("manifest_run_identity_hash"))
    if not payload.get("run_identity_hash"):
        raise RuntimeError(f"{context} checkpoint is missing run_identity_hash")
    if not isinstance(payload.get("eval_metrics"), Mapping):
        raise RuntimeError(f"{context} checkpoint is missing eval_metrics")


def run_two_stage_decoupling(args: argparse.Namespace) -> dict[str, Any]:
    _reject_amp(args)
    _seed_everything(int(args.seed))
    resolved = _resolved_paths(args)
    stage1, stage2 = _stage_specs(args)
    if stage2.backbone_lr != 0.0:
        raise RuntimeError(f"Stage 2 backbone lr must be exactly 0.0 before training starts, got {stage2.backbone_lr!r}")

    fp32_teacher_stage = stage_fp32_teacher_checkpoint(
        dataset_name=args.dataset,
        seed=int(args.seed),
        baseline_root=Path(resolved["teacher_checkpoint"]).parent.parent.parent,
        explicit_teacher_checkpoint=resolved.get("requested_teacher_checkpoint"),
        staged_checkpoint=resolved["teacher_checkpoint"],
        force_restage=bool(args.force_stage_fp32_teacher),
        dry_run=bool(args.dry_run),
    )
    resolved["teacher_checkpoint"] = Path(str(fp32_teacher_stage["staged_checkpoint"]))
    teacher_checkpoint = resolved["teacher_checkpoint"]
    paths = dataset_paths(args.dataset, str(teacher_checkpoint))
    resolved = _resolved_paths(args, dataset_root=paths.root)
    resolved["teacher_checkpoint"] = teacher_checkpoint
    qparams_csv = resolved["qparams_csv"]

    if not bool(args.dry_run) and not teacher_checkpoint.exists():
        raise FileNotFoundError(f"FP32 teacher checkpoint not found after staging: {teacher_checkpoint}")
    if not qparams_csv.exists():
        raise FileNotFoundError(f"INT4 qparams CSV not found: {qparams_csv}; run run_int4_student.py first")
    if not resolved["int4_manifest"].exists():
        raise FileNotFoundError(f"INT4 calibrated student manifest not found: {resolved['int4_manifest']}; run run_int4_student.py first")
    if not resolved["int4_checkpoint"].exists():
        raise FileNotFoundError(f"INT4 calibrated student checkpoint not found: {resolved['int4_checkpoint']}; run run_int4_student.py first")
    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "resolved": {key: path_str(value) for key, value in resolved.items()},
                    "fp32_teacher_stage": fp32_teacher_stage,
                    "stage_specs": [stage1.__dict__, stage2.__dict__],
                    "resume_from_stage1_best": bool(args.resume_from_stage1_best),
                    "required_checks": {
                        "stage1": {
                            "backbone_lr": ">0",
                            "backbone_requires_grad": True,
                            "classifier_requires_grad": True,
                            "kd_loss": "enabled",
                            "class_prior": "disabled",
                            "long_tail_reweight": "disabled",
                            "class_correction": "disabled",
                        },
                    "stage2": {
                        "backbone_lr": "==0",
                        "backbone_requires_grad": False,
                        "backbone_optimizer_group_exists": False,
                        "classifier_requires_grad": True,
                        "kd_loss": "disabled",
                        "loss": "clean_cross_entropy",
                        },
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return {"dry_run": True, "resolved": resolved, "fp32_teacher_stage": fp32_teacher_stage}

    label_mapping = load_label_mapping(paths.label_mapping)
    num_classes = num_predictor_classes(label_mapping)
    validate_split_labels(paths.train_csv, num_classes=num_classes, name="train")
    validate_split_labels(paths.eval_csv, num_classes=num_classes, name="eval")
    validate_split_labels(paths.test_csv, num_classes=num_classes, name="test")
    label_buckets = infer_label_buckets(paths)
    label_names = label_names_by_id(label_mapping)

    int4_manifest = _load_required_json(resolved["int4_manifest"], name="INT4 student manifest")
    int4_provenance = _validate_int4_student_provenance(
        manifest_path=resolved["int4_manifest"],
        manifest=int4_manifest,
        teacher_checkpoint=teacher_checkpoint,
        int4_checkpoint=resolved["int4_checkpoint"],
        qparams_csv=qparams_csv,
        paths=paths,
    )
    calibration_csv = int4_provenance["calibration_csv"]
    calibration_hash = str(int4_provenance["calibration_hash"])
    layers = [dict(row) for row in int4_provenance["quantized_layers"]]

    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    teacher, teacher_payload = load_trained_checkpoint(teacher_checkpoint, device=device)
    _assert_fp32_teacher_payload(teacher_payload, context="two-stage teacher")
    student, student_init_payload = load_trained_checkpoint(resolved["int4_checkpoint"], device=device)
    student_kind = str(student_init_payload.get("model_kind") or "")
    if student_kind != "int4_student_init":
        raise RuntimeError(f"two-stage student init must be INT4 calibrated student, got model_kind={student_kind!r}")
    _validate_loaded_int4_checkpoint_payload(
        checkpoint_path=resolved["int4_checkpoint"],
        checkpoint=student_init_payload,
        int4_provenance=int4_provenance,
        teacher_checkpoint=teacher_checkpoint,
        paths=paths,
    )
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()

    controller = getattr(student, "_baseline_quant_controller", None)
    if controller is None:
        raise RuntimeError("loaded INT4 calibrated student is missing _baseline_quant_controller")
    initial_quant_audit = _assert_quant_training_state(student, controller, int4_provenance, context="before_two_stage_training")

    train_loader = build_loader(paths, split="train", batch_size=int(args.batch_size), num_workers=int(args.num_workers), seed=int(args.seed))
    eval_loader = build_loader(paths, split="eval", batch_size=int(args.eval_batch_size), num_workers=int(args.num_workers), seed=int(args.seed))
    stage_contracts: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    stage1_train_rows: list[dict[str, Any]] = []
    stage2_train_rows: list[dict[str, Any]] = []
    best_eval_macro_f1_label_universe_nanmean = float("-inf")
    stage2_best_eval_macro_f1_label_universe_nanmean = float("-inf")
    best_checkpoint_path: Path | None = None
    best_epoch = 0
    best_stage_name = ""
    best_eval: dict[str, Any] = {}
    best_quant_audit: dict[str, Any] = {}
    stage1_best_train_row: dict[str, Any] = {}
    stage1_best_eval_payload: dict[str, Any] = {}
    stage1_best_quant_audit: dict[str, Any] = {}
    stage1_best_contract: dict[str, Any] = {}
    stage1_best_checkpoint_payload: dict[str, Any] | None = None
    stage1_best_stage_epoch = 0
    stage1_best_global_epoch = 0
    stage1_split_results: dict[str, Any] = {}
    stage1_eval_metric_row: dict[str, Any] = {}
    stage2_best_train_row: dict[str, Any] = {}
    stage2_best_eval_payload: dict[str, Any] = {}
    stage2_best_quant_audit: dict[str, Any] = {}
    stage2_best_contract: dict[str, Any] = {}
    stage2_best_checkpoint_payload: dict[str, Any] | None = None
    stage2_best_stage_epoch = 0
    stage2_best_global_epoch = 0
    stage2_split_results: dict[str, Any] = {}
    stage2_eval_metric_row: dict[str, Any] = {}
    global_epoch = 0

    for stage in (stage1, stage2):
        resume_stage1_best = bool(args.resume_from_stage1_best) and int(stage.index) == 1
        stage1_resume_checkpoint_payload: dict[str, Any] | None = None
        if resume_stage1_best:
            if not resolved["stage1_checkpoint"].exists():
                raise FileNotFoundError(f"--resume-from-stage1-best requires existing Stage 1 best checkpoint: {resolved['stage1_checkpoint']}")
            if not resolved["stage1_last_checkpoint"].exists():
                raise FileNotFoundError(f"--resume-from-stage1-best requires existing Stage 1 audit checkpoint: {resolved['stage1_last_checkpoint']}")
            if "student" in locals():
                del student
            controller = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            student, stage1_resume_checkpoint_payload = load_trained_checkpoint(resolved["stage1_checkpoint"], device=device)
            _validate_stage1_resume_checkpoint_payload(
                payload=stage1_resume_checkpoint_payload,
                stage=stage,
                args=args,
                teacher_checkpoint=teacher_checkpoint,
                qparams_csv=qparams_csv,
                paths=paths,
                int4_provenance=int4_provenance,
            )
        optimizer, stage_contract = _configure_stage(student, stage, weight_decay=float(args.weight_decay))
        stage_contracts.append(dict(stage_contract))
        stage_protocol = _stage_protocol_payload(args, stage_contracts)
        scheduler = _make_scheduler(
            optimizer,
            total_steps=int(stage.epochs) * max(1, len(train_loader)),
            warmup_ratio=float(args.warmup_ratio),
        )
        _append_stage_log(
            resolved["stage_log"],
            {
                "event": "stage_start",
                "stage": stage.name,
                "stage_index": int(stage.index),
                "objective": stage.objective,
                "epochs": int(stage.epochs),
                "resume_from_stage1_best": bool(resume_stage1_best),
                "contract": stage_contract,
            },
        )
        print(
            f"[innovation-two-stage] start {stage.name} "
            f"backbone_lr={stage_contract['backbone_lr']} classifier_lr={stage_contract['classifier_lr']} "
            f"backbone_optimizer_group_exists={stage_contract['backbone_optimizer_group_exists']} "
            f"kd_enabled={stage_contract['kd_loss_enabled']}"
        )

        last_eval: dict[str, Any] = {}
        last_eval_breakdown: dict[str, Any] = {}
        last_quant_audit: dict[str, Any] = {}
        stage_epoch_values = range(1, int(stage.epochs) + 1)
        if resume_stage1_best:
            if stage1_resume_checkpoint_payload is None:
                raise RuntimeError("--resume-from-stage1-best did not load a Stage 1 checkpoint payload")
            stage1_log_rows = _read_train_log_rows(resolved["stage1_train_log"], stage_index=1, context="resume_from_stage1_best")
            stage1_best_stage_epoch = _payload_int(stage1_resume_checkpoint_payload, "stage_epoch", context="resume_from_stage1_best")
            stage1_best_global_epoch = _payload_int(stage1_resume_checkpoint_payload, "epoch", context="resume_from_stage1_best")
            stage1_best_train_row = _find_train_log_row(
                stage1_log_rows,
                stage_epoch=stage1_best_stage_epoch,
                global_epoch=stage1_best_global_epoch,
                context="resume_from_stage1_best",
            )
            stage1_last_train_row = _last_train_log_row(stage1_log_rows, context="resume_from_stage1_best")
            stage1_completed_stage_epoch = _train_log_int(stage1_last_train_row, "stage_epoch", context="resume_from_stage1_best")
            stage1_completed_global_epoch = _train_log_int(stage1_last_train_row, "global_epoch", context="resume_from_stage1_best")
            if stage1_completed_stage_epoch < int(stage.epochs):
                raise RuntimeError(
                    "--resume-from-stage1-best requires a completed Stage 1 train log: "
                    f"last_stage_epoch={stage1_completed_stage_epoch}, expected_at_least={int(stage.epochs)}"
                )
            global_epoch = max(int(global_epoch), int(stage1_completed_global_epoch))
            stage1_best_checkpoint_payload = dict(stage1_resume_checkpoint_payload)
            stage1_best_eval_payload = dict(stage1_resume_checkpoint_payload.get("eval_metrics") or {})
            stage1_best_quant_audit = dict(stage1_resume_checkpoint_payload.get("training_quant_audit") or {})
            stage1_best_contract = dict(stage_contract)
            best_eval_macro_f1_label_universe_nanmean = float(stage1_best_eval_payload.get(STAGE1_SELECTION_METRIC, float("nan")))
            if not math.isfinite(best_eval_macro_f1_label_universe_nanmean):
                raise RuntimeError(f"--resume-from-stage1-best checkpoint has no finite {STAGE1_SELECTION_METRIC}")
            best_epoch = int(stage1_best_global_epoch)
            best_stage_name = stage.name
            best_eval = dict(stage1_best_eval_payload)
            best_quant_audit = dict(stage1_best_quant_audit)
            best_checkpoint_path = resolved["stage1_checkpoint"]
            last_eval = {
                "loss": float(stage1_last_train_row.get("eval_loss", float("nan"))),
                "accuracy": float(stage1_last_train_row.get("eval_accuracy", float("nan"))),
            }
            last_eval_breakdown = {
                STAGE1_SELECTION_METRIC: float(stage1_last_train_row.get("eval_macro_f1_label_universe_nanmean", float("nan"))),
            }
            last_quant_audit = _quant_audit_from_train_log_row(stage1_last_train_row)
            stage_epoch_values = range(0)
            _append_stage_log(
                resolved["stage_log"],
                {
                    "event": "stage1_resume_from_best",
                    "checkpoint": path_str(resolved["stage1_checkpoint"]),
                    "checkpoint_sha256": file_sha256(resolved["stage1_checkpoint"]),
                    "checkpoint_run_identity_hash": stage1_resume_checkpoint_payload.get("run_identity_hash"),
                    "stage1_best_stage_epoch": int(stage1_best_stage_epoch),
                    "stage1_best_global_epoch": int(stage1_best_global_epoch),
                    "stage1_completed_stage_epoch": int(stage1_completed_stage_epoch),
                    "stage1_completed_global_epoch": int(stage1_completed_global_epoch),
                    "stage1_last_checkpoint": path_str(resolved["stage1_last_checkpoint"]),
                    "stage1_last_checkpoint_sha256": file_sha256(resolved["stage1_last_checkpoint"]),
                },
            )
            print(
                f"[innovation-two-stage] resume {stage.name} "
                f"from checkpoint={resolved['stage1_checkpoint']} "
                f"best_stage_epoch={stage1_best_stage_epoch} completed_stage_epoch={stage1_completed_stage_epoch}"
            )
        for stage_epoch in stage_epoch_values:
            global_epoch += 1
            train_dataset = getattr(train_loader, "dataset", None)
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(global_epoch)
            if stage.kd_enabled:
                train_metrics = train_stage_one_epoch(
                    student=student,
                    teacher=teacher,
                    loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    alpha=float(args.alpha),
                    temperature=float(args.temperature),
                    num_classes=num_classes,
                )
            else:
                train_metrics = train_stage_two_epoch(
                    student=student,
                    loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    num_classes=num_classes,
                )
            if stage.index == 2 and float(train_metrics.get("teacher_forward_count", 0.0)) != 0.0:
                raise RuntimeError("Stage 2 contract violation: teacher distillation loss was computed")

            last_quant_audit = _assert_quant_training_state(student, controller, int4_provenance, context=f"after_{stage.name}_epoch_{stage_epoch}")
            eval_metrics = evaluate_student(model=student, loader=eval_loader, device=device, amp=False, num_classes=num_classes)
            eval_breakdown, _per_class_rows = compute_eval_breakdown(
                eval_metrics,
                label_names=label_names,
                label_buckets=label_buckets,
            )
            last_eval = dict(eval_metrics)
            last_eval_breakdown = dict(eval_breakdown)
            current = float(eval_breakdown.get(STAGE1_SELECTION_METRIC, float("nan")))
            row_backbone_lr = _optimizer_lr(optimizer, "backbone") if stage.index == 1 else 0.0
            row_backbone_optimizer_group_exists = bool(stage_contract.get("backbone_optimizer_group_exists", False))
            row = {
                "stage": stage.name,
                "stage_index": int(stage.index),
                "stage_epoch": int(stage_epoch),
                "global_epoch": int(global_epoch),
                "objective": stage.objective,
                "backbone_lr": row_backbone_lr,
                "backbone_optimizer_group_exists": row_backbone_optimizer_group_exists,
                "classifier_lr": _optimizer_lr(optimizer, "classifier"),
                "kd_loss_enabled": bool(stage.kd_enabled),
                "backbone_trainable": bool(stage.train_backbone),
                "classifier_trainable": bool(stage.train_classifier),
                **{key: float(value) for key, value in train_metrics.items()},
                "eval_loss": float(eval_metrics["loss"]),
                "eval_accuracy": float(eval_metrics["accuracy"]),
                "eval_macro_f1_label_universe_nanmean": current,
                "eval_macro_f1_observed_classes": float(eval_breakdown.get("eval_macro_f1_observed_classes", float("nan"))),
                "eval_macro_f1_legacy_zero_absent_diagnostic": float(eval_breakdown.get("eval_macro_f1_legacy_zero_absent_diagnostic", float("nan"))),
                "quant_audit_observer_enabled": bool(last_quant_audit.get("observer_enabled")),
                "quant_audit_fake_quant_enabled": bool(last_quant_audit.get("fake_quant_enabled")),
                "quant_audit_qparams_frozen": bool(last_quant_audit.get("qparams_frozen")),
            }
            if stage.index == 1:
                row.update(
                    {
                        "stage1_train_total_loss": row["train_loss"],
                        "stage1_train_ce_loss": row["train_ce_loss"],
                        "stage1_train_kd_loss": row["train_kd_term"],
                        "stage1_train_kl_loss": row["train_kl_loss"],
                        "stage1_train_temperature": float(args.temperature),
                        "stage1_train_alpha": float(args.alpha),
                        "stage1_backbone_lr": row["backbone_lr"],
                        "stage1_classifier_lr": row["classifier_lr"],
                        "stage1_backbone_trainable_param_count": int(stage_contract["parameter_status"]["backbone"]["trainable_num_params"]),
                        "stage1_classifier_trainable_param_count": int(stage_contract["parameter_status"]["classifier"]["trainable_num_params"]),
                        "stage1_teacher_kd_enabled": True,
                        "stage1_qparams_frozen": bool(last_quant_audit.get("qparams_frozen")),
                        "stage1_fake_quant_enabled": bool(last_quant_audit.get("fake_quant_enabled")),
                        "stage1_observer_enabled": bool(last_quant_audit.get("observer_enabled")),
                    }
                )
                stage1_train_rows.append(row)
                _write_train_log(resolved["stage1_train_log"], stage1_train_rows)
            elif stage.index == 2:
                row.update(
                    {
                        "stage2_train_ce_loss": row["train_ce_loss"],
                        "stage2_train_total_loss": row["train_loss"],
                        "stage2_train_kd_loss": 0.0,
                        "stage2_teacher_kd_enabled": False,
                        "stage2_eval_accuracy": float(eval_metrics["accuracy"]),
                        "stage2_eval_macro_f1_label_universe_nanmean": current,
                        "stage2_eval_many_accuracy": float(eval_breakdown.get("eval_many_acc", float("nan"))),
                        "stage2_eval_medium_accuracy": float(eval_breakdown.get("eval_medium_acc", float("nan"))),
                        "stage2_eval_few_accuracy": float(eval_breakdown.get("eval_few_acc", float("nan"))),
                        "stage2_eval_tail_accuracy": float(eval_breakdown.get("eval_tail_acc", float("nan"))),
                        "stage2_backbone_lr": 0.0,
                        "stage2_backbone_optimizer_group_exists": False,
                        "stage2_classifier_lr": row["classifier_lr"],
                        "stage2_backbone_trainable_param_count": 0,
                        "stage2_classifier_trainable_param_count": int(stage_contract["parameter_status"]["classifier"]["trainable_num_params"]),
                    }
                )
                stage2_train_rows.append(row)
                _write_train_log(resolved["stage2_train_log"], stage2_train_rows)
            print(
                f"[innovation-two-stage] stage={stage.index} epoch={stage_epoch}/{stage.epochs} "
                f"train_loss={row['train_loss']:.6f} eval_macro_f1_label_universe_nanmean={current:.6f}"
            )
            if stage.index == 1 and math.isfinite(current) and current > best_eval_macro_f1_label_universe_nanmean:
                best_eval_macro_f1_label_universe_nanmean = current
                best_epoch = int(global_epoch)
                best_stage_name = stage.name
                best_eval = {**eval_metrics, **eval_breakdown}
                best_quant_audit = dict(last_quant_audit)
                best_checkpoint_path = resolved["stage1_checkpoint"]
                stage1_best_train_row = dict(row)
                stage1_best_eval_payload = dict(best_eval)
                stage1_best_quant_audit = dict(last_quant_audit)
                stage1_best_contract = dict(stage_contract)
                stage1_best_stage_epoch = int(stage_epoch)
                stage1_best_global_epoch = int(global_epoch)
                stage1_best_checkpoint_payload = _save_stage_checkpoint(
                    resolved["stage1_checkpoint"],
                    model=student,
                    stage=stage,
                    global_epoch=global_epoch,
                    stage_epoch=int(stage_epoch),
                    eval_metrics=stage1_best_eval_payload,
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
                    quant_audit=last_quant_audit,
                    stage_protocol=stage_protocol,
                    best_metric_so_far=best_eval_macro_f1_label_universe_nanmean,
                )
                _append_stage_log(
                    resolved["stage_log"],
                    {
                        "event": "stage1_best_checkpoint",
                        "stage": stage.name,
                        "stage_index": int(stage.index),
                        "stage_epoch": int(stage_epoch),
                        "global_epoch": int(global_epoch),
                        "selection_metric": STAGE1_SELECTION_METRIC,
                        "selection_metric_value": float(current),
                        "checkpoint": path_str(resolved["stage1_checkpoint"]),
                        "checkpoint_sha256": file_sha256(resolved["stage1_checkpoint"]),
                    },
                )
            if stage.index == 2 and math.isfinite(current) and current > stage2_best_eval_macro_f1_label_universe_nanmean:
                stage2_best_eval_macro_f1_label_universe_nanmean = current
                best_epoch = int(global_epoch)
                best_stage_name = stage.name
                best_eval = {**eval_metrics, **eval_breakdown}
                best_quant_audit = dict(last_quant_audit)
                best_checkpoint_path = resolved["stage2_checkpoint"]
                stage2_best_train_row = dict(row)
                stage2_best_eval_payload = dict(best_eval)
                stage2_best_quant_audit = dict(last_quant_audit)
                stage2_best_contract = dict(stage_contract)
                stage2_best_stage_epoch = int(stage_epoch)
                stage2_best_global_epoch = int(global_epoch)
                stage2_best_checkpoint_payload = _save_stage_checkpoint(
                    resolved["stage2_checkpoint"],
                    model=student,
                    stage=stage,
                    global_epoch=global_epoch,
                    stage_epoch=int(stage_epoch),
                    eval_metrics=stage2_best_eval_payload,
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
                    quant_audit=last_quant_audit,
                    stage_protocol=stage_protocol,
                    best_metric_so_far=stage2_best_eval_macro_f1_label_universe_nanmean,
                )
                _append_stage_log(
                    resolved["stage_log"],
                    {
                        "event": "stage2_best_checkpoint",
                        "stage": stage.name,
                        "stage_index": int(stage.index),
                        "stage_epoch": int(stage_epoch),
                        "global_epoch": int(global_epoch),
                        "selection_metric": STAGE2_SELECTION_METRIC,
                        "selection_metric_value": float(current),
                        "checkpoint": path_str(resolved["stage2_checkpoint"]),
                        "checkpoint_sha256": file_sha256(resolved["stage2_checkpoint"]),
                    },
                )

        if not last_eval:
            raise RuntimeError(f"{stage.name} produced no evaluation metrics")
        stage_eval_payload = {**last_eval, **last_eval_breakdown}
        if stage.index == 1:
            if stage1_best_checkpoint_payload is None or not stage1_best_eval_payload:
                raise RuntimeError(f"Stage 1 produced no finite {STAGE1_SELECTION_METRIC} checkpoint")
            stage_checkpoint_path = resolved["stage1_checkpoint"]
            if not resume_stage1_best:
                _save_stage_checkpoint(
                    resolved["stage1_last_checkpoint"],
                    model=student,
                    stage=stage,
                    global_epoch=global_epoch,
                    stage_epoch=int(stage.epochs),
                    eval_metrics=stage_eval_payload,
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
                    quant_audit=last_quant_audit,
                    stage_protocol=stage_protocol,
                    best_metric_so_far=best_eval_macro_f1_label_universe_nanmean,
                )
            checkpoint_payload = dict(stage1_best_checkpoint_payload)
            stage1_split_results, _stage1_export_checkpoint, _stage1_token_layers, _stage1_token_export, stage1_export_run_identity_hash, _stage1_export_quant_audit = _run_export_eval(
                args=args,
                resolved=resolved,
                checkpoint_path=resolved["stage1_checkpoint"],
                calibration_hash=calibration_hash,
                label_buckets=label_buckets,
                device=device,
                splits=("eval",),
                export_root=resolved["seed_dir"] / "stage1_export",
                export_name=f"{INNOVATION_NAME}_stage1",
            )
            stage1_eval_metric_row = build_stage1_eval_metric_row(
                split_result=stage1_split_results["eval"],
                best_train_row=stage1_best_train_row,
                stage_contract=stage1_best_contract,
                quant_audit=stage1_best_quant_audit,
                seed=int(args.seed),
                stage_epoch=stage1_best_stage_epoch,
                global_epoch=stage1_best_global_epoch,
                checkpoint_path=resolved["stage1_checkpoint"],
                checkpoint_sha256=file_sha256(resolved["stage1_checkpoint"]),
                alpha=float(args.alpha),
                temperature=float(args.temperature),
            )
            write_stage1_eval_metrics(resolved["stage1_eval_metrics"], stage1_eval_metric_row)
            stage1_manifest = attach_run_manifest_hash(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "artifact_type": "innovation_two_stage_decoupling_stage1_manifest",
                    "innovation_name": INNOVATION_NAME,
                    "dataset_name": paths.name,
                    "seed": int(args.seed),
                    "resume_from_stage1_best": bool(resume_stage1_best),
                    "selection_metric": STAGE1_SELECTION_METRIC,
                    "selection_metric_value": float(best_eval_macro_f1_label_universe_nanmean),
                    "stage1_checkpoint": path_str(resolved["stage1_checkpoint"]),
                    "stage1_checkpoint_sha256": file_sha256(resolved["stage1_checkpoint"]),
                    "stage1_checkpoint_usage": "eval_best",
                    "stage1_last": path_str(resolved["stage1_last_checkpoint"]),
                    "stage1_last_sha256": file_sha256(resolved["stage1_last_checkpoint"]),
                    "stage1_last_usage": "audit_only",
                    "stage1_last_preserved_from_resume": bool(resume_stage1_best),
                    "stage1_last_in_stage1_eval_metrics": False,
                    "stage1_train_log": path_str(resolved["stage1_train_log"]),
                    "stage1_eval_metrics": path_str(resolved["stage1_eval_metrics"]),
                    "stage1_best_epoch": int(stage1_best_stage_epoch),
                    "stage1_best_global_epoch": int(stage1_best_global_epoch),
                    "stage1_best_train_row": dict(stage1_best_train_row),
                    "stage1_best_eval": dict(stage1_best_eval_payload),
                    "stage1_eval_metric_row": dict(stage1_eval_metric_row),
                    "stage1_contract": dict(stage1_best_contract),
                    "stage1_quant_audit_best": dict(stage1_best_quant_audit),
                    "stage1_quant_audit_last": dict(last_quant_audit),
                    "stage1_export": {
                        "run_identity_hash": stage1_export_run_identity_hash,
                        "split_results": stage1_split_results,
                    },
                    "teacher_checkpoint": path_str(teacher_checkpoint),
                    "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
                    "int4_calibrated_student_checkpoint": path_str(resolved["int4_checkpoint"]),
                    "int4_calibrated_student_checkpoint_sha256": file_sha256(resolved["int4_checkpoint"]),
                    "qparams_csv": path_str(qparams_csv),
                    "qparams_csv_sha256": file_sha256(qparams_csv),
                    "critical_checks": {
                        "stage1_backbone_lr_gt_0": float(stage1_best_contract.get("backbone_lr", 0.0)) > 0.0,
                        "stage1_kd_loss_enabled": bool(stage1_best_contract.get("kd_loss_enabled")),
                        "stage1_checkpoint_is_eval_best": True,
                        "stage1_last_is_audit_only": True,
                        "fp32_reference_loaded": bool(stage1_eval_metric_row.get("stage1_reference_checkpoint_hash")),
                    },
                }
            )
            _write_json(resolved["stage1_manifest"], stage1_manifest)
            student, stage1_resume_payload = load_trained_checkpoint(resolved["stage1_checkpoint"], device=device)
            stage1_init_checkpoint_sha256 = file_sha256(resolved["stage1_checkpoint"])
            _require_loaded_checkpoint_identity(
                context="stage1_best_loaded_for_stage2",
                loaded_payload=stage1_resume_payload,
                expected_payload=stage1_best_checkpoint_payload,
                expected_stage_index=1,
                expected_stage_epoch=stage1_best_stage_epoch,
                expected_global_epoch=stage1_best_global_epoch,
            )
            controller = getattr(student, "_baseline_quant_controller", None)
            if controller is None:
                raise RuntimeError("Stage 1 best checkpoint reload is missing _baseline_quant_controller")
            stage1_resume_quant_audit = _assert_quant_training_state(
                student,
                controller,
                int4_provenance,
                context="stage1_best_loaded_for_stage2",
            )
            _append_stage_log(
                resolved["stage_log"],
                {
                    "event": "stage1_best_loaded_for_stage2",
                    "checkpoint": path_str(resolved["stage1_checkpoint"]),
                    "checkpoint_sha256": stage1_init_checkpoint_sha256,
                    "checkpoint_run_identity_hash": stage1_resume_payload.get("run_identity_hash"),
                    "checkpoint_stage_epoch": int(stage1_resume_payload.get("stage_epoch")),
                    "checkpoint_global_epoch": int(stage1_resume_payload.get("epoch")),
                    "stage1_last_is_audit_only": True,
                    "quant_audit": dict(stage1_resume_quant_audit),
                },
            )
            _append_stage_log(
                resolved["stage_log"],
                {
                    "event": "stage_transition",
                    "from_stage": stage1.name,
                    "to_stage": stage2.name,
                    "transition_checkpoint": path_str(resolved["stage1_checkpoint"]),
                    "transition_checkpoint_sha256": stage1_init_checkpoint_sha256,
                    "transition_checkpoint_usage": "stage1_eval_best",
                    "transition_run_identity_hash": stage1_resume_payload.get("run_identity_hash"),
                    "transition_stage1_best_stage_epoch": int(stage1_best_stage_epoch),
                    "transition_stage1_best_global_epoch": int(stage1_best_global_epoch),
                    "stage2_contract_required": {
                        "backbone_lr": 0.0,
                        "backbone_requires_grad": False,
                        "backbone_optimizer_group_exists": False,
                        "classifier_requires_grad": True,
                        "kd_loss_enabled": False,
                    },
                },
            )
            print(
                f"[innovation-two-stage] transition {stage1.name} -> {stage2.name} "
                f"from checkpoint={resolved['stage1_checkpoint']}"
            )
        else:
            if stage2_best_checkpoint_payload is None or not stage2_best_eval_payload:
                raise RuntimeError(f"Stage 2 produced no finite {STAGE2_SELECTION_METRIC} checkpoint")
            stage_checkpoint_path = resolved["stage2_checkpoint"]
            _save_stage_checkpoint(
                resolved["stage2_last_checkpoint"],
                model=student,
                stage=stage,
                global_epoch=global_epoch,
                stage_epoch=int(stage.epochs),
                eval_metrics=stage_eval_payload,
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
                quant_audit=last_quant_audit,
                stage_protocol=stage_protocol,
                best_metric_so_far=stage2_best_eval_macro_f1_label_universe_nanmean,
            )
            checkpoint_payload = dict(stage2_best_checkpoint_payload)
            stage2_split_results, _stage2_export_checkpoint, _stage2_token_layers, _stage2_token_export, stage2_export_run_identity_hash, _stage2_export_quant_audit = _run_export_eval(
                args=args,
                resolved=resolved,
                checkpoint_path=resolved["stage2_checkpoint"],
                calibration_hash=calibration_hash,
                label_buckets=label_buckets,
                device=device,
                splits=("eval",),
                export_root=resolved["seed_dir"] / "stage2_export",
                export_name=f"{INNOVATION_NAME}_stage2",
            )
            stage2_eval_metric_row = build_stage2_clean_eval_metric_row(
                split_result=stage2_split_results["eval"],
                best_train_row=stage2_best_train_row,
                stage_contract=stage2_best_contract,
                seed=int(args.seed),
                stage_epoch=stage2_best_stage_epoch,
                global_epoch=stage2_best_global_epoch,
                checkpoint_path=resolved["stage2_checkpoint"],
                checkpoint_sha256=file_sha256(resolved["stage2_checkpoint"]),
            )
            write_stage2_eval_metrics(resolved["stage2_eval_metrics"], stage2_eval_metric_row)
            stage2_manifest = attach_run_manifest_hash(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "artifact_type": "innovation_two_stage_decoupling_stage2_manifest",
                    "innovation_name": INNOVATION_NAME,
                    "dataset_name": paths.name,
                    "seed": int(args.seed),
                    "selection_metric": STAGE2_SELECTION_METRIC,
                    "selection_metric_value": float(stage2_best_eval_macro_f1_label_universe_nanmean),
                    "stage2_checkpoint": path_str(resolved["stage2_checkpoint"]),
                    "stage2_checkpoint_sha256": file_sha256(resolved["stage2_checkpoint"]),
                    "stage2_checkpoint_usage": "eval_best",
                    "stage2_last": path_str(resolved["stage2_last_checkpoint"]),
                    "stage2_last_sha256": file_sha256(resolved["stage2_last_checkpoint"]),
                    "stage2_last_usage": "audit_only",
                    "stage2_last_in_stage2_eval_metrics": False,
                    "stage2_train_log": path_str(resolved["stage2_train_log"]),
                    "stage2_eval_metrics": path_str(resolved["stage2_eval_metrics"]),
                    "stage2_best_epoch": int(stage2_best_stage_epoch),
                    "stage2_best_global_epoch": int(stage2_best_global_epoch),
                    "stage2_best_train_row": dict(stage2_best_train_row),
                    "stage2_best_eval": dict(stage2_best_eval_payload),
                    "stage2_eval_metric_row": dict(stage2_eval_metric_row),
                    "stage2_contract": dict(stage2_best_contract),
                    "stage2_quant_audit_best": dict(stage2_best_quant_audit),
                    "stage2_quant_audit_last": dict(last_quant_audit),
                    "stage2_export": {
                        "run_identity_hash": stage2_export_run_identity_hash,
                        "split_results": stage2_split_results,
                    },
                    "teacher_checkpoint": path_str(teacher_checkpoint),
                    "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
                    "stage1_checkpoint": path_str(resolved["stage1_checkpoint"]),
                    "stage1_checkpoint_sha256": file_sha256(resolved["stage1_checkpoint"]),
                    "critical_checks": {
                        "stage2_backbone_lr_eq_0": float(stage2_best_contract.get("backbone_lr", -1.0)) == 0.0,
                        "stage2_backbone_trainable_param_count_eq_0": int(stage2_best_contract["parameter_status"]["backbone"]["trainable_num_params"]) == 0,
                        "stage2_backbone_optimizer_group_absent": not bool(stage2_best_contract.get("backbone_optimizer_group_exists", True)),
                        "stage2_kd_loss_disabled": True,
                        "stage2_clean_ce_only": True,
                        "stage2_class_prior_disabled": True,
                        "stage2_checkpoint_is_eval_best": True,
                        "stage2_last_is_audit_only": True,
                        "fp32_reference_loaded": bool(stage2_eval_metric_row.get("stage2_reference_checkpoint_hash")),
                    },
                }
            )
            _write_json(resolved["stage2_manifest"], stage2_manifest)
            student, stage2_resume_payload = load_trained_checkpoint(resolved["stage2_checkpoint"], device=device)
            stage2_final_checkpoint_sha256 = file_sha256(resolved["stage2_checkpoint"])
            _require_loaded_checkpoint_identity(
                context="stage2_best_loaded_for_final_export",
                loaded_payload=stage2_resume_payload,
                expected_payload=stage2_best_checkpoint_payload,
                expected_stage_index=2,
                expected_stage_epoch=stage2_best_stage_epoch,
                expected_global_epoch=stage2_best_global_epoch,
            )
            controller = getattr(student, "_baseline_quant_controller", None)
            if controller is None:
                raise RuntimeError("Stage 2 best checkpoint reload is missing _baseline_quant_controller")
            stage2_resume_quant_audit = _assert_quant_training_state(
                student,
                controller,
                int4_provenance,
                context="stage2_best_loaded_for_final_export",
            )
            _append_stage_log(
                resolved["stage_log"],
                {
                    "event": "stage2_best_loaded_for_final_export",
                    "checkpoint": path_str(resolved["stage2_checkpoint"]),
                    "checkpoint_sha256": stage2_final_checkpoint_sha256,
                    "checkpoint_run_identity_hash": stage2_resume_payload.get("run_identity_hash"),
                    "checkpoint_stage_epoch": int(stage2_resume_payload.get("stage_epoch")),
                    "checkpoint_global_epoch": int(stage2_resume_payload.get("epoch")),
                    "stage2_last_is_audit_only": True,
                    "quant_audit": dict(stage2_resume_quant_audit),
                },
            )
        stage_row_eval_metric = (
            float(stage1_best_eval_payload.get(STAGE1_SELECTION_METRIC, float("nan")))
            if stage.index == 1
            else float(stage2_best_eval_payload.get(STAGE2_SELECTION_METRIC, float("nan")))
        )
        stage_row = {
            "stage": stage.name,
            "stage_index": int(stage.index),
            "checkpoint": path_str(stage_checkpoint_path),
            "checkpoint_sha256": file_sha256(stage_checkpoint_path),
            "checkpoint_run_identity_hash": checkpoint_payload.get("run_identity_hash"),
            "last_checkpoint": path_str(resolved["stage1_last_checkpoint"]) if stage.index == 1 else path_str(resolved["stage2_last_checkpoint"]),
            "last_checkpoint_sha256": file_sha256(resolved["stage1_last_checkpoint"]) if stage.index == 1 else file_sha256(resolved["stage2_last_checkpoint"]),
            "checkpoint_usage": "eval_best",
            "last_checkpoint_usage": "audit_only" if stage.index in {1, 2} else "stage_final",
            "final_stage_epoch": int(stage.epochs),
            "global_epoch": int(global_epoch),
            "eval_macro_f1_label_universe_nanmean": stage_row_eval_metric,
            "last_eval_macro_f1_label_universe_nanmean": float(stage_eval_payload.get(STAGE1_SELECTION_METRIC, float("nan"))),
            "stage_contract": stage_contract,
        }
        stage_rows.append(stage_row)
        _append_stage_log(resolved["stage_log"], {"event": "stage_end", **stage_row})
        print(f"[innovation-two-stage] saved {stage.name} checkpoint={stage_checkpoint_path}")

    if best_checkpoint_path is None or best_epoch <= 0:
        raise RuntimeError("Two-stage decoupling never produced a finite eval_macro_f1_label_universe_nanmean checkpoint")

    final_quant_audit = _assert_quant_training_state(student, controller, int4_provenance, context="after_two_stage_final")
    final_eval = evaluate_student(model=student, loader=eval_loader, device=device, amp=False, num_classes=num_classes)
    final_eval_breakdown, _per_class_rows = compute_eval_breakdown(
        final_eval,
        label_names=label_names,
        label_buckets=label_buckets,
    )
    stage_protocol = _stage_protocol_payload(args, stage_contracts)
    final_checkpoint_payload = _save_stage_checkpoint(
        resolved["final_checkpoint"],
        model=student,
        stage=stage2,
        global_epoch=int(stage2_best_global_epoch),
        stage_epoch=int(stage2_best_stage_epoch),
        eval_metrics={**final_eval, **final_eval_breakdown},
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
        stage_protocol=stage_protocol,
        best_metric_so_far=stage2_best_eval_macro_f1_label_universe_nanmean,
    )

    split_results, export_checkpoint, token_layer_indices, token_export, export_run_identity_hash, export_quant_audit = _run_export_eval(
        args=args,
        resolved=resolved,
        checkpoint_path=resolved["final_checkpoint"],
        calibration_hash=calibration_hash,
        label_buckets=label_buckets,
        device=device,
    )
    _write_combined_metrics(split_results, resolved["metrics_csv"], seed=int(args.seed))
    oer_protocol_by_split = {str(split): result.get("oer_protocol") for split, result in split_results.items()}
    first_oer_protocol = next((value for value in oer_protocol_by_split.values() if isinstance(value, Mapping)), {})

    audio_cfg = audio_config_from_project()
    int4_provenance_payload = {
        "manifest_path": int4_provenance.get("manifest_path"),
        "manifest_sha256": int4_provenance.get("manifest_sha256"),
        "run_manifest_hash": int4_provenance.get("manifest_hash"),
        "manifest_run_identity_hash": int4_provenance.get("manifest_run_identity_hash"),
        "calibration_csv": path_str(int4_provenance.get("calibration_csv")),
        "calibration_csv_hash": int4_provenance.get("calibration_csv_hash"),
        "calibration_manifest_hash": int4_provenance.get("calibration_manifest_hash"),
        "calibration_hash": int4_provenance.get("calibration_hash"),
        "qparams_hash": int4_provenance.get("qparams_hash"),
        "qparams_csv_sha256": int4_provenance.get("qparams_csv_sha256"),
        "quantized_layers_hash": int4_provenance.get("quantized_layers_hash"),
        "quantized_layers": layers,
    }
    manifest = baseline_common_manifest(
        stage="innovation_two_stage_decoupling",
        dataset_name=paths.name,
        checkpoint=resolved["final_checkpoint"],
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
            "innovation_name": INNOVATION_NAME,
            "teacher_checkpoint": path_str(teacher_checkpoint),
            "teacher_checkpoint_sha256": file_sha256(teacher_checkpoint),
            "fp32_teacher_stage": dict(fp32_teacher_stage),
            "student_init": f"{INT4_BASELINE_NAME}_from_teacher_seed_{int(args.seed)}",
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
            "model_kind": INNOVATION_NAME,
            "stage1_checkpoint": path_str(resolved["stage1_checkpoint"]),
            "stage1_checkpoint_sha256": file_sha256(resolved["stage1_checkpoint"]),
            "stage2_checkpoint": path_str(resolved["stage2_checkpoint"]),
            "stage2_checkpoint_sha256": file_sha256(resolved["stage2_checkpoint"]),
            "stage2_checkpoint_usage": "eval_best",
            "stage2_last": path_str(resolved["stage2_last_checkpoint"]),
            "stage2_last_sha256": file_sha256(resolved["stage2_last_checkpoint"]),
            "stage2_last_usage": "audit_only",
            "final_checkpoint": path_str(resolved["final_checkpoint"]),
            "final_checkpoint_sha256": file_sha256(resolved["final_checkpoint"]),
            "checkpoint_run_identity_hash": final_checkpoint_payload.get("run_identity_hash"),
            "run_identity_hash": export_run_identity_hash,
            "selection_metric": STAGE2_SELECTION_METRIC,
            "best_epoch": int(best_epoch),
            "best_stage": best_stage_name,
            "best_eval_macro_f1_label_universe_nanmean": float(stage2_best_eval_macro_f1_label_universe_nanmean),
            "selected_checkpoint": path_str(best_checkpoint_path),
            "selected_checkpoint_sha256": file_sha256(best_checkpoint_path),
            "best_eval": best_eval,
            "resume_from_stage1_best": bool(args.resume_from_stage1_best),
            "stage1_last": path_str(resolved["stage1_last_checkpoint"]),
            "stage1_last_sha256": file_sha256(resolved["stage1_last_checkpoint"]),
            "stage1_last_role": "audit_only",
            "stage1_last_preserved_from_resume": bool(args.resume_from_stage1_best),
            "stage1_train_log": path_str(resolved["stage1_train_log"]),
            "stage1_eval_metrics": path_str(resolved["stage1_eval_metrics"]),
            "stage1_manifest": path_str(resolved["stage1_manifest"]),
            "stage1_eval_metric_row": stage1_eval_metric_row,
            "stage1_stage2_init_checkpoint": path_str(resolved["stage1_checkpoint"]),
            "stage1_stage2_init_policy": "Stage 2 starts from Stage 1 eval-best checkpoint; stage1_last.pt is audit-only",
            "stage2_train_log": path_str(resolved["stage2_train_log"]),
            "stage2_eval_metrics": path_str(resolved["stage2_eval_metrics"]),
            "stage2_manifest": path_str(resolved["stage2_manifest"]),
            "stage2_eval_metric_row": stage2_eval_metric_row,
            "stage2_stage_final_checkpoint_policy": "Stage 2 checkpoint is eval-best raw student logits; stage2_last.pt is audit-only",
            "stage_log": path_str(resolved["stage_log"]),
            "metrics_csv": path_str(resolved["metrics_csv"]),
            "two_stage_protocol": stage_protocol,
            "stage_checkpoints": stage_rows,
            "critical_checks": {
                "stage1_backbone_lr_gt_0": float(args.stage1_backbone_lr) > 0.0,
                "stage1_kd_loss_enabled": True,
                "stage2_backbone_lr_eq_0": float(stage2_best_contract.get("backbone_lr", -1.0)) == 0.0,
                "stage2_backbone_requires_grad_false": True,
                "stage2_backbone_optimizer_group_absent": not bool(stage2_best_contract.get("backbone_optimizer_group_exists", True)),
                "stage2_kd_loss_disabled": True,
                "stage2_clean_ce_only": True,
                "stage2_class_prior_disabled": True,
            },
            "token_layers": [int(item) for item in token_layer_indices],
            "token_export": token_export,
            "export_all_token_layers": token_export == "all",
            "teacher_frozen": True,
            "qparams_frozen": True,
            "student_backbone_trainable_final": False,
            "train_quant_audit_initial": initial_quant_audit,
            "train_quant_audit_best": best_quant_audit,
            "train_quant_audit_final": final_quant_audit,
            "export_quant_audit": export_quant_audit,
            "artifact_sha256": {
                "stage1_checkpoint.pt": file_sha256(resolved["stage1_checkpoint"]),
                "stage1_last.pt": file_sha256(resolved["stage1_last_checkpoint"]),
                "stage1_train_log.csv": file_sha256(resolved["stage1_train_log"]),
                "stage1_eval_metrics.csv": file_sha256(resolved["stage1_eval_metrics"]),
                "stage1_manifest.json": file_sha256(resolved["stage1_manifest"]),
                "stage2_checkpoint.pt": file_sha256(resolved["stage2_checkpoint"]),
                "stage2_last.pt": file_sha256(resolved["stage2_last_checkpoint"]),
                "stage2_train_log.csv": file_sha256(resolved["stage2_train_log"]),
                "stage2_eval_metrics.csv": file_sha256(resolved["stage2_eval_metrics"]),
                "stage2_manifest.json": file_sha256(resolved["stage2_manifest"]),
                "final_checkpoint.pt": file_sha256(resolved["final_checkpoint"]),
                "qparams.csv": file_sha256(qparams_csv),
                "int4_calibrated_student_checkpoint.pt": file_sha256(resolved["int4_checkpoint"]),
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
    write_run_manifests({"seed": resolved["seed_dir"], **_export_dirs(resolved["export_root"], str(args.export_name))}, payload)
    print(f"[innovation-two-stage] stage1_checkpoint={resolved['stage1_checkpoint']}")
    print(f"[innovation-two-stage] stage2_checkpoint={resolved['stage2_checkpoint']}")
    print(f"[innovation-two-stage] final_checkpoint={resolved['final_checkpoint']}")
    print(f"[innovation-two-stage] run_manifest_hash={payload['run_manifest_hash']}")
    return {"manifest": payload, "resolved": resolved, "splits": split_results}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_two_stage_decoupling(args)


if __name__ == "__main__":
    main()
