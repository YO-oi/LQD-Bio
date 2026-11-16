from __future__ import annotations

import argparse
import csv
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
from src.baseline.baseline_metrics import SELECTION_MACRO_F1_METRIC, compute_baseline_metrics, correct_logit_margin
from src.baseline.export_eval import DEFAULT_POOLING, forward_export_batch, load_trained_checkpoint
from src.baseline.manifest import file_sha256, path_str
from src.baseline.quant_checkpoint import load_qparams_csv, qparams_hash
from src.metrics.common_metrics import DEFAULT_EPS, embedding_distortion, tail_and_few_damage_gaps
from src.metrics.geometry_metrics import class_trace_variance, compute_group_fisher, compute_sample_margin
from src.train.train_beats_classifier_head import (
    build_loader,
    dataset_paths,
    load_label_mapping,
    num_predictor_classes,
    validate_split_labels,
)
from src.innovation.stage2_class_prior_adjustment import build_train_label_buckets, validate_split_bucket_coverage


PROTOCOL_VERSION = "innovation_geometry_milestones.v1"
EMBEDDING_SOURCE = "pooled_embedding_before_classifier"
POOLING = DEFAULT_POOLING
SPLIT_NAME = "eval"


@dataclass(frozen=True)
class MilestoneSpec:
    name: str
    model_kind: str
    checkpoint_path: Path
    qparams_csv: Path | None = None


@dataclass(frozen=True)
class ForwardCache:
    sample_ids: list[str]
    labels: Tensor
    buckets: list[str]
    logits: Tensor
    embeddings: Tensor
    checkpoint_payload: Mapping[str, Any]


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


def _dataset_root(dataset_name: str | None) -> Path:
    return dataset_paths(dataset_name, None).root


def _default_output_csv(dataset_name: str | None, seed: int) -> Path:
    return _dataset_root(dataset_name) / "innovation" / "geometry_milestones" / f"seed_{int(seed)}" / "geometry_milestone_metrics.csv"


def _default_fp32_teacher_checkpoint(dataset_name: str | None, seed: int) -> Path:
    return (
        _dataset_root(dataset_name)
        / "baseline"
        / "teacher"
        / f"seed_{int(seed)}"
        / "checkpoints"
        / f"fp32_teacher_seed_{int(seed)}.pt"
    )


def _default_int4_init_checkpoint(dataset_name: str | None, seed: int) -> Path:
    return (
        _dataset_root(dataset_name)
        / "baseline"
        / "student"
        / f"seed_{int(seed)}"
        / "checkpoints"
        / f"int4_student_calibrated_seed_{int(seed)}.pt"
    )


def _default_qparams_csv(dataset_name: str | None, seed: int) -> Path:
    return (
        _dataset_root(dataset_name)
        / "baseline"
        / "student"
        / f"seed_{int(seed)}"
        / "qparams"
        / "int4_qparams.csv"
    )


def _default_stage1_checkpoint(dataset_name: str | None, seed: int) -> Path:
    return _dataset_root(dataset_name) / "innovation" / "two_stage" / f"seed_{int(seed)}" / "stage1_checkpoint.pt"


def _default_two_stage_final_checkpoint(dataset_name: str | None, seed: int) -> Path:
    return _dataset_root(dataset_name) / "innovation" / "two_stage" / f"seed_{int(seed)}" / "final_checkpoint.pt"


def _default_vanilla_kd_checkpoint(dataset_name: str | None, seed: int) -> Path:
    return (
        _dataset_root(dataset_name)
        / "baseline"
        / "kd"
        / f"seed_{int(seed)}"
        / "checkpoints"
        / f"vanilla_kd_best_seed_{int(seed)}.pt"
    )


def _resolve_checkpoint_path(path: str | Path, *, label: str) -> Path:
    target = Path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"{label} checkpoint path must be a concrete .pt file, not a directory: {target}")
    if target.suffix.lower() != ".pt":
        raise ValueError(f"{label} checkpoint path must be a concrete .pt file: {target}")
    return target


def _resolve_csv_path(path: str | Path, *, label: str) -> Path:
    target = Path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"{label} path must be a concrete .csv file, not a directory: {target}")
    if target.suffix.lower() != ".csv":
        raise ValueError(f"{label} path must be a concrete .csv file: {target}")
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export geometry milestone metrics on the full eval split. "
            "This script only runs inference and writes geometry_milestone_metrics.csv."
        )
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fp32-teacher-checkpoint", default=None)
    parser.add_argument("--int4-init-checkpoint", default=None)
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--two-stage-final-checkpoint", default=None)
    parser.add_argument("--vanilla-kd-checkpoint", default=None)
    parser.add_argument("--qparams-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    seed = int(args.seed)
    return {
        "fp32_teacher_checkpoint": _resolve_checkpoint_path(
            args.fp32_teacher_checkpoint or _default_fp32_teacher_checkpoint(args.dataset, seed),
            label="FP32 teacher reference",
        ),
        "int4_init_checkpoint": _resolve_checkpoint_path(
            args.int4_init_checkpoint or _default_int4_init_checkpoint(args.dataset, seed),
            label="INT4 init",
        ),
        "stage1_checkpoint": _resolve_checkpoint_path(
            args.stage1_checkpoint or _default_stage1_checkpoint(args.dataset, seed),
            label="Stage1 best",
        ),
        "two_stage_final_checkpoint": _resolve_checkpoint_path(
            args.two_stage_final_checkpoint or _default_two_stage_final_checkpoint(args.dataset, seed),
            label="Two-Stage final",
        ),
        "vanilla_kd_checkpoint": _resolve_checkpoint_path(
            args.vanilla_kd_checkpoint or _default_vanilla_kd_checkpoint(args.dataset, seed),
            label="Vanilla KD best",
        ),
        "qparams_csv": _resolve_csv_path(
            args.qparams_csv or _default_qparams_csv(args.dataset, seed),
            label="INT4 qparams CSV",
        ),
        "output_csv": Path(args.output_csv) if args.output_csv else _default_output_csv(args.dataset, seed),
    }


def _milestones(resolved: Mapping[str, Path]) -> tuple[MilestoneSpec, ...]:
    return (
        MilestoneSpec(
            name="fp32_teacher_reference",
            model_kind="fp32_teacher",
            checkpoint_path=Path(resolved["fp32_teacher_checkpoint"]),
        ),
        MilestoneSpec(
            name="int4_init_checkpoint",
            model_kind="int4_student_init",
            checkpoint_path=Path(resolved["int4_init_checkpoint"]),
            qparams_csv=Path(resolved["qparams_csv"]),
        ),
        MilestoneSpec(
            name="stage1_best_checkpoint",
            model_kind="two_stage_stage1_best",
            checkpoint_path=Path(resolved["stage1_checkpoint"]),
            qparams_csv=Path(resolved["qparams_csv"]),
        ),
        MilestoneSpec(
            name="two_stage_final_checkpoint",
            model_kind="two_stage_decoupling",
            checkpoint_path=Path(resolved["two_stage_final_checkpoint"]),
            qparams_csv=Path(resolved["qparams_csv"]),
        ),
        MilestoneSpec(
            name="vanilla_kd_best_checkpoint",
            model_kind="vanilla_kd",
            checkpoint_path=Path(resolved["vanilla_kd_checkpoint"]),
            qparams_csv=Path(resolved["qparams_csv"]),
        ),
    )


def _batch_sample_ids(batch: Mapping[str, Any]) -> list[str]:
    raw = batch.get("sample_id")
    if raw is None:
        raise KeyError("eval batch missing sample_id; milestone export requires stable sample order")
    if isinstance(raw, Tensor):
        return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    return [str(raw)]


def _buckets_for_labels(labels: Tensor, label_buckets: Mapping[int, str]) -> list[str]:
    buckets: list[str] = []
    missing: set[int] = set()
    for value in labels.detach().cpu().long().reshape(-1).tolist():
        label_id = int(value)
        bucket = label_buckets.get(label_id)
        if bucket is None:
            missing.add(label_id)
        else:
            buckets.append(str(bucket))
    if missing:
        raise ValueError(f"eval split contains labels missing from train-derived bucket map: {sorted(missing)[:8]}")
    return buckets


def _checkpoint_model_kind(payload: Mapping[str, Any]) -> str:
    value = payload.get("model_kind")
    return "" if value is None else str(value).strip()


def _loader_inferred_model_kind(payload: Mapping[str, Any]) -> str:
    value = payload.get("_baseline_model_kind")
    return "" if value is None else str(value).strip()


def _collect_forward_cache(
    *,
    spec: MilestoneSpec,
    paths: Any,
    loader: Any,
    label_buckets: Mapping[int, str],
    device: torch.device,
) -> ForwardCache:
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(f"{spec.name} checkpoint not found: {spec.checkpoint_path}")
    model, checkpoint_payload = load_trained_checkpoint(
        spec.checkpoint_path,
        device=device,
        qparams_csv_override=spec.qparams_csv,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None

    sample_ids: list[str] = []
    buckets: list[str] = []
    label_chunks: list[Tensor] = []
    logit_chunks: list[Tensor] = []
    embedding_chunks: list[Tensor] = []
    seen: set[str] = set()

    for batch in loader:
        batch_sample_ids = _batch_sample_ids(batch)
        duplicates = sorted(set(batch_sample_ids) & seen)
        if duplicates:
            raise ValueError(f"duplicate sample_id values during eval milestone export: {duplicates[:8]}")
        seen.update(batch_sample_ids)

        labels = batch["label_id"].detach().cpu().long().reshape(-1)
        if int(labels.numel()) != len(batch_sample_ids):
            raise ValueError(f"eval batch label count differs from sample_id count for {spec.name}")
        batch_buckets = _buckets_for_labels(labels, label_buckets)

        logits, embeddings, _token_padding_mask, _valid_token_count, records, oer_activations = forward_export_batch(
            model,
            batch,
            sample_ids=batch_sample_ids,
            token_layer_indices=(),
            device=device,
            oer_mode="none",
            oer_hook_list=(),
        )
        if records:
            raise RuntimeError("geometry milestone export unexpectedly collected token trace records")
        if oer_activations:
            raise RuntimeError("geometry milestone export unexpectedly collected OER activations")

        sample_ids.extend(batch_sample_ids)
        buckets.extend(batch_buckets)
        label_chunks.append(labels.contiguous())
        logit_chunks.append(logits.detach().cpu().float().contiguous())
        embedding_chunks.append(embeddings.detach().cpu().float().contiguous())

    if not sample_ids:
        raise RuntimeError("eval loader produced no samples for geometry milestone export")
    return ForwardCache(
        sample_ids=sample_ids,
        labels=torch.cat(label_chunks, dim=0).long().contiguous(),
        buckets=buckets,
        logits=torch.cat(logit_chunks, dim=0).float().contiguous(),
        embeddings=torch.cat(embedding_chunks, dim=0).float().contiguous(),
        checkpoint_payload=dict(checkpoint_payload),
    )


def _finite_values(values: Any) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    return tensor[torch.isfinite(tensor)]


def _finite_mean(values: Any) -> float:
    finite = _finite_values(values)
    if int(finite.numel()) == 0:
        return float("nan")
    return float(finite.mean().item())


def _finite_std(values: Any) -> float:
    finite = _finite_values(values)
    if int(finite.numel()) <= 1:
        return 0.0 if int(finite.numel()) == 1 else float("nan")
    return float(finite.std(unbiased=False).item())


def _finite_median(values: Any) -> float:
    finite = _finite_values(values)
    if int(finite.numel()) == 0:
        return float("nan")
    return float(torch.median(finite).item())


def _finite_p95(values: Any) -> float:
    finite = _finite_values(values)
    if int(finite.numel()) == 0:
        return float("nan")
    return float(torch.quantile(finite, 0.95).item())


def _safe_ratio(numerator: float, denominator: float, *, eps: float) -> float:
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return float("nan")
    return float(numerator) / (float(denominator) + float(eps))


def _scalar_float(value: Any) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if int(tensor.numel()) != 1:
        raise ValueError(f"expected scalar tensor, got shape={tuple(torch.as_tensor(value).shape)}")
    return float(tensor[0].item())


def _aligned_or_raise(reference: ForwardCache, current: ForwardCache, *, milestone_name: str) -> None:
    if reference.sample_ids != current.sample_ids:
        bad = [
            index
            for index, (left, right) in enumerate(zip(reference.sample_ids, current.sample_ids))
            if str(left) != str(right)
        ][:8]
        raise ValueError(f"{milestone_name} eval sample order differs from FP32 reference; mismatch_indices={bad}")
    if not torch.equal(reference.labels, current.labels):
        bad = torch.nonzero(reference.labels.reshape(-1) != current.labels.reshape(-1), as_tuple=False).reshape(-1)[:8]
        raise ValueError(f"{milestone_name} eval labels differ from FP32 reference; mismatch_indices={bad.tolist()}")
    if list(reference.buckets) != list(current.buckets):
        bad = [index for index, (left, right) in enumerate(zip(reference.buckets, current.buckets)) if left != right][:8]
        raise ValueError(f"{milestone_name} eval buckets differ from FP32 reference; mismatch_indices={bad}")
    if tuple(reference.embeddings.shape) != tuple(current.embeddings.shape):
        raise ValueError(
            f"{milestone_name} embedding shape differs from FP32 reference: "
            f"{tuple(current.embeddings.shape)} vs {tuple(reference.embeddings.shape)}"
        )
    if tuple(reference.logits.shape) != tuple(current.logits.shape):
        raise ValueError(
            f"{milestone_name} logits shape differs from FP32 reference: "
            f"{tuple(current.logits.shape)} vs {tuple(reference.logits.shape)}"
        )


def _variance_metrics(reference: ForwardCache, current: ForwardCache, *, eps: float) -> dict[str, Any]:
    ref_var, ref_details = class_trace_variance(
        reference.embeddings,
        reference.labels,
        normalization="dof",
        reduction="none",
        return_details=True,
    )
    cur_var, cur_details = class_trace_variance(
        current.embeddings,
        current.labels,
        normalization="dof",
        reduction="none",
        return_details=True,
    )
    assert isinstance(ref_var, Tensor)
    assert isinstance(cur_var, Tensor)
    ref_class_ids = torch.as_tensor(ref_details["class_ids"]).detach().cpu().long()
    cur_class_ids = torch.as_tensor(cur_details["class_ids"]).detach().cpu().long()
    if not torch.equal(ref_class_ids, cur_class_ids):
        raise ValueError("class_trace_variance class ids differ between FP32 reference and current milestone")

    ratio = cur_var / (ref_var + float(eps))
    ratio = torch.where(
        torch.isfinite(ref_var) & torch.isfinite(cur_var) & torch.isfinite(ratio),
        ratio,
        torch.full_like(ratio, float("nan")),
    )
    deviation = torch.abs(torch.log(ratio))
    deviation = torch.where(
        torch.isfinite(ratio) & (ratio > 0.0) & torch.isfinite(deviation),
        deviation,
        torch.full_like(deviation, float("nan")),
    )
    valid = torch.isfinite(deviation)
    ref_counts = torch.as_tensor(ref_details["counts"]).detach().cpu().long()
    return {
        "variance_ratio_mean": _finite_mean(ratio),
        "variance_ratio_median": _finite_median(ratio),
        "variance_deviation": _finite_mean(deviation),
        "variance_deviation_median": _finite_median(deviation),
        "variance_deviation_p95": _finite_p95(deviation),
        "variance_valid_class_count": int(valid.sum().item()),
        "variance_total_class_count": int(ref_class_ids.numel()),
        "variance_singleton_class_count": int((ref_counts < 2).sum().item()),
        "variance_definition": "class-level Tr(Cov({z_i | y_i = c})) with dof normalization; deviation=abs(log(candidate/(fp32+eps)))",
    }


def _collapse_metrics(distortion_values: Tensor, buckets: Sequence[str]) -> dict[str, Any]:
    try:
        gap_payload = tail_and_few_damage_gaps(distortion_values, list(buckets), higher_is_better=False)
    except ValueError:
        return {
            "many_embedding_distortion_mean": float("nan"),
            "medium_embedding_distortion_mean": float("nan"),
            "few_embedding_distortion_mean": float("nan"),
            "tail_embedding_distortion_mean": float("nan"),
            "tail_collapse_gap": float("nan"),
            "few_collapse_gap": float("nan"),
            "many_sample_count": 0,
            "medium_sample_count": 0,
            "few_sample_count": 0,
            "tail_sample_count": 0,
        }
    return {
        "many_embedding_distortion_mean": _scalar_float(gap_payload["many_value"]),
        "medium_embedding_distortion_mean": _scalar_float(gap_payload["medium_value"]),
        "few_embedding_distortion_mean": _scalar_float(gap_payload["few_value"]),
        "tail_embedding_distortion_mean": _scalar_float(gap_payload["tail_value"]),
        "tail_collapse_gap": _scalar_float(gap_payload["tail_vs_many_gap"]),
        "few_collapse_gap": _scalar_float(gap_payload["few_vs_many_gap"]),
        "many_sample_count": int(_scalar_float(gap_payload["many_count"])),
        "medium_sample_count": int(_scalar_float(gap_payload["medium_count"])),
        "few_sample_count": int(_scalar_float(gap_payload["few_count"])),
        "tail_sample_count": int(_scalar_float(gap_payload["tail_count"])),
    }


def _analysis_mask_from_buckets(buckets: Sequence[str], analysis_group: str) -> Tensor:
    target = str(analysis_group)
    return torch.as_tensor([str(item) == target for item in buckets], dtype=torch.bool)


def _group_metric_row(metrics: Mapping[str, Sequence[Mapping[str, Any]]], *, metric_name: str, analysis_group: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in metrics.get("group_metrics", [])
        if str(row.get("analysis_group")) == str(analysis_group) and str(row.get("metric_name")) == str(metric_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one group metric row for analysis_group={analysis_group!r}, "
            f"metric_name={metric_name!r}; got {len(matches)}"
        )
    return matches[0]


def _few_classification_diagnostics(
    *,
    sample_ids: Sequence[str],
    labels: Tensor,
    logits: Tensor,
    embeddings: Tensor,
    correct_logit_margin_values: Tensor,
    buckets: Sequence[str],
    label_mapping: Mapping[Any, Any],
    label_buckets: Mapping[int, str],
    model_kind: str,
    seed: int,
) -> dict[str, Any]:
    few_mask = _analysis_mask_from_buckets(buckets, "few")
    few_sample_count = int(few_mask.sum().item())
    metrics = compute_baseline_metrics(
        sample_ids=sample_ids,
        labels=labels,
        logits=logits,
        embeddings=embeddings,
        bucket=buckets,
        label_mapping=label_mapping,
        label_buckets=label_buckets,
        split=SPLIT_NAME,
        model_kind=model_kind,
        seed=int(seed),
        pooling=POOLING,
    )
    few_accuracy_row = _group_metric_row(metrics, analysis_group="few", metric_name="accuracy")
    few_f1_row = _group_metric_row(metrics, analysis_group="few", metric_name=SELECTION_MACRO_F1_METRIC)
    if few_sample_count <= 0:
        return {
            "few_accuracy": float(few_accuracy_row.get("metric_value", float("nan"))),
            "few_f1": float(few_f1_row.get("metric_value", float("nan"))),
            "few_f1_effective_class_count": int(few_f1_row.get("effective_class_count", 0)),
            "few_correct_logit_margin": float("nan"),
            "few_classification_source": "src.baseline.baseline_metrics.compute_baseline_metrics.group_metrics",
            "few_classification_definition": "eval samples whose train-derived label bucket is few",
        }

    return {
        "few_accuracy": float(few_accuracy_row["metric_value"]),
        "few_f1": float(few_f1_row["metric_value"]),
        "few_f1_effective_class_count": int(few_f1_row.get("effective_class_count", 0)),
        "few_correct_logit_margin": _finite_mean(correct_logit_margin_values.detach().cpu().float()[few_mask]),
        "few_classification_source": "src.baseline.baseline_metrics.compute_baseline_metrics.group_metrics",
        "few_classification_definition": "few_accuracy=group_metrics[analysis_group=few,metric_name=accuracy]; few_f1=group_metrics[analysis_group=few,metric_name=macro_f1_label_universe_nanmean]",
    }


def _metric_row(
    *,
    spec: MilestoneSpec,
    cache: ForwardCache,
    reference: ForwardCache,
    paths: Any,
    label_mapping: Mapping[Any, Any],
    label_buckets: Mapping[int, str],
    seed: int,
    eps: float,
) -> dict[str, Any]:
    _aligned_or_raise(reference, cache, milestone_name=spec.name)
    labels = reference.labels
    ref_embeddings = reference.embeddings
    cur_embeddings = cache.embeddings
    ref_logits = reference.logits
    cur_logits = cache.logits

    ref_fisher = _scalar_float(compute_group_fisher(ref_embeddings, labels, None, normalize_by_dof=False, eps=eps))
    cur_fisher = _scalar_float(compute_group_fisher(cur_embeddings, labels, None, normalize_by_dof=False, eps=eps))

    ref_geo_margin = compute_sample_margin(ref_embeddings, labels, reduction="none")
    cur_geo_margin = compute_sample_margin(cur_embeddings, labels, reduction="none")
    assert isinstance(ref_geo_margin, Tensor)
    assert isinstance(cur_geo_margin, Tensor)
    ref_geo_margin_mean = _finite_mean(ref_geo_margin)
    cur_geo_margin_mean = _finite_mean(cur_geo_margin)

    ref_logit_margin = correct_logit_margin(ref_logits, labels)
    cur_logit_margin = correct_logit_margin(cur_logits, labels)
    ref_logit_margin_mean = _finite_mean(ref_logit_margin)
    cur_logit_margin_mean = _finite_mean(cur_logit_margin)

    distortion_values = embedding_distortion(ref_embeddings, cur_embeddings, reduction="none")
    assert isinstance(distortion_values, Tensor)
    cosine_values = F.cosine_similarity(cur_embeddings.float(), ref_embeddings.float(), dim=1, eps=1e-12)

    checkpoint_hash = file_sha256(spec.checkpoint_path)
    qparams_csv = spec.qparams_csv
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_name": paths.name,
        "split": SPLIT_NAME,
        "subset_policy": "full_eval_csv",
        "seed": int(seed),
        "milestone": spec.name,
        "model_kind": spec.model_kind,
        "model_kind_authority": "milestone_spec",
        "checkpoint_model_kind": _checkpoint_model_kind(cache.checkpoint_payload),
        "checkpoint_model_kind_source": "checkpoint.model_kind",
        "loader_inferred_model_kind": _loader_inferred_model_kind(cache.checkpoint_payload),
        "checkpoint_identity_fields_are_diagnostic": True,
        "checkpoint_path": path_str(spec.checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "qparams_csv": path_str(qparams_csv) if qparams_csv is not None else "",
        "qparams_csv_sha256": file_sha256(qparams_csv) if qparams_csv is not None else "",
        "reference_milestone": "fp32_teacher_reference",
        "reference_checkpoint_path": path_str(Path(paths.output_checkpoint)),
        "reference_checkpoint_sha256": file_sha256(Path(paths.output_checkpoint)),
        "eval_csv": path_str(paths.eval_csv),
        "eval_csv_sha256": file_sha256(paths.eval_csv),
        "train_csv": path_str(paths.train_csv),
        "train_csv_sha256": file_sha256(paths.train_csv),
        "label_mapping": path_str(paths.label_mapping),
        "label_mapping_sha256": file_sha256(paths.label_mapping),
        "num_samples": int(labels.numel()),
        "num_classes_present": int(torch.unique(labels).numel()),
        "embedding_dim": int(cur_embeddings.shape[1]),
        "embedding_source": EMBEDDING_SOURCE,
        "pooling": POOLING,
        "logit_adjustment": "before_logit_adjustment",
        "token_features_saved": False,
        "training_performed": False,
        "backward_performed": False,
        "fisher_abs": cur_fisher,
        "fisher_fp32_reference": ref_fisher,
        "fisher_retention": _safe_ratio(cur_fisher, ref_fisher, eps=eps),
        "fisher_definition": "Tr(S_B)/(Tr(S_W)+eps), raw scatter ratio, normalize_by_dof=False",
        "geo_margin_main": cur_geo_margin_mean,
        "geo_margin_fp32_reference": ref_geo_margin_mean,
        "geo_margin_retention": _safe_ratio(cur_geo_margin_mean, ref_geo_margin_mean, eps=eps),
        "geo_margin_definition": "min_{c != y} ||z - mu_c||_2 - ||z - mu_y||_2",
        "correct_logit_margin": cur_logit_margin_mean,
        "correct_logit_margin_fp32_reference": ref_logit_margin_mean,
        "correct_logit_margin_retention": _safe_ratio(cur_logit_margin_mean, ref_logit_margin_mean, eps=eps),
        "correct_logit_margin_definition": "logit_y - max_{j != y} logit_j",
        "embedding_distortion": _finite_mean(distortion_values),
        "embedding_distortion_std": _finite_std(distortion_values),
        "embedding_distortion_median": _finite_median(distortion_values),
        "embedding_distortion_p95": _finite_p95(distortion_values),
        "embedding_distortion_definition": "mean_i ||E_candidate_i - E_fp32_i||_2 / (||E_fp32_i||_2 + eps)",
        "embedding_cosine_similarity": _finite_mean(cosine_values),
        "embedding_cosine_similarity_std": _finite_std(cosine_values),
    }
    row.update(_variance_metrics(reference, cache, eps=eps))
    row.update(_collapse_metrics(torch.as_tensor(distortion_values), reference.buckets))
    row.update(
        _few_classification_diagnostics(
            sample_ids=cache.sample_ids,
            labels=labels,
            logits=cur_logits,
            embeddings=cur_embeddings,
            correct_logit_margin_values=cur_logit_margin,
            buckets=reference.buckets,
            label_mapping=label_mapping,
            label_buckets=label_buckets,
            model_kind=spec.model_kind,
            seed=int(seed),
        )
    )
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no milestone rows to write: {path}")
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


def _validate_checkpoint_paths(milestones: Sequence[MilestoneSpec]) -> None:
    missing = [f"{item.name}: {item.checkpoint_path}" for item in milestones if not item.checkpoint_path.exists()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"missing required geometry milestone checkpoint(s):\n  {joined}")
    non_files = [f"{item.name}: {item.checkpoint_path}" for item in milestones if not item.checkpoint_path.is_file()]
    if non_files:
        joined = "\n  ".join(non_files)
        raise FileNotFoundError(f"geometry milestone checkpoint path(s) must be concrete files:\n  {joined}")


def _validate_qparams_csv_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required INT4 qparams CSV: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"INT4 qparams CSV path must be a concrete file: {path}")


def _validate_quantized_milestone_qparams(milestones: Sequence[MilestoneSpec]) -> None:
    mismatches: list[str] = []
    payload_errors: list[str] = []
    expected_hash_cache: dict[Path, str] = {}

    for item in milestones:
        if item.qparams_csv is None:
            continue
        qparams_csv = item.qparams_csv
        if qparams_csv not in expected_hash_cache:
            expected_hash_cache[qparams_csv] = qparams_hash(load_qparams_csv(qparams_csv))
        expected_hash = expected_hash_cache[qparams_csv]

        checkpoint = torch.load(item.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            payload_errors.append(f"{item.name}: checkpoint payload is not a mapping: {type(checkpoint)!r}")
            continue

        stored_hash = checkpoint.get("qparams_hash")
        qparams_payload = checkpoint.get("qparams")
        if not stored_hash:
            payload_errors.append(f"{item.name}: checkpoint is missing qparams_hash")
            continue
        if not isinstance(qparams_payload, Mapping) or not qparams_payload:
            payload_errors.append(f"{item.name}: checkpoint is missing non-empty qparams payload")
            continue

        computed_hash = qparams_hash(qparams_payload)
        if computed_hash != str(stored_hash):
            payload_errors.append(
                f"{item.name}: checkpoint qparams payload hash mismatch, stored={stored_hash}, computed={computed_hash}"
            )
            continue
        if str(stored_hash) != expected_hash:
            recorded_csv = checkpoint.get("qparams_csv", "")
            mismatches.append(
                f"{item.name}: checkpoint_qparams_hash={stored_hash}, expected_qparams_hash={expected_hash}, "
                f"checkpoint={item.checkpoint_path}, qparams_csv={qparams_csv}, recorded_qparams_csv={recorded_csv}"
            )

    if payload_errors:
        joined = "\n  ".join(payload_errors)
        raise RuntimeError(f"invalid quantized milestone qparams payload(s):\n  {joined}")
    if mismatches:
        joined = "\n  ".join(mismatches)
        raise RuntimeError(
            "geometry milestone qparams mismatch; all quantized milestones must match the explicit INT4 qparams CSV. "
            "Rerun the mismatched checkpoint from the same INT4 init/qparams before exporting comparable milestones:\n  "
            f"{joined}"
        )


def run_geometry_milestones(args: argparse.Namespace) -> dict[str, Any]:
    resolved = _resolved_paths(args)
    milestones = _milestones(resolved)
    paths = dataset_paths(args.dataset, str(resolved["fp32_teacher_checkpoint"]))

    label_mapping = load_label_mapping(paths.label_mapping)
    num_classes = num_predictor_classes(label_mapping)
    validate_split_labels(paths.train_csv, num_classes=num_classes, name="train")
    validate_split_labels(paths.eval_csv, num_classes=num_classes, name="eval")
    label_buckets = build_train_label_buckets(paths.train_csv)
    validate_split_bucket_coverage(paths.eval_csv, label_buckets, split_name="eval")

    if bool(args.dry_run):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "dataset_name": paths.name,
            "split": SPLIT_NAME,
            "train_csv": path_str(paths.train_csv),
            "eval_csv": path_str(paths.eval_csv),
            "qparams_csv": path_str(resolved["qparams_csv"]),
            "qparams_csv_exists": Path(resolved["qparams_csv"]).exists(),
            "output_csv": path_str(resolved["output_csv"]),
            "milestones": [
                {
                    "milestone": item.name,
                    "model_kind": item.model_kind,
                    "checkpoint_path": path_str(item.checkpoint_path),
                    "qparams_csv": path_str(item.qparams_csv) if item.qparams_csv is not None else "",
                    "exists": item.checkpoint_path.exists(),
                }
                for item in milestones
            ],
            "test_split_touched": False,
            "full_csv_touched": False,
        }

    _validate_checkpoint_paths(milestones)
    _validate_qparams_csv_path(Path(resolved["qparams_csv"]))
    _validate_quantized_milestone_qparams(milestones)
    device = torch.device(args.device or getattr(project_config, "DEVICE", "cpu"))
    loader = build_loader(
        paths,
        split=SPLIT_NAME,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
    )

    caches: dict[str, ForwardCache] = {}
    for spec in milestones:
        caches[spec.name] = _collect_forward_cache(
            spec=spec,
            paths=paths,
            loader=loader,
            label_buckets=label_buckets,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference = caches["fp32_teacher_reference"]
    rows = [
        _metric_row(
            spec=spec,
            cache=caches[spec.name],
            reference=reference,
            paths=paths,
            label_mapping=label_mapping,
            label_buckets=label_buckets,
            seed=int(args.seed),
            eps=float(DEFAULT_EPS),
        )
        for spec in milestones
    ]
    _write_csv(resolved["output_csv"], rows)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_name": paths.name,
        "split": SPLIT_NAME,
        "output_csv": path_str(resolved["output_csv"]),
        "output_csv_sha256": file_sha256(resolved["output_csv"]),
        "num_rows": len(rows),
        "num_samples": int(reference.labels.numel()),
        "test_split_touched": False,
        "full_csv_touched": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_geometry_milestones(args)
    if bool(args.dry_run):
        import json

        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"[geometry-milestones] metrics_csv={result['output_csv']}")
    print(f"[geometry-milestones] metrics_csv_sha256={result['output_csv_sha256']}")


if __name__ == "__main__":
    main()
