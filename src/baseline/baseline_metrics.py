from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.metrics.common_metrics import (
    DEFAULT_MARGIN_MIN_ABS,
    collapse_gap,
    embedding_distortion,
    fisher_retention,
    margin_retention,
    margin_value_retention,
    scalar_oer_retention,
)
from src.metrics.geometry_metrics import class_trace_variance, compute_group_fisher, compute_sample_margin
from src.utils.group_utils import AGGREGATION_FNS, ANALYSIS_GROUPS, TAIL_BUCKETS, analysis_group_mask, validate_bucket_values


BASELINE_METRIC_TABLES = (
    "metrics",
    "main_metrics",
    "group_metrics",
    "per_class_metrics",
    "geometry_metrics",
    "retention_metrics",
)
GEOMETRY_METRIC_NAMES = {
    "fisher",
    "geo_margin_main",
    "correct_logit_margin_detail",
    "top1_top2_gap_detail",
    "class_trace_variance",
    "class_covariance_trace",
}
CANONICAL_METRIC_COLUMNS = (
    "model_kind",
    "split",
    "seed",
    "metric_name",
    "analysis_group",
    "metric_value",
    "metric_direction",
    "aggregation_unit",
    "aggregation_fn",
    "metric_role",
    "support_count",
    "effective_class_count",
    "reference_checkpoint_hash",
    "run_identity_hash",
)
SELECTION_MACRO_F1_METRIC = "macro_f1_label_universe_nanmean"
LEGACY_MACRO_F1_METRIC = "macro_f1_legacy_zero_absent_diagnostic"


@dataclass(frozen=True)
class ReferenceBundle:
    embeddings: Any
    logits: Any
    sample_ids: Sequence[Any]
    labels: Any
    label_mapping_hash: str
    split_hash: str
    sample_order_hash: str
    pooling: str
    model_kind: str = "fp32_teacher"
    checkpoint_hash: str = ""
    embedding_dim: int | None = None


@dataclass(frozen=True)
class _MetricContext:
    split: str
    model_kind: str
    seed: int | None
    run_identity_hash: str
    reference_checkpoint_hash: str
    label_mapping_hash: str
    split_hash: str
    sample_order_hash: str
    pooling: str


@dataclass(frozen=True)
class _ReferenceTensors:
    embeddings: Tensor
    logits: Tensor


def _is_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, Tensor):
        return value.dtype == torch.bool
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Sequence):
        return any(_is_bool_like(item) for item in value)
    return False


def _as_float_cpu(value: Any, *, name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().cpu().float()
    if tensor.numel() <= 0:
        raise ValueError(f"{name} must not be empty")
    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        bad_count = int((~finite).sum().item())
        raise ValueError(f"{name} contains {bad_count} NaN/Inf value(s); exported tensors must be finite")
    return tensor


def _require_finite_tensor(tensor: Tensor, *, name: str) -> None:
    finite = torch.isfinite(tensor)
    if not bool(finite.all().item()):
        bad_count = int((~finite).sum().item())
        raise ValueError(f"{name} contains {bad_count} NaN/Inf value(s)")


def _numeric_tensor(value: Any, *, name: str) -> Tensor:
    if _is_bool_like(value):
        raise ValueError(f"{name} must be numeric integer-like values, not bool")
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception:
            try:
                tensor = torch.tensor([float(str(item).strip()) for item in value], dtype=torch.float64)
            except Exception as exc:
                raise ValueError(f"{name} must be numeric integer-like values") from exc
    if tensor.dtype == torch.bool:
        raise ValueError(f"{name} must be numeric integer-like values, not bool")
    if tensor.numel() <= 0:
        raise ValueError(f"{name} must not be empty")
    try:
        numeric = tensor.detach().cpu().to(dtype=torch.float64).reshape(-1)
    except RuntimeError as exc:
        raise ValueError(f"{name} must be numeric integer-like values") from exc
    finite = torch.isfinite(numeric)
    if not bool(finite.all().item()):
        bad_count = int((~finite).sum().item())
        raise ValueError(f"{name} contains {bad_count} NaN/Inf value(s)")
    rounded = numeric.round()
    if not bool(torch.equal(numeric, rounded)):
        bad = numeric[numeric != rounded][:8].tolist()
        raise ValueError(f"{name} must be integer-like before casting; examples={bad}")
    return rounded.to(dtype=torch.long)


def _as_label_tensor(value: Any, *, name: str, num_classes: int) -> Tensor:
    tensor = _numeric_tensor(value, name=name)
    if int(tensor.min().item()) < 0 or int(tensor.max().item()) >= int(num_classes):
        raise ValueError(f"{name} contain values outside [0, {int(num_classes) - 1}]")
    return tensor


def _finite_mean(values: Tensor) -> float:
    x = values.detach().float().reshape(-1)
    finite = torch.isfinite(x)
    if not bool(finite.any().item()):
        return float("nan")
    return float(x[finite].mean().item())


def _nanmean(values: Sequence[float]) -> tuple[float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), 0
    return sum(finite) / float(len(finite)), len(finite)


def _scalar_float(value: Any, *, name: str = "value") -> float:
    try:
        tensor = torch.as_tensor(value).detach().float().reshape(-1)
    except Exception:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be a scalar, got shape={tuple(torch.as_tensor(value).shape)}")
    result = float(tensor[0].item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sample_order_hash(sample_ids: Sequence[Any]) -> str:
    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_label_mapping(label_mapping: Mapping[Any, Any] | None) -> dict[str, int]:
    if not isinstance(label_mapping, Mapping) or not label_mapping:
        raise ValueError("compute_baseline_metrics requires a complete label_mapping for label-universe metrics")
    result = {str(label_name): int(label_id) for label_name, label_id in label_mapping.items()}
    ids = sorted(result.values())
    expected = list(range(len(ids)))
    if ids != expected or len(set(ids)) != len(ids):
        raise ValueError(f"label_mapping ids must be dense 0..C-1, observed={ids}, expected={expected}")
    return result


def _label_names_by_id(label_mapping: Mapping[str, int]) -> dict[int, str]:
    return {int(label_id): str(label_name) for label_name, label_id in label_mapping.items()}


def _analysis_mask(buckets: Sequence[str], analysis_group: str) -> Tensor:
    if analysis_group == "global":
        return torch.ones(len(buckets), dtype=torch.bool)
    mask = analysis_group_mask([str(item) for item in buckets], str(analysis_group))
    return torch.tensor([bool(item) for item in mask], dtype=torch.bool)


def _label_ids_for_group(label_ids: Sequence[int], label_buckets: Mapping[int, str], group_name: str) -> list[int]:
    if group_name == "global":
        return [int(item) for item in label_ids]
    if group_name == "tail":
        return [int(label_id) for label_id in label_ids if str(label_buckets.get(int(label_id), "")) in set(TAIL_BUCKETS)]
    return [int(label_id) for label_id in label_ids if str(label_buckets.get(int(label_id), "")) == str(group_name)]


def _unique_label_count(labels: Tensor, mask: Tensor) -> int:
    if not bool(mask.any().item()):
        return 0
    return int(torch.unique(labels[mask].detach().cpu().long()).numel())


def _validate_inputs(
    *,
    sample_ids: Sequence[Any],
    labels: Tensor,
    predictions: Tensor,
    logits: Tensor,
    embeddings: Tensor,
    buckets: Sequence[str],
) -> None:
    n = int(labels.numel())
    if len(sample_ids) != n:
        raise ValueError(f"sample_ids length must match labels, got {len(sample_ids)} vs {n}")
    if len(buckets) != n:
        raise ValueError(f"bucket length must match labels, got {len(buckets)} vs {n}")
    if int(predictions.numel()) != n:
        raise ValueError(f"predictions length must match labels, got {int(predictions.numel())} vs {n}")
    if logits.ndim != 2 or int(logits.shape[0]) != n:
        raise ValueError(f"logits must be [N,C] with N={n}, got shape={tuple(logits.shape)}")
    if embeddings.ndim != 2 or int(embeddings.shape[0]) != n:
        raise ValueError(f"embeddings must be [N,D] with N={n}, got shape={tuple(embeddings.shape)}")


def _validate_external_predictions(external: Any | None, *, derived: Tensor, num_classes: int) -> None:
    if external is None:
        return
    audited = _as_label_tensor(external, name="external_predictions", num_classes=num_classes)
    if tuple(audited.shape) != tuple(derived.reshape(-1).shape):
        raise ValueError(f"external_predictions shape must match logits.argmax, got {tuple(audited.shape)} vs {tuple(derived.shape)}")
    if not bool(torch.equal(audited.reshape(-1), derived.reshape(-1).cpu().long())):
        mismatch = torch.nonzero(audited.reshape(-1) != derived.reshape(-1).cpu().long(), as_tuple=False).reshape(-1)[:8].tolist()
        raise ValueError(f"external predictions differ from logits.argmax(dim=1); mismatch indices={mismatch}")


def _canonical_label_buckets(label_buckets: Mapping[int, str] | None, label_ids: Sequence[int]) -> dict[int, str]:
    if not isinstance(label_buckets, Mapping) or not label_buckets:
        raise ValueError("label_buckets must be provided from train split train_label_n; sample bucket is audit-only")
    result = {int(label_id): str(bucket) for label_id, bucket in label_buckets.items()}
    missing = sorted(set(int(item) for item in label_ids) - set(result))
    if missing:
        raise ValueError(f"label_buckets missing label ids from label_mapping: {missing[:8]}")
    validate_bucket_values([result[int(label_id)] for label_id in label_ids], context="label_buckets")
    return result


def _audit_sample_buckets(labels: Tensor, buckets: Sequence[str], label_buckets: Mapping[int, str]) -> list[str]:
    canonical = [str(label_buckets[int(label_id)]) for label_id in labels.detach().cpu().long().tolist()]
    observed = validate_bucket_values([str(item) for item in buckets], context="sample bucket audit")
    mismatches = [
        (index, int(label_id), str(observed_bucket), str(expected_bucket))
        for index, (label_id, observed_bucket, expected_bucket) in enumerate(zip(labels.tolist(), observed, canonical))
        if str(observed_bucket) != str(expected_bucket)
    ]
    if mismatches:
        preview = "; ".join(
            f"idx={idx}: label_id={label_id} bucket={observed} expected={expected}"
            for idx, label_id, observed, expected in mismatches[:8]
        )
        raise ValueError(f"sample bucket differs from canonical train-label bucket; {preview}")
    return canonical


def _per_class_stats(labels: Tensor, predictions: Tensor, label_ids: Sequence[int]) -> dict[int, dict[str, float | int]]:
    label_list = [int(item) for item in labels.detach().cpu().long().tolist()]
    pred_list = [int(item) for item in predictions.detach().cpu().long().tolist()]
    true_count: Counter[int] = Counter(label_list)
    pred_count: Counter[int] = Counter(pred_list)
    correct_count: Counter[int] = Counter()
    for label_id, prediction in zip(label_list, pred_list):
        if int(label_id) == int(prediction):
            correct_count[int(label_id)] += 1
    stats: dict[int, dict[str, float | int]] = {}
    for label_id in [int(item) for item in label_ids]:
        support = int(true_count[label_id])
        predicted = int(pred_count[label_id])
        correct = int(correct_count[label_id])
        false_positive = max(0, predicted - correct)
        false_negative = max(0, support - correct)
        precision = correct / float(correct + false_positive) if (correct + false_positive) > 0 else float("nan")
        recall = correct / float(support) if support > 0 else float("nan")
        f1_nan_absent = (2 * correct) / float((2 * correct) + false_positive + false_negative) if support > 0 else float("nan")
        legacy_denom = (2 * correct) + false_positive + false_negative
        f1_legacy = (2 * correct) / float(legacy_denom) if legacy_denom > 0 else 0.0
        stats[label_id] = {
            "support": support,
            "predicted": predicted,
            "correct": correct,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1_nan_absent": f1_nan_absent,
            "f1_legacy_zero_absent": f1_legacy,
        }
    return stats


def correct_logit_margin(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 2 or int(logits.shape[1]) < 2:
        return logits.new_full((int(logits.shape[0]),), float("nan"))
    labels = labels.to(device=logits.device, dtype=torch.long).reshape(-1)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    rival_logits = logits.clone()
    rival_logits.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - rival_logits.max(dim=1).values


def top1_top2_gap(logits: Tensor) -> Tensor:
    if logits.ndim != 2 or int(logits.shape[1]) < 2:
        return logits.new_full((int(logits.shape[0]),), float("nan"))
    top2 = torch.topk(logits, k=2, dim=1, largest=True, sorted=True).values
    return top2[:, 0] - top2[:, 1]


def _metric_row(
    *,
    context: _MetricContext,
    metric_name: str,
    analysis_group: str,
    metric_value: float,
    metric_direction: str,
    aggregation_unit: str,
    aggregation_fn: str,
    metric_role: str,
    support_count: int,
    effective_class_count: int,
    definition: str,
    **extra: Any,
) -> dict[str, Any]:
    if str(aggregation_fn) not in AGGREGATION_FNS:
        raise ValueError(f"metric_name={metric_name}: aggregation_fn={aggregation_fn!r} is not registered in group_utils.AGGREGATION_FNS")
    row: dict[str, Any] = {
        "model_kind": str(context.model_kind),
        "split": str(context.split),
        "seed": "" if context.seed is None else int(context.seed),
        "metric_name": str(metric_name),
        "analysis_group": str(analysis_group),
        "metric_value": float(metric_value),
        "metric_direction": str(metric_direction),
        "aggregation_unit": str(aggregation_unit),
        "aggregation_fn": str(aggregation_fn),
        "metric_role": str(metric_role),
        "support_count": int(support_count),
        "effective_class_count": int(effective_class_count),
        "reference_checkpoint_hash": str(context.reference_checkpoint_hash),
        "run_identity_hash": str(context.run_identity_hash),
        "definition": str(definition),
        "label_mapping_hash": str(context.label_mapping_hash),
        "split_hash": str(context.split_hash),
        "sample_order_hash": str(context.sample_order_hash),
        "pooling": str(context.pooling),
    }
    row.update(extra)
    return row


def _classification_rows(
    *,
    context: _MetricContext,
    labels: Tensor,
    predictions: Tensor,
    logits: Tensor,
    buckets: Sequence[str],
    label_ids: Sequence[int],
    label_buckets: Mapping[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    correct = (predictions == labels).float()
    main_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for group_name in ("global", *ANALYSIS_GROUPS):
        mask = _analysis_mask(buckets, group_name)
        sample_count = int(mask.sum().item())
        rows = main_rows if group_name == "global" else group_rows
        group_label_ids = _label_ids_for_group(label_ids, label_buckets, group_name)
        stats = _per_class_stats(labels[mask], predictions[mask], group_label_ids)
        observed_f1_values = [float(row["f1_nan_absent"]) for row in stats.values() if int(row["support"]) > 0]
        observed_f1, observed_count = _nanmean(observed_f1_values)
        universe_f1_values = [float(row["f1_nan_absent"]) for row in stats.values()]
        universe_f1, universe_effective_count = _nanmean(universe_f1_values)
        legacy_f1_values = [float(row["f1_legacy_zero_absent"]) for row in stats.values()]
        legacy_f1, legacy_effective_count = _nanmean(legacy_f1_values)
        sample_class_count = _unique_label_count(labels, mask)
        for metric_name, value, direction, unit, fn, role, definition, effective_count in (
            (
                "accuracy",
                _finite_mean(correct[mask]),
                "higher_is_better",
                "sample",
                "mean",
                "main",
                "mean(logits.argmax(dim=1) == label_id)",
                sample_class_count,
            ),
            (
                "cross_entropy_loss",
                _finite_mean(ce_loss[mask]),
                "lower_is_better",
                "sample",
                "mean",
                "main",
                "plain CrossEntropyLoss per-sample mean",
                sample_class_count,
            ),
            (
                "macro_f1_observed_classes",
                observed_f1,
                "higher_is_better",
                "class",
                "macro_mean_by_class",
                "main",
                "macro F1 over label classes with support > 0 in this analysis group",
                observed_count,
            ),
            (
                SELECTION_MACRO_F1_METRIC,
                universe_f1,
                "higher_is_better",
                "class",
                "nanmean",
                "main",
                "macro F1 over the complete label universe; classes with support=0 are NaN and skipped by nanmean",
                universe_effective_count,
            ),
            (
                LEGACY_MACRO_F1_METRIC,
                legacy_f1,
                "diagnostic",
                "class",
                "macro_mean_by_class",
                "detail",
                "legacy macro F1 over the complete label universe with absent classes forced to 0.0; not a selection metric",
                legacy_effective_count,
            ),
        ):
            rows.append(
                _metric_row(
                    context=context,
                    metric_name=metric_name,
                    analysis_group=group_name,
                    metric_value=float(value),
                    metric_direction=direction,
                    aggregation_unit=unit,
                    aggregation_fn=fn,
                    metric_role=role,
                    support_count=sample_count,
                    effective_class_count=int(effective_count),
                    definition=definition,
                    label_universe_class_count=int(len(group_label_ids)),
                )
            )
    return main_rows, group_rows


def _per_class_rows(
    *,
    context: _MetricContext,
    labels: Tensor,
    predictions: Tensor,
    label_ids: Sequence[int],
    label_names: Mapping[int, str],
    label_buckets: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = _per_class_stats(labels, predictions, label_ids)
    for label_id in [int(item) for item in label_ids]:
        item = stats[label_id]
        support = int(item["support"])
        recall = float(item["recall"])
        rows.append(
            _metric_row(
                context=context,
                metric_name="per_class_recall",
                analysis_group=f"class:{label_id}",
                metric_value=recall,
                metric_direction="higher_is_better",
                aggregation_unit="class",
                aggregation_fn="custom",
                metric_role="main",
                support_count=support,
                effective_class_count=1 if support > 0 else 0,
                definition="correct predictions within this label_id divided by class support; support=0 emits NaN",
                label_id=label_id,
                label_name=str(label_names.get(label_id, str(label_id))),
                bucket=str(label_buckets.get(label_id, "unknown")),
                correct=int(item["correct"]),
                predicted=int(item["predicted"]),
                false_positive=int(item["false_positive"]),
                false_negative=int(item["false_negative"]),
                support=support,
                precision=float(item["precision"]),
                recall=recall,
                f1=float(item["f1_nan_absent"]),
                f1_legacy_zero_absent_diagnostic=float(item["f1_legacy_zero_absent"]),
            )
        )
    return rows


def _geometry_rows(
    *,
    context: _MetricContext,
    labels: Tensor,
    logits: Tensor,
    embeddings: Tensor,
    buckets: Sequence[str],
    label_buckets: Mapping[int, str],
) -> list[dict[str, Any]]:
    sample_margins, _margin_details = compute_sample_margin(embeddings, labels, reduction="none", return_details=True)
    assert isinstance(sample_margins, Tensor)
    correct_margin = correct_logit_margin(logits, labels)
    top_gap = top1_top2_gap(logits)
    class_variance, variance_details = class_trace_variance(
        embeddings,
        labels,
        normalization="dof",
        reduction="none",
        return_details=True,
    )
    assert isinstance(class_variance, Tensor)
    class_ids = variance_details["class_ids"]
    counts = variance_details["counts"]
    assert isinstance(class_ids, Tensor)
    assert isinstance(counts, Tensor)
    class_buckets = [str(label_buckets.get(int(class_id), "unknown")) for class_id in class_ids.detach().cpu().long().tolist()]

    rows: list[dict[str, Any]] = []
    for group_name in ("global", *ANALYSIS_GROUPS):
        sample_mask = _analysis_mask(buckets, group_name)
        class_mask = torch.ones(int(class_variance.numel()), dtype=torch.bool) if group_name == "global" else _analysis_mask(class_buckets, group_name)
        sample_count = int(sample_mask.sum().item())
        sample_class_count = _unique_label_count(labels, sample_mask)
        if group_name == "global":
            fisher_value, fisher_details = compute_group_fisher(embeddings, labels, None, normalize_by_dof=False, return_details=True)
        else:
            fisher_value, fisher_details = compute_group_fisher(
                embeddings,
                labels,
                None,
                bucket=list(buckets),
                analysis_group=group_name,
                normalize_by_dof=False,
                return_details=True,
            )
        fisher_class_count = int(torch.as_tensor(fisher_details.get("effective_class_count", sample_class_count)).reshape(()).item())
        try:
            fisher_metric_value = _scalar_float(fisher_value, name="fisher")
        except ValueError:
            if fisher_class_count < 2:
                fisher_metric_value = float("nan")
            else:
                raise
        variance_values = class_variance[class_mask]
        variance_effective_count = int(torch.isfinite(variance_values).sum().item())

        for metric_name, value, role, direction, unit, fn, definition, effective_count in (
            (
                "fisher",
                fisher_metric_value,
                "main",
                "higher_is_better",
                "group_recompute",
                "fisher_global_anchored_recompute",
                "Tr(S_B)/(Tr(S_W)+eps), raw scatter ratio, normalize_by_dof=False",
                fisher_class_count,
            ),
            (
                "geo_margin_main",
                _finite_mean(sample_margins[sample_mask]),
                "main",
                "higher_is_better",
                "sample",
                "nanmean_valid_only",
                "min_{c != y} ||z - mu_c||_2 - ||z - mu_y||_2",
                sample_class_count,
            ),
            (
                "correct_logit_margin_detail",
                _finite_mean(correct_margin[sample_mask]),
                "detail",
                "higher_is_better",
                "sample",
                "nanmean_valid_only",
                "logit_y - max_{j != y} logit_j",
                sample_class_count,
            ),
            (
                "top1_top2_gap_detail",
                _finite_mean(top_gap[sample_mask]),
                "detail",
                "diagnostic",
                "sample",
                "nanmean_valid_only",
                "top1_logit - top2_logit",
                sample_class_count,
            ),
            (
                "class_trace_variance",
                _finite_mean(variance_values),
                "main",
                "diagnostic",
                "class",
                "macro_mean_by_class",
                "macro_mean_c Tr(Cov({z_i | y_i = c})); singleton classes are NaN",
                variance_effective_count,
            ),
        ):
            rows.append(
                _metric_row(
                    context=context,
                    metric_name=metric_name,
                    analysis_group=group_name,
                    metric_value=float(value),
                    metric_direction=direction,
                    aggregation_unit=unit,
                    aggregation_fn=fn,
                    metric_role=role,
                    support_count=sample_count,
                    effective_class_count=int(effective_count),
                    definition=definition,
                )
            )

    for class_id, trace_value, count in zip(
        class_ids.detach().cpu().long().tolist(),
        class_variance.detach().cpu().float().tolist(),
        counts.detach().cpu().long().tolist(),
    ):
        label_key = int(class_id)
        rows.append(
            _metric_row(
                context=context,
                metric_name="class_covariance_trace",
                analysis_group=f"class:{label_key}",
                metric_value=float(trace_value),
                metric_direction="diagnostic",
                aggregation_unit="class",
                aggregation_fn="custom",
                metric_role="main",
                support_count=int(count),
                effective_class_count=1 if math.isfinite(float(trace_value)) else 0,
                definition="Tr(Cov({z_i | y_i = c})) with dof normalization; singleton classes are NaN",
                label_id=label_key,
                bucket=str(label_buckets.get(label_key, "unknown")),
            )
        )
    return rows


def _validate_reference_bundle(
    reference: ReferenceBundle | None,
    *,
    sample_ids: Sequence[str],
    labels: Tensor,
    logits: Tensor,
    embeddings: Tensor,
    label_mapping_hash: str,
    split_hash: str,
    sample_order_hash: str,
    pooling: str,
    num_classes: int,
) -> _ReferenceTensors | None:
    if reference is None:
        return None
    if str(reference.model_kind) != "fp32_teacher":
        raise ValueError(f"ReferenceBundle.model_kind must be 'fp32_teacher', got {reference.model_kind!r}")
    identity_fields = {
        "current_label_mapping_hash": label_mapping_hash,
        "reference_label_mapping_hash": reference.label_mapping_hash,
        "current_split_hash": split_hash,
        "reference_split_hash": reference.split_hash,
        "current_sample_order_hash": sample_order_hash,
        "reference_sample_order_hash": reference.sample_order_hash,
        "current_pooling": pooling,
        "reference_pooling": reference.pooling,
        "reference_checkpoint_hash": reference.checkpoint_hash,
    }
    missing_identity = [name for name, value in identity_fields.items() if not str(value).strip()]
    if missing_identity:
        raise ValueError(f"ReferenceBundle identity fields must be non-empty: {missing_identity}")
    if reference.embedding_dim is None:
        raise ValueError("ReferenceBundle.embedding_dim must be provided")
    reference_sample_ids = [str(item) for item in reference.sample_ids]
    if list(sample_ids) != reference_sample_ids:
        if len(sample_ids) != len(reference_sample_ids):
            raise ValueError(f"ReferenceBundle sample_ids length mismatch: current={len(sample_ids)} reference={len(reference_sample_ids)}")
        mismatch = [idx for idx, (cur, ref) in enumerate(zip(sample_ids, reference_sample_ids)) if cur != ref][:8]
        raise ValueError(f"ReferenceBundle sample_ids order differs from current export; mismatch indices={mismatch}")
    reference_labels = _as_label_tensor(reference.labels, name="reference_labels", num_classes=num_classes)
    if tuple(reference_labels.shape) != tuple(labels.shape) or not bool(torch.equal(reference_labels, labels.cpu().long())):
        mismatch = torch.nonzero(reference_labels.reshape(-1) != labels.cpu().long().reshape(-1), as_tuple=False).reshape(-1)[:8].tolist()
        raise ValueError(f"ReferenceBundle labels differ from current labels; mismatch indices={mismatch}")
    if str(sample_order_hash) != str(reference.sample_order_hash):
        raise ValueError("current sample_order_hash differs from ReferenceBundle.sample_order_hash")
    if str(label_mapping_hash) != str(reference.label_mapping_hash):
        raise ValueError("current label_mapping_hash differs from ReferenceBundle.label_mapping_hash")
    if str(split_hash) != str(reference.split_hash):
        raise ValueError("current split_hash differs from ReferenceBundle.split_hash")
    if str(pooling) != str(reference.pooling):
        raise ValueError("current pooling differs from ReferenceBundle.pooling")
    ref_embeddings = _as_float_cpu(reference.embeddings, name="reference_embeddings")
    ref_logits = _as_float_cpu(reference.logits, name="reference_logits")
    _require_finite_tensor(ref_embeddings, name="reference_embeddings")
    _require_finite_tensor(ref_logits, name="reference_logits")
    if tuple(ref_embeddings.shape) != tuple(embeddings.shape):
        raise ValueError(f"reference_embeddings must have shape={tuple(embeddings.shape)}, got {tuple(ref_embeddings.shape)}")
    if tuple(ref_logits.shape) != tuple(logits.shape):
        raise ValueError(f"reference_logits must have shape={tuple(logits.shape)}, got {tuple(ref_logits.shape)}")
    if reference.embedding_dim is not None and int(reference.embedding_dim) != int(embeddings.shape[1]):
        raise ValueError(f"ReferenceBundle.embedding_dim={reference.embedding_dim} differs from current embedding_dim={int(embeddings.shape[1])}")
    if int(ref_embeddings.shape[1]) != int(embeddings.shape[1]):
        raise ValueError("reference embedding_dim differs from current embedding_dim")
    return _ReferenceTensors(embeddings=ref_embeddings, logits=ref_logits)


def _oer_record_value(record: Mapping[str, Any], *, index: int, oer_vector_reduce: str | None) -> float:
    if "oer_retention" in record:
        return _scalar_float(record["oer_retention"], name=f"oer_records[{index}].oer_retention")
    if "oer_retention_values" in record:
        if str(record.get("oer_vector_reduce", "")).strip():
            raise ValueError("oer_vector_reduce must be passed to compute_baseline_metrics, not embedded in oer_records")
        reduce = str(oer_vector_reduce or "").strip().lower()
        if reduce != "mean":
            raise ValueError("oer_retention_values requires caller oer_vector_reduce='mean'")
        values = _as_float_cpu(record["oer_retention_values"], name=f"oer_records[{index}].oer_retention_values").reshape(-1)
        finite = torch.isfinite(values)
        if not bool(finite.all().item()):
            bad_count = int((~finite).sum().item())
            raise ValueError(f"oer_records[{index}].oer_retention_values contains {bad_count} NaN/Inf value(s)")
        return float(values.mean().item())
    reference = record.get("reference_activation", record.get("fp32_activation"))
    current = record.get("current_activation", record.get("quant_activation"))
    if reference is None or current is None:
        raise ValueError("oer_records must provide scalar oer_retention, explicit oer_retention_values, or reference/current activation tensors")
    return _scalar_float(
        scalar_oer_retention(
            reference,
            current,
            topk=record.get("topk"),
            topk_ratio=record.get("topk_ratio"),
            topk_indices=record.get("topk_indices"),
            batch_dim=record.get("batch_dim", 0),
            reduction="mean",
        ),
        name=f"oer_records[{index}].scalar_oer_retention",
    )



def _oer_record_float(record: Mapping[str, Any], field: str, *, index: int) -> float:
    value = record.get(field)
    if value is None:
        raise ValueError(f"oer_records[{index}] missing required field {field!r}")
    try:
        tensor = torch.as_tensor(value).detach().float().reshape(-1)
    except Exception as exc:
        raise ValueError(f"oer_records[{index}].{field} must be scalar float or NaN") from exc
    if int(tensor.numel()) != 1:
        raise ValueError(f"oer_records[{index}].{field} must be scalar, got shape={tuple(torch.as_tensor(value).shape)}")
    result = float(tensor[0].item())
    if math.isinf(result):
        raise ValueError(f"oer_records[{index}].{field} must not be Inf")
    return result


def _sample_oer_rows(
    *,
    context: _MetricContext,
    sample_ids: Sequence[str],
    labels: Tensor,
    buckets: Sequence[str],
    oer_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = [dict(item) for item in oer_records]
    if not records:
        raise ValueError("oer_records was passed but is empty; formal OER export must either pass sample-level records or None")
    if len(records) != len(sample_ids) or len(records) != int(labels.numel()) or len(records) != len(buckets):
        raise ValueError("oer_records must be sample-level and match current sample dimension exactly")
    values_by_metric = {
        "oer_retention": [],
        "oer_abs_deviation_from_1": [],
        "oer_valid_rate": [],
    }
    for index, record in enumerate(records):
        record_sample_id = str(record.get("sample_id", ""))
        if record_sample_id != str(sample_ids[index]):
            raise ValueError(f"oer_records[{index}].sample_id={record_sample_id!r} does not match current sample_id={sample_ids[index]!r}")
        record_label = _as_label_tensor([record.get("label_id")], name=f"oer_records[{index}].label_id", num_classes=int(labels.max().item()) + 1)[0]
        if int(record_label.item()) != int(labels[index].item()):
            raise ValueError(f"oer_records[{index}].label_id does not match current labels")
        record_bucket = str(record.get("bucket", ""))
        if record_bucket != str(buckets[index]):
            raise ValueError(f"oer_records[{index}].bucket={record_bucket!r} does not match current bucket={buckets[index]!r}")
        retention = _oer_record_float(record, "oer_retention", index=index)
        abs_deviation = _oer_record_float(record, "oer_abs_deviation_from_1", index=index)
        valid_rate = _oer_record_float(record, "oer_valid_rate", index=index)
        if math.isfinite(retention):
            expected_abs = abs(retention - 1.0)
            if math.isfinite(abs_deviation) and abs(abs_deviation - expected_abs) > 1e-6:
                raise ValueError(f"oer_records[{index}].oer_abs_deviation_from_1 does not equal |oer_retention - 1|")
        if math.isfinite(valid_rate) and not (0.0 <= valid_rate <= 1.0):
            raise ValueError(f"oer_records[{index}].oer_valid_rate must be in [0,1]")
        values_by_metric["oer_retention"].append(retention)
        values_by_metric["oer_abs_deviation_from_1"].append(abs_deviation)
        values_by_metric["oer_valid_rate"].append(valid_rate)

    metric_specs = {
        "oer_retention": ("diagnostic", "sample-level nanmean over H_QI of E_current/E_fp32; target value is 1, values above 1 indicate energy expansion rather than strictly better retention"),
        "oer_abs_deviation_from_1": ("lower_is_better", "sample-level |oer_retention - 1| after H_QI nanmean"),
        "oer_valid_rate": ("higher_is_better", "fraction of H_QI hooks with valid FP32 oracle energy and non-empty legal oracle positions"),
    }
    rows: list[dict[str, Any]] = []
    for metric_name, values in values_by_metric.items():
        value_tensor = torch.tensor(values, dtype=torch.float32)
        metric_direction, definition = metric_specs[metric_name]
        for analysis_group in ("global", *ANALYSIS_GROUPS):
            mask = _analysis_mask(buckets, analysis_group)
            selected = value_tensor[mask]
            selected_labels = labels.detach().cpu().long().reshape(-1)[mask]
            support_count = int(mask.sum().item())
            if support_count <= 0:
                continue
            rows.append(
                _metric_row(
                    context=context,
                    metric_name=metric_name,
                    analysis_group=analysis_group,
                    metric_value=_finite_mean(selected),
                    metric_direction=metric_direction,
                    aggregation_unit="sample",
                    aggregation_fn="nanmean_valid_only",
                    metric_role="main",
                    support_count=support_count,
                    effective_class_count=int(torch.unique(selected_labels).numel()) if int(selected_labels.numel()) else 0,
                    definition=definition,
                )
            )
    return rows
def _retention_rows(
    *,
    context: _MetricContext,
    sample_ids: Sequence[str],
    labels: Tensor,
    logits: Tensor,
    embeddings: Tensor,
    buckets: Sequence[str],
    reference: _ReferenceTensors | None,
    oer_records: Sequence[Mapping[str, Any]] | None,
    oer_vector_reduce: str | None,
    min_abs_margin: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if reference is not None:
        ref_embeddings = reference.embeddings
        ref_logits = reference.logits
        distortion_values = embedding_distortion(ref_embeddings, embeddings, reduction="none")
        rows.append(_metric_row(context=context, metric_name="embedding_distortion", analysis_group="global", metric_value=_finite_mean(torch.as_tensor(distortion_values)), metric_direction="lower_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only", metric_role="main", support_count=int(labels.numel()), effective_class_count=int(torch.unique(labels).numel()), definition="mean ||E_current - E_reference||_2 / ||E_reference||_2"))
        cosine_values = F.cosine_similarity(embeddings.float(), ref_embeddings.float(), dim=1, eps=1e-12)
        rows.append(_metric_row(context=context, metric_name="embedding_cosine_similarity", analysis_group="global", metric_value=_finite_mean(cosine_values), metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only", metric_role="main", support_count=int(labels.numel()), effective_class_count=int(torch.unique(labels).numel()), definition="mean cosine_similarity(E_current, E_reference) over pooled embeddings"))
        try:
            fisher_value_float = _scalar_float(fisher_retention(ref_embeddings, embeddings, labels, normalize_by_dof=False), name="fisher_retention")
        except ValueError:
            fisher_value_float = float("nan")
        rows.append(_metric_row(context=context, metric_name="fisher_retention", analysis_group="global", metric_value=fisher_value_float, metric_direction="higher_is_better", aggregation_unit="global_derived", aggregation_fn="ratio", metric_role="main", support_count=int(labels.numel()), effective_class_count=int(torch.unique(labels).numel()), definition="Fisher_current / Fisher_reference, raw scatter ratio"))
        ref_margin, _ = compute_sample_margin(ref_embeddings, labels, reduction="none", return_details=True)
        cur_margin, _ = compute_sample_margin(embeddings, labels, reduction="none", return_details=True)
        try:
            geo_retention_float = _scalar_float(margin_value_retention(ref_margin, cur_margin, reduction="mean", min_abs_margin=float(min_abs_margin)), name="geo_margin_retention")
        except ValueError:
            geo_retention_float = float("nan")
        rows.append(_metric_row(context=context, metric_name="geo_margin_retention", analysis_group="global", metric_value=geo_retention_float, metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only", metric_role="main", support_count=int(labels.numel()), effective_class_count=int(torch.unique(labels).numel()), definition="sample geometric margin retention against reference embeddings"))
        try:
            tail_gap_float = _scalar_float(collapse_gap(distortion_values, list(buckets), higher_is_better=False), name="tail_collapse_gap")
        except ValueError:
            tail_gap_float = float("nan")
        rows.append(_metric_row(context=context, metric_name="tail_collapse_gap", analysis_group="tail_vs_many", metric_value=tail_gap_float, metric_direction="lower_is_better", aggregation_unit="global_derived", aggregation_fn="damage_gap_from_group_values", metric_role="main", support_count=int(labels.numel()), effective_class_count=int(torch.unique(labels).numel()), definition="tail-vs-many damage gap over per-sample embedding distortion, using common_metrics.collapse_gap"))
        try:
            logit_retention_float = _scalar_float(margin_retention(ref_logits, logits, label_id=labels, reduction="mean", min_abs_margin=float(min_abs_margin)), name="logit_margin_retention")
        except ValueError:
            logit_retention_float = float("nan")
        rows.append(_metric_row(context=context, metric_name="logit_margin_retention", analysis_group="global", metric_value=logit_retention_float, metric_direction="higher_is_better", aggregation_unit="sample", aggregation_fn="nanmean_valid_only", metric_role="main", support_count=int(labels.numel()), effective_class_count=int(torch.unique(labels).numel()), definition="correct-class logit margin retention against reference logits"))

    if oer_records is not None:
        if oer_vector_reduce is not None:
            raise ValueError("Baseline sample-level OER records do not accept oer_vector_reduce")
        rows.extend(
            _sample_oer_rows(
                context=context,
                sample_ids=sample_ids,
                labels=labels,
                buckets=buckets,
                oer_records=oer_records,
            )
        )
    return rows


def compute_baseline_metrics(
    *,
    sample_ids: Sequence[Any],
    labels: Any,
    logits: Any,
    embeddings: Any,
    bucket: Sequence[Any],
    label_mapping: Mapping[Any, Any] | None = None,
    label_buckets: Mapping[int, str] | None = None,
    split: str = "",
    predictions: Any | None = None,
    reference: ReferenceBundle | None = None,
    reference_bundle: ReferenceBundle | None = None,
    reference_embeddings: Any | None = None,
    reference_logits: Any | None = None,
    oer_records: Sequence[Mapping[str, Any]] | None = None,
    oer_vector_reduce: str | None = None,
    min_abs_margin: float = DEFAULT_MARGIN_MIN_ABS,
    model_kind: str = "",
    seed: int | None = None,
    run_identity_hash: str = "",
    label_mapping_hash: str = "",
    split_hash: str = "",
    sample_order_hash: str | None = None,
    pooling: str = "mask_aware_mean",
) -> dict[str, list[dict[str, Any]]]:
    """Compute Baseline Pipeline metrics from already-exported tensors.

    ``predictions`` is an audit input only. The canonical prediction used by all
    metrics is always ``logits.argmax(dim=1)``.
    """
    if reference_embeddings is not None or reference_logits is not None:
        raise ValueError("reference_embeddings/reference_logits are no longer accepted; pass a ReferenceBundle")
    if reference is not None and reference_bundle is not None:
        raise ValueError("pass only one of reference or reference_bundle")
    reference = reference if reference is not None else reference_bundle
    if float(min_abs_margin) != float(DEFAULT_MARGIN_MIN_ABS):
        raise ValueError(
            "margin_retention uses fixed margin_min_abs="
            f"{float(DEFAULT_MARGIN_MIN_ABS)}; got {float(min_abs_margin)}"
        )

    logits_tensor = _as_float_cpu(logits, name="logits")
    embeddings_tensor = _as_float_cpu(embeddings, name="embeddings")
    _require_finite_tensor(logits_tensor, name="logits")
    _require_finite_tensor(embeddings_tensor, name="embeddings")
    if logits_tensor.ndim != 2 or int(logits_tensor.shape[1]) <= 0:
        raise ValueError(f"logits must be [N,C], got shape={tuple(logits_tensor.shape)}")
    if embeddings_tensor.ndim != 2 or int(embeddings_tensor.shape[1]) <= 0:
        raise ValueError(f"embeddings must be [N,D], got shape={tuple(embeddings_tensor.shape)}")
    num_classes = int(logits_tensor.shape[1])
    label_mapping_normalized = _normalise_label_mapping(label_mapping)
    label_ids = sorted(int(item) for item in label_mapping_normalized.values())
    if label_ids != list(range(num_classes)):
        raise ValueError(f"label_mapping universe {label_ids} does not match logits class dimension C={num_classes}")
    label_names = _label_names_by_id(label_mapping_normalized)

    labels_tensor = _as_label_tensor(labels, name="labels", num_classes=num_classes)
    predictions_tensor = logits_tensor.argmax(dim=1).detach().cpu().long().contiguous()
    _validate_external_predictions(predictions, derived=predictions_tensor, num_classes=num_classes)

    sample_id_list = [str(item) for item in sample_ids]
    current_sample_order_hash = str(sample_order_hash or _sample_order_hash(sample_id_list))
    if current_sample_order_hash != _sample_order_hash(sample_id_list):
        raise ValueError("provided sample_order_hash does not match current sample_ids order")

    _validate_inputs(
        sample_ids=sample_id_list,
        labels=labels_tensor,
        predictions=predictions_tensor,
        logits=logits_tensor,
        embeddings=embeddings_tensor,
        buckets=[str(item) for item in bucket],
    )
    label_bucket_map = _canonical_label_buckets(label_buckets, label_ids)
    canonical_buckets = _audit_sample_buckets(labels_tensor, [str(item) for item in bucket], label_bucket_map)

    reference_tensors = _validate_reference_bundle(
        reference,
        sample_ids=sample_id_list,
        labels=labels_tensor,
        logits=logits_tensor,
        embeddings=embeddings_tensor,
        label_mapping_hash=str(label_mapping_hash),
        split_hash=str(split_hash),
        sample_order_hash=current_sample_order_hash,
        pooling=str(pooling),
        num_classes=num_classes,
    )
    context = _MetricContext(
        split=str(split),
        model_kind=str(model_kind or "unknown"),
        seed=None if seed is None else int(seed),
        run_identity_hash=str(run_identity_hash),
        reference_checkpoint_hash="" if reference is None else str(reference.checkpoint_hash),
        label_mapping_hash=str(label_mapping_hash),
        split_hash=str(split_hash),
        sample_order_hash=current_sample_order_hash,
        pooling=str(pooling),
    )

    main_rows, group_rows = _classification_rows(
        context=context,
        labels=labels_tensor,
        predictions=predictions_tensor,
        logits=logits_tensor,
        buckets=canonical_buckets,
        label_ids=label_ids,
        label_buckets=label_bucket_map,
    )
    per_class_rows = _per_class_rows(
        context=context,
        labels=labels_tensor,
        predictions=predictions_tensor,
        label_ids=label_ids,
        label_names=label_names,
        label_buckets=label_bucket_map,
    )
    geometry_rows = _geometry_rows(
        context=context,
        labels=labels_tensor,
        logits=logits_tensor,
        embeddings=embeddings_tensor,
        buckets=canonical_buckets,
        label_buckets=label_bucket_map,
    )
    retention_rows = _retention_rows(
        context=context,
        sample_ids=[str(item) for item in sample_ids],
        labels=labels_tensor,
        logits=logits_tensor,
        embeddings=embeddings_tensor,
        buckets=canonical_buckets,
        reference=reference_tensors,
        oer_records=oer_records,
        oer_vector_reduce=oer_vector_reduce,
        min_abs_margin=float(DEFAULT_MARGIN_MIN_ABS),
    )
    all_rows = [*main_rows, *group_rows, *per_class_rows, *geometry_rows, *retention_rows]
    return {
        "metrics": all_rows,
        "main_metrics": main_rows,
        "group_metrics": group_rows,
        "per_class_metrics": per_class_rows,
        "geometry_metrics": geometry_rows,
        "retention_metrics": retention_rows,
    }


def compute_export_metric_rows(
    *,
    split: str,
    logits: Tensor,
    embeddings: Tensor,
    labels: Tensor,
    buckets: Sequence[str],
    label_buckets: Mapping[int, str],
    label_mapping: Mapping[Any, Any] | None = None,
    sample_ids: Sequence[Any] | None = None,
    predictions: Tensor | None = None,
    reference: ReferenceBundle | None = None,
    oer_records: Sequence[Mapping[str, Any]] | None = None,
    oer_vector_reduce: str | None = None,
    model_kind: str = "",
    seed: int | None = None,
    run_identity_hash: str = "",
    label_mapping_hash: str = "",
    split_hash: str = "",
    sample_order_hash: str | None = None,
    pooling: str = "mask_aware_mean",
) -> list[dict[str, Any]]:
    if sample_ids is None:
        sample_ids = [str(index) for index in range(int(labels.detach().reshape(-1).numel()))]
    bundle = compute_baseline_metrics(
        sample_ids=sample_ids,
        labels=labels,
        predictions=predictions,
        logits=logits,
        embeddings=embeddings,
        bucket=buckets,
        label_mapping=label_mapping,
        label_buckets=label_buckets,
        split=split,
        reference=reference,
        oer_records=oer_records,
        oer_vector_reduce=oer_vector_reduce,
        model_kind=model_kind,
        seed=seed,
        run_identity_hash=run_identity_hash,
        label_mapping_hash=label_mapping_hash,
        split_hash=split_hash,
        sample_order_hash=sample_order_hash,
        pooling=pooling,
    )
    return list(bundle["metrics"])


__all__ = [
    "BASELINE_METRIC_TABLES",
    "CANONICAL_METRIC_COLUMNS",
    "GEOMETRY_METRIC_NAMES",
    "LEGACY_MACRO_F1_METRIC",
    "ReferenceBundle",
    "SELECTION_MACRO_F1_METRIC",
    "compute_baseline_metrics",
    "compute_export_metric_rows",
    "correct_logit_margin",
    "top1_top2_gap",
]
