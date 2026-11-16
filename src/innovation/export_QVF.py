from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from src.baseline.export_eval import (
    DEFAULT_TOKEN_EXPORT,
    DEFAULT_TOKEN_LAYERS,
    ExportEvalConfig,
    load_fp32_reference_bundle,
    run_export_eval_config,
)
from src.baseline.manifest import file_sha256
from src.metrics.common_metrics import DEFAULT_EPS, embedding_distortion, tail_and_few_damage_gaps
from src.metrics.geometry_metrics import class_trace_variance
from src.train.train_beats_classifier_head import DatasetPaths, dataset_paths
from src.utils.group_utils import group_protocol_metadata, tail_included, validate_bucket_values


PROTOCOL_VERSION = "innovation_qvf_export.v1"
METHOD_VARIANT = "stage1_vanilla_kd_plus_qvf_token_fidelity"
SPLIT_NAME = "eval"
DEFAULT_SEED = 2026
DEFAULT_BETAS = ("0.01", "0.05", "0.10")
FORMAL_QVF_BETAS = (0.01, 0.05, 0.10)
SENSITIVE_TRAIN_LAYERS = (1, 9)
MECHANISM_EXPORT_LAYERS = (1, 2, 3, 4, 9)
MECHANISM_MODULE_NAME = "fc1"
MECHANISM_QUANT_MODES = ("A4-only", "W4A4")
MECHANISM_ANALYSIS_GROUPS = ("many", "medium", "few", "tail")
OVERALL_ACC_GUARDRAIL = -0.005
OVERALL_MACRO_F1_GUARDRAIL = -0.005
MANY_ACC_GUARDRAIL = -0.010
SELECTION_METRIC = "tail_macro_f1"
TIE_BREAK_METRICS = ("tail_accuracy", "medium_macro_f1", "medium_accuracy")

SUMMARY_COLUMNS = [
    "candidate_index",
    "selected",
    "passes_guardrail",
    "guardrail_failure_reasons",
    "dataset_name",
    "seed",
    "qvf_beta",
    "candidate_dir",
    "checkpoint",
    "checkpoint_sha256",
    "train_log",
    "train_log_sha256",
    "candidate_manifest",
    "candidate_manifest_sha256",
    "eval_export_dir",
    "eval_export_generated",
    "eval_export_run_identity_hash",
    "overall_accuracy",
    "macro_f1_label_universe_nanmean",
    "many_accuracy",
    "many_macro_f1",
    "medium_accuracy",
    "medium_macro_f1",
    "tail_accuracy",
    "tail_macro_f1",
    "few_accuracy",
    "few_f1",
    "delta_overall_accuracy_vs_vanilla_kd",
    "delta_macro_f1_label_universe_nanmean_vs_vanilla_kd",
    "delta_many_accuracy_vs_vanilla_kd",
    "embedding_distortion",
    "embedding_cosine_similarity",
    "fisher_retention",
    "geo_margin_retention",
    "variance_deviation_abs_log",
    "variance_deviation_nonfinite_class_count",
    "tail_collapse_gap",
    "few_collapse_gap",
    "geometry_all_required_finite",
    "mechanism_summary_available",
    "test_split_touched",
]

MECHANISM_COLUMNS = [
    "dataset_name",
    "seed",
    "model_role",
    "qvf_beta",
    "source_dir",
    "source_kind",
    "layer",
    "layer_idx",
    "module_name",
    "quant_mode",
    "analysis_group",
    "metric_name",
    "metric_value",
    "n_total",
    "n_valid",
    "topk_unit",
    "topk_source",
    "topk_selected_on",
    "topk_ratio",
    "topk",
    "topk_k",
    "topk_k_policy",
    "topk_policy",
    "padding_policy",
    "student_activation_semantic",
]

MECHANISM_METRICS = {
    "oer_retention": "token_oer_retention",
    "mean_norm_retention": "mean_norm_retention",
    "teacher_topk_relative_distortion": "mean_token_distortion",
    "topk_overlap": "topk_overlap",
    "high_energy_scalar_exact_zero_rate": "high_energy_scalar_exact_zero_rate",
    "high_energy_scalar_strict_overflow_rate": "high_energy_scalar_strict_overflow_rate",
    "topk_token_exact_zero_rate": "topk_token_exact_zero_rate",
    "topk_token_strict_overflow_rate": "topk_token_strict_overflow_rate",
}

REQUIRED_GEOMETRY_METRICS = (
    "embedding_distortion",
    "fisher_retention",
    "geo_margin_retention",
    "variance_deviation_abs_log",
    "variance_deviation_nonfinite_class_count",
    "tail_collapse_gap",
    "few_collapse_gap",
)


@dataclass(frozen=True)
class BetaSpec:
    raw: str
    value: float
    dir_name: str


@dataclass(frozen=True)
class CandidateSpec:
    index: int
    beta: BetaSpec
    candidate_dir: Path
    checkpoint_path: Path
    train_log_path: Path
    manifest_path: Path
    eval_export_dir: Path
    token_survival_dir: Path


@dataclass
class NumericAccumulator:
    values: list[float]

    @classmethod
    def empty(cls) -> "NumericAccumulator":
        return cls(values=[])

    def add(self, value: Any) -> None:
        self.values.append(_as_float_or_nan(value))

    @property
    def n_total(self) -> int:
        return len(self.values)

    @property
    def n_valid(self) -> int:
        return sum(1 for value in self.values if math.isfinite(float(value)))

    def nanmean(self, *, context: str) -> float:
        valid = [float(value) for value in self.values if math.isfinite(float(value))]
        if not valid:
            raise RuntimeError(f"{context} has n_valid=0")
        return float(sum(valid) / float(len(valid)))


def _path_str(path: str | Path | None) -> str:
    if path is None:
        return ""
    return str(Path(path).resolve(strict=False))


def _json_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    tmp.replace(target)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    tmp.replace(target)


def _read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"CSV not found: {target}")
    with target.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise RuntimeError(f"CSV has no header: {target}")
    return rows, fieldnames


def _csv_row_count(path: str | Path) -> int:
    rows, _fieldnames = _read_csv_rows(path)
    return int(len(rows))


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"JSON not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON must contain an object: {target}")
    return payload


def _load_torch_mapping(path: str | Path, *, map_location: Any = "cpu") -> Mapping[str, Any]:
    target = Path(path)
    try:
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(target, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError(f"torch artifact must contain a mapping, got {type(payload)!r}: {target}")
    return payload


def _as_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, float):
        return float(value)
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _finite_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if math.isinf(parsed):
        raise ValueError(f"{name} must be finite or NaN, got {value!r}")
    return parsed


def _parse_betas(value: str | Sequence[str] | None) -> tuple[BetaSpec, ...]:
    raw_items = DEFAULT_BETAS if value is None else value
    if isinstance(raw_items, str):
        parts = tuple(part.strip() for part in raw_items.split(",") if part.strip())
    else:
        parts = tuple(str(item).strip() for item in raw_items if str(item).strip())
    if not parts:
        raise argparse.ArgumentTypeError("beta list must not be empty")
    result: list[BetaSpec] = []
    seen: set[float] = set()
    for raw in parts:
        beta = float(raw)
        if not math.isfinite(beta) or beta < 0.0:
            raise argparse.ArgumentTypeError(f"qvf_beta must be finite and non-negative, got {raw!r}")
        if beta in seen:
            raise argparse.ArgumentTypeError(f"duplicate qvf_beta value: {raw!r}")
        seen.add(beta)
        stem = raw.lower().replace("+", "").replace("-", "m").replace(".", "p")
        result.append(BetaSpec(raw=raw, value=float(beta), dir_name=f"beta_{stem}"))
    return tuple(result)


def _qvf_float_stem(value: float) -> str:
    return f"{float(value):.10g}".replace("+", "").replace("-", "m").replace(".", "p")


def _parse_expected_layer_weights(value: str | None) -> dict[str, float] | None:
    text = str(value or "").strip()
    if not text:
        return None
    result: dict[int, float] = {}
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"expected layer weight must be layer:weight, got {item!r}")
        raw_layer, raw_weight = item.split(":", 1)
        layer_text = raw_layer.strip().lower()
        if layer_text.startswith("l"):
            layer_text = layer_text[1:]
        if "." in layer_text:
            layer_text = layer_text.split(".", 1)[0]
        layer_idx = int(layer_text)
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise argparse.ArgumentTypeError(f"expected layer weight must be finite and non-negative, got {raw_weight!r}")
        result[layer_idx] = float(weight)
    expected_layers = tuple(int(item) for item in SENSITIVE_TRAIN_LAYERS)
    if tuple(sorted(result)) != tuple(sorted(expected_layers)):
        raise argparse.ArgumentTypeError(f"expected layer weights must cover exactly {expected_layers}, got {sorted(result)}")
    total = sum(float(item) for item in result.values())
    if not math.isfinite(total) or total <= 0.0:
        raise argparse.ArgumentTypeError("expected layer weights must sum to a positive finite value")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise argparse.ArgumentTypeError(f"expected layer weights must sum to 1.0, got {total}")
    return {f"L{layer}.{MECHANISM_MODULE_NAME}": float(result[layer]) for layer in expected_layers}


def _qvf_weighted_variant(layer_weights: Mapping[str, float]) -> str:
    ordered: list[tuple[int, float]] = []
    for key, value in layer_weights.items():
        text = str(key).strip().lower()
        if text.startswith("l"):
            text = text[1:]
        if "." in text:
            text = text.split(".", 1)[0]
        ordered.append((int(text), float(value)))
    return "_".join(f"L{layer}_{_qvf_float_stem(weight)}" for layer, weight in sorted(ordered))


def _dataset_root(dataset_name: str | None, dataset_root: str | Path | None) -> Path:
    return Path(dataset_root) if dataset_root else dataset_paths(dataset_name, None).root


def _rebase_dataset_path(path: Path, *, old_root: Path, new_root: Path, fallback_relative: str) -> Path:
    source = Path(path)
    try:
        relative = source.resolve(strict=False).relative_to(old_root.resolve(strict=False))
    except ValueError:
        relative = Path(fallback_relative)
    return new_root / relative


def _dataset_paths_for_root(dataset_name: str | None, dataset_root: str | Path | None, output: str | Path | None = None) -> DatasetPaths:
    paths = dataset_paths(dataset_name, None if output is None else str(output))
    if dataset_root is None or not str(dataset_root).strip():
        return paths
    root = Path(dataset_root)
    old_root = Path(paths.root)
    return DatasetPaths(
        name=paths.name,
        root=root,
        train_csv=_rebase_dataset_path(paths.train_csv, old_root=old_root, new_root=root, fallback_relative="splits/train.csv"),
        eval_csv=_rebase_dataset_path(paths.eval_csv, old_root=old_root, new_root=root, fallback_relative="splits/eval.csv"),
        test_csv=_rebase_dataset_path(paths.test_csv, old_root=old_root, new_root=root, fallback_relative="splits/test.csv"),
        full_csv=_rebase_dataset_path(paths.full_csv, old_root=old_root, new_root=root, fallback_relative="splits/full_dataset.csv"),
        label_mapping=_rebase_dataset_path(paths.label_mapping, old_root=old_root, new_root=root, fallback_relative="splits/label_mapping.json"),
        output_checkpoint=Path(output) if output is not None else root / f"BEATs_{paths.name}.pt",
    )


def _seed_dir(dataset_root: Path, seed: int, output_root: str | Path | None) -> Path:
    root = Path(output_root) if output_root else dataset_root / "innovation" / "QVF"
    return root / f"seed_{int(seed)}"


def _default_reference_root(dataset_root: Path, seed: int) -> Path:
    return dataset_root / "baseline" / "teacher" / f"seed_{int(seed)}"


def _default_teacher_checkpoint(dataset_root: Path, seed: int) -> Path:
    return _default_reference_root(dataset_root, seed) / "checkpoints" / f"fp32_teacher_seed_{int(seed)}.pt"


def _default_qparams_csv(dataset_root: Path, seed: int) -> Path:
    return dataset_root / "baseline" / "student" / f"seed_{int(seed)}" / "qparams" / "int4_qparams.csv"


def _default_int4_checkpoint(dataset_root: Path, seed: int) -> Path:
    return dataset_root / "baseline" / "student" / f"seed_{int(seed)}" / "checkpoints" / f"int4_student_calibrated_seed_{int(seed)}.pt"


def _default_calibration_csv(dataset_root: Path) -> Path:
    return dataset_root / "splits" / "calibration.csv"


def _default_vanilla_metrics(dataset_root: Path, seed: int) -> Path:
    return dataset_root / "baseline" / "kd" / f"seed_{int(seed)}" / "vanilla_kd_metrics.csv"


def _default_vanilla_manifest(dataset_root: Path, seed: int) -> Path:
    return dataset_root / "baseline" / "kd" / f"seed_{int(seed)}" / "run_manifest.json"


def _default_vanilla_checkpoint(dataset_root: Path, seed: int) -> Path:
    return dataset_root / "baseline" / "kd" / f"seed_{int(seed)}" / "checkpoints" / f"vanilla_kd_best_seed_{int(seed)}.pt"


def _default_vanilla_token_survival_dir(dataset_root: Path, seed: int) -> Path:
    del seed
    return dataset_root / "token_survival"


def _candidate_checkpoint(candidate_dir: Path, explicit_name: str | None) -> Path:
    if explicit_name:
        return candidate_dir / explicit_name
    candidates = (
        "qvf_stage1_checkpoint.pt",
        "stage1_checkpoint.pt",
        "checkpoint.pt",
        "best_checkpoint.pt",
    )
    for name in candidates:
        path = candidate_dir / name
        if path.exists():
            return path
    return candidate_dir / candidates[0]


def _candidate_specs(args: argparse.Namespace, seed_dir: Path) -> tuple[CandidateSpec, ...]:
    betas = _parse_betas(args.betas)
    result: list[CandidateSpec] = []
    for index, beta in enumerate(betas, start=1):
        candidate_dir = seed_dir / beta.dir_name
        result.append(
            CandidateSpec(
                index=index,
                beta=beta,
                candidate_dir=candidate_dir,
                checkpoint_path=_candidate_checkpoint(candidate_dir, args.checkpoint_name),
                train_log_path=candidate_dir / "qvf_train_log.csv",
                manifest_path=candidate_dir / "qvf_manifest.json",
                eval_export_dir=candidate_dir / "eval_export",
                token_survival_dir=candidate_dir / "token_survival",
            )
        )
    return tuple(result)


def _metric_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
    analysis_group: str,
    split: str = SPLIT_NAME,
    required: bool = True,
) -> float:
    matches = [
        row
        for row in rows
        if str(row.get("metric_name")) == str(metric_name)
        and str(row.get("analysis_group")) == str(analysis_group)
        and (not row.get("split") or str(row.get("split")) == str(split))
    ]
    if not matches:
        if required:
            raise KeyError(f"missing metric: split={split}, metric_name={metric_name}, analysis_group={analysis_group}")
        return float("nan")
    return _finite_float(matches[0].get("metric_value"), name=f"{metric_name}:{analysis_group}")


def _read_metric_tables_from_export(paths: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in ("metrics", "group_metrics", "geometry_metrics", "retention_metrics"):
        path = paths.get(key)
        if not path:
            continue
        table_rows, _fieldnames = _read_csv_rows(path)
        rows.extend(table_rows)
    if not rows:
        raise RuntimeError(f"export result contains no metric tables: {paths}")
    return rows


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
            suffix = parts[parts.index(anchor):]
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


def _baseline_rows_from_manifest(manifest_path: Path, *, dataset_root: Path) -> list[dict[str, str]]:
    if not manifest_path.exists() or not manifest_path.is_file():
        return []
    manifest = _read_json(manifest_path)
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or SPLIT_NAME not in splits:
        return []
    split_result = splits[SPLIT_NAME]
    if not isinstance(split_result, Mapping):
        return []
    raw_paths = split_result.get("paths")
    if not isinstance(raw_paths, Mapping):
        raw_paths = manifest.get("outputs")
        if isinstance(raw_paths, Mapping):
            raw_paths = raw_paths.get(SPLIT_NAME)
    if not isinstance(raw_paths, Mapping):
        return []
    resolved_paths: dict[str, str] = {}
    for key in ("metrics", "group_metrics", "geometry_metrics", "retention_metrics"):
        value = raw_paths.get(key)
        if not value:
            continue
        resolved_paths[key] = _path_str(
            _resolve_uploaded_artifact_path(value, manifest_path=manifest_path, dataset_root=dataset_root)
        )
    if "metrics" not in resolved_paths or "group_metrics" not in resolved_paths:
        return []
    return _read_metric_tables_from_export(resolved_paths)


def _baseline_eval_rows(path: Path, *, manifest_path: Path | None = None, dataset_root: Path | None = None) -> list[dict[str, str]]:
    if manifest_path is not None and dataset_root is not None:
        manifest_rows = _baseline_rows_from_manifest(manifest_path, dataset_root=dataset_root)
        if manifest_rows:
            return manifest_rows
    rows, _fieldnames = _read_csv_rows(path)
    eval_rows = [row for row in rows if str(row.get("split")) == SPLIT_NAME]
    if not eval_rows:
        raise RuntimeError(f"Vanilla KD metrics table contains no eval rows: {path}")
    return eval_rows


def _classification_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "overall_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="global"),
        "macro_f1_label_universe_nanmean": _metric_value(
            rows,
            metric_name="macro_f1_label_universe_nanmean",
            analysis_group="global",
        ),
        "many_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="many"),
        "many_macro_f1": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="many"),
        "medium_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="medium"),
        "medium_macro_f1": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="medium"),
        "tail_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="tail"),
        "tail_macro_f1": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="tail"),
        "few_accuracy": _metric_value(rows, metric_name="accuracy", analysis_group="few"),
        "few_f1": _metric_value(rows, metric_name="macro_f1_label_universe_nanmean", analysis_group="few"),
    }


def _global_metric_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        "embedding_distortion": _metric_value(rows, metric_name="embedding_distortion", analysis_group="global"),
        "embedding_cosine_similarity": _metric_value(
            rows,
            metric_name="embedding_cosine_similarity",
            analysis_group="global",
            required=False,
        ),
        "fisher_retention": _metric_value(rows, metric_name="fisher_retention", analysis_group="global"),
        "geo_margin_retention": _metric_value(rows, metric_name="geo_margin_retention", analysis_group="global"),
    }


def _tensor_payload(path: str | Path) -> Mapping[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"tensor artifact not found: {target}")
    return _load_torch_mapping(target, map_location="cpu")


def _payload_tensor(payload: Mapping[str, Any], keys: Sequence[str], *, name: str) -> Tensor:
    for key in keys:
        if key in payload:
            return torch.as_tensor(payload[key]).detach().cpu().float().contiguous()
    raise KeyError(f"payload missing tensor key for {name}: {keys}")


def _payload_labels(payload: Mapping[str, Any]) -> Tensor:
    for key in ("labels", "label_id"):
        if key in payload:
            return torch.as_tensor(payload[key]).detach().cpu().long().reshape(-1).contiguous()
    raise KeyError("payload missing labels/label_id")


def _payload_sample_ids(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("sample_ids", payload.get("sample_id"))
    if isinstance(raw, Tensor):
        return [str(item) for item in raw.detach().cpu().reshape(-1).tolist()]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    raise KeyError("payload missing sample_ids/sample_id")


def _finite_mean(values: Tensor) -> float:
    flat = values.detach().cpu().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if int(finite.numel()) <= 0:
        return float("nan")
    return float(finite.mean().item())


def _variance_extra(current_embeddings: Tensor, labels: Tensor, reference_embeddings: Tensor) -> dict[str, Any]:
    ref_var, ref_details = class_trace_variance(
        reference_embeddings,
        labels,
        normalization="dof",
        reduction="none",
        return_details=True,
    )
    cur_var, cur_details = class_trace_variance(
        current_embeddings,
        labels,
        normalization="dof",
        reduction="none",
        return_details=True,
    )
    assert isinstance(ref_var, Tensor)
    assert isinstance(cur_var, Tensor)
    ref_ids = torch.as_tensor(ref_details["class_ids"]).detach().cpu().long()
    cur_ids = torch.as_tensor(cur_details["class_ids"]).detach().cpu().long()
    if not torch.equal(ref_ids, cur_ids):
        raise ValueError("class_trace_variance class ids differ between reference and QVF candidate")
    ratio = cur_var.float() / (ref_var.float() + float(DEFAULT_EPS))
    deviation = torch.abs(torch.log(ratio))
    valid = torch.isfinite(ratio) & (ratio > 0.0) & torch.isfinite(deviation)
    finite_deviation = torch.where(valid, deviation, torch.full_like(deviation, float("nan")))
    return {
        "variance_deviation_abs_log": _finite_mean(finite_deviation),
        "variance_deviation_nonfinite_class_count": int((~valid).sum().item()),
        "variance_deviation_total_class_count": int(ref_ids.numel()),
    }


def _tail_few_extra(
    current_embeddings: Tensor,
    reference_embeddings: Tensor,
    buckets: Sequence[str],
) -> dict[str, float]:
    distortion = embedding_distortion(reference_embeddings, current_embeddings, reduction="none")
    assert isinstance(distortion, Tensor)
    payload = tail_and_few_damage_gaps(distortion, list(buckets), higher_is_better=False)
    return {
        "tail_collapse_gap": float(torch.as_tensor(payload["tail_vs_many_gap"]).reshape(-1)[0].item()),
        "few_collapse_gap": float(torch.as_tensor(payload["few_vs_many_gap"]).reshape(-1)[0].item()),
    }


def _geometry_extra_from_export(paths: Mapping[str, Any], *, reference_root: Path) -> dict[str, Any]:
    embeddings_payload = _tensor_payload(paths["embeddings"])
    current_embeddings = _payload_tensor(embeddings_payload, ("embeddings", "embedding"), name="embeddings")
    labels = _payload_labels(embeddings_payload)
    sample_ids = _payload_sample_ids(embeddings_payload)
    buckets = [str(item) for item in embeddings_payload.get("bucket", [])]
    if len(buckets) != len(sample_ids):
        raise RuntimeError("embedding artifact bucket length must match sample_ids for QVF geometry summary")
    reference = load_fp32_reference_bundle(reference_root, split=SPLIT_NAME)
    if sample_ids != [str(item) for item in reference.sample_ids]:
        raise ValueError("QVF export sample order differs from FP32 reference artifacts")
    if not torch.equal(labels.reshape(-1), reference.labels.long().reshape(-1)):
        raise ValueError("QVF export labels differ from FP32 reference artifacts")
    if tuple(current_embeddings.shape) != tuple(reference.embeddings.shape):
        raise ValueError(
            f"QVF embedding shape differs from FP32 reference: "
            f"{tuple(current_embeddings.shape)} vs {tuple(reference.embeddings.shape)}"
        )
    result: dict[str, Any] = {}
    result.update(_variance_extra(current_embeddings, labels, reference.embeddings.float()))
    result.update(_tail_few_extra(current_embeddings, reference.embeddings.float(), buckets))
    return result


def _candidate_export_model_name(spec: CandidateSpec) -> str:
    return f"qvf_beta_{spec.beta.raw.replace('.', 'p')}"


def _resume_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "resume", True)) and not bool(getattr(args, "force_export", False))


def _parse_token_layers_csv(value: Any) -> tuple[int, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    result: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        result.append(int(item))
    return tuple(result)


def _cross_platform_path_name(path_value: str | Path) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    return raw.replace("\\", "/").rstrip("/").split("/")[-1]


def _portable_path_parts(path_value: str | Path) -> tuple[str, ...]:
    raw = str(path_value or "").strip()
    if not raw:
        return ()
    return tuple(part for part in raw.replace("\\", "/").split("/") if part and part != ".")


def _unique_artifact_hash(matches: Sequence[tuple[str, Any]]) -> str:
    values = sorted({str(value) for _key, value in matches if value})
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        keys = [str(key) for key, _value in matches[:6]]
        raise RuntimeError(f"ambiguous artifact_sha256 matches for migrated QVF export path: {keys}")
    return ""


def _artifact_hash_from_manifest(manifest: Mapping[str, Any], path: Path) -> str:
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        return ""
    resolved = str(path.resolve(strict=False))
    candidates = (resolved, str(path), path.name)
    for key in candidates:
        value = artifact_hashes.get(key)
        if value:
            return str(value)
    path_parts = _portable_path_parts(resolved)
    for width in range(min(8, len(path_parts)), 1, -1):
        suffix = tuple(path_parts[-width:])
        matched = _unique_artifact_hash(
            [
                (str(key), value)
                for key, value in artifact_hashes.items()
                if value
                and len(_portable_path_parts(str(key))) >= width
                and tuple(_portable_path_parts(str(key))[-width:]) == suffix
            ]
        )
        if matched:
            return matched
    filename = _cross_platform_path_name(path)
    matched = _unique_artifact_hash(
        [
            (str(key), value)
            for key, value in artifact_hashes.items()
            if value and _cross_platform_path_name(str(key)) == filename
        ]
    )
    if matched:
        return matched
    return ""


_FLAT_EVAL_EXPORT_FILENAMES = {
    "logits": "logits.pt",
    "embeddings": "embeddings.pt",
    "predictions": "predictions.csv",
    "metrics": "metrics.csv",
    "group_metrics": "group_metrics.csv",
    "per_class_metrics": "per_class_metrics.csv",
    "geometry_metrics": "geometry_metrics.csv",
    "token_features_index": "token_features.index.csv",
}


def _resolve_resumable_export_file(path: str | Path, *, manifest_path: Path, key: str) -> Path:
    recorded = str(path or "").strip()
    target = Path(recorded)
    if target.exists() and target.is_file():
        return target.resolve(strict=False)
    root = manifest_path.parent
    candidates: list[Path] = []
    expected_name = _FLAT_EVAL_EXPORT_FILENAMES.get(str(key))
    if expected_name:
        if str(key) == "token_features_index":
            candidates.append(root / "token_features" / expected_name)
        candidates.append(root / expected_name)
    filename = _cross_platform_path_name(recorded)
    if filename:
        candidates.append(root / filename)
        candidates.append(root / "token_features" / filename)
    seen: set[str] = set()
    for candidate in candidates:
        marker = str(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve(strict=False)
    checked = ", ".join(str(item) for item in candidates[:8])
    raise FileNotFoundError(f"resumable QVF export missing {key}: recorded={target}; checked={checked}")


def _assert_existing_file_matches_manifest(manifest: Mapping[str, Any], path: str | Path, *, name: str) -> str:
    target = Path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"resumable QVF export missing {name}: {target}")
    actual = file_sha256(target)
    expected = _artifact_hash_from_manifest(manifest, target)
    if not expected:
        raise RuntimeError(f"resumable QVF export {name} is not registered in artifact_sha256: {target}")
    if str(actual) != str(expected):
        raise RuntimeError(f"resumable QVF export {name} sha256 mismatch: actual={actual}, expected={expected}")
    return str(actual)


def _resolve_token_feature_shard_path(row: Mapping[str, Any], *, index_path: Path) -> Path:
    file_path = str(row.get("file_path") or "").strip()
    if not file_path:
        raise RuntimeError(f"resumable QVF token feature index has empty file_path: {index_path}")
    target = Path(file_path)
    if target.exists() and target.is_file():
        return target.resolve(strict=False)
    filename = _cross_platform_path_name(file_path)
    candidates: list[Path] = []
    layer_value = row.get("layer", row.get("layer_idx"))
    if layer_value is not None and str(layer_value).strip():
        candidates.append(index_path.parent / f"layer_{int(float(str(layer_value))):02d}" / filename)
    candidates.append(index_path.parent / filename)
    seen: set[str] = set()
    for candidate in candidates:
        marker = str(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve(strict=False)
    checked = ", ".join(str(item) for item in candidates[:8])
    raise FileNotFoundError(f"resumable QVF token feature shard missing: recorded={target}; checked={checked}")


def _validate_existing_token_index(manifest: Mapping[str, Any], token_index_path: str | Path) -> dict[str, Any]:
    index_path = Path(token_index_path)
    index_hash = _assert_existing_file_matches_manifest(manifest, index_path, name="token feature index")
    rows, fieldnames = _read_csv_rows(index_path)
    if not rows:
        raise RuntimeError(f"resumable QVF token feature index is empty: {index_path}")
    required = {"file_path", "file_sha256", "layer", "shard_id"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise RuntimeError(f"resumable QVF token feature index missing columns {missing}: {index_path}")
    checked = 0
    for row in rows:
        expected = row.get("file_sha256")
        shard_path = _resolve_token_feature_shard_path(row, index_path=index_path)
        actual = file_sha256(shard_path)
        if str(actual) != str(expected or ""):
            raise RuntimeError(
                f"resumable QVF token feature shard sha256 mismatch: {shard_path} "
                f"actual={actual}, expected={expected}"
            )
        checked += 1
    return {"token_features_index_sha256": index_hash, "token_feature_shards_checked": int(checked)}


def _validate_existing_export_manifest(
    *,
    spec: CandidateSpec,
    args: argparse.Namespace,
    dataset_paths_obj: Any,
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(manifest.get("stage", "")) != "baseline_export_eval":
        raise RuntimeError("existing eval_export manifest stage is not baseline_export_eval")
    if str(manifest.get("model_name", "")) != _candidate_export_model_name(spec):
        raise RuntimeError(f"existing eval_export model_name mismatch: {manifest.get('model_name')!r}")
    if str(manifest.get("split_name", "")) != SPLIT_NAME:
        raise RuntimeError(f"existing eval_export split_name mismatch: {manifest.get('split_name')!r}")
    if str(manifest.get("pooling", "")) != "mask_aware_mean":
        raise RuntimeError(f"existing eval_export pooling mismatch: {manifest.get('pooling')!r}")
    if int(manifest.get("seed", -1)) != int(args.seed):
        raise RuntimeError(f"existing eval_export seed mismatch: {manifest.get('seed')!r} != {int(args.seed)}")

    checkpoint_hash = file_sha256(spec.checkpoint_path)
    if str(manifest.get("checkpoint_sha256") or "") != str(checkpoint_hash):
        raise RuntimeError("existing eval_export checkpoint hash differs from current QVF checkpoint")
    label_mapping_hash = file_sha256(dataset_paths_obj.label_mapping)
    if str(manifest.get("label_mapping_hash") or "") != str(label_mapping_hash):
        raise RuntimeError("existing eval_export label_mapping hash differs from current dataset")
    split_hashes = manifest.get("split_hashes")
    if not isinstance(split_hashes, Mapping):
        raise RuntimeError("existing eval_export manifest missing split_hashes mapping")
    eval_hash = file_sha256(dataset_paths_obj.eval_csv)
    if str(split_hashes.get(SPLIT_NAME) or "") != str(eval_hash):
        raise RuntimeError("existing eval_export eval split hash differs from current dataset")
    if str(split_hashes.get("test", "")).strip():
        raise RuntimeError("existing eval_export unexpectedly records test split hash")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or SPLIT_NAME not in splits:
        raise RuntimeError("existing eval_export manifest missing eval split result")
    if "test" in splits:
        raise RuntimeError("existing eval_export unexpectedly contains test split result")

    expected_token_export = str(args.token_export)
    if str(manifest.get("token_export", "")) != expected_token_export:
        raise RuntimeError(
            f"existing eval_export token_export mismatch: {manifest.get('token_export')!r} != {expected_token_export!r}"
        )
    manifest_token_layers = tuple(int(item) for item in (manifest.get("token_layers") or []))
    if expected_token_export == "none" and manifest_token_layers:
        raise RuntimeError("existing eval_export has token layers but current token_export=none")
    if expected_token_export == "sparse":
        expected_layers = _parse_token_layers_csv(args.token_layers)
        if manifest_token_layers != expected_layers:
            raise RuntimeError(
                f"existing eval_export token layers mismatch: {manifest_token_layers!r} != {expected_layers!r}"
            )

    split_result = dict(splits[SPLIT_NAME])
    if str(split_result.get("split", "")) != SPLIT_NAME:
        raise RuntimeError(f"existing eval_export split payload mismatch: {split_result.get('split')!r}")
    paths = dict(split_result.get("paths") or manifest.get("outputs") or {})
    required_path_keys = (
        "logits",
        "embeddings",
        "predictions",
        "metrics",
        "group_metrics",
        "per_class_metrics",
        "geometry_metrics",
    )
    checked_hashes: dict[str, str] = {}
    for key in required_path_keys:
        path_value = paths.get(key)
        if not path_value:
            raise RuntimeError(f"existing eval_export missing output path {key!r}")
        resolved_path = _resolve_resumable_export_file(path_value, manifest_path=manifest_path, key=key)
        paths[key] = _path_str(resolved_path)
        checked_hashes[key] = _assert_existing_file_matches_manifest(manifest, resolved_path, name=key)

    token_audit: dict[str, Any] = {}
    if expected_token_export != "none":
        token_index_path = paths.get("token_features_index")
        if not token_index_path:
            raise RuntimeError("existing eval_export missing token_features_index for token export")
        resolved_index_path = _resolve_resumable_export_file(
            token_index_path,
            manifest_path=manifest_path,
            key="token_features_index",
        )
        paths["token_features_index"] = _path_str(resolved_index_path)
        token_audit = _validate_existing_token_index(manifest, resolved_index_path)

    metric_rows = _read_metric_tables_from_export(paths)
    _classification_payload(metric_rows)
    _global_metric_payload(metric_rows)

    paths["manifest"] = _path_str(manifest_path)
    split_result["paths"] = paths
    summary = {
        "run_identity_hash": str(manifest.get("run_identity_hash") or split_result.get("run_identity_hash") or ""),
        "run_manifest_hash": str(manifest.get("run_manifest_hash") or ""),
        "manifest_path": _path_str(manifest_path),
        "split": split_result,
    }
    if not summary["run_identity_hash"]:
        raise RuntimeError("existing eval_export missing run_identity_hash")
    return summary, {
        "status": "resumed",
        "manifest": _path_str(manifest_path),
        "run_identity_hash": summary["run_identity_hash"],
        "checked_output_sha256": checked_hashes,
        **token_audit,
    }


def _load_resumable_candidate_export(
    *,
    spec: CandidateSpec,
    args: argparse.Namespace,
    dataset_paths_obj: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = spec.eval_export_dir / "run_manifest.json"
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"resumable QVF eval_export manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    return _validate_existing_export_manifest(
        spec=spec,
        args=args,
        dataset_paths_obj=dataset_paths_obj,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def _export_candidate_eval(
    *,
    spec: CandidateSpec,
    args: argparse.Namespace,
    paths: DatasetPaths,
    dataset_root: Path,
    reference_root: Path,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    manifest_path = spec.eval_export_dir / "run_manifest.json"
    if _resume_enabled(args):
        if manifest_path.exists():
            summary, audit = _load_resumable_candidate_export(
                spec=spec,
                args=args,
                dataset_paths_obj=paths,
            )
            print(f"[qvf-export] resume beta={spec.beta.raw} eval_export={spec.eval_export_dir}")
            return summary, False, audit
        print(f"[qvf-export] generate beta={spec.beta.raw}; no resumable eval_export manifest")
    else:
        if manifest_path.exists() and not bool(getattr(args, "force_export", False)):
            raise RuntimeError(
                f"existing eval_export manifest found for beta={spec.beta.raw}: {manifest_path}. "
                "Use default --resume to validate/reuse it, or pass --force-export to overwrite it."
            )
        print(f"[qvf-export] generate beta={spec.beta.raw}; resume disabled")
    summary = run_export_eval_config(
        ExportEvalConfig(
            model_name=_candidate_export_model_name(spec),
            checkpoint_path=spec.checkpoint_path,
            split_csv=paths.eval_csv,
            split_name=SPLIT_NAME,
            output_dir=spec.eval_export_dir,
            label_mapping_path=paths.label_mapping,
            bucket_protocol=paths.train_csv,
            pooling="mask_aware_mean",
            token_export=str(args.token_export),
            token_layers=str(args.token_layers),
            device=args.device,
            seed=int(args.seed),
            dataset_root=dataset_root,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            reference_root=reference_root,
            qparams_csv_override=_default_qparams_csv(dataset_root, int(args.seed)),
        )
    )
    validated_summary, validation_audit = _load_resumable_candidate_export(
        spec=spec,
        args=args,
        dataset_paths_obj=paths,
    )
    validation_audit = {
        **dict(validation_audit),
        "status": "generated",
        "validated_after_generation": True,
        "eval_export_dir": _path_str(spec.eval_export_dir),
    }
    if str(validated_summary.get("run_identity_hash") or "") != str(summary.get("run_identity_hash") or ""):
        raise RuntimeError("generated QVF eval_export run_identity_hash changed during post-generation validation")
    return validated_summary, True, validation_audit


def _candidate_row(
    *,
    spec: CandidateSpec,
    export_result: Mapping[str, Any],
    baseline_payload: Mapping[str, float],
    reference_root: Path,
    mechanism_available: bool,
) -> dict[str, Any]:
    split = export_result.get("split")
    if not isinstance(split, Mapping):
        raise KeyError("candidate export result missing split")
    paths = split.get("paths")
    if not isinstance(paths, Mapping):
        raise KeyError("candidate export split missing paths")
    metric_rows = _read_metric_tables_from_export(paths)
    classification = _classification_payload(metric_rows)
    metrics = _global_metric_payload(metric_rows)
    metrics.update(_geometry_extra_from_export(paths, reference_root=reference_root))

    delta_overall = classification["overall_accuracy"] - baseline_payload["overall_accuracy"]
    delta_macro = classification["macro_f1_label_universe_nanmean"] - baseline_payload["macro_f1_label_universe_nanmean"]
    delta_many = classification["many_accuracy"] - baseline_payload["many_accuracy"]
    failures: list[str] = []
    if delta_overall < OVERALL_ACC_GUARDRAIL:
        failures.append("delta_overall_accuracy")
    if delta_macro < OVERALL_MACRO_F1_GUARDRAIL:
        failures.append("delta_macro_f1_label_universe_nanmean")
    if delta_many < MANY_ACC_GUARDRAIL:
        failures.append("delta_many_accuracy")

    geometry_required = {name: metrics.get(name, float("nan")) for name in REQUIRED_GEOMETRY_METRICS}
    geometry_all_finite = all(math.isfinite(float(value)) for value in geometry_required.values())
    return {
        "candidate_index": int(spec.index),
        "selected": False,
        "passes_guardrail": not failures,
        "guardrail_failure_reasons": ";".join(failures),
        "qvf_beta": float(spec.beta.value),
        "candidate_dir": _path_str(spec.candidate_dir),
        "checkpoint": _path_str(spec.checkpoint_path),
        "checkpoint_sha256": file_sha256(spec.checkpoint_path),
        "train_log": _path_str(spec.train_log_path) if spec.train_log_path.exists() else "",
        "train_log_sha256": file_sha256(spec.train_log_path),
        "candidate_manifest": _path_str(spec.manifest_path) if spec.manifest_path.exists() else "",
        "candidate_manifest_sha256": file_sha256(spec.manifest_path),
        "eval_export_dir": _path_str(spec.eval_export_dir),
        "eval_export_run_identity_hash": str(export_result.get("run_identity_hash") or split.get("run_identity_hash") or ""),
        "overall_accuracy": classification["overall_accuracy"],
        "macro_f1_label_universe_nanmean": classification["macro_f1_label_universe_nanmean"],
        "many_accuracy": classification["many_accuracy"],
        "many_macro_f1": classification["many_macro_f1"],
        "medium_accuracy": classification["medium_accuracy"],
        "medium_macro_f1": classification["medium_macro_f1"],
        "tail_accuracy": classification["tail_accuracy"],
        "tail_macro_f1": classification["tail_macro_f1"],
        "few_accuracy": classification["few_accuracy"],
        "few_f1": classification["few_f1"],
        "delta_overall_accuracy_vs_vanilla_kd": delta_overall,
        "delta_macro_f1_label_universe_nanmean_vs_vanilla_kd": delta_macro,
        "delta_many_accuracy_vs_vanilla_kd": delta_many,
        "embedding_distortion": metrics["embedding_distortion"],
        "embedding_cosine_similarity": metrics["embedding_cosine_similarity"],
        "fisher_retention": metrics["fisher_retention"],
        "geo_margin_retention": metrics["geo_margin_retention"],
        "variance_deviation_abs_log": metrics["variance_deviation_abs_log"],
        "variance_deviation_nonfinite_class_count": metrics["variance_deviation_nonfinite_class_count"],
        "tail_collapse_gap": metrics["tail_collapse_gap"],
        "few_collapse_gap": metrics["few_collapse_gap"],
        "geometry_all_required_finite": geometry_all_finite,
        "mechanism_summary_available": bool(mechanism_available),
        "test_split_touched": False,
    }


def _select_best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    feasible = [row for row in rows if bool(row.get("passes_guardrail"))]
    if not feasible:
        return None

    def key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row.get(SELECTION_METRIC, float("nan"))),
            float(row.get("tail_accuracy", float("nan"))),
            float(row.get("medium_macro_f1", float("nan"))),
            float(row.get("medium_accuracy", float("nan"))),
            -float(row.get("qvf_beta", float("inf"))),
        )

    return max(feasible, key=key)


def _str_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _analysis_groups_for_bucket(bucket: str) -> tuple[str, ...]:
    canonical = validate_bucket_values([bucket], context="qvf_mechanism.bucket")[0]
    if canonical == "many":
        return ("many",)
    if canonical == "medium":
        return ("medium", "tail")
    if canonical == "few":
        return ("few", "tail")
    raise AssertionError("unreachable bucket")


def _source_files(token_survival_dir: Path, *, layers: Sequence[int], quant_modes: Sequence[str]) -> tuple[str, list[Path]]:
    combined = token_survival_dir / "token_survival_metrics.csv"
    shard_dir = token_survival_dir / "shards"
    if shard_dir.exists():
        files: list[Path] = []
        for quant_mode in quant_modes:
            for layer_idx in layers:
                pattern = (
                    f"eval__{quant_mode}__intervene_L{int(layer_idx):02d}_{MECHANISM_MODULE_NAME}"
                    f"__trace_L{int(layer_idx):02d}_{MECHANISM_MODULE_NAME}_input_activation__shard*.csv"
                )
                matched = sorted(shard_dir.glob(pattern))
                if not matched:
                    raise FileNotFoundError(
                        "QVF mechanism target shard missing while shards directory exists; refusing to fall back "
                        f"to possibly stale combined CSV. missing_pattern={pattern}, combined_csv={combined}"
                    )
                files.extend(matched)
        return "target_shards_csv", list(dict.fromkeys(files))
    if combined.exists():
        return "combined_metrics_csv_no_shards_dir", [combined]
    raise FileNotFoundError(
        f"QVF mechanism token survival source missing: neither {combined} nor {shard_dir} exists"
    )


def _is_mechanism_target_row(row: Mapping[str, Any], *, layer_idx: int, quant_mode: str) -> bool:
    if str(row.get("quant_mode", "")).strip() != str(quant_mode):
        return False
    if int(float(str(row.get("layer_idx", "-1")).strip())) != int(layer_idx):
        return False
    if str(row.get("module_name", "")).strip() != MECHANISM_MODULE_NAME:
        return False
    if str(row.get("tensor_position", "")).strip() != "input_activation":
        return False
    if str(row.get("intervention_module_name", "")).strip() != MECHANISM_MODULE_NAME:
        return False
    if int(float(str(row.get("intervention_layer_idx", "-1")).strip())) != int(layer_idx):
        return False
    if int(float(str(row.get("trace_layer_idx", "-1")).strip())) != int(layer_idx):
        return False
    expected_trace = f"Layer{int(layer_idx)}.fc1.input"
    if str(row.get("trace_target_name", "")).strip() != expected_trace:
        return False
    if str(row.get("trace_scope", "")).strip() != "input_activation_mechanism":
        return False
    if str(row.get("metric_role", "")).strip() != "main":
        return False
    if str(row.get("topk_unit", "")).strip() != "token":
        return False
    if str(row.get("topk_source", "")).strip() != "fp32_token_energy":
        return False
    if str(row.get("topk_selected_on", "")).strip() != "fp32":
        return False
    if not _str_bool(row.get("metric_valid_for_paper", False)):
        return False
    return True


def _iter_csv_rows(paths: Sequence[Path]) -> Iterable[tuple[Path, dict[str, str]]]:
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            missing = ["sample_id", "bucket", *MECHANISM_METRICS.values()]
            absent = [column for column in missing if column not in fieldnames]
            if absent:
                raise KeyError(f"QVF mechanism source missing columns {absent}: {path}")
            for row in reader:
                yield path, dict(row)


def _collect_mechanism_rows(
    *,
    dataset_name: str,
    seed: int,
    model_role: str,
    beta: float | None,
    token_survival_dir: Path,
    layers: Sequence[int],
    quant_modes: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_kind, sources = _source_files(token_survival_dir, layers=layers, quant_modes=quant_modes)
    accumulators: dict[tuple[int, str, str, str], NumericAccumulator] = {}
    row_counts: Counter[str] = Counter()
    protocol_fields: dict[tuple[int, str], dict[str, str]] = {}
    matched = 0
    for _path, row in _iter_csv_rows(sources):
        for layer_idx in layers:
            matched_layer = False
            for quant_mode in quant_modes:
                if not _is_mechanism_target_row(row, layer_idx=int(layer_idx), quant_mode=str(quant_mode)):
                    continue
                matched_layer = True
                matched += 1
                protocol_fields.setdefault(
                    (int(layer_idx), str(quant_mode)),
                    {
                        "topk_unit": str(row.get("topk_unit", "")),
                        "topk_source": str(row.get("topk_source", "")),
                        "topk_selected_on": str(row.get("topk_selected_on", "")),
                        "topk_ratio": str(row.get("topk_ratio", row.get("token_topk_ratio", ""))),
                        "topk": str(row.get("topk", "")),
                        "topk_k": str(row.get("topk_k", "")),
                        "topk_k_policy": str(row.get("topk_k_policy", "")),
                        "topk_policy": str(row.get("topk_policy", "")),
                        "padding_policy": str(row.get("valid_token_policy", row.get("mask_status", ""))),
                    },
                )
                bucket = validate_bucket_values([row.get("bucket")], context="qvf_mechanism.bucket")[0]
                expected_tail = bool(tail_included([bucket])[0])
                if "tail_included" in row and str(row.get("tail_included", "")).strip():
                    observed_tail = _str_bool(row.get("tail_included"))
                    if observed_tail != expected_tail:
                        raise ValueError("QVF mechanism row tail_included disagrees with bucket")
                for analysis_group in _analysis_groups_for_bucket(bucket):
                    for metric_name, source_column in MECHANISM_METRICS.items():
                        key = (int(layer_idx), str(quant_mode), str(analysis_group), str(metric_name))
                        accumulators.setdefault(key, NumericAccumulator.empty()).add(row.get(source_column))
                break
            if matched_layer:
                break
    if matched <= 0:
        raise RuntimeError(f"QVF mechanism source contains no strict target rows: {token_survival_dir}")
    rows: list[dict[str, Any]] = []
    for layer_idx in layers:
        for quant_mode in quant_modes:
            protocol = protocol_fields.get((int(layer_idx), str(quant_mode)), {})
            for analysis_group in MECHANISM_ANALYSIS_GROUPS:
                for metric_name in MECHANISM_METRICS:
                    key = (int(layer_idx), str(quant_mode), str(analysis_group), str(metric_name))
                    accumulator = accumulators.get(key, NumericAccumulator.empty())
                    value = accumulator.nanmean(
                        context=(
                            f"model_role={model_role}, beta={beta}, layer=L{layer_idx}.fc1, "
                            f"quant_mode={quant_mode}, group={analysis_group}, metric={metric_name}"
                        )
                    )
                    row_counts[f"L{layer_idx}.fc1:{quant_mode}:{analysis_group}:{metric_name}"] = accumulator.n_total
                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "seed": int(seed),
                            "model_role": model_role,
                            "qvf_beta": "" if beta is None else float(beta),
                            "source_dir": _path_str(token_survival_dir),
                            "source_kind": source_kind,
                            "layer": f"L{int(layer_idx)}.fc1",
                            "layer_idx": int(layer_idx),
                            "module_name": MECHANISM_MODULE_NAME,
                            "quant_mode": str(quant_mode),
                            "analysis_group": str(analysis_group),
                            "metric_name": str(metric_name),
                            "metric_value": value,
                            "n_total": accumulator.n_total,
                            "n_valid": accumulator.n_valid,
                            "topk_unit": protocol.get("topk_unit", ""),
                            "topk_source": protocol.get("topk_source", ""),
                            "topk_selected_on": protocol.get("topk_selected_on", ""),
                            "topk_ratio": protocol.get("topk_ratio", ""),
                            "topk": protocol.get("topk", ""),
                            "topk_k": protocol.get("topk_k", ""),
                            "topk_k_policy": protocol.get("topk_k_policy", ""),
                            "topk_policy": protocol.get("topk_policy", ""),
                            "padding_policy": protocol.get("padding_policy", ""),
                            "student_activation_semantic": "post_fake_quant_dequant_input_activation",
                        }
                    )
    return rows, {
        "source_kind": source_kind,
        "source_files": [_path_str(path) for path in sources],
        "source_file_sha256": {_path_str(path): file_sha256(path) for path in sources},
        "matched_rows": int(matched),
        "row_counts": dict(row_counts),
    }


def _require_close(name: str, actual: Any, expected: float, *, atol: float = 1e-12) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric, got {actual!r}") from exc
    if not math.isfinite(value) or not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=float(atol)):
        raise RuntimeError(f"{name} mismatch: actual={actual!r}, expected={expected!r}")


def _require_truthy(payload: Mapping[str, Any], key: str, *, context: str) -> None:
    if not _str_bool(payload.get(key)):
        raise RuntimeError(f"{context} must set {key}=true")


def _require_false(payload: Mapping[str, Any], key: str, *, context: str) -> None:
    if _str_bool(payload.get(key)):
        raise RuntimeError(f"{context} must set {key}=false")


def _path_from_payload(base: Path, payload: Mapping[str, Any], key: str) -> Path:
    value = payload.get(key)
    if value is None or not str(value).strip():
        raise RuntimeError(f"QVF manifest missing required path field {key!r}: {base}")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return base.parent / path


def _resolve_file_by_hash(recorded_path: Path, fallback_path: Path, expected: Any, *, name: str) -> Path:
    expected_hash = "" if expected is None else str(expected).strip()
    candidates = (Path(recorded_path), Path(fallback_path))
    mismatches: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        marker = str(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        if not candidate.exists() or not candidate.is_file():
            continue
        actual = file_sha256(candidate)
        if not expected_hash or str(actual) == expected_hash:
            return candidate.resolve(strict=False)
        mismatches.append(f"{candidate} actual={actual}")
    if mismatches:
        raise RuntimeError(f"{name} sha256 mismatch after path migration: expected={expected_hash}; checked={mismatches[:4]}")
    raise FileNotFoundError(
        f"{name} not found after path migration: "
        f"recorded={recorded_path.resolve(strict=False)} fallback={fallback_path.resolve(strict=False)}"
    )


def _require_file_hash(path: Path, expected: Any, *, name: str) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    actual = file_sha256(path)
    if expected is not None and str(expected).strip() and str(actual) != str(expected):
        raise RuntimeError(f"{name} sha256 mismatch: actual={actual}, expected={expected}")
    return str(actual)


def _validate_formal_beta_set(candidates: Sequence[CandidateSpec], formal_betas: Sequence[float]) -> None:
    values = sorted(float(item.beta.value) for item in candidates)
    expected = sorted(float(item) for item in formal_betas)
    if len(values) != len(expected) or any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(values, expected)):
        raise RuntimeError(
            f"Formal QVF export requires exactly beta sweep {expected}; got {[item.beta.raw for item in candidates]}"
        )


def _validate_qvf_train_log(
    spec: CandidateSpec,
    *,
    expected_method_variant: str,
    expected_protocol_version: str | None,
    expected_layer_weights: Mapping[str, float] | None,
    expected_weighted_variant: str | None,
) -> dict[str, Any]:
    if not spec.train_log_path.exists() or not spec.train_log_path.is_file():
        raise FileNotFoundError(f"missing QVF train log for beta={spec.beta.raw}: {spec.train_log_path}")
    rows, fieldnames = _read_csv_rows(spec.train_log_path)
    if not rows:
        raise RuntimeError(f"QVF train log is empty: {spec.train_log_path}")
    required = {
        "train_total_loss",
        "train_ce_loss",
        "train_kd_loss",
        "train_qvf_loss",
        "qvf_beta",
        "use_qvf_token_fidelity",
    }
    missing = sorted(required - set(fieldnames))
    if missing:
        raise RuntimeError(f"QVF train log missing required columns {missing}: {spec.train_log_path}")
    last = rows[-1]
    _require_close(f"train_log.qvf_beta beta={spec.beta.raw}", last.get("qvf_beta"), float(spec.beta.value))
    if not _str_bool(last.get("use_qvf_token_fidelity", "")):
        raise RuntimeError(f"QVF train log must set use_qvf_token_fidelity=true: {spec.train_log_path}")
    for key in ("train_total_loss", "train_ce_loss", "train_kd_loss", "train_qvf_loss"):
        value = _as_float_or_nan(last.get(key))
        if not math.isfinite(value):
            raise RuntimeError(f"QVF train log final {key} must be finite: {spec.train_log_path}")
    if expected_layer_weights:
        weighted_required = {
            "qvf_method_variant",
            "qvf_protocol_version",
            "qvf_layer_weight_policy",
            "qvf_weighted_variant",
            *{f"qvf_layer_weight_{layer.replace('.', '_')}" for layer in expected_layer_weights},
        }
        missing_weighted = sorted(weighted_required - set(fieldnames))
        if missing_weighted:
            raise RuntimeError(f"weighted QVF train log missing required columns {missing_weighted}: {spec.train_log_path}")
        if str(last.get("qvf_method_variant", "")).strip() != str(expected_method_variant):
            raise RuntimeError(f"QVF train log method variant mismatch: {spec.train_log_path}")
        if expected_protocol_version and str(last.get("qvf_protocol_version", "")).strip() != str(expected_protocol_version):
            raise RuntimeError(f"QVF train log protocol version mismatch: {spec.train_log_path}")
        if str(last.get("qvf_layer_weight_policy", "")).strip() != "weighted_sensitive_layer_mean":
            raise RuntimeError(f"QVF train log must record weighted_sensitive_layer_mean: {spec.train_log_path}")
        if expected_weighted_variant and str(last.get("qvf_weighted_variant", "")).strip() != str(expected_weighted_variant):
            raise RuntimeError(f"QVF train log weighted variant mismatch: {spec.train_log_path}")
        for layer, expected_weight in expected_layer_weights.items():
            _require_close(f"train_log.{layer}.weight beta={spec.beta.raw}", last.get(f"qvf_layer_weight_{layer.replace('.', '_')}"), float(expected_weight))
    return {
        "path": _path_str(spec.train_log_path),
        "sha256": file_sha256(spec.train_log_path),
        "num_rows": int(len(rows)),
        "columns": list(fieldnames),
    }


def _qvf_protocol_payload(payload: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    protocol = payload.get("qvf_protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError(f"{context} missing qvf_protocol mapping")
    return protocol


def _validate_qvf_topk_protocol(protocol: Mapping[str, Any], *, context: str) -> None:
    ratio_valid = False
    ratio_text = str(protocol.get("topk_ratio", "")).strip() if "topk_ratio" in protocol else ""
    if ratio_text:
        ratio = _as_float_or_nan(ratio_text)
        if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
            raise RuntimeError(f"{context} QVF topk_ratio must be finite in (0, 1], got {ratio_text!r}")
        if not str(protocol.get("topk_k_policy", "")).strip():
            raise RuntimeError(f"{context} QVF topk_ratio requires non-empty topk_k_policy")
        ratio_valid = True

    actual_k_valid = False
    for key in ("topk", "topk_k"):
        value_text = str(protocol.get(key, "")).strip() if key in protocol else ""
        if not value_text:
            continue
        value = _as_float_or_nan(value_text)
        if not math.isfinite(value) or value < 1.0 or not float(value).is_integer():
            raise RuntimeError(f"{context} QVF {key} must be a positive integer, got {value_text!r}")
        actual_k_valid = True

    if not ratio_valid and not actual_k_valid:
        raise RuntimeError(f"{context} QVF protocol must record a valid topk_ratio or actual K")


def _require_qvf_weight_payload(name: str, actual: Any, expected: Mapping[str, float]) -> None:
    if not isinstance(actual, Mapping):
        raise RuntimeError(f"{name} must be a mapping, got {type(actual)!r}")
    actual_keys = set(str(key) for key in actual.keys())
    expected_keys = set(str(key) for key in expected.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(f"{name} keys mismatch: actual={sorted(actual_keys)!r}, expected={sorted(expected_keys)!r}")
    for key, expected_value in expected.items():
        _require_close(f"{name}.{key}", actual.get(key), float(expected_value))


def _validate_qvf_protocol_fields(
    payload: Mapping[str, Any],
    *,
    beta: float,
    context: str,
    expected_method_variant: str,
    expected_protocol_version: str | None,
    expected_layer_weights: Mapping[str, float] | None,
    expected_weighted_variant: str | None,
) -> None:
    _require_truthy(payload, "use_qvf_token_fidelity", context=context)
    _require_close(f"{context}.qvf_beta", payload.get("qvf_beta"), float(beta))
    if str(payload.get("method_variant", "")) != str(expected_method_variant):
        raise RuntimeError(f"{context} method_variant mismatch: {payload.get('method_variant')!r}")
    protocol = _qvf_protocol_payload(payload, context=context)
    _require_truthy(protocol, "use_qvf_token_fidelity", context=f"{context}.qvf_protocol")
    _require_close(f"{context}.qvf_protocol.qvf_beta", protocol.get("qvf_beta"), float(beta))
    if str(protocol.get("method_variant", "")) != str(expected_method_variant):
        raise RuntimeError(f"{context} qvf_protocol.method_variant mismatch: {protocol.get('method_variant')!r}")
    if expected_protocol_version and str(protocol.get("protocol_version", "")) != str(expected_protocol_version):
        raise RuntimeError(f"{context} qvf_protocol.protocol_version mismatch: {protocol.get('protocol_version')!r}")
    if str(protocol.get("topk_unit", "")) != "token":
        raise RuntimeError(f"{context} QVF topk_unit must be token")
    if str(protocol.get("topk_source", "")) != "fp32_token_energy":
        raise RuntimeError(f"{context} QVF topk_source must be fp32_token_energy")
    if str(protocol.get("topk_selected_on", "")) != "fp32":
        raise RuntimeError(f"{context} QVF topk_selected_on must be fp32")
    _validate_qvf_topk_protocol(protocol, context=context)
    student_semantic = str(protocol.get("student_activation_semantic", payload.get("student_activation_semantic", "")))
    if "post_fake_quant" not in student_semantic or "autograd" not in student_semantic:
        raise RuntimeError(f"{context} student activation semantic must be post-fake-quant with autograd graph")
    sensitive = list(protocol.get("sensitive_layer_list") or payload.get("sensitive_layer_list") or [])
    expected_sensitive = [f"L{layer}.{MECHANISM_MODULE_NAME}" for layer in SENSITIVE_TRAIN_LAYERS]
    if [str(item) for item in sensitive] != expected_sensitive:
        raise RuntimeError(f"{context} sensitive layer list mismatch: {sensitive!r} != {expected_sensitive!r}")
    exact_paths = list(protocol.get("exact_module_paths") or payload.get("exact_module_paths") or [])
    expected_paths = [f"encoder.layers.{layer}.{MECHANISM_MODULE_NAME}" for layer in SENSITIVE_TRAIN_LAYERS]
    if [str(item) for item in exact_paths] != expected_paths:
        raise RuntimeError(f"{context} exact module paths mismatch: {exact_paths!r} != {expected_paths!r}")
    if expected_layer_weights:
        if str(protocol.get("layer_weight_policy", "")) != "weighted_sensitive_layer_mean":
            raise RuntimeError(f"{context} qvf_protocol.layer_weight_policy must be weighted_sensitive_layer_mean")
        if expected_weighted_variant:
            if str(payload.get("qvf_weighted_variant", "")) != str(expected_weighted_variant):
                raise RuntimeError(f"{context} qvf_weighted_variant mismatch: {payload.get('qvf_weighted_variant')!r}")
            if str(protocol.get("qvf_weighted_variant", "")) != str(expected_weighted_variant):
                raise RuntimeError(f"{context} qvf_protocol.qvf_weighted_variant mismatch: {protocol.get('qvf_weighted_variant')!r}")
        _require_qvf_weight_payload(f"{context}.sensitive_layer_weights", payload.get("sensitive_layer_weights"), expected_layer_weights)
        _require_qvf_weight_payload(f"{context}.qvf_layer_weights", payload.get("qvf_layer_weights"), expected_layer_weights)
        _require_qvf_weight_payload(f"{context}.qvf_protocol.sensitive_layer_weights", protocol.get("sensitive_layer_weights"), expected_layer_weights)
        _require_qvf_weight_payload(f"{context}.qvf_protocol.qvf_layer_weights", protocol.get("qvf_layer_weights"), expected_layer_weights)
    _require_false(payload, "test_split_touched", context=context)


def _validate_qvf_manifest_hashes(
    spec: CandidateSpec,
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    paths: Any,
) -> dict[str, Any]:
    checkpoint_sha256 = _require_file_hash(spec.checkpoint_path, manifest.get("checkpoint_sha256"), name="QVF checkpoint")
    _require_equal = {
        "checkpoint_sha256": checkpoint_sha256,
        "label_mapping_hash": file_sha256(paths.label_mapping),
    }
    for key, expected in _require_equal.items():
        actual = manifest.get(key)
        if str(actual or "") != str(expected or ""):
            raise RuntimeError(f"QVF manifest {key} mismatch: actual={actual!r}, expected={expected!r}")
    if str(checkpoint.get("qparams_hash") or "") != str(manifest.get("qparams_hash") or ""):
        raise RuntimeError("QVF checkpoint qparams_hash differs from qvf_manifest.json")
    qparams_identity_hash = str(manifest.get("qparams_hash") or checkpoint.get("qparams_hash") or "")
    if not qparams_identity_hash.strip():
        raise RuntimeError("QVF candidate must record non-empty qparams_hash")

    train_hashes = manifest.get("split_hashes")
    if not isinstance(train_hashes, Mapping):
        raise RuntimeError(f"QVF manifest missing split_hashes mapping: {spec.manifest_path}")
    expected_splits = {
        "train": file_sha256(paths.train_csv),
        "eval": file_sha256(paths.eval_csv),
    }
    for key, expected in expected_splits.items():
        if str(train_hashes.get(key) or "") != str(expected or ""):
            raise RuntimeError(f"QVF manifest split_hashes.{key} mismatch: actual={train_hashes.get(key)!r}, expected={expected!r}")
    if str(train_hashes.get("test", "")).strip():
        raise RuntimeError("QVF manifest must not record test split hash")

    seed = int(manifest.get("seed", DEFAULT_SEED))
    paths_root = getattr(paths, "root", None)
    dataset_root = Path(paths_root) if paths_root is not None else _dataset_root(None, None)
    teacher_path = _resolve_file_by_hash(
        _path_from_payload(spec.manifest_path, manifest, "teacher_checkpoint"),
        _default_teacher_checkpoint(dataset_root, seed),
        manifest.get("teacher_checkpoint_sha256"),
        name="QVF teacher checkpoint",
    )
    teacher_sha = _require_file_hash(teacher_path, manifest.get("teacher_checkpoint_sha256"), name="QVF teacher checkpoint")
    qparams_csv = _resolve_file_by_hash(
        _path_from_payload(spec.manifest_path, manifest, "qparams_csv"),
        _default_qparams_csv(dataset_root, seed),
        manifest.get("qparams_csv_sha256"),
        name="QVF qparams CSV",
    )
    qparams_sha = _require_file_hash(qparams_csv, manifest.get("qparams_csv_sha256"), name="QVF qparams CSV")
    int4_path = _resolve_file_by_hash(
        _path_from_payload(spec.manifest_path, manifest, "int4_calibrated_student_checkpoint"),
        _default_int4_checkpoint(dataset_root, seed),
        manifest.get("int4_calibrated_student_checkpoint_sha256") or manifest.get("source_int4_init_hash"),
        name="QVF INT4 init checkpoint",
    )
    int4_sha = _require_file_hash(
        int4_path,
        manifest.get("int4_calibrated_student_checkpoint_sha256") or manifest.get("source_int4_init_hash"),
        name="QVF INT4 init checkpoint",
    )
    calibration_path = _resolve_file_by_hash(
        _path_from_payload(spec.manifest_path, manifest, "calibration_csv"),
        _default_calibration_csv(dataset_root),
        manifest.get("calibration_csv_hash"),
        name="QVF calibration CSV",
    )
    calibration_sha = _require_file_hash(calibration_path, manifest.get("calibration_csv_hash"), name="QVF calibration CSV")
    if str(manifest.get("source_int4_init_hash") or "") != str(int4_sha):
        raise RuntimeError("QVF manifest source_int4_init_hash must equal INT4 calibrated student checkpoint sha256")
    if str(checkpoint.get("teacher_checkpoint_sha256") or "") != str(teacher_sha):
        raise RuntimeError("QVF checkpoint teacher hash differs from qvf_manifest.json")
    if str(checkpoint.get("qparams_csv_sha256") or "") != str(qparams_sha):
        raise RuntimeError("QVF checkpoint qparams CSV hash differs from qvf_manifest.json")
    if str(checkpoint.get("calibration_csv_hash") or "") != str(calibration_sha):
        raise RuntimeError("QVF checkpoint calibration CSV hash differs from qvf_manifest.json")
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "teacher_checkpoint_sha256": teacher_sha,
        "qparams_csv_sha256": qparams_sha,
        "qparams_hash": qparams_identity_hash,
        "int4_init_sha256": int4_sha,
        "calibration_csv_sha256": calibration_sha,
        "split_hashes": dict(train_hashes),
    }


def _validate_qvf_test_isolation(manifest: Mapping[str, Any], *, context: str) -> None:
    if "test_split_touched" not in manifest:
        raise RuntimeError(f"{context} must explicitly record test_split_touched=false")
    _require_false(manifest, "test_split_touched", context=context)
    isolation = manifest.get("test_isolation")
    if isinstance(isolation, Mapping):
        for key in (
            "test_split_touched",
            "test_csv_read",
            "test_loader_built",
            "test_forward_run",
            "test_metrics_exported",
            "test_predictions_exported",
            "test_path_recorded",
        ):
            _require_false(isolation, key, context=f"{context}.test_isolation")
    forbidden = ("test_csv", "test_path", "test_hash", "test_predictions", "test_metrics")
    for key in forbidden:
        if key in manifest and str(manifest.get(key) or "").strip():
            raise RuntimeError(f"{context} must not record {key}")


def _validate_cross_candidate_consistency(audits_by_beta: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not audits_by_beta:
        raise RuntimeError("QVF cross-candidate consistency requires at least one candidate")
    ordered = [(str(beta), dict(audit)) for beta, audit in audits_by_beta.items()]
    reference_beta, reference_audit = ordered[0]
    reference_hashes = reference_audit.get("hashes")
    if not isinstance(reference_hashes, Mapping):
        raise RuntimeError(f"QVF candidate beta={reference_beta} missing hash audit payload")
    hash_keys = (
        "teacher_checkpoint_sha256",
        "qparams_csv_sha256",
        "qparams_hash",
        "int4_init_sha256",
        "calibration_csv_sha256",
    )
    for beta, audit in ordered[1:]:
        hashes = audit.get("hashes")
        if not isinstance(hashes, Mapping):
            raise RuntimeError(f"QVF candidate beta={beta} missing hash audit payload")
        for key in hash_keys:
            if str(hashes.get(key) or "") != str(reference_hashes.get(key) or ""):
                raise RuntimeError(
                    f"QVF candidates must share {key}; "
                    f"beta={reference_beta} has {reference_hashes.get(key)!r}, beta={beta} has {hashes.get(key)!r}"
                )
        reference_splits = reference_hashes.get("split_hashes")
        splits = hashes.get("split_hashes")
        if not isinstance(reference_splits, Mapping) or not isinstance(splits, Mapping):
            raise RuntimeError("QVF candidate hash audit missing split_hashes mapping")
        for split_name in ("train", "eval"):
            if str(splits.get(split_name) or "") != str(reference_splits.get(split_name) or ""):
                raise RuntimeError(
                    f"QVF candidates must share split_hashes.{split_name}; "
                    f"beta={reference_beta} has {reference_splits.get(split_name)!r}, "
                    f"beta={beta} has {splits.get(split_name)!r}"
                )
    return {
        "status": "passed",
        "reference_beta": reference_beta,
        "checked_betas": [beta for beta, _audit in ordered],
        "shared_hash_keys": list(hash_keys),
        "shared_split_hashes": ["train", "eval"],
    }


def _validate_candidate_artifacts(
    candidates: Sequence[CandidateSpec],
    *,
    paths: Any,
    formal_betas: Sequence[float],
    expected_method_variant: str,
    expected_protocol_version: str | None,
    expected_layer_weights: Mapping[str, float] | None,
    expected_weighted_variant: str | None,
) -> dict[str, Any]:
    _validate_formal_beta_set(candidates, formal_betas)
    missing = [f"beta={item.beta.raw}: {item.checkpoint_path}" for item in candidates if not item.checkpoint_path.exists()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"missing QVF candidate checkpoint(s):\n  {joined}")
    non_files = [f"beta={item.beta.raw}: {item.checkpoint_path}" for item in candidates if not item.checkpoint_path.is_file()]
    if non_files:
        joined = "\n  ".join(non_files)
        raise FileNotFoundError(f"QVF candidate checkpoints must be files:\n  {joined}")
    audits: dict[str, Any] = {}
    for spec in candidates:
        train_log_audit = _validate_qvf_train_log(
            spec,
            expected_method_variant=expected_method_variant,
            expected_protocol_version=expected_protocol_version,
            expected_layer_weights=expected_layer_weights,
            expected_weighted_variant=expected_weighted_variant,
        )
        manifest = _read_json(spec.manifest_path)
        checkpoint = _load_torch_mapping(spec.checkpoint_path, map_location="cpu")
        _validate_qvf_protocol_fields(
            manifest,
            beta=float(spec.beta.value),
            context=f"qvf_manifest beta={spec.beta.raw}",
            expected_method_variant=expected_method_variant,
            expected_protocol_version=expected_protocol_version,
            expected_layer_weights=expected_layer_weights,
            expected_weighted_variant=expected_weighted_variant,
        )
        _validate_qvf_protocol_fields(
            checkpoint,
            beta=float(spec.beta.value),
            context=f"qvf_checkpoint beta={spec.beta.raw}",
            expected_method_variant=expected_method_variant,
            expected_protocol_version=expected_protocol_version,
            expected_layer_weights=expected_layer_weights,
            expected_weighted_variant=expected_weighted_variant,
        )
        _validate_qvf_test_isolation(manifest, context=f"qvf_manifest beta={spec.beta.raw}")
        hashes = _validate_qvf_manifest_hashes(spec, manifest, checkpoint, paths=paths)
        audits[spec.beta.raw] = {
            "train_log": train_log_audit,
            "manifest": _path_str(spec.manifest_path),
            "manifest_sha256": file_sha256(spec.manifest_path),
            "hashes": hashes,
        }
    return {
        "by_beta": audits,
        "cross_candidate_consistency": _validate_cross_candidate_consistency(audits),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QVF Stage1 eval-only formal export and mechanism summary. This script does not train and never reads test.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-root", default=None, help="Default: <dataset_root>/innovation/QVF.")
    parser.add_argument("--betas", default=",".join(DEFAULT_BETAS))
    parser.add_argument("--formal-betas", default=None, help="Expected formal beta set. Default: 0.01,0.05,0.10.")
    parser.add_argument("--expected-method-variant", default=METHOD_VARIANT)
    parser.add_argument("--expected-protocol-version", default=None)
    parser.add_argument("--expected-layer-weights", default=None, help="Expected weighted QVF layer weights, e.g. 1:0.25,9:0.75.")
    parser.add_argument("--expected-qvf-weighted-variant", default=None)
    parser.add_argument("--checkpoint-name", default=None, help="Candidate checkpoint filename inside each beta dir. Default searches qvf_stage1_checkpoint.pt, stage1_checkpoint.pt, checkpoint.pt, best_checkpoint.pt.")
    parser.add_argument("--vanilla-kd-metrics", default=None)
    parser.add_argument("--vanilla-kd-manifest", default=None)
    parser.add_argument("--vanilla-kd-checkpoint", default=None)
    parser.add_argument("--vanilla-token-survival-dir", default=None)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--token-export", choices=("all", "sparse", "none"), default=DEFAULT_TOKEN_EXPORT)
    parser.add_argument("--token-layers", default=DEFAULT_TOKEN_LAYERS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-export", action="store_true", help="Regenerate candidate eval exports even when complete resumable exports exist.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Reuse complete validated candidate eval exports when available.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resumable candidate eval export reuse.")
    parser.add_argument("--skip-mechanism-summary", action="store_true")
    parser.add_argument("--allow-nonfinite-geometry", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run_qvf_export(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = _dataset_root(args.dataset, args.dataset_root)
    paths = _dataset_paths_for_root(args.dataset, args.dataset_root)
    seed_dir = _seed_dir(dataset_root, int(args.seed), args.output_root)
    formal_betas = tuple(float(item.value) for item in _parse_betas(args.formal_betas)) if args.formal_betas else tuple(float(item) for item in FORMAL_QVF_BETAS)
    expected_layer_weights = _parse_expected_layer_weights(args.expected_layer_weights)
    expected_weighted_variant = str(args.expected_qvf_weighted_variant or "").strip()
    if expected_layer_weights and not expected_weighted_variant:
        expected_weighted_variant = _qvf_weighted_variant(expected_layer_weights)
    expected_method_variant = str(args.expected_method_variant or METHOD_VARIANT)
    expected_protocol_version = str(args.expected_protocol_version or "").strip() or None
    reference_root = Path(args.reference_root) if args.reference_root else _default_reference_root(dataset_root, int(args.seed))
    vanilla_metrics_path = Path(args.vanilla_kd_metrics) if args.vanilla_kd_metrics else _default_vanilla_metrics(dataset_root, int(args.seed))
    vanilla_manifest_path = Path(args.vanilla_kd_manifest) if args.vanilla_kd_manifest else _default_vanilla_manifest(dataset_root, int(args.seed))
    vanilla_checkpoint = Path(args.vanilla_kd_checkpoint) if args.vanilla_kd_checkpoint else _default_vanilla_checkpoint(dataset_root, int(args.seed))
    vanilla_token_survival_dir = Path(args.vanilla_token_survival_dir) if args.vanilla_token_survival_dir else _default_vanilla_token_survival_dir(dataset_root, int(args.seed))
    candidates = _candidate_specs(args, seed_dir)

    dry_run_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_name": paths.name,
        "dataset_root": _path_str(paths.root),
        "train_csv": _path_str(paths.train_csv),
        "eval_csv": _path_str(paths.eval_csv),
        "label_mapping": _path_str(paths.label_mapping),
        "seed": int(args.seed),
        "seed_dir": _path_str(seed_dir),
        "formal_betas": list(formal_betas),
        "expected_method_variant": expected_method_variant,
        "expected_protocol_version": expected_protocol_version,
        "expected_qvf_weighted_variant": expected_weighted_variant,
        "expected_layer_weights": dict(expected_layer_weights or {}),
        "resume_enabled": _resume_enabled(args),
        "force_export": bool(args.force_export),
        "reference_root": _path_str(reference_root),
        "vanilla_kd_metrics": _path_str(vanilla_metrics_path),
        "vanilla_kd_metrics_exists": vanilla_metrics_path.exists(),
        "vanilla_kd_manifest": _path_str(vanilla_manifest_path),
        "vanilla_kd_manifest_exists": vanilla_manifest_path.exists(),
        "vanilla_kd_checkpoint": _path_str(vanilla_checkpoint),
        "vanilla_kd_checkpoint_exists": vanilla_checkpoint.exists(),
        "vanilla_token_survival_dir": _path_str(vanilla_token_survival_dir),
        "vanilla_token_survival_dir_exists": vanilla_token_survival_dir.exists(),
        "candidate_plan": [
            {
                "qvf_beta": item.beta.value,
                "candidate_dir": _path_str(item.candidate_dir),
                "checkpoint": _path_str(item.checkpoint_path),
                "checkpoint_exists": item.checkpoint_path.exists(),
                "eval_export_dir": _path_str(item.eval_export_dir),
                "eval_export_manifest": _path_str(item.eval_export_dir / "run_manifest.json"),
                "eval_export_manifest_exists": (item.eval_export_dir / "run_manifest.json").exists(),
                "token_survival_dir": _path_str(item.token_survival_dir),
                "token_survival_dir_exists": item.token_survival_dir.exists(),
            }
            for item in candidates
        ],
        "test_split_touched": False,
    }
    if bool(args.dry_run):
        return dry_run_payload

    candidate_artifact_audits = _validate_candidate_artifacts(
        candidates,
        paths=paths,
        formal_betas=formal_betas,
        expected_method_variant=expected_method_variant,
        expected_protocol_version=expected_protocol_version,
        expected_layer_weights=expected_layer_weights,
        expected_weighted_variant=expected_weighted_variant,
    )
    baseline_rows = _baseline_eval_rows(
        vanilla_metrics_path,
        manifest_path=vanilla_manifest_path,
        dataset_root=dataset_root,
    )
    baseline_payload = _classification_payload(baseline_rows)

    summary_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    mechanism_manifest: dict[str, Any] = {}

    if not bool(args.skip_mechanism_summary):
        vanilla_mech, vanilla_mech_manifest = _collect_mechanism_rows(
            dataset_name=paths.name,
            seed=int(args.seed),
            model_role="vanilla_kd",
            beta=None,
            token_survival_dir=vanilla_token_survival_dir,
            layers=MECHANISM_EXPORT_LAYERS,
            quant_modes=MECHANISM_QUANT_MODES,
        )
        mechanism_rows.extend(vanilla_mech)
        mechanism_manifest["vanilla_kd"] = vanilla_mech_manifest

    generated_by_beta: dict[str, bool] = {}
    resume_audit_by_beta: dict[str, Mapping[str, Any]] = {}
    export_results: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        export_result, generated, resume_audit = _export_candidate_eval(
            spec=candidate,
            args=args,
            paths=paths,
            dataset_root=dataset_root,
            reference_root=reference_root,
        )
        export_results[candidate.beta.raw] = export_result
        generated_by_beta[candidate.beta.raw] = bool(generated)
        resume_audit_by_beta[candidate.beta.raw] = dict(resume_audit)
        candidate_mechanism_available = False
        if not bool(args.skip_mechanism_summary):
            candidate_mech, candidate_mech_manifest = _collect_mechanism_rows(
                dataset_name=paths.name,
                seed=int(args.seed),
                model_role="qvf_candidate",
                beta=float(candidate.beta.value),
                token_survival_dir=candidate.token_survival_dir,
                layers=MECHANISM_EXPORT_LAYERS,
                quant_modes=MECHANISM_QUANT_MODES,
            )
            mechanism_rows.extend(candidate_mech)
            mechanism_manifest[candidate.beta.raw] = candidate_mech_manifest
            candidate_mechanism_available = True
        row = _candidate_row(
            spec=candidate,
            export_result=export_result,
            baseline_payload=baseline_payload,
            reference_root=reference_root,
            mechanism_available=candidate_mechanism_available,
        )
        row["dataset_name"] = paths.name
        row["seed"] = int(args.seed)
        row["eval_export_generated"] = bool(generated)
        if not bool(row["geometry_all_required_finite"]) and not bool(args.allow_nonfinite_geometry):
            raise RuntimeError(f"QVF candidate beta={candidate.beta.raw} has non-finite required geometry metrics")
        summary_rows.append(row)

    best = _select_best(summary_rows)
    if best is not None:
        best_beta = float(best["qvf_beta"])
        for row in summary_rows:
            row["selected"] = float(row["qvf_beta"]) == best_beta

    summary_csv = seed_dir / "qvf_eval_summary.csv"
    mechanism_csv = seed_dir / "qvf_mechanism_summary.csv"
    manifest_path = seed_dir / "qvf_export_manifest.json"
    _write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    if not bool(args.skip_mechanism_summary):
        _write_csv(mechanism_csv, mechanism_rows, MECHANISM_COLUMNS)

    manifest_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "expected_training_protocol_version": expected_protocol_version,
        "dataset_name": paths.name,
        "seed": int(args.seed),
        "qvf_betas": [float(item.beta.value) for item in candidates],
        "formal_betas": list(formal_betas),
        "method_variant": expected_method_variant,
        "qvf_weighted_variant": expected_weighted_variant,
        "sensitive_layer_weights": dict(expected_layer_weights or {}),
        "candidate_checkpoint_sha256": {
            item.beta.raw: file_sha256(item.checkpoint_path)
            for item in candidates
        },
        "vanilla_kd_metrics_sha256": file_sha256(vanilla_metrics_path),
        "reference_root": _path_str(reference_root),
    }
    manifest = {
        **manifest_identity,
        "artifact_type": "qvf_stage1_eval_export_summary",
        "method_variant": expected_method_variant,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_identity_hash": _json_hash(manifest_identity),
        "output_dir": _path_str(seed_dir),
        "outputs": {
            "qvf_eval_summary_csv": _path_str(summary_csv),
            "qvf_mechanism_summary_csv": "" if bool(args.skip_mechanism_summary) else _path_str(mechanism_csv),
            "qvf_export_manifest_json": _path_str(manifest_path),
        },
        "qvf_loss_protocol": {
            "stage": "Stage1 Vanilla KD only",
            "training_implemented_here": False,
            "loss_definition": "L_CE + alpha * L_KD + beta * L_QVF",
            "use_qvf_token_fidelity": True,
            "method_variant": expected_method_variant,
            "expected_training_protocol_version": expected_protocol_version,
            "layer_weight_policy": "weighted_sensitive_layer_mean" if expected_layer_weights else "equal_sensitive_layer_mean",
            "qvf_weighted_variant": expected_weighted_variant,
            "sensitive_layer_weights": dict(expected_layer_weights or {}),
            "sensitive_layer_list_for_training": [f"L{layer}.fc1" for layer in SENSITIVE_TRAIN_LAYERS],
            "exact_module_paths": [f"encoder.layers.{layer}.fc1" for layer in SENSITIVE_TRAIN_LAYERS],
            "activation_hook_semantic": "FFN fc1 input activation",
            "student_activation_required_for_loss": "post_fake_quant_with_autograd_graph",
            "student_activation_summarized_here": "post_fake_quant_dequant_input_activation_from_token_survival_artifacts",
            "teacher_token_selection_source": "FP32 teacher token norm / fp32_token_energy; Top-K indices are fixed for student",
            "epsilon": float(DEFAULT_EPS),
        },
        "candidate_selection": {
            "selection_split": SPLIT_NAME,
            "baseline": "vanilla_kd",
            "guardrails": {
                "delta_overall_accuracy_min": OVERALL_ACC_GUARDRAIL,
                "delta_macro_f1_label_universe_nanmean_min": OVERALL_MACRO_F1_GUARDRAIL,
                "delta_many_accuracy_min": MANY_ACC_GUARDRAIL,
            },
            "primary": SELECTION_METRIC,
            "tie_break": list(TIE_BREAK_METRICS),
            "few_used_for_selection": False,
            "geometry_hard_gate": "finite-only",
            "selected_qvf_beta": None if best is None else float(best["qvf_beta"]),
            "selected_candidate_dir": "" if best is None else str(best["candidate_dir"]),
        },
        "test_isolation": {
            "test_split_touched": False,
            "test_csv_read": False,
            "test_loader_built": False,
            "test_forward_run": False,
            "test_metrics_exported": False,
            "test_predictions_exported": False,
            "test_path_recorded": False,
        },
        "formal_export": {
            "source": "src.baseline.export_eval.run_export_eval_config",
            "split": SPLIT_NAME,
            "canonical_reload_based_eval": True,
            "force_export": bool(args.force_export),
            "resume_enabled": _resume_enabled(args),
            "reuse_existing_eval_export": _resume_enabled(args),
            "generated_by_beta": generated_by_beta,
            "resume_audit_by_beta": resume_audit_by_beta,
            "token_export": str(args.token_export),
            "token_layers": str(args.token_layers),
        },
        "candidate_artifact_validation": candidate_artifact_audits,
        "mechanism_summary": {
            "skipped": bool(args.skip_mechanism_summary),
            "target_layers": [f"L{layer}.fc1" for layer in MECHANISM_EXPORT_LAYERS],
            "training_sensitive_layers_only": [f"L{layer}.fc1" for layer in SENSITIVE_TRAIN_LAYERS],
            "quant_modes": list(MECHANISM_QUANT_MODES),
            "metrics": dict(MECHANISM_METRICS),
            "topk_protocol_fields_reused": [
                "topk_unit",
                "topk_source",
                "topk_selected_on",
                "topk_ratio",
                "topk",
                "topk_k",
                "topk_k_policy",
                "topk_policy",
                "valid_token_policy",
            ],
            "sources": mechanism_manifest,
        },
        "paths": {
            "dataset_root": _path_str(dataset_root),
            "train_csv": _path_str(paths.train_csv),
            "eval_csv": _path_str(paths.eval_csv),
            "label_mapping": _path_str(paths.label_mapping),
            "reference_root": _path_str(reference_root),
            "vanilla_kd_metrics": _path_str(vanilla_metrics_path),
            "vanilla_kd_manifest": _path_str(vanilla_manifest_path) if vanilla_manifest_path.exists() else "",
            "vanilla_kd_checkpoint": _path_str(vanilla_checkpoint) if vanilla_checkpoint.exists() else "",
            "vanilla_token_survival_dir": "" if bool(args.skip_mechanism_summary) else _path_str(vanilla_token_survival_dir),
        },
        "artifact_sha256": {
            "qvf_eval_summary.csv": file_sha256(summary_csv),
            "qvf_mechanism_summary.csv": None if bool(args.skip_mechanism_summary) else file_sha256(mechanism_csv),
            "vanilla_kd_metrics.csv": file_sha256(vanilla_metrics_path),
            "vanilla_kd_manifest.json": file_sha256(vanilla_manifest_path),
            "vanilla_kd_checkpoint.pt": file_sha256(vanilla_checkpoint),
        },
        "summary": summary_rows,
    }
    manifest["run_manifest_hash"] = _json_hash(manifest)
    _write_json(manifest_path, manifest)
    return {
        "output_dir": seed_dir,
        "summary_csv": summary_csv,
        "mechanism_csv": None if bool(args.skip_mechanism_summary) else mechanism_csv,
        "manifest_path": manifest_path,
        "selected_qvf_beta": None if best is None else float(best["qvf_beta"]),
        "num_candidates": len(summary_rows),
        "num_resumed_candidates": sum(1 for value in generated_by_beta.values() if not bool(value)),
        "num_generated_candidates": sum(1 for value in generated_by_beta.values() if bool(value)),
        "num_mechanism_rows": len(mechanism_rows),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_qvf_export(args)
    if bool(args.dry_run):
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return result
    print(f"[qvf-export] output_dir={result['output_dir']}")
    print(f"[qvf-export] summary_csv={result['summary_csv']}")
    if result.get("mechanism_csv") is not None:
        print(f"[qvf-export] mechanism_csv={result['mechanism_csv']}")
    print(f"[qvf-export] manifest={result['manifest_path']}")
    print(f"[qvf-export] selected_qvf_beta={result['selected_qvf_beta']}")
    print(
        f"[qvf-export] candidates resumed={result['num_resumed_candidates']} "
        f"generated={result['num_generated_candidates']}"
    )
    return result


if __name__ == "__main__":
    main()
