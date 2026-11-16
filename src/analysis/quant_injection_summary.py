from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config


PROTOCOL_VERSION = "quant_injection_summary.v1"
ANALYSIS_GROUPS = ("many", "medium", "few", "tail")
BASE_BUCKETS = ("many", "medium", "few")
TAIL_BUCKETS = ("medium", "few")
DEFAULT_TOP_SENSITIVE_K = 8
DEFAULT_RANKING_CHUNKSIZE = 50000

LAYER_LEVEL_METRICS = (
    "margin_retention",
    "fisher_retention",
    "oer_retention",
    "projection_oer_retention",
    "embedding_distortion",
    "clipping_rate",
)

TOKEN_METRICS = (
    "token_oer_retention",
    "mean_norm_retention",
    "topk_survival_rate",
    "topk_overlap",
    "topk_token_exact_zero_rate",
    "topk_token_near_zero_pm1_rate",
    "high_energy_scalar_exact_zero_rate",
    "high_energy_scalar_near_zero_pm1_rate",
    "topk_token_boundary_hit_rate",
    "topk_token_strict_overflow_rate",
    "token_distortion",
    "embedding_distortion",
)

DANGER_ZONE_METRICS = (
    "oer_retention",
    "projection_oer_retention",
    "projection_rel_l2_error",
    "projection_cosine_drift",
    "output_energy_distortion",
    "exact_zero_rate",
    "near_zero_pm1_rate",
    "high_energy_near_zero_rate",
    "topk_token_exact_zero_rate",
    "topk_token_near_zero_pm1_rate",
    "topk_token_boundary_hit_rate",
    "topk_token_strict_overflow_rate",
    "high_energy_scalar_exact_zero_rate",
    "high_energy_scalar_near_zero_pm1_rate",
    "high_energy_scalar_boundary_hit_rate",
    "high_energy_scalar_strict_overflow_rate",
    "saturation_clip_energy_share",
    "key_token_clip_energy_share",
    "key_token_near_zero_energy_share",
    "key_token_near_zero_loss_share",
    "oer_scalar_clip_rate",
    "scale_mean",
    "scale_median",
    "scale_p95",
    "max_abs_over_p99",
    "p999_over_p99",
)

GEOMETRY_METRICS = (
    "fisher_retention",
    "margin_retention",
    "embedding_centroid_margin_retention",
    "center_cosine_margin_retention",
    "variance_inflation",
    "variance_inflation_deviation",
    "embedding_distortion",
    "fisher_damage",
    "margin_damage",
    "embedding_centroid_margin_damage",
)

ACTIVATION_DISTRIBUTION_METRICS = (
    "max_abs",
    "p99",
    "p999",
    "std",
    "kurtosis",
    "sparsity",
    "sparsity_ratio",
    "near_zero_rate",
    "near_zero_ratio",
    "outlier_ratio",
    "outlier_energy",
    "outlier_energy_ratio",
    "scale_mean",
    "scale_median",
    "scale_p95",
    "scale_max",
    "clipping_rate",
    "oer_retention",
    "fixed_per_tensor_scale",
    "fixed_clipping_rate_per_tensor",
    "fixed_oer_retention_per_tensor",
)

QUANT_MODE_ENABLED_FLAGS = {
    "A4-only": {"weight_qparams_enabled": False, "activation_qparams_enabled": True},
    "W4-only": {"weight_qparams_enabled": True, "activation_qparams_enabled": False},
    "W4A4": {"weight_qparams_enabled": True, "activation_qparams_enabled": True},
}

FRONT_COLUMNS = (
    "source_experiment",
    "source_file",
    "scope",
    "target",
    "target_name",
    "intervention_target",
    "intervention_layer_idx",
    "intervention_module_name",
    "trace_target_name",
    "trace_layer_idx",
    "trace_module_name",
    "layer_idx",
    "module_name",
    "tensor_position",
    "activation_kind",
    "quant_mode",
    "granularity",
    "metric_name",
    "metric_role",
    "metric_direction",
    "aggregation_unit",
    "aggregation_fn",
    "many_value",
    "medium_value",
    "few_value",
    "tail_value",
    "many_damage",
    "medium_damage",
    "few_damage",
    "tail_damage",
    "tail_minus_many_damage_gap",
    "few_minus_many_damage_gap",
    "tail_vs_many_gap",
    "few_vs_many_gap",
    "many_sample_count",
    "medium_sample_count",
    "few_sample_count",
    "tail_sample_count",
    "many_effective_class_count",
    "medium_effective_class_count",
    "few_effective_class_count",
    "tail_effective_class_count",
    "many_class_count",
    "medium_class_count",
    "few_class_count",
    "tail_class_count",
    "many_margin_valid_rate",
    "medium_margin_valid_rate",
    "few_margin_valid_rate",
    "tail_margin_valid_rate",
)


@dataclass(frozen=True)
class SummaryPaths:
    dataset_name: str
    dataset_root: Path
    output_dir: Path

    @property
    def ffn_layer_heatmap_summary(self) -> Path:
        return self.dataset_root / "ffn_layer_heatmap" / "ffn_layer_heatmap_summary.csv"

    @property
    def ffn_layer_heatmap_manifest(self) -> Path:
        return self.dataset_root / "ffn_layer_heatmap" / "ffn_layer_heatmap_manifest.json"

    @property
    def quant_mode_summary(self) -> Path:
        return self.dataset_root / "quant_mode_ablation" / "ffn_quant_mode_comparison_summary.csv"

    @property
    def quant_mode_detail(self) -> Path:
        return self.dataset_root / "quant_mode_ablation" / "ffn_quant_mode_ablation.csv"

    @property
    def quant_mode_qparams_summary(self) -> Path:
        return self.dataset_root / "quant_mode_ablation" / "ffn_quant_mode_qparams_summary.csv"

    @property
    def quant_mode_manifest(self) -> Path:
        return self.dataset_root / "quant_mode_ablation" / "ffn_quant_mode_ablation_manifest.json"

    @property
    def token_survival_summary(self) -> Path:
        return self.dataset_root / "token_survival" / "token_survival_comparison_summary.csv"

    @property
    def token_survival_metrics(self) -> Path:
        return self.dataset_root / "token_survival" / "token_survival_metrics.csv"

    @property
    def token_survival_manifest(self) -> Path:
        return self.dataset_root / "token_survival" / "token_survival_manifest.json"

    @property
    def danger_zone_summary(self) -> Path:
        return self.dataset_root / "quant_danger_zone" / "quant_danger_zone_comparison_summary.csv"

    @property
    def danger_zone_manifest(self) -> Path:
        return self.dataset_root / "quant_danger_zone" / "ffn_quant_danger_zone_manifest.json"

    @property
    def semantic_geometry_summary(self) -> Path:
        return self.dataset_root / "semantic_geometry" / "semantic_geometry_comparison_summary.csv"

    @property
    def semantic_geometry_global_summary(self) -> Path:
        return self.dataset_root / "semantic_geometry" / "semantic_summary_metrics.csv"

    @property
    def semantic_geometry_manifest(self) -> Path:
        return self.dataset_root / "semantic_geometry" / "semantic_geometry_projection_summary.json"

    @property
    def activation_distribution_summary(self) -> Path:
        return self.dataset_root / "ffn_activation_distribution_probe" / "ffn_activation_distribution_plot_summary.csv"

    @property
    def activation_distribution_manifest(self) -> Path:
        return self.dataset_root / "ffn_activation_distribution_probe" / "ffn_activation_distribution_manifest.json"

    @property
    def core_reference_manifest(self) -> Path:
        return self.dataset_root / "cache" / "fp32_reference" / "core_eval.pt.manifest.json"

    @property
    def oer_oracle_manifest(self) -> Path:
        return self.dataset_root / "cache" / "fp32_reference" / "oer_oracle_eval.pt.manifest.json"

    @property
    def token_trace_oracle_manifest(self) -> Path:
        return self.dataset_root / "cache" / "fp32_reference" / "token_trace_oracle_eval.manifest.json"

    @property
    def split_dir(self) -> Path:
        return self.dataset_root / "splits"


@dataclass
class MetricAccumulator:
    total: float = 0.0
    valid_count: int = 0
    sample_count: int = 0

    def add(self, value: Any) -> None:
        self.sample_count += 1
        number = _as_float(value)
        if math.isfinite(number):
            self.total += number
            self.valid_count += 1

    @property
    def mean(self) -> float:
        if self.valid_count == 0:
            return float("nan")
        return self.total / self.valid_count


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_path_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def _normalize_quant_mode_artifact_path(value: Any, paths: SummaryPaths) -> tuple[str, bool]:
    text = _normalize_path_text(value)
    if not text:
        return "", False
    legacy = str(paths.dataset_root / "ffn_layer_heatmap" / "quant_mode_ablation")
    current = str(paths.dataset_root / "quant_mode_ablation")
    candidates = (
        (legacy, current),
        (legacy.replace("\\", "/"), current.replace("\\", "/")),
    )
    for old, new in candidates:
        if text.startswith(old):
            return new + text[len(old) :], True
    return text, False


def _fix_legacy_quant_mode_paths(df: pd.DataFrame, paths: SummaryPaths) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    path_columns = [
        column
        for column in result.columns
        if column.endswith("_path") or column in {"artifact_path", "embedding_path", "qparams_path"}
    ]
    for column in path_columns:
        result[column] = [
            _normalize_quant_mode_artifact_path(value, paths)[0] if _normalize_path_text(value) else value
            for value in result[column]
        ]
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"value": value}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kwargs)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def _metric_direction(metric_name: Any, current: Any = None) -> str:
    current_text = str(current).strip()
    if current_text in {"higher_is_better", "lower_is_better", "diagnostic"}:
        return current_text
    name = str(metric_name).strip()
    if name.endswith("_retention") or name in {"topk_overlap", "topk_survival_rate"}:
        return "higher_is_better"
    lower_markers = (
        "damage",
        "distortion",
        "drift",
        "loss",
        "clipping",
        "clip",
        "near_zero",
        "exact_zero",
        "boundary",
        "overflow",
        "inflation_deviation",
        "quant_error",
    )
    if any(marker in name for marker in lower_markers):
        return "lower_is_better"
    return "diagnostic"


def _damage_value(value: Any, metric_direction: str) -> float:
    number = _as_float(value)
    if not math.isfinite(number):
        return float("nan")
    if metric_direction == "higher_is_better":
        return 1.0 - number
    if metric_direction == "lower_is_better":
        return number
    return float("nan")


def _damage_gap(group_values: Mapping[str, Any], metric_direction: str, comparison_group: str) -> float:
    many_damage = _damage_value(group_values.get("many"), metric_direction)
    comparison_damage = _damage_value(group_values.get(comparison_group), metric_direction)
    if not math.isfinite(many_damage) or not math.isfinite(comparison_damage):
        return float("nan")
    return comparison_damage - many_damage


def _add_damage_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    if "metric_direction" not in result.columns:
        result["metric_direction"] = result.get("metric_name", "").map(_metric_direction)
    else:
        result["metric_direction"] = [
            _metric_direction(metric_name, current)
            for metric_name, current in zip(result.get("metric_name", ""), result["metric_direction"])
        ]
    for group in ANALYSIS_GROUPS:
        value_col = f"{group}_value"
        damage_col = f"{group}_damage"
        if value_col in result.columns:
            result[damage_col] = [
                _damage_value(value, direction)
                for value, direction in zip(result[value_col], result["metric_direction"])
            ]
    if {"many_value", "tail_value"}.issubset(result.columns):
        result["tail_minus_many_damage_gap"] = [
            _damage_gap({"many": many, "tail": tail}, direction, "tail")
            for many, tail, direction in zip(result["many_value"], result["tail_value"], result["metric_direction"])
        ]
    if {"many_value", "few_value"}.issubset(result.columns):
        result["few_minus_many_damage_gap"] = [
            _damage_gap({"many": many, "few": few}, direction, "few")
            for many, few, direction in zip(result["many_value"], result["few_value"], result["metric_direction"])
        ]
    return result


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    front = [column for column in FRONT_COLUMNS if column in df.columns]
    rest = [column for column in df.columns if column not in front]
    return df[front + rest]


def _tag_source(df: pd.DataFrame, *, source_experiment: str, source_file: Path) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result.insert(0, "source_file", str(source_file))
    result.insert(0, "source_experiment", source_experiment)
    return _order_columns(_add_damage_columns(result))


def _filter_metric_rows(df: pd.DataFrame, metric_names: Sequence[str]) -> pd.DataFrame:
    if df.empty or "metric_name" not in df.columns:
        return pd.DataFrame()
    return df[df["metric_name"].isin(metric_names)].copy()


def build_layer_level_summary(paths: SummaryPaths) -> pd.DataFrame:
    heatmap = _filter_metric_rows(_read_csv(paths.ffn_layer_heatmap_summary), LAYER_LEVEL_METRICS)
    heatmap = _tag_source(
        heatmap,
        source_experiment="layer_heatmap_w4a4",
        source_file=paths.ffn_layer_heatmap_summary,
    )

    quant_mode = _read_csv(paths.quant_mode_summary)
    quant_mode_source = paths.quant_mode_summary
    if quant_mode.empty:
        quant_mode = _read_csv(paths.quant_mode_detail)
        quant_mode_source = paths.quant_mode_detail
        if not quant_mode.empty and "row_type" in quant_mode.columns:
            quant_mode = quant_mode[quant_mode["row_type"].astype(str).str.contains("aggregate", case=False, na=False)]
    quant_mode = _fix_legacy_quant_mode_paths(_filter_metric_rows(quant_mode, LAYER_LEVEL_METRICS), paths)
    quant_mode = _tag_source(
        quant_mode,
        source_experiment="layer_level_quant_mode_ablation",
        source_file=quant_mode_source,
    )

    pieces = [frame for frame in (heatmap, quant_mode) if not frame.empty]
    if not pieces:
        return pd.DataFrame()
    return _order_columns(pd.concat(pieces, ignore_index=True, sort=False))


def build_sensitive_layer_candidates(layer_summary: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    if layer_summary.empty:
        return pd.DataFrame()
    required = {"layer_idx", "module_name", "quant_mode", "metric_name", "tail_minus_many_damage_gap"}
    if not required.issubset(layer_summary.columns):
        return pd.DataFrame()
    candidates = layer_summary.copy()
    candidates = candidates[candidates["metric_name"].isin(LAYER_LEVEL_METRICS)]
    candidates = candidates[candidates["metric_direction"].isin(["higher_is_better", "lower_is_better"])]
    if "quant_mode" in candidates.columns:
        w4a4 = candidates[candidates["quant_mode"].astype(str).str.lower().isin({"w4a4", "w4-a4", "w4_a4"})]
        if not w4a4.empty:
            candidates = w4a4
    if "source_experiment" in candidates.columns:
        heatmap = candidates[candidates["source_experiment"].astype(str).eq("layer_heatmap_w4a4")]
        if not heatmap.empty:
            candidates = heatmap
    candidates["score_value"] = pd.to_numeric(candidates["tail_minus_many_damage_gap"], errors="coerce")
    candidates = candidates[candidates["score_value"].notna()]
    if candidates.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (layer_idx, module_name), part in candidates.groupby(["layer_idx", "module_name"], dropna=False):
        part = part.sort_values("score_value", ascending=False)
        top = part.iloc[0]
        positive = part[part["score_value"] > 0]
        rows.append(
            {
                "layer_idx": layer_idx,
                "module_name": module_name,
                "quant_mode": top.get("quant_mode", ""),
                "sensitivity_score": top.get("score_value", float("nan")),
                "sensitivity_metric": top.get("metric_name", ""),
                "evidence_source": top.get("source_experiment", ""),
                "positive_damage_metric_count": int(len(positive)),
                "max_tail_minus_many_damage_gap": top.get("score_value", float("nan")),
                "max_few_minus_many_damage_gap": pd.to_numeric(part.get("few_minus_many_damage_gap"), errors="coerce").max(),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["sensitivity_score", "positive_damage_metric_count"],
        ascending=[False, False],
        na_position="last",
    )
    if top_k > 0:
        result = result.head(int(top_k))
    return result.reset_index(drop=True)


def build_geometry_summary(paths: SummaryPaths) -> pd.DataFrame:
    summary = _filter_metric_rows(_read_csv(paths.semantic_geometry_summary), GEOMETRY_METRICS)
    summary = _tag_source(summary, source_experiment="semantic_geometry", source_file=paths.semantic_geometry_summary)
    collapse = build_geometry_collapse_summary(paths)
    pieces = [frame for frame in (summary, collapse) if not frame.empty]
    if not pieces:
        return pd.DataFrame()
    return _order_columns(pd.concat(pieces, ignore_index=True, sort=False))


def build_geometry_collapse_summary(paths: SummaryPaths) -> pd.DataFrame:
    df = _read_csv(paths.semantic_geometry_global_summary)
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, source_row in df.iterrows():
        tail_gap = _as_float(source_row.get("tail_collapse_gap", source_row.get("collapse_gap")))
        few_gap = _as_float(source_row.get("few_collapse_gap"))
        collapse_gap = _as_float(source_row.get("collapse_gap", tail_gap))
        row = {
            "scope": source_row.get("scope", ""),
            "target": source_row.get("target", ""),
            "layer_idx": source_row.get("layer_idx", ""),
            "module_name": source_row.get("module_name", ""),
            "target_name": source_row.get("target_name", ""),
            "embedding_cache_path": source_row.get("embedding_cache_path", ""),
            "qparams_path": source_row.get("qparams_path", ""),
            "quant_mode": source_row.get("quant_mode", ""),
            "granularity": source_row.get("granularity", ""),
            "metric_name": "collapse_gap",
            "metric_role": "main",
            "metric_direction": "lower_is_better",
            "aggregation_unit": "global_derived",
            "aggregation_fn": "custom",
            "source_column": "collapse_gap/tail_collapse_gap/few_collapse_gap",
            "gap_enabled": True,
            "comparable_across_groups": True,
            "many_value": 0.0,
            "medium_value": float("nan"),
            "few_value": few_gap,
            "tail_value": tail_gap if math.isfinite(tail_gap) else collapse_gap,
            "tail_vs_many_gap": tail_gap if math.isfinite(tail_gap) else collapse_gap,
            "few_vs_many_gap": few_gap,
            "num_samples": source_row.get("num_samples", ""),
            "num_classes": source_row.get("num_classes", ""),
            "fisher_normalization": source_row.get("fisher_normalization", ""),
            "geometry_metric_note": "collapse gap is imported from semantic_summary_metrics.csv; embedding cosine is not recomputed here",
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    return _tag_source(
        result,
        source_experiment="semantic_geometry_collapse_gap",
        source_file=paths.semantic_geometry_global_summary,
    )


def build_activation_distribution_summary(
    paths: SummaryPaths,
    sensitive_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = _read_csv(paths.activation_distribution_summary)
    if summary.empty:
        empty = pd.DataFrame()
        return empty, empty
    if "row_type" in summary.columns:
        summary = summary[summary["row_type"].astype(str).eq("metric_summary")].copy()
    if "analysis_group" in summary.columns:
        tail_rows = summary[summary["analysis_group"].astype(str).eq("tail")].copy()
        if not tail_rows.empty:
            summary = tail_rows
    summary = _filter_metric_rows(summary, ACTIVATION_DISTRIBUTION_METRICS)
    all_summary = _tag_source(
        summary,
        source_experiment="activation_distribution",
        source_file=paths.activation_distribution_summary,
    )

    sensitive_summary = all_summary
    if not sensitive_targets.empty and {"layer_idx", "module_name"}.issubset(all_summary.columns):
        keys = {
            (str(row["layer_idx"]), str(row["module_name"]))
            for _, row in sensitive_targets[["layer_idx", "module_name"]].dropna().iterrows()
        }
        mask = [
            (str(layer_idx), str(module_name)) in keys
            for layer_idx, module_name in zip(all_summary["layer_idx"], all_summary["module_name"])
        ]
        filtered = all_summary[mask].copy()
        if not filtered.empty:
            sensitive_summary = filtered
    return sensitive_summary.reset_index(drop=True), all_summary.reset_index(drop=True)


def build_group_activation_token_summary(paths: SummaryPaths, ranking_summary: pd.DataFrame) -> pd.DataFrame:
    token_summary = _filter_metric_rows(_read_csv(paths.token_survival_summary), TOKEN_METRICS)
    token_summary = _tag_source(token_summary, source_experiment="token_survival", source_file=paths.token_survival_summary)

    danger_summary = _filter_metric_rows(_read_csv(paths.danger_zone_summary), DANGER_ZONE_METRICS)
    danger_summary = _tag_source(danger_summary, source_experiment="quant_danger_zone", source_file=paths.danger_zone_summary)

    pieces = [frame for frame in (token_summary, danger_summary, ranking_summary) if not frame.empty]
    if not pieces:
        return pd.DataFrame()
    return _order_columns(pd.concat(pieces, ignore_index=True, sort=False))


def _parse_id_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return list(parsed)


def _ranking_drift(fp32_ids: Any, int4_ids: Any) -> tuple[float, float]:
    fp32 = _parse_id_list(fp32_ids)
    int4 = _parse_id_list(int4_ids)
    if not fp32:
        return float("nan"), float("nan")

    fp32_rank = {token: rank for rank, token in enumerate(fp32)}
    int4_rank = {token: rank for rank, token in enumerate(int4)}
    common = [token for token in fp32 if token in int4_rank]

    if common:
        common_denominator = max(len(fp32) - 1, 1)
        common_drift = sum(abs(fp32_rank[token] - int4_rank[token]) for token in common) / (
            len(common) * common_denominator
        )
    else:
        common_drift = float("nan")

    missing_rank = len(fp32)
    penalized_denominator = max(len(fp32), 1)
    penalized_drift = sum(
        abs(rank - int4_rank.get(token, missing_rank)) / penalized_denominator
        for rank, token in enumerate(fp32)
    ) / len(fp32)
    return float(common_drift), float(penalized_drift)


def _ranking_group_for_row(row: Mapping[str, Any]) -> str:
    shot_group = str(row.get("shot_group", "")).strip().lower()
    if shot_group in BASE_BUCKETS:
        return shot_group
    bucket = str(row.get("bucket", "")).strip().lower()
    if bucket in BASE_BUCKETS:
        return bucket
    return ""


def _ranking_tail_included(row: Mapping[str, Any], group: str) -> bool:
    if "tail_included" in row and str(row.get("tail_included", "")).strip() != "":
        return _boolish(row.get("tail_included"))
    return group in TAIL_BUCKETS


def compute_token_ranking_drift_summary(paths: SummaryPaths, *, chunksize: int) -> pd.DataFrame:
    path = paths.token_survival_metrics
    if not path.exists():
        return pd.DataFrame()

    desired_columns = [
        "sample_id",
        "bucket",
        "shot_group",
        "tail_included",
        "intervention_target",
        "intervention_layer_idx",
        "intervention_module_name",
        "trace_target_name",
        "trace_layer_idx",
        "trace_module_name",
        "target",
        "layer_idx",
        "module_name",
        "tensor_position",
        "quant_mode",
        "granularity",
        "integer_mapping_source",
        "actual_integer_mapping_available",
        "integer_mapping_is_direct_quantlinear_forward",
        "fp32_topk_token_ids",
        "int4_topk_token_ids",
        "metric_valid_for_paper",
    ]
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in desired_columns if column in header.columns]
    required = {"fp32_topk_token_ids", "int4_topk_token_ids"}
    if not required.issubset(usecols):
        return pd.DataFrame()

    facet_columns = [
        column
        for column in (
            "intervention_target",
            "intervention_layer_idx",
            "intervention_module_name",
            "trace_target_name",
            "trace_layer_idx",
            "trace_module_name",
            "target",
            "layer_idx",
            "module_name",
            "tensor_position",
            "quant_mode",
            "granularity",
            "integer_mapping_source",
            "actual_integer_mapping_available",
            "integer_mapping_is_direct_quantlinear_forward",
        )
        if column in usecols
    ]

    accumulators: dict[tuple[Any, ...], MetricAccumulator] = defaultdict(MetricAccumulator)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        if "metric_valid_for_paper" in chunk.columns:
            valid_mask = ~chunk["metric_valid_for_paper"].astype(str).str.lower().isin({"false", "0", "no"})
            chunk = chunk[valid_mask].copy()
        if chunk.empty:
            continue
        drifts = [
            _ranking_drift(fp32_ids, int4_ids)
            for fp32_ids, int4_ids in zip(chunk["fp32_topk_token_ids"], chunk["int4_topk_token_ids"])
        ]
        chunk["ranking_drift_common"] = [pair[0] for pair in drifts]
        chunk["ranking_drift_penalized"] = [pair[1] for pair in drifts]

        row_dicts = chunk.to_dict(orient="records")
        for row in row_dicts:
            base_group = _ranking_group_for_row(row)
            groups: list[str] = []
            if base_group in BASE_BUCKETS:
                groups.append(base_group)
            if _ranking_tail_included(row, base_group):
                groups.append("tail")
            for metric_name in ("ranking_drift_common", "ranking_drift_penalized"):
                for group in groups:
                    key = tuple(row.get(column, "") for column in facet_columns) + (metric_name, group)
                    accumulators[key].add(row.get(metric_name))

    if not accumulators:
        return pd.DataFrame()

    grouped: dict[tuple[Any, ...], dict[str, MetricAccumulator]] = defaultdict(dict)
    for key, accumulator in accumulators.items():
        facet_key = key[: len(facet_columns)] + (key[len(facet_columns)],)
        group = str(key[len(facet_columns) + 1])
        grouped[facet_key][group] = accumulator

    rows: list[dict[str, Any]] = []
    for facet_key, by_group in grouped.items():
        metric_name = str(facet_key[-1])
        values = {group: by_group.get(group, MetricAccumulator()).mean for group in ANALYSIS_GROUPS}
        row = {column: value for column, value in zip(facet_columns, facet_key[:-1])}
        row.update(
            {
                "metric_name": metric_name,
                "metric_role": "formal_main",
                "metric_direction": "lower_is_better",
                "aggregation_unit": "token_sample_summary",
                "aggregation_fn": "mean",
                "source_column": metric_name,
                "weighting": "sample_equal",
                "comparable_across_groups": True,
                "gap_enabled": True,
            }
        )
        for group in ANALYSIS_GROUPS:
            accumulator = by_group.get(group, MetricAccumulator())
            row[f"{group}_value"] = values[group]
            row[f"{group}_sample_count"] = accumulator.sample_count
            row[f"{group}_valid_count"] = accumulator.valid_count
        row["tail_vs_many_gap"] = _damage_gap(values, "lower_is_better", "tail")
        row["few_vs_many_gap"] = _damage_gap(values, "lower_is_better", "few")
        rows.append(row)

    result = pd.DataFrame(rows)
    return _tag_source(result, source_experiment="token_survival_ranking_drift", source_file=path)


def build_qparams_reuse_validation(paths: SummaryPaths) -> pd.DataFrame:
    source = _read_csv(paths.quant_mode_qparams_summary)
    if source.empty:
        return pd.DataFrame(
            [
                {
                    "status": "ERROR",
                    "category": "quantization_protocol",
                    "check_name": "qparams_summary_exists",
                    "detail": f"Missing or empty qparams summary: {paths.quant_mode_qparams_summary}",
                    "source_file": str(paths.quant_mode_qparams_summary),
                }
            ]
        )

    required_columns = {
        "layer_idx",
        "module_name",
        "quant_mode",
        "qparams_path",
        "weight_qparams_enabled",
        "activation_qparams_enabled",
    }
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        return pd.DataFrame(
            [
                {
                    "status": "ERROR",
                    "category": "quantization_protocol",
                    "check_name": "qparams_summary_schema",
                    "detail": f"Missing columns: {missing_columns}",
                    "source_file": str(paths.quant_mode_qparams_summary),
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    hash_cache: dict[str, str] = {}
    for _, row in source.iterrows():
        quant_mode = str(row.get("quant_mode", "")).strip()
        raw_path = _normalize_path_text(row.get("qparams_path"))
        resolved_path, rewritten = _normalize_quant_mode_artifact_path(raw_path, paths)
        path = Path(resolved_path) if resolved_path else None
        exists = bool(path and path.exists())
        qparams_sha256 = ""
        if exists:
            key = str(path)
            if key not in hash_cache:
                hash_cache[key] = _file_sha256(path)
            qparams_sha256 = hash_cache[key]

        expected = QUANT_MODE_ENABLED_FLAGS.get(quant_mode)
        actual_weight = _boolish(row.get("weight_qparams_enabled"))
        actual_activation = _boolish(row.get("activation_qparams_enabled"))
        expected_weight = expected["weight_qparams_enabled"] if expected else None
        expected_activation = expected["activation_qparams_enabled"] if expected else None
        flags_ok = (
            expected is not None
            and actual_weight == bool(expected_weight)
            and actual_activation == bool(expected_activation)
        )
        rows.append(
            {
                "status": "PASS" if exists and flags_ok else "ERROR",
                "category": "quantization_protocol",
                "check_name": "qparams_row",
                "layer_idx": row.get("layer_idx", ""),
                "module_name": row.get("module_name", ""),
                "target_name": row.get("target_name", ""),
                "quant_mode": quant_mode,
                "raw_qparams_path": raw_path,
                "qparams_path": resolved_path,
                "qparams_path_was_rewritten": bool(rewritten),
                "qparams_path_exists": exists,
                "qparams_sha256": qparams_sha256,
                "weight_qparams_enabled": actual_weight,
                "activation_qparams_enabled": actual_activation,
                "expected_weight_qparams_enabled": expected_weight,
                "expected_activation_qparams_enabled": expected_activation,
                "enabled_flags_match": flags_ok,
                "source_file": str(paths.quant_mode_qparams_summary),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    for (layer_idx, module_name), part in result.groupby(["layer_idx", "module_name"], dropna=False):
        index = part.index
        mode_set = set(part["quant_mode"].astype(str))
        paths_for_target = {
            str(value)
            for value in part["qparams_path"].dropna()
            if _normalize_path_text(value)
        }
        hashes_for_target = {
            str(value)
            for value in part["qparams_sha256"].dropna()
            if _normalize_path_text(value)
        }
        modes_ok = mode_set == set(QUANT_MODE_ENABLED_FLAGS)
        reuse_ok = len(paths_for_target) == 1 and len(hashes_for_target) == 1
        result.loc[index, "expected_quant_modes_present"] = modes_ok
        result.loc[index, "qparams_reused_across_modes"] = reuse_ok
        result.loc[index, "reuse_group_status"] = "PASS" if modes_ok and reuse_ok else "ERROR"
        result.loc[index, "reuse_group_detail"] = (
            f"layer_idx={layer_idx}, module_name={module_name}, "
            f"modes={sorted(mode_set)}, unique_qparams_paths={len(paths_for_target)}, unique_qparams_hashes={len(hashes_for_target)}"
        )
        if not (modes_ok and reuse_ok):
            result.loc[index, "status"] = "ERROR"
    return result


def _validation_row(status: str, category: str, check_name: str, detail: str, **extra: Any) -> dict[str, Any]:
    row = {
        "status": status,
        "category": category,
        "check_name": check_name,
        "detail": detail,
    }
    row.update(extra)
    return row


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("checkpoint_hash") or manifest.get("checkpoint_sha256") or "").strip()


def _same_nonempty_values(values: Mapping[str, Any]) -> tuple[bool, str]:
    normalized = {key: str(value).strip() for key, value in values.items() if str(value).strip()}
    if len(normalized) != len(values):
        missing = sorted(set(values) - set(normalized))
        return False, f"missing values for {missing}; observed={normalized}"
    unique = set(normalized.values())
    return len(unique) == 1, f"observed={normalized}"


def _path_same(left: Any, right: Any, paths: SummaryPaths | None = None) -> bool:
    left_text = _normalize_path_text(left)
    right_text = _normalize_path_text(right)
    if paths is not None:
        left_text = _normalize_quant_mode_artifact_path(left_text, paths)[0]
        right_text = _normalize_quant_mode_artifact_path(right_text, paths)[0]
    if not left_text or not right_text:
        return False
    return str(Path(left_text)) == str(Path(right_text))


def _expected_all_ffn_targets() -> set[tuple[int, str]]:
    return {(layer_idx, module_name) for layer_idx in range(12) for module_name in ("fc1", "fc2")}


def _target_name(layer_idx: Any, module_name: Any) -> str:
    try:
        layer = int(float(layer_idx))
    except (TypeError, ValueError):
        return ""
    module = str(module_name).strip()
    if module not in {"fc1", "fc2"}:
        return ""
    return f"encoder.layers.{layer}.{module}"


def _short_target(layer_idx: Any, module_name: Any) -> str:
    try:
        layer = int(float(layer_idx))
    except (TypeError, ValueError):
        return ""
    module = str(module_name).strip()
    if module not in {"fc1", "fc2"}:
        return ""
    return f"L{layer}_{module}"


def _parse_layer_module_from_text(value: Any) -> tuple[int, str] | None:
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"encoder\.layers\.(\d+)\.(fc[12])", text)
    if match:
        return int(match.group(1)), match.group(2)
    match = re.search(r"\bL(\d+)_(fc[12])\b", text)
    if match:
        return int(match.group(1)), match.group(2)
    match = re.search(r"\bLayer(\d+)\.(fc[12])\.", text)
    if match:
        return int(match.group(1)), match.group(2)
    return None


def _target_set_from_manifest_items(items: Any) -> set[tuple[int, str]]:
    targets: set[tuple[int, str]] = set()
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return targets
    for item in items:
        parsed: tuple[int, str] | None = None
        if isinstance(item, Mapping):
            layer = item.get("layer_idx")
            module = item.get("module_name")
            if layer is not None and module is not None:
                try:
                    parsed = (int(float(layer)), str(module).strip())
                except (TypeError, ValueError):
                    parsed = None
            if parsed is None:
                parsed = _parse_layer_module_from_text(item.get("target_name", "") or item.get("target", ""))
        else:
            parsed = _parse_layer_module_from_text(item)
        if parsed is not None and parsed[1] in {"fc1", "fc2"}:
            targets.add(parsed)
    return targets


def _target_set_from_frame(df: pd.DataFrame, *, layer_column: str = "layer_idx", module_column: str = "module_name") -> set[tuple[int, str]]:
    targets: set[tuple[int, str]] = set()
    if df.empty or layer_column not in df.columns or module_column not in df.columns:
        return targets
    pairs = df[[layer_column, module_column]].drop_duplicates().itertuples(index=False, name=None)
    for layer, module in pairs:
        module_text = str(module).strip()
        try:
            layer_int = int(float(layer))
        except (TypeError, ValueError):
            continue
        if module_text in {"fc1", "fc2"}:
            targets.add((layer_int, module_text))
    return targets


def _format_target_set(targets: Iterable[tuple[int, str]], *, limit: int = 12) -> str:
    values = [f"L{layer}_{module}" for layer, module in sorted(set(targets))]
    if len(values) > limit:
        return f"{values[:limit]} ... (+{len(values) - limit} more)"
    return str(values)


def _target_set_check_rows(
    *,
    source: str,
    actual: set[tuple[int, str]],
    expected: set[tuple[int, str]],
    check_name: str,
) -> list[dict[str, Any]]:
    extra = actual - expected
    missing = expected - actual
    ok = not extra and not missing and bool(expected)
    return [
        _validation_row(
            "PASS" if ok else "ERROR",
            "hook_target_position",
            check_name,
            f"source={source}; actual={_format_target_set(actual)}; expected={_format_target_set(expected)}; "
            f"extra={_format_target_set(extra)}; missing={_format_target_set(missing)}",
        )
    ]


def _validate_split_reference_identity(manifests: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs = {
        "fp32_core_reference": manifests.get("fp32_core_reference", {}),
        "fp32_oer_oracle": manifests.get("fp32_oer_oracle", {}),
        "fp32_token_trace_oracle": manifests.get("fp32_token_trace_oracle", {}),
    }
    checks = {
        "eval_csv_sha256": {name: manifest.get("eval_csv_sha256", "") for name, manifest in refs.items()},
        "sample_order_hash": {name: manifest.get("sample_order_hash", "") for name, manifest in refs.items()},
        "checkpoint_hash": {name: _manifest_hash(manifest) for name, manifest in refs.items()},
    }
    rows: list[dict[str, Any]] = []
    for check_name, values in checks.items():
        ok, detail = _same_nonempty_values(values)
        rows.append(_validation_row("PASS" if ok else "ERROR", "split_reference_identity", check_name, detail))
    return rows


def _validate_oracle_identity(paths: SummaryPaths, manifests: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quant_manifest = manifests.get("quant_mode_ablation", {})
    danger_manifest = manifests.get("quant_danger_zone", {})
    token_manifest = manifests.get("token_survival", {})
    expected_oer_path = paths.dataset_root / "cache" / "fp32_reference" / "oer_oracle_eval.pt"
    expected_trace_path = paths.dataset_root / "cache" / "fp32_reference" / "token_trace_oracle_eval.pt"

    for source, manifest in (("quant_mode_ablation", quant_manifest), ("quant_danger_zone", danger_manifest)):
        actual = manifest.get("fp32_reference_oer_oracle_path", "")
        ok = _path_same(actual, expected_oer_path, paths)
        rows.append(
            _validation_row(
                "PASS" if ok else "ERROR",
                "oracle_identity",
                f"{source}.oer_oracle_path",
                f"actual={actual}; expected={expected_oer_path}",
            )
        )

    trace_cache = token_manifest.get("token_trace_oracle_cache", {})
    trace_path = trace_cache.get("path", "") if isinstance(trace_cache, Mapping) else ""
    ok = _path_same(trace_path, expected_trace_path, paths)
    rows.append(
        _validation_row(
            "PASS" if ok else "ERROR",
            "oracle_identity",
            "token_survival.token_trace_oracle_path",
            f"actual={trace_path}; expected={expected_trace_path}",
        )
    )
    return rows


def _validate_metric_protocol(manifests: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_manifest = manifests.get("token_survival", {})
    trace_manifest = manifests.get("fp32_token_trace_oracle", {})
    trace_cache = token_manifest.get("token_trace_oracle_cache", {})
    trace_cache = trace_cache if isinstance(trace_cache, Mapping) else {}

    for key in ("topk_unit", "topk_source", "topk_ratio", "topk_k_policy"):
        token_value = trace_cache.get(key, token_manifest.get(key, ""))
        oracle_value = trace_manifest.get(key, "")
        ok = str(token_value) == str(oracle_value) and str(token_value).strip() != ""
        rows.append(
            _validation_row(
                "PASS" if ok else "ERROR",
                "metric_protocol",
                f"token_topk.{key}",
                f"token_survival={token_value}; token_trace_oracle={oracle_value}",
            )
        )

    available_positions = set(trace_manifest.get("available_positions", []) or trace_cache.get("available_positions", []))
    required_positions = {"block_output", "input_activation"}
    ok = required_positions.issubset(available_positions)
    rows.append(
        _validation_row(
            "PASS" if ok else "ERROR",
            "metric_protocol",
            "token_trace.available_positions",
            f"required={sorted(required_positions)}; observed={sorted(available_positions)}",
        )
    )

    quant_manifest = manifests.get("quant_mode_ablation", {})
    oer_manifest = manifests.get("fp32_oer_oracle", {})
    observed_oer_positions = set(oer_manifest.get("activation_positions", []) or [])
    ok = {"input", "output"}.issubset(observed_oer_positions)
    rows.append(
        _validation_row(
            "PASS" if ok else "ERROR",
            "metric_protocol",
            "scalar_oer.activation_positions",
            f"required=['input', 'output']; observed={sorted(observed_oer_positions)}",
        )
    )
    expected_oer_fields = {
        "input_oer_activation_kind": "input_activation",
        "projection_oer_activation_kind": "linear_projection_output",
    }
    for key, expected in expected_oer_fields.items():
        actual = quant_manifest.get(key, "")
        ok = str(actual) == expected
        rows.append(
            _validation_row(
                "PASS" if ok else "ERROR",
                "metric_protocol",
                f"quant_mode_ablation.{key}",
                f"actual={actual}; expected={expected}",
            )
        )
    return rows


def _validate_hook_target_position(paths: SummaryPaths, manifests: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_ffn_targets = _expected_all_ffn_targets()

    heatmap = _read_csv(paths.ffn_layer_heatmap_summary, usecols=["layer_idx", "module_name"])
    heatmap_targets = _target_set_from_frame(heatmap)
    rows.extend(
        _target_set_check_rows(
            source="ffn_layer_heatmap_summary.csv",
            actual=heatmap_targets,
            expected=all_ffn_targets,
            check_name="layer_heatmap.intervention_targets",
        )
    )

    quant_manifest_targets = _target_set_from_manifest_items(manifests.get("quant_mode_ablation", {}).get("target_modules", []))
    quant_cols = [
        "layer_idx",
        "module_name",
        "target_name",
        "input_oer_activation_kind",
        "input_oer_oracle_source",
        "projection_oer_activation_kind",
        "projection_oer_oracle_source",
    ]
    quant_df = _read_csv(paths.quant_mode_summary, usecols=quant_cols)
    quant_targets = _target_set_from_frame(quant_df)
    rows.extend(
        _target_set_check_rows(
            source="ffn_quant_mode_comparison_summary.csv",
            actual=quant_targets,
            expected=quant_manifest_targets or all_ffn_targets,
            check_name="quant_mode_ablation.targets_match_manifest",
        )
    )
    bad_target_names = []
    for _, row in quant_df[["layer_idx", "module_name", "target_name"]].drop_duplicates().iterrows():
        expected = _target_name(row.get("layer_idx"), row.get("module_name"))
        if expected and str(row.get("target_name", "")).strip() != expected:
            bad_target_names.append((row.get("layer_idx"), row.get("module_name"), row.get("target_name"), expected))
    rows.append(
        _validation_row(
            "PASS" if not bad_target_names else "ERROR",
            "hook_target_position",
            "quant_mode_ablation.target_name_consistency",
            f"bad_rows={bad_target_names[:8]}",
        )
    )
    oer_checks = {
        "input_oer_activation_kind": "input_activation",
        "input_oer_oracle_source": "oer_oracle_eval.pt.input.topk_scalar_indices",
        "projection_oer_activation_kind": "linear_projection_output",
        "projection_oer_oracle_source": "oer_oracle_eval.pt.output.topk_scalar_indices",
    }
    for column, expected in oer_checks.items():
        observed = sorted(str(value) for value in quant_df[column].dropna().unique())
        ok = observed == [expected]
        rows.append(
            _validation_row(
                "PASS" if ok else "ERROR",
                "hook_target_position",
                f"quant_mode_ablation.{column}",
                f"observed={observed}; expected={[expected]}",
            )
        )

    oer_oracle_targets = _target_set_from_manifest_items(manifests.get("fp32_oer_oracle", {}).get("target_modules", []))
    rows.extend(
        _target_set_check_rows(
            source="oer_oracle_eval.pt.manifest.json",
            actual=oer_oracle_targets,
            expected=all_ffn_targets,
            check_name="oer_oracle.targets",
        )
    )

    token_manifest_targets = _target_set_from_manifest_items(manifests.get("token_survival", {}).get("target_modules", []))
    token_cols = [
        "intervention_target",
        "intervention_layer_idx",
        "intervention_module_name",
        "trace_target_name",
        "trace_layer_idx",
        "trace_module_name",
        "target",
        "layer_idx",
        "module_name",
        "tensor_position",
    ]
    token_df = _read_csv(paths.token_survival_summary, usecols=token_cols)
    token_facets = token_df.drop_duplicates()
    token_targets = _target_set_from_frame(
        token_facets,
        layer_column="intervention_layer_idx",
        module_column="intervention_module_name",
    )
    rows.extend(
        _target_set_check_rows(
            source="token_survival_comparison_summary.csv",
            actual=token_targets,
            expected=token_manifest_targets or all_ffn_targets,
            check_name="token_survival.intervention_targets_match_manifest",
        )
    )
    token_positions = set(str(value) for value in token_facets["tensor_position"].dropna().unique())
    expected_token_positions = {"block_output", "input_activation"}
    rows.append(
        _validation_row(
            "PASS" if token_positions == expected_token_positions else "ERROR",
            "hook_target_position",
            "token_survival.trace_positions",
            f"observed={sorted(token_positions)}; expected={sorted(expected_token_positions)}",
        )
    )

    token_bad_targets = []
    token_bad_block = []
    token_bad_input = []
    for _, row in token_facets.iterrows():
        intervention_expected = _short_target(row.get("intervention_layer_idx"), row.get("intervention_module_name"))
        if intervention_expected and str(row.get("intervention_target", "")).strip() != intervention_expected:
            token_bad_targets.append((row.get("intervention_target"), intervention_expected))
        if intervention_expected and str(row.get("target", "")).strip() != intervention_expected:
            token_bad_targets.append((row.get("target"), intervention_expected))

        position = str(row.get("tensor_position", "")).strip()
        if position == "block_output":
            try:
                trace_layer = int(float(row.get("trace_layer_idx")))
                row_layer = int(float(row.get("layer_idx")))
            except (TypeError, ValueError):
                token_bad_block.append(row.to_dict())
                continue
            expected_trace = f"Layer{trace_layer}.block_output"
            if (
                str(row.get("trace_module_name", "")).strip() != "block"
                or str(row.get("module_name", "")).strip() != "block"
                or str(row.get("trace_target_name", "")).strip() != expected_trace
                or row_layer != trace_layer
            ):
                token_bad_block.append(row.to_dict())
        elif position == "input_activation":
            try:
                intervention_layer = int(float(row.get("intervention_layer_idx")))
                trace_layer = int(float(row.get("trace_layer_idx")))
                row_layer = int(float(row.get("layer_idx")))
            except (TypeError, ValueError):
                token_bad_input.append(row.to_dict())
                continue
            intervention_module = str(row.get("intervention_module_name", "")).strip()
            expected_trace = f"Layer{intervention_layer}.{intervention_module}.input"
            if (
                trace_layer != intervention_layer
                or row_layer != intervention_layer
                or str(row.get("trace_module_name", "")).strip() != intervention_module
                or str(row.get("module_name", "")).strip() != intervention_module
                or str(row.get("trace_target_name", "")).strip() != expected_trace
            ):
                token_bad_input.append(row.to_dict())

    rows.append(
        _validation_row(
            "PASS" if not token_bad_targets else "ERROR",
            "hook_target_position",
            "token_survival.intervention_target_name_consistency",
            f"bad_rows={token_bad_targets[:8]}",
        )
    )
    rows.append(
        _validation_row(
            "PASS" if not token_bad_block else "ERROR",
            "hook_target_position",
            "token_survival.block_output_separated_from_ffn_input",
            f"bad_rows={token_bad_block[:3]}",
        )
    )
    rows.append(
        _validation_row(
            "PASS" if not token_bad_input else "ERROR",
            "hook_target_position",
            "token_survival.input_activation_matches_intervention_module",
            f"bad_rows={token_bad_input[:3]}",
        )
    )

    danger_manifest_targets = _target_set_from_manifest_items(manifests.get("quant_danger_zone", {}).get("target_modules", []))
    danger_df = _read_csv(paths.danger_zone_summary, usecols=["target", "tensor_position"])
    danger_facets = danger_df.drop_duplicates()
    danger_targets = {
        parsed
        for parsed in (_parse_layer_module_from_text(value) for value in danger_facets.get("target", pd.Series(dtype=str)))
        if parsed is not None
    }
    rows.extend(
        _target_set_check_rows(
            source="quant_danger_zone_comparison_summary.csv",
            actual=danger_targets,
            expected=danger_manifest_targets,
            check_name="quant_danger_zone.targets_match_manifest",
        )
    )
    expected_danger_positions = set(manifests.get("quant_danger_zone", {}).get("positions", []) or ["input_activation", "linear_output_projection"])
    danger_positions = set(str(value) for value in danger_facets["tensor_position"].dropna().unique())
    rows.append(
        _validation_row(
            "PASS" if danger_positions == expected_danger_positions else "ERROR",
            "hook_target_position",
            "quant_danger_zone.positions_match_manifest",
            f"observed={sorted(danger_positions)}; expected={sorted(expected_danger_positions)}",
        )
    )

    activation_manifest_targets = _target_set_from_manifest_items(
        manifests.get("activation_distribution", {}).get("target_modules", [])
    )
    activation_cols = ["row_type", "layer_idx", "module_name", "activation_kind", "target_name"]
    activation_df = _read_csv(paths.activation_distribution_summary, usecols=activation_cols)
    if "row_type" in activation_df.columns:
        activation_df = activation_df[activation_df["row_type"].astype(str).eq("metric_summary")]
    activation_facets = activation_df[["layer_idx", "module_name", "activation_kind", "target_name"]].drop_duplicates()
    activation_targets = _target_set_from_frame(activation_facets)
    rows.extend(
        _target_set_check_rows(
            source="ffn_activation_distribution_plot_summary.csv",
            actual=activation_targets,
            expected=activation_manifest_targets,
            check_name="activation_distribution.targets_match_manifest",
        )
    )
    activation_kinds = set(str(value) for value in activation_facets["activation_kind"].dropna().unique())
    expected_activation_kinds = {"input_activation", "output_activation"}
    rows.append(
        _validation_row(
            "PASS" if activation_kinds == expected_activation_kinds else "ERROR",
            "hook_target_position",
            "activation_distribution.activation_kinds",
            f"observed={sorted(activation_kinds)}; expected={sorted(expected_activation_kinds)}",
        )
    )
    bad_activation_names = []
    for _, row in activation_facets.iterrows():
        kind = str(row.get("activation_kind", "")).strip()
        suffix = "input" if kind == "input_activation" else "output" if kind == "output_activation" else ""
        short = _short_target(row.get("layer_idx"), row.get("module_name"))
        expected_name = short.replace("_", " ") + f" {suffix}" if short and suffix else ""
        if expected_name and str(row.get("target_name", "")).strip() != expected_name:
            bad_activation_names.append((row.get("target_name"), expected_name))
    rows.append(
        _validation_row(
            "PASS" if not bad_activation_names else "ERROR",
            "hook_target_position",
            "activation_distribution.target_name_consistency",
            f"bad_rows={bad_activation_names[:8]}",
        )
    )

    return rows


def _validate_quantization_protocol(
    manifests: Mapping[str, Mapping[str, Any]],
    qparams_validation: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quant_modes = set(manifests.get("quant_mode_ablation", {}).get("quant_modes", []) or [])
    expected_modes = set(QUANT_MODE_ENABLED_FLAGS)
    modes_ok = quant_modes == expected_modes
    rows.append(
        _validation_row(
            "PASS" if modes_ok else "ERROR",
            "quantization_protocol",
            "quant_mode_ablation.quant_modes",
            f"observed={sorted(quant_modes)}; expected={sorted(expected_modes)}",
        )
    )

    if qparams_validation.empty:
        rows.append(
            _validation_row(
                "ERROR",
                "quantization_protocol",
                "qparams_reuse_validation",
                "qparams validation table is empty",
            )
        )
        return rows

    error_rows = qparams_validation[qparams_validation.get("status", "") == "ERROR"]
    ok = error_rows.empty
    rows.append(
        _validation_row(
            "PASS" if ok else "ERROR",
            "quantization_protocol",
            "qparams_reuse_and_enabled_flags",
            f"rows={len(qparams_validation)}, error_rows={len(error_rows)}",
        )
    )
    rewritten_count = int(qparams_validation.get("qparams_path_was_rewritten", pd.Series(dtype=bool)).sum())
    rows.append(
        _validation_row(
            "WARN" if rewritten_count else "PASS",
            "quantization_protocol",
            "legacy_qparams_path_rewrite",
            f"rewritten_rows={rewritten_count}; generated summaries use corrected qparams_path",
        )
    )
    return rows


def validate_protocol(
    paths: SummaryPaths,
    manifests: Mapping[str, Mapping[str, Any]],
    qparams_validation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_validate_split_reference_identity(manifests))
    rows.extend(_validate_oracle_identity(paths, manifests))
    rows.extend(_validate_metric_protocol(manifests))
    rows.extend(_validate_hook_target_position(paths, manifests))
    rows.extend(_validate_quantization_protocol(manifests, qparams_validation))
    return pd.DataFrame(rows)


def _manifest_rows(paths: SummaryPaths) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_paths = {
        "ffn_layer_heatmap": paths.ffn_layer_heatmap_manifest,
        "quant_mode_ablation": paths.quant_mode_manifest,
        "activation_distribution": paths.activation_distribution_manifest,
        "quant_danger_zone": paths.danger_zone_manifest,
        "token_survival": paths.token_survival_manifest,
        "semantic_geometry": paths.semantic_geometry_manifest,
        "fp32_core_reference": paths.core_reference_manifest,
        "fp32_oer_oracle": paths.oer_oracle_manifest,
        "fp32_token_trace_oracle": paths.token_trace_oracle_manifest,
    }
    rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}

    for source, path in manifest_paths.items():
        data = _read_json(path)
        manifests[source] = data
        if not data:
            rows.append(
                {
                    "section": "manifest",
                    "source": source,
                    "key": "__missing__",
                    "value": str(path),
                    "source_file": str(path),
                }
            )
            continue
        for key, value in data.items():
            rows.append(
                {
                    "section": "manifest",
                    "source": source,
                    "key": key,
                    "value": json.dumps(value, ensure_ascii=False, default=_json_default),
                    "source_file": str(path),
                }
            )

    derived = [
        ("bucket_rule", "many", "train_label_n >= 100"),
        ("bucket_rule", "medium", "20 <= train_label_n <= 99"),
        ("bucket_rule", "few", "train_label_n <= 19"),
        ("bucket_rule", "tail", "medium + few; not the same concept as few"),
        (
            "ranking_drift_definition",
            "ranking_drift_common",
            "mean absolute rank displacement over common fp32/int4 top-k tokens, normalized by max(k-1, 1)",
        ),
        (
            "ranking_drift_definition",
            "ranking_drift_penalized",
            "mean absolute rank displacement over fp32 top-k tokens; missing int4 tokens use one-past-last rank, normalized by k",
        ),
        ("quant_mode_definition", "A4-only", "activation quantization enabled; weight quantization disabled"),
        ("quant_mode_definition", "W4-only", "weight quantization enabled; activation quantization disabled"),
        ("quant_mode_definition", "W4A4", "weight and activation quantization enabled"),
        (
            "geometry_metric_note",
            "center_cosine_margin_retention",
            "center-relative geometry metric; not a direct embedding cosine similarity recomputation",
        ),
        (
            "geometry_metric_note",
            "embedding_cosine",
            "not emitted by this summary because source CSVs do not contain it and .pt embeddings are not recomputed",
        ),
        ("split", "train_csv", str(paths.split_dir / "train.csv")),
        ("split", "eval_csv", str(paths.split_dir / "eval.csv")),
        ("split", "full_dataset_csv", str(paths.split_dir / "full_dataset.csv")),
        ("split", "calibration_csv", str(paths.split_dir / "calibration.csv")),
        ("split", "label_mapping", str(paths.split_dir / "label_mapping.json")),
    ]
    for section, key, value in derived:
        rows.append(
            {
                "section": section,
                "source": "derived_summary_protocol",
                "key": key,
                "value": value,
                "source_file": "",
            }
        )

    return rows, manifests


def build_protocol_outputs(paths: SummaryPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, manifests = _manifest_rows(paths)
    return pd.DataFrame(rows), manifests


def _dataset_root_from_config(dataset_name: str) -> Path:
    datasets = getattr(project_config, "DATASETS", {})
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset={dataset_name!r}; choose one of {tuple(datasets)}")
    root = datasets[dataset_name].get("ROOT")
    if root is None:
        raise ValueError(f"Dataset {dataset_name!r} does not define ROOT")
    return Path(root)


def _default_output_dir(dataset_root: Path) -> Path:
    return dataset_root / "quant_injection_summary"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Quant Injection summary tables from existing result files.")
    parser.add_argument("--dataset", default=getattr(project_config, "ACTIVE_DATASET", "local_data"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-sensitive-k", type=int, default=DEFAULT_TOP_SENSITIVE_K)
    parser.add_argument("--ranking-chunksize", type=int, default=DEFAULT_RANKING_CHUNKSIZE)
    parser.add_argument("--skip-ranking-drift", action="store_true")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else _dataset_root_from_config(args.dataset)
    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_output_dir(dataset_root)
    paths = SummaryPaths(dataset_name=str(args.dataset), dataset_root=dataset_root, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    layer_summary = build_layer_level_summary(paths)
    layer_path = output_dir / "layer_level_quant_injection_summary.csv"
    _write_csv(layer_summary, layer_path)
    generated_files["layer_level_quant_injection_summary"] = str(layer_path)
    row_counts["layer_level_quant_injection_summary"] = int(len(layer_summary))

    sensitive_targets = build_sensitive_layer_candidates(layer_summary, top_k=max(int(args.top_sensitive_k), 0))
    sensitive_path = output_dir / "sensitive_layer_candidates.csv"
    _write_csv(sensitive_targets, sensitive_path)
    generated_files["sensitive_layer_candidates"] = str(sensitive_path)
    row_counts["sensitive_layer_candidates"] = int(len(sensitive_targets))

    qparams_validation = build_qparams_reuse_validation(paths)
    qparams_validation_path = output_dir / "qparams_reuse_validation.csv"
    _write_csv(qparams_validation, qparams_validation_path)
    generated_files["qparams_reuse_validation"] = str(qparams_validation_path)
    row_counts["qparams_reuse_validation"] = int(len(qparams_validation))

    ranking_summary = pd.DataFrame()
    if not args.skip_ranking_drift:
        ranking_summary = compute_token_ranking_drift_summary(paths, chunksize=max(int(args.ranking_chunksize), 1))
    ranking_path = output_dir / "token_ranking_drift_summary.csv"
    _write_csv(ranking_summary, ranking_path)
    generated_files["token_ranking_drift_summary"] = str(ranking_path)
    row_counts["token_ranking_drift_summary"] = int(len(ranking_summary))

    group_summary = build_group_activation_token_summary(paths, ranking_summary)
    group_path = output_dir / "group_activation_token_summary.csv"
    _write_csv(group_summary, group_path)
    generated_files["group_activation_token_summary"] = str(group_path)
    row_counts["group_activation_token_summary"] = int(len(group_summary))

    geometry_summary = build_geometry_summary(paths)
    geometry_path = output_dir / "embedding_geometry_group_summary.csv"
    _write_csv(geometry_summary, geometry_path)
    generated_files["embedding_geometry_group_summary"] = str(geometry_path)
    row_counts["embedding_geometry_group_summary"] = int(len(geometry_summary))

    activation_sensitive, activation_comparison = build_activation_distribution_summary(paths, sensitive_targets)
    activation_sensitive_path = output_dir / "activation_distribution_sensitive_summary.csv"
    activation_comparison_path = output_dir / "activation_distribution_comparison_summary.csv"
    _write_csv(activation_sensitive, activation_sensitive_path)
    _write_csv(activation_comparison, activation_comparison_path)
    legacy_activation_all_path = output_dir / "activation_distribution_all_summary.csv"
    if legacy_activation_all_path.exists():
        legacy_activation_all_path.unlink()
    generated_files["activation_distribution_sensitive_summary"] = str(activation_sensitive_path)
    generated_files["activation_distribution_comparison_summary"] = str(activation_comparison_path)
    row_counts["activation_distribution_sensitive_summary"] = int(len(activation_sensitive))
    row_counts["activation_distribution_comparison_summary"] = int(len(activation_comparison))

    protocol_summary, source_manifests = build_protocol_outputs(paths)
    protocol_validation = validate_protocol(paths, source_manifests, qparams_validation)
    protocol_validation_path = output_dir / "protocol_validation.csv"
    _write_csv(protocol_validation, protocol_validation_path)
    generated_files["protocol_validation"] = str(protocol_validation_path)
    row_counts["protocol_validation"] = int(len(protocol_validation))

    protocol_csv_path = output_dir / "quant_injection_protocol_summary.csv"
    _write_csv(protocol_summary, protocol_csv_path)
    generated_files["quant_injection_protocol_summary"] = str(protocol_csv_path)
    row_counts["quant_injection_protocol_summary"] = int(len(protocol_summary))

    summary_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": paths.dataset_name,
        "dataset_root": str(paths.dataset_root),
        "output_dir": str(paths.output_dir),
        "group_definitions": {
            "many": "train_label_n >= 100",
            "medium": "20 <= train_label_n <= 99",
            "few": "train_label_n <= 19",
            "tail": "medium + few",
            "note": "tail and few are kept as separate analysis groups.",
        },
        "ranking_drift_definitions": {
            "ranking_drift_common": "mean absolute rank displacement over common fp32/int4 top-k tokens, normalized by max(k-1, 1)",
            "ranking_drift_penalized": "mean absolute rank displacement over fp32 top-k tokens; missing int4 tokens use one-past-last rank, normalized by k",
        },
        "metric_damage_rule": {
            "higher_is_better": "damage = 1 - value",
            "lower_is_better": "damage = value",
            "diagnostic": "damage is left blank",
            "gap": "comparison_damage - many_damage; positive means the comparison group is more damaged than many",
        },
        "top_sensitive_k": int(args.top_sensitive_k),
        "ranking_chunksize": int(args.ranking_chunksize),
        "skip_ranking_drift": bool(args.skip_ranking_drift),
        "generated_files": generated_files,
        "row_counts": row_counts,
        "protocol_validation": {
            "error_count": int((protocol_validation["status"] == "ERROR").sum()) if "status" in protocol_validation else 0,
            "warn_count": int((protocol_validation["status"] == "WARN").sum()) if "status" in protocol_validation else 0,
            "hard_fail_categories": [
                "split_reference_identity",
                "oracle_identity",
                "metric_protocol",
                "hook_target_position",
                "quantization_protocol",
            ],
        },
        "source_files": {
            "ffn_layer_heatmap_summary": str(paths.ffn_layer_heatmap_summary),
            "quant_mode_summary": str(paths.quant_mode_summary),
            "quant_mode_qparams_summary": str(paths.quant_mode_qparams_summary),
            "token_survival_summary": str(paths.token_survival_summary),
            "token_survival_metrics": str(paths.token_survival_metrics),
            "danger_zone_summary": str(paths.danger_zone_summary),
            "semantic_geometry_summary": str(paths.semantic_geometry_summary),
            "activation_distribution_summary": str(paths.activation_distribution_summary),
            "core_reference_manifest": str(paths.core_reference_manifest),
            "oer_oracle_manifest": str(paths.oer_oracle_manifest),
            "token_trace_oracle_manifest": str(paths.token_trace_oracle_manifest),
        },
        "source_manifest_top_level": {
            name: sorted(data.keys()) if isinstance(data, dict) else []
            for name, data in source_manifests.items()
        },
    }
    manifest_path = output_dir / "quant_injection_summary_manifest.json"
    _write_json(summary_manifest, manifest_path)
    generated_files["quant_injection_summary_manifest"] = str(manifest_path)

    error_rows = protocol_validation[protocol_validation["status"] == "ERROR"] if "status" in protocol_validation else pd.DataFrame()
    if not error_rows.empty:
        details = "; ".join(
            f"{row.get('category')}.{row.get('check_name')}: {row.get('detail')}"
            for _, row in error_rows.head(10).iterrows()
        )
        raise RuntimeError(f"Quant Injection protocol validation failed with {len(error_rows)} error(s): {details}")

    return summary_manifest


def main() -> None:
    manifest = run()
    print(json.dumps({"generated_files": manifest["generated_files"], "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
