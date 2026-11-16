from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import warnings

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.utils.group_utils import ANALYSIS_GROUPS, GROUP_PROTOCOL_VERSION, normalize_analysis_group
except ImportError:  # pragma: no cover
    from ..utils.group_utils import ANALYSIS_GROUPS, GROUP_PROTOCOL_VERSION, normalize_analysis_group


SATURATION_BOUNDARY_HIT_RATE_FILENAME = "boundary_hit_rate_by_target.png"
HIGH_ENERGY_NEAR_ZERO_RATE_FILENAME = "high_energy_near_zero_rate_by_target.png"
DANGER_COMPONENT_GAP_FILENAME = "danger_component_tail_many_gap.png"
COMPONENT_OER_CORRELATION_FILENAME = "danger_component_oer_correlation.png"
HIGH_ENERGY_BOUNDARY_HIT_HEATMAP_FILENAME = "high_energy_scalar_boundary_hit_rate_heatmap.png"
HIGH_ENERGY_CLIP_HEATMAP_FILENAME = HIGH_ENERGY_BOUNDARY_HIT_HEATMAP_FILENAME

GROUP_COLUMNS = ("target", "tensor_position", "granularity", "quant_mode", "analysis_group")
DEFAULT_ANALYSIS_GROUP_ORDER = tuple(ANALYSIS_GROUPS)


def _require_columns(df: Any, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"quant danger zone plot df missing columns: {', '.join(missing)}")


def _require_protocol_v2(df: Any, *, context: str, require_analysis_group: bool = True) -> None:
    import pandas as pd

    required = ["group_protocol_version", "metric_name", "metric_direction", "aggregation_unit", "aggregation_fn"]
    if require_analysis_group:
        required.append("analysis_group")
    _require_columns(df, required)
    version = pd.to_numeric(df["group_protocol_version"], errors="coerce")
    if bool(version.isna().any()) or bool((version < GROUP_PROTOCOL_VERSION).any()):
        raise ValueError(f"{context} requires group_protocol_version >= {GROUP_PROTOCOL_VERSION}")


def _to_numeric_series(series: Any) -> Any:
    import pandas as pd
    import torch

    def scalarize(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            return value.item() if value.numel() == 1 else float("nan")
        if isinstance(value, np.ndarray):
            return float(value.reshape(-1)[0]) if value.size == 1 else float("nan")
        return value

    return pd.to_numeric(series.map(scalarize), errors="coerce")


def _prepare_save_path(save_dir: str | Path, filename: str) -> Path:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def _ordered_unique(values: Sequence[Any], preferred: Sequence[str] | None = None) -> list[str]:
    items = [str(value) for value in values if str(value).strip().lower() not in {"", "nan", "<na>", "none"}]
    unique = list(dict.fromkeys(items))
    if preferred is None:
        return unique
    preferred_set = {str(item) for item in preferred}
    ordered = [str(item) for item in preferred if str(item) in unique]
    ordered.extend([item for item in unique if item not in preferred_set])
    return ordered


def _target_sort_key(target: Any) -> tuple[int, str, str]:
    import re

    text = str(target)
    match = re.search(r"L(\d+)[_\s-]*(fc[12])", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).lower(), text
    return 10_000, text, text


def _target_order(work: Any) -> list[str]:
    return sorted(_ordered_unique(work["target"].tolist()), key=_target_sort_key)


def _facet_order(work: Any) -> list[tuple[str, str, str]]:
    facets = work[["tensor_position", "granularity", "quant_mode"]].astype(str).drop_duplicates().sort_values(["tensor_position", "granularity", "quant_mode"])
    return [tuple(row) for row in facets.to_numpy()]


def _assert_unique_plot_keys(work: Any, keys: Sequence[str], *, context: str) -> None:
    duplicate_keys = work.duplicated(list(keys), keep=False)
    if bool(duplicate_keys.any()):
        examples = work.loc[duplicate_keys, list(keys)].drop_duplicates().head(8).to_dict(orient="records")
        raise ValueError(f"{context} has duplicate plot keys; build the v2 aggregate summary first: {examples}")


def _warn_or_raise_metric_request(problems: Sequence[str], *, context: str, formal: bool) -> bool:
    if not problems:
        return True
    message = f"{context} requested metrics are missing or all-NaN: " + "; ".join(str(item) for item in problems)
    if formal:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return False


def _requested_metric_problems(
    df: Any,
    *,
    metric_column: str,
    metrics: Sequence[str],
    value_columns: Sequence[str],
) -> list[str]:
    metric_values = df[metric_column].astype(str)
    problems: list[str] = []
    for metric in metrics:
        part = df[metric_values.eq(str(metric))]
        if len(part) == 0:
            problems.append(f"{metric}: missing")
            continue
        finite_count = 0
        for column in value_columns:
            if column in part.columns:
                finite_count += int(_to_numeric_series(part[column]).notna().sum())
        if finite_count <= 0:
            problems.append(f"{metric}: all NaN in {', '.join(value_columns)}")
    return problems


def _require_requested_metrics(
    df: Any,
    *,
    metric_column: str,
    metrics: Sequence[str],
    value_columns: Sequence[str],
    context: str,
    formal: bool,
) -> bool:
    _require_columns(df, (metric_column, *value_columns))
    problems = _requested_metric_problems(
        df,
        metric_column=metric_column,
        metrics=metrics,
        value_columns=value_columns,
    )
    return _warn_or_raise_metric_request(problems, context=context, formal=formal)


def _prepare_metric_frame(df: Any, metric_name: str, *, formal: bool = True) -> Any:
    import pandas as pd

    _require_protocol_v2(df, context=f"summary metric {metric_name}")
    _require_columns(df, ("target", "tensor_position", "granularity", "quant_mode", "metric_value"))
    _require_requested_metrics(
        df,
        metric_column="metric_name",
        metrics=(metric_name,),
        value_columns=("metric_value",),
        context=f"summary metric {metric_name}",
        formal=formal,
    )
    work = df[df["metric_name"].astype(str).eq(str(metric_name))][list(GROUP_COLUMNS) + ["metric_value"]].copy()
    for column in ("target", "tensor_position", "granularity", "quant_mode"):
        work[column] = work[column].astype(str)
    work["analysis_group"] = work["analysis_group"].map(normalize_analysis_group)
    work["metric_value"] = _to_numeric_series(work["metric_value"])
    work = work.dropna(subset=["metric_value"]).reset_index(drop=True)
    if len(work) == 0:
        message = f"{metric_name} has no plottable data"
        if formal:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return pd.DataFrame(work)
    _assert_unique_plot_keys(work, GROUP_COLUMNS, context=metric_name)
    return pd.DataFrame(work)


def _values_for_facet(work: Any, facet: tuple[str, str, str], analysis_group: str, targets: Sequence[str]) -> np.ndarray:
    tensor_position, granularity, quant_mode = facet
    part = work[
        (work["tensor_position"].astype(str) == tensor_position)
        & (work["granularity"].astype(str) == granularity)
        & (work["quant_mode"].astype(str) == quant_mode)
        & (work["analysis_group"] == analysis_group)
    ]
    values = part.set_index("target")["metric_value"]
    return values.reindex(list(targets)).to_numpy(dtype=float)


def _set_target_ticks(ax: Any, x_pos: np.ndarray, labels: Sequence[str]) -> None:
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)


def _plot_metric_bar_by_target(
    df: Any,
    save_dir: str | Path,
    *,
    metric_name: str,
    filename: str,
    title: str,
    ylabel: str,
    formal: bool = True,
) -> Path | None:
    work = _prepare_metric_frame(df, metric_name, formal=formal)
    if len(work) == 0:
        return None
    analysis_groups = _ordered_unique(work["analysis_group"].tolist(), DEFAULT_ANALYSIS_GROUP_ORDER)
    targets = _target_order(work)
    facets = _facet_order(work)

    x_pos = np.arange(len(targets), dtype=float)
    width = min(0.22, 0.72 / max(1, len(analysis_groups)))
    offsets = (np.arange(len(analysis_groups), dtype=float) - (len(analysis_groups) - 1) / 2.0) * width

    fig, axes = plt.subplots(len(facets), 1, figsize=(max(9.0, 0.7 * len(targets)), max(3.0 * len(facets), 3.2)), squeeze=False, constrained_layout=True)
    for row_idx, facet in enumerate(facets):
        ax = axes[row_idx][0]
        for group_idx, analysis_group in enumerate(analysis_groups):
            values = _values_for_facet(work, facet, analysis_group, targets)
            ax.bar(x_pos + offsets[group_idx], values, width=width, label=analysis_group)
        _set_target_ticks(ax, x_pos, targets)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} | {facet[0]} | {facet[1]} | {facet[2]}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=min(4, len(analysis_groups)), fontsize=8)

    output_path = _prepare_save_path(save_dir, filename)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_saturation_boundary_hit_rate_bar(df: Any, save_dir: str | Path, *, formal: bool = True) -> Path | None:
    return _plot_metric_bar_by_target(
        df,
        save_dir,
        metric_name="boundary_hit_rate",
        filename=SATURATION_BOUNDARY_HIT_RATE_FILENAME,
        title="Saturation Boundary Pressure by Target",
        ylabel="boundary_hit_rate",
        formal=formal,
    )


def plot_high_energy_near_zero_rate_bar(df: Any, save_dir: str | Path, *, formal: bool = True) -> Path | None:
    return _plot_metric_bar_by_target(
        df,
        save_dir,
        metric_name="high_energy_near_zero_rate",
        filename=HIGH_ENERGY_NEAR_ZERO_RATE_FILENAME,
        title="High-Energy Near-Zero Rate by Target",
        ylabel="high_energy_near_zero_rate",
        formal=formal,
    )


def plot_danger_component_gap(comparison_df: Any, save_dir: str | Path, *, formal: bool = True) -> Path | None:
    import pandas as pd

    _require_protocol_v2(comparison_df, context="danger component comparison", require_analysis_group=False)
    _require_columns(comparison_df, ("target", "tensor_position", "granularity", "quant_mode", "tail_vs_many_gap", "few_vs_many_gap"))
    metrics = [
        "strict_overflow_rate",
        "overflow_energy_proxy",
        "high_energy_near_zero_rate",
        "topk_token_boundary_hit_rate",
        "high_energy_scalar_boundary_hit_rate",
        "oer_loss_total",
    ]
    _require_requested_metrics(
        comparison_df,
        metric_column="metric_name",
        metrics=metrics,
        value_columns=("tail_vs_many_gap", "few_vs_many_gap"),
        context="danger component comparison",
        formal=formal,
    )
    work = comparison_df[comparison_df["metric_name"].astype(str).isin(metrics)].copy()
    work["tail_vs_many_gap"] = _to_numeric_series(work["tail_vs_many_gap"])
    work["few_vs_many_gap"] = _to_numeric_series(work["few_vs_many_gap"])
    work = work.dropna(subset=["tail_vs_many_gap", "few_vs_many_gap"], how="all")
    if len(work) == 0:
        message = "danger component comparison has no plottable gaps"
        if formal:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return None
    work["series"] = (
        work["metric_name"].astype(str)
        + " | " + work["tensor_position"].astype(str)
        + " | " + work["granularity"].astype(str)
        + " | " + work["quant_mode"].astype(str)
    )
    targets = _target_order(work)
    series_order = _ordered_unique(work["series"].tolist())
    fig, axes = plt.subplots(len(series_order), 1, figsize=(max(9.0, 0.7 * len(targets)), max(2.6 * len(series_order), 3.2)), squeeze=False, constrained_layout=True)
    x_pos = np.arange(len(targets), dtype=float)
    for row_idx, series in enumerate(series_order):
        ax = axes[row_idx][0]
        part = work[work["series"] == series]
        tail_values = part.set_index("target")["tail_vs_many_gap"].reindex(targets).to_numpy(dtype=float)
        few_values = part.set_index("target")["few_vs_many_gap"].reindex(targets).to_numpy(dtype=float)
        ax.bar(x_pos - 0.17, tail_values, width=0.32, label="tail_vs_many_gap")
        ax.bar(x_pos + 0.17, few_values, width=0.32, label="few_vs_many_gap")
        _set_target_ticks(ax, x_pos, targets)
        ax.set_ylabel("damage gap")
        ax.set_title(series)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    output_path = _prepare_save_path(save_dir, DANGER_COMPONENT_GAP_FILENAME)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_component_oer_correlation(correlation_df: Any, save_dir: str | Path) -> Path:
    _require_protocol_v2(correlation_df, context="component OER correlation")
    _require_columns(correlation_df, ("target", "tensor_position", "granularity", "quant_mode", "component_metric", "pearson"))
    work = correlation_df.copy()
    work["analysis_group"] = work["analysis_group"].map(normalize_analysis_group)
    work["pearson"] = _to_numeric_series(work["pearson"])
    work = work.dropna(subset=["pearson"])
    if len(work) == 0:
        raise ValueError("component OER correlation has no plottable data")
    targets = _target_order(work)
    work["row_key"] = (
        work["component_metric"].astype(str)
        + " | " + work["tensor_position"].astype(str)
        + " | " + work["analysis_group"].astype(str)
        + " | " + work["granularity"].astype(str)
        + " | " + work["quant_mode"].astype(str)
    )
    row_order = _ordered_unique(work["row_key"].tolist())
    _assert_unique_plot_keys(work, ["row_key", "target"], context="component OER correlation")
    heatmap = work.pivot(index="row_key", columns="target", values="pearson").reindex(index=row_order, columns=targets)
    matrix = heatmap.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.7 * len(targets)), max(3.2, 0.42 * len(row_order))), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8)
    ax.set_title("Danger Component vs OER Loss Correlation")
    ax.set_xlabel("target")
    ax.set_ylabel("component | tensor_position | analysis_group | granularity | quant_mode")
    fig.colorbar(image, ax=ax).set_label("Pearson r")
    output_path = _prepare_save_path(save_dir, COMPONENT_OER_CORRELATION_FILENAME)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_high_energy_boundary_hit_heatmap(df: Any, save_dir: str | Path, *, formal: bool = True) -> Path | None:
    work = _prepare_metric_frame(df, "high_energy_scalar_boundary_hit_rate", formal=formal)
    if len(work) == 0:
        return None
    targets = _target_order(work)
    work["row_key"] = work["tensor_position"].astype(str) + " | " + work["granularity"].astype(str) + " | " + work["quant_mode"].astype(str) + " | " + work["analysis_group"].astype(str)
    row_order = _ordered_unique(work["row_key"].tolist())
    heatmap = work.pivot(index="row_key", columns="target", values="metric_value").reindex(index=row_order, columns=targets)
    matrix = heatmap.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.7 * len(targets)), max(3.2, 0.42 * len(row_order))), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8)
    ax.set_title("High-Energy Scalar Boundary-Hit Rate")
    ax.set_xlabel("target")
    ax.set_ylabel("tensor_position | granularity | quant_mode | analysis_group")
    fig.colorbar(image, ax=ax).set_label("high_energy_scalar_boundary_hit_rate")
    output_path = _prepare_save_path(save_dir, HIGH_ENERGY_CLIP_HEATMAP_FILENAME)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_high_energy_clip_heatmap(df: Any, save_dir: str | Path, *, formal: bool = True) -> Path | None:
    return plot_high_energy_boundary_hit_heatmap(df, save_dir, formal=formal)


def plot_all_quant_danger_zone_figures(
    df: Any,
    save_dir: str | Path,
    *,
    comparison_df: Any | None = None,
    correlation_df: Any | None = None,
    formal: bool = True,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, path in (
        ("boundary_hit_rate", plot_saturation_boundary_hit_rate_bar(df, save_dir, formal=formal)),
        ("high_energy_near_zero_rate", plot_high_energy_near_zero_rate_bar(df, save_dir, formal=formal)),
        ("high_energy_scalar_boundary_hit_rate", plot_high_energy_boundary_hit_heatmap(df, save_dir, formal=formal)),
    ):
        if path is not None:
            paths[key] = path
    if comparison_df is not None:
        path = plot_danger_component_gap(comparison_df, save_dir, formal=formal)
        if path is not None:
            paths["danger_component_gap"] = path
    if correlation_df is not None:
        paths["component_oer_correlation"] = plot_component_oer_correlation(correlation_df, save_dir)
    return paths


__all__ = [
    "COMPONENT_OER_CORRELATION_FILENAME",
    "DANGER_COMPONENT_GAP_FILENAME",
    "HIGH_ENERGY_BOUNDARY_HIT_HEATMAP_FILENAME",
    "HIGH_ENERGY_CLIP_HEATMAP_FILENAME",
    "HIGH_ENERGY_NEAR_ZERO_RATE_FILENAME",
    "SATURATION_BOUNDARY_HIT_RATE_FILENAME",
    "plot_all_quant_danger_zone_figures",
    "plot_component_oer_correlation",
    "plot_danger_component_gap",
    "plot_high_energy_boundary_hit_heatmap",
    "plot_high_energy_clip_heatmap",
    "plot_high_energy_near_zero_rate_bar",
    "plot_saturation_boundary_hit_rate_bar",
]
