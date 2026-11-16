from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.utils.group_utils import GROUP_PROTOCOL_VERSION
except ImportError:  # pragma: no cover - package-relative debugging.
    from ..utils.group_utils import GROUP_PROTOCOL_VERSION


EMBEDDING_DISTORTION_FILENAME = "embedding_distortion_tail_many_gap.png"
OER_RETENTION_FILENAME = "oer_retention_tail_many_gap.png"
GEOMETRY_RETENTION_FILENAME = "geometry_retention_tail_many_gap.png"

DEFAULT_QUANT_MODE_ORDER = ("W4-only", "A4-only", "W4A4")
DEFAULT_GEOMETRY_METRICS = ("margin_retention", "fisher_retention")

_QUANT_MODE_LABELS = {
    "w4": "W4-only",
    "w4-only": "W4-only",
    "w4_only": "W4-only",
    "a4": "A4-only",
    "a4-only": "A4-only",
    "a4_only": "A4-only",
    "w4a4": "W4A4",
    "w4-a4": "W4A4",
    "w4_a4": "W4A4",
}


def _require_columns(df: Any, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"ffn quant mode plot df missing columns: {', '.join(missing)}")


def _to_numeric_series(series: Any) -> Any:
    import pandas as pd
    import torch

    def scalarize(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return float("nan")
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.reshape(-1)[0])
            return float("nan")
        return value

    return pd.to_numeric(series.map(scalarize), errors="coerce")


def _require_protocol_v2(df: Any, *, context: str) -> None:
    _require_columns(df, ("group_protocol_version",))
    versions = _to_numeric_series(df["group_protocol_version"])
    finite = versions.dropna()
    if len(finite) == 0 or bool((finite < int(GROUP_PROTOCOL_VERSION)).any()):
        raise ValueError(f"{context} requires group_protocol_version >= {int(GROUP_PROTOCOL_VERSION)}")


def _assert_no_duplicate_keys(df: Any, keys: Sequence[str], *, context: str) -> None:
    duplicated = df.duplicated(subset=list(keys), keep=False)
    if bool(duplicated.any()):
        preview = df.loc[duplicated, list(keys)].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{context} has duplicate rows for {list(keys)}: {preview}")


def _prepare_save_path(save_dir: str | Path, filename: str) -> Path:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def _normalize_quant_mode(value: Any) -> str:
    text = str(value)
    key = text.strip().lower()
    return _QUANT_MODE_LABELS.get(key, text)


def _target_label(layer_idx: Any, module_name: Any) -> str:
    try:
        layer_text = str(int(layer_idx))
    except (TypeError, ValueError):
        layer_text = str(layer_idx)
    return f"L{layer_text} {module_name}"


def _ordered_unique(values: Sequence[Any], preferred: Sequence[str] | None = None) -> list[str]:
    seen = [str(value) for value in values if str(value).strip().lower() not in {"", "nan", "<na>", "none"}]
    unique = list(dict.fromkeys(seen))
    if preferred is None:
        return unique
    preferred_set = {str(item) for item in preferred}
    ordered = [str(item) for item in preferred if str(item) in unique]
    ordered.extend([item for item in unique if item not in preferred_set])
    return ordered


def _target_order(work: Any) -> list[str]:
    ordered = work[["layer_idx", "module_name", "target"]].drop_duplicates()
    ordered = ordered.sort_values(["layer_idx", "module_name"])
    return ordered["target"].astype(str).tolist()


def _set_target_ticks(ax: Any, x_pos: np.ndarray, labels: Sequence[str]) -> None:
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)


def _series_for_target(frame: Any, *, value_column: str, target_order: Sequence[str]) -> np.ndarray:
    _assert_no_duplicate_keys(frame, ("target",), context=f"{value_column} gap series")
    values = {str(target): float(value) for target, value in zip(frame["target"], frame[value_column])}
    return np.asarray([values.get(str(target), np.nan) for target in target_order], dtype=float)


def _comparison_source(metric_df: Any, comparison_df: Any | None) -> Any:
    source = comparison_df if comparison_df is not None else metric_df
    _require_columns(
        source,
        (
            "layer_idx",
            "module_name",
            "quant_mode",
            "metric_name",
            "tail_vs_many_gap",
            "few_vs_many_gap",
        ),
    )
    return source


def _prepare_gap_frame(metric_df: Any, comparison_df: Any | None, metric_names: Sequence[str]) -> Any:
    source = _comparison_source(metric_df, comparison_df)
    _require_protocol_v2(source, context="ffn quant mode comparison plot")
    _require_columns(source, ("metric_direction", "aggregation_unit", "aggregation_fn"))

    work = source.copy()
    work["metric_name"] = work["metric_name"].astype(str)
    work = work[work["metric_name"].isin([str(name) for name in metric_names])]
    if "gap_enabled" in work.columns:
        gap_enabled = work["gap_enabled"].astype(str).str.lower().isin({"true", "1", "yes"})
        work = work[gap_enabled]
    if len(work) == 0:
        raise ValueError(f"No comparable comparison rows found for metrics: {', '.join(metric_names)}")

    work["layer_idx"] = _to_numeric_series(work["layer_idx"])
    work["module_name"] = work["module_name"].astype(str)
    work["quant_mode"] = work["quant_mode"].map(_normalize_quant_mode)
    work["tail_vs_many_gap"] = _to_numeric_series(work["tail_vs_many_gap"])
    work["few_vs_many_gap"] = _to_numeric_series(work["few_vs_many_gap"])
    work = work.dropna(subset=["layer_idx"])
    work["layer_idx"] = work["layer_idx"].astype(int)
    work["target"] = [_target_label(layer, module) for layer, module in zip(work["layer_idx"], work["module_name"])]

    key_columns = ["layer_idx", "module_name", "target", "quant_mode", "metric_name"]
    _assert_no_duplicate_keys(work, key_columns, context="ffn quant mode comparison summary")
    return work.reset_index(drop=True)


def _plot_gap_metrics(
    metric_df: Any,
    save_dir: str | Path,
    *,
    metric_names: Sequence[str],
    filename: str,
    title: str,
    ylabel: str = "damage gap",
    comparison_df: Any | None = None,
) -> Path:
    work = _prepare_gap_frame(metric_df, comparison_df, metric_names)
    quant_modes = _ordered_unique(work["quant_mode"].tolist(), DEFAULT_QUANT_MODE_ORDER)
    metrics = _ordered_unique(work["metric_name"].tolist(), metric_names)
    targets = _target_order(work)
    x_pos = np.arange(len(targets), dtype=float)

    n_rows = len(metrics)
    n_cols = len(quant_modes)
    fig_width = max(4.6 * n_cols, 8.0)
    fig_height = max(3.2 * n_rows, 3.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=True)

    for metric_idx, metric_name in enumerate(metrics):
        for mode_idx, quant_mode in enumerate(quant_modes):
            ax = axes[metric_idx][mode_idx]
            part = work[(work["metric_name"] == metric_name) & (work["quant_mode"] == quant_mode)]
            tail_gap = _series_for_target(part, value_column="tail_vs_many_gap", target_order=targets)
            few_gap = _series_for_target(part, value_column="few_vs_many_gap", target_order=targets)
            ax.bar(x_pos, tail_gap, width=0.62, label="tail_vs_many_gap")
            ax.scatter(
                x_pos,
                few_gap,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=0.9,
                label="few_vs_many_gap",
                zorder=3,
            )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
            _set_target_ticks(ax, x_pos, targets)
            ax.set_title(f"{metric_name} | {quant_mode}")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=8)

    fig.suptitle(title)
    output_path = _prepare_save_path(save_dir, filename)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_embedding_distortion_bar(df: Any, save_dir: str | Path, *, comparison_df: Any | None = None) -> Path:
    """Plot tail-vs-many comparison for embedding distortion."""

    return _plot_gap_metrics(
        df,
        save_dir,
        metric_names=("embedding_distortion",),
        filename=EMBEDDING_DISTORTION_FILENAME,
        title="Embedding Distortion Tail-vs-Many Gap",
        ylabel="tail damage gap",
        comparison_df=comparison_df,
    )


def plot_oer_retention_bar(df: Any, save_dir: str | Path, *, comparison_df: Any | None = None) -> Path:
    """Plot tail-vs-many comparison for input-activation OER retention."""

    return _plot_gap_metrics(
        df,
        save_dir,
        metric_names=("oer_retention",),
        filename=OER_RETENTION_FILENAME,
        title="Input Activation OER Retention Tail-vs-Many Gap",
        ylabel="tail retention gap",
        comparison_df=comparison_df,
    )


def plot_geometry_retention_comparison(
    df: Any,
    save_dir: str | Path,
    *,
    metric_columns: Sequence[str] = DEFAULT_GEOMETRY_METRICS,
    comparison_df: Any | None = None,
) -> Path:
    """Plot tail-vs-many comparison for geometry retention metrics."""

    return _plot_gap_metrics(
        df,
        save_dir,
        metric_names=tuple(metric_columns),
        filename=GEOMETRY_RETENTION_FILENAME,
        title="Geometry Retention Tail-vs-Many Gap",
        ylabel="tail retention gap",
        comparison_df=comparison_df,
    )


def plot_all_quant_mode_plots(df: Any, save_dir: str | Path, *, comparison_df: Any | None = None) -> dict[str, Path]:
    """Draw all quant-mode ablation figures from protocol-v2 comparison summaries."""

    embedding = plot_embedding_distortion_bar(df, save_dir, comparison_df=comparison_df)
    oer = plot_oer_retention_bar(df, save_dir, comparison_df=comparison_df)
    geometry = plot_geometry_retention_comparison(df, save_dir, comparison_df=comparison_df)
    return {
        "embedding_distortion": embedding,
        "embedding_distortion_gap": embedding,
        "oer_retention": oer,
        "oer_retention_gap": oer,
        "geometry_retention": geometry,
        "geometry_retention_gap": geometry,
    }


plot_embedding_distortion_by_quant_mode = plot_embedding_distortion_bar
plot_oer_retention_by_quant_mode = plot_oer_retention_bar
plot_few_many_geometry_retention_comparison = plot_geometry_retention_comparison


__all__ = [
    "EMBEDDING_DISTORTION_FILENAME",
    "GEOMETRY_RETENTION_FILENAME",
    "OER_RETENTION_FILENAME",
    "plot_all_quant_mode_plots",
    "plot_embedding_distortion_bar",
    "plot_embedding_distortion_by_quant_mode",
    "plot_few_many_geometry_retention_comparison",
    "plot_geometry_retention_comparison",
    "plot_oer_retention_bar",
    "plot_oer_retention_by_quant_mode",
]
