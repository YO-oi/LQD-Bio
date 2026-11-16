from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.utils.group_utils import normalize_analysis_group
except ImportError:  # pragma: no cover - package-relative debugging.
    from ..utils.group_utils import normalize_analysis_group


HISTOGRAM_FILENAME = "activation_histogram_log.png"
ACTIVATION_RANGE_FILENAME = "activation_range_bar.png"
ADAPTIVE_CLIPPING_FILENAME = "adaptive_clipping_rate_tensor_vs_channel.png"
FIXED_CLIPPING_FILENAME = "fixed_clipping_rate_tensor_vs_channel.png"
ADAPTIVE_OER_FILENAME = "adaptive_oer_retention_tensor_vs_channel.png"
FIXED_OER_FILENAME = "fixed_oer_retention_tensor_vs_channel.png"


def _require_columns(df: Any, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"activation distribution df missing columns: {', '.join(missing)}")


def _require_protocol_v2(df: Any) -> None:
    import pandas as pd

    _require_columns(df, ("group_protocol_version",))
    versions = pd.to_numeric(df["group_protocol_version"], errors="coerce")
    if versions.isna().any():
        raise ValueError("activation distribution plots require group_protocol_version=2 on every row")
    if (versions < 2).any():
        bad = sorted({str(item) for item in df.loc[versions < 2, "group_protocol_version"].tolist()})
        raise ValueError(f"activation distribution plots refuse pre-v2 group protocol rows: {bad}")


def _row_type_series(df: Any) -> Any:
    _require_columns(df, ("row_type",))
    return df["row_type"].astype(str).str.strip()


def _histogram_df(df: Any) -> Any:
    _require_protocol_v2(df)
    row_type = _row_type_series(df)
    work = df.loc[row_type == "histogram_summary"].copy()
    if len(work) == 0:
        raise ValueError("activation histogram requires histogram_summary rows")
    return _ordered_df(work)


def _summary_wide_df(df: Any) -> Any:
    _require_protocol_v2(df)
    _require_columns(df, ("row_type", "metric_name", "metric_value", "analysis_group"))
    row_type = _row_type_series(df)
    work = df.loc[row_type == "metric_summary"].copy()
    if len(work) == 0:
        raise ValueError("activation summary plots require metric_summary rows")

    work["analysis_group"] = work["analysis_group"].map(normalize_analysis_group)
    work["metric_value"] = _to_numeric_series(work["metric_value"])
    index_candidates = (
        "layer_idx",
        "module_name",
        "activation_kind",
        "target_name",
        "analysis_group",
        "sample_count",
        "class_count",
    )
    index_columns = [column for column in index_candidates if column in work.columns]
    if not index_columns:
        raise KeyError("activation summary rows need at least one target/index column")

    duplicate_mask = work.duplicated(index_columns + ["metric_name"], keep=False)
    if duplicate_mask.any():
        example = work.loc[duplicate_mask, index_columns + ["metric_name"]].head(3).to_dict("records")
        raise ValueError(f"duplicate activation metric_summary rows would make plotting ambiguous: {example}")

    wide = work.set_index(index_columns + ["metric_name"])["metric_value"].unstack("metric_name").reset_index()
    wide.columns = [str(column) for column in wide.columns]
    return _ordered_df(wide)


def _summary_gap_wide_df(df: Any, metric_names: Sequence[str]) -> Any:
    _require_protocol_v2(df)
    _require_columns(df, ("row_type", "metric_name", "analysis_group", "tail_vs_many_gap", "few_vs_many_gap"))
    row_type = _row_type_series(df)
    metric_set = {str(metric) for metric in metric_names}
    work = df.loc[(row_type == "metric_summary") & df["metric_name"].astype(str).isin(metric_set)].copy()
    if len(work) == 0:
        raise ValueError(f"activation summary gap plots require metric_summary rows for {sorted(metric_set)}")

    work["analysis_group"] = work["analysis_group"].map(normalize_analysis_group)
    work["tail_vs_many_gap"] = _to_numeric_series(work["tail_vs_many_gap"])
    work["few_vs_many_gap"] = _to_numeric_series(work["few_vs_many_gap"])
    group_rank = {"tail": 0, "many": 1, "medium": 2, "few": 3}
    work["__group_rank"] = work["analysis_group"].map(lambda value: group_rank.get(str(value), 99))

    index_candidates = ("layer_idx", "module_name", "activation_kind", "target_name", "tensor_name")
    index_columns = [column for column in index_candidates if column in work.columns]
    if not index_columns:
        work["__target_index"] = range(len(work))
        index_columns = ["__target_index"]

    work = work.sort_values(index_columns + ["metric_name", "__group_rank"])
    work = work.drop_duplicates(index_columns + ["metric_name"], keep="first")
    duplicate_mask = work.duplicated(index_columns + ["metric_name"], keep=False)
    if duplicate_mask.any():
        example = work.loc[duplicate_mask, index_columns + ["metric_name"]].head(3).to_dict("records")
        raise ValueError(f"duplicate activation gap rows would make plotting ambiguous: {example}")

    tail_wide = work.set_index(index_columns + ["metric_name"])["tail_vs_many_gap"].unstack("metric_name")
    few_wide = work.set_index(index_columns + ["metric_name"])["few_vs_many_gap"].unstack("metric_name")
    tail_wide.columns = [str(column) for column in tail_wide.columns]
    few_wide.columns = [f"{column}_few_vs_many_gap" for column in few_wide.columns]
    wide = tail_wide.join(few_wide).reset_index()
    wide.columns = [str(column) for column in wide.columns]
    if "__target_index" in wide.columns:
        wide = wide.drop(columns=["__target_index"])
    return _ordered_df(wide)


def _plot_tail_gap_bars(
    df: Any,
    save_dir: str | Path,
    *,
    metric_columns: Sequence[str],
    ylabel: str,
    title: str,
    filename: str,
    metric_labels: Mapping[str, str] | None = None,
) -> Path:
    work = _summary_gap_wide_df(df, metric_columns)
    _require_columns(work, metric_columns)
    labels = _target_labels(work)
    x_pos = np.arange(len(work), dtype=float)
    metric_count = max(len(metric_columns), 1)
    width = min(0.34, 0.76 / float(metric_count))
    offsets = (np.arange(metric_count, dtype=float) - (metric_count - 1) / 2.0) * width
    label_map = {} if metric_labels is None else dict(metric_labels)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(work)), 4.2), constrained_layout=True)
    few_marker_added = False
    for idx, metric in enumerate(metric_columns):
        display = label_map.get(metric, metric)
        values = _to_numeric_series(work[metric]).to_numpy(dtype=float)
        positions = x_pos + offsets[idx]
        ax.bar(positions, values, width=width, label=f"{display} tail-many")
        few_column = f"{metric}_few_vs_many_gap"
        if few_column in work.columns:
            few_values = _to_numeric_series(work[few_column]).to_numpy(dtype=float)
            if np.isfinite(few_values).any():
                ax.scatter(
                    positions,
                    few_values,
                    marker="o",
                    s=24,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.8,
                    label="few-many" if not few_marker_added else None,
                    zorder=3,
                )
                few_marker_added = True

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    _set_target_ticks(ax, x_pos, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    output_path = _prepare_save_path(save_dir, filename)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


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


def _list_like_to_numpy(value: Any, *, name: str) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} is empty")
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{name} is not a parseable list/tuple string: {text[:80]}") from exc
        array = np.asarray(parsed)
    else:
        raise TypeError(f"{name} must be list/tuple/ndarray/tensor or a string representation, got {type(value)!r}")

    array = np.asarray(array, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return array


def _first_existing_column(df: Any, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _histogram_columns(df: Any) -> tuple[str, str, bool, bool]:
    density_column = _first_existing_column(
        df,
        (
            "hist_density",
            "histogram_density",
            "activation_hist_density",
            "bin_density",
            "density",
        ),
    )
    count_column = _first_existing_column(
        df,
        (
            "hist_counts",
            "histogram_counts",
            "activation_hist_counts",
            "bin_counts",
            "counts",
        ),
    )
    value_column = density_column if density_column is not None else count_column
    is_density = density_column is not None
    if value_column is None:
        raise KeyError("activation histogram requires hist_density or precomputed count columns, e.g. hist_counts")

    edge_column = _first_existing_column(
        df,
        (
            "hist_edges",
            "histogram_edges",
            "activation_hist_edges",
            "bin_edges",
        ),
    )
    if edge_column is not None:
        return value_column, edge_column, True, is_density

    center_column = _first_existing_column(
        df,
        (
            "hist_centers",
            "histogram_centers",
            "activation_hist_centers",
            "bin_centers",
        ),
    )
    if center_column is None:
        raise KeyError("activation histogram requires precomputed bin edges or centers")
    return value_column, center_column, False, is_density


def _analysis_group_suffix(row: Any) -> str:
    if "analysis_group" not in row.index:
        return ""
    value = str(row["analysis_group"]).strip()
    if not value or value.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return f" | {normalize_analysis_group(value)}"


def _target_label(row: Any, index: Any) -> str:
    suffix = _analysis_group_suffix(row)
    if "layer_idx" in row.index and "module_name" in row.index:
        if "activation_kind" in row.index:
            kind = str(row["activation_kind"]).replace("_activation", "")
            return f"L{row['layer_idx']} {row['module_name']} {kind}{suffix}"
        return f"L{row['layer_idx']} {row['module_name']}{suffix}"
    if "target_name" in row.index:
        return f"{row['target_name']}{suffix}"
    if "tensor_name" in row.index:
        return f"{row['tensor_name']}{suffix}"
    if "module_name" in row.index:
        return f"{row['module_name']}{suffix}"
    return f"{index}{suffix}"


def _ordered_df(df: Any) -> Any:
    work = df.copy()
    sort_columns = [column for column in ("layer_idx", "module_name", "target_name", "tensor_name") if column in work.columns]
    if sort_columns:
        return work.sort_values(sort_columns).reset_index(drop=True)
    return work.reset_index(drop=True)


def _target_labels(df: Any) -> list[str]:
    return [_target_label(row, index) for index, row in df.iterrows()]


def _prepare_save_path(save_dir: str | Path, filename: str) -> Path:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def _set_target_ticks(ax: Any, positions: np.ndarray, labels: Sequence[str]) -> None:
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)


def plot_activation_histograms(df: Any, save_dir: str | Path) -> Path:
    """Plot precomputed activation histograms with a log-scaled density axis."""

    work = _histogram_df(df)
    if len(work) == 0:
        raise ValueError("activation distribution df cannot be empty")
    value_column, bin_column, is_edges, is_density = _histogram_columns(work)

    n_items = len(work)
    n_cols = min(4, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig_width = max(4.0 * n_cols, 6.0)
    fig_height = max(2.7 * n_rows, 3.0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=True)

    for item_idx, (row_index, row) in enumerate(work.iterrows()):
        ax = axes[item_idx // n_cols][item_idx % n_cols]
        values = _list_like_to_numpy(row[value_column], name=value_column)
        bins = _list_like_to_numpy(row[bin_column], name=bin_column)
        if not is_density:
            total = float(np.sum(values))
            values = values / total if total > 0.0 else np.full_like(values, np.nan, dtype=float)
        if is_edges:
            if bins.size != values.size + 1:
                raise ValueError(
                    f"{_target_label(row, row_index)}: histogram edges length must equal value length + 1; "
                    f"got {bins.size} vs {values.size}"
                )
            x_values = (bins[:-1] + bins[1:]) * 0.5
        else:
            if bins.size != values.size:
                raise ValueError(
                    f"{_target_label(row, row_index)}: histogram centers length must equal value length; "
                    f"got {bins.size} vs {values.size}"
                )
            x_values = bins

        y_values = np.where(values > 0, values, np.nan)
        ax.plot(x_values, y_values, linewidth=1.2)
        ax.set_yscale("log")
        ax.set_title(_target_label(row, row_index), fontsize=9)
        ax.set_xlabel("activation")
        ax.set_ylabel("normalized density")
        ax.grid(True, axis="y", alpha=0.25)

    for empty_idx in range(n_items, n_rows * n_cols):
        axes[empty_idx // n_cols][empty_idx % n_cols].axis("off")

    output_path = _prepare_save_path(save_dir, HISTOGRAM_FILENAME)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_activation_range_bar(df: Any, save_dir: str | Path) -> Path:
    """Plot tail-vs-many activation range gaps for max_abs and p99."""

    return _plot_tail_gap_bars(
        df,
        save_dir,
        metric_columns=("max_abs", "p99"),
        ylabel="tail_vs_many_gap",
        title="Activation Heavy-Tail Range Gap (Tail vs Many)",
        filename=ACTIVATION_RANGE_FILENAME,
    )


def _plot_tensor_channel_bars(
    df: Any,
    save_dir: str | Path,
    *,
    per_tensor_column: str,
    per_channel_column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> Path:
    return _plot_tail_gap_bars(
        df,
        save_dir,
        metric_columns=(per_tensor_column, per_channel_column),
        ylabel=f"{ylabel} tail_vs_many_gap",
        title=f"{title} Gap (Tail vs Many)",
        filename=filename,
        metric_labels={per_tensor_column: "per-tensor", per_channel_column: "per-channel"},
    )


def plot_adaptive_clipping_rate_comparison(df: Any, save_dir: str | Path) -> Path:
    """Plot per-sample/adaptive activation clipping pressure."""

    return _plot_tensor_channel_bars(
        df,
        save_dir,
        per_tensor_column="adaptive_clipping_rate_per_tensor",
        per_channel_column="adaptive_clipping_rate_per_channel",
        ylabel="clipping rate",
        title="Adaptive Activation Clipping Rate",
        filename=ADAPTIVE_CLIPPING_FILENAME,
    )


def plot_fixed_clipping_rate_comparison(df: Any, save_dir: str | Path) -> Path:
    """Plot calibration-fixed activation clipping pressure."""

    return _plot_tensor_channel_bars(
        df,
        save_dir,
        per_tensor_column="fixed_clipping_rate_per_tensor",
        per_channel_column="fixed_clipping_rate_per_channel",
        ylabel="clipping rate",
        title="Calibration-Fixed Activation Clipping Rate",
        filename=FIXED_CLIPPING_FILENAME,
    )


def plot_adaptive_oer_retention_comparison(df: Any, save_dir: str | Path) -> Path:
    """Plot per-sample/adaptive OER retention."""

    return _plot_tensor_channel_bars(
        df,
        save_dir,
        per_tensor_column="adaptive_oer_retention_per_tensor",
        per_channel_column="adaptive_oer_retention_per_channel",
        ylabel="OER retention",
        title="Adaptive OER Retention",
        filename=ADAPTIVE_OER_FILENAME,
    )


def plot_fixed_oer_retention_comparison(df: Any, save_dir: str | Path) -> Path:
    """Plot calibration-fixed OER retention."""

    return _plot_tensor_channel_bars(
        df,
        save_dir,
        per_tensor_column="fixed_oer_retention_per_tensor",
        per_channel_column="fixed_oer_retention_per_channel",
        ylabel="OER retention",
        title="Calibration-Fixed OER Retention",
        filename=FIXED_OER_FILENAME,
    )


def plot_clipping_rate_comparison(df: Any, save_dir: str | Path) -> Path:
    """Backward-compatible alias for adaptive clipping pressure."""

    return plot_adaptive_clipping_rate_comparison(df, save_dir)


def plot_oer_retention_comparison(df: Any, save_dir: str | Path) -> Path:
    """Backward-compatible alias for adaptive OER retention."""

    return plot_adaptive_oer_retention_comparison(df, save_dir)


__all__ = [
    "plot_activation_histograms",
    "plot_adaptive_clipping_rate_comparison",
    "plot_adaptive_oer_retention_comparison",
    "plot_clipping_rate_comparison",
    "plot_fixed_clipping_rate_comparison",
    "plot_fixed_oer_retention_comparison",
    "plot_activation_range_bar",
    "plot_oer_retention_comparison",
]