from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.utils.group_utils import ANALYSIS_GROUPS, BASE_BUCKETS, GROUP_PROTOCOL_VERSION, normalize_analysis_group, read_canonical_bucket
except ImportError:  # pragma: no cover - package-relative debugging.
    from ..utils.group_utils import ANALYSIS_GROUPS, BASE_BUCKETS, GROUP_PROTOCOL_VERSION, normalize_analysis_group, read_canonical_bucket


FISHER_RETENTION_FILENAME = "fisher_retention_by_quant_mode.png"
MARGIN_RETENTION_FILENAME = "margin_retention_many_few_gap.png"
VARIANCE_INFLATION_FILENAME = "variance_inflation_by_class_count.png"
EMBEDDING_2D_FILENAME = "embedding_projection_2d.png"
GEOMETRY_DAMAGE_HEATMAP_FILENAME = "semantic_geometry_damage_heatmap.png"

DEFAULT_QUANT_MODE_ORDER = ("FP32", "A4-only", "W4A4", "W4-only")
DEFAULT_ANALYSIS_GROUP_ORDER = tuple(ANALYSIS_GROUPS)
DEFAULT_BUCKET_ORDER = tuple(BASE_BUCKETS)
DEFAULT_PROJECTION_MODES = ("FP32", "A4-only", "W4A4")
DAMAGE_HEATMAP_METRICS = ("fisher_damage", "margin_damage", "embedding_distortion", "variance_inflation_deviation")
PROJECTION_COLUMN_PAIRS = (
    ("projection_x", "projection_y"),
    ("embedding_2d_x", "embedding_2d_y"),
    ("x", "y"),
    ("dim1", "dim2"),
)

_QUANT_MODE_ALIASES = {
    "fp32": "FP32",
    "a4": "A4-only",
    "a4-only": "A4-only",
    "a4_only": "A4-only",
    "w4a4": "W4A4",
    "w4-a4": "W4A4",
    "w4_a4": "W4A4",
    "w4": "W4-only",
    "w4-only": "W4-only",
    "w4_only": "W4-only",
}


def _load_frame(data: Any) -> Any:
    import pandas as pd

    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"metrics CSV does not exist: {path}")
        return pd.read_csv(path)
    if hasattr(data, "columns"):
        return data.copy()
    raise TypeError("plotter input must be a metrics CSV path or pandas.DataFrame")


def _require_columns(df: Any, columns: Sequence[str], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{context} missing columns: {', '.join(missing)}")


def _to_numeric(series: Any) -> Any:
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


def _save_path(save_dir: str | Path, filename: str) -> Path:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def _normalize_quant_mode(value: Any) -> str:
    text = str(value).strip()
    key = text.lower().replace(" ", "_")
    return _QUANT_MODE_ALIASES.get(key, text)


def _ordered_unique(values: Sequence[Any], preferred: Sequence[str] | None = None) -> list[str]:
    items = [str(value) for value in values if str(value).strip().lower() not in {"", "nan", "none", "<na>"}]
    unique = list(dict.fromkeys(items))
    if preferred is None:
        return unique
    preferred_set = {str(item) for item in preferred}
    ordered = [str(item) for item in preferred if str(item) in unique]
    ordered.extend([item for item in unique if item not in preferred_set])
    return ordered


def _require_protocol_v2(df: Any, *, context: str, require_analysis_group: bool) -> None:
    required = ["group_protocol_version", "metric_name", "metric_direction", "aggregation_unit", "aggregation_fn"]
    if require_analysis_group:
        required.append("analysis_group")
    _require_columns(df, required, context=context)
    version = _to_numeric(df["group_protocol_version"])
    if bool(version.isna().any()) or bool((version != GROUP_PROTOCOL_VERSION).any()):
        raise ValueError(f"{context} requires group_protocol_version == {GROUP_PROTOCOL_VERSION}")


def _nonempty_text(series: Any) -> Any:
    text = series.astype(str)
    return text.where(~text.str.strip().str.lower().isin({"", "nan", "none", "<na>"}), "")


def _position_column(work: Any) -> str:
    target = _nonempty_text(work["target"]) if "target" in work.columns else _nonempty_text(work["target_name"])
    label = target.where(target != "", "unknown_target")
    if "granularity" in work.columns:
        granularity = _nonempty_text(work["granularity"])
        label = label + granularity.map(lambda value: f" | {value}" if value else "")
    work["plot_position"] = label
    return "plot_position"


def _assert_no_duplicate_plot_points(work: Any, key_columns: Sequence[str], *, context: str) -> None:
    duplicated = work.duplicated(subset=list(key_columns), keep=False)
    if bool(duplicated.any()):
        examples = work.loc[duplicated, list(key_columns)].drop_duplicates().head(8).to_dict(orient="records")
        raise ValueError(f"{context} has duplicate plot points; summary must be unique before plotting: {examples}")


def _prepare_summary_metric(data: Any, metric_name: str) -> Any:
    import pandas as pd

    df = _load_frame(data)
    _require_protocol_v2(df, context=f"semantic geometry summary metric {metric_name}", require_analysis_group=True)
    _require_columns(df, ("target", "layer_idx", "module_name", "quant_mode", "granularity", "value"), context=metric_name)
    work = df[df["metric_name"].astype(str).eq(str(metric_name))].copy()
    if len(work) == 0:
        raise ValueError(f"{metric_name} has no rows in v2 summary")
    work["quant_mode"] = work["quant_mode"].map(_normalize_quant_mode)
    work["analysis_group"] = work["analysis_group"].map(normalize_analysis_group)
    work["value"] = _to_numeric(work["value"])
    _position_column(work)
    work = work.dropna(subset=["value"]).reset_index(drop=True)
    if len(work) == 0:
        raise ValueError(f"{metric_name} has no finite plottable values")
    _assert_no_duplicate_plot_points(work, ("plot_position", "quant_mode", "analysis_group"), context=metric_name)
    return pd.DataFrame(work)


def _prepare_comparison_metrics(data: Any, metric_names: Sequence[str]) -> Any:
    import pandas as pd

    df = _load_frame(data)
    _require_protocol_v2(df, context="semantic geometry comparison summary", require_analysis_group=False)
    _require_columns(df, ("target", "quant_mode", "granularity", "tail_vs_many_gap", "gap_enabled"), context="comparison summary")
    work = df[df["metric_name"].astype(str).isin([str(item) for item in metric_names])].copy()
    if len(work) == 0:
        raise ValueError("comparison summary has no requested metric rows")
    work = work[work["gap_enabled"].astype(str).str.lower().isin({"true", "1", "yes"})]
    work["quant_mode"] = work["quant_mode"].map(_normalize_quant_mode)
    work["tail_vs_many_gap"] = _to_numeric(work["tail_vs_many_gap"])
    _position_column(work)
    work = work.dropna(subset=["tail_vs_many_gap"]).reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("comparison summary has no finite tail_vs_many_gap values")
    _assert_no_duplicate_plot_points(work, ("plot_position", "quant_mode", "metric_name"), context="geometry damage heatmap")
    return pd.DataFrame(work)


def _bar_offsets(num_bars: int, width: float) -> np.ndarray:
    center = (num_bars - 1) / 2.0
    return (np.arange(num_bars, dtype=float) - center) * width


def _plot_group_metric_bar(data: Any, save_dir: str | Path, *, metric_name: str, filename: str, title: str, ylabel: str) -> Path:
    work = _prepare_summary_metric(data, metric_name)
    quant_modes = _ordered_unique(work["quant_mode"].tolist(), DEFAULT_QUANT_MODE_ORDER)
    groups = _ordered_unique(work["analysis_group"].tolist(), DEFAULT_ANALYSIS_GROUP_ORDER)
    positions = _ordered_unique(work["plot_position"].tolist())
    x_pos = np.arange(len(positions), dtype=float)
    width = min(0.22, 0.78 / max(1, len(groups)))
    offsets = _bar_offsets(len(groups), width)

    fig, axes = plt.subplots(len(quant_modes), 1, figsize=(max(8.5, 0.95 * len(positions)), max(3.0 * len(quant_modes), 3.5)), squeeze=False, constrained_layout=True)
    for row_idx, quant_mode in enumerate(quant_modes):
        ax = axes[row_idx][0]
        part = work[work["quant_mode"] == quant_mode]
        for group_idx, group in enumerate(groups):
            values = part[part["analysis_group"] == group].set_index("plot_position")["value"].reindex(positions).to_numpy(dtype=float)
            ax.bar(x_pos + offsets[group_idx], values, width=width, label=group)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(positions, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} | {quant_mode}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(title="analysis_group", fontsize=8)
    output = _save_path(save_dir, filename)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_fisher_retention_bar(summary_v2_long_csv: Any, save_dir: str | Path) -> Path:
    return _plot_group_metric_bar(summary_v2_long_csv, save_dir, metric_name="fisher_retention", filename=FISHER_RETENTION_FILENAME, title="Embedding-space Fisher Retention", ylabel="fisher_retention")


def plot_margin_retention_gap_bar(summary_v2_long_csv: Any, save_dir: str | Path) -> Path:
    return _plot_group_metric_bar(summary_v2_long_csv, save_dir, metric_name="margin_retention", filename=MARGIN_RETENTION_FILENAME, title="Margin Retention by Analysis Group", ylabel="margin_retention")


def plot_variance_inflation_scatter(summary_v2_long_csv: Any, save_dir: str | Path) -> Path:
    return _plot_group_metric_bar(summary_v2_long_csv, save_dir, metric_name="variance_inflation", filename=VARIANCE_INFLATION_FILENAME, title="Variance Inflation Diagnostic", ylabel="variance_inflation")


def plot_geometry_damage_heatmap(comparison_summary_csv: Any, save_dir: str | Path) -> Path:
    work = _prepare_comparison_metrics(comparison_summary_csv, DAMAGE_HEATMAP_METRICS)
    work["row_key"] = work["plot_position"].astype(str) + " | " + work["quant_mode"].astype(str)
    rows = _ordered_unique(work["row_key"].tolist())
    metrics = [metric for metric in DAMAGE_HEATMAP_METRICS if metric in set(work["metric_name"].astype(str))]
    matrix_df = work.pivot(index="row_key", columns="metric_name", values="tail_vs_many_gap").reindex(index=rows, columns=metrics)
    matrix = matrix_df.to_numpy(dtype=float)
    if not np.isfinite(matrix).any():
        raise ValueError("geometry damage heatmap has no finite v2 comparison values")
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=(max(7.8, 1.25 * len(metrics)), max(2.8, 0.42 * len(rows) + 1.8)), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title("Semantic Geometry Tail-vs-Many Damage Summary")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.3g}", ha="center", va="center", color="white", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("tail_vs_many_gap (larger is worse)")
    output = _save_path(save_dir, GEOMETRY_DAMAGE_HEATMAP_FILENAME)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def _bucket_column(df: Any) -> str:
    if "bucket" in df.columns:
        return "bucket"
    raise KeyError("projection metrics must contain canonical bucket")


def _canonical_bucket_values(df: Any, bucket_col: str) -> list[str]:
    shot_values = df["shot_group"] if "shot_group" in df.columns else None
    result = read_canonical_bucket(
        bucket=df[bucket_col],
        shot_group=shot_values,
        context="semantic geometry projection bucket",
        allow_shot_group_fallback=False,
    )
    return [str(item) for item in result.bucket]


def _projection_columns(df: Any) -> tuple[str, str]:
    for x_col, y_col in PROJECTION_COLUMN_PAIRS:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    raise KeyError("2D projection plot requires precomputed projection_x/projection_y or equivalent columns")


def plot_embedding_projection_2d(metrics_csv: Any, save_dir: str | Path) -> Path:
    df = _load_frame(metrics_csv)
    bucket_col = _bucket_column(df)
    x_col, y_col = _projection_columns(df)
    _require_columns(df, ("quant_mode", bucket_col, x_col, y_col), context="Embedding 2D projection")
    work = df[["quant_mode", bucket_col, x_col, y_col]].copy()
    if "label" in df.columns:
        work["label"] = df["label"].astype(str)
    work = work.rename(columns={bucket_col: "bucket"})
    work["quant_mode"] = work["quant_mode"].map(_normalize_quant_mode)
    work["bucket"] = _canonical_bucket_values(work, "bucket")
    work[x_col] = _to_numeric(work[x_col])
    work[y_col] = _to_numeric(work[y_col])
    work = work[work["quant_mode"].isin(DEFAULT_PROJECTION_MODES)].dropna(subset=[x_col, y_col]).reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("2D projection plot has no FP32 / A4-only / W4A4 plottable data")

    quant_modes = [mode for mode in DEFAULT_PROJECTION_MODES if mode in set(work["quant_mode"])]
    buckets = _ordered_unique(work["bucket"].tolist(), DEFAULT_BUCKET_ORDER)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(buckets))))
    color_map = {bucket: colors[idx] for idx, bucket in enumerate(buckets)}

    fig, axes = plt.subplots(1, len(quant_modes), figsize=(max(4.2 * len(quant_modes), 8.0), 4.4), squeeze=False, constrained_layout=True)
    for idx, mode in enumerate(quant_modes):
        ax = axes[0][idx]
        part = work[work["quant_mode"] == mode]
        for bucket in buckets:
            series = part[part["bucket"] == bucket]
            if len(series) == 0:
                continue
            ax.scatter(series[x_col].to_numpy(dtype=float), series[y_col].to_numpy(dtype=float), s=18, alpha=0.72, label=bucket, color=color_map[bucket])
        ax.set_title(f"{mode} | auxiliary 2D view")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.18)
        ax.legend(title="bucket", fontsize=8)
    output = _save_path(save_dir, EMBEDDING_2D_FILENAME)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_all_semantic_geometry_figures(
    *,
    summary_v2_long_csv: Any,
    comparison_summary_csv: Any,
    save_dir: str | Path,
    projection_metrics_csv: Any | None = None,
) -> dict[str, Path]:
    outputs = {
        "fisher_retention": plot_fisher_retention_bar(summary_v2_long_csv, save_dir),
        "margin_retention": plot_margin_retention_gap_bar(summary_v2_long_csv, save_dir),
        "variance_inflation": plot_variance_inflation_scatter(summary_v2_long_csv, save_dir),
        "geometry_damage_heatmap": plot_geometry_damage_heatmap(comparison_summary_csv, save_dir),
    }
    if projection_metrics_csv is not None:
        outputs["embedding_projection_2d"] = plot_embedding_projection_2d(projection_metrics_csv, save_dir)
    return outputs


__all__ = [
    "EMBEDDING_2D_FILENAME",
    "FISHER_RETENTION_FILENAME",
    "GEOMETRY_DAMAGE_HEATMAP_FILENAME",
    "MARGIN_RETENTION_FILENAME",
    "VARIANCE_INFLATION_FILENAME",
    "plot_all_semantic_geometry_figures",
    "plot_embedding_projection_2d",
    "plot_fisher_retention_bar",
    "plot_geometry_damage_heatmap",
    "plot_margin_retention_gap_bar",
    "plot_variance_inflation_scatter",
]
