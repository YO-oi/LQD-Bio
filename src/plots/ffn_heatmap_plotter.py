from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.utils.group_utils import normalize_analysis_group
except ImportError:  # pragma: no cover - package-relative debugging.
    from ..utils.group_utils import normalize_analysis_group


LAYER_INDICES = list(range(12))
MODULE_NAMES = ["fc1", "fc2"]
GROUP_PROTOCOL_VERSION = 2
DETAIL_GROUP_COLUMN = "analysis_group"


def _require_columns(df: Any, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"heatmap df missing columns: {', '.join(missing)}")


def _to_numeric_series(series: Any) -> Any:
    import pandas as pd
    import torch

    def scalarize(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return float("nan")
        return value

    return pd.to_numeric(series.map(scalarize), errors="coerce")


def _require_group_protocol(df: Any, *, context: str) -> None:
    _require_columns(df, ("group_protocol_version",))
    versions = _to_numeric_series(df["group_protocol_version"])
    invalid = versions.isna() | (versions < GROUP_PROTOCOL_VERSION)
    if bool(invalid.any()):
        preview = versions[invalid].head(5).tolist()
        raise ValueError(
            f"{context} requires group_protocol_version >= {GROUP_PROTOCOL_VERSION}; "
            f"invalid values: {preview}"
        )


def _assert_no_duplicate_keys(df: Any, keys: Sequence[str], *, context: str) -> None:
    duplicated = df.duplicated(subset=list(keys), keep=False)
    if bool(duplicated.any()):
        preview = df.loc[duplicated, list(keys)].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{context} has duplicate rows for {list(keys)}: {preview}")


def _normalise_group_values(series: Any) -> Any:
    return series.map(normalize_analysis_group)


def _filter_group(df: Any, analysis_group: str) -> Any:
    _require_group_protocol(df, context=f"{analysis_group} detail heatmap")
    _require_columns(df, (DETAIL_GROUP_COLUMN,))
    work = df.copy()
    work[DETAIL_GROUP_COLUMN] = _normalise_group_values(work[DETAIL_GROUP_COLUMN])
    return work[work[DETAIL_GROUP_COLUMN] == normalize_analysis_group(analysis_group)].copy()


def _metric_matrix(df: Any, metric_column: str) -> np.ndarray:
    """Return a [fc1/fc2, layer0..11] matrix for one metric."""

    _require_columns(df, ("layer_idx", "module_name", metric_column))
    work = df[["layer_idx", "module_name", metric_column]].copy()
    work["layer_idx"] = _to_numeric_series(work["layer_idx"]).astype("Int64")
    work[metric_column] = _to_numeric_series(work[metric_column])
    work["module_name"] = work["module_name"].astype(str)
    work = work[work["layer_idx"].isin(LAYER_INDICES) & work["module_name"].isin(MODULE_NAMES)]

    _assert_no_duplicate_keys(work, ("module_name", "layer_idx"), context=f"{metric_column} heatmap")

    matrix = np.full((len(MODULE_NAMES), len(LAYER_INDICES)), np.nan, dtype=float)
    for _, row in work.iterrows():
        module_name = str(row["module_name"])
        layer_idx = int(row["layer_idx"])
        row_idx = MODULE_NAMES.index(module_name)
        col_idx = LAYER_INDICES.index(layer_idx)
        matrix[row_idx, col_idx] = float(row[metric_column])
    return matrix


def _save_heatmap(
    df: Any,
    save_dir: str | Path,
    *,
    metric_column: str,
    filename: str,
    title: str,
    colorbar_label: str,
) -> Path:
    matrix = _metric_matrix(df, metric_column)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    output_path = save_path / filename

    fig, ax = plt.subplots(figsize=(10.5, 2.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(LAYER_INDICES)))
    ax.set_xticklabels([str(layer_idx) for layer_idx in LAYER_INDICES])
    ax.set_yticks(np.arange(len(MODULE_NAMES)))
    ax.set_yticklabels(MODULE_NAMES)
    ax.set_xlabel("Layer")
    ax.set_ylabel("FFN module")
    ax.set_title(title)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.3g}", ha="center", va="center", color="white", fontsize=7)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _tail_margin_drop_frame(detail_df: Any) -> Any:
    work = _filter_group(detail_df, "tail")
    if "tail_margin_drop" not in work.columns:
        _require_columns(work, ("delta_margin",))
        work["tail_margin_drop"] = _to_numeric_series(work["delta_margin"])
    return work


def _margin_retention_gap_frame(summary_df: Any) -> Any:
    _require_group_protocol(summary_df, context="summary gap heatmap")
    _require_columns(summary_df, ("metric_name", "tail_vs_many_gap"))
    work = summary_df.copy()
    work = work[work["metric_name"].astype(str) == "margin_retention"].copy()
    if work.empty:
        raise ValueError("summary df missing metric_name='margin_retention' rows")
    return work


def plot_tail_margin_drop_heatmap(detail_df: Any, save_dir: str | Path) -> Path:
    work = _tail_margin_drop_frame(detail_df)
    return _save_heatmap(
        work,
        save_dir,
        metric_column="tail_margin_drop",
        filename="tail_margin_drop_heatmap.png",
        title="Tail Margin Drop",
        colorbar_label="tail delta_margin",
    )


def plot_tail_damage_gap_heatmap(summary_df: Any, save_dir: str | Path) -> Path:
    work = _margin_retention_gap_frame(summary_df)
    return _save_heatmap(
        work,
        save_dir,
        metric_column="tail_vs_many_gap",
        filename="tail_vs_many_damage_gap_heatmap.png",
        title="Tail vs Many Margin Retention Gap",
        colorbar_label="tail_vs_many_gap",
    )


def plot_tail_clipping_rate_heatmap(detail_df: Any, save_dir: str | Path) -> Path:
    work = _filter_group(detail_df, "tail")
    return _save_heatmap(
        work,
        save_dir,
        metric_column="clipping_rate",
        filename="tail_clipping_rate_heatmap.png",
        title="Tail Clipping Rate",
        colorbar_label="tail clipping_rate",
    )


def plot_tail_oer_loss_heatmap(detail_df: Any, save_dir: str | Path) -> Path:
    work = _filter_group(detail_df, "tail")
    return _save_heatmap(
        work,
        save_dir,
        metric_column="oer_loss",
        filename="tail_oer_loss_heatmap.png",
        title="Tail OER Loss",
        colorbar_label="tail oer_loss",
    )


def plot_all_heatmaps(detail_df: Any, summary_df: Any, save_dir: str | Path) -> dict[str, Path]:
    """Create the four FFN layer-sensitivity heatmaps from protocol-v2 tables."""

    return {
        "tail_margin_drop": plot_tail_margin_drop_heatmap(detail_df, save_dir),
        "tail_vs_many_damage_gap": plot_tail_damage_gap_heatmap(summary_df, save_dir),
        "tail_clipping_rate": plot_tail_clipping_rate_heatmap(detail_df, save_dir),
        "tail_oer_loss": plot_tail_oer_loss_heatmap(detail_df, save_dir),
    }


__all__ = [
    "plot_all_heatmaps",
    "plot_tail_clipping_rate_heatmap",
    "plot_tail_damage_gap_heatmap",
    "plot_tail_margin_drop_heatmap",
    "plot_tail_oer_loss_heatmap",
]
