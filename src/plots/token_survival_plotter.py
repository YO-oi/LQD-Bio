from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.utils.group_utils import (
        ANALYSIS_GROUPS,
        BASE_BUCKETS,
        GROUP_PROTOCOL_VERSION,
        analysis_group_mask,
        normalize_analysis_group,
        normalize_bucket,
        read_canonical_bucket,
    )
except ImportError:  # pragma: no cover
    from ..utils.group_utils import (
        ANALYSIS_GROUPS,
        BASE_BUCKETS,
        GROUP_PROTOCOL_VERSION,
        analysis_group_mask,
        normalize_analysis_group,
        normalize_bucket,
        read_canonical_bucket,
    )

TOPK_SURVIVAL_CURVE_FILENAME = "topk_survival_curve_by_layer.png"
TOKEN_DISTORTION_CURVE_FILENAME = "token_distortion_curve_by_layer.png"
TOPK_NEAR_ZERO_HEATMAP_FILENAME = "topk_near_zero_heatmap.png"
TOPK_OVERLAP_CURVE_FILENAME = "topk_overlap_curve_by_layer.png"
CASE_TOKEN_ENERGY_TRAJECTORY_FILENAME = "case_token_energy_trajectory.png"
TAIL_MANY_GAP_BAR_FILENAME = "token_survival_tail_many_gap.png"
QUANT_MODE_DELTA_BAR_FILENAME = "token_survival_quant_mode_delta.png"

DEFAULT_LAYER_ORDER = tuple(range(12))
DEFAULT_ANALYSIS_GROUP_ORDER = tuple(ANALYSIS_GROUPS)
DEFAULT_BUCKET_ORDER = tuple(BASE_BUCKETS)
DEFAULT_QUANT_MODE_ORDER = ("A4-only", "W4A4", "W4-only", "FP32")
DEFAULT_CASE_GROUPS = DEFAULT_BUCKET_ORDER
DEFAULT_CASE_ENERGY_METRIC = "token_oer_retention"

_QUANT_MODE_ALIASES = {
    "a4": "A4-only",
    "a4-only": "A4-only",
    "a4_only": "A4-only",
    "activation": "A4-only",
    "activation-only": "A4-only",
    "activation_only": "A4-only",
    "w4a4": "W4A4",
    "w4-a4": "W4A4",
    "w4_a4": "W4A4",
    "w4": "W4-only",
    "w4-only": "W4-only",
    "w4_only": "W4-only",
    "weight": "W4-only",
    "weight-only": "W4-only",
    "weight_only": "W4-only",
    "fp32": "FP32",
}


def _require_columns(df: Any, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"token survival plot df missing columns: {', '.join(missing)}")


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


def _normalize_quant_mode(value: Any) -> str:
    text = str(value).strip()
    return _QUANT_MODE_ALIASES.get(text.lower().replace(" ", "_"), text)


def _prepare_save_path(save_dir: str | Path, filename: str) -> Path:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def _require_protocol_v2(df: Any, *, context: str, require_analysis_group: bool) -> None:
    required = ["group_protocol_version", "metric_name", "metric_direction", "aggregation_unit", "aggregation_fn"]
    if require_analysis_group:
        required.append("analysis_group")
    _require_columns(df, required)
    version = _to_numeric_series(df["group_protocol_version"])
    if bool(version.isna().any()) or bool((version != GROUP_PROTOCOL_VERSION).any()):
        raise ValueError(f"{context} requires group_protocol_version == {GROUP_PROTOCOL_VERSION}")


def _ordered_unique(values: Sequence[Any], preferred: Sequence[str] | None = None) -> list[str]:
    items = [str(value) for value in values if str(value).strip().lower() not in {"", "nan", "<na>", "none"}]
    unique = list(dict.fromkeys(items))
    if preferred is None:
        return unique
    preferred_set = {str(item) for item in preferred}
    ordered = [str(item) for item in preferred if str(item) in unique]
    ordered.extend([item for item in unique if item not in preferred_set])
    return ordered


def _layers_from_frame(work: Any) -> list[int]:
    observed = sorted({int(value) for value in work["layer_idx"].dropna().tolist()})
    if observed and min(observed) >= 0 and max(observed) <= 11:
        return list(DEFAULT_LAYER_ORDER)
    return observed


def _nonempty_text(series: Any) -> Any:
    text = series.astype(str)
    return text.where(~text.str.strip().str.lower().isin({"", "nan", "<na>", "none"}), "")


def _bool_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return "true"
    if text in {"false", "0", "no", "n", "f", "", "nan", "<na>", "none"}:
        return "false"
    return text


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _formal_main_mask(work: Any) -> Any:
    import pandas as pd

    mask = work["metric_valid_for_paper"].map(_truthy) if "metric_valid_for_paper" in work.columns else pd.Series([False] * len(work), index=work.index)
    if "metric_valid" in work.columns:
        mask = mask & work["metric_valid"].map(_truthy)
    else:
        mask = mask & pd.Series([False] * len(work), index=work.index)
    if "metric_role" in work.columns:
        mask = mask & work["metric_role"].astype(str).eq("formal_main")
    else:
        mask = mask & pd.Series([False] * len(work), index=work.index)
    return mask


def _position_column(work: Any) -> str:
    parts = []
    if "target" in work.columns:
        target = _nonempty_text(work["target"])
        if (target != "").any():
            parts.append("Q:" + target)
    elif "intervention_target" in work.columns:
        target = _nonempty_text(work["intervention_target"])
        if (target != "").any():
            parts.append("Q:" + target)
    if "integer_mapping_source" in work.columns or "actual_integer_mapping_available" in work.columns:
        if "integer_mapping_source" in work.columns:
            source = _nonempty_text(work["integer_mapping_source"]).where(lambda s: s != "", "unknown")
        else:
            source = work["layer_idx"].map(lambda _item: "unknown")
        label = "map:" + source
        if "actual_integer_mapping_available" in work.columns:
            label = label + " actual:" + work["actual_integer_mapping_available"].map(_bool_label)
        parts.append(label)
    if "module_name" in work.columns:
        module = _nonempty_text(work["module_name"])
        if (module != "").any():
            parts.append(module)
    if "tensor_position" in work.columns:
        pos = _nonempty_text(work["tensor_position"])
        if (pos != "").any():
            parts.append(pos)
    if not parts:
        work["plot_position"] = "token_feature"
    else:
        value = parts[0]
        for part in parts[1:]:
            value = value + " | " + part
        work["plot_position"] = value.str.replace(r"(\s*\|\s*)+", " | ", regex=True).str.strip(" |")
    return "plot_position"


def _assert_no_duplicate_plot_points(work: Any, key_columns: Sequence[str], *, context: str) -> None:
    duplicated = work.duplicated(subset=list(key_columns), keep=False)
    if bool(duplicated.any()):
        examples = work.loc[duplicated, list(key_columns)].drop_duplicates().head(8).to_dict(orient="records")
        raise ValueError(f"{context} has duplicate plot points; aggregate in summary first: {examples}")


def _prepare_summary_metric(df: Any, metric_name: str) -> Any:
    import pandas as pd

    _require_protocol_v2(df, context=f"token survival summary metric {metric_name}", require_analysis_group=True)
    _require_columns(df, ("target", "layer_idx", "module_name", "tensor_position", "quant_mode", "value"))
    work = df[df["metric_name"].astype(str).eq(str(metric_name))].copy()
    work = work[_formal_main_mask(work)].copy()
    if len(work) == 0:
        raise ValueError(f"{metric_name} has no rows in v2 summary")
    work["layer_idx"] = _to_numeric_series(work["layer_idx"])
    work["quant_mode"] = work["quant_mode"].map(_normalize_quant_mode)
    work["analysis_group"] = work["analysis_group"].map(normalize_analysis_group)
    work["value"] = _to_numeric_series(work["value"])
    _position_column(work)
    work = work.dropna(subset=["layer_idx", "value"]).reset_index(drop=True)
    if len(work) == 0:
        raise ValueError(f"{metric_name} has no plottable finite values")
    work["layer_idx"] = work["layer_idx"].astype(int)
    _assert_no_duplicate_plot_points(work, ["layer_idx", "quant_mode", "plot_position", "analysis_group"], context=metric_name)
    return pd.DataFrame(work)


def _set_target_ticks(ax: Any, x_pos: np.ndarray, labels: Sequence[str]) -> None:
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)


def _plot_layer_curve(df: Any, save_dir: str | Path, *, metric_name: str, filename: str, title: str, ylabel: str) -> Path:
    work = _prepare_summary_metric(df, metric_name)
    quant_modes = _ordered_unique(work["quant_mode"].tolist(), DEFAULT_QUANT_MODE_ORDER)
    groups = _ordered_unique(work["analysis_group"].tolist(), DEFAULT_ANALYSIS_GROUP_ORDER)
    positions = _ordered_unique(work["plot_position"].tolist())
    layers = _layers_from_frame(work)
    fig, axes = plt.subplots(len(positions), len(quant_modes), figsize=(max(4.4 * len(quant_modes), 8.0), max(2.9 * len(positions), 3.2)), squeeze=False, constrained_layout=True)
    for row_idx, position in enumerate(positions):
        for col_idx, quant_mode in enumerate(quant_modes):
            ax = axes[row_idx][col_idx]
            part = work[(work["plot_position"] == position) & (work["quant_mode"] == quant_mode)]
            for group in groups:
                series = part[part["analysis_group"] == group]
                if len(series) == 0:
                    continue
                values = series.set_index("layer_idx")["value"].reindex(layers)
                ax.plot(layers, values.to_numpy(dtype=float), marker="o", linewidth=1.7, label=group)
            ax.set_title(f"{title} | {position} | {quant_mode}")
            ax.set_xlabel("layer_idx")
            ax.set_ylabel(ylabel)
            ax.set_xticks(layers)
            ax.grid(True, alpha=0.25)
            ax.legend(title="analysis_group", fontsize=8)
    out = _prepare_save_path(save_dir, filename)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_topk_survival_curve_by_layer(df: Any, save_dir: str | Path) -> Path:
    return _plot_layer_curve(df, save_dir, metric_name="topk_survival_rate", filename=TOPK_SURVIVAL_CURVE_FILENAME, title="Top-K Token Survival", ylabel="topk_survival_rate")


def plot_token_distortion_curve_by_layer(df: Any, save_dir: str | Path) -> Path:
    return _plot_layer_curve(df, save_dir, metric_name="token_distortion", filename=TOKEN_DISTORTION_CURVE_FILENAME, title="Token Direction Distortion", ylabel="token_distortion")


def plot_topk_overlap_curve_by_layer(df: Any, save_dir: str | Path) -> Path:
    return _plot_layer_curve(df, save_dir, metric_name="topk_overlap", filename=TOPK_OVERLAP_CURVE_FILENAME, title="Top-K Token Overlap Diagnostic", ylabel="topk_overlap")


def plot_topk_near_zero_heatmap(df: Any, save_dir: str | Path) -> Path:
    work = _prepare_summary_metric(df, "topk_token_near_zero_pm1_rate")
    quant_modes = _ordered_unique(work["quant_mode"].tolist(), DEFAULT_QUANT_MODE_ORDER)
    groups = _ordered_unique(work["analysis_group"].tolist(), DEFAULT_ANALYSIS_GROUP_ORDER)
    positions = _ordered_unique(work["plot_position"].tolist())
    layers = _layers_from_frame(work)
    values = work["value"].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError("topk_token_near_zero_pm1_rate has no finite values")
    vmin = float(np.nanmin(values)); vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    fig, axes = plt.subplots(len(groups), len(quant_modes), figsize=(max(4.2 * len(quant_modes), 8.0), max(2.8 * len(groups), 3.2)), squeeze=False, constrained_layout=True)
    image = None
    for row_idx, group in enumerate(groups):
        for col_idx, quant_mode in enumerate(quant_modes):
            ax = axes[row_idx][col_idx]
            part = work[(work["analysis_group"] == group) & (work["quant_mode"] == quant_mode)]
            _assert_no_duplicate_plot_points(part, ["plot_position", "layer_idx"], context=f"topk_near_zero heatmap {group} {quant_mode}")
            heatmap = part.pivot(index="plot_position", columns="layer_idx", values="value").reindex(index=positions, columns=layers)
            image = ax.imshow(heatmap.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"Top-K Near-Zero | {group} | {quant_mode}")
            ax.set_xlabel("layer_idx")
            ax.set_ylabel("target | mapping | module | tensor_position")
            ax.set_xticks(np.arange(len(layers)))
            ax.set_xticklabels(layers, fontsize=8)
            ax.set_yticks(np.arange(len(positions)))
            ax.set_yticklabels(positions, fontsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88).set_label("topk_token_near_zero_pm1_rate")
    out = _prepare_save_path(save_dir, TOPK_NEAR_ZERO_HEATMAP_FILENAME)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_tail_many_gap_bar(comparison_df: Any, save_dir: str | Path) -> Path:
    _require_protocol_v2(comparison_df, context="token survival comparison summary", require_analysis_group=False)
    _require_columns(comparison_df, ("layer_idx", "quant_mode", "metric_name", "tail_vs_many_gap", "gap_enabled"))
    metrics = [
        "topk_survival_rate",
        "token_survival_drop",
        "mean_norm_retention",
        "token_oer_retention",
        "token_oer_loss",
        "topk_token_near_zero_pm1_rate",
        "topk_token_strict_overflow_rate",
        "high_energy_scalar_near_zero_pm1_rate",
        "high_energy_scalar_strict_overflow_rate",
        "token_distortion",
    ]
    work = comparison_df[comparison_df["metric_name"].astype(str).isin(metrics)].copy()
    work = work[_formal_main_mask(work)].copy()
    work = work[work["gap_enabled"].astype(bool)]
    work["tail_vs_many_gap"] = _to_numeric_series(work["tail_vs_many_gap"])
    work["layer_idx"] = _to_numeric_series(work["layer_idx"])
    work["quant_mode"] = work["quant_mode"].map(_normalize_quant_mode)
    _position_column(work)
    work = work.dropna(subset=["layer_idx", "tail_vs_many_gap"]).reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("comparison summary has no plottable tail_vs_many_gap")
    work["series"] = work["metric_name"].astype(str) + " | " + work["quant_mode"].astype(str)
    series_order = _ordered_unique(work["series"].tolist())
    positions = _ordered_unique(work["plot_position"].tolist())
    x_labels = [f"L{int(layer)}" for layer in sorted(work["layer_idx"].dropna().astype(int).unique().tolist())]
    layers = [int(label[1:]) for label in x_labels]
    fig, axes = plt.subplots(len(series_order), 1, figsize=(max(9.0, 0.7 * len(layers)), max(2.6 * len(series_order), 3.2)), squeeze=False, constrained_layout=True)
    x_pos = np.arange(len(layers), dtype=float)
    width = min(0.22, 0.7 / max(1, len(positions)))
    offsets = (np.arange(len(positions), dtype=float) - (len(positions) - 1) / 2.0) * width
    for row_idx, series in enumerate(series_order):
        ax = axes[row_idx][0]
        part = work[work["series"] == series]
        for pos_idx, position in enumerate(positions):
            values = part[part["plot_position"] == position].set_index("layer_idx")["tail_vs_many_gap"].reindex(layers).to_numpy(dtype=float)
            ax.bar(x_pos + offsets[pos_idx], values, width=width, label=position)
        _set_target_ticks(ax, x_pos, x_labels)
        ax.set_ylabel("tail_vs_many_gap")
        ax.set_title(series)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    out = _prepare_save_path(save_dir, TAIL_MANY_GAP_BAR_FILENAME)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_quant_mode_delta_bar(quant_mode_comparison_df: Any, save_dir: str | Path) -> Path:
    _require_protocol_v2(quant_mode_comparison_df, context="token survival quant-mode comparison", require_analysis_group=True)
    _require_columns(quant_mode_comparison_df, ("layer_idx", "metric_name", "comparison_name", "delta_value"))
    work = quant_mode_comparison_df.copy()
    work = work[_formal_main_mask(work)].copy()
    work = work[work["metric_name"].astype(str).isin(["topk_survival_rate", "token_oer_retention", "token_oer_loss", "token_distortion"])]
    work["delta_value"] = _to_numeric_series(work["delta_value"])
    work["layer_idx"] = _to_numeric_series(work["layer_idx"])
    work["analysis_group"] = work["analysis_group"].map(normalize_analysis_group)
    _position_column(work)
    work = work.dropna(subset=["layer_idx", "delta_value"]).reset_index(drop=True)
    if len(work) == 0:
        raise ValueError("quant-mode comparison has no plottable deltas")
    work["series"] = (
        work["metric_name"].astype(str)
        + " | " + work["comparison_name"].astype(str)
        + " | " + work["analysis_group"].astype(str)
        + " | " + work["plot_position"].astype(str)
    )
    _assert_no_duplicate_plot_points(work, ["series", "layer_idx"], context="token survival quant-mode delta")
    series_order = _ordered_unique(work["series"].tolist())[:12]
    layers = sorted(work["layer_idx"].dropna().astype(int).unique().tolist())
    fig, axes = plt.subplots(len(series_order), 1, figsize=(max(9.0, 0.7 * len(layers)), max(2.5 * len(series_order), 3.2)), squeeze=False, constrained_layout=True)
    x_pos = np.arange(len(layers), dtype=float)
    for row_idx, series in enumerate(series_order):
        ax = axes[row_idx][0]
        part = work[work["series"] == series]
        values = part.set_index("layer_idx")["delta_value"].reindex(layers).to_numpy(dtype=float)
        ax.bar(x_pos, values, width=0.62)
        _set_target_ticks(ax, x_pos, [f"L{layer}" for layer in layers])
        ax.set_ylabel("delta_value")
        ax.set_title(series)
        ax.grid(True, axis="y", alpha=0.25)
    out = _prepare_save_path(save_dir, QUANT_MODE_DELTA_BAR_FILENAME)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _case_group_order(case_groups: Sequence[str]) -> list[str]:
    return [normalize_bucket(group, field_name="case_groups") for group in case_groups]


def _canonical_bucket_values(work: Any) -> list[str]:
    if "bucket" not in work.columns:
        raise KeyError("sample-level token survival plot df must contain bucket")
    shot_values = work["shot_group"] if "shot_group" in work.columns else None
    result = read_canonical_bucket(bucket=work["bucket"], shot_group=shot_values, context="token survival case plot bucket", allow_shot_group_fallback=False)
    return [str(item) for item in result.bucket]


def _select_case_sample_ids(work: Any, *, case_groups: Sequence[str], case_sample_ids: Mapping[str, Any] | None) -> dict[str, str]:
    selected: dict[str, str] = {}
    groups = _case_group_order(case_groups)
    if case_sample_ids is not None:
        for group_name, sample_id in case_sample_ids.items():
            selected[normalize_bucket(group_name, field_name="case_sample_ids")] = str(sample_id)
        return selected
    for bucket in groups:
        group_df = work[analysis_group_mask(work["bucket"], bucket)]
        if len(group_df) == 0:
            continue
        coverage = group_df.groupby("sample_id", dropna=False).agg(num_positions=("plot_position", "nunique"), num_layers=("layer_idx", "nunique"), num_points=("layer_idx", "count")).reset_index().sort_values(["num_positions", "num_layers", "num_points", "sample_id"], ascending=[False, False, False, True])
        selected[bucket] = str(coverage.iloc[0]["sample_id"])
    return selected


def plot_case_token_energy_trajectory(df: Any, save_dir: str | Path, *, energy_metric: str = DEFAULT_CASE_ENERGY_METRIC, case_groups: Sequence[str] = DEFAULT_CASE_GROUPS, case_sample_ids: Mapping[str, Any] | None = None) -> Path:
    _require_columns(df, ("sample_id", "bucket", "layer_idx", "quant_mode", "module_name", "tensor_position", energy_metric))
    optional = [column for column in ("shot_group", "intervention_target", "integer_mapping_source", "actual_integer_mapping_available") if column in df.columns]
    work = df[["sample_id", "bucket", "layer_idx", "quant_mode", "module_name", "tensor_position", energy_metric, *optional]].copy()
    work["sample_id"] = work["sample_id"].astype(str)
    work["bucket"] = _canonical_bucket_values(work)
    work["quant_mode"] = work["quant_mode"].map(_normalize_quant_mode)
    work["layer_idx"] = _to_numeric_series(work["layer_idx"])
    work[energy_metric] = _to_numeric_series(work[energy_metric])
    _position_column(work)
    work = work.dropna(subset=["layer_idx", energy_metric]).reset_index(drop=True)
    if len(work) == 0:
        raise ValueError(f"{energy_metric} has no case trajectory data")
    work["layer_idx"] = work["layer_idx"].astype(int)
    _assert_no_duplicate_plot_points(work, ["sample_id", "bucket", "quant_mode", "layer_idx", "plot_position"], context="case_token_energy_trajectory")
    selected = _select_case_sample_ids(work, case_groups=case_groups, case_sample_ids=case_sample_ids)
    if not selected:
        raise ValueError("case_token_energy_trajectory has no base bucket sample")
    layers = _layers_from_frame(work)
    quant_modes = _ordered_unique(work["quant_mode"].tolist(), DEFAULT_QUANT_MODE_ORDER)
    groups = [group for group in _case_group_order(case_groups) if group in selected]
    positions = _ordered_unique(work["plot_position"].tolist())
    fig, axes = plt.subplots(len(positions), len(groups), figsize=(max(4.5 * len(groups), 7.5), max(2.8 * len(positions), 3.8)), squeeze=False, constrained_layout=True)
    for row_idx, position in enumerate(positions):
        for col_idx, bucket in enumerate(groups):
            ax = axes[row_idx][col_idx]
            sample_id = selected[bucket]
            part = work[analysis_group_mask(work["bucket"], bucket) & (work["sample_id"] == sample_id) & (work["plot_position"] == position)]
            for quant_mode in quant_modes:
                mode_df = part[part["quant_mode"] == quant_mode]
                if len(mode_df) == 0:
                    continue
                trajectory = mode_df.set_index("layer_idx")[energy_metric].reindex(layers)
                ax.plot(layers, trajectory.to_numpy(dtype=float), marker="o", linewidth=1.8, label=quant_mode)
            ax.set_title(f"{bucket} | {sample_id} | {position}")
            ax.set_xlabel("layer_idx")
            ax.set_ylabel(energy_metric)
            ax.set_xticks(layers)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            ax.text(0.02, 0.02, "case display only", transform=ax.transAxes, fontsize=8, color="dimgray", ha="left", va="bottom")
    out = _prepare_save_path(save_dir, CASE_TOKEN_ENERGY_TRAJECTORY_FILENAME)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _is_missing_formal_plot_data_error(exc: ValueError) -> bool:
    text = str(exc)
    patterns = (
        "has no rows",
        "has no plottable finite values",
        "has no finite values",
        "has no plottable tail_vs_many_gap",
        "has no plottable deltas",
        "has no case trajectory data",
        "has no base bucket sample",
    )
    return any(pattern in text for pattern in patterns)


def plot_all_token_survival_figures(summary_df: Any, save_dir: str | Path, *, raw_sample_df: Any | None = None, comparison_df: Any | None = None, quant_mode_comparison_df: Any | None = None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, fn in (
        ("topk_survival_curve", plot_topk_survival_curve_by_layer),
        ("token_distortion_curve", plot_token_distortion_curve_by_layer),
        ("topk_near_zero_heatmap", plot_topk_near_zero_heatmap),
        ("topk_overlap_curve", plot_topk_overlap_curve_by_layer),
    ):
        try:
            paths[key] = fn(summary_df, save_dir)
        except ValueError as exc:
            if not _is_missing_formal_plot_data_error(exc):
                raise
    if comparison_df is not None:
        try:
            paths["tail_many_gap"] = plot_tail_many_gap_bar(comparison_df, save_dir)
        except ValueError as exc:
            if not _is_missing_formal_plot_data_error(exc):
                raise
    if quant_mode_comparison_df is not None and len(quant_mode_comparison_df) > 0:
        try:
            paths["quant_mode_delta"] = plot_quant_mode_delta_bar(quant_mode_comparison_df, save_dir)
        except ValueError as exc:
            if not _is_missing_formal_plot_data_error(exc):
                raise
    if raw_sample_df is not None:
        try:
            paths["case_token_energy_trajectory"] = plot_case_token_energy_trajectory(raw_sample_df, save_dir)
        except ValueError as exc:
            if not _is_missing_formal_plot_data_error(exc):
                raise
    return paths


__all__ = [
    "CASE_TOKEN_ENERGY_TRAJECTORY_FILENAME",
    "QUANT_MODE_DELTA_BAR_FILENAME",
    "TAIL_MANY_GAP_BAR_FILENAME",
    "TOPK_NEAR_ZERO_HEATMAP_FILENAME",
    "TOPK_OVERLAP_CURVE_FILENAME",
    "TOPK_SURVIVAL_CURVE_FILENAME",
    "TOKEN_DISTORTION_CURVE_FILENAME",
    "plot_all_token_survival_figures",
    "plot_case_token_energy_trajectory",
    "plot_quant_mode_delta_bar",
    "plot_tail_many_gap_bar",
    "plot_token_distortion_curve_by_layer",
    "plot_topk_near_zero_heatmap",
    "plot_topk_overlap_curve_by_layer",
    "plot_topk_survival_curve_by_layer",
]
