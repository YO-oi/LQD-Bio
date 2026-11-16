from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

try:
    from src.utils.group_utils import (
        ANALYSIS_GROUPS,
        GROUP_PROTOCOL_NAME,
        GROUP_PROTOCOL_VERSION,
        analysis_group_mask,
        compute_tail_and_few_gaps,
        define_metric_spec,
        metric_spec_metadata,
        read_canonical_bucket,
    )
except ImportError:  # pragma: no cover - allow package-relative debugging
    from ..utils.group_utils import (
        ANALYSIS_GROUPS,
        GROUP_PROTOCOL_NAME,
        GROUP_PROTOCOL_VERSION,
        analysis_group_mask,
        compute_tail_and_few_gaps,
        define_metric_spec,
        metric_spec_metadata,
        read_canonical_bucket,
    )
try:
    from .common_metrics import (
        DEFAULT_EPS,
        DEFAULT_FISHER_NORMALIZE_BY_DOF,
        DEFAULT_MARGIN_MIN_ABS,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
        activation_clipping_rate,
        embedding_distortion as common_embedding_distortion,
        margin_value_retention as common_geometry_margin_retention,
        metric_drop,
        scalar_oer_energy_metrics,
        weight_clipping_rate,
    )
    from .geometry_metrics import compute_group_fisher, compute_group_margin
except ImportError:  # pragma: no cover - 兼容直接运行调试。
    from src.metrics.common_metrics import (
        DEFAULT_EPS,
        DEFAULT_FISHER_NORMALIZE_BY_DOF,
        DEFAULT_MARGIN_MIN_ABS,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
        DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
        activation_clipping_rate,
        embedding_distortion as common_embedding_distortion,
        margin_value_retention as common_geometry_margin_retention,
        metric_drop,
        scalar_oer_energy_metrics,
        weight_clipping_rate,
    )
    from src.metrics.geometry_metrics import compute_group_fisher, compute_group_margin


QuantTensorKind = Literal["activation", "weight"]
DEFAULT_GROUP_NAMES = ANALYSIS_GROUPS
FISHER_NORMALIZATION = "raw_scatter_ratio"

_QUANT_MODE_GROUP_SPECS = (
    define_metric_spec(
        "embedding_distortion",
        metric_direction="lower_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "activation_quant_error",
        metric_direction="lower_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "projection_quant_error",
        metric_direction="lower_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "projection_oer_retention",
        metric_direction="higher_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "projection_oer_loss",
        metric_direction="lower_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "projection_off_oracle_energy_ratio",
        metric_direction="diagnostic",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
        metric_role="detail",
    ),
    define_metric_spec(
        "oer_retention",
        metric_direction="higher_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "oer_loss",
        metric_direction="lower_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "off_oracle_energy_ratio",
        metric_direction="diagnostic",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
        metric_role="detail",
    ),
    define_metric_spec(
        "clipping_rate",
        metric_direction="lower_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "exact_zero_rate",
        metric_direction="lower_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean",
    ),
    define_metric_spec(
        "margin_retention",
        metric_direction="higher_is_better",
        aggregation_unit="sample",
        aggregation_fn="nanmean_valid_only",
    ),
    define_metric_spec(
        "fisher_retention",
        metric_direction="higher_is_better",
        aggregation_unit="group_recompute",
        aggregation_fn="fisher_global_anchored_recompute",
    ),
)
QUANT_MODE_GROUP_METRIC_SPECS = {spec.metric_name: spec for spec in _QUANT_MODE_GROUP_SPECS}

QUANT_MODE_TARGET_DETAIL_SPECS: dict[str, dict[str, str]] = {
    "weight_quant_error": {
        "metric_name": "weight_quant_error",
        "metric_direction": "lower_is_better",
        "aggregation_unit": "target_module",
        "aggregation_fn": "module_scalar",
        "metric_role": "detail",
    },
    "weight_clipping_rate": {
        "metric_name": "weight_clipping_rate",
        "metric_direction": "lower_is_better",
        "aggregation_unit": "target_module",
        "aggregation_fn": "module_scalar",
        "metric_role": "detail",
    },
}

_PROTOCOL_COLUMNS = [
    "row_type",
    "group_protocol_version",
    "group_protocol_name",
    "main_comparison_group",
    "supplementary_extreme_group",
]
_ID_COLUMNS = ["layer_idx", "module_name", "target_name", "quant_mode", "granularity"]
_SPEC_COLUMNS = ["metric_name", "metric_direction", "aggregation_unit", "aggregation_fn", "metric_role"]
_FLAG_COLUMNS = ["comparable_across_groups", "gap_enabled"]

QUANT_MODE_METRIC_COLUMNS = [
    *_PROTOCOL_COLUMNS,
    *_ID_COLUMNS,
    *_SPEC_COLUMNS,
    *_FLAG_COLUMNS,
    "analysis_group",
    "shot_group",
    "bucket",
    "tail_included",
    "metric_value",
    "sample_count",
    "effective_class_count",
    "fisher_normalization",
    "margin_fp32",
    "margin_quant",
    "delta_margin",
    "margin_valid_count",
    "margin_total_count",
    "margin_valid_rate",
    "margin_invalid_count",
    "margin_nonpositive_count",
    "margin_near_threshold_count",
    "margin_nan_count",
    "margin_inf_count",
    "margin_invalid_reason",
    "fisher_fp32",
    "fisher_quant",
    "delta_fisher",
]

_MARGIN_COMPARISON_FIELDS = [
    "margin_valid_count",
    "margin_total_count",
    "margin_valid_rate",
    "margin_invalid_count",
    "margin_nonpositive_count",
    "margin_near_threshold_count",
    "margin_nan_count",
    "margin_inf_count",
    "margin_invalid_reason",
]
_MARGIN_COMPARISON_COLUMNS = [
    f"{group}_{field}"
    for group in ANALYSIS_GROUPS
    for field in _MARGIN_COMPARISON_FIELDS
]

QUANT_MODE_COMPARISON_COLUMNS = [
    *_PROTOCOL_COLUMNS,
    *_ID_COLUMNS,
    *_SPEC_COLUMNS,
    *_FLAG_COLUMNS,
    "many_value",
    "medium_value",
    "few_value",
    "tail_value",
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
    *_MARGIN_COMPARISON_COLUMNS,
    "fisher_normalization",
]

QUANT_MODE_TARGET_DETAIL_COLUMNS = [
    *_PROTOCOL_COLUMNS,
    *_ID_COLUMNS,
    *_SPEC_COLUMNS,
    *_FLAG_COLUMNS,
    "metric_value",
    "comparable_across_groups",
    "qparams_enabled",
    "fisher_normalization",
]


def _as_float_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _as_long_tensor(value: Any, *, device: torch.device | None = None) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach().long()
    else:
        result = torch.as_tensor(value, dtype=torch.long)
    if device is not None:
        result = result.to(device=device)
    return result


def _as_bool_mask(value: Any, *, num_samples: int, device: torch.device | None = None) -> Tensor:
    if isinstance(value, Tensor):
        mask = value.detach().bool().reshape(-1)
    else:
        mask = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
    if int(mask.numel()) != int(num_samples):
        raise ValueError(f"sample mask 长度必须等于样本数，当前 {int(mask.numel())} vs {int(num_samples)}")
    if device is not None:
        mask = mask.to(device=device)
    return mask


def _nan_scalar(device: torch.device | None = None) -> Tensor:
    return torch.tensor(float("nan"), dtype=torch.float32, device=device)


def _nanmean(values: Any) -> Tensor:
    tensor = _as_float_tensor(values).reshape(-1)
    finite = torch.isfinite(tensor)
    if not finite.any().item():
        return tensor.new_tensor(float("nan"))
    return tensor[finite].mean()


def _extract_tensor(cache: Any) -> Any:
    if isinstance(cache, Mapping):
        for key in ("activation", "feature", "features", "tensor", "value", "embedding", "weight"):
            if key in cache:
                return cache[key]
        if len(cache) == 1:
            return next(iter(cache.values()))
    return cache


def _num_samples_from_embedding(embeddings: Any) -> int:
    tensor = _as_float_tensor(embeddings)
    if tensor.ndim < 2:
        raise ValueError("embedding 必须至少包含 [N, C] 两个维度")
    return int(tensor.reshape(-1, int(tensor.shape[-1])).shape[0])


def _canonical_buckets(
    bucket: Sequence[Any] | Tensor,
    shot_group: Sequence[Any] | Tensor | None,
    *,
    num_samples: int,
) -> list[str]:
    result = read_canonical_bucket(
        bucket=bucket,
        shot_group=shot_group,
        context="ffn_quant_mode_metrics",
        allow_shot_group_fallback=False,
    )
    buckets = list(result.bucket)
    if len(buckets) != int(num_samples):
        raise ValueError(f"bucket length must equal num_samples, got {len(buckets)} vs {int(num_samples)}")
    return buckets


def _sample_mask_from_analysis_group(
    bucket: Sequence[Any],
    analysis_group: Any,
    *,
    num_samples: int,
    device: torch.device | None = None,
) -> Tensor:
    """Build the sample mask through group_utils; tail is medium + few."""

    if analysis_group is None or str(analysis_group).lower() in {"all", "eval", "overall"}:
        return torch.ones(num_samples, dtype=torch.bool, device=device)
    mask_values = list(analysis_group_mask(bucket, analysis_group))
    if len(mask_values) != int(num_samples):
        raise ValueError(f"analysis_group_mask length must equal num_samples, got {len(mask_values)} vs {int(num_samples)}")
    mask = torch.as_tensor(mask_values, dtype=torch.bool)
    return mask.to(device=device) if device is not None else mask

def _filter_first_dim(value: Any, sample_mask: Tensor | None) -> Tensor:
    tensor = _as_float_tensor(_extract_tensor(value))
    if sample_mask is None:
        return tensor
    if tensor.ndim == 0:
        raise ValueError("不能对标量应用样本 mask")
    if int(tensor.shape[0]) != int(sample_mask.numel()):
        raise ValueError(f"张量 batch 维与 sample_mask 不一致: {int(tensor.shape[0])} vs {int(sample_mask.numel())}")
    return tensor[sample_mask.to(device=tensor.device)]


def _qparam_get(qparams: Mapping[str, Any] | None, name: str, default: Any = None) -> Any:
    if qparams is None:
        return default
    return qparams[name] if name in qparams else default


def _safe_scalar_retention(numerator: Any, denominator: Any, *, eps: float = DEFAULT_EPS) -> Tensor:
    num = _as_float_tensor(numerator).reshape(-1)
    den = _as_float_tensor(denominator).to(device=num.device).reshape(-1)
    if tuple(num.shape) != tuple(den.shape):
        raise ValueError(f"retention 分子分母形状必须一致，当前 {tuple(num.shape)} vs {tuple(den.shape)}")
    valid = torch.isfinite(num) & torch.isfinite(den) & (den > float(eps))
    value = num / torch.clamp(den, min=float(eps))
    value = torch.where(valid, value, torch.full_like(value, float("nan")))
    return _nanmean(value)


def _current_sample_ids(sample_id: Sequence[Any] | Tensor | None, *, batch_size: int) -> list[str] | None:
    if sample_id is None:
        return None
    if isinstance(sample_id, Tensor):
        values = sample_id.detach().cpu().reshape(-1).tolist()
    elif isinstance(sample_id, (str, bytes)):
        raise TypeError("sample_id 必须是 batch 对齐序列，不能是单个字符串")
    else:
        values = list(sample_id)
    if len(values) != int(batch_size):
        raise ValueError(f"当前 sample_id 数量必须等于 batch size，当前 {len(values)} vs {int(batch_size)}")
    return [str(item) for item in values]


def _require_sample_ids(sample_id: Sequence[Any] | Tensor | None, *, num_samples: int, context: str) -> list[str]:
    """三组因果拆解必须显式携带 sample_id，作为跨缓存对齐唯一键。"""

    ids = _current_sample_ids(sample_id, batch_size=int(num_samples))
    if ids is None:
        raise ValueError(f"{context} 必须显式传入 sample_id，用于校验 embedding / activation / OER oracle 顺序")
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in ids:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        preview = ", ".join(list(dict.fromkeys(duplicates))[:5])
        raise ValueError(f"{context} 的 sample_id 必须唯一，发现重复: {preview}")
    return ids


def _assert_same_sample_ids(expected: Sequence[str], current: Sequence[str], *, context: str) -> None:
    if len(expected) != len(current):
        raise ValueError(f"{context} sample_id 数量不一致: expected={len(expected)}, current={len(current)}")
    mismatches = [idx for idx, (left, right) in enumerate(zip(expected, current)) if str(left) != str(right)]
    if mismatches:
        preview = ", ".join(
            f"idx={idx}: expected={expected[idx]!r}, current={current[idx]!r}" for idx in mismatches[:5]
        )
        raise ValueError(f"{context} sample_id 顺序不一致: {preview}")


def _sample_ids_from_cache(cache: Any, *, batch_size: int, context: str) -> list[str] | None:
    if not isinstance(cache, Mapping):
        return None
    for key in ("sample_id", "sample_ids"):
        if key in cache and cache[key] is not None:
            ids = _current_sample_ids(cache[key], batch_size=int(batch_size))
            if ids is None:
                raise ValueError(f"{context} 包含空 sample_id")
            return ids
    return None


def _validate_cache_first_dim_and_sample_id(cache: Any, expected_ids: Sequence[str], *, context: str) -> None:
    tensor = _as_float_tensor(_extract_tensor(cache))
    if tensor.ndim == 0:
        raise ValueError(f"{context} 必须至少包含 batch 维")
    if int(tensor.shape[0]) != len(expected_ids):
        raise ValueError(f"{context} batch 维必须等于 sample_id 数量: {int(tensor.shape[0])} vs {len(expected_ids)}")
    cache_ids = _sample_ids_from_cache(cache, batch_size=len(expected_ids), context=context)
    if cache_ids is not None:
        _assert_same_sample_ids(expected_ids, cache_ids, context=context)


def _validate_quant_mode_sample_alignment(
    *,
    fp32_embeddings: Any,
    quant_embeddings: Any,
    fp32_activation: Any | None,
    quant_activation: Any | None,
    fp32_projection: Any | None,
    quant_projection: Any | None,
    oer_oracle: Mapping[str, Any] | None,
    sample_id: Sequence[Any] | Tensor | None,
) -> list[str]:
    """强制校验 embedding、activation、OER oracle 三者的 sample_id 顺序一致。"""

    num_samples = _num_samples_from_embedding(fp32_embeddings)
    ids = _require_sample_ids(sample_id, num_samples=num_samples, context="compute_group_statistics")
    _validate_cache_first_dim_and_sample_id(fp32_embeddings, ids, context="fp32_embeddings")
    _validate_cache_first_dim_and_sample_id(quant_embeddings, ids, context="quant_embeddings")

    if (fp32_activation is None) != (quant_activation is None):
        raise ValueError("fp32_activation 与 quant_activation 必须同时提供或同时省略")
    if fp32_activation is not None and quant_activation is not None:
        _validate_cache_first_dim_and_sample_id(fp32_activation, ids, context="fp32_activation")
        _validate_cache_first_dim_and_sample_id(quant_activation, ids, context="quant_activation")

    if (fp32_projection is None) != (quant_projection is None):
        raise ValueError("fp32_projection and quant_projection must be provided together")
    if fp32_projection is not None and quant_projection is not None:
        _validate_cache_first_dim_and_sample_id(fp32_projection, ids, context="fp32_projection")
        _validate_cache_first_dim_and_sample_id(quant_projection, ids, context="quant_projection")

    if oer_oracle is not None:
        if not isinstance(oer_oracle, Mapping):
            raise TypeError("oer_oracle 必须是带 sample_id/token_idx/channel_idx/topk_indices 的 mapping")
        if "sample_id" not in oer_oracle or oer_oracle["sample_id"] is None:
            raise KeyError("oer_oracle 缺少 sample_id，无法校验 FP32 oracle 顺序")
        oracle_ids = _oracle_sample_ids(oer_oracle["sample_id"], batch_size=len(ids))
        _assert_same_sample_ids(ids, oracle_ids, context="oer_oracle")
        if fp32_activation is None or quant_activation is None:
            raise ValueError("计算 OER Retention 时必须同时提供 fp32_activation 与 quant_activation")
    return ids


def _oracle_sample_ids(oracle_sample_id: Any, *, batch_size: int) -> list[str]:
    """从 FP32 oracle metadata 里恢复每个 batch row 的 sample_id。"""

    if isinstance(oracle_sample_id, Tensor):
        tensor = oracle_sample_id.detach().cpu()
        if tensor.ndim == 1:
            values = tensor.reshape(-1).tolist()
            if len(values) != int(batch_size):
                raise ValueError(f"OER oracle sample_id 数量必须等于 batch size，当前 {len(values)} vs {int(batch_size)}")
            return [str(item) for item in values]
        rows = tensor.reshape(int(tensor.shape[0]), -1).tolist()
    elif isinstance(oracle_sample_id, (str, bytes)):
        raise TypeError("OER oracle sample_id 必须是 batch 对齐序列，不能是单个字符串")
    else:
        rows = list(oracle_sample_id)

    if len(rows) != int(batch_size):
        raise ValueError(f"OER oracle sample_id 数量必须等于 batch size，当前 {len(rows)} vs {int(batch_size)}")

    result: list[str] = []
    for row_idx, row in enumerate(rows):
        if isinstance(row, Tensor):
            row_values = row.detach().cpu().reshape(-1).tolist()
        elif isinstance(row, (str, bytes)):
            row_values = [row]
        elif isinstance(row, Sequence):
            row_values = list(row)
        else:
            row_values = [row]
        if not row_values:
            raise ValueError(f"OER oracle sample_id 第 {row_idx} 行为空")
        first = str(row_values[0])
        if any(str(item) != first for item in row_values):
            raise ValueError(f"OER oracle sample_id 第 {row_idx} 行内部不一致")
        result.append(first)
    return result


def _validate_oer_sample_id_alignment(
    oracle_indices: Mapping[str, Any],
    *,
    sample_id: Sequence[Any] | Tensor | None,
    batch_size: int,
    require_sample_id_alignment: bool,
) -> None:
    current = _current_sample_ids(sample_id, batch_size=batch_size)
    if current is None:
        if require_sample_id_alignment:
            raise ValueError("计算 OER Retention 必须传入当前 quant activation 对应的 sample_id 以校验顺序")
        return

    if "sample_id" not in oracle_indices or oracle_indices["sample_id"] is None:
        raise KeyError("OER oracle metadata 缺少 sample_id，无法校验样本顺序")
    oracle = _oracle_sample_ids(oracle_indices["sample_id"], batch_size=batch_size)
    mismatches = [idx for idx, (left, right) in enumerate(zip(current, oracle)) if left != right]
    if mismatches:
        preview = ", ".join(
            f"idx={idx}: current={current[idx]!r}, oracle={oracle[idx]!r}" for idx in mismatches[:5]
        )
        raise ValueError(f"OER sample_id 顺序与 FP32 oracle 不一致: {preview}")


def _geometry_group_selector(
    *,
    label_id: Any,
    num_samples: int,
    group_sample_mask: Any | None,
    group_label_ids: Any | None,
    device: torch.device,
) -> Tensor | Any:
    """显式区分样本级 group mask 与类别 id group。"""

    if group_sample_mask is not None and group_label_ids is not None:
        raise ValueError("group_sample_mask 与 group_label_ids 只能传入一个")
    if group_sample_mask is not None:
        return _as_bool_mask(group_sample_mask, num_samples=num_samples, device=device)
    if group_label_ids is not None:
        return group_label_ids
    return torch.ones(num_samples, dtype=torch.bool, device=device)


def _oracle_indices_from_metadata(
    oracle_indices: Mapping[str, Any],
    *,
    batch_size: int,
    num_scalars: int,
    device: torch.device,
    sample_mask: Tensor | None = None,
) -> Tensor:
    """读取 FP32 阶段缓存的固定 top-k 位置，禁止量化后重新找 top-k。"""

    required = ("sample_id", "layer_idx", "module_name", "token_idx", "channel_idx")
    missing = [key for key in required if key not in oracle_indices or oracle_indices[key] is None]
    if missing:
        raise KeyError(f"OER oracle metadata 缺少字段: {missing}")
    if "topk_indices" not in oracle_indices and "flat_index" not in oracle_indices:
        raise KeyError("OER oracle 必须包含 topk_indices 或 flat_index")

    raw = oracle_indices.get("topk_indices", oracle_indices.get("flat_index"))
    indices = _as_long_tensor(raw, device=device)
    if indices.ndim == 1:
        indices = indices.reshape(batch_size, -1)
    else:
        indices = indices.reshape(int(indices.shape[0]), -1)
    if int(indices.shape[0]) != int(batch_size):
        raise ValueError(f"OER oracle batch 维必须等于 activation batch，当前 {int(indices.shape[0])} vs {int(batch_size)}")
    if sample_mask is not None:
        indices = indices[sample_mask.to(device=device)]
    if int(indices.numel()) == 0:
        raise ValueError("OER oracle top-k 索引不能为空")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(num_scalars):
        raise ValueError("OER oracle top-k 索引越界")

    token_idx = _as_long_tensor(oracle_indices["token_idx"], device=device).reshape(batch_size, -1)
    channel_idx = _as_long_tensor(oracle_indices["channel_idx"], device=device).reshape(batch_size, -1)
    if sample_mask is not None:
        token_idx = token_idx[sample_mask.to(device=device)]
        channel_idx = channel_idx[sample_mask.to(device=device)]
    if tuple(token_idx.shape) != tuple(indices.shape) or tuple(channel_idx.shape) != tuple(indices.shape):
        raise ValueError("OER oracle 的 token_idx/channel_idx 形状必须与 topk_indices 一致")
    return indices


def compute_embedding_distortion(
    fp32_embeddings: Any,
    quant_embeddings: Any,
    *,
    sample_mask: Any | None = None,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """计算最终 embedding 的相对畸变: ||E_quant - E_fp32||_2 / ||E_fp32||_2。"""

    fp32 = _as_float_tensor(fp32_embeddings)
    quant = _as_float_tensor(quant_embeddings).to(device=fp32.device)
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"fp32_embeddings 与 quant_embeddings 形状必须一致，当前 {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    if sample_mask is not None:
        mask = _as_bool_mask(sample_mask, num_samples=int(fp32.shape[0]), device=fp32.device)
        fp32 = fp32[mask]
        quant = quant[mask]
    if int(fp32.numel()) == 0:
        return _nan_scalar(fp32.device)
    values = common_embedding_distortion(fp32, quant, dim=-1, reduction="none", eps=eps)
    return _nanmean(values)


def compute_oer_retention(
    fp32_feature_cache: Any,
    quant_feature_cache: Any,
    oracle_indices: Mapping[str, Any],
    *,
    sample_id: Sequence[Any] | Tensor,
    sample_mask: Any | None = None,
    require_sample_id_alignment: bool = True,
    eps: float = DEFAULT_EPS,
) -> dict[str, Tensor]:
    """用 FP32 阶段固定 top-k scalar oracle 计算 OER Retention。"""

    if not isinstance(oracle_indices, Mapping):
        raise TypeError("OER Retention 必须传入带 sample_id/layer_idx/module_name/token_idx/channel_idx 的 oracle metadata")
    fp32_all = _as_float_tensor(_extract_tensor(fp32_feature_cache))
    quant_all = _as_float_tensor(_extract_tensor(quant_feature_cache)).to(device=fp32_all.device)
    if tuple(fp32_all.shape) != tuple(quant_all.shape):
        raise ValueError(f"FP32/quant activation 形状必须一致，当前 {tuple(fp32_all.shape)} vs {tuple(quant_all.shape)}")
    if fp32_all.ndim < 2:
        raise ValueError("OER activation 必须至少包含 batch 和 scalar 维度")

    batch_size = int(fp32_all.shape[0])
    _validate_oer_sample_id_alignment(
        oracle_indices,
        sample_id=sample_id,
        batch_size=batch_size,
        require_sample_id_alignment=bool(require_sample_id_alignment),
    )
    mask = None if sample_mask is None else _as_bool_mask(sample_mask, num_samples=batch_size, device=fp32_all.device)
    num_scalars = int(fp32_all.reshape(batch_size, -1).shape[1])
    indices = _oracle_indices_from_metadata(
        oracle_indices,
        batch_size=batch_size,
        num_scalars=num_scalars,
        device=fp32_all.device,
        sample_mask=mask,
    )
    fp32 = fp32_all if mask is None else fp32_all[mask]
    quant = quant_all if mask is None else quant_all[mask]
    if int(fp32.shape[0]) == 0:
        nan = _nan_scalar(fp32_all.device)
        return {"oer_retention": nan, "oer_loss": nan, "off_oracle_energy_ratio": nan}
    return scalar_oer_energy_metrics(
        fp32,
        quant,
        topk_indices=indices,
        batch_dim=0,
        reduction="mean",
        eps=eps,
    )


def compute_clipping_rate(
    values: Any,
    *,
    scale: Any | None = None,
    zero_point: Any = 0.0,
    qmin: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMIN,
    qmax: int = DEFAULT_SIGNED_INT4_SYMMETRIC_QMAX,
    qparams: Mapping[str, Any] | None = None,
    tensor_kind: QuantTensorKind = "activation",
    sample_mask: Any | None = None,
    channel_axis: int = 0,
) -> Tensor:
    """计算 activation 或 weight 在真实量化 qparams 下的截断率。"""

    resolved_scale = _qparam_get(qparams, "scale", scale)
    if resolved_scale is None:
        raise ValueError("clipping_rate 必须传入真实量化使用的 scale")
    resolved_zero_point = _qparam_get(qparams, "zero_point", zero_point)
    resolved_qmin = int(_qparam_get(qparams, "qmin", qmin))
    resolved_qmax = int(_qparam_get(qparams, "qmax", qmax))

    if tensor_kind == "activation":
        return _sample_level_clipping_rate(
            values,
            scale=resolved_scale,
            zero_point=resolved_zero_point,
            qmin=resolved_qmin,
            qmax=resolved_qmax,
            sample_mask=sample_mask,
        )
    if tensor_kind == "weight":
        return weight_clipping_rate(
            _extract_tensor(values),
            scale=resolved_scale,
            zero_point=resolved_zero_point,
            qmin=resolved_qmin,
            qmax=resolved_qmax,
            channel_axis=channel_axis,
            reduction="mean",
        )
    raise ValueError(f"tensor_kind 只允许 activation / weight，当前为 {tensor_kind!r}")


def _target_name(layer_idx: int | None, module_name: str | None) -> str:
    if layer_idx is None or module_name is None:
        return ""
    return f"encoder.layers.{int(layer_idx)}.{module_name}"


def _protocol_metadata() -> dict[str, Any]:
    return {
        "group_protocol_version": int(GROUP_PROTOCOL_VERSION),
        "group_protocol_name": GROUP_PROTOCOL_NAME,
        "main_comparison_group": "tail",
        "supplementary_extreme_group": "few",
    }


def _metric_is_comparable(spec: Any) -> bool:
    return str(spec.metric_direction) in {"higher_is_better", "lower_is_better"}


def _metric_metadata(metric_name: str, *, target_detail: bool = False) -> dict[str, Any]:
    registry = QUANT_MODE_TARGET_DETAIL_SPECS if target_detail else QUANT_MODE_GROUP_METRIC_SPECS
    if metric_name not in registry:
        raise KeyError(f"MetricSpec is missing for {metric_name!r}")
    spec = registry[metric_name]
    if target_detail:
        metadata = dict(spec)
        comparable = False
    else:
        metadata = metric_spec_metadata(spec)
        comparable = _metric_is_comparable(spec)
    return {
        **metadata,
        "comparable_across_groups": bool(comparable),
        "gap_enabled": bool(comparable),
    }


def _id_metadata(
    *,
    layer_idx: int | None,
    module_name: str | None,
    quant_mode: str,
    granularity: str,
) -> dict[str, Any]:
    return {
        "layer_idx": None if layer_idx is None else int(layer_idx),
        "module_name": module_name,
        "target_name": _target_name(layer_idx, module_name),
        "quant_mode": str(quant_mode),
        "granularity": str(granularity),
    }


def _to_float(value: Any) -> float:
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if int(value.numel()) != 1:
            return float("nan")
        return float(value.reshape(()).item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_int(value: Any, *, default: int = 0) -> int:
    number = _to_float(value)
    if not math.isfinite(number):
        return int(default)
    return int(number)


def _empty_margin_fields() -> dict[str, Any]:
    return {
        "margin_fp32": float("nan"),
        "margin_quant": float("nan"),
        "delta_margin": float("nan"),
        "margin_valid_count": "",
        "margin_total_count": "",
        "margin_valid_rate": "",
        "margin_invalid_count": "",
        "margin_nonpositive_count": "",
        "margin_near_threshold_count": "",
        "margin_nan_count": "",
        "margin_inf_count": "",
        "margin_invalid_reason": "",
    }


def _empty_fisher_fields() -> dict[str, Any]:
    return {
        "fisher_fp32": float("nan"),
        "fisher_quant": float("nan"),
        "delta_fisher": float("nan"),
    }


def _effective_class_count(label_id: Any, sample_mask: Tensor) -> int:
    labels = _as_long_tensor(label_id, device=sample_mask.device).reshape(-1)
    if int(labels.numel()) != int(sample_mask.numel()):
        raise ValueError(f"label_id length must equal sample_mask length, got {int(labels.numel())} vs {int(sample_mask.numel())}")
    selected = labels[sample_mask]
    if int(selected.numel()) == 0:
        return 0
    return int(torch.unique(selected, sorted=True).numel())


def _sample_level_clipping_rate(
    values: Any,
    *,
    scale: Any,
    zero_point: Any = 0.0,
    qmin: int,
    qmax: int,
    sample_mask: Any | None = None,
) -> Tensor:
    activation = _filter_first_dim(values, sample_mask)
    if activation.ndim == 0 or int(activation.shape[0]) == 0 or int(activation.numel()) == 0:
        return _nan_scalar(activation.device)
    scale_t = _as_float_tensor(scale).to(device=activation.device)
    zero_point_t = _as_float_tensor(zero_point).to(device=activation.device)
    q = activation / torch.clamp(scale_t, min=float(DEFAULT_EPS)) + zero_point_t
    clipped = (q < float(qmin)) | (q > float(qmax))
    per_sample = clipped.reshape(int(activation.shape[0]), -1).float().mean(dim=1)
    return _nanmean(per_sample)


def compute_exact_zero_rate(values: Any, *, sample_mask: Any | None = None, eps: float = 0.0) -> Tensor:
    activation = _filter_first_dim(values, sample_mask)
    if activation.ndim == 0 or int(activation.shape[0]) == 0 or int(activation.numel()) == 0:
        return _nan_scalar(activation.device)
    per_sample = (activation.abs() <= float(eps)).reshape(int(activation.shape[0]), -1).float().mean(dim=1)
    return _nanmean(per_sample)


def compute_near_zero_rate(values: Any, *, sample_mask: Any | None = None, eps: float = 0.0) -> Tensor:
    """Deprecated Quant Mode alias: this is exact_zero_rate on fake-quant output."""

    return compute_exact_zero_rate(values, sample_mask=sample_mask, eps=eps)


def compute_margin_retention(
    fp32_embeddings: Any,
    quant_embeddings: Any,
    label_id: Any,
    *,
    group_sample_mask: Any | None = None,
    group_label_ids: Any | None = None,
    eps: float = DEFAULT_EPS,
) -> dict[str, Tensor | int | float]:
    """Compute margin retention from raw per-sample margins with explicit validity counts."""

    fp32 = _as_float_tensor(fp32_embeddings)
    group_selector = _geometry_group_selector(
        label_id=label_id,
        num_samples=int(fp32.shape[0]),
        group_sample_mask=group_sample_mask,
        group_label_ids=group_label_ids,
        device=fp32.device,
    )
    fp32_margin = _as_float_tensor(compute_group_margin(fp32, label_id, group_selector, reduction="none")).reshape(-1)
    quant_margin = _as_float_tensor(compute_group_margin(quant_embeddings, label_id, group_selector, reduction="none")).to(
        device=fp32_margin.device
    ).reshape(-1)
    if tuple(fp32_margin.shape) != tuple(quant_margin.shape):
        raise ValueError(f"margin shapes must match, got {tuple(fp32_margin.shape)} vs {tuple(quant_margin.shape)}")

    margin_retention, margin_diag = common_geometry_margin_retention(
        fp32_margin,
        quant_margin,
        reduction="mean",
        min_abs_margin=DEFAULT_MARGIN_MIN_ABS,
        return_diagnostics=True,
        eps=eps,
    )
    total_count = int(margin_diag["total_count"].detach().cpu().item())
    nan_mask = torch.isnan(fp32_margin) | torch.isnan(quant_margin)
    inf_mask = torch.isinf(fp32_margin) | torch.isinf(quant_margin)
    nonpositive = margin_diag["nonpositive_margin_mask"].to(device=fp32_margin.device)
    near_threshold = margin_diag["near_threshold_margin_mask"].to(device=fp32_margin.device)
    valid_count = int(margin_diag["valid_count"].detach().cpu().item())
    invalid_count = int(total_count - valid_count)
    valid_rate = float(valid_count / total_count) if total_count > 0 else float("nan")

    return {
        "margin_fp32": _nanmean(fp32_margin),
        "margin_quant": _nanmean(quant_margin),
        "delta_margin": metric_drop(fp32_margin, quant_margin, reduction="mean"),
        "margin_retention": margin_retention,
        "margin_valid_count": valid_count,
        "margin_total_count": total_count,
        "margin_valid_rate": valid_rate,
        "margin_invalid_count": invalid_count,
        "margin_nonpositive_count": int(nonpositive.sum().item()),
        "margin_near_threshold_count": int(near_threshold.sum().item()),
        "margin_nan_count": int(nan_mask.sum().item()),
        "margin_inf_count": int(inf_mask.sum().item()),
    }


def compute_fisher_retention(
    fp32_embeddings: Any,
    quant_embeddings: Any,
    label_id: Any,
    *,
    group_sample_mask: Any | None = None,
    group_label_ids: Any | None = None,
    normalize_by_dof: bool = False,
    eps: float = DEFAULT_EPS,
) -> dict[str, Tensor | int | str]:
    """Compute raw global-anchored group Fisher retention for the selected analysis group."""

    if bool(normalize_by_dof):
        raise ValueError("Quant Mode Fisher main output is fixed to raw_scatter_ratio; normalize_by_dof=True is disabled for this experiment")
    fp32 = _as_float_tensor(fp32_embeddings)
    group_selector = _geometry_group_selector(
        label_id=label_id,
        num_samples=int(fp32.shape[0]),
        group_sample_mask=group_sample_mask,
        group_label_ids=group_label_ids,
        device=fp32.device,
    )
    fp32_fisher, fp32_details = compute_group_fisher(
        fp32,
        label_id,
        group_selector,
        normalize_by_dof=False,
        eps=eps,
        return_details=True,
    )
    quant_fisher, quant_details = compute_group_fisher(
        quant_embeddings,
        label_id,
        group_selector,
        normalize_by_dof=False,
        eps=eps,
        return_details=True,
    )
    return {
        "fisher_fp32": fp32_fisher,
        "fisher_quant": quant_fisher,
        "delta_fisher": metric_drop(fp32_fisher.reshape(1), quant_fisher.reshape(1), reduction="mean"),
        "fisher_retention": _safe_scalar_retention(quant_fisher, fp32_fisher, eps=eps),
        "fisher_normalization": FISHER_NORMALIZATION,
        "sample_count": _to_int(fp32_details.get("sample_count", quant_details.get("sample_count", 0))),
        "effective_class_count": _to_int(
            fp32_details.get("effective_class_count", quant_details.get("effective_class_count", 0))
        ),
    }


def compute_collapse_gap(
    group_rows: Sequence[Mapping[str, Any]],
    *,
    metric_name: str = "margin_retention",
    many_label: str = "many",
    few_label: str = "few",
    higher_is_better: bool = True,
) -> Tensor:
    """Backward-compatible helper; new outputs use tail_vs_many_gap/few_vs_many_gap."""

    group_values: dict[str, Any] = {}
    for row in group_rows:
        if str(row.get("metric_name", metric_name)) not in {metric_name, ""}:
            continue
        group_value = row.get("analysis_group", row.get("shot_group", row.get("group_name")))
        if group_value is None:
            continue
        text = str(group_value).strip()
        if not text:
            continue
        value = row.get("metric_value", row.get(metric_name, float("nan")))
        group_values[text] = value
    if many_label not in group_values or few_label not in group_values:
        return _nan_scalar()
    direction = "higher_is_better" if higher_is_better else "lower_is_better"
    return torch.as_tensor(
        compute_tail_and_few_gaps(group_values, metric_direction=direction).get("few_vs_many_gap", float("nan")),
        dtype=torch.float32,
    )


def weight_quant_error(fp32_weight: Any, quant_weight: Any, *, eps: float = DEFAULT_EPS) -> Tensor:
    """Relative L2 error between the FP32 and fake-quantized target-module weight."""

    return common_embedding_distortion(
        _as_float_tensor(_extract_tensor(fp32_weight)),
        _as_float_tensor(_extract_tensor(quant_weight)),
        dim=None,
        reduction="mean",
        eps=eps,
    )


def activation_quant_error(
    fp32_activation: Any,
    quant_activation: Any,
    *,
    sample_mask: Any | None = None,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Sample-level activation relative L2 error, aggregated with finite-only nanmean."""

    fp32 = _filter_first_dim(fp32_activation, None)
    quant = _as_float_tensor(_extract_tensor(quant_activation)).to(device=fp32.device)
    if tuple(fp32.shape) != tuple(quant.shape):
        raise ValueError(f"FP32/quant activation shapes must match, got {tuple(fp32.shape)} vs {tuple(quant.shape)}")
    if sample_mask is not None:
        mask = _as_bool_mask(sample_mask, num_samples=int(fp32.shape[0]), device=fp32.device)
        fp32 = fp32[mask]
        quant = quant[mask]
    if fp32.ndim == 0 or int(fp32.shape[0]) == 0 or int(fp32.numel()) == 0:
        return _nan_scalar(fp32.device)
    fp32_flat = fp32.reshape(int(fp32.shape[0]), -1)
    quant_flat = quant.reshape(int(quant.shape[0]), -1)
    values = common_embedding_distortion(fp32_flat, quant_flat, dim=-1, reduction="none", eps=eps)
    return _nanmean(values)


def compute_weight_quant_error(fp32_weight: Any, quant_weight: Any, *, eps: float = DEFAULT_EPS) -> Tensor:
    return weight_quant_error(fp32_weight, quant_weight, eps=eps)


def compute_activation_quant_error(
    fp32_activation: Any,
    quant_activation: Any,
    *,
    sample_mask: Any | None = None,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    return activation_quant_error(fp32_activation, quant_activation, sample_mask=sample_mask, eps=eps)


def compute_projection_quant_error(
    fp32_projection: Any,
    quant_projection: Any,
    *,
    sample_mask: Any | None = None,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Projection-output relative L2 error for the target QuantLinear y tensor."""

    return activation_quant_error(fp32_projection, quant_projection, sample_mask=sample_mask, eps=eps)


def compute_projection_oer_retention(
    fp32_projection: Any,
    quant_projection: Any,
    *,
    sample_mask: Any | None = None,
    topk_ratio: float | None = None,
    eps: float = DEFAULT_EPS,
) -> dict[str, Tensor]:
    """Projection-level OER using FP32 projection top-k indices as the fixed oracle."""

    fp32_all = _as_float_tensor(_extract_tensor(fp32_projection))
    quant_all = _as_float_tensor(_extract_tensor(quant_projection)).to(device=fp32_all.device)
    if tuple(fp32_all.shape) != tuple(quant_all.shape):
        raise ValueError(
            f"FP32/quant projection shapes must match, got {tuple(fp32_all.shape)} vs {tuple(quant_all.shape)}"
        )
    if fp32_all.ndim < 2:
        raise ValueError("Projection OER tensor must include batch and scalar dimensions")

    batch_size = int(fp32_all.shape[0])
    mask = None if sample_mask is None else _as_bool_mask(sample_mask, num_samples=batch_size, device=fp32_all.device)
    fp32 = fp32_all if mask is None else fp32_all[mask]
    quant = quant_all if mask is None else quant_all[mask]
    if fp32.ndim == 0 or int(fp32.shape[0]) == 0 or int(fp32.numel()) == 0:
        nan = _nan_scalar(fp32_all.device)
        return {
            "projection_oer_retention": nan,
            "projection_oer_loss": nan,
            "projection_off_oracle_energy_ratio": nan,
        }

    values = scalar_oer_energy_metrics(
        fp32,
        quant,
        topk_ratio=topk_ratio,
        batch_dim=0,
        reduction="mean",
        eps=eps,
    )
    return {
        "projection_oer_retention": values["oer_retention"],
        "projection_oer_loss": values["oer_loss"],
        "projection_off_oracle_energy_ratio": values["off_oracle_energy_ratio"],
    }


def _build_group_metric_row(
    *,
    layer_idx: int | None,
    module_name: str | None,
    quant_mode: str,
    granularity: str,
    metric_name: str,
    metric_value: Any,
    analysis_group: str,
    sample_count: int,
    effective_class_count: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        **_protocol_metadata(),
        **_id_metadata(layer_idx=layer_idx, module_name=module_name, quant_mode=quant_mode, granularity=granularity),
        **_metric_metadata(metric_name),
        **_empty_margin_fields(),
        **_empty_fisher_fields(),
        "row_type": "aggregate",
        "analysis_group": str(analysis_group),
        "shot_group": "",
        "bucket": "",
        "tail_included": "",
        "metric_value": metric_value,
        "sample_count": int(sample_count),
        "effective_class_count": int(effective_class_count),
        "fisher_normalization": "",
    }
    if extra:
        row.update(dict(extra))
    return row


def _build_target_detail_row(
    *,
    layer_idx: int | None,
    module_name: str | None,
    quant_mode: str,
    granularity: str,
    metric_name: str,
    metric_value: Any,
    qparams_enabled: bool,
) -> dict[str, Any]:
    return {
        **_protocol_metadata(),
        **_id_metadata(layer_idx=layer_idx, module_name=module_name, quant_mode=quant_mode, granularity=granularity),
        **_metric_metadata(metric_name, target_detail=True),
        "row_type": "target_detail",
        "metric_value": metric_value,
        "comparable_across_groups": False,
        "gap_enabled": False,
        "qparams_enabled": bool(qparams_enabled),
        "fisher_normalization": "",
    }


def _build_comparison_summary_rows(metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    id_keys = ("layer_idx", "module_name", "target_name", "quant_mode", "granularity", "metric_name")
    for row in metric_rows:
        key = tuple(row.get(column, "") for column in id_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        rows_by_group: dict[str, Mapping[str, Any]] = {}
        duplicate_groups: list[str] = []
        invalid_groups: list[str] = []
        for row in rows:
            group_name = str(row.get("analysis_group", "")).strip()
            if group_name not in ANALYSIS_GROUPS:
                invalid_groups.append(group_name)
                continue
            if group_name in rows_by_group:
                duplicate_groups.append(group_name)
                continue
            rows_by_group[group_name] = row
        if invalid_groups:
            raise ValueError(f"quant-mode comparison summary has invalid analysis_group values for key={key}: {sorted(set(invalid_groups))}")
        if duplicate_groups:
            raise ValueError(f"quant-mode comparison summary has duplicate analysis_group rows for key={key}: {sorted(set(duplicate_groups))}")
        missing_groups = [group for group in ANALYSIS_GROUPS if group not in rows_by_group]
        if missing_groups:
            raise ValueError(f"quant-mode comparison summary is missing analysis_group rows for key={key}: {missing_groups}")
        first = rows[0]
        metric_name = str(first.get("metric_name"))
        spec_meta = _metric_metadata(metric_name)
        group_values = {group: _to_float(rows_by_group.get(group, {}).get("metric_value", float("nan"))) for group in ANALYSIS_GROUPS}
        if bool(spec_meta["gap_enabled"]):
            gaps = compute_tail_and_few_gaps(group_values, metric_direction=str(spec_meta["metric_direction"]))
        else:
            gaps = {"tail_vs_many_gap": float("nan"), "few_vs_many_gap": float("nan")}
        row: dict[str, Any] = {
            **_protocol_metadata(),
            "row_type": "comparison_summary",
            "layer_idx": key[0],
            "module_name": key[1],
            "target_name": key[2],
            "quant_mode": key[3],
            "granularity": key[4],
            **spec_meta,
            "many_value": group_values.get("many", float("nan")),
            "medium_value": group_values.get("medium", float("nan")),
            "few_value": group_values.get("few", float("nan")),
            "tail_value": group_values.get("tail", float("nan")),
            "tail_vs_many_gap": gaps["tail_vs_many_gap"],
            "few_vs_many_gap": gaps["few_vs_many_gap"],
            "fisher_normalization": first.get("fisher_normalization", "") if metric_name == "fisher_retention" else "",
        }
        for group in ANALYSIS_GROUPS:
            group_row = rows_by_group.get(group, {})
            row[f"{group}_sample_count"] = _to_int(group_row.get("sample_count", 0))
            row[f"{group}_effective_class_count"] = _to_int(group_row.get("effective_class_count", 0))
            for field in _MARGIN_COMPARISON_FIELDS:
                row[f"{group}_{field}"] = group_row.get(field, "") if metric_name == "margin_retention" else ""
        summary_rows.append(row)
    return summary_rows



def _per_sample_metric_vector(
    sample_metrics: Mapping[str, Any],
    metric_name: str,
    *,
    num_samples: int,
    device: torch.device | None = None,
) -> Tensor:
    value = sample_metrics.get(metric_name)
    if value is None:
        result = torch.full((int(num_samples),), float("nan"), dtype=torch.float32)
    else:
        result = _as_float_tensor(value).reshape(-1)
        if int(result.numel()) != int(num_samples):
            raise ValueError(
                f"sample metric {metric_name!r} length must equal num_samples, "
                f"got {int(result.numel())} vs {int(num_samples)}"
            )
    return result.to(device=device) if device is not None else result


def _masked_sample_metric_mean(
    sample_metrics: Mapping[str, Any],
    metric_name: str,
    *,
    sample_mask: Tensor,
    num_samples: int,
) -> Tensor:
    values = _per_sample_metric_vector(
        sample_metrics,
        metric_name,
        num_samples=num_samples,
        device=sample_mask.device,
    )
    selected = values[sample_mask]
    if int(selected.numel()) == 0:
        return _nan_scalar(sample_mask.device)
    return _nanmean(selected)


def compute_quant_mode_metric_tables_from_sample_metrics(
    *,
    fp32_embeddings: Any,
    quant_embeddings: Any,
    label_id: Any,
    bucket: Sequence[Any] | Tensor,
    quant_mode: str,
    sample_id: Sequence[Any] | Tensor,
    sample_metrics: Mapping[str, Any],
    layer_idx: int | None = None,
    module_name: str | None = None,
    granularity: str = "per-channel-weight/per-tensor-activation",
    group_names: Sequence[str] = DEFAULT_GROUP_NAMES,
    shot_group: Sequence[Any] | Tensor | None = None,
    fp32_weight: Any | None = None,
    quant_weight: Any | None = None,
    weight_qparams: Mapping[str, Any] | None = None,
    normalize_fisher_by_dof: bool = DEFAULT_FISHER_NORMALIZE_BY_DOF,
) -> dict[str, list[dict[str, Any]]]:
    """Build quant-mode tables from compact per-sample activation/projection metrics.

    This variant keeps large activation/projection tensors out of the long-lived
    experiment state. Embedding geometry is still recomputed from the saved
    embedding tensors; activation, projection, OER, clipping, and exact-zero
    metrics are aggregated from per-sample vectors.
    """

    if bool(normalize_fisher_by_dof):
        raise ValueError("normalize_fisher_by_dof=True is disabled: Quant Mode main Fisher uses raw_scatter_ratio only")
    num_samples = _num_samples_from_embedding(fp32_embeddings)
    ids = _require_sample_ids(sample_id, num_samples=num_samples, context="compute_quant_mode_metric_tables_from_sample_metrics")
    _validate_cache_first_dim_and_sample_id(fp32_embeddings, ids, context="fp32_embeddings")
    _validate_cache_first_dim_and_sample_id(quant_embeddings, ids, context="quant_embeddings")
    buckets = _canonical_buckets(bucket, shot_group, num_samples=num_samples)

    compact_metric_names = (
        "activation_quant_error",
        "oer_retention",
        "oer_loss",
        "off_oracle_energy_ratio",
        "projection_quant_error",
        "projection_oer_retention",
        "projection_oer_loss",
        "projection_off_oracle_energy_ratio",
        "clipping_rate",
        "exact_zero_rate",
    )

    metric_rows: list[dict[str, Any]] = []
    for group_name in group_names:
        sample_mask = _sample_mask_from_analysis_group(buckets, group_name, num_samples=num_samples)
        sample_count = int(sample_mask.sum().item())
        effective_classes = _effective_class_count(label_id, sample_mask)

        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="embedding_distortion",
                metric_value=compute_embedding_distortion(fp32_embeddings, quant_embeddings, sample_mask=sample_mask),
                analysis_group=str(group_name),
                sample_count=sample_count,
                effective_class_count=effective_classes,
            )
        )

        for metric_name in compact_metric_names:
            metric_rows.append(
                _build_group_metric_row(
                    layer_idx=layer_idx,
                    module_name=module_name,
                    quant_mode=quant_mode,
                    granularity=granularity,
                    metric_name=metric_name,
                    metric_value=_masked_sample_metric_mean(
                        sample_metrics,
                        metric_name,
                        sample_mask=sample_mask,
                        num_samples=num_samples,
                    ),
                    analysis_group=str(group_name),
                    sample_count=sample_count,
                    effective_class_count=effective_classes,
                )
            )

        margin_values = compute_margin_retention(
            fp32_embeddings,
            quant_embeddings,
            label_id,
            group_sample_mask=sample_mask,
        )
        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="margin_retention",
                metric_value=margin_values["margin_retention"],
                analysis_group=str(group_name),
                sample_count=sample_count,
                effective_class_count=effective_classes,
                extra={
                    "margin_fp32": margin_values["margin_fp32"],
                    "margin_quant": margin_values["margin_quant"],
                    "delta_margin": margin_values["delta_margin"],
                    "margin_valid_count": margin_values["margin_valid_count"],
                    "margin_total_count": margin_values["margin_total_count"],
                    "margin_valid_rate": margin_values["margin_valid_rate"],
                    "margin_invalid_count": margin_values["margin_invalid_count"],
                    "margin_nonpositive_count": margin_values["margin_nonpositive_count"],
                    "margin_near_threshold_count": margin_values["margin_near_threshold_count"],
                    "margin_nan_count": margin_values["margin_nan_count"],
                    "margin_inf_count": margin_values["margin_inf_count"],
                    "margin_invalid_reason": "see_count_fields",
                },
            )
        )

        fisher_values = compute_fisher_retention(
            fp32_embeddings,
            quant_embeddings,
            label_id,
            group_sample_mask=sample_mask,
            normalize_by_dof=False,
        )
        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="fisher_retention",
                metric_value=fisher_values["fisher_retention"],
                analysis_group=str(group_name),
                sample_count=_to_int(fisher_values.get("sample_count", sample_count)),
                effective_class_count=_to_int(fisher_values.get("effective_class_count", effective_classes)),
                extra={
                    "fisher_fp32": fisher_values["fisher_fp32"],
                    "fisher_quant": fisher_values["fisher_quant"],
                    "delta_fisher": fisher_values["delta_fisher"],
                    "fisher_normalization": FISHER_NORMALIZATION,
                },
            )
        )

    target_detail_rows: list[dict[str, Any]] = []
    if fp32_weight is not None and quant_weight is not None and weight_qparams is not None:
        weight_error_value = weight_quant_error(fp32_weight, quant_weight)
    else:
        weight_error_value = _nan_scalar()
    if fp32_weight is not None and weight_qparams is not None:
        weight_clipping_value = compute_clipping_rate(
            fp32_weight,
            qparams=weight_qparams,
            tensor_kind="weight",
            channel_axis=0,
        )
    else:
        weight_clipping_value = _nan_scalar()
    target_detail_rows.append(
        _build_target_detail_row(
            layer_idx=layer_idx,
            module_name=module_name,
            quant_mode=quant_mode,
            granularity=granularity,
            metric_name="weight_quant_error",
            metric_value=weight_error_value,
            qparams_enabled=weight_qparams is not None,
        )
    )
    target_detail_rows.append(
        _build_target_detail_row(
            layer_idx=layer_idx,
            module_name=module_name,
            quant_mode=quant_mode,
            granularity=granularity,
            metric_name="weight_clipping_rate",
            metric_value=weight_clipping_value,
            qparams_enabled=weight_qparams is not None,
        )
    )

    return {
        "metric_rows": metric_rows,
        "comparison_rows": _build_comparison_summary_rows(metric_rows),
        "target_detail_rows": target_detail_rows,
    }

def compute_group_statistics(
    *,
    fp32_embeddings: Any,
    quant_embeddings: Any,
    label_id: Any,
    bucket: Sequence[Any] | Tensor,
    quant_mode: str,
    sample_id: Sequence[Any] | Tensor,
    layer_idx: int | None = None,
    module_name: str | None = None,
    granularity: str = "per-channel-weight/per-tensor-activation",
    group_names: Sequence[str] = DEFAULT_GROUP_NAMES,
    shot_group: Sequence[Any] | Tensor | None = None,
    fp32_activation: Any | None = None,
    quant_activation: Any | None = None,
    fp32_projection: Any | None = None,
    quant_projection: Any | None = None,
    projection_oer_top_ratio: float | None = None,
    oer_oracle: Mapping[str, Any] | None = None,
    activation_qparams: Mapping[str, Any] | None = None,
    fp32_weight: Any | None = None,
    quant_weight: Any | None = None,
    weight_qparams: Mapping[str, Any] | None = None,
    normalize_fisher_by_dof: bool = DEFAULT_FISHER_NORMALIZE_BY_DOF,
) -> list[dict[str, Any]]:
    """Backward-compatible alias returning the v2 long group metric rows."""

    return compute_quant_mode_metric_tables(
        fp32_embeddings=fp32_embeddings,
        quant_embeddings=quant_embeddings,
        label_id=label_id,
        bucket=bucket,
        quant_mode=quant_mode,
        sample_id=sample_id,
        layer_idx=layer_idx,
        module_name=module_name,
        granularity=granularity,
        group_names=group_names,
        shot_group=shot_group,
        fp32_activation=fp32_activation,
        quant_activation=quant_activation,
        fp32_projection=fp32_projection,
        quant_projection=quant_projection,
        projection_oer_top_ratio=projection_oer_top_ratio,
        oer_oracle=oer_oracle,
        activation_qparams=activation_qparams,
        fp32_weight=fp32_weight,
        quant_weight=quant_weight,
        weight_qparams=weight_qparams,
        normalize_fisher_by_dof=False,
    )["metric_rows"]


def compute_quant_mode_metric_tables(
    *,
    fp32_embeddings: Any,
    quant_embeddings: Any,
    label_id: Any,
    bucket: Sequence[Any] | Tensor,
    quant_mode: str,
    sample_id: Sequence[Any] | Tensor,
    layer_idx: int | None = None,
    module_name: str | None = None,
    granularity: str = "per-channel-weight/per-tensor-activation",
    group_names: Sequence[str] = DEFAULT_GROUP_NAMES,
    shot_group: Sequence[Any] | Tensor | None = None,
    fp32_activation: Any | None = None,
    quant_activation: Any | None = None,
    fp32_projection: Any | None = None,
    quant_projection: Any | None = None,
    projection_oer_top_ratio: float | None = None,
    oer_oracle: Mapping[str, Any] | None = None,
    activation_qparams: Mapping[str, Any] | None = None,
    fp32_weight: Any | None = None,
    quant_weight: Any | None = None,
    weight_qparams: Mapping[str, Any] | None = None,
    normalize_fisher_by_dof: bool = DEFAULT_FISHER_NORMALIZE_BY_DOF,
) -> dict[str, list[dict[str, Any]]]:
    """Build protocol-v2 long group metrics, comparison summary, and target diagnostics."""

    if bool(normalize_fisher_by_dof):
        raise ValueError("normalize_fisher_by_dof=True is disabled: Quant Mode main Fisher uses raw_scatter_ratio only")
    num_samples = _num_samples_from_embedding(fp32_embeddings)
    sample_ids = _validate_quant_mode_sample_alignment(
        fp32_embeddings=fp32_embeddings,
        quant_embeddings=quant_embeddings,
        fp32_activation=fp32_activation,
        quant_activation=quant_activation,
        fp32_projection=fp32_projection,
        quant_projection=quant_projection,
        oer_oracle=oer_oracle,
        sample_id=sample_id,
    )
    buckets = _canonical_buckets(bucket, shot_group, num_samples=num_samples)

    metric_rows: list[dict[str, Any]] = []
    for group_name in group_names:
        sample_mask = _sample_mask_from_analysis_group(buckets, group_name, num_samples=num_samples)
        sample_count = int(sample_mask.sum().item())
        effective_classes = _effective_class_count(label_id, sample_mask)

        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="embedding_distortion",
                metric_value=compute_embedding_distortion(fp32_embeddings, quant_embeddings, sample_mask=sample_mask),
                analysis_group=str(group_name),
                sample_count=sample_count,
                effective_class_count=effective_classes,
            )
        )

        if fp32_activation is not None and quant_activation is not None:
            activation_error = activation_quant_error(fp32_activation, quant_activation, sample_mask=sample_mask)
            oer_values = (
                compute_oer_retention(
                    fp32_activation,
                    quant_activation,
                    oer_oracle,
                    sample_id=sample_ids,
                    sample_mask=sample_mask,
                )
                if oer_oracle is not None
                else {
                    "oer_retention": _nan_scalar(),
                    "oer_loss": _nan_scalar(),
                    "off_oracle_energy_ratio": _nan_scalar(),
                }
            )
        else:
            activation_error = _nan_scalar()
            oer_values = {
                "oer_retention": _nan_scalar(),
                "oer_loss": _nan_scalar(),
                "off_oracle_energy_ratio": _nan_scalar(),
            }

        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="activation_quant_error",
                metric_value=activation_error,
                analysis_group=str(group_name),
                sample_count=sample_count,
                effective_class_count=effective_classes,
            )
        )
        for oer_name in ("oer_retention", "oer_loss", "off_oracle_energy_ratio"):
            metric_rows.append(
                _build_group_metric_row(
                    layer_idx=layer_idx,
                    module_name=module_name,
                    quant_mode=quant_mode,
                    granularity=granularity,
                    metric_name=oer_name,
                    metric_value=oer_values[oer_name],
                    analysis_group=str(group_name),
                    sample_count=sample_count,
                    effective_class_count=effective_classes,
                )
            )

        if fp32_projection is not None and quant_projection is not None:
            projection_error = compute_projection_quant_error(
                fp32_projection,
                quant_projection,
                sample_mask=sample_mask,
            )
            projection_oer_values = compute_projection_oer_retention(
                fp32_projection,
                quant_projection,
                sample_mask=sample_mask,
                topk_ratio=projection_oer_top_ratio,
            )
        else:
            projection_error = _nan_scalar()
            projection_oer_values = {
                "projection_oer_retention": _nan_scalar(),
                "projection_oer_loss": _nan_scalar(),
                "projection_off_oracle_energy_ratio": _nan_scalar(),
            }
        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="projection_quant_error",
                metric_value=projection_error,
                analysis_group=str(group_name),
                sample_count=sample_count,
                effective_class_count=effective_classes,
            )
        )
        for projection_oer_name in (
            "projection_oer_retention",
            "projection_oer_loss",
            "projection_off_oracle_energy_ratio",
        ):
            metric_rows.append(
                _build_group_metric_row(
                    layer_idx=layer_idx,
                    module_name=module_name,
                    quant_mode=quant_mode,
                    granularity=granularity,
                    metric_name=projection_oer_name,
                    metric_value=projection_oer_values[projection_oer_name],
                    analysis_group=str(group_name),
                    sample_count=sample_count,
                    effective_class_count=effective_classes,
                )
            )

        if fp32_activation is not None and activation_qparams is not None:
            clipping_value = compute_clipping_rate(
                fp32_activation,
                qparams=activation_qparams,
                tensor_kind="activation",
                sample_mask=sample_mask,
            )
        else:
            clipping_value = _nan_scalar()
        if quant_activation is not None and activation_qparams is not None:
            near_zero_value = compute_exact_zero_rate(quant_activation, sample_mask=sample_mask)
        else:
            near_zero_value = _nan_scalar()
        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="clipping_rate",
                metric_value=clipping_value,
                analysis_group=str(group_name),
                sample_count=sample_count,
                effective_class_count=effective_classes,
            )
        )
        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="exact_zero_rate",
                metric_value=near_zero_value,
                analysis_group=str(group_name),
                sample_count=sample_count,
                effective_class_count=effective_classes,
            )
        )

        margin_values = compute_margin_retention(
            fp32_embeddings,
            quant_embeddings,
            label_id,
            group_sample_mask=sample_mask,
        )
        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="margin_retention",
                metric_value=margin_values["margin_retention"],
                analysis_group=str(group_name),
                sample_count=sample_count,
                effective_class_count=effective_classes,
                extra={
                    "margin_fp32": margin_values["margin_fp32"],
                    "margin_quant": margin_values["margin_quant"],
                    "delta_margin": margin_values["delta_margin"],
                    "margin_valid_count": margin_values["margin_valid_count"],
                    "margin_total_count": margin_values["margin_total_count"],
                    "margin_valid_rate": margin_values["margin_valid_rate"],
                    "margin_invalid_count": margin_values["margin_invalid_count"],
                    "margin_nonpositive_count": margin_values["margin_nonpositive_count"],
                    "margin_near_threshold_count": margin_values["margin_near_threshold_count"],
                    "margin_nan_count": margin_values["margin_nan_count"],
                    "margin_inf_count": margin_values["margin_inf_count"],
                    "margin_invalid_reason": "see_count_fields",
                },
            )
        )

        fisher_values = compute_fisher_retention(
            fp32_embeddings,
            quant_embeddings,
            label_id,
            group_sample_mask=sample_mask,
            normalize_by_dof=False,
        )
        metric_rows.append(
            _build_group_metric_row(
                layer_idx=layer_idx,
                module_name=module_name,
                quant_mode=quant_mode,
                granularity=granularity,
                metric_name="fisher_retention",
                metric_value=fisher_values["fisher_retention"],
                analysis_group=str(group_name),
                sample_count=_to_int(fisher_values.get("sample_count", sample_count)),
                effective_class_count=_to_int(fisher_values.get("effective_class_count", effective_classes)),
                extra={
                    "fisher_fp32": fisher_values["fisher_fp32"],
                    "fisher_quant": fisher_values["fisher_quant"],
                    "delta_fisher": fisher_values["delta_fisher"],
                    "fisher_normalization": FISHER_NORMALIZATION,
                },
            )
        )

    target_detail_rows: list[dict[str, Any]] = []
    if fp32_weight is not None and quant_weight is not None and weight_qparams is not None:
        weight_error_value = weight_quant_error(fp32_weight, quant_weight)
    else:
        weight_error_value = _nan_scalar()
    if fp32_weight is not None and weight_qparams is not None:
        weight_clipping_value = compute_clipping_rate(
            fp32_weight,
            qparams=weight_qparams,
            tensor_kind="weight",
            channel_axis=0,
        )
    else:
        weight_clipping_value = _nan_scalar()
    target_detail_rows.append(
        _build_target_detail_row(
            layer_idx=layer_idx,
            module_name=module_name,
            quant_mode=quant_mode,
            granularity=granularity,
            metric_name="weight_quant_error",
            metric_value=weight_error_value,
            qparams_enabled=weight_qparams is not None,
        )
    )
    target_detail_rows.append(
        _build_target_detail_row(
            layer_idx=layer_idx,
            module_name=module_name,
            quant_mode=quant_mode,
            granularity=granularity,
            metric_name="weight_clipping_rate",
            metric_value=weight_clipping_value,
            qparams_enabled=weight_qparams is not None,
        )
    )

    return {
        "metric_rows": metric_rows,
        "comparison_rows": _build_comparison_summary_rows(metric_rows),
        "target_detail_rows": target_detail_rows,
    }


def compute_quant_mode_metrics(**kwargs: Any) -> list[dict[str, Any]]:
    """Backward-compatible alias returning protocol-v2 long group metric rows."""

    return compute_group_statistics(**kwargs)


__all__ = [
    "DEFAULT_GROUP_NAMES",
    "FISHER_NORMALIZATION",
    "QUANT_MODE_COMPARISON_COLUMNS",
    "QUANT_MODE_GROUP_METRIC_SPECS",
    "QUANT_MODE_METRIC_COLUMNS",
    "QUANT_MODE_TARGET_DETAIL_COLUMNS",
    "QUANT_MODE_TARGET_DETAIL_SPECS",
    "activation_quant_error",
    "compute_activation_quant_error",
    "compute_clipping_rate",
    "compute_collapse_gap",
    "compute_embedding_distortion",
    "compute_fisher_retention",
    "compute_group_statistics",
    "compute_margin_retention",
    "compute_exact_zero_rate",
    "compute_near_zero_rate",
    "compute_oer_retention",
    "compute_projection_oer_retention",
    "compute_projection_quant_error",
    "compute_quant_mode_metric_tables",
    "compute_quant_mode_metric_tables_from_sample_metrics",
    "compute_quant_mode_metrics",
    "compute_weight_quant_error",
    "weight_quant_error",
]
