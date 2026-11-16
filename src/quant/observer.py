from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor


ObserverType = Literal["activation", "weight"]
ObserverGranularity = Literal["per_tensor", "per_channel"]
ObserverPhase = Literal["range", "histogram"]
MaskKind = Literal["none", "token", "element"]
PercentileBackend = Literal["two_pass_histogram"]


def _percentile_key(percentile: float) -> str:
    text = f"{float(percentile):g}".replace(".", "_")
    return f"p{text}"


def _as_percentile_tuple(values: Sequence[float]) -> tuple[float, ...]:
    percentiles = tuple(float(item) for item in values)
    if not percentiles:
        raise ValueError("observer percentiles 不能为空")
    for percentile in percentiles:
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(f"observer percentile 必须位于 [0, 100]，当前为 {percentile}")
    return tuple(sorted(set(percentiles)))


def _to_json_value(value: Tensor | float | int | None) -> Any:
    if value is None:
        return None
    if isinstance(value, Tensor):
        value_cpu = value.detach().cpu()
        if value_cpu.ndim == 0:
            return float(value_cpu.item())
        return [float(item) for item in value_cpu.reshape(-1).tolist()]
    return value




def _qparams_to_json(qparams: Mapping[str, Any] | None) -> dict[str, Any]:
    empty = {"scale": None, "zero_point": None, "qmin": None, "qmax": None}
    if qparams is None:
        return empty
    result = dict(empty)
    for key, value in qparams.items():
        if key in {"scale", "zero_point"}:
            result[key] = _to_json_value(value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class ObserverConfig:
    """observer 的纯统计配置。

    observer 主要记录 activation / weight 的 FP32 数值分布摘要。
    compute_qparams()/freeze() 只把已有统计导出为 qparams，不做 fake quant，也不修改模型参数。
    percentile 后端固定为两遍 calibration 的有界内存直方图。
    """

    observer_type: ObserverType
    granularity: ObserverGranularity
    qmin: int = -7
    qmax: int = 7
    symmetric: bool = True
    percentiles: tuple[float, ...] = (99.0, 99.9, 99.99)
    percentile_low: float | None = None
    percentile_high: float | None = 99.9
    channel_axis: int = 0
    eps: float = 1e-8
    histogram_bins: int = 2048
    percentile_backend: PercentileBackend = "two_pass_histogram"

    def __post_init__(self) -> None:
        if self.observer_type not in {"activation", "weight"}:
            raise ValueError(f"不支持的 observer_type={self.observer_type!r}")
        if self.granularity not in {"per_tensor", "per_channel"}:
            raise ValueError(f"不支持的 granularity={self.granularity!r}")
        if int(self.qmin) >= int(self.qmax):
            raise ValueError(f"qmin 必须小于 qmax，当前 qmin={self.qmin}, qmax={self.qmax}")
        if self.percentile_backend != "two_pass_histogram":
            raise ValueError("observer percentile 后端必须是 two_pass_histogram")
        if float(self.eps) <= 0.0:
            raise ValueError(f"eps 必须为正数，当前为 {self.eps}")
        if int(self.histogram_bins) <= 1:
            raise ValueError(f"histogram_bins 必须大于 1，当前为 {self.histogram_bins}")
        object.__setattr__(self, "percentiles", _as_percentile_tuple(self.percentiles))
        if self.percentile_low is not None and not (0.0 <= float(self.percentile_low) <= 100.0):
            raise ValueError(f"percentile_low 必须位于 [0, 100]，当前为 {self.percentile_low}")
        if self.percentile_high is not None and not (0.0 <= float(self.percentile_high) <= 100.0):
            raise ValueError(f"percentile_high 必须位于 [0, 100]，当前为 {self.percentile_high}")


def observer_config_from_mapping(
    cfg: Mapping[str, Any],
    *,
    observer_type: ObserverType,
    default_granularity: ObserverGranularity,
    default_qmin: int = -7,
    default_qmax: int = 7,
    default_symmetric: bool = True,
    default_channel_axis: int = 0,
    default_eps: float = 1e-8,
    default_percentile_high: float | None = 99.9,
    default_histogram_bins: int = 2048,
) -> ObserverConfig:
    """从 config.py 风格映射构造 observer 配置。"""

    return ObserverConfig(
        observer_type=observer_type,
        granularity=str(cfg.get("GRANULARITY", default_granularity)),  # type: ignore[arg-type]
        qmin=int(cfg.get("QMIN", default_qmin)),
        qmax=int(cfg.get("QMAX", default_qmax)),
        symmetric=bool(cfg.get("SYMMETRIC", default_symmetric)),
        percentiles=tuple(cfg.get("PERCENTILES", (99.0, 99.9, 99.99))),
        percentile_low=cfg.get("PERCENTILE_LOW", None),
        percentile_high=cfg.get("PERCENTILE_HIGH", default_percentile_high),
        channel_axis=int(cfg.get("CHANNEL_AXIS", default_channel_axis)),
        eps=float(cfg.get("EPS", default_eps)),
        histogram_bins=int(cfg.get("HISTOGRAM_BINS", default_histogram_bins)),
        percentile_backend=str(cfg.get("PERCENTILE_BACKEND", "two_pass_histogram")),  # type: ignore[arg-type]
    )


def load_project_observer_configs() -> dict[str, ObserverConfig]:
    """读取项目根 config.OBSERVER；公共 observer 参数不做静默 fallback。"""

    import config as project_config

    if not hasattr(project_config, "OBSERVER"):
        raise AttributeError("config.py 必须定义 OBSERVER，供 src.quant.observer 使用")
    observer_cfg = getattr(project_config, "OBSERVER")
    if not isinstance(observer_cfg, Mapping):
        raise TypeError("config.OBSERVER 必须是 mapping")

    quant_cfg = getattr(project_config, "QUANT")
    weight_quant = dict(quant_cfg.get("WEIGHT", {}))
    activation_quant = dict(quant_cfg.get("ACTIVATION", {}))
    common_cfg = {
        "PERCENTILES": tuple(observer_cfg.get("DEFAULT_PERCENTILES", (99.0, 99.9, 99.99))),
        "PERCENTILE_BACKEND": str(observer_cfg.get("PERCENTILE_BACKEND", "two_pass_histogram")),
        "HISTOGRAM_BINS": int(observer_cfg.get("HISTOGRAM_BINS", activation_quant.get("OBSERVER_BINS", 2048))),
    }
    activation_cfg = {**common_cfg, **dict(observer_cfg.get("ACTIVATION", {}))}
    weight_cfg = {**common_cfg, **dict(observer_cfg.get("WEIGHT", {}))}
    return {
        "activation": observer_config_from_mapping(
            activation_cfg,
            observer_type="activation",
            default_granularity=str(activation_quant.get("GRANULARITY", "per_tensor")),  # type: ignore[arg-type]
            default_qmin=int(activation_quant.get("QMIN", -7)),
            default_qmax=int(activation_quant.get("QMAX", 7)),
            default_symmetric=bool(activation_quant.get("SYMMETRIC", True)),
            default_channel_axis=0,
            default_eps=float(activation_quant.get("EPS", 1e-8)),
            default_percentile_high=float(activation_quant.get("PERCENTILE", 99.9)),
            default_histogram_bins=int(activation_quant.get("OBSERVER_BINS", 2048)),
        ),
        "weight": observer_config_from_mapping(
            weight_cfg,
            observer_type="weight",
            default_granularity=str(weight_quant.get("GRANULARITY", "per_channel")),  # type: ignore[arg-type]
            default_qmin=int(weight_quant.get("QMIN", -7)),
            default_qmax=int(weight_quant.get("QMAX", 7)),
            default_symmetric=bool(weight_quant.get("SYMMETRIC", True)),
            default_channel_axis=int(weight_quant.get("CHANNEL_AXIS", 0)),
            default_eps=float(weight_quant.get("EPS", 1e-8)),
            default_percentile_high=99.9,
            default_histogram_bins=int(observer_cfg.get("HISTOGRAM_BINS", 2048)),
        ),
    }


class TensorDistributionObserver:
    """activation / weight 的通用两遍分布 observer。

    placement 由调用方负责；observer 不把输入强制搬到 CPU 或 GPU。
    第一遍 range 只统计范围和基础矩，第二遍 histogram 用固定范围累积直方图。
    """

    def __init__(
        self,
        config: ObserverConfig,
        *,
        tensor_name: str | None = None,
        module_name: str | None = None,
        layer_idx: int | None = None,
        module_type: str | None = None,
        enabled: bool = False,
    ) -> None:
        self.config = config
        self.tensor_name = tensor_name
        self.module_name = module_name
        self.layer_idx = layer_idx
        self.module_type = module_type
        self.enabled = bool(enabled)
        self.reset()

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def reset(self) -> None:
        self.phase: ObserverPhase = "range"
        self._stats_device: torch.device | None = None
        self._min: Tensor | None = None
        self._max: Tensor | None = None
        self._abs_max: Tensor | None = None
        self._sum: Tensor | None = None
        self._sum_sq: Tensor | None = None
        self._num_elements: Tensor | None = None
        self._hist: Tensor | None = None
        self._abs_hist: Tensor | None = None
        self._hist_min: Tensor | None = None
        self._hist_max: Tensor | None = None
        self._hist_abs_max: Tensor | None = None
        self._hist_num_elements: Tensor | None = None
        self._range_finalized = False
        self._histogram_finalized = False
        self._frozen_qparams: dict[str, Any] | None = None
        self.range_num_batches = 0
        self.range_num_samples = 0
        self.range_num_sample_updates_explicit = 0
        self.range_num_sample_updates_inferred = 0
        self.range_num_sample_updates_unknown = 0
        self.range_num_valid_tokens = 0
        self.range_num_padding_tokens = 0
        self.range_num_valid_elements = 0
        self.range_num_masked_elements = 0
        self.range_num_excluded_elements = 0
        self.range_num_updates_with_token_mask = 0
        self.range_num_updates_with_element_mask = 0
        self.range_num_updates_without_mask = 0
        self.range_num_updates_with_padding_mask = 0
        self.range_num_updates_without_padding_mask = 0
        self.histogram_num_batches = 0
        self.histogram_num_samples = 0
        self.histogram_num_sample_updates_explicit = 0
        self.histogram_num_sample_updates_inferred = 0
        self.histogram_num_sample_updates_unknown = 0
        self.histogram_num_valid_tokens = 0
        self.histogram_num_padding_tokens = 0
        self.histogram_num_valid_elements = 0
        self.histogram_num_masked_elements = 0
        self.histogram_num_excluded_elements = 0
        self.histogram_num_updates_with_token_mask = 0
        self.histogram_num_updates_with_element_mask = 0
        self.histogram_num_updates_without_mask = 0
        self.histogram_num_updates_with_padding_mask = 0
        self.histogram_num_updates_without_padding_mask = 0

    @property
    def has_data(self) -> bool:
        return self._num_elements is not None and bool(torch.any(self._num_elements > 0).item())

    @property
    def has_histogram(self) -> bool:
        return (
            self._hist is not None
            and self._hist_num_elements is not None
            and bool(torch.any(self._hist_num_elements > 0).item())
        )

    @property
    def num_elements(self) -> int:
        if self._num_elements is None:
            return 0
        return int(self._num_elements.sum().item())

    @property
    def histogram_num_elements(self) -> int:
        if self._hist_num_elements is None:
            return 0
        return int(self._hist_num_elements.sum().item())

    @property
    def qparams_frozen(self) -> bool:
        return self._frozen_qparams is not None

    @property
    def min_value(self) -> Tensor:
        if self._min is None:
            raise RuntimeError("observer 尚未统计 min，请先调用 update")
        return self._min

    @property
    def max_value(self) -> Tensor:
        if self._max is None:
            raise RuntimeError("observer 尚未统计 max，请先调用 update")
        return self._max

    @property
    def abs_max_value(self) -> Tensor:
        if self._abs_max is None:
            raise RuntimeError("observer 尚未统计 abs_max，请先调用 update")
        return self._abs_max

    @property
    def frozen_qparams(self) -> dict[str, Any] | None:
        return self._frozen_qparams

    def set_frozen_qparams(self, qparams: Mapping[str, Any]) -> dict[str, Any]:
        """保存由 calibration.py 计算得到的 qparams。"""

        required = {"scale", "zero_point", "qmin", "qmax"}
        missing = required.difference(qparams)
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"frozen qparams 缺少字段: {names}")
        frozen = dict(qparams)
        self.finalize()
        self._frozen_qparams = frozen
        self.disable()
        return frozen

    def start_range_pass(self, *, reset: bool = True) -> None:
        """开始第一遍 calibration：只统计范围和基础矩。"""

        if reset:
            self.reset()
        else:
            self.phase = "range"
            self._range_finalized = False

    def start_histogram_pass(self) -> None:
        """开始第二遍 calibration：用第一遍范围初始化固定直方图。"""

        if not self.has_data:
            raise RuntimeError("开始 histogram pass 前必须先完成 range pass")
        self._range_finalized = True
        self.phase = "histogram"
        self._init_histograms()

    def finalize(self) -> None:
        if self.phase == "range":
            self._range_finalized = True
        else:
            self._histogram_finalized = True

    def update(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
        num_samples: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self.qparams_frozen:
            raise RuntimeError("observer 已 freeze；如需重新统计请 reset")
        if self.phase == "range" and self._range_finalized:
            raise RuntimeError("range pass 已 finalized；如需重新统计请 reset")
        if self.phase == "histogram" and self._histogram_finalized:
            raise RuntimeError("histogram pass 已 finalized；如需重新统计请 reset")
        if not torch.is_floating_point(x):
            raise TypeError(f"observer 需要浮点 Tensor，当前 dtype={x.dtype}")

        resolved_num_samples, num_samples_source = self._resolve_num_samples(
            x,
            valid_mask=valid_mask,
            padding_mask=padding_mask,
            num_samples=num_samples,
        )
        values, valid_tokens, padding_tokens, valid_elements, masked_elements, mask_kind = self._select_values(
            x,
            valid_mask=valid_mask,
            padding_mask=padding_mask,
        )
        if values.numel() == 0:
            return

        values = values.detach().float()
        if not torch.isfinite(values).all().item():
            raise ValueError(
                f"{self.config.observer_type} observer 收到 NaN 或 Inf："
                f"tensor_name={self.tensor_name!r}, module_name={self.module_name!r}"
            )
        prepared = self._prepare_values_for_granularity(values)
        self._check_or_set_device(prepared.device)

        if self.phase == "range":
            self._update_streaming_stats(prepared)
            self._update_range_coverage(
                valid_tokens,
                padding_tokens,
                valid_elements,
                masked_elements,
                mask_kind,
                resolved_num_samples,
                num_samples_source,
            )
        else:
            self._update_histograms(prepared)
            self._update_histogram_coverage(
                valid_tokens,
                padding_tokens,
                valid_elements,
                masked_elements,
                mask_kind,
                resolved_num_samples,
                num_samples_source,
            )

    def _resolve_num_samples(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None,
        padding_mask: Tensor | None,
        num_samples: int | None,
    ) -> tuple[int, str]:
        """解析样本覆盖量。

        外部显式传入 num_samples 时优先使用；否则按可见结构兜底推断，
        并在 summary 里记录来源，避免覆盖量静默变成假的 0。
        """

        if num_samples is not None:
            resolved = int(num_samples)
            if resolved < 0:
                raise ValueError(f"num_samples 不能为负数，当前为 {resolved}")
            return resolved, "explicit"
        if padding_mask is not None and padding_mask.ndim >= 1:
            return int(padding_mask.shape[0]), "inferred_from_padding_mask_dim0"
        if valid_mask is not None and valid_mask.ndim >= 1:
            return int(valid_mask.shape[0]), "inferred_from_valid_mask_dim0"
        if x.ndim >= 1:
            return int(x.shape[0]), "inferred_from_x_dim0"
        return 0, "unknown"
    def _select_values(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None,
        padding_mask: Tensor | None,
    ) -> tuple[Tensor, int | None, int | None, int, int, MaskKind]:
        if valid_mask is not None and padding_mask is not None:
            raise ValueError("valid_mask 和 padding_mask 只能传入一个")
        if padding_mask is not None:
            # 项目统一语义：padding_mask == True 表示 padding，需要反转为 valid_mask。
            valid_mask = ~padding_mask.detach().to(device=x.device, dtype=torch.bool)
        elif valid_mask is not None:
            valid_mask = valid_mask.detach().to(device=x.device, dtype=torch.bool)

        if valid_mask is None:
            return x.detach(), None, None, int(x.numel()), 0, "none"
        if tuple(valid_mask.shape) == tuple(x.shape[:-1]):
            valid_tokens = int(valid_mask.sum().item())
            padding_tokens = int(valid_mask.numel() - valid_tokens)
            feature_dim = int(x.shape[-1])
            valid_elements = int(valid_tokens * feature_dim)
            masked_elements = int(padding_tokens * feature_dim)
            return x.detach()[valid_mask], valid_tokens, padding_tokens, valid_elements, masked_elements, "token"
        if tuple(valid_mask.shape) == tuple(x.shape):
            valid_elements = int(valid_mask.sum().item())
            masked_elements = int(valid_mask.numel() - valid_elements)
            return x.detach()[valid_mask], None, None, valid_elements, masked_elements, "element"
        raise ValueError(
            "observer mask shape 必须匹配 token shape 或完整 tensor shape，"
            f"当前 mask={tuple(valid_mask.shape)}，x={tuple(x.shape)}"
        )
    def _prepare_values_for_granularity(self, values: Tensor) -> Tensor:
        if self.config.granularity == "per_tensor":
            return values.reshape(1, -1)
        axis = int(self.config.channel_axis)
        if values.ndim == 0:
            raise ValueError("per-channel observer 需要 rank >= 1 的 tensor")
        if axis < 0:
            axis += values.ndim
        if axis < 0 or axis >= values.ndim:
            raise ValueError(f"channel_axis={self.config.channel_axis} 对 shape={tuple(values.shape)} 非法")
        moved = values.movedim(axis, 0).contiguous()
        return moved.reshape(int(moved.shape[0]), -1)

    def _check_or_set_device(self, device: torch.device) -> None:
        if self._stats_device is None:
            self._stats_device = device
            return
        if self._stats_device != device:
            raise RuntimeError(
                "同一个 observer 的两遍 calibration 必须由外部放在同一 device 上运行；"
                f"range device={self._stats_device}, 当前 device={device}"
            )

    def _update_streaming_stats(self, prepared: Tensor) -> None:
        current_min = prepared.amin(dim=1)
        current_max = prepared.amax(dim=1)
        current_abs_max = prepared.abs().amax(dim=1)
        current_sum = prepared.sum(dim=1, dtype=torch.float64)
        current_sum_sq = prepared.square().sum(dim=1, dtype=torch.float64)
        current_num = torch.full((prepared.shape[0],), int(prepared.shape[1]), device=prepared.device, dtype=torch.float64)
        if self._num_elements is None:
            self._min = current_min
            self._max = current_max
            self._abs_max = current_abs_max
            self._sum = current_sum
            self._sum_sq = current_sum_sq
            self._num_elements = current_num
            return
        if int(self._num_elements.numel()) != int(current_num.numel()):
            raise ValueError(
                "per-channel observer 的 channel 数在不同 update 间发生变化："
                f"old={int(self._num_elements.numel())}, new={int(current_num.numel())}"
            )
        self._min = torch.minimum(self._min, current_min)  # type: ignore[arg-type]
        self._max = torch.maximum(self._max, current_max)  # type: ignore[arg-type]
        self._abs_max = torch.maximum(self._abs_max, current_abs_max)  # type: ignore[arg-type]
        self._sum = self._sum + current_sum  # type: ignore[operator]
        self._sum_sq = self._sum_sq + current_sum_sq  # type: ignore[operator]
        self._num_elements = self._num_elements + current_num

    def _init_histograms(self) -> None:
        if self._min is None or self._max is None or self._abs_max is None:
            raise RuntimeError("初始化 histogram 前缺少 range pass 统计")
        bins = int(self.config.histogram_bins)
        channels = int(self._min.numel())
        device = self._min.device
        self._hist_min = self._min.detach().clone()
        self._hist_max = torch.maximum(self._max.detach().clone(), self._hist_min + float(self.config.eps))
        self._hist_abs_max = torch.clamp(self._abs_max.detach().clone(), min=float(self.config.eps))
        self._hist = torch.zeros((channels, bins), device=device, dtype=torch.float64)
        self._abs_hist = torch.zeros((channels, bins), device=device, dtype=torch.float64)
        self._hist_num_elements = torch.zeros((channels,), device=device, dtype=torch.float64)
        self._histogram_finalized = False

    def _update_histograms(self, prepared: Tensor) -> None:
        if self._hist is None or self._abs_hist is None or self._hist_min is None or self._hist_max is None:
            raise RuntimeError("histogram pass 尚未初始化，请先调用 start_histogram_pass()")
        if self._hist_abs_max is None or self._hist_num_elements is None:
            raise RuntimeError("histogram 状态不完整")
        if int(prepared.shape[0]) != int(self._hist.shape[0]):
            raise ValueError(
                "histogram pass 的 channel 数与 range pass 不一致："
                f"range={int(self._hist.shape[0])}, current={int(prepared.shape[0])}"
            )
        bins = int(self.config.histogram_bins)
        signed_indices = self._bucket_indices(prepared, lower=self._hist_min, upper=self._hist_max, bins=bins)
        abs_indices = self._bucket_indices(prepared.abs(), lower=torch.zeros_like(self._hist_abs_max), upper=self._hist_abs_max, bins=bins)
        signed_counts = torch.zeros_like(self._hist)
        abs_counts = torch.zeros_like(self._abs_hist)
        ones = torch.ones_like(signed_indices, dtype=torch.float64)
        signed_counts.scatter_add_(1, signed_indices, ones)
        abs_counts.scatter_add_(1, abs_indices, ones)
        self._hist = self._hist + signed_counts
        self._abs_hist = self._abs_hist + abs_counts
        self._hist_num_elements = self._hist_num_elements + torch.full(
            (prepared.shape[0],), int(prepared.shape[1]), device=prepared.device, dtype=torch.float64
        )

    def _bucket_indices(self, values: Tensor, *, lower: Tensor, upper: Tensor, bins: int) -> Tensor:
        width = torch.clamp((upper - lower) / float(bins), min=float(self.config.eps))
        indices = torch.floor((values - lower[:, None]) / width[:, None]).to(dtype=torch.long)
        return torch.clamp(indices, min=0, max=int(bins) - 1)

    def _record_sample_source(self, *, phase: ObserverPhase, source: str) -> None:
        if phase == "range":
            if source == "explicit":
                self.range_num_sample_updates_explicit += 1
            elif source == "unknown":
                self.range_num_sample_updates_unknown += 1
            else:
                self.range_num_sample_updates_inferred += 1
            return
        if source == "explicit":
            self.histogram_num_sample_updates_explicit += 1
        elif source == "unknown":
            self.histogram_num_sample_updates_unknown += 1
        else:
            self.histogram_num_sample_updates_inferred += 1

    def _record_mask_kind(self, *, phase: ObserverPhase, mask_kind: MaskKind) -> None:
        if phase == "range":
            if mask_kind == "token":
                self.range_num_updates_with_token_mask += 1
            elif mask_kind == "element":
                self.range_num_updates_with_element_mask += 1
            else:
                self.range_num_updates_without_mask += 1
            return
        if mask_kind == "token":
            self.histogram_num_updates_with_token_mask += 1
        elif mask_kind == "element":
            self.histogram_num_updates_with_element_mask += 1
        else:
            self.histogram_num_updates_without_mask += 1

    def _update_range_coverage(
        self,
        valid_tokens: int | None,
        padding_tokens: int | None,
        valid_elements: int,
        masked_elements: int,
        mask_kind: MaskKind,
        num_samples: int,
        num_samples_source: str,
    ) -> None:
        self.range_num_batches += 1
        self.range_num_samples += int(num_samples)
        self._record_sample_source(phase="range", source=num_samples_source)
        self._record_mask_kind(phase="range", mask_kind=mask_kind)
        self.range_num_valid_tokens += int(valid_tokens or 0)
        self.range_num_padding_tokens += int(padding_tokens or 0)
        self.range_num_valid_elements += int(valid_elements)
        self.range_num_masked_elements += int(masked_elements)
        self.range_num_excluded_elements += int(masked_elements)
        self.range_num_updates_with_padding_mask += int(mask_kind != "none")
        self.range_num_updates_without_padding_mask += int(mask_kind == "none")

    def _update_histogram_coverage(
        self,
        valid_tokens: int | None,
        padding_tokens: int | None,
        valid_elements: int,
        masked_elements: int,
        mask_kind: MaskKind,
        num_samples: int,
        num_samples_source: str,
    ) -> None:
        self.histogram_num_batches += 1
        self.histogram_num_samples += int(num_samples)
        self._record_sample_source(phase="histogram", source=num_samples_source)
        self._record_mask_kind(phase="histogram", mask_kind=mask_kind)
        self.histogram_num_valid_tokens += int(valid_tokens or 0)
        self.histogram_num_padding_tokens += int(padding_tokens or 0)
        self.histogram_num_valid_elements += int(valid_elements)
        self.histogram_num_masked_elements += int(masked_elements)
        self.histogram_num_excluded_elements += int(masked_elements)
        self.histogram_num_updates_with_padding_mask += int(mask_kind != "none")
        self.histogram_num_updates_without_padding_mask += int(mask_kind == "none")
    def percentile_value(self, percentile: float, *, absolute: bool = False) -> Tensor:
        if not self.has_histogram:
            raise RuntimeError("percentile 需要第二遍 histogram pass 统计，当前没有 histogram 数据")
        percentile = float(percentile)
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(f"percentile 必须位于 [0, 100]，当前为 {percentile}")
        hist = self._abs_hist if absolute else self._hist
        lower = torch.zeros_like(self._hist_abs_max) if absolute else self._hist_min
        upper = self._hist_abs_max if absolute else self._hist_max
        if hist is None or lower is None or upper is None:
            raise RuntimeError("histogram 状态不完整")
        return self._percentile_from_histogram(hist, lower=lower, upper=upper, percentile=percentile)

    def _percentile_from_histogram(self, hist: Tensor, *, lower: Tensor, upper: Tensor, percentile: float) -> Tensor:
        total = torch.clamp(hist.sum(dim=1), min=1.0)
        rank = torch.clamp(torch.ceil(total * (float(percentile) / 100.0)), min=1.0)
        cdf = torch.cumsum(hist, dim=1)
        indices = torch.argmax((cdf >= rank[:, None]).to(dtype=torch.int64), dim=1)
        bins = int(hist.shape[1])
        width = torch.clamp((upper - lower) / float(bins), min=float(self.config.eps))
        return lower + (indices.to(dtype=width.dtype) + 0.5) * width

    def _mean(self) -> Tensor | None:
        if self._sum is None or self._num_elements is None:
            return None
        return self._sum / torch.clamp(self._num_elements, min=1.0)

    def _std(self) -> Tensor | None:
        """总体标准差，ddof=0。"""

        if self._sum is None or self._sum_sq is None or self._num_elements is None:
            return None
        count = torch.clamp(self._num_elements, min=1.0)
        mean = self._sum / count
        variance = torch.clamp(self._sum_sq / count - mean.square(), min=0.0)
        return torch.sqrt(variance)


    def summary(self) -> dict[str, Any]:
        percentiles: dict[str, Any] = {}
        abs_percentiles: dict[str, Any] = {}
        if self.has_histogram:
            for percentile in self.config.percentiles:
                key = _percentile_key(percentile)
                percentiles[key] = _to_json_value(self.percentile_value(percentile))
                abs_percentiles[key] = _to_json_value(self.percentile_value(percentile, absolute=True))
        percentile_low = None
        percentile_high = None
        if self.config.percentile_low is not None and self.has_histogram:
            percentile_low = _to_json_value(self.percentile_value(float(self.config.percentile_low)))
        if self.config.percentile_high is not None and self.has_histogram:
            percentile_high = _to_json_value(self.percentile_value(float(self.config.percentile_high)))
        total_batches = self.range_num_batches + self.histogram_num_batches
        total_samples = self.range_num_samples + self.histogram_num_samples
        total_valid_tokens = self.range_num_valid_tokens + self.histogram_num_valid_tokens
        total_padding_tokens = self.range_num_padding_tokens + self.histogram_num_padding_tokens
        total_valid_elements = self.range_num_valid_elements + self.histogram_num_valid_elements
        total_masked_elements = self.range_num_masked_elements + self.histogram_num_masked_elements
        total_excluded = self.range_num_excluded_elements + self.histogram_num_excluded_elements
        total_token_mask_updates = self.range_num_updates_with_token_mask + self.histogram_num_updates_with_token_mask
        total_element_mask_updates = self.range_num_updates_with_element_mask + self.histogram_num_updates_with_element_mask
        total_no_mask_updates = self.range_num_updates_without_mask + self.histogram_num_updates_without_mask
        mask_kinds = []
        if total_token_mask_updates > 0:
            mask_kinds.append("token")
        if total_element_mask_updates > 0:
            mask_kinds.append("element")
        if total_no_mask_updates > 0:
            mask_kinds.append("none")
        mask_coverage_kind = mask_kinds[0] if len(mask_kinds) == 1 else "mixed"
        may_include_padding = self.config.observer_type == "activation" and (
            total_no_mask_updates > 0 or total_element_mask_updates > 0
        )
        qparams_json = _qparams_to_json(self._frozen_qparams)
        return {
            "tensor_name": self.tensor_name,
            "module_name": self.module_name,
            "layer_idx": self.layer_idx,
            "module_type": self.module_type,
            "observer_type": self.config.observer_type,
            "granularity": self.config.granularity,
            "channel_axis": int(self.config.channel_axis),
            "enabled": bool(self.enabled),
            "phase": self.phase,
            "range_finalized": bool(self._range_finalized),
            "histogram_finalized": bool(self._histogram_finalized),
            "finalized": bool(self._range_finalized and self._histogram_finalized),
            "percentile_backend": self.config.percentile_backend,
            "histogram_bins": int(self.config.histogram_bins),
            "histogram_built": bool(self.has_histogram),
            "qparams_frozen": bool(self.qparams_frozen),
            "qparams": qparams_json,
            "scale": qparams_json["scale"],
            "zero_point": qparams_json["zero_point"],
            "qmin": qparams_json["qmin"],
            "qmax": qparams_json["qmax"],
            "device": str(self._stats_device) if self._stats_device is not None else None,
            "min": _to_json_value(self._min),
            "max": _to_json_value(self._max),
            "abs_max": _to_json_value(self._abs_max),
            "mean": _to_json_value(self._mean()),
            "std": _to_json_value(self._std()),
            "std_population": _to_json_value(self._std()),
            "std_definition": "population",
            "std_ddof": 0,
            "numel": int(self.num_elements),
            "num_elements": int(self.num_elements),
            "histogram_num_elements": int(self.histogram_num_elements),
            "num_batches": int(total_batches),
            "num_samples": int(total_samples),
            "num_valid_tokens": int(total_valid_tokens),
            "num_padding_tokens": int(total_padding_tokens),
            "num_valid_elements": int(total_valid_elements),
            "num_masked_elements": int(total_masked_elements),
            "num_excluded_elements": int(total_excluded),
            "mask_coverage_kind": mask_coverage_kind,
            "mask_kinds_seen": mask_kinds,
            "range_num_batches": int(self.range_num_batches),
            "range_num_samples": int(self.range_num_samples),
            "range_num_sample_updates_explicit": int(self.range_num_sample_updates_explicit),
            "range_num_sample_updates_inferred": int(self.range_num_sample_updates_inferred),
            "range_num_sample_updates_unknown": int(self.range_num_sample_updates_unknown),
            "range_num_valid_tokens": int(self.range_num_valid_tokens),
            "range_num_padding_tokens": int(self.range_num_padding_tokens),
            "range_num_valid_elements": int(self.range_num_valid_elements),
            "range_num_masked_elements": int(self.range_num_masked_elements),
            "range_num_excluded_elements": int(self.range_num_excluded_elements),
            "range_num_updates_with_token_mask": int(self.range_num_updates_with_token_mask),
            "range_num_updates_with_element_mask": int(self.range_num_updates_with_element_mask),
            "range_num_updates_without_mask": int(self.range_num_updates_without_mask),
            "range_num_updates_with_padding_mask": int(self.range_num_updates_with_padding_mask),
            "range_num_updates_without_padding_mask": int(self.range_num_updates_without_padding_mask),
            "histogram_num_batches": int(self.histogram_num_batches),
            "histogram_num_samples": int(self.histogram_num_samples),
            "histogram_num_sample_updates_explicit": int(self.histogram_num_sample_updates_explicit),
            "histogram_num_sample_updates_inferred": int(self.histogram_num_sample_updates_inferred),
            "histogram_num_sample_updates_unknown": int(self.histogram_num_sample_updates_unknown),
            "histogram_num_valid_tokens": int(self.histogram_num_valid_tokens),
            "histogram_num_padding_tokens": int(self.histogram_num_padding_tokens),
            "histogram_num_valid_elements": int(self.histogram_num_valid_elements),
            "histogram_num_masked_elements": int(self.histogram_num_masked_elements),
            "histogram_num_excluded_elements": int(self.histogram_num_excluded_elements),
            "histogram_num_updates_with_token_mask": int(self.histogram_num_updates_with_token_mask),
            "histogram_num_updates_with_element_mask": int(self.histogram_num_updates_with_element_mask),
            "histogram_num_updates_without_mask": int(self.histogram_num_updates_without_mask),
            "histogram_num_updates_with_padding_mask": int(self.histogram_num_updates_with_padding_mask),
            "histogram_num_updates_without_padding_mask": int(self.histogram_num_updates_without_padding_mask),
            "num_updates_with_token_mask": int(total_token_mask_updates),
            "num_updates_with_element_mask": int(total_element_mask_updates),
            "num_updates_without_mask": int(total_no_mask_updates),
            "num_updates_with_padding_mask": int(self.range_num_updates_with_padding_mask + self.histogram_num_updates_with_padding_mask),
            "num_updates_without_padding_mask": int(self.range_num_updates_without_padding_mask + self.histogram_num_updates_without_padding_mask),
            "token_mask_provided": bool(total_token_mask_updates > 0),
            "element_mask_provided": bool(total_element_mask_updates > 0),
            "mask_provided": bool(total_token_mask_updates > 0 or total_element_mask_updates > 0),
            "padding_mask_provided": bool(self.range_num_updates_with_padding_mask > 0 or self.histogram_num_updates_with_padding_mask > 0),
            "num_sample_updates_explicit": int(self.range_num_sample_updates_explicit + self.histogram_num_sample_updates_explicit),
            "num_sample_updates_inferred": int(self.range_num_sample_updates_inferred + self.histogram_num_sample_updates_inferred),
            "num_sample_updates_unknown": int(self.range_num_sample_updates_unknown + self.histogram_num_sample_updates_unknown),
            "may_include_padding_tokens": bool(may_include_padding),
            "cpu_transfer_num_elements": 0,
            "num_cached_chunks": 0,
            "percentiles": percentiles,
            "abs_percentiles": abs_percentiles,
            "percentile_low": percentile_low,
            "percentile_high": percentile_high,
            "percentile_low_value": percentile_low,
            "percentile_high_value": percentile_high,
        }

    def state_summary(self) -> dict[str, Any]:
        """兼容量化控制器的轻量摘要入口。"""

        return self.summary()


class ActivationObserver(TensorDistributionObserver):
    """用于 FFN 线性层输入 activation 的 per-tensor observer。"""

    def __init__(self, config: ObserverConfig, **kwargs: Any) -> None:
        if config.observer_type != "activation":
            raise ValueError("ActivationObserver 需要 observer_type='activation'")
        super().__init__(config, **kwargs)


class WeightObserver(TensorDistributionObserver):
    """用于 nn.Linear.weight 的 observer；默认按 out_features 做 per-channel 统计。"""

    def __init__(self, config: ObserverConfig, **kwargs: Any) -> None:
        if config.observer_type != "weight":
            raise ValueError("WeightObserver 需要 observer_type='weight'")
        super().__init__(config, **kwargs)


def observe_weight(
    weight: Tensor,
    *,
    config: ObserverConfig,
    tensor_name: str = "weight",
    module_name: str | None = None,
    layer_idx: int | None = None,
    module_type: str | None = "nn.Linear",
) -> dict[str, Any]:
    """两遍观察 FP32 weight 分布，不修改 weight。"""

    observer = WeightObserver(
        config,
        tensor_name=tensor_name,
        module_name=module_name,
        layer_idx=layer_idx,
        module_type=module_type,
        enabled=True,
    )
    observer.start_range_pass(reset=True)
    observer.update(weight)
    observer.start_histogram_pass()
    observer.update(weight)
    from .calibration import freeze as freeze_calibration

    freeze_calibration(observer)
    return observer.summary()


__all__ = [
    "ActivationObserver",
    "ObserverConfig",
    "ObserverGranularity",
    "ObserverPhase",
    "ObserverType",
    "TensorDistributionObserver",
    "WeightObserver",
    "load_project_observer_configs",
    "observe_weight",
    "observer_config_from_mapping",
]






