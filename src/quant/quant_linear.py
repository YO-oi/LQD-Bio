from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from src.metrics.common_metrics import activation_clipping_rate, clipping_rate
except ImportError:  # pragma: no cover - package-relative debugging fallback.
    from ..metrics.common_metrics import activation_clipping_rate, clipping_rate

try:
    from src.utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
except ImportError:  # pragma: no cover - package-relative debugging fallback.
    from ..utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields

try:
    from .calibration import calculate_symmetric_weight_qparams_from_tensor, freeze as freeze_calibration
    from .observer import ActivationObserver, ObserverConfig, load_project_observer_configs
    from .quant_core import (
        Int4QuantConfig,
        QuantMode,
        default_quant_mode_from_mapping,
        int4_quant_config_from_mapping,
    )
    from .ste import quantize_dequantize_ste
except ImportError:  # pragma: no cover - direct script debugging fallback.
    from src.quant.calibration import calculate_symmetric_weight_qparams_from_tensor, freeze as freeze_calibration
    from src.quant.observer import ActivationObserver, ObserverConfig, load_project_observer_configs
    from src.quant.quant_core import (
        Int4QuantConfig,
        QuantMode,
        default_quant_mode_from_mapping,
        int4_quant_config_from_mapping,
    )
    from src.quant.ste import quantize_dequantize_ste


def _coerce_mode_and_config(
    quant_config: Int4QuantConfig | Mapping[str, Any] | None,
) -> tuple[QuantMode, Int4QuantConfig]:
    if quant_config is None:
        return "w4a4", Int4QuantConfig()
    if isinstance(quant_config, Int4QuantConfig):
        return "w4a4", quant_config
    if isinstance(quant_config, Mapping):
        if "QUANT" in quant_config and isinstance(quant_config["QUANT"], Mapping):
            quant_config = quant_config["QUANT"]  # type: ignore[assignment]
        return default_quant_mode_from_mapping(quant_config), int4_quant_config_from_mapping(quant_config)
    raise TypeError(f"quant_config must be Int4QuantConfig, mapping, or None, got {type(quant_config)!r}")


def _default_activation_observer_config() -> ObserverConfig:
    return load_project_observer_configs()["activation"]


class QuantLinear(nn.Module):
    """nn.Linear wrapper supporting W4/A4/W4A4 fake quantization.

    The wrapper keeps FP32 master weights in the original linear module,
    uses per-tensor or explicitly requested per-channel activation qparams,
    per-output-channel weight qparams, applies fake quantization during forward,
    and can store OER activations.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        quant_config: Int4QuantConfig | Mapping[str, Any] | None = None,
        activation_observer_config: ObserverConfig | None = None,
        target_name: str | None = None,
        store_oer_activation: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"QuantLinear can only wrap nn.Linear, got {type(linear)!r}")

        mode, config = _coerce_mode_and_config(quant_config)
        self.linear = linear
        self.config = config
        self.target_name = str(target_name or "quant_linear")
        self.weight_quant_enabled = mode in {"w4", "w4a4"}
        self.activation_quant_enabled = mode in {"a4", "w4a4"}
        self.calibration_enabled = False
        self.fake_quant_enabled = False
        self.qparams_frozen = False
        self.store_oer_activation = bool(store_oer_activation)
        self.capture_integer_grid = False
        self.integer_grid_capture_scope = "selected_targets_only"
        self.integer_grid_save_device = "cpu"
        self.integer_grid_detach = True
        self.integer_grid_canonical_layout = "BTC"
        self._activation_padding_mask_provider: Callable[[], Any] | None = None

        observer_config = activation_observer_config or _default_activation_observer_config()
        if observer_config.observer_type != "activation":
            raise ValueError("QuantLinear activation observer_config must have observer_type='activation'")
        if observer_config.granularity == "per_channel" and int(observer_config.channel_axis) != -1:
            raise ValueError("QuantLinear activation per-channel observer requires channel_axis=-1")
        self.activation_granularity = str(observer_config.granularity)
        self.activation_channel_axis = int(observer_config.channel_axis)
        self.activation_observer = ActivationObserver(
            observer_config,
            tensor_name=f"{self.target_name}.input_activation",
            module_name=self.target_name,
            module_type="nn.Linear",
            enabled=False,
        )

        activation_qparam_shape = (self.in_features,) if self.activation_granularity == "per_channel" else ()
        self.register_buffer("activation_scale", torch.full(activation_qparam_shape, float("nan"), dtype=torch.float32))
        self.register_buffer("activation_zero_point", torch.zeros(activation_qparam_shape, dtype=torch.float32))
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float32))
        self.register_buffer("weight_zero_point", torch.empty(0, dtype=torch.float32))
        self.fixed_weight_qparams_enabled = False
        self.last_activation_clipping_rate: Tensor | None = None
        self.last_activation_clipping_rate_by_sample: Tensor | None = None
        self.last_fp32_activation_for_oer: Tensor | None = None
        self.last_quant_activation_for_oer: Tensor | None = None
        self.last_activation_valid_mask_for_oer: Tensor | None = None
        self.last_fp32_projection_for_oer: Tensor | None = None
        self.last_quant_projection_for_oer: Tensor | None = None
        self.last_activation_layout_info_for_oer: dict[str, Any] | None = None
        self.qvf_activation_capture_enabled = False
        self.last_qvf_activation: Tensor | None = None
        self.last_qvf_valid_mask: Tensor | None = None
        self.last_qvf_layout_info: dict[str, Any] | None = None
        self.last_integer_grid_diagnostics: dict[str, Any] | None = None
        self._weight_qparams_cache_key: tuple[Any, ...] | None = None
        self._weight_qparams_cache: tuple[Tensor, Tensor] | None = None
        self._activation_qparams_cache_key: tuple[Any, ...] | None = None
        self._activation_qparams_cache: tuple[Tensor, Tensor] | None = None

    @property
    def weight(self) -> nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.linear.bias

    @property
    def in_features(self) -> int:
        return int(self.linear.in_features)

    @property
    def out_features(self) -> int:
        return int(self.linear.out_features)

    def _apply(self, fn: Any) -> "QuantLinear":
        result = super()._apply(fn)
        self._clear_weight_qparams_cache()
        self._clear_activation_qparams_cache()
        return result

    @property
    def weight_qparams_cache_is_valid(self) -> bool:
        return self._weight_qparams_cache is not None

    @property
    def activation_qparams_cache_is_valid(self) -> bool:
        return self._activation_qparams_cache is not None

    def set_activation_padding_mask_provider(
        self,
        provider: Callable[[], Any] | None,
    ) -> None:
        """Register a provider for the current activation padding mask."""

        self._activation_padding_mask_provider = provider

    def reset_activation_observer(self) -> None:
        self.activation_observer.reset()
        self.activation_observer.set_enabled(self.calibration_enabled)
        self.invalidate_activation_qparams()

    def invalidate_activation_qparams(self) -> None:
        self.qparams_frozen = False
        shape = (self.in_features,) if self.activation_granularity == "per_channel" else ()
        device = self.activation_scale.device
        self.activation_scale = torch.full(shape, float("nan"), device=device, dtype=torch.float32)
        self.activation_zero_point = torch.zeros(shape, device=device, dtype=torch.float32)
        self._clear_activation_qparams_cache()

    def invalidate_weight_qparams_cache(self) -> None:
        self._clear_weight_qparams_cache()

    def _validate_external_qrange(self, qmin: int | None, qmax: int | None, *, kind: str) -> None:
        """Validate externally supplied qrange against module config."""

        expected_qmin = int(self.config.weight_qmin if kind == "weight" else self.config.activation_qmin)
        expected_qmax = int(self.config.weight_qmax if kind == "weight" else self.config.activation_qmax)
        if qmin is not None and int(qmin) != expected_qmin:
            raise ValueError(f"{self.target_name}: {kind} qmin={qmin} does not match configured qmin={expected_qmin}")
        if qmax is not None and int(qmax) != expected_qmax:
            raise ValueError(f"{self.target_name}: {kind} qmax={qmax} does not match configured qmax={expected_qmax}")

    def _normalize_activation_qparams(self, scale: Tensor, zero_point: Tensor) -> tuple[Tensor, Tensor]:
        scale = scale.detach().float().reshape(-1)
        zero_point = zero_point.detach().float().reshape(-1)
        if self.activation_granularity == "per_tensor":
            if int(scale.numel()) != 1 or int(zero_point.numel()) != 1:
                raise ValueError(f"{self.target_name}: per-tensor activation qparams must be scalar")
            return scale.reshape(()), zero_point.reshape(())
        if self.activation_granularity == "per_channel":
            expected = int(self.in_features)
            if int(scale.numel()) != expected or int(zero_point.numel()) != expected:
                raise ValueError(
                    f"{self.target_name}: per-channel activation qparams must have numel=in_features={expected}; "
                    f"got scale={int(scale.numel())}, zero_point={int(zero_point.numel())}"
                )
            return scale.contiguous(), zero_point.contiguous()
        raise ValueError(f"{self.target_name}: unsupported activation granularity {self.activation_granularity!r}")

    def _set_activation_qparam_buffers(self, scale: Tensor, zero_point: Tensor) -> None:
        scale, zero_point = self._normalize_activation_qparams(scale, zero_point)
        if not torch.isfinite(scale).all().item() or not (scale > 0).all().item():
            raise ValueError(f"{self.target_name}: activation scale must be finite and positive")
        if not torch.isfinite(zero_point).all().item():
            raise ValueError(f"{self.target_name}: activation zero_point must be finite")
        device = self.activation_scale.device
        self.activation_scale = scale.to(device=device, dtype=torch.float32).clone()
        self.activation_zero_point = zero_point.to(device=device, dtype=torch.float32).clone()

    def set_fixed_activation_qparams(
        self,
        scale: Tensor,
        zero_point: Tensor,
        *,
        qmin: int | None = None,
        qmax: int | None = None,
    ) -> None:
        """Set fixed activation qparams; activation quantization happens at the linear input."""

        self._validate_external_qrange(qmin, qmax, kind="activation")
        self._set_activation_qparam_buffers(scale, zero_point)
        self.calibration_enabled = False
        self.activation_observer.disable()
        self.qparams_frozen = True
        self._clear_activation_qparams_cache()

    def set_fixed_weight_qparams(
        self,
        scale: Tensor,
        zero_point: Tensor,
        *,
        qmin: int | None = None,
        qmax: int | None = None,
    ) -> None:
        """Set fixed weight qparams; fake quant does not modify FP32 master weights."""

        self._validate_external_qrange(qmin, qmax, kind="weight")
        scale = scale.detach().float()
        zero_point = zero_point.detach().float()
        if scale.ndim == 1 and int(scale.numel()) == int(self.out_features):
            scale = scale.reshape(self.out_features, 1)
        if zero_point.ndim == 1 and int(zero_point.numel()) == int(self.out_features):
            zero_point = zero_point.reshape(self.out_features, 1)
        if tuple(scale.shape) != tuple(zero_point.shape):
            raise ValueError(
                f"{self.target_name}: weight scale/zero_point shape 韫囧懘銆忔稉鈧懛杈剧礉"
                f"瑜版挸澧?{tuple(scale.shape)} vs {tuple(zero_point.shape)}"
            )
        if int(scale.shape[0]) != int(self.out_features):
            raise ValueError(
                f"{self.target_name}: weight qparams dim 0 must equal out_features={self.out_features}; "
                f"got shape={tuple(scale.shape)}"
            )
        if not torch.isfinite(scale).all().item() or not (scale > 0).all().item():
            raise ValueError(f"{self.target_name}: weight scale must be finite and positive")
        if not torch.isfinite(zero_point).all().item():
            raise ValueError(f"{self.target_name}: weight zero_point must be finite")
        self.weight_scale = scale.to(device=self.linear.weight.device, dtype=torch.float32).clone()
        self.weight_zero_point = zero_point.to(device=self.linear.weight.device, dtype=torch.float32).clone()
        self.fixed_weight_qparams_enabled = True
        self._clear_weight_qparams_cache()

    def clear_fixed_weight_qparams(self) -> None:
        """Clear externally fixed weight qparams."""

        self.weight_scale = torch.empty(0, device=self.linear.weight.device, dtype=torch.float32)
        self.weight_zero_point = torch.empty(0, device=self.linear.weight.device, dtype=torch.float32)
        self.fixed_weight_qparams_enabled = False
        self._clear_weight_qparams_cache()

    def freeze_activation_qparams(self) -> None:
        """Backward-compatible alias for freeze_calibration."""

        self.freeze_calibration()

    def enable_calibration(self) -> None:
        """Start or advance the two-pass activation calibration flow."""

        self.disable_fake_quant()
        if not self.activation_quant_enabled:
            self.calibration_enabled = False
            self.activation_observer.disable()
            self.qparams_frozen = True
            return

        if self.qparams_frozen or not self.activation_observer.has_data:
            self.activation_observer.start_range_pass(reset=True)
            self.qparams_frozen = False
        elif self.activation_observer.phase == "range":
            self.activation_observer.start_histogram_pass()

        self.calibration_enabled = True
        self.activation_observer.enable()

    def freeze_calibration(self) -> None:
        """Freeze activation qparams from observer statistics."""

        if self.activation_quant_enabled:
            if not self.activation_observer.has_histogram:
                raise RuntimeError(
                    f"{self.target_name}: freeze_calibration requires completed histogram calibration"
                )
            lower_percentile: float | None = None
            upper_percentile: float | None = None
            if not self.config.activation_symmetric:
                lower_percentile, upper_percentile = self.config.asymmetric_percentile_bounds()
            qparams = freeze_calibration(
                self.activation_observer,
                symmetric=bool(self.config.activation_symmetric),
                retained_percentile=self.config.symmetric_abs_percentile(),
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                qmin=int(self.config.activation_qmin),
                qmax=int(self.config.activation_qmax),
                eps=float(self.config.eps),
            )
            scale = qparams["scale"]
            zero_point = qparams["zero_point"]
            self._set_activation_qparam_buffers(scale, zero_point)

        self.calibration_enabled = False
        self.activation_observer.disable()
        self.qparams_frozen = True
        self._clear_activation_qparams_cache()

    def enable_fake_quant(self) -> None:
        if self.activation_quant_enabled and not self.qparams_frozen:
            raise RuntimeError(f"{self.target_name}: activation fake quant requires freeze_calibration() first")
        self.calibration_enabled = False
        self.activation_observer.disable()
        self.fake_quant_enabled = True

    def disable_fake_quant(self) -> None:
        self.fake_quant_enabled = False

    def set_integer_grid_capture(
        self,
        enabled: bool,
        *,
        scope: str = "selected_targets_only",
        save_device: str = "cpu",
        detach: bool = True,
        canonical_layout: str = "BTC",
    ) -> None:
        """Enable optional integer-grid diagnostics without changing forward math."""

        save_device = str(save_device).lower()
        canonical_layout = str(canonical_layout).upper()
        if save_device not in {"cpu", "module"}:
            raise ValueError(f"{self.target_name}: INTEGER_GRID_SAVE_DEVICE must be 'cpu' or 'module', got {save_device!r}")
        if canonical_layout != "BTC":
            raise ValueError(f"{self.target_name}: INTEGER_GRID_CANONICAL_LAYOUT must be 'BTC', got {canonical_layout!r}")
        self.capture_integer_grid = bool(enabled)
        self.integer_grid_capture_scope = str(scope)
        self.integer_grid_save_device = save_device
        self.integer_grid_detach = bool(detach)
        self.integer_grid_canonical_layout = canonical_layout
        if not self.capture_integer_grid:
            self.last_integer_grid_diagnostics = None

    def _clear_weight_qparams_cache(self) -> None:
        self._weight_qparams_cache_key = None
        self._weight_qparams_cache = None

    def _clear_activation_qparams_cache(self) -> None:
        self._activation_qparams_cache_key = None
        self._activation_qparams_cache = None

    def _provider_padding_context(self, provider_value: Any) -> tuple[Tensor | None, dict[str, Any]]:
        if provider_value is None or isinstance(provider_value, Tensor):
            return provider_value, {}
        if not isinstance(provider_value, Mapping):
            raise TypeError(
                f"{self.target_name}: activation padding mask provider must return Tensor, mapping, or None, "
                f"got {type(provider_value)!r}"
            )
        padding_mask = provider_value.get("padding_mask", provider_value.get("mask"))
        if padding_mask is not None and not isinstance(padding_mask, Tensor):
            padding_mask = torch.as_tensor(padding_mask)
        metadata = {
            "activation_layout": provider_value.get("activation_layout", provider_value.get("raw_layout")),
            "mask_layout": provider_value.get("mask_layout", "BT" if padding_mask is not None else None),
            "raw_shape": provider_value.get("raw_shape", provider_value.get("activation_raw_shape")),
            "padding_mask_shape": provider_value.get("padding_mask_shape"),
        }
        return padding_mask, metadata

    def _resolve_valid_mask(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None,
        padding_mask: Tensor | None,
    ) -> Tensor | None:
        if valid_mask is not None and padding_mask is not None:
            raise ValueError(f"{self.target_name}: valid_mask and padding_mask are mutually exclusive")
        if padding_mask is not None:
            # Project convention: padding_mask == True means padding; invert so True means valid.
            mask = ~padding_mask.detach().to(device=x.device, dtype=torch.bool)
        elif valid_mask is not None:
            mask = valid_mask.detach().to(device=x.device, dtype=torch.bool)
        else:
            return None

        if tuple(mask.shape) == tuple(x.shape):
            return mask.contiguous()
        if tuple(mask.shape) == tuple(x.shape[:-1]):
            return mask.contiguous()
        raise ValueError(
            f"{self.target_name}: mask shape {tuple(mask.shape)} does not exactly align with activation shape "
            f"{tuple(x.shape)}. QuantLinear no longer transposes masks from shape heuristics; provide a mask "
            "already aligned to the raw activation layout, or pass a padding_mask for diagnostics only."
        )

    def _diagnostic_activation_context(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None,
        padding_mask: Tensor | None,
        provider_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        metadata = dict(provider_metadata or {})
        if padding_mask is not None:
            canonical = canonicalize_activation_btc(
                x.detach().float(),
                raw_layout=metadata.get("activation_layout"),
                padding_mask=padding_mask,
                context=f"{self.target_name}.activation_diagnostics",
            )
            mask_bt = canonical.padding_mask
            valid_bt = None if mask_bt is None else (~mask_bt).to(device=canonical.tensor.device, dtype=torch.bool)
            return canonical.tensor, valid_bt, layout_info_to_public_fields(canonical.layout_info)

        if valid_mask is not None:
            resolved = self._resolve_valid_mask(x, valid_mask=valid_mask, padding_mask=None)
            if x.ndim == 3:
                raise ValueError(
                    f"{self.target_name}: 3D activation diagnostics require a padding_mask with explicit [B,T] semantics; "
                    "direct valid_mask cannot disambiguate BTC/TBC layout"
                )
            return x.detach().float(), resolved, {
                "original_layout": "NC" if x.ndim == 2 else "",
                "original_shape": tuple(int(item) for item in x.shape),
                "canonical_layout": "NC" if x.ndim == 2 else "",
                "canonical_shape": tuple(int(item) for item in x.shape),
                "layout_inference_source": "non_sequence_tensor",
                "padding_mask_available": False,
                "padding_mask_aligned": False,
                "mask_status": "not_applicable",
            }

        if x.ndim == 3:
            raise ValueError(
                f"{self.target_name}: 3D activation diagnostics require padding_mask or valid_mask; "
                "QuantLinear will not infer BTC/TBC layout without mask context"
            )
        return x.detach().float(), None, {
            "original_layout": "NC" if x.ndim == 2 else "",
            "original_shape": tuple(int(item) for item in x.shape),
            "canonical_layout": "NC" if x.ndim == 2 else "",
            "canonical_shape": tuple(int(item) for item in x.shape),
            "layout_inference_source": "non_sequence_tensor",
            "padding_mask_available": False,
            "padding_mask_aligned": False,
            "mask_status": "not_applicable",
        }

    def _canonicalize_like_diagnostics(
        self,
        value: Tensor,
        *,
        layout_info: Mapping[str, Any] | None,
        padding_mask: Tensor | None,
    ) -> Tensor:
        if value.ndim != 3:
            return value.detach()
        raw_layout = None if layout_info is None else layout_info.get("original_layout")
        if not raw_layout:
            raise ValueError(f"{self.target_name}: cannot canonicalize diagnostic tensor without original_layout")
        canonical = canonicalize_activation_btc(
            value.detach().float(),
            raw_layout=str(raw_layout),
            padding_mask=padding_mask,
            context=f"{self.target_name}.diagnostic_tensor",
        )
        return canonical.tensor

    def _canonicalize_qvf_activation(
        self,
        value: Tensor,
        *,
        layout_info: Mapping[str, Any] | None,
        padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, dict[str, Any]]:
        if value.ndim != 3:
            raise ValueError(f"{self.target_name}: QVF activation capture requires rank-3 token tensor")
        raw_layout = None if layout_info is None else layout_info.get("original_layout")
        if not raw_layout:
            raise ValueError(f"{self.target_name}: cannot canonicalize QVF tensor without original_layout")
        canonical = canonicalize_activation_btc(
            value,
            raw_layout=str(raw_layout),
            padding_mask=padding_mask,
            context=f"{self.target_name}.qvf_activation",
        )
        valid_mask = None
        if canonical.padding_mask is not None:
            valid_mask = (~canonical.padding_mask).to(device=canonical.tensor.device, dtype=torch.bool)
        return canonical.tensor, valid_mask, dict(canonical.layout_info)

    def _integer_grid_store_tensor(self, value: Tensor) -> Tensor:
        tensor = value.detach() if self.integer_grid_detach else value
        if self.integer_grid_save_device == "cpu":
            tensor = tensor.cpu()
        return tensor.contiguous()

    def _capture_integer_grid_diagnostics(
        self,
        x_btc: Tensor | None,
        *,
        scale: Tensor,
        zero_point: Tensor,
        valid_mask: Tensor | None,
        layout_info: Mapping[str, Any] | None,
    ) -> None:
        if not self.capture_integer_grid:
            self.last_integer_grid_diagnostics = None
            return
        if not (self.fake_quant_enabled and self.activation_quant_enabled):
            self.last_integer_grid_diagnostics = None
            return
        if x_btc is None or x_btc.ndim != 3:
            raise RuntimeError(f"{self.target_name}: integer-grid capture requires canonical [B,T,C] diagnostic activation")
        scale_t = scale.to(device=x_btc.device, dtype=x_btc.dtype)
        zero_t = zero_point.to(device=x_btc.device, dtype=x_btc.dtype)
        q_preclip = torch.round(x_btc / torch.clamp(scale_t, min=float(self.config.eps)) + zero_t)
        q_int = torch.clamp(q_preclip, min=float(self.config.activation_qmin), max=float(self.config.activation_qmax))
        boundary_hit = (q_int <= float(self.config.activation_qmin)) | (q_int >= float(self.config.activation_qmax))
        strict_overflow = (q_preclip < float(self.config.activation_qmin)) | (q_preclip > float(self.config.activation_qmax))
        exact_zero = q_int == zero_t
        near_zero_pm1 = (q_int - zero_t).abs() <= 1.0
        payload: dict[str, Any] = {
            "x": self._integer_grid_store_tensor(x_btc.float()),
            "q_preclip": self._integer_grid_store_tensor(q_preclip.float()),
            "q_int": self._integer_grid_store_tensor(q_int.float()),
            "boundary_hit_mask": self._integer_grid_store_tensor(boundary_hit.bool()),
            "strict_overflow_mask": self._integer_grid_store_tensor(strict_overflow.bool()),
            "exact_zero_mask": self._integer_grid_store_tensor(exact_zero.bool()),
            "near_zero_pm1_mask": self._integer_grid_store_tensor(near_zero_pm1.bool()),
            "scale": self._integer_grid_store_tensor(scale_t.reshape(-1).float()),
            "zero_point": self._integer_grid_store_tensor(zero_t.reshape(-1).float()),
            "qmin": int(self.config.activation_qmin),
            "qmax": int(self.config.activation_qmax),
            "input_layout_info": dict(layout_info or {}),
            "mapping_source": "quantlinear_forward",
            "integer_grid_source": "quantlinear_forward",
            "is_direct_quantlinear_forward_grid": True,
            "canonical_layout": "BTC",
        }
        if valid_mask is not None:
            payload["valid_token_mask"] = self._integer_grid_store_tensor(valid_mask.bool())
        # Deprecated aliases are kept inside the diagnostic payload only.
        payload["q_preclip_deprecated_alias_of"] = "q_preclip"
        payload["clip_mask"] = payload["strict_overflow_mask"]
        payload["clip_mask_deprecated_alias_of"] = "strict_overflow_mask"
        payload["near_zero_mask"] = payload["exact_zero_mask"]
        payload["near_zero_mask_deprecated_alias_of"] = "exact_zero_mask"
        self.last_integer_grid_diagnostics = payload

    def get_integer_grid_diagnostics(self) -> dict[str, Any] | None:
        return None if self.last_integer_grid_diagnostics is None else dict(self.last_integer_grid_diagnostics)

    def clear_integer_grid_diagnostics(self) -> None:
        self.last_integer_grid_diagnostics = None

    def _expand_valid_mask_for_clipping(self, x: Tensor, valid_mask: Tensor | None) -> Tensor | None:
        if valid_mask is None:
            return None
        if tuple(valid_mask.shape) == tuple(x.shape):
            return valid_mask
        if tuple(valid_mask.shape) == tuple(x.shape[:-1]):
            return valid_mask.unsqueeze(-1).expand_as(x)
        raise ValueError(
            f"{self.target_name}: clipping valid_mask shape {tuple(valid_mask.shape)} "
            f"must match activation shape {tuple(x.shape)} or token shape {tuple(x.shape[:-1])}"
        )

    def _sample_clipping_reduce_dims(self, x: Tensor) -> tuple[int, ...] | None:
        if x.ndim < 2:
            return None
        if x.ndim == 2:
            return (1,)
        # Do not guess the batch dimension for raw 3D activations inside forward.
        # Layout canonicalization is performed in analysis/cache code where batch
        # context and padding masks are available.
        return None

    def _activation_clipping_rate_by_sample(
        self,
        x: Tensor,
        *,
        scale: Tensor,
        zero_point: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor | None:
        reduce_dims = self._sample_clipping_reduce_dims(x)
        if reduce_dims is None:
            return None
        return clipping_rate(
            x,
            scale=scale,
            zero_point=zero_point,
            qmin=int(self.config.activation_qmin),
            qmax=int(self.config.activation_qmax),
            valid_mask=valid_mask,
            reduce_dims=reduce_dims,
            reduction="none",
        ).detach()

    def _activation_qparams_for(self, x: Tensor) -> tuple[Tensor, Tensor]:
        cache_key = (
            x.device,
            x.dtype,
            tuple(int(item) for item in self.activation_scale.shape),
            tuple(int(item) for item in self.activation_zero_point.shape),
            int(getattr(self.activation_scale, "_version", -1)),
            int(getattr(self.activation_zero_point, "_version", -1)),
        )
        if self._activation_qparams_cache_key == cache_key and self._activation_qparams_cache is not None:
            return self._activation_qparams_cache

        scale = self.activation_scale
        zero_point = self.activation_zero_point
        if scale.device != x.device or scale.dtype != x.dtype:
            scale = scale.to(device=x.device, dtype=x.dtype)
        if zero_point.device != x.device or zero_point.dtype != x.dtype:
            zero_point = zero_point.to(device=x.device, dtype=x.dtype)

        self._activation_qparams_cache_key = cache_key
        self._activation_qparams_cache = (scale, zero_point)
        return scale, zero_point

    def _weight_qparams_for(self, weight: Tensor) -> tuple[Tensor, Tensor]:
        if self.fixed_weight_qparams_enabled:
            if int(self.weight_scale.numel()) <= 0 or int(self.weight_zero_point.numel()) <= 0:
                raise RuntimeError(f"{self.target_name}: fixed_weight_qparams_enabled=True but weight qparams are empty")
            cache_key = (
                "fixed",
                weight.device,
                weight.dtype,
                int(getattr(self.weight_scale, "_version", -1)),
                int(getattr(self.weight_zero_point, "_version", -1)),
            )
            if self._weight_qparams_cache_key == cache_key and self._weight_qparams_cache is not None:
                return self._weight_qparams_cache
            scale = self.weight_scale
            zero_point = self.weight_zero_point
            if scale.device != weight.device or scale.dtype != weight.dtype:
                scale = scale.to(device=weight.device, dtype=weight.dtype)
            if zero_point.device != weight.device or zero_point.dtype != weight.dtype:
                zero_point = zero_point.to(device=weight.device, dtype=weight.dtype)
            self._weight_qparams_cache_key = cache_key
            self._weight_qparams_cache = (scale, zero_point)
            return scale, zero_point

        cache_key = (
            int(getattr(weight, "_version", -1)),
            tuple(weight.shape),
            weight.device,
            weight.dtype,
            int(self.config.weight_qmax),
            float(self.config.eps),
        )
        if self._weight_qparams_cache_key == cache_key and self._weight_qparams_cache is not None:
            return self._weight_qparams_cache

        scale, zero_point = calculate_symmetric_weight_qparams_from_tensor(
            weight,
            qmax=int(self.config.weight_qmax),
            eps=float(self.config.eps),
            channel_axis=0,
        )
        self._weight_qparams_cache_key = cache_key
        self._weight_qparams_cache = (scale, zero_point)
        return scale, zero_point

    def activation_quant(
        self,
        x: Tensor,
        *,
        diagnostic_x: Tensor | None = None,
        diagnostic_valid_mask: Tensor | None = None,
        diagnostic_layout_info: Mapping[str, Any] | None = None,
    ) -> Tensor:
        if not (self.fake_quant_enabled and self.activation_quant_enabled):
            self.last_activation_clipping_rate = None
            self.last_activation_clipping_rate_by_sample = None
            self.last_integer_grid_diagnostics = None
            return x
        if not self.qparams_frozen:
            raise RuntimeError(f"{self.target_name}: activation fake quant requires freeze_calibration() first")
        scale, zero_point = self._activation_qparams_for(x)
        stats_x = diagnostic_x if diagnostic_x is not None else x.detach().float()
        stats_valid_mask = diagnostic_valid_mask
        clipping_mask = self._expand_valid_mask_for_clipping(stats_x, stats_valid_mask)
        self.last_activation_clipping_rate = activation_clipping_rate(
            stats_x,
            scale=scale,
            zero_point=zero_point,
            qmin=int(self.config.activation_qmin),
            qmax=int(self.config.activation_qmax),
            valid_mask=clipping_mask,
            reduction="mean",
        ).detach()
        self.last_activation_clipping_rate_by_sample = self._activation_clipping_rate_by_sample(
            stats_x,
            scale=scale,
            zero_point=zero_point,
            valid_mask=clipping_mask,
        )
        self._capture_integer_grid_diagnostics(
            diagnostic_x,
            scale=scale,
            zero_point=zero_point,
            valid_mask=diagnostic_valid_mask,
            layout_info=diagnostic_layout_info,
        )
        return quantize_dequantize_ste(
            x,
            scale,
            zero_point,
            qmin=int(self.config.activation_qmin),
            qmax=int(self.config.activation_qmax),
        )

    def weight_quant(self, weight: Tensor) -> Tensor:
        if not (self.fake_quant_enabled and self.weight_quant_enabled):
            return weight
        scale, zero_point = self._weight_qparams_for(weight)
        return quantize_dequantize_ste(
            weight,
            scale,
            zero_point,
            qmin=int(self.config.weight_qmin),
            qmax=int(self.config.weight_qmax),
        )

    def get_oer_activation_tensors(self) -> tuple[Tensor | None, Tensor | None]:

        return self.last_fp32_activation_for_oer, self.last_quant_activation_for_oer

    def get_oer_projection_tensors(self) -> tuple[Tensor | None, Tensor | None]:

        return self.last_fp32_projection_for_oer, self.last_quant_projection_for_oer

    def set_qvf_activation_capture(self, enabled: bool) -> None:

        self.clear_qvf_activation_tensor()
        self.qvf_activation_capture_enabled = bool(enabled)

    def get_qvf_activation_tensor(self) -> tuple[Tensor | None, Tensor | None]:

        return self.last_qvf_activation, self.last_qvf_valid_mask

    def clear_qvf_activation_tensor(self) -> None:

        self.last_qvf_activation = None
        self.last_qvf_valid_mask = None
        self.last_qvf_layout_info = None

    def clear_oer_activation_tensors(self) -> None:

        self.last_fp32_activation_for_oer = None
        self.last_quant_activation_for_oer = None
        self.last_activation_valid_mask_for_oer = None
        self.last_fp32_projection_for_oer = None
        self.last_quant_projection_for_oer = None
        self.last_activation_layout_info_for_oer = None
        self.clear_qvf_activation_tensor()
        self.last_integer_grid_diagnostics = None

    def clear_oer_projection_tensors(self) -> None:

        self.clear_oer_activation_tensors()

    def forward(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        provider_metadata: dict[str, Any] = {}
        if padding_mask is None and valid_mask is None and self._activation_padding_mask_provider is not None:
            padding_mask, provider_metadata = self._provider_padding_context(self._activation_padding_mask_provider())
        needs_activation_diagnostics = bool(
            self.store_oer_activation
            or self.capture_integer_grid
            or (self.calibration_enabled and self.activation_quant_enabled)
            or (self.fake_quant_enabled and self.activation_quant_enabled)
        )
        if needs_activation_diagnostics:
            diagnostic_x, diagnostic_valid_mask, diagnostic_layout_info = self._diagnostic_activation_context(
                x,
                valid_mask=valid_mask,
                padding_mask=padding_mask,
                provider_metadata=provider_metadata,
            )
        else:
            diagnostic_x = None
            diagnostic_valid_mask = None
            diagnostic_layout_info = None
        padding_mask_for_diagnostics = None if padding_mask is None else padding_mask.detach().to(dtype=torch.bool)

        if self.calibration_enabled and self.activation_quant_enabled:
            if diagnostic_x is None:
                raise RuntimeError(f"{self.target_name}: missing diagnostic activation for calibration")
            self.activation_observer.update(
                diagnostic_x,
                valid_mask=diagnostic_valid_mask,
                num_samples=int(diagnostic_x.shape[0]) if diagnostic_x.ndim >= 1 else None,
            )

        if self.store_oer_activation:
            if diagnostic_x is None or diagnostic_layout_info is None:
                raise RuntimeError(f"{self.target_name}: missing diagnostic activation for OER capture")
            self.last_fp32_activation_for_oer = diagnostic_x.detach()
            self.last_activation_valid_mask_for_oer = (
                diagnostic_valid_mask.detach() if diagnostic_valid_mask is not None else None
            )
            self.last_activation_layout_info_for_oer = dict(diagnostic_layout_info)

        x_q = self.activation_quant(
            x,
            diagnostic_x=diagnostic_x,
            diagnostic_valid_mask=diagnostic_valid_mask,
            diagnostic_layout_info=diagnostic_layout_info,
        )
        if self.store_oer_activation:
            self.last_quant_activation_for_oer = self._canonicalize_like_diagnostics(
                x_q,
                layout_info=diagnostic_layout_info,
                padding_mask=padding_mask_for_diagnostics,
            ).detach()
        if self.qvf_activation_capture_enabled:
            if diagnostic_layout_info is None:
                raise RuntimeError(f"{self.target_name}: missing diagnostic layout for QVF capture")
            qvf_activation, qvf_valid_mask, qvf_layout_info = self._canonicalize_qvf_activation(
                x_q,
                layout_info=diagnostic_layout_info,
                padding_mask=padding_mask_for_diagnostics,
            )
            self.last_qvf_activation = qvf_activation
            self.last_qvf_valid_mask = qvf_valid_mask
            self.last_qvf_layout_info = qvf_layout_info

        w_q = self.weight_quant(self.linear.weight)
        y = F.linear(x_q, w_q, self.linear.bias)
        if self.store_oer_activation:
            if self.fake_quant_enabled:
                with torch.no_grad():
                    bias = self.linear.bias.detach() if self.linear.bias is not None else None
                    fp32_projection = F.linear(
                        x.detach(),
                        self.linear.weight.detach(),
                        bias,
                    )
            else:
                fp32_projection = y.detach()
            self.last_fp32_projection_for_oer = self._canonicalize_like_diagnostics(
                fp32_projection,
                layout_info=diagnostic_layout_info,
                padding_mask=padding_mask_for_diagnostics,
            ).detach()
            self.last_quant_projection_for_oer = self._canonicalize_like_diagnostics(
                y,
                layout_info=diagnostic_layout_info,
                padding_mask=padding_mask_for_diagnostics,
            ).detach()
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, target_name={self.target_name!r}, "
            f"weight_quant_enabled={self.weight_quant_enabled}, "
            f"activation_quant_enabled={self.activation_quant_enabled}"
        )


__all__ = ["QuantLinear"]







