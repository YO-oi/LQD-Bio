from __future__ import annotations
import csv
import hashlib
import os
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
try:
    from src.data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from src.metrics.common_metrics import DEFAULT_EPS, scalar_oer_topk_index_metadata, scalar_oer_topk_indices
    from src.metrics.geometry_metrics import compute_group_margin
    from src.quant.ffn_quant_injector import restore_clean_model_from_checkpoint
    from src.utils.activation_hook import get_target_ffn_module
    from src.utils.group_utils import ANALYSIS_GROUPS, analysis_group_mask, group_protocol_metadata, read_canonical_bucket, tail_included
    from src.utils.layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
except ImportError:  # pragma: no cover - support package-relative debugging.
    from ..data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from ..metrics.common_metrics import DEFAULT_EPS, scalar_oer_topk_index_metadata, scalar_oer_topk_indices
    from ..metrics.geometry_metrics import compute_group_margin
    from ..quant.ffn_quant_injector import restore_clean_model_from_checkpoint
    from .activation_hook import get_target_ffn_module
    from .group_utils import ANALYSIS_GROUPS, analysis_group_mask, group_protocol_metadata, read_canonical_bucket, tail_included
    from .layout_utils import canonicalize_activation_btc, layout_info_to_public_fields
CACHE_VERSION = 4
CACHE_FILENAME = "fp32_reference_cache.pt"
DEFAULT_TARGET_MODULE_NAMES = ("fc1", "fc2")
DEFAULT_GROUP_ORDER = ANALYSIS_GROUPS
OER_FEATURE_LAYOUT = "B,T,C"
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
def _nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default
def _reference_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FFN_FP32_REFERENCE_CACHE", {})
    return value if isinstance(value, Mapping) else {}
def _heatmap_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FFN_LAYER_HEATMAP", {})
    return value if isinstance(value, Mapping) else {}
def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}
def _metrics_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "METRICS", {})
    return value if isinstance(value, Mapping) else {}
def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    ref_cfg = _reference_cfg(cfg)
    if key in ref_cfg:
        return ref_cfg[key]
    heatmap_cfg = _heatmap_cfg(cfg)
    if key in heatmap_cfg:
        return heatmap_cfg[key]
    return _cfg_get(cfg, key, default)
def _dataset_name(cfg: Any) -> str:
    dataset_cfg = _dataset_cfg(cfg)
    for key in ("NAME", "name", "DATASET_NAME", "dataset_name"):
        value = _nested_get(dataset_cfg, key, None)
        if value:
            return str(value)
    active_dataset = _cfg_get(cfg, "ACTIVE_DATASET", None)
    if active_dataset:
        return str(active_dataset)
    dataset_root = _nested_get(dataset_cfg, "ROOT", None)
    if dataset_root is not None:
        return Path(dataset_root).name
    current_result_dir = _cfg_get(cfg, "CURRENT_RESULT_DIR", None)
    if current_result_dir is not None:
        return Path(current_result_dir).name
    raise AttributeError("Cannot determine dataset name: set ACTIVE_DATASET or CURRENT_DATASET['ROOT'].")
def _reference_cache_dir(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "ffn_layer_heatmap" / "reference_cache"
def _default_cache_path(cfg: Any) -> Path:
    configured = _setting_get(cfg, "CACHE_PATH", None)
    if configured is not None:
        return Path(configured)
    return _reference_cache_dir(cfg) / CACHE_FILENAME
def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", None)
    if configured is None:
        configured = _cfg_get(cfg, "DEVICE", None)
    if configured is None:
        raise AttributeError("ffn_fp32_reference_cache requires DEVICE from config.py.")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py DEVICE={configured!r}, but CUDA is not available.")
    return device
def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))
def _batch_size(cfg: Any) -> int:
    return int(_setting_get(cfg, "EVAL_BATCH_SIZE", _setting_get(cfg, "BATCH_SIZE", 4)))
def _num_workers(cfg: Any) -> int:
    return int(_setting_get(cfg, "NUM_WORKERS", 0))
def _pin_memory(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "PIN_MEMORY", _device(cfg).type == "cuda"))
def _oer_top_ratio(cfg: Any) -> float:
    return float(
        _setting_get(
            cfg,
            "SCALAR_OER_TOP_RATIO",
            _metrics_cfg(cfg).get("SCALAR_OER_TOP_RATIO", _cfg_get(cfg, "OER_SCALAR_TOP_RATIO", 0.001)),
        )
    )
def _make_audio_config(cfg: Any) -> AudioInputConfig:
    audio = _cfg_get(cfg, "AUDIO", {})
    if not isinstance(audio, Mapping):
        audio = {}
    return AudioInputConfig(
        sample_rate=int(audio.get("TARGET_SR", 16000)),
        target_seconds=float(audio.get("TARGET_SECONDS", 10.0)),
        normalization=str(audio.get("NORMALIZATION", "peak")),
        eps=float(audio.get("EPS", 1e-8)),
        deterministic_crop=str(audio.get("CROP_MODE", "center")),  # type: ignore[arg-type]
        fixed_crop_start=int(audio.get("FIXED_CROP_START", 0)),
    )
def _eval_csv(cfg: Any) -> Path:
    for key in ("EVAL_CSV", "REFERENCE_EVAL_CSV"):
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
    dataset_value = _nested_get(_dataset_cfg(cfg), "EVAL_CSV", None)
    if dataset_value is not None:
        return Path(dataset_value)
    raise AttributeError("Cannot find fixed evaluation EVAL_CSV.")
def _normal_manifest_path(value: Any) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())
def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _checkpoint_path_for_manifest(cfg: Any) -> Path | None:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    if isinstance(beats_cfg, Mapping):
        for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT"):
            value = beats_cfg.get(key)
            if value is not None:
                return Path(value)
    for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT"):
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return Path(value)
    return None
def _target_modules_manifest(target_modules: Sequence[tuple[int, str]] | None) -> list[tuple[int, str]] | None:
    if target_modules is None:
        return None
    result: list[tuple[int, str]] = []
    for layer_idx, module_name in target_modules:
        result.append((int(layer_idx), str(module_name)))
    return result
def _coerce_target_modules_manifest(value: Any) -> list[tuple[int, str]] | None:
    if value is None:
        return None
    result: list[tuple[int, str]] = []
    for item in value:
        layer_idx, module_name = item
        result.append((int(layer_idx), str(module_name)))
    return result
def _pooling_policy(cfg: Any) -> dict[str, Any]:
    return {
        "require_mask_aware_pooling": bool(_setting_get(cfg, "REQUIRE_MASK_AWARE_POOLING", False)),
        "sequence_pooling": "mask_aware_mean_or_explicit_unmasked_fallback",
    }
def _reference_manifest(
    cfg: Any,
    target_modules: Sequence[tuple[int, str]] | None,
    *,
    embedding_pooling_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = group_protocol_metadata(
        tail_rule="tail is derived by analysis_group_mask from raw sample rows",
    )
    eval_csv_path = _eval_csv(cfg)
    checkpoint_path = _checkpoint_path_for_manifest(cfg)
    manifest: dict[str, Any] = {
        "eval_csv": _normal_manifest_path(eval_csv_path),
        "checkpoint_path": _normal_manifest_path(checkpoint_path),
        "hash_algorithm": "sha256",
        "eval_csv_sha256": _sha256_file(eval_csv_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "dataset_name": _dataset_name(cfg),
        "oer_top_ratio": float(_oer_top_ratio(cfg)),
        "oer_feature_layout": OER_FEATURE_LAYOUT,
        "oer_activation_kind": "input_activation",
        "oer_oracle_source": "linear_forward_pre_hook_input_topk",
        "oer_projection_output_included": False,
        "target_modules": _target_modules_manifest(target_modules),
        "pooling_policy": _pooling_policy(cfg),
        "contains_full_oer_activation": False,
        **protocol,
        "grouping_protocol": dict(protocol),
    }
    if embedding_pooling_summary is not None:
        manifest["embedding_pooling_summary"] = dict(embedding_pooling_summary)
    return manifest
def _validate_reference_cache_manifest(
    cache: Mapping[str, Any],
    cfg: Any,
    *,
    target_modules: Sequence[tuple[int, str]] | None,
    model: nn.Module | None,
) -> None:
    manifest = cache.get("manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("FP32 reference cache is missing manifest; refusing reuse. Set force_rebuild=True to rebuild.")
    expected_targets = target_modules
    if expected_targets is None and model is not None:
        expected_targets = _normalize_target_modules(model, None)
    expected = _reference_manifest(cfg, expected_targets)
    mismatches: list[str] = []
    for key in (
        "eval_csv",
        "checkpoint_path",
        "hash_algorithm",
        "eval_csv_sha256",
        "checkpoint_sha256",
        "dataset_name",
        "oer_feature_layout",
        "contains_full_oer_activation",
    ):
        if manifest.get(key) != expected.get(key):
            mismatches.append(f"{key}: cache={manifest.get(key)!r}, current={expected.get(key)!r}")
    cached_top_ratio = manifest.get("oer_top_ratio")
    if cached_top_ratio is None or abs(float(cached_top_ratio) - float(expected["oer_top_ratio"])) > 1e-12:
        mismatches.append(f"oer_top_ratio: cache={cached_top_ratio!r}, current={expected['oer_top_ratio']!r}")
    cached_targets = _coerce_target_modules_manifest(manifest.get("target_modules", cache.get("target_modules")))
    expected_target_manifest = expected.get("target_modules")
    if expected_target_manifest is not None and cached_targets != expected_target_manifest:
        mismatches.append(f"target_modules: cache={cached_targets!r}, current={expected_target_manifest!r}")
    if manifest.get("pooling_policy") != expected.get("pooling_policy"):
        mismatches.append(f"pooling_policy: cache={manifest.get('pooling_policy')!r}, current={expected.get('pooling_policy')!r}")
    pooling_summary = manifest.get("embedding_pooling_summary")
    if not isinstance(pooling_summary, Mapping):
        mismatches.append("embedding_pooling_summary: cache manifest 缂哄け")
    elif bool(expected["pooling_policy"].get("require_mask_aware_pooling")) and bool(pooling_summary.get("any_unmasked_mean_fallback", False)):
        mismatches.append("embedding_pooling_summary: current run requires mask-aware pooling, but cache used unmasked fallback")
    oracle_cache = cache.get("oer_oracle_cache")
    if not isinstance(oracle_cache, Mapping):
        mismatches.append("oer_oracle_cache: cache is missing or has the wrong type")
    else:
        if expected_target_manifest is not None:
            cached_oracle_targets = _coerce_target_modules_manifest(oracle_cache.keys())
            if cached_oracle_targets != expected_target_manifest:
                mismatches.append(f"oer_oracle_cache keys: cache={cached_oracle_targets!r}, current={expected_target_manifest!r}")
        for key, oracle in oracle_cache.items():
            if not isinstance(oracle, Mapping):
                mismatches.append(f"oer_oracle_cache[{key!r}]: 绫诲瀷閿欒")
                continue
            if oracle.get("layout") != OER_FEATURE_LAYOUT:
                mismatches.append(f"oer_oracle_cache[{key!r}].layout: cache={oracle.get('layout')!r}, current={OER_FEATURE_LAYOUT!r}")
            oracle_top_ratio = oracle.get("topk_ratio")
            if oracle_top_ratio is None or abs(float(oracle_top_ratio) - float(expected["oer_top_ratio"])) > 1e-12:
                mismatches.append(f"oer_oracle_cache[{key!r}].topk_ratio: cache={oracle_top_ratio!r}, current={expected['oer_top_ratio']!r}")
    if mismatches:
        detail = "; ".join(mismatches)
        raise RuntimeError(
            "FP32 reference cache manifest does not match the current experiment; refusing reuse. "
            f"Set force_rebuild=True to rebuild. Differences: {detail}"
        )
def _build_eval_loader(cfg: Any) -> DataLoader:
    override = _cfg_get(cfg, "build_fp32_reference_eval_loader", None)
    if callable(override):
        return override(cfg)
    override = _cfg_get(cfg, "build_eval_loader", None)
    if callable(override):
        return override(cfg)
    dataset = AudioInputDataset(
        _eval_csv(cfg),
        mode=str(_setting_get(cfg, "MODE", "eval")),
        dataset_root=_nested_get(_dataset_cfg(cfg), "ROOT", None),
        project_root=_cfg_get(cfg, "BASE_DIR", None),
        audio_config=_make_audio_config(cfg),
        global_seed=_seed(cfg),
        epoch=0,
        return_dict=True,
        require_label_id=True,
        include_audio_stats=False,
    )
    return DataLoader(
        dataset,
        batch_size=_batch_size(cfg),
        shuffle=False,
        num_workers=_num_workers(cfg),
        pin_memory=_pin_memory(cfg),
        collate_fn=partial(audio_input_collate_fn, include_audio_stats=False),
    )
def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved
def _extract_tensor_and_mask(output: Any) -> tuple[Tensor, Tensor | None]:
    if isinstance(output, Tensor):
        return output, None
    if isinstance(output, Mapping):
        tensor = None
        for key in ("embedding", "embeddings", "features", "feature", "x", "logits", "lprobs"):
            value = output.get(key)
            if isinstance(value, Tensor):
                tensor = value
                break
        if tensor is None:
            raise TypeError("妯″瀷杈撳嚭 dict 涓病鏈夊彲鐢ㄧ殑 tensor 瀛楁")
        mask = output.get("padding_mask")
        return tensor, mask if isinstance(mask, Tensor) else None
    if isinstance(output, (tuple, list)) and output:
        tensor = output[0]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"妯″瀷杈撳嚭绗竴涓厓绱犲繀椤绘槸 Tensor锛屽綋鍓嶄负 {type(tensor)!r}")
        mask = output[1] if len(output) > 1 and isinstance(output[1], Tensor) else None
        return tensor, mask
    raise TypeError(f"涓嶆敮鎸佺殑妯″瀷杈撳嚭绫诲瀷: {type(output)!r}")
def _forward_features(model: nn.Module, batch: Mapping[str, Any], cfg: Any) -> tuple[Tensor, Tensor | None]:
    for name in ("ffn_fp32_reference_forward", "ffn_heatmap_forward"):
        forward_fn = _cfg_get(cfg, name, None)
        if callable(forward_fn):
            return _extract_tensor_and_mask(forward_fn(model, batch, cfg))
    waveform = batch["waveform"]
    padding_mask = batch.get("padding_mask")
    if hasattr(model, "extract_features"):
        return _extract_tensor_and_mask(model.extract_features(waveform, padding_mask=padding_mask))
    try:
        return _extract_tensor_and_mask(model(waveform, padding_mask=padding_mask))
    except TypeError:
        return _extract_tensor_and_mask(model(waveform))
def _pool_embedding(
    features: Tensor,
    padding_mask: Tensor | None,
    *,
    require_mask_aware: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    """Generate final embeddings and record whether padding masks were used."""
    info: dict[str, Any] = {
        "feature_shape": tuple(int(item) for item in features.shape),
        "padding_mask_shape": None if padding_mask is None else tuple(int(item) for item in padding_mask.shape),
        "mask_used": False,
        "mask_source": "none",
        "method": "unknown",
        "fallback_to_unmasked_mean": False,
    }
    if features.ndim == 2:
        info["method"] = "identity_2d"
        return features.detach(), info
    if features.ndim == 3:
        if padding_mask is not None and tuple(padding_mask.shape) == tuple(features.shape[:2]):
            mask = padding_mask.to(device=features.device, dtype=torch.bool)
            valid = (~mask).unsqueeze(-1)
            summed = features.masked_fill(~valid, 0.0).sum(dim=1)
            valid_tokens = valid.sum(dim=1).clamp_min(1)
            denom = valid_tokens.to(dtype=features.dtype)
            info.update(
                {
                    "method": "mask_aware_mean",
                    "mask_used": True,
                    "mask_source": "forward_output_padding_mask",
                    "valid_tokens_min": int(valid_tokens.min().item()),
                    "valid_tokens_max": int(valid_tokens.max().item()),
                    "padding_tokens_total": int(mask.sum().item()),
                }
            )
            return (summed / denom).detach(), info
        if require_mask_aware:
            raise RuntimeError(
                "FP32 reference embedding pooling requires mask-aware mean, but "
                f"padding_mask shape={info['padding_mask_shape']} does not match features shape={info['feature_shape']}"
            )
        info.update(
            {
                "method": "unmasked_mean_fallback",
                "mask_source": "missing" if padding_mask is None else "shape_mismatch",
                "fallback_to_unmasked_mean": True,
            }
        )
        return features.mean(dim=1).detach(), info
    if features.ndim > 3:
        info["method"] = "flatten_nonsequence"
        return features.flatten(start_dim=1).detach(), info
    info["method"] = "reshape_1d_or_scalar"
    return features.reshape(features.shape[0], -1).detach(), info
def _pooling_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    method_counts: dict[str, int] = {}
    mask_source_counts: dict[str, int] = {}
    for record in records:
        method = str(record.get("method", "unknown"))
        source = str(record.get("mask_source", "unknown"))
        method_counts[method] = method_counts.get(method, 0) + 1
        mask_source_counts[source] = mask_source_counts.get(source, 0) + 1
    return {
        "num_batches": int(len(records)),
        "method_counts": method_counts,
        "mask_source_counts": mask_source_counts,
        "mask_used_batches": int(sum(1 for item in records if bool(item.get("mask_used", False)))),
        "fallback_batches": int(sum(1 for item in records if bool(item.get("fallback_to_unmasked_mean", False)))),
        "all_mask_aware_mean": bool(records) and all(str(item.get("method")) == "mask_aware_mean" for item in records),
        "any_unmasked_mean_fallback": any(bool(item.get("fallback_to_unmasked_mean", False)) for item in records),
    }
def _autocast_context(cfg: Any, device: torch.device) -> Any:
    use_amp = bool(_nested_get(_cfg_get(cfg, "BEATS", {}), "USE_AMP", False))
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()
def _get_encoder_layers(model: nn.Module) -> nn.ModuleList:
    if not hasattr(model, "encoder"):
        raise AttributeError("model is missing encoder; cannot enumerate BEATs Transformer FFN modules.")
    encoder = model.encoder
    if not hasattr(encoder, "layers"):
        raise AttributeError("model.encoder is missing layers; cannot enumerate BEATs Transformer FFN modules.")
    layers = encoder.layers
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"model.encoder.layers must be nn.ModuleList, got {type(layers)!r}")
    return layers
def _all_target_modules(model: nn.Module) -> list[tuple[int, str]]:
    layers = _get_encoder_layers(model)
    targets: list[tuple[int, str]] = []
    for layer_idx, layer in enumerate(layers):
        for module_name in DEFAULT_TARGET_MODULE_NAMES:
            if not hasattr(layer, module_name):
                raise AttributeError(f"encoder.layers.{layer_idx}.{module_name} does not exist")
            targets.append((int(layer_idx), module_name))
    if not targets:
        raise ValueError("target_modules 涓嶈兘涓虹┖")
    return targets
def _normalize_target_modules(model: nn.Module, target_modules: Sequence[tuple[int, str]] | None) -> list[tuple[int, str]]:
    if target_modules is None:
        return _all_target_modules(model)
    normalized: list[tuple[int, str]] = []
    for item in target_modules:
        try:
            layer_idx, module_name = item
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target_modules entries must be (layer_idx, module_name), got {item!r}") from exc
        module_name = str(module_name)
        if module_name not in DEFAULT_TARGET_MODULE_NAMES:
            raise ValueError(f"module_name must be fc1 or fc2, got {module_name!r}")
        get_target_ffn_module(model, int(layer_idx), module_name)
        normalized.append((int(layer_idx), module_name))
    if not normalized:
        raise ValueError("target_modules 涓嶈兘涓虹┖")
    return normalized
def _activation_to_btc(activation: Tensor, batch_size: int, target_name: str) -> Tensor:
    return canonicalize_activation_btc(
        activation.detach().float().cpu(),
        batch_size=int(batch_size),
        context=target_name,
    ).tensor


def _activation_to_btc_with_layout(activation: Tensor, batch_size: int, target_name: str):
    return canonicalize_activation_btc(
        activation.detach().float().cpu(),
        batch_size=int(batch_size),
        context=target_name,
    )
def _batch_sample_ids(batch: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in batch["sample_id"]]
def _batch_labels(batch: Mapping[str, Any]) -> Tensor:
    value = batch["label_id"]
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.long)
    return value.detach().cpu().long().reshape(-1)
def _metadata_value_present(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}
def _parse_bool_metadata(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{context}: tail_included must be bool-like, got {value!r}")
def _sample_id_from_metadata_row(row: Mapping[str, Any]) -> str | None:
    for key in ("sample_id", "audio_id", "file_id"):
        value = row.get(key)
        if _metadata_value_present(value):
            return str(value).strip()
    return None
def _full_dataset_csv(cfg: Any) -> Path | None:
    dataset_cfg = _dataset_cfg(cfg)
    for key in ("FULL_CSV", "FULL_DATASET_CSV"):
        value = _setting_get(cfg, key, None)
        if value is not None:
            return Path(value)
        dataset_value = _nested_get(dataset_cfg, key, None)
        if dataset_value is not None:
            return Path(dataset_value)
    return None
def _metadata_csv_paths(cfg: Any) -> list[Path]:
    paths: list[Path] = [_eval_csv(cfg)]
    full_csv = _full_dataset_csv(cfg)
    if full_csv is not None:
        paths.append(full_csv)
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(Path(path))
        if key not in seen:
            seen.add(key)
            result.append(Path(path))
    return result
def _read_eval_metadata(cfg: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in _metadata_csv_paths(cfg):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader, start=2):
                sample_id = _sample_id_from_metadata_row(row)
                if sample_id is None:
                    continue
                previous = result.get(sample_id)
                if previous is not None and _metadata_value_present(previous.get("bucket")) and _metadata_value_present(row.get("bucket")):
                    prev_bucket = read_canonical_bucket(
                        bucket=[previous.get("bucket")],
                        shot_group=[previous.get("shot_group")] if _metadata_value_present(previous.get("shot_group")) else None,
                        train_label_n=[previous.get("train_label_n")] if _metadata_value_present(previous.get("train_label_n")) else None,
                        context=f"metadata sample_id={sample_id!r}",
                        allow_shot_group_fallback=False,
                    ).bucket[0]
                    next_bucket = read_canonical_bucket(
                        bucket=[row.get("bucket")],
                        shot_group=[row.get("shot_group")] if _metadata_value_present(row.get("shot_group")) else None,
                        train_label_n=[row.get("train_label_n")] if _metadata_value_present(row.get("train_label_n")) else None,
                        context=f"{path}:{row_idx}",
                        allow_shot_group_fallback=False,
                    ).bucket[0]
                    if prev_bucket != next_bucket:
                        raise ValueError(f"Conflicting bucket metadata for sample_id={sample_id!r}: {prev_bucket!r} vs {next_bucket!r}")
                result[sample_id] = dict(row)
    return result
def _metadata_for_samples(sample_ids: Sequence[str], labels: Tensor, cfg: Any) -> dict[str, Any]:
    del labels
    eval_metadata = _read_eval_metadata(cfg)
    buckets: list[str] = []
    tail_flags: list[bool] = []
    audio_id: list[str] = []
    label_name: list[str] = []
    for sample_idx, sample_id in enumerate(sample_ids):
        row = eval_metadata.get(str(sample_id), {})
        has_bucket = _metadata_value_present(row.get("bucket"))
        has_shot_group = _metadata_value_present(row.get("shot_group"))
        has_train_label_n = _metadata_value_present(row.get("train_label_n"))
        if not has_bucket:
            raise ValueError(
                "FP32 reference cache requires sample-level bucket metadata; "
                f"missing bucket for sample_id={sample_id!r} at eval index {sample_idx}"
            )
        result = read_canonical_bucket(
            bucket=[row.get("bucket")],
            shot_group=[row.get("shot_group")] if has_shot_group else None,
            train_label_n=[row.get("train_label_n")] if has_train_label_n else None,
            context=f"metadata sample_id={sample_id!r}",
            allow_shot_group_fallback=False,
        )
        bucket = result.bucket[0]
        computed_tail = bool(tail_included([bucket])[0])
        if _metadata_value_present(row.get("tail_included")):
            observed_tail = _parse_bool_metadata(row.get("tail_included"), context=f"metadata sample_id={sample_id!r}")
            if observed_tail != computed_tail:
                raise ValueError(f"metadata sample_id={sample_id!r}: tail_included does not match bucket={bucket!r}")
        buckets.append(bucket)
        tail_flags.append(computed_tail)
        audio_value = None
        for key in ("audio_id", "file_id", "file_hash", "file_name", "rel_path", "filepath"):
            if row.get(key):
                audio_value = row[key]
                break
        audio_id.append(str(audio_value if audio_value is not None else sample_id))
        label_value = None
        for key in ("label", "ebird_code", "class_name"):
            if row.get(key):
                label_value = row[key]
                break
        label_name.append("" if label_value is None else str(label_value))
    return {
        "bucket": buckets,
        "shot_group": list(buckets),
        "tail_included": tail_flags,
        "audio_id": audio_id,
        "label_name": label_name,
    }
def _group_label_ids(labels: Tensor, bucket: Sequence[str]) -> dict[str, Tensor]:
    labels_cpu = labels.detach().cpu().long().reshape(-1)
    result: dict[str, Tensor] = {}
    for group in DEFAULT_GROUP_ORDER:
        mask = torch.as_tensor(list(analysis_group_mask(bucket, group)), dtype=torch.bool)
        if int(mask.numel()) != int(labels_cpu.numel()):
            raise ValueError(f"analysis_group_mask length must equal label count for {group!r}")
        selected = labels_cpu[mask]
        result[str(group)] = torch.tensor(sorted({int(item) for item in selected.tolist()}), dtype=torch.long)
    return result
def _full_eval_class_geometry(embeddings: Tensor, labels: Tensor) -> dict[str, Any]:
    samples = embeddings.detach().cpu().float().reshape(int(embeddings.shape[0]), -1)
    labels_cpu = labels.detach().cpu().long().reshape(-1)
    if int(samples.shape[0]) != int(labels_cpu.numel()):
        raise ValueError(f"embedding count differs from label_id count: {int(samples.shape[0])} vs {int(labels_cpu.numel())}")
    class_ids = torch.unique(labels_cpu, sorted=True)
    global_mean = samples.mean(dim=0)
    centers: dict[int, Tensor] = {}
    counts: dict[int, int] = {}
    within_scatter: dict[int, Tensor] = {}
    for class_id in class_ids.tolist():
        cls = int(class_id)
        part = samples[labels_cpu == cls]
        center = part.mean(dim=0)
        centers[cls] = center
        counts[cls] = int(part.shape[0])
        within_scatter[cls] = (part - center).square().sum()
    return {
        "samples": samples,
        "labels": labels_cpu,
        "class_ids": class_ids,
        "global_mean": global_mean,
        "centers": centers,
        "counts": counts,
        "within_scatter": within_scatter,
    }
def _safe_group_fisher(
    geometry: Mapping[str, Any],
    group_labels: Tensor,
    *,
    normalize_by_dof: bool = True,
) -> Tensor:
    """Compute group Fisher from the global eval geometry reference."""
    group_labels = group_labels.detach().cpu().long().reshape(-1)
    if int(group_labels.numel()) == 0:
        return torch.tensor(float("nan"))
    centers: Mapping[int, Tensor] = geometry["centers"]
    counts: Mapping[int, int] = geometry["counts"]
    within_scatter: Mapping[int, Tensor] = geometry["within_scatter"]
    global_mean: Tensor = geometry["global_mean"]
    between = global_mean.new_tensor(0.0)
    within = global_mean.new_tensor(0.0)
    num_classes = 0
    num_samples = 0
    for raw_class_id in group_labels.tolist():
        class_id = int(raw_class_id)
        if class_id not in centers:
            continue
        count = int(counts[class_id])
        center = centers[class_id]
        between = between + float(count) * (center - global_mean).square().sum()
        within = within + within_scatter[class_id]
        num_classes += 1
        num_samples += count
    if num_classes < 2 or num_samples <= num_classes:
        return torch.tensor(float("nan"))
    if normalize_by_dof:
        between = between / float(num_classes - 1)
        within = within / float(num_samples - num_classes)
    return (between / torch.clamp(within, min=float(DEFAULT_EPS))).detach().cpu()
def _safe_group_margin(embeddings: Tensor, labels: Tensor, group_labels: Tensor) -> Tensor:
    if int(group_labels.numel()) == 0:
        return torch.tensor(float("nan"))
    return compute_group_margin(embeddings, labels, group_labels, reduction="mean").detach().cpu()
def compute_fp32_group_geometry_cache(embedding_cache: Mapping[str, Any], cfg: Any | None = None) -> dict[str, Any]:
    """Compute FP32 geometry baselines for many/medium/few/tail analysis groups."""
    cfg = {} if cfg is None else cfg
    embeddings = embedding_cache["embedding"].detach().cpu().float()
    labels = embedding_cache["label_id"].detach().cpu().long()
    bucket_values = embedding_cache.get("bucket")
    if bucket_values is None:
        raise KeyError("embedding_cache is missing sample-level bucket metadata")
    shot_values = embedding_cache.get("shot_group")
    buckets = read_canonical_bucket(
        bucket=bucket_values,
        shot_group=shot_values,
        context="fp32 geometry cache",
        allow_shot_group_fallback=False,
    ).bucket
    if len(buckets) != int(labels.numel()):
        raise ValueError(f"bucket length must equal label count, got {len(buckets)} vs {int(labels.numel())}")
    group_labels = _group_label_ids(labels, buckets)
    full_geometry = _full_eval_class_geometry(embeddings, labels)
    normalize_by_dof = bool(_metrics_cfg(cfg).get("FISHER_NORMALIZE_BY_DOF", _cfg_get(cfg, "FISHER_NORMALIZE_BY_DOF", True)))
    group_metrics: dict[str, dict[str, Any]] = {}
    for group in DEFAULT_GROUP_ORDER:
        ids = group_labels[str(group)]
        sample_mask = torch.as_tensor(list(analysis_group_mask(buckets, group)), dtype=torch.bool)
        group_metrics[str(group)] = {
            "label_ids": ids,
            "num_labels": int(ids.numel()),
            "num_samples": int(sample_mask.sum().item()),
            "margin": _safe_group_margin(embeddings, labels, ids),
            "fisher": _safe_group_fisher(full_geometry, ids, normalize_by_dof=normalize_by_dof),
            "fisher_reference": "full_eval_global_mean_and_class_centers",
        }
    result: dict[str, Any] = {
        "group_order": tuple(DEFAULT_GROUP_ORDER),
        "fisher_reference": "full_eval_global_mean_and_class_centers",
        "fisher_normalize_by_dof": bool(normalize_by_dof),
        "full_eval_num_samples": int(labels.numel()),
        "full_eval_num_classes": int(full_geometry["class_ids"].numel()),
        "groups": group_metrics,
    }
    for group, metrics in group_metrics.items():
        result[f"{group}_margin_fp32"] = metrics["margin"]
        result[f"{group}_fisher_fp32"] = metrics["fisher"]
    return result
def _oracle_chunk(activation: Tensor, sample_ids: Sequence[str], layer_idx: int, module_name: str, topk_ratio: float) -> dict[str, Any]:
    batch_size = len(sample_ids)
    canonical = _activation_to_btc_with_layout(activation, batch_size, f"L{int(layer_idx)}.{module_name}.input")
    x = canonical.tensor
    public_layout = layout_info_to_public_fields(canonical.layout_info)
    topk_indices = scalar_oer_topk_indices(x, topk_ratio=topk_ratio, batch_dim=0).detach().cpu().long()
    flat = x.reshape(batch_size, -1)
    selected = torch.gather(flat, dim=1, index=topk_indices)
    metadata = scalar_oer_topk_index_metadata(
        x,
        sample_id=sample_ids,
        layer_idx=int(layer_idx),
        module_name=module_name,
        topk_indices=topk_indices,
        batch_dim=0,
        token_dim=1,
        channel_dim=2,
    )
    return {
        "topk_indices": topk_indices,
        "token_idx": metadata["token_idx"].detach().cpu().long(),
        "channel_idx": metadata["channel_idx"].detach().cpu().long(),
        "sample_index": metadata["sample_index"].detach().cpu().long(),
        "sample_id": list(metadata["sample_id"]),
        "fp32_oracle_energy": selected.square().sum(dim=1).detach().cpu(),
        "fp32_total_energy": flat.square().sum(dim=1).detach().cpu(),
        "activation_shape": tuple(int(item) for item in x.shape),
        "original_layout": public_layout["original_layout"],
        "original_shape": public_layout["original_shape"],
        "canonical_layout": public_layout["canonical_layout"],
        "canonical_shape": public_layout["canonical_shape"],
        "layout_inference_source": public_layout["layout_inference_source"],
        "padding_mask_available": public_layout["padding_mask_available"],
        "padding_mask_aligned": public_layout["padding_mask_aligned"],
        "mask_status": public_layout["mask_status"],
        "topk_ratio": float(topk_ratio),
    }
def _merge_oracle_chunks(chunks: Sequence[Mapping[str, Any]], layer_idx: int, module_name: str) -> dict[str, Any]:
    if not chunks:
        raise RuntimeError(f"L{int(layer_idx)}.{module_name}: no FP32 input activation was captured")
    activation_shapes = [tuple(chunk["activation_shape"]) for chunk in chunks]
    nonbatch_shapes = {shape[1:] for shape in activation_shapes}
    if len(nonbatch_shapes) != 1:
        raise RuntimeError(f"L{int(layer_idx)}.{module_name}: OER activation batch dimension is inconsistent: {activation_shapes}")
    sample_id_nested: list[Any] = []
    for chunk in chunks:
        sample_id_nested.extend(list(chunk["sample_id"]))
    topk_indices = torch.cat([chunk["topk_indices"] for chunk in chunks], dim=0)
    token_idx = torch.cat([chunk["token_idx"] for chunk in chunks], dim=0)
    channel_idx = torch.cat([chunk["channel_idx"] for chunk in chunks], dim=0)
    sample_index = torch.arange(int(topk_indices.shape[0]), dtype=torch.long).reshape(-1, 1).expand_as(topk_indices)
    total_batch = int(topk_indices.shape[0])
    first_shape = activation_shapes[0]
    return {
        "layout": OER_FEATURE_LAYOUT,
        "contains_full_activation": False,
        "oracle_source": "linear_forward_pre_hook_input_topk",
        "projection_output_included": False,
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "target_name": f"encoder.layers.{int(layer_idx)}.{module_name}",
        "activation_kind": "input_activation",
        "activation_shape": (total_batch, *first_shape[1:]),
        "original_layout": chunks[0].get("original_layout", ""),
        "original_shape": chunks[0].get("original_shape", ""),
        "canonical_layout": "BTC",
        "canonical_shape": (total_batch, *first_shape[1:]),
        "layout_inference_source": chunks[0].get("layout_inference_source", ""),
        "padding_mask_available": bool(chunks[0].get("padding_mask_available", False)),
        "padding_mask_aligned": bool(chunks[0].get("padding_mask_aligned", False)),
        "mask_status": chunks[0].get("mask_status", ""),
        "topk_ratio": float(chunks[0]["topk_ratio"]),
        "topk_indices": topk_indices,
        "flat_index": topk_indices,
        "sample_index": sample_index,
        "sample_id": sample_id_nested,
        "token_idx": token_idx,
        "channel_idx": channel_idx,
        "batch_dim": 0,
        "token_dim": 1,
        "channel_dim": 2,
        "fp32_oracle_energy": torch.cat([chunk["fp32_oracle_energy"] for chunk in chunks], dim=0),
        "fp32_total_energy": torch.cat([chunk["fp32_total_energy"] for chunk in chunks], dim=0),
    }
def _register_input_oer_hooks(
    model: nn.Module,
    target_modules: Sequence[tuple[int, str]],
    oracle_chunks: dict[tuple[int, str], list[dict[str, Any]]],
    state: dict[str, Any],
    topk_ratio: float,
) -> list[Any]:
    handles: list[Any] = []
    for layer_idx, module_name in target_modules:
        key = (int(layer_idx), str(module_name))
        module = get_target_ffn_module(model, int(layer_idx), str(module_name))
        oracle_chunks[key] = []
        def hook(_module: nn.Module, inputs: tuple[Any, ...], *, capture_key: tuple[int, str] = key) -> None:
            del _module
            sample_ids = state.get("sample_ids")
            batch_counts = state.get("batch_capture_counts")
            if sample_ids is None or batch_counts is None:
                raise RuntimeError("OER hook captured activation without current batch metadata")
            if not inputs or not isinstance(inputs[0], Tensor):
                raise RuntimeError(f"L{capture_key[0]}.{capture_key[1]}: 鏃犳硶鎹曡幏杈撳叆 activation")
            batch_counts[capture_key] = int(batch_counts.get(capture_key, 0)) + 1
            oracle_chunks[capture_key].append(
                _oracle_chunk(inputs[0], sample_ids, capture_key[0], capture_key[1], topk_ratio)
            )
        handles.append(module.register_forward_pre_hook(hook))
    return handles
def _remove_hooks(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()
def _collect_fp32_reference(
    cfg: Any,
    model: nn.Module,
    eval_loader: DataLoader,
    target_modules: Sequence[tuple[int, str]] | None,
    *,
    collect_embeddings: bool,
    collect_oer: bool,
) -> dict[str, Any]:
    device = _device(cfg)
    model.to(device)
    model.eval()
    active_targets = _normalize_target_modules(model, target_modules) if collect_oer else []
    topk_ratio = _oer_top_ratio(cfg)
    sample_ids: list[str] = []
    labels: list[Tensor] = []
    embeddings: list[Tensor] = []
    embedding_pooling_records: list[dict[str, Any]] = []
    batch_sizes: list[int] = []
    oracle_chunks: dict[tuple[int, str], list[dict[str, Any]]] = {}
    hook_state: dict[str, Any] = {"sample_ids": None, "batch_capture_counts": None}
    handles = _register_input_oer_hooks(model, active_targets, oracle_chunks, hook_state, topk_ratio) if collect_oer else []
    try:
        with torch.no_grad():
            for batch in eval_loader:
                batch_index = len(batch_sizes)
                batch_sample_ids = _batch_sample_ids(batch)
                batch_label = _batch_labels(batch)
                if len(batch_sample_ids) != int(batch_label.numel()):
                    raise RuntimeError("sample_id count must equal label_id count")
                batch_sizes.append(int(batch_label.numel()))
                sample_ids.extend(batch_sample_ids)
                labels.append(batch_label)
                hook_state["sample_ids"] = batch_sample_ids
                hook_state["batch_capture_counts"] = {target: 0 for target in active_targets}
                moved = _move_batch_to_device(batch, device)
                with _autocast_context(cfg, device):
                    features, feature_padding_mask = _forward_features(model, moved, cfg)
                if collect_embeddings:
                    pooled, pooling_info = _pool_embedding(
                        features.float(),
                        feature_padding_mask,
                        require_mask_aware=bool(_setting_get(cfg, "REQUIRE_MASK_AWARE_POOLING", False)),
                    )
                    pooling_info["batch_index"] = int(batch_index)
                    pooling_info["sample_id"] = list(batch_sample_ids)
                    embedding_pooling_records.append(pooling_info)
                    embeddings.append(pooled.cpu())
                if collect_oer:
                    missing = [target for target, count in hook_state["batch_capture_counts"].items() if int(count) != 1]
                    if missing:
                        names = ", ".join(f"L{idx}.{name}" for idx, name in missing)
                        raise RuntimeError(
                            f"FP32 reference forward 娌℃湁鍑嗙‘缁忚繃鐩爣 FFN 杈撳叆妯″潡: {names}; "
                            f"褰撳墠 batch sample_id={batch_sample_ids}"
                        )
                hook_state["sample_ids"] = None
                hook_state["batch_capture_counts"] = None
                del moved, features, feature_padding_mask
    finally:
        _remove_hooks(handles)
    if not labels:
        raise RuntimeError("鍥哄畾 eval_loader 娌℃湁浜х敓浠讳綍 batch")
    label_tensor = torch.cat(labels, dim=0)
    metadata = _metadata_for_samples(sample_ids, label_tensor, cfg)
    result: dict[str, Any] = {
        "sample_id": sample_ids,
        "label_id": label_tensor,
        "audio_id": metadata["audio_id"],
        "bucket": metadata["bucket"],
        "shot_group": metadata["shot_group"],
        "tail_included": metadata["tail_included"],
        "label_name": metadata["label_name"],
        "batch_sizes": batch_sizes,
        "target_modules": active_targets,
    }
    if collect_embeddings:
        result["embedding"] = torch.cat(embeddings, dim=0)
        result["embedding_pooling_records"] = embedding_pooling_records
        result["embedding_pooling_summary"] = _pooling_summary(embedding_pooling_records)
    if collect_oer:
        result["oer_oracle"] = {
            key: _merge_oracle_chunks(chunks, key[0], key[1])
            for key, chunks in oracle_chunks.items()
        }
    return result
def build_fp32_embedding_cache(
    cfg: Any,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Load a clean FP32 model and cache final embeddings for the fixed eval set."""
    if model is None:
        model = restore_clean_model_from_checkpoint(cfg)
    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)
    collected = _collect_fp32_reference(
        cfg,
        model,
        eval_loader,
        target_modules=None,
        collect_embeddings=True,
        collect_oer=False,
    )
    return {
        "sample_id": collected["sample_id"],
        "label_id": collected["label_id"],
        "audio_id": collected["audio_id"],
        "bucket": collected["bucket"],
        "shot_group": collected["shot_group"],
        "tail_included": collected["tail_included"],
        "label_name": collected["label_name"],
        "embedding": collected["embedding"],
        "embedding_pooling_records": collected["embedding_pooling_records"],
        "embedding_pooling_summary": collected["embedding_pooling_summary"],
        "batch_sizes": collected["batch_sizes"],
    }
def build_fp32_oer_oracle_cache(
    cfg: Any,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
    target_modules: Sequence[tuple[int, str]] | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Build the OER oracle on target FFN linear-layer input activations."""
    if model is None:
        model = restore_clean_model_from_checkpoint(cfg)
    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)
    collected = _collect_fp32_reference(
        cfg,
        model,
        eval_loader,
        target_modules=target_modules,
        collect_embeddings=False,
        collect_oer=True,
    )
    return collected["oer_oracle"]
def _build_full_reference_cache(
    cfg: Any,
    *,
    model: nn.Module | None,
    eval_loader: DataLoader | None,
    target_modules: Sequence[tuple[int, str]] | None,
) -> dict[str, Any]:
    if model is None:
        model = restore_clean_model_from_checkpoint(cfg)
    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)
    collected = _collect_fp32_reference(
        cfg,
        model,
        eval_loader,
        target_modules=target_modules,
        collect_embeddings=True,
        collect_oer=True,
    )
    embedding_cache = {
        "sample_id": collected["sample_id"],
        "label_id": collected["label_id"],
        "audio_id": collected["audio_id"],
        "bucket": collected["bucket"],
        "shot_group": collected["shot_group"],
        "tail_included": collected["tail_included"],
        "label_name": collected["label_name"],
        "embedding": collected["embedding"],
        "embedding_pooling_records": collected["embedding_pooling_records"],
        "embedding_pooling_summary": collected["embedding_pooling_summary"],
        "batch_sizes": collected["batch_sizes"],
    }
    geometry_cache = compute_fp32_group_geometry_cache(embedding_cache, cfg)
    return {
        "cache_version": CACHE_VERSION,
        "cache_type": "fp32_reference",
        "embedding_cache": embedding_cache,
        "oer_oracle_cache": collected["oer_oracle"],
        "geometry_cache": geometry_cache,
        "target_modules": collected["target_modules"],
        "manifest": _reference_manifest(
            cfg,
            collected["target_modules"],
            embedding_pooling_summary=embedding_cache["embedding_pooling_summary"],
        ),
    }
def load_or_build_fp32_reference_cache(
    cfg: Any,
    *,
    cache_path: Path | str | None = None,
    force_rebuild: bool = False,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
    target_modules: Sequence[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    """Read or build the shared FP32 reference cache for W4/A4/W4A4 causal ablations."""
    path = Path(cache_path) if cache_path is not None else _default_cache_path(cfg)
    if path.exists() and not force_rebuild:
        cache = torch.load(path, map_location="cpu")
        if not isinstance(cache, Mapping):
            raise TypeError(f"FP32 reference cache must be a mapping, got {type(cache)!r}")
        if int(cache.get("cache_version", -1)) != CACHE_VERSION:
            raise RuntimeError(
                f"FP32 reference cache version mismatch: {cache.get('cache_version')} vs {CACHE_VERSION}; "
                "set force_rebuild=True to rebuild"
            )
        _validate_reference_cache_manifest(cache, cfg, target_modules=target_modules, model=model)
        return dict(cache)
    cache = _build_full_reference_cache(
        cfg,
        model=model,
        eval_loader=eval_loader,
        target_modules=target_modules,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(cache, tmp_path)
    os.replace(tmp_path, path)
    cache["cache_path"] = path
    return cache
__all__ = [
    "CACHE_FILENAME",
    "CACHE_VERSION",
    "OER_FEATURE_LAYOUT",
    "build_fp32_embedding_cache",
    "build_fp32_oer_oracle_cache",
    "compute_fp32_group_geometry_cache",
    "load_or_build_fp32_reference_cache",
]

