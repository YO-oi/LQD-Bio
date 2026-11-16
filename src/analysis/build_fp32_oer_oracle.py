from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from src.data.audio_input import AudioInputDataset, audio_input_collate_fn
    from src.export.export_fp32_core_cache import (
        _atomic_json_dump,
        _atomic_torch_save,
        _autocast_context,
        _cfg_get,
        _checkpoint_path_for_manifest,
        _checkpoint_sha256_for_manifest,
        _file_sha256,
        _relative_to_parent,
        _sample_shard_size,
        _sample_order_hash,
        _dataset_cfg,
        _default_cfg,
        _make_audio_config,
        _move_batch_to_device,
        _nested_get,
        _restore_fp32_teacher,
        backbone_features_for_cache,
    )
    from src.metrics.common_metrics import DEFAULT_SCALAR_OER_TOP_RATIO
    from src.utils.activation_hook import ActivationHookManager, get_target_ffn_module
    from src.utils.layout_utils import canonicalize_activation_btc, canonicalize_padding_mask_bt, layout_info_to_public_fields
except ImportError:  # pragma: no cover - script execution fallback.
    from ..data.audio_input import AudioInputDataset, audio_input_collate_fn
    from ..export.export_fp32_core_cache import (
        _atomic_json_dump,
        _atomic_torch_save,
        _autocast_context,
        _cfg_get,
        _checkpoint_path_for_manifest,
        _checkpoint_sha256_for_manifest,
        _file_sha256,
        _relative_to_parent,
        _sample_shard_size,
        _sample_order_hash,
        _dataset_cfg,
        _default_cfg,
        _make_audio_config,
        _move_batch_to_device,
        _nested_get,
        _restore_fp32_teacher,
        backbone_features_for_cache,
    )
    from ..metrics.common_metrics import DEFAULT_SCALAR_OER_TOP_RATIO
    from ..utils.activation_hook import ActivationHookManager, get_target_ffn_module
    from ..utils.layout_utils import canonicalize_activation_btc, canonicalize_padding_mask_bt, layout_info_to_public_fields


ORACLE_VERSION = 3
ORACLE_FILENAME = "fp32_oer_oracle.pt"
MANIFEST_FILENAME = "fp32_oer_oracle.manifest.json"
ORACLE_STORAGE_FORMAT = "layer_sample_sharded_v1"
ORACLE_SHARD_DIR_SUFFIX = "_shards"
NUM_BEATS_LAYERS = 12
FFN_MODULE_NAMES = ("fc1", "fc2")
DEFAULT_ACTIVATION_POSITIONS = ("input", "output")
ALLOWED_ACTIVATION_POSITIONS = {"input", "output"}
TOPK_UNIT = "scalar"
TOPK_SOURCE = "fp32_scalar_energy"
VALID_UNIT = "valid_scalar"
TOPK_K_POLICY = "ceil(topk_ratio * valid_unit_count)"
TOPK_MIN_K = 1
TOPK_MAX_K = None
PADDING_EXCLUDED = True
LAYOUT_PROTOCOL_VERSION = 1
CHECKPOINT_HASH_TYPE = "sha256"
SCALAR_FLATTEN_COORDINATE_SYSTEM = (
    "canonical BTC per sample: scalar_index = token_idx * feature_dim + feature_idx; "
    "token_idx is the absolute canonical token coordinate, feature_idx is in [0, feature_dim), "
    "and padding tokens are excluded from the candidate set before top-k selection"
)


def _oracle_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_OER_ORACLE", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    oracle_cfg = _oracle_cfg(cfg)
    if key in oracle_cfg:
        return oracle_cfg[key]
    metrics_cfg = _cfg_get(cfg, "METRICS", {})
    if isinstance(metrics_cfg, Mapping) and key in metrics_cfg:
        return metrics_cfg[key]
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
    raise AttributeError("missing dataset name; configure ACTIVE_DATASET or CURRENT_DATASET['ROOT']")


def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", _cfg_get(cfg, "DEVICE", None))
    if configured is None:
        raise AttributeError("build_fp32_oer_oracle requires config.py DEVICE")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py DEVICE={configured!r}, but CUDA is not available")
    return device


def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


def _batch_size(cfg: Any) -> int:
    return int(_setting_get(cfg, "EVAL_BATCH_SIZE", _setting_get(cfg, "BATCH_SIZE", 8)))


def _num_workers(cfg: Any) -> int:
    return int(_setting_get(cfg, "NUM_WORKERS", 0))


def _pin_memory(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "PIN_MEMORY", _device(cfg).type == "cuda"))


def _eval_csv(cfg: Any) -> Path:
    value = _setting_get(cfg, "EVAL_CSV", None)
    if value is not None:
        return Path(value)
    dataset_value = _nested_get(_dataset_cfg(cfg), "EVAL_CSV", None)
    if dataset_value is not None:
        return Path(dataset_value)
    raise AttributeError("missing eval csv; configure CURRENT_DATASET['EVAL_CSV'] or FP32_OER_ORACLE['EVAL_CSV']")


def _output_dir(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "fp32_oer_oracle"


def _oracle_path(cfg: Any, output_path: str | Path | None = None) -> Path:
    if output_path is not None:
        return Path(output_path)
    configured = _setting_get(cfg, "CACHE_PATH", None)
    if configured is None:
        configured = _setting_get(cfg, "ORACLE_PATH", None)
    if configured is not None:
        return Path(configured)
    return _output_dir(cfg) / ORACLE_FILENAME


def _manifest_path(oracle_path: str | Path) -> Path:
    path = Path(oracle_path)
    if path.name == ORACLE_FILENAME:
        return path.with_name(MANIFEST_FILENAME)
    return path.with_suffix(path.suffix + ".manifest.json")


def _oracle_shard_dir(oracle_path: str | Path) -> Path:
    path = Path(oracle_path)
    return path.with_name(path.stem + ORACLE_SHARD_DIR_SUFFIX)


def _resolve_oracle_relative_path(oracle_path: str | Path, value: str | Path) -> Path:
    text = str(value).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    path = Path(text)
    if path.is_absolute():
        return path
    oracle_parent = Path(oracle_path).parent
    is_windows_absolute = len(text) >= 3 and text[0].isalpha() and text[1] == ":" and text[2] == "/"
    if text.startswith("local_data/") or is_windows_absolute:
        local_root = next((item for item in (oracle_parent, *oracle_parent.parents) if item.name == "local_data"), None)
        marker = "local_data/"
        if local_root is not None and marker in text:
            return local_root / text.split(marker, 1)[1]
    marker = f"/{oracle_parent.name}/"
    if is_windows_absolute and marker in text:
        return oracle_parent / text.split(marker, 1)[1]
    return oracle_parent / path


def is_sharded_fp32_oer_oracle(oracle: Mapping[str, Any]) -> bool:
    manifest = oracle.get("manifest", {})
    manifest_format = manifest.get("storage_format") if isinstance(manifest, Mapping) else None
    return bool(oracle.get("sharded", False)) or str(oracle.get("storage_format", manifest_format)) == ORACLE_STORAGE_FORMAT


def iter_fp32_oer_oracle_shards(oracle: Mapping[str, Any], oracle_path: str | Path, *, layer_idx: int | None = None):
    if not is_sharded_fp32_oer_oracle(oracle):
        yield dict(oracle)
        return
    shards = oracle.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise ValueError("sharded OER oracle manifest is missing shards")
    for entry in shards:
        if not isinstance(entry, Mapping):
            raise TypeError(f"OER shard entry must be a mapping, got {type(entry)!r}")
        if layer_idx is not None and int(entry.get("layer_idx", -1)) != int(layer_idx):
            continue
        shard_rel = entry.get("path")
        if shard_rel is None:
            raise KeyError("OER shard entry missing path")
        shard_path = _resolve_oracle_relative_path(oracle_path, shard_rel)
        shard = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard, Mapping):
            raise TypeError(f"OER oracle shard must save a mapping, got {type(shard)!r}: {shard_path}")
        yield dict(shard)


def load_fp32_oer_oracle(oracle_path: str | Path, *, materialize: bool = False) -> dict[str, Any]:
    path = Path(oracle_path)
    if not path.exists():
        raise FileNotFoundError(f"FP32 OER oracle does not exist: {path}")
    oracle = torch.load(path, map_location="cpu")
    if not isinstance(oracle, Mapping):
        raise TypeError(f"FP32 OER oracle must be a mapping, got {type(oracle)!r}")
    result = dict(oracle)
    if not materialize or not is_sharded_fp32_oer_oracle(result):
        return result

    columns = _empty_oracle_columns()
    for shard in iter_fp32_oer_oracle_shards(result, path):
        sample_ids = list(shard.get("sample_id", []))
        row_count = len(sample_ids)
        layer_idx_tensor = torch.as_tensor(shard["layer_idx"], dtype=torch.long).reshape(-1)
        energies = torch.as_tensor(shard["fp32_oracle_energy"], dtype=torch.float32).reshape(-1)
        total_energy = torch.as_tensor(shard["fp32_total_energy"], dtype=torch.float32).reshape(-1)
        topk_ratio = torch.as_tensor(shard["topk_ratio"], dtype=torch.float32).reshape(-1)
        valid_count = torch.as_tensor(shard["valid_token_count"], dtype=torch.long).reshape(-1)
        aux_token_values = shard.get("oer_topk_token_indices_aux", shard.get("topk_token_indices"))
        if aux_token_values is None:
            raise KeyError("OER oracle shard missing oer_topk_token_indices_aux")
        topk_k_tensor = torch.as_tensor(shard["topk_k"], dtype=torch.long).reshape(-1) if "topk_k" in shard else None
        aux_topk_k_tensor = torch.as_tensor(shard["oer_aux_token_topk_k"], dtype=torch.long).reshape(-1) if "oer_aux_token_topk_k" in shard else None
        valid_scalar_tensor = torch.as_tensor(shard["valid_scalar_count"], dtype=torch.long).reshape(-1) if "valid_scalar_count" in shard else None
        feature_dim_tensor = torch.as_tensor(shard["feature_dim"], dtype=torch.long).reshape(-1) if "feature_dim" in shard else None
        for idx in range(row_count):
            columns["sample_id"].append(str(sample_ids[idx]))
            columns["layer_idx"].append(int(layer_idx_tensor[idx].item()))
            columns["module_name"].append(str(shard["module_name"][idx]))
            columns["activation_position"].append(str(shard["activation_position"][idx]))
            scalar_values = shard["topk_scalar_indices"][idx]
            aux_values = aux_token_values[idx]
            canonical_shape = tuple(shard["canonical_shape"][idx])
            feature_dim_value = int(feature_dim_tensor[idx].item()) if feature_dim_tensor is not None else int(canonical_shape[-1])
            valid_token_value = int(valid_count[idx].item())
            columns["topk_scalar_indices"].append(scalar_values)
            columns["oer_topk_token_indices_aux"].append(aux_values)
            columns["topk_k"].append(int(topk_k_tensor[idx].item()) if topk_k_tensor is not None else int(torch.as_tensor(scalar_values).numel()))
            columns["oer_aux_token_topk_k"].append(int(aux_topk_k_tensor[idx].item()) if aux_topk_k_tensor is not None else int(torch.as_tensor(aux_values).numel()))
            columns["valid_scalar_count"].append(int(valid_scalar_tensor[idx].item()) if valid_scalar_tensor is not None else int(valid_token_value * feature_dim_value))
            columns["feature_dim"].append(int(feature_dim_value))
            columns["fp32_oracle_energy"].append(float(energies[idx].item()))
            columns["fp32_total_energy"].append(float(total_energy[idx].item()))
            columns["topk_ratio"].append(float(topk_ratio[idx].item()))
            columns["activation_shape"].append(tuple(shard["activation_shape"][idx]))
            columns["original_layout"].append(str(shard["original_layout"][idx]))
            columns["original_shape"].append(tuple(shard["original_shape"][idx]))
            columns["canonical_layout"].append(str(shard["canonical_layout"][idx]))
            columns["canonical_shape"].append(tuple(shard["canonical_shape"][idx]))
            columns["layout_inference_source"].append(str(shard["layout_inference_source"][idx]))
            columns["padding_mask_available"].append(bool(shard["padding_mask_available"][idx]))
            columns["padding_mask_aligned"].append(bool(shard["padding_mask_aligned"][idx]))
            columns["mask_status"].append(str(shard["mask_status"][idx]))
            columns["valid_token_count"].append(int(valid_token_value))
    data = _tensorize_columns(columns)
    result.update(data)
    result["row_index"] = _build_row_index(data["sample_id"], data["layer_idx"], data["module_name"], data["activation_position"])
    result["sample_id_set"] = sorted(set(str(item) for item in data["sample_id"]))
    return result

def _topk_ratio(cfg: Any) -> float:
    ratio = float(_setting_get(cfg, "TOPK_RATIO", _setting_get(cfg, "SCALAR_OER_TOP_RATIO", DEFAULT_SCALAR_OER_TOP_RATIO)))
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"TOPK_RATIO must be in (0, 1], got {ratio}")
    return ratio


def _checkpoint_protocol_fields(cfg: Any) -> dict[str, Any]:
    checkpoint_sha256 = _checkpoint_sha256_for_manifest(cfg)
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hash": checkpoint_sha256,
        "checkpoint_hash_type": CHECKPOINT_HASH_TYPE,
    }


def _scalar_topk_protocol_fields(topk_ratio: float) -> dict[str, Any]:
    return {
        "topk_unit": TOPK_UNIT,
        "topk_source": TOPK_SOURCE,
        "topk_ratio": float(topk_ratio),
        "topk_k": "per_row",
        "topk_k_policy": TOPK_K_POLICY,
        "topk_min_k": TOPK_MIN_K,
        "topk_max_k": TOPK_MAX_K,
        "valid_unit": VALID_UNIT,
        "valid_scalar_count": "per_row",
        "valid_unit_count_field": "valid_scalar_count",
        "feature_dim": "per_row",
        "padding_excluded": PADDING_EXCLUDED,
        "layout_protocol_version": LAYOUT_PROTOCOL_VERSION,
        "scalar_flatten_coordinate_system": SCALAR_FLATTEN_COORDINATE_SYSTEM,
        "auxiliary_token_indices_field": "oer_topk_token_indices_aux",
    }


def _activation_positions(cfg: Any) -> tuple[str, ...]:
    configured = _setting_get(cfg, "ACTIVATION_POSITIONS", DEFAULT_ACTIVATION_POSITIONS)
    if isinstance(configured, str):
        raw_positions = (configured,)
    else:
        raw_positions = tuple(configured)
    positions: list[str] = []
    for item in raw_positions:
        name = str(item).strip().lower()
        if name in {"input_activation", "pre_linear", "linear_input"}:
            name = "input"
        if name in {"output_activation", "post_linear", "linear_output"}:
            name = "output"
        if name not in ALLOWED_ACTIVATION_POSITIONS:
            raise ValueError(f"activation_position must be input/output, got {item!r}")
        if name not in positions:
            positions.append(name)
    if not positions:
        raise ValueError("ACTIVATION_POSITIONS cannot be empty")
    for required_position in DEFAULT_ACTIVATION_POSITIONS:
        if required_position not in positions:
            positions.append(required_position)
    return tuple(positions)


def all_ffn_targets(num_layers: int = NUM_BEATS_LAYERS) -> tuple[tuple[int, str], ...]:
    """FP32 OER oracle helper."""

    return tuple((layer_idx, module_name) for layer_idx in range(int(num_layers)) for module_name in FFN_MODULE_NAMES)


def _build_eval_loader(cfg: Any) -> DataLoader:
    override = _cfg_get(cfg, "build_eval_loader", None)
    if callable(override):
        return override(cfg)
    mode = str(_setting_get(cfg, "AUDIO_MODE", "eval"))
    if mode.lower() == "train":
        raise ValueError("FP32 OER oracle must not use train random crop")
    dataset = AudioInputDataset(
        _eval_csv(cfg),
        mode=mode,
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


def _sample_ids_from_batch(batch: Mapping[str, Any]) -> list[str]:
    if "sample_id" not in batch:
        raise KeyError("eval batch must include sample_id to build a reproducible OER oracle")
    return [str(item) for item in batch["sample_id"]]


def _validate_targets(model: nn.Module, targets: Sequence[tuple[int, str]]) -> None:
    for layer_idx, module_name in targets:
        module = get_target_ffn_module(model, layer_idx, module_name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"encoder.layers.{layer_idx}.{module_name} must be nn.Linear, got {type(module)!r}")


def _activation_to_btc(
    x: Tensor,
    *,
    batch_size: int,
    target_name: str,
    activation_position: str,
    padding_mask: Tensor | None = None,
) -> Tensor:
    return canonicalize_activation_btc(
        x.detach().float(),
        padding_mask=padding_mask,
        batch_size=int(batch_size),
        context=f"{target_name}.{activation_position}",
    ).tensor


def _activation_to_btc_with_layout(
    x: Tensor,
    *,
    batch_size: int,
    target_name: str,
    activation_position: str,
    padding_mask: Tensor | None = None,
):
    return canonicalize_activation_btc(
        x.detach().float(),
        padding_mask=padding_mask,
        batch_size=int(batch_size),
        context=f"{target_name}.{activation_position}",
    )


def _align_token_padding_mask(token_padding_mask: Tensor, *, batch_size: int, num_tokens: int, device: torch.device) -> Tensor:
    return canonicalize_padding_mask_bt(
        token_padding_mask,
        batch_size=int(batch_size),
        token_count=int(num_tokens),
        context="fp32_oer_oracle.padding_mask",
    ).to(device=device)


def _resolve_topk(num_items: int, topk_ratio: float) -> int:
    if int(num_items) <= 0:
        raise ValueError("empty activation cannot build an OER oracle")
    return max(1, min(int(num_items), int(math.ceil(float(num_items) * float(topk_ratio)))))


def _topk_indices_for_single_sample(
    sample_activation: Tensor,
    valid_token_mask: Tensor,
    *,
    topk_ratio: float,
) -> tuple[Tensor, Tensor, int, int, int, int, int, Tensor, Tensor]:
    """Return scalar-level OER top-k plus an auxiliary token aggregation."""

    if sample_activation.ndim != 2:
        raise ValueError(f"single-sample activation must have shape [T,C], got shape={tuple(sample_activation.shape)}")
    if tuple(valid_token_mask.shape) != (int(sample_activation.shape[0]),):
        raise ValueError("valid_token_mask must align with the activation token dimension")
    if not torch.isfinite(sample_activation).all().item():
        raise ValueError("FP32 activation contains NaN/Inf and cannot build an OER oracle")

    valid_positions = torch.nonzero(valid_token_mask.to(dtype=torch.bool), as_tuple=False).flatten()
    valid_token_count = int(valid_positions.numel())
    if valid_token_count <= 0:
        raise ValueError("sample has zero valid tokens and cannot build an OER oracle")

    valid_activation = sample_activation.index_select(0, valid_positions)
    feature_dim = int(valid_activation.shape[1])
    valid_scalar_count = int(valid_activation.numel())
    scalar_k = _resolve_topk(valid_scalar_count, topk_ratio)
    flat_valid = valid_activation.reshape(-1)
    scalar_local = torch.topk(flat_valid.square(), k=scalar_k, largest=True, sorted=False).indices.long()
    scalar_token_local = torch.div(scalar_local, feature_dim, rounding_mode="floor")
    scalar_feature = scalar_local.remainder(feature_dim)
    scalar_token_original = valid_positions.index_select(0, scalar_token_local)
    scalar_flat_original = (scalar_token_original * feature_dim + scalar_feature).long()

    token_energy = valid_activation.square().sum(dim=1)
    token_k = _resolve_topk(valid_token_count, topk_ratio)
    token_local = torch.topk(token_energy, k=token_k, largest=True, sorted=False).indices.long()
    token_original = valid_positions.index_select(0, token_local).long()
    fp32_oracle_energy = flat_valid.index_select(0, scalar_local).square().sum().detach().cpu().reshape(())
    fp32_total_energy = flat_valid.square().sum().detach().cpu().reshape(())
    return (
        scalar_flat_original.detach().cpu().contiguous(),
        token_original.detach().cpu().contiguous(),
        int(scalar_k),
        int(token_k),
        int(valid_token_count),
        int(valid_scalar_count),
        int(feature_dim),
        fp32_oracle_energy,
        fp32_total_energy,
    )


def _append_oracle_rows_from_activation(
    columns: dict[str, list[Any]],
    *,
    sample_ids: Sequence[str],
    layer_idx: int,
    module_name: str,
    activation_position: str,
    activation: Tensor,
    token_padding_mask: Tensor,
    topk_ratio: float,
    target_name: str,
) -> None:
    batch_size = len(sample_ids)
    canonical = _activation_to_btc_with_layout(
        activation,
        batch_size=batch_size,
        target_name=target_name,
        activation_position=activation_position,
        padding_mask=token_padding_mask,
    )
    activation_btc = canonical.tensor
    public_layout = layout_info_to_public_fields(canonical.layout_info)
    if activation_btc.ndim != 3:
        raise ValueError("canonicalized activation must have shape [B,T,C]")
    mask = _align_token_padding_mask(
        token_padding_mask,
        batch_size=batch_size,
        num_tokens=int(activation_btc.shape[1]),
        device=activation_btc.device,
    )
    valid_mask = ~mask
    valid_counts = valid_mask.sum(dim=1).detach().cpu().long()
    if bool((valid_counts <= 0).any().item()):
        bad = [str(sample_ids[idx]) for idx, count in enumerate(valid_counts.tolist()) if int(count) <= 0]
        raise ValueError(f"{target_name}.{activation_position}: samples have zero valid tokens: {bad[:8]}")

    for batch_idx, sample_id in enumerate(sample_ids):
        (
            scalar_indices,
            token_indices_aux,
            scalar_k,
            token_k_aux,
            valid_token_count,
            valid_scalar_count,
            feature_dim,
            fp32_oracle_energy,
            fp32_total_energy,
        ) = _topk_indices_for_single_sample(
            activation_btc[batch_idx],
            valid_mask[batch_idx],
            topk_ratio=topk_ratio,
        )
        columns["sample_id"].append(str(sample_id))
        columns["layer_idx"].append(int(layer_idx))
        columns["module_name"].append(str(module_name))
        columns["activation_position"].append(str(activation_position))
        columns["topk_scalar_indices"].append(scalar_indices)
        columns["oer_topk_token_indices_aux"].append(token_indices_aux)
        columns["topk_k"].append(int(scalar_k))
        columns["oer_aux_token_topk_k"].append(int(token_k_aux))
        columns["valid_scalar_count"].append(int(valid_scalar_count))
        columns["feature_dim"].append(int(feature_dim))
        columns["fp32_oracle_energy"].append(float(fp32_oracle_energy.item()))
        columns["fp32_total_energy"].append(float(fp32_total_energy.item()))
        columns["topk_ratio"].append(float(topk_ratio))
        columns["activation_shape"].append(public_layout["canonical_shape"])
        columns["original_layout"].append(public_layout["original_layout"])
        columns["original_shape"].append(public_layout["original_shape"])
        columns["canonical_layout"].append(public_layout["canonical_layout"])
        columns["canonical_shape"].append(public_layout["canonical_shape"])
        columns["layout_inference_source"].append(public_layout["layout_inference_source"])
        columns["padding_mask_available"].append(public_layout["padding_mask_available"])
        columns["padding_mask_aligned"].append(public_layout["padding_mask_aligned"])
        columns["mask_status"].append(public_layout["mask_status"])
        columns["valid_token_count"].append(int(valid_token_count))


def _empty_oracle_columns() -> dict[str, list[Any]]:
    return {
        "sample_id": [],
        "layer_idx": [],
        "module_name": [],
        "activation_position": [],
        "topk_scalar_indices": [],
        "oer_topk_token_indices_aux": [],
        "topk_k": [],
        "oer_aux_token_topk_k": [],
        "valid_scalar_count": [],
        "feature_dim": [],
        "fp32_oracle_energy": [],
        "fp32_total_energy": [],
        "topk_ratio": [],
        "activation_shape": [],
        "original_layout": [],
        "original_shape": [],
        "canonical_layout": [],
        "canonical_shape": [],
        "layout_inference_source": [],
        "padding_mask_available": [],
        "padding_mask_aligned": [],
        "mask_status": [],
        "valid_token_count": [],
    }


def _tensorize_columns(columns: Mapping[str, list[Any]]) -> dict[str, Any]:
    row_count = len(columns["sample_id"])
    expected = {
        "layer_idx",
        "module_name",
        "activation_position",
        "topk_scalar_indices",
        "oer_topk_token_indices_aux",
        "topk_k",
        "oer_aux_token_topk_k",
        "valid_scalar_count",
        "feature_dim",
        "fp32_oracle_energy",
        "fp32_total_energy",
        "topk_ratio",
        "activation_shape",
        "original_layout",
        "original_shape",
        "canonical_layout",
        "canonical_shape",
        "layout_inference_source",
        "padding_mask_available",
        "padding_mask_aligned",
        "mask_status",
        "valid_token_count",
    }
    for key in expected:
        if len(columns[key]) != row_count:
            raise ValueError(f"oracle field {key} row count {len(columns[key])} does not match sample_id row count {row_count}")
    return {
        "sample_id": list(columns["sample_id"]),
        "layer_idx": torch.as_tensor(columns["layer_idx"], dtype=torch.long),
        "module_name": list(columns["module_name"]),
        "activation_position": list(columns["activation_position"]),
        "topk_scalar_indices": [item.long().cpu().contiguous() for item in columns["topk_scalar_indices"]],
        "oer_topk_token_indices_aux": [item.long().cpu().contiguous() for item in columns["oer_topk_token_indices_aux"]],
        "topk_k": torch.as_tensor(columns["topk_k"], dtype=torch.long),
        "oer_aux_token_topk_k": torch.as_tensor(columns["oer_aux_token_topk_k"], dtype=torch.long),
        "valid_scalar_count": torch.as_tensor(columns["valid_scalar_count"], dtype=torch.long),
        "feature_dim": torch.as_tensor(columns["feature_dim"], dtype=torch.long),
        "fp32_oracle_energy": torch.as_tensor(columns["fp32_oracle_energy"], dtype=torch.float32),
        "fp32_total_energy": torch.as_tensor(columns["fp32_total_energy"], dtype=torch.float32),
        "topk_ratio": torch.as_tensor(columns["topk_ratio"], dtype=torch.float32),
        "activation_shape": [tuple(int(dim) for dim in item) for item in columns["activation_shape"]],
        "original_layout": list(columns["original_layout"]),
        "original_shape": list(columns["original_shape"]),
        "canonical_layout": list(columns["canonical_layout"]),
        "canonical_shape": list(columns["canonical_shape"]),
        "layout_inference_source": list(columns["layout_inference_source"]),
        "padding_mask_available": list(columns["padding_mask_available"]),
        "padding_mask_aligned": list(columns["padding_mask_aligned"]),
        "mask_status": list(columns["mask_status"]),
        "valid_token_count": torch.as_tensor(columns["valid_token_count"], dtype=torch.long),
    }


def _build_row_index(sample_ids: Sequence[str], layer_idx: Tensor, module_names: Sequence[str], positions: Sequence[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for row_idx, sample_id in enumerate(sample_ids):
        key = f"{sample_id}|L{int(layer_idx[row_idx].item())}|{module_names[row_idx]}|{positions[row_idx]}"
        if key in index:
            raise ValueError(f"duplicate oracle row key: {key}")
        index[key] = int(row_idx)
    return index


def build_fp32_oer_oracle(
    cfg: Any,
    *,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Build FP32 OER oracle top-k indices for target FFN activations."""

    device = _device(cfg)
    topk_ratio = _topk_ratio(cfg)
    positions = _activation_positions(cfg)
    targets = all_ffn_targets()

    owns_model = model is None
    if model is None:
        model = _restore_fp32_teacher(cfg)
    model = model.to(device)
    model.eval()
    _validate_targets(model, targets)

    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)

    columns = _empty_oracle_columns()
    seen_sample_ids: set[str] = set()
    sample_id_order: list[str] = []
    captured_payloads: list[dict[str, Any]] = []

    def on_capture(payload: dict[str, Any]) -> None:
        captured_payloads.append(payload)

    def append_payload_rows(
        payload: dict[str, Any],
        *,
        sample_ids: Sequence[str],
        token_padding_mask: Tensor,
    ) -> None:
        for position in positions:
            activation_key = "input_activation" if position == "input" else "output_activation"
            _append_oracle_rows_from_activation(
                columns,
                sample_ids=sample_ids,
                layer_idx=int(payload["layer_idx"]),
                module_name=str(payload["module_name"]),
                activation_position=position,
                activation=payload[activation_key],
                token_padding_mask=token_padding_mask,
                topk_ratio=topk_ratio,
                target_name=str(payload["target_name"]),
            )

    for batch in eval_loader:
        batch_sample_ids = _sample_ids_from_batch(batch)
        duplicated_in_batch = [item for item in batch_sample_ids if batch_sample_ids.count(item) > 1]
        if duplicated_in_batch:
            raise ValueError(f"duplicate sample_id values within batch: {sorted(set(duplicated_in_batch))[:8]}")
        repeated = sorted(set(batch_sample_ids) & seen_sample_ids)
        if repeated:
            raise ValueError(f"duplicate sample_id values across eval_loader: {repeated[:8]}")
        seen_sample_ids.update(batch_sample_ids)
        sample_id_order.extend(batch_sample_ids)

        moved = _move_batch_to_device(batch, device)
        captured_payloads.clear()
        manager = ActivationHookManager(
            model,
            targets,
            cache_activations=False,
            cache_device=None,
            on_capture=on_capture,
        )
        manager.register_hooks()
        try:
            with torch.no_grad(), _autocast_context(cfg, device):
                _, token_padding_mask = backbone_features_for_cache(
                    model,
                    moved["waveform"],
                    moved.get("padding_mask"),
                )
            token_padding_mask = token_padding_mask.detach()
            if len(captured_payloads) != len(targets):
                raise ValueError(
                    f"hook capture count mismatch: captured={len(captured_payloads)}, expected={len(targets)}"
                )
            for payload in captured_payloads:
                append_payload_rows(
                    payload,
                    sample_ids=batch_sample_ids,
                    token_padding_mask=token_padding_mask,
                )
        finally:
            manager.clear_hooks()
            captured_payloads.clear()

    if not columns["sample_id"]:
        raise ValueError("eval_loader produced no OER oracle rows")

    data = _tensorize_columns(columns)
    row_index = _build_row_index(
        data["sample_id"],
        data["layer_idx"],
        data["module_name"],
        data["activation_position"],
    )
    row_count = len(data["sample_id"])
    expected_targets_per_sample = len(targets) * len(positions)
    if row_count != len(seen_sample_ids) * expected_targets_per_sample:
        raise ValueError(
            f"oracle row_count mismatch: row_count={row_count}, "
            f"num_samples={len(seen_sample_ids)}, expected_per_sample={expected_targets_per_sample}"
        )

    manifest = {
        "oracle_version": ORACLE_VERSION,
        "cache_type": "fp32_oer_oracle",
        "oracle_source": "fp32_real_activation",
        "stage": "quant_injection_pre_fp32_oer_oracle",
        "eval_csv": str(_eval_csv(cfg).expanduser().resolve()),
        "eval_csv_sha256": _file_sha256(_eval_csv(cfg)),
        "checkpoint_path": _checkpoint_path_for_manifest(cfg),
        **_checkpoint_protocol_fields(cfg),
        "dataset_name": _dataset_name(cfg),
        "num_samples": len(seen_sample_ids),
        "sample_order_hash": _sample_order_hash(sample_id_order),
        "sample_order_hash_algorithm": "sha256-json-list",
        "row_count": row_count,
        "num_layers": NUM_BEATS_LAYERS,
        "target_modules": [{"layer_idx": layer_idx, "module_name": module_name} for layer_idx, module_name in targets],
        "activation_positions": list(positions),
        **_scalar_topk_protocol_fields(topk_ratio),
        "activation_layout": "hook capture canonical_layout is BTC; deprecated activation_shape equals canonical_shape; scalar index uses scalar_index = token_idx * feature_dim + feature_idx",
        "padding_policy": "padding token excluded before top-k; token padding_mask True means padding",
        "contains_fp32_oracle_energy": True,
        "contains_fp32_total_energy": True,
        "energy_scope": "valid_tokens_only",
        "contains_oer_retention": False,
        "forward_passes_per_batch": 1,
    }
    oracle = {
        "oracle_version": ORACLE_VERSION,
        "manifest": manifest,
        **data,
        "row_index": row_index,
        "sample_id_set": sorted(seen_sample_ids),
    }
    if owns_model and device.type == "cuda":
        torch.cuda.empty_cache()
    return oracle


def build_and_save_fp32_oer_oracle_sharded(
    cfg: Any,
    oracle_path: str | Path,
    *,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Build FP32 OER oracle as per-layer, per-sample-shard files."""

    path = Path(oracle_path)
    shard_dir = _oracle_shard_dir(path)
    shard_size = _sample_shard_size(cfg)
    device = _device(cfg)
    topk_ratio = _topk_ratio(cfg)
    positions = _activation_positions(cfg)
    targets = all_ffn_targets()
    layer_indices = sorted({int(layer_idx) for layer_idx, _module_name in targets})

    owns_model = model is None
    if model is None:
        model = _restore_fp32_teacher(cfg)
    model = model.to(device)
    model.eval()
    _validate_targets(model, targets)

    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)

    seen_sample_ids: set[str] = set()
    sample_id_order: list[str] = []
    shard_records: list[dict[str, Any]] = []
    current_shard_sample_ids: list[str] = []
    current_sample_start = 0
    sample_shard_id = 0
    total_row_count = 0
    layer_columns: dict[int, dict[str, list[Any]]] = {layer_idx: _empty_oracle_columns() for layer_idx in layer_indices}
    captured_payloads: list[dict[str, Any]] = []

    def reset_layer_columns() -> None:
        layer_columns.clear()
        layer_columns.update({layer_idx: _empty_oracle_columns() for layer_idx in layer_indices})

    def flush_sample_shard() -> None:
        nonlocal current_sample_start, sample_shard_id, total_row_count
        if not current_shard_sample_ids:
            return
        sample_end = current_sample_start + len(current_shard_sample_ids)
        sample_hash = _sample_order_hash(current_shard_sample_ids)
        expected_rows_per_layer = len(current_shard_sample_ids) * len(FFN_MODULE_NAMES) * len(positions)
        for layer_idx in layer_indices:
            columns = layer_columns[layer_idx]
            if len(columns["sample_id"]) != expected_rows_per_layer:
                raise RuntimeError(
                    f"OER shard row_count mismatch for layer {layer_idx}: "
                    f"{len(columns['sample_id'])} vs {expected_rows_per_layer}"
                )
            data = _tensorize_columns(columns)
            row_count = len(data["sample_id"])
            shard_path = shard_dir / f"layer_{int(layer_idx):02d}_shard_{int(sample_shard_id):05d}.pt"
            shard_manifest = {
                "oracle_version": ORACLE_VERSION,
                "cache_type": "fp32_oer_oracle_shard",
                "parent_cache_type": "fp32_oer_oracle",
                "oracle_source": "fp32_real_activation",
                "stage": "quant_injection_pre_fp32_oer_oracle",
                "storage_format": ORACLE_STORAGE_FORMAT,
                "layer_idx": int(layer_idx),
                "sample_shard_id": int(sample_shard_id),
                "sample_start": int(current_sample_start),
                "sample_end": int(sample_end),
                "num_samples": int(len(current_shard_sample_ids)),
                "sample_order_hash": sample_hash,
                "row_count": int(row_count),
                "activation_positions": list(positions),
                **_scalar_topk_protocol_fields(topk_ratio),
            }
            payload = {
                "oracle_version": ORACLE_VERSION,
                "cache_type": "fp32_oer_oracle_shard",
                "manifest": shard_manifest,
                **data,
                "row_index": _build_row_index(data["sample_id"], data["layer_idx"], data["module_name"], data["activation_position"]),
                "sample_id_set": sorted(set(str(item) for item in data["sample_id"])),
            }
            _atomic_torch_save(payload, shard_path)
            shard_hash = _file_sha256(shard_path)
            shard_records.append(
                {
                    "path": _relative_to_parent(shard_path, path.parent),
                    "sha256": shard_hash,
                    "layer_idx": int(layer_idx),
                    "sample_shard_id": int(sample_shard_id),
                    "sample_start": int(current_sample_start),
                    "sample_end": int(sample_end),
                    "num_samples": int(len(current_shard_sample_ids)),
                    "row_count": int(row_count),
                    "sample_order_hash": sample_hash,
                }
            )
            total_row_count += row_count
        current_sample_start = sample_end
        sample_shard_id += 1
        current_shard_sample_ids.clear()
        reset_layer_columns()

    def on_capture(payload: dict[str, Any]) -> None:
        captured_payloads.append(payload)

    def append_payload_rows(
        payload: dict[str, Any],
        *,
        sample_ids: Sequence[str],
        token_padding_mask: Tensor,
    ) -> None:
        layer_idx = int(payload["layer_idx"])
        if layer_idx not in layer_columns:
            raise KeyError(f"unexpected OER layer_idx captured: {layer_idx}")
        for position in positions:
            activation_key = "input_activation" if position == "input" else "output_activation"
            _append_oracle_rows_from_activation(
                layer_columns[layer_idx],
                sample_ids=sample_ids,
                layer_idx=layer_idx,
                module_name=str(payload["module_name"]),
                activation_position=position,
                activation=payload[activation_key],
                token_padding_mask=token_padding_mask,
                topk_ratio=topk_ratio,
                target_name=str(payload["target_name"]),
            )

    for batch in eval_loader:
        batch_sample_ids = _sample_ids_from_batch(batch)
        duplicated_in_batch = [item for item in batch_sample_ids if batch_sample_ids.count(item) > 1]
        if duplicated_in_batch:
            raise ValueError(f"duplicate sample_id values within batch: {sorted(set(duplicated_in_batch))[:8]}")
        repeated = sorted(set(batch_sample_ids) & seen_sample_ids)
        if repeated:
            raise ValueError(f"duplicate sample_id values across eval_loader: {repeated[:8]}")

        moved = _move_batch_to_device(batch, device)
        captured_payloads.clear()
        manager = ActivationHookManager(
            model,
            targets,
            cache_activations=False,
            cache_device=None,
            on_capture=on_capture,
        )
        manager.register_hooks()
        try:
            with torch.no_grad(), _autocast_context(cfg, device):
                _, token_padding_mask = backbone_features_for_cache(
                    model,
                    moved["waveform"],
                    moved.get("padding_mask"),
                )
            token_padding_mask = token_padding_mask.detach()
            if len(captured_payloads) != len(targets):
                raise ValueError(
                    f"hook capture count mismatch: captured={len(captured_payloads)}, expected={len(targets)}"
                )
            for payload in captured_payloads:
                append_payload_rows(
                    payload,
                    sample_ids=batch_sample_ids,
                    token_padding_mask=token_padding_mask,
                )
        finally:
            manager.clear_hooks()
            captured_payloads.clear()

        seen_sample_ids.update(batch_sample_ids)
        sample_id_order.extend(batch_sample_ids)
        current_shard_sample_ids.extend(batch_sample_ids)
        if len(current_shard_sample_ids) >= shard_size:
            flush_sample_shard()

    flush_sample_shard()
    if not sample_id_order:
        raise ValueError("eval_loader produced no OER oracle rows")

    expected_targets_per_sample = len(targets) * len(positions)
    expected_total_rows = len(sample_id_order) * expected_targets_per_sample
    if total_row_count != expected_total_rows:
        raise ValueError(
            f"oracle row_count mismatch: row_count={total_row_count}, "
            f"num_samples={len(sample_id_order)}, expected_per_sample={expected_targets_per_sample}"
        )

    manifest = {
        "oracle_version": ORACLE_VERSION,
        "cache_type": "fp32_oer_oracle",
        "oracle_source": "fp32_real_activation",
        "stage": "quant_injection_pre_fp32_oer_oracle",
        "storage_format": ORACLE_STORAGE_FORMAT,
        "sharded": True,
        "sample_shard_size": int(shard_size),
        "eval_csv": str(_eval_csv(cfg).expanduser().resolve()),
        "eval_csv_sha256": _file_sha256(_eval_csv(cfg)),
        "checkpoint_path": _checkpoint_path_for_manifest(cfg),
        **_checkpoint_protocol_fields(cfg),
        "dataset_name": _dataset_name(cfg),
        "num_samples": len(sample_id_order),
        "sample_order_hash": _sample_order_hash(sample_id_order),
        "sample_order_hash_algorithm": "sha256-json-list",
        "row_count": int(total_row_count),
        "num_layers": NUM_BEATS_LAYERS,
        "target_modules": [{"layer_idx": layer_idx, "module_name": module_name} for layer_idx, module_name in targets],
        "activation_positions": list(positions),
        **_scalar_topk_protocol_fields(topk_ratio),
        "activation_layout": "hook capture canonical_layout is BTC; deprecated activation_shape equals canonical_shape; scalar index uses scalar_index = token_idx * feature_dim + feature_idx",
        "padding_policy": "padding token excluded before top-k; token padding_mask True means padding",
        "contains_fp32_oracle_energy": True,
        "contains_fp32_total_energy": True,
        "energy_scope": "valid_tokens_only",
        "contains_oer_retention": False,
        "forward_passes_per_batch": 1,
        "num_shards": len(shard_records),
        "shard_dir": _relative_to_parent(shard_dir, path.parent),
        "shard_paths": [str(item["path"]) for item in shard_records],
        "shard_sha256": {str(item["path"]): item.get("sha256") for item in shard_records},
        "shards": list(shard_records),
    }
    oracle = {
        "oracle_version": ORACLE_VERSION,
        "cache_type": "fp32_oer_oracle",
        "manifest": manifest,
        "storage_format": ORACLE_STORAGE_FORMAT,
        "sharded": True,
        "sample_id_order": list(sample_id_order),
        "sample_id_set": sorted(seen_sample_ids),
        "shard_paths": [str(item["path"]) for item in shard_records],
        "shard_sha256": {str(item["path"]): item.get("sha256") for item in shard_records},
        "shards": list(shard_records),
    }
    _atomic_torch_save(oracle, path)
    _atomic_json_dump(manifest, _manifest_path(path))
    if owns_model and device.type == "cuda":
        torch.cuda.empty_cache()
    return {"oracle": oracle, "oracle_path": path, "manifest_path": _manifest_path(path)}

def save_fp32_oer_oracle(oracle: Mapping[str, Any], oracle_path: str | Path) -> Path:
    path = Path(oracle_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(oracle), path)
    manifest = oracle.get("manifest", {})
    with _manifest_path(path).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return path


def export_fp32_oer_oracle(
    cfg: Any | None = None,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """FP32 OER oracle helper."""

    cfg = _default_cfg() if cfg is None else cfg
    path = _oracle_path(cfg, output_path)
    if path.exists() and not force:
        raise FileExistsError(f"FP32 OER oracle already exists: {path}; pass force=True to rebuild")
    return build_and_save_fp32_oer_oracle_sharded(
        cfg,
        path,
        model=model,
        eval_loader=eval_loader,
    )

def main(cfg: Any | None = None) -> dict[str, Any]:
    return export_fp32_oer_oracle(cfg)


__all__ = [
    "ALLOWED_ACTIVATION_POSITIONS",
    "DEFAULT_ACTIVATION_POSITIONS",
    "FFN_MODULE_NAMES",
    "MANIFEST_FILENAME",
    "NUM_BEATS_LAYERS",
    "ORACLE_FILENAME",
    "ORACLE_STORAGE_FORMAT",
    "ORACLE_VERSION",
    "all_ffn_targets",
    "build_and_save_fp32_oer_oracle_sharded",
    "build_fp32_oer_oracle",
    "export_fp32_oer_oracle",
    "is_sharded_fp32_oer_oracle",
    "iter_fp32_oer_oracle_shards",
    "load_fp32_oer_oracle",
    "main",
    "save_fp32_oer_oracle",
]
