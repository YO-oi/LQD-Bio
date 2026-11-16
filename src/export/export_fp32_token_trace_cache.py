from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
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
        _autocast_context,
        _cfg_get,
        _checkpoint_path_for_manifest,
        _checkpoint_sha256_for_manifest,
        _file_sha256,
        _sample_order_hash,
        _dataset_cfg,
        _default_cfg,
        _make_audio_config,
        _move_batch_to_device,
        _nested_get,
        _restore_fp32_teacher,
        backbone_features_for_cache,
    )
    from src.utils.layout_utils import canonicalize_activation_btc, canonicalize_padding_mask_bt, layout_info_to_public_fields
    from src.utils.token_trace_hook import (
        FFN_MODULE_NAMES,
        FULL_LAYER_INDICES,
        TokenTraceHookManager,
        TokenTraceRecord,
        get_transformer_block,
    )
except ImportError:  # pragma: no cover - script execution fallback.
    from ..data.audio_input import AudioInputDataset, audio_input_collate_fn
    from ..export.export_fp32_core_cache import (
        _autocast_context,
        _cfg_get,
        _checkpoint_path_for_manifest,
        _checkpoint_sha256_for_manifest,
        _file_sha256,
        _sample_order_hash,
        _dataset_cfg,
        _default_cfg,
        _make_audio_config,
        _move_batch_to_device,
        _nested_get,
        _restore_fp32_teacher,
        backbone_features_for_cache,
    )
    from ..utils.layout_utils import canonicalize_activation_btc, canonicalize_padding_mask_bt, layout_info_to_public_fields
    from ..utils.token_trace_hook import (
        FFN_MODULE_NAMES,
        FULL_LAYER_INDICES,
        TokenTraceHookManager,
        TokenTraceRecord,
        get_transformer_block,
    )


CACHE_VERSION = 1
CACHE_DIRNAME = "fp32_token_trace_cache"
MANIFEST_FILENAME = "fp32_token_trace_cache.manifest.json"
INDEX_FILENAME = "fp32_token_trace_cache_index.jsonl"
BLOCK_TRACE_DIRNAME = "block_token_trace"
FFN_TRACE_DIRNAME = "ffn_activation_trace"
DEFAULT_TRACE_DTYPE = "float32"


def _trace_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_TOKEN_TRACE_CACHE", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    trace_cfg = _trace_cfg(cfg)
    if key in trace_cfg:
        return trace_cfg[key]
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
        raise AttributeError("export_fp32_token_trace_cache requires config.py DEVICE")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py DEVICE={configured!r}, but CUDA is not available")
    return device


def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


def _batch_size(cfg: Any) -> int:
    return int(_setting_get(cfg, "EVAL_BATCH_SIZE", _setting_get(cfg, "BATCH_SIZE", 4)))


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
    raise AttributeError(
        "missing eval csv; configure CURRENT_DATASET['EVAL_CSV'] or FP32_TOKEN_TRACE_CACHE['EVAL_CSV']"
    )

def _output_dir(cfg: Any, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / CACHE_DIRNAME


def _index_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / INDEX_FILENAME


def _manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / MANIFEST_FILENAME


def _include_ffn_activation_trace(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "INCLUDE_FFN_ACTIVATION_TRACE", False))


def _check_finite(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "CHECK_FINITE", True))


def _trace_dtype(cfg: Any) -> torch.dtype:
    name = str(_setting_get(cfg, "TRACE_DTYPE", DEFAULT_TRACE_DTYPE)).lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"TRACE_DTYPE must be float32/float16/bfloat16, got {name!r}")
    return mapping[name]


def _trace_dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    return str(dtype).replace("torch.", "")


def _layer_indices(cfg: Any) -> tuple[int, ...]:
    configured = _setting_get(cfg, "LAYER_INDICES", FULL_LAYER_INDICES)
    values = tuple(int(item) for item in configured)
    if not values:
        raise ValueError("LAYER_INDICES cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"LAYER_INDICES contains duplicate values: {values}")
    missing = sorted(set(values) - set(FULL_LAYER_INDICES))
    if missing:
        raise ValueError(f"LAYER_INDICES must be within Layer0..Layer11, missing={missing}")
    return values


def _target_modules(layer_indices: Sequence[int]) -> tuple[tuple[int, str], ...]:
    return tuple((int(layer_idx), module_name) for layer_idx in layer_indices for module_name in FFN_MODULE_NAMES)


def _build_eval_loader(cfg: Any) -> DataLoader:
    override = _cfg_get(cfg, "build_eval_loader", None)
    if callable(override):
        return override(cfg)
    mode = str(_setting_get(cfg, "AUDIO_MODE", "eval"))
    if mode.lower() == "train":
        raise ValueError("FP32 token trace cache must not use train random crop")
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
        raise KeyError("eval batch must include sample_id to build a reproducible token trace cache")
    return [str(item) for item in batch["sample_id"]]


def _label_ids_from_batch(batch: Mapping[str, Any]) -> Tensor | None:
    if "label_id" not in batch:
        return None
    value = batch["label_id"]
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.long)
    return value.detach().cpu().long().reshape(-1)


def _normalise_token_padding_mask(mask: Tensor | None, *, batch_size: int, num_tokens: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)
    return canonicalize_padding_mask_bt(
        mask.to(device=device, dtype=torch.bool),
        batch_size=int(batch_size),
        token_count=int(num_tokens),
        context="fp32_token_trace_cache.token_padding_mask",
    )


def _prepare_encoder_input_and_mask(
    model: nn.Module,
    waveform: Tensor,
    padding_mask: Tensor | None,
) -> tuple[Tensor, Tensor | None, Tensor] | None:
    """Prepare encoder input and token padding mask."""

    required = ("preprocess", "patch_embedding", "encoder")
    if not all(hasattr(model, name) for name in required):
        return None

    fbank = model.preprocess(waveform)
    feature_padding_mask = padding_mask
    if feature_padding_mask is not None:
        feature_padding_mask = model.forward_padding_mask(fbank, feature_padding_mask)

    features = model.patch_embedding(fbank.unsqueeze(1))
    features = features.reshape(features.shape[0], features.shape[1], -1)
    features = features.transpose(1, 2)
    features = model.layer_norm(features)

    if feature_padding_mask is not None:
        feature_padding_mask = model.forward_padding_mask(features, feature_padding_mask)

    if getattr(model, "post_extract_proj", None) is not None:
        features = model.post_extract_proj(features)

    encoder_input = model.dropout_input(features)
    token_padding_mask = _normalise_token_padding_mask(
        feature_padding_mask,
        batch_size=int(encoder_input.shape[0]),
        num_tokens=int(encoder_input.shape[1]),
        device=encoder_input.device,
    )
    return encoder_input, feature_padding_mask, token_padding_mask


def _first_feature_tensor(value: Any, *, target_name: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    candidates: list[Any]
    if isinstance(value, Mapping):
        preferred_keys = ("x", "features", "feature", "encoder_out", "hidden_states", "last_hidden_state")
        for key in preferred_keys:
            item = value.get(key)
            if isinstance(item, Tensor):
                return item
        candidates = list(value.values())
    elif isinstance(value, (tuple, list)):
        candidates = list(value)
    else:
        raise TypeError(f"{target_name}: hook output must be Tensor/tuple/list/dict, got {type(value)!r}")
    tensors = [item for item in candidates if isinstance(item, Tensor)]
    for item in tensors:
        if item.is_floating_point() and item.ndim >= 3:
            return item
    if tensors:
        return tensors[0]
    raise TypeError(f"{target_name}: hook output did not contain a feature Tensor")


def _feature_to_btd(value: Any, *, batch_size: int, padding_mask: Tensor, target_name: str):
    tensor = _first_feature_tensor(value, target_name=target_name).detach()
    return canonicalize_activation_btc(
        tensor,
        padding_mask=padding_mask,
        batch_size=int(batch_size),
        context=target_name,
    )


def _record_from_block_output(
    *,
    output: Any,
    layer_idx: int,
    sample_ids: Sequence[str],
    padding_mask: Tensor,
    target_name: str,
) -> dict[str, Any]:
    canonical = _feature_to_btd(output, batch_size=len(sample_ids), padding_mask=padding_mask, target_name=target_name)
    tensor = canonical.tensor.detach().to(device=torch.device("cpu")).contiguous()
    public_layout = layout_info_to_public_fields(canonical.layout_info)
    if tuple(padding_mask.shape) != (len(sample_ids), int(tensor.shape[1])):
        raise ValueError(
            f"{target_name}: padding_mask must align with token tensor [B,T]; mask={tuple(padding_mask.shape)}, tensor={tuple(tensor.shape)}"
        )
    return {
        "layer_idx": int(layer_idx),
        "module_name": "block",
        "tensor_position": "block_output",
        "target_name": target_name,
        "tensor": tensor,
        "activation_shape": tuple(int(item) for item in tensor.shape),
        "original_layout": public_layout["original_layout"],
        "original_shape": public_layout["original_shape"],
        "canonical_layout": public_layout["canonical_layout"],
        "canonical_shape": public_layout["canonical_shape"],
        "layout_inference_source": public_layout["layout_inference_source"],
        "padding_mask_available": public_layout["padding_mask_available"],
        "padding_mask_aligned": public_layout["padding_mask_aligned"],
        "mask_status": public_layout["mask_status"],
        "sample_id": list(sample_ids),
        "padding_mask": padding_mask.detach().cpu().bool().contiguous(),
        "valid_token_mask": (~padding_mask.detach().cpu().bool()).contiguous(),
    }


def _capture_block_only_records(
    model: nn.Module,
    *,
    layer_indices: Sequence[int],
    sample_ids: Sequence[str],
    token_padding_mask: Tensor,
    forward_fn: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    handles: list[Any] = []
    cpu_mask = token_padding_mask.detach().cpu().bool().contiguous()
    try:
        for layer_idx in layer_indices:
            block = get_transformer_block(model, int(layer_idx))
            target_name = f"Layer{int(layer_idx)}.block_output"

            def hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                hook_layer_idx: int = int(layer_idx),
                hook_target_name: str = target_name,
            ) -> None:
                del _module, _inputs
                records.append(
                    _record_from_block_output(
                        output=output,
                        layer_idx=hook_layer_idx,
                        sample_ids=sample_ids,
                        padding_mask=cpu_mask,
                        target_name=hook_target_name,
                    )
                )

            handles.append(block.register_forward_hook(hook))
        forward_fn()
    finally:
        for handle in handles:
            handle.remove()
    return records


def _record_to_payload(record: TokenTraceRecord | Mapping[str, Any], *, store_dtype: torch.dtype, check_finite: bool) -> dict[str, Any]:
    if isinstance(record, TokenTraceRecord):
        tensor = record.tensor
        padding_mask = record.padding_mask
        sample_ids = record.sample_id
        layer_idx = record.layer_idx
        module_name = record.module_name
        tensor_position = record.tensor_position
        target_name = record.target_name
        activation_shape = record.activation_shape
        original_layout = record.original_layout
        original_shape = record.original_shape
        canonical_layout = record.canonical_layout
        canonical_shape = record.canonical_shape
        layout_inference_source = record.layout_inference_source
        padding_mask_available = record.padding_mask_available
        padding_mask_aligned = record.padding_mask_aligned
        mask_status = record.mask_status
    else:
        tensor = record["tensor"]
        padding_mask = record["padding_mask"]
        sample_ids = record["sample_id"]
        layer_idx = record["layer_idx"]
        module_name = record["module_name"]
        tensor_position = record["tensor_position"]
        target_name = record["target_name"]
        activation_shape = record["activation_shape"]
        original_layout = record.get("original_layout", "BTC")
        original_shape = record.get("original_shape", activation_shape)
        canonical_layout = record.get("canonical_layout", "BTC")
        canonical_shape = record.get("canonical_shape", activation_shape)
        layout_inference_source = record.get("layout_inference_source", "legacy_or_block_record")
        padding_mask_available = record.get("padding_mask_available", True)
        padding_mask_aligned = record.get("padding_mask_aligned", True)
        mask_status = record.get("mask_status", "aligned")

    tensor = tensor.detach().cpu().to(dtype=store_dtype).contiguous()
    padding_mask = padding_mask.detach().cpu().bool().contiguous()
    if check_finite and not torch.isfinite(tensor.float()).all().item():
        raise ValueError(f"{target_name}: FP32 token trace contains NaN/Inf")
    valid_token_count = (~padding_mask).sum(dim=1).long()
    if bool((valid_token_count <= 0).any().item()):
        bad = [str(sample_ids[idx]) for idx, value in enumerate(valid_token_count.tolist()) if int(value) <= 0]
        raise ValueError(f"{target_name}: samples have zero valid tokens: {bad[:8]}")
    return {
        "layer_idx": int(layer_idx),
        "module_name": str(module_name),
        "tensor_position": str(tensor_position),
        "target_name": str(target_name),
        "activation_shape": tuple(int(item) for item in canonical_shape),
        "original_layout": str(original_layout),
        "original_shape": tuple(int(item) for item in original_shape),
        "canonical_layout": str(canonical_layout),
        "canonical_shape": tuple(int(item) for item in canonical_shape),
        "layout_inference_source": str(layout_inference_source),
        "padding_mask_available": bool(padding_mask_available),
        "padding_mask_aligned": bool(padding_mask_aligned),
        "mask_status": str(mask_status),
        "sample_id": [str(item) for item in sample_ids],
        "valid_token_count": valid_token_count,
        "tensor": tensor,
    }


def _records_by_group(records: Sequence[TokenTraceRecord | Mapping[str, Any]]) -> tuple[list[Any], list[Any]]:
    block_records: list[Any] = []
    ffn_records: list[Any] = []
    for record in records:
        position = record.tensor_position if isinstance(record, TokenTraceRecord) else record["tensor_position"]
        if str(position) == "block_output":
            block_records.append(record)
        else:
            ffn_records.append(record)
    return block_records, ffn_records


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_jsonl_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _save_trace_shard(
    *,
    output_dir: Path,
    trace_group: str,
    batch_index: int,
    sample_ids: Sequence[str],
    label_id: Tensor | None,
    token_padding_mask: Tensor,
    records: Sequence[TokenTraceRecord | Mapping[str, Any]],
    store_dtype: torch.dtype,
    check_finite: bool,
) -> tuple[Path | None, list[dict[str, Any]]]:
    if not records:
        return None, []

    if trace_group == "block_token_trace":
        trace_dir = output_dir / BLOCK_TRACE_DIRNAME
        filename = f"block_trace_batch_{batch_index:06d}.pt"
    elif trace_group == "ffn_activation_trace":
        trace_dir = output_dir / FFN_TRACE_DIRNAME
        filename = f"ffn_trace_batch_{batch_index:06d}.pt"
    else:
        raise ValueError(f"unknown trace_group: {trace_group!r}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    shard_path = trace_dir / filename

    padding_mask = token_padding_mask.detach().cpu().bool().contiguous()
    valid_token_count = (~padding_mask).sum(dim=1).long()
    payload_records = [
        _record_to_payload(record, store_dtype=store_dtype, check_finite=check_finite)
        for record in records
    ]
    payload = {
        "cache_version": CACHE_VERSION,
        "trace_group": trace_group,
        "batch_index": int(batch_index),
        "sample_id": [str(item) for item in sample_ids],
        "label_id": None if label_id is None else label_id.detach().cpu().long().contiguous(),
        "padding_mask": padding_mask,
        "valid_token_count": valid_token_count,
        "tensor_layout": "[B,T,D]",
        "padding_mask_semantics": "True means padding token; False means real token",
        "records": payload_records,
    }
    torch.save(payload, shard_path)

    index_rows: list[dict[str, Any]] = []
    for record in payload_records:
        index_rows.append(
            {
                "trace_group": trace_group,
                "shard_path": _relative_path(shard_path, output_dir),
                "batch_index": int(batch_index),
                "sample_id": [str(item) for item in sample_ids],
                "layer_idx": int(record["layer_idx"]),
                "module_name": str(record["module_name"]),
                "tensor_position": str(record["tensor_position"]),
                "activation_shape": [int(item) for item in record["canonical_shape"]],
                "original_layout": record.get("original_layout", ""),
                "original_shape": [int(item) for item in record.get("original_shape", [])],
                "canonical_layout": record.get("canonical_layout", "BTC"),
                "canonical_shape": [int(item) for item in record.get("canonical_shape", record["activation_shape"])],
                "layout_inference_source": record.get("layout_inference_source", ""),
                "padding_mask_available": bool(record.get("padding_mask_available", False)),
                "padding_mask_aligned": bool(record.get("padding_mask_aligned", False)),
                "mask_status": record.get("mask_status", ""),
                "valid_token_count": [int(item) for item in valid_token_count.tolist()],
            }
        )
    return shard_path, index_rows


def _clean_output_dir_for_force(output_dir: Path) -> None:
    for name in (BLOCK_TRACE_DIRNAME, FFN_TRACE_DIRNAME):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
    for path in (_index_path(output_dir), _manifest_path(output_dir)):
        if path.exists():
            path.unlink()


def _forward_for_trace(
    model: nn.Module,
    *,
    waveform: Tensor,
    padding_mask: Tensor | None,
    prepared: tuple[Tensor, Tensor | None, Tensor] | None,
) -> None:
    if prepared is not None:
        encoder_input, encoder_padding_mask, _token_padding_mask = prepared
        model.encoder(encoder_input, padding_mask=encoder_padding_mask)
        return
    backbone_features_for_cache(model, waveform, padding_mask)


def build_fp32_token_trace_cache(
    cfg: Any,
    *,
    output_dir: str | Path | None = None,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build the FP32 token trace cache for block and FFN activation tensors."""
    device = _device(cfg)
    output_root = _output_dir(cfg, output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"FP32 token trace cache output directory is not empty: {output_root}; pass force=True to rebuild")
    output_root.mkdir(parents=True, exist_ok=True)
    if force:
        _clean_output_dir_for_force(output_root)

    include_ffn = _include_ffn_activation_trace(cfg)
    layer_indices = _layer_indices(cfg)
    target_modules = _target_modules(layer_indices)
    store_dtype = _trace_dtype(cfg)
    check_finite = _check_finite(cfg)

    owns_model = model is None
    if model is None:
        model = _restore_fp32_teacher(cfg)
    model = model.to(device)
    model.eval()

    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)

    index_path = _index_path(output_root)
    if index_path.exists():
        index_path.unlink()

    seen_sample_ids: set[str] = set()
    sample_id_order: list[str] = []
    block_shards: list[str] = []
    ffn_shards: list[str] = []
    block_record_count = 0
    ffn_record_count = 0
    num_batches = 0
    num_samples = 0
    used_frontend_shortcut = True

    for batch_index, batch in enumerate(eval_loader):
        batch_sample_ids = _sample_ids_from_batch(batch)
        repeated = sorted(set(batch_sample_ids) & seen_sample_ids)
        if repeated:
            raise ValueError(f"duplicate sample_id values across eval_loader: {repeated[:8]}")
        if len(set(batch_sample_ids)) != len(batch_sample_ids):
            raise ValueError("duplicate sample_id values within batch; token trace cache requires unique sample ids")
        seen_sample_ids.update(batch_sample_ids)
        sample_id_order.extend(batch_sample_ids)
        num_batches += 1
        num_samples += len(batch_sample_ids)

        moved = _move_batch_to_device(batch, device)
        waveform = moved["waveform"]
        waveform_padding_mask = moved.get("padding_mask")
        label_id = _label_ids_from_batch(batch)

        with torch.no_grad(), _autocast_context(cfg, device):
            prepared = _prepare_encoder_input_and_mask(model, waveform, waveform_padding_mask)
            if prepared is None:
                used_frontend_shortcut = False
                _, token_padding_mask = backbone_features_for_cache(model, waveform, waveform_padding_mask)
            else:
                token_padding_mask = prepared[2]

            if include_ffn:
                manager = TokenTraceHookManager(
                    model,
                    layer_indices=layer_indices,
                    target_modules=target_modules,
                    require_padding_mask=True,
                )
                manager.set_batch_context(sample_id=batch_sample_ids, padding_mask=token_padding_mask.detach().cpu(), clear_cache=True)
                manager.register_hooks()
                try:
                    _forward_for_trace(
                        model,
                        waveform=waveform,
                        padding_mask=waveform_padding_mask,
                        prepared=prepared,
                    )
                    records: list[Any] = manager.pop_cache()
                finally:
                    manager.clear_hooks()
                    manager.clear_batch_context()
            else:
                records = _capture_block_only_records(
                    model,
                    layer_indices=layer_indices,
                    sample_ids=batch_sample_ids,
                    token_padding_mask=token_padding_mask,
                    forward_fn=lambda: _forward_for_trace(
                        model,
                        waveform=waveform,
                        padding_mask=waveform_padding_mask,
                        prepared=prepared,
                    ),
                )

        block_records, ffn_records = _records_by_group(records)
        expected_block_records = len(layer_indices)
        if len(block_records) != expected_block_records:
            raise RuntimeError(f"block_token_trace record count mismatch: {len(block_records)} vs {expected_block_records}")
        if include_ffn:
            expected_ffn_records = len(target_modules) * 2
            if len(ffn_records) != expected_ffn_records:
                raise RuntimeError(f"ffn_activation_trace record count mismatch: {len(ffn_records)} vs {expected_ffn_records}")
        elif ffn_records:
            raise RuntimeError("INCLUDE_FFN_ACTIVATION_TRACE=False but FFN trace records were captured")

        block_path, block_rows = _save_trace_shard(
            output_dir=output_root,
            trace_group="block_token_trace",
            batch_index=batch_index,
            sample_ids=batch_sample_ids,
            label_id=label_id,
            token_padding_mask=token_padding_mask,
            records=block_records,
            store_dtype=store_dtype,
            check_finite=check_finite,
        )
        if block_path is not None:
            block_shards.append(_relative_path(block_path, output_root))
            block_record_count += len(block_rows)
            _write_jsonl_rows(index_path, block_rows)

        ffn_path, ffn_rows = _save_trace_shard(
            output_dir=output_root,
            trace_group="ffn_activation_trace",
            batch_index=batch_index,
            sample_ids=batch_sample_ids,
            label_id=label_id,
            token_padding_mask=token_padding_mask,
            records=ffn_records,
            store_dtype=store_dtype,
            check_finite=check_finite,
        )
        if ffn_path is not None:
            ffn_shards.append(_relative_path(ffn_path, output_root))
            ffn_record_count += len(ffn_rows)
            _write_jsonl_rows(index_path, ffn_rows)

    if num_samples <= 0:
        raise ValueError("eval_loader produced no samples for token trace cache")

    manifest = {
        "cache_version": CACHE_VERSION,
        "cache_type": "fp32_token_trace_cache",
        "stage": "optional_secondary_token_trace_cache",
        "dataset_name": _dataset_name(cfg),
        "eval_csv": str(_eval_csv(cfg).expanduser().resolve()),
        "eval_csv_sha256": _file_sha256(_eval_csv(cfg)),
        "checkpoint_path": _checkpoint_path_for_manifest(cfg),
        "checkpoint_sha256": _checkpoint_sha256_for_manifest(cfg),
        "output_dir": str(output_root.expanduser().resolve()),
        "index_file": INDEX_FILENAME,
        "num_samples": int(num_samples),
        "sample_order_hash": _sample_order_hash(sample_id_order),
        "sample_order_hash_algorithm": "sha256-json-list",
        "num_batches": int(num_batches),
        "layer_indices": [int(item) for item in layer_indices],
        "block_token_trace": {
            "enabled": True,
            "record_count": int(block_record_count),
            "shard_count": len(block_shards),
            "shards": block_shards,
        },
        "ffn_activation_trace": {
            "enabled": bool(include_ffn),
            "record_count": int(ffn_record_count),
            "shard_count": len(ffn_shards),
            "shards": ffn_shards,
            "target_modules": [
                {"layer_idx": int(layer_idx), "module_name": str(module_name)}
                for layer_idx, module_name in target_modules
            ],
        },
        "tensor_layout": "[B,T,D]",
        "trace_dtype": _trace_dtype_name(store_dtype),
        "padding_mask_semantics": "True means padding token; False means real token",
        "topk_policy_note": "padding token must be excluded by downstream Token Survival metrics",
        "contains_core_cache": False,
        "contains_oer_oracle": False,
        "contains_metrics": False,
        "used_frontend_shortcut": bool(used_frontend_shortcut),
    }
    with _manifest_path(output_root).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    if owns_model and device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "manifest": manifest,
        "output_dir": output_root,
        "manifest_path": _manifest_path(output_root),
        "index_path": index_path,
    }


def export_fp32_token_trace_cache(
    cfg: Any | None = None,
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    cfg = _default_cfg() if cfg is None else cfg
    return build_fp32_token_trace_cache(
        cfg,
        output_dir=output_dir,
        model=model,
        eval_loader=eval_loader,
        force=force,
    )


def main(cfg: Any | None = None) -> dict[str, Any]:
    return export_fp32_token_trace_cache(cfg)


__all__ = [
    "BLOCK_TRACE_DIRNAME",
    "CACHE_DIRNAME",
    "CACHE_VERSION",
    "DEFAULT_TRACE_DTYPE",
    "FFN_TRACE_DIRNAME",
    "INDEX_FILENAME",
    "MANIFEST_FILENAME",
    "build_fp32_token_trace_cache",
    "export_fp32_token_trace_cache",
    "main",
]



