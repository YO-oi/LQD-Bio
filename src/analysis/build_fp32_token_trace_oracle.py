from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from src.export.export_fp32_core_cache import (
        _atomic_json_dump,
        _atomic_torch_save,
        _autocast_context,
        _build_eval_loader,
        _cfg_get,
        _checkpoint_path_for_manifest,
        _checkpoint_sha256_for_manifest,
        _dataset_cfg,
        _default_cfg,
        _device,
        _eval_csv,
        _file_sha256,
        _move_batch_to_device,
        _relative_to_parent,
        _nested_get,
        _restore_fp32_teacher,
        _sample_order_hash,
        _sample_ids_from_batch,
        backbone_features_for_cache,
    )
    from src.export.export_fp32_token_trace_cache import _forward_for_trace, _prepare_encoder_input_and_mask
    from src.utils.token_trace_hook import FFN_MODULE_NAMES, TokenTraceHookManager, TokenTraceRecord, resolve_full_layer_indices
except ImportError:  # pragma: no cover - direct package-relative debugging.
    from ..export.export_fp32_core_cache import (
        _atomic_json_dump,
        _atomic_torch_save,
        _autocast_context,
        _build_eval_loader,
        _cfg_get,
        _checkpoint_path_for_manifest,
        _checkpoint_sha256_for_manifest,
        _dataset_cfg,
        _default_cfg,
        _device,
        _eval_csv,
        _file_sha256,
        _move_batch_to_device,
        _relative_to_parent,
        _nested_get,
        _restore_fp32_teacher,
        _sample_order_hash,
        _sample_ids_from_batch,
        backbone_features_for_cache,
    )
    from ..export.export_fp32_token_trace_cache import _forward_for_trace, _prepare_encoder_input_and_mask
    from ..utils.token_trace_hook import FFN_MODULE_NAMES, TokenTraceHookManager, TokenTraceRecord, resolve_full_layer_indices


ORACLE_VERSION = 2
ORACLE_PROTOCOL_VERSION = "token_trace_oracle.v2"
ORACLE_FILENAME = "token_trace_oracle_eval.pt"
MANIFEST_FILENAME = "token_trace_oracle_eval.manifest.json"
ORACLE_STORAGE_FORMAT = "multi_position_sharded_v2"
LEGACY_ORACLE_STORAGE_FORMAT = "layer_sample_sharded_v1"
ORACLE_SHARD_DIRNAME = "token_trace_oracle_eval"
POSITIONS = ("block_output", "input_activation")
INTERNAL_TRACE_POSITIONS = ("block_output", "input")
BLOCK_MODULE_NAME = "block"
TOPK_UNIT = "token"
TOPK_SOURCE = "fp32_token_energy"
VALID_UNIT = "valid_token"
TOPK_K_POLICY = "ceil(topk_ratio * valid_unit_count)"
TOPK_MIN_K = 1
TOPK_MAX_K = None
PADDING_EXCLUDED = True
LAYOUT_PROTOCOL_VERSION = 1
CHECKPOINT_HASH_TYPE = "sha256"


def _oracle_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_TOKEN_TRACE_ORACLE", {})
    return value if isinstance(value, Mapping) else {}


def _token_survival_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FFN_TOKEN_SURVIVAL_PROBE", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    oracle_cfg = _oracle_cfg(cfg)
    if key in oracle_cfg:
        return oracle_cfg[key]
    token_cfg = _token_survival_cfg(cfg)
    if key in token_cfg:
        return token_cfg[key]
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


def _output_path(cfg: Any, output_path: str | Path | None = None) -> Path:
    if output_path is not None:
        return Path(output_path)
    configured = _setting_get(cfg, "OUTPUT_PATH", None)
    if configured is not None:
        return Path(configured)
    output_dir = _setting_get(cfg, "OUTPUT_DIR", None)
    if output_dir is not None:
        return Path(output_dir) / ORACLE_FILENAME
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "cache" / "fp32_reference" / ORACLE_FILENAME


def _manifest_path(oracle_path: str | Path) -> Path:
    path = Path(oracle_path)
    if path.name == ORACLE_FILENAME:
        return path.with_name(MANIFEST_FILENAME)
    return path.with_suffix(path.suffix + ".manifest.json")


def _oracle_shard_root(oracle_path: str | Path) -> Path:
    path = Path(oracle_path)
    if path.name == "token_trace_oracle_eval.pt":
        return path.with_suffix("")
    return path.parent / ORACLE_SHARD_DIRNAME


def _resolve_oracle_relative_path(oracle_path: str | Path, value: str | Path) -> Path:
    text = _portable_path_text(value)
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


def _portable_path_text(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def _normalise_oracle_header_paths(header: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(header)
    if isinstance(result.get("shards"), Mapping):
        result["shards"] = {str(key): _portable_path_text(value) for key, value in dict(result["shards"]).items()}
    manifest = result.get("manifest")
    if isinstance(manifest, Mapping):
        manifest_copy = dict(manifest)
        if isinstance(manifest_copy.get("shards"), Mapping):
            manifest_copy["shards"] = {
                str(key): _portable_path_text(value) for key, value in dict(manifest_copy["shards"]).items()
            }
        for key in ("shard_root", "manifest_path"):
            if key in manifest_copy and manifest_copy[key] not in (None, ""):
                manifest_copy[key] = _portable_path_text(manifest_copy[key])
        if isinstance(manifest_copy.get("shard_paths"), Sequence) and not isinstance(
            manifest_copy.get("shard_paths"), (str, bytes)
        ):
            manifest_copy["shard_paths"] = [_portable_path_text(value) for value in manifest_copy["shard_paths"]]
        result["manifest"] = manifest_copy
    return result


def _row_key(sample_id: Any, layer_idx: int, module_name: str, position: str) -> str:
    return f"{str(sample_id)}|L{int(layer_idx)}|{str(module_name)}|{str(position)}"


def _position_shard_path(shard_root: Path, position: str) -> Path:
    if position not in POSITIONS:
        raise ValueError(f"unsupported token trace oracle position: {position!r}")
    return shard_root / f"{position}.pt"


def _position_from_record(record: TokenTraceRecord) -> str:
    if str(record.tensor_position) == "block_output":
        if str(record.module_name) != BLOCK_MODULE_NAME:
            raise ValueError(f"block_output oracle rows must use module_name='block', got {record.module_name!r}")
        return "block_output"
    if str(record.tensor_position) == "input":
        if str(record.module_name) not in FFN_MODULE_NAMES:
            raise ValueError(f"input_activation oracle rows must use fc1/fc2, got {record.module_name!r}")
        return "input_activation"
    raise ValueError(f"token trace oracle v2 does not accept tensor_position={record.tensor_position!r}")


def _input_target_modules(layer_indices: Sequence[int]) -> tuple[tuple[int, str], ...]:
    return tuple((int(layer_idx), str(module_name)) for layer_idx in layer_indices for module_name in FFN_MODULE_NAMES)


def _make_rows() -> dict[str, list[Any]]:
    return {
        "sample_id": [],
        "layer_idx": [],
        "module_name": [],
        "position": [],
        "topk_token_indices": [],
        "topk_token_energy": [],
        "topk_source": [],
        "topk_k": [],
        "valid_token_count": [],
        "layout_info": [],
    }


def _topk_ratio(cfg: Any) -> float:
    metrics_cfg = _cfg_get(cfg, "METRICS", {})
    fallback = _nested_get(metrics_cfg, "SCALAR_OER_TOP_RATIO", 0.001) if isinstance(metrics_cfg, Mapping) else 0.001
    ratio = float(_setting_get(cfg, "TOKEN_TOPK_RATIO", fallback))
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"TOKEN_TOPK_RATIO must be in (0, 1], got {ratio}")
    return ratio


def _checkpoint_protocol_fields(cfg: Any) -> dict[str, Any]:
    checkpoint_sha256 = _checkpoint_sha256_for_manifest(cfg)
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hash": checkpoint_sha256,
        "checkpoint_hash_type": CHECKPOINT_HASH_TYPE,
    }


def _token_topk_protocol_fields(topk_ratio: float) -> dict[str, Any]:
    return {
        "topk_unit": TOPK_UNIT,
        "topk_source": TOPK_SOURCE,
        "topk_ratio": float(topk_ratio),
        "topk_k": "per_row",
        "topk_k_policy": TOPK_K_POLICY,
        "topk_min_k": TOPK_MIN_K,
        "topk_max_k": TOPK_MAX_K,
        "valid_unit": VALID_UNIT,
        "valid_token_count": "per_row",
        "valid_unit_count_field": "valid_token_count",
        "padding_excluded": PADDING_EXCLUDED,
        "layout_protocol_version": LAYOUT_PROTOCOL_VERSION,
    }


def _layout_info(record: TokenTraceRecord) -> dict[str, Any]:
    return {
        "original_layout": str(record.original_layout),
        "original_shape": tuple(int(item) for item in record.original_shape),
        "canonical_layout": str(record.canonical_layout),
        "canonical_shape": tuple(int(item) for item in record.canonical_shape),
        "layout_inference_source": str(record.layout_inference_source),
        "padding_mask_available": bool(record.padding_mask_available),
        "padding_mask_aligned": bool(record.padding_mask_aligned),
        "mask_status": str(record.mask_status),
    }


def _resolve_topk_k(valid_unit_count: int, topk_ratio: float) -> int:
    count = int(valid_unit_count)
    if count <= 0:
        raise ValueError("valid_unit_count must be positive when resolving token top-k")
    k = int(math.ceil(count * float(topk_ratio)))
    k = max(int(TOPK_MIN_K), k)
    if TOPK_MAX_K is not None:
        k = min(k, int(TOPK_MAX_K))
    return min(k, count)


def _topk_token_energy_indices(
    tensor_btc: Tensor,
    padding_mask_bt: Tensor,
    *,
    topk_ratio: float,
    position: str,
) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
    if tensor_btc.ndim != 3:
        raise ValueError(f"{position} tensor must be canonical [B,T,C], got shape={tuple(tensor_btc.shape)}")
    if padding_mask_bt.ndim != 2 or tuple(padding_mask_bt.shape) != tuple(tensor_btc.shape[:2]):
        raise ValueError(
            f"padding_mask must be [B,T] and align with {position}; "
            f"mask_shape={tuple(padding_mask_bt.shape)}, tensor_shape={tuple(tensor_btc.shape)}"
        )
    tensor = tensor_btc.detach().float().cpu()
    mask = padding_mask_bt.detach().cpu().bool()
    energy = tensor.pow(2).sum(dim=-1)
    valid_token_count = (~mask).sum(dim=1).long()
    topk_indices: list[Tensor] = []
    topk_energy: list[Tensor] = []
    topk_k_rows: list[int] = []
    for sample_idx in range(int(tensor.shape[0])):
        valid_indices = torch.nonzero(~mask[sample_idx], as_tuple=False).flatten().long()
        valid_count = int(valid_indices.numel())
        if valid_count <= 0:
            raise ValueError(f"sample index {sample_idx} has zero valid tokens")
        k = _resolve_topk_k(valid_count, topk_ratio)
        values = energy[sample_idx, valid_indices]
        top_values, local_indices = torch.topk(values, k=k, largest=True, sorted=True)
        topk_indices.append(valid_indices[local_indices].long().contiguous())
        topk_energy.append(top_values.float().contiguous())
        topk_k_rows.append(int(k))
    return topk_indices, topk_energy, torch.tensor(topk_k_rows, dtype=torch.long), valid_token_count


def _append_record_rows(*, record: TokenTraceRecord, topk_ratio: float, rows: dict[str, list[Any]]) -> None:
    position = _position_from_record(record)
    padding_mask = record.padding_mask.detach().cpu().bool().contiguous()
    topk_indices, topk_energy, topk_k, valid_counts = _topk_token_energy_indices(
        record.tensor,
        padding_mask,
        topk_ratio=topk_ratio,
        position=position,
    )
    layout_info = _layout_info(record)
    for idx, sample_id in enumerate(record.sample_id):
        rows["sample_id"].append(str(sample_id))
        rows["layer_idx"].append(int(record.layer_idx))
        rows["module_name"].append(str(record.module_name))
        rows["position"].append(position)
        rows["topk_token_indices"].append(topk_indices[idx])
        rows["topk_token_energy"].append(topk_energy[idx])
        rows["topk_source"].append(TOPK_SOURCE)
        rows["topk_k"].append(int(topk_k[idx].item()))
        rows["valid_token_count"].append(int(valid_counts[idx].item()))
        rows["layout_info"].append(dict(layout_info))


def _collect_oracle_rows(
    cfg: Any,
    *,
    model: nn.Module,
    eval_loader: DataLoader,
) -> dict[str, Any]:
    device = _device(cfg)
    topk_ratio = _topk_ratio(cfg)
    model = model.to(device)
    model.eval()
    layer_indices = resolve_full_layer_indices(model)
    input_targets = _input_target_modules(layer_indices)
    expected_batch_keys = {
        (int(layer_idx), BLOCK_MODULE_NAME, "block_output")
        for layer_idx in layer_indices
    } | {
        (int(layer_idx), str(module_name), "input_activation")
        for layer_idx, module_name in input_targets
    }
    rows_by_position = {position: _make_rows() for position in POSITIONS}
    sample_id_order: list[str] = []
    seen_sample_ids: set[str] = set()
    num_batches = 0
    used_frontend_shortcut = True

    for batch in eval_loader:
        batch_sample_ids = _sample_ids_from_batch(batch)
        repeated = sorted(set(batch_sample_ids) & seen_sample_ids)
        if repeated:
            raise ValueError(f"duplicate sample_id values across eval_loader: {repeated[:8]}")
        if len(set(batch_sample_ids)) != len(batch_sample_ids):
            raise ValueError("duplicate sample_id values within batch; token trace oracle requires unique sample ids")
        num_batches += 1

        moved = _move_batch_to_device(batch, device)
        waveform = moved["waveform"]
        waveform_padding_mask = moved.get("padding_mask")
        with torch.no_grad(), _autocast_context(cfg, device):
            prepared = _prepare_encoder_input_and_mask(model, waveform, waveform_padding_mask)
            if prepared is None:
                used_frontend_shortcut = False
                _, token_padding_mask = backbone_features_for_cache(model, waveform, waveform_padding_mask)
            else:
                token_padding_mask = prepared[2]

            manager = TokenTraceHookManager(
                model,
                layer_indices=layer_indices,
                target_modules=input_targets,
                positions=INTERNAL_TRACE_POSITIONS,
                require_padding_mask=True,
            )
            manager.set_batch_context(sample_id=batch_sample_ids, padding_mask=token_padding_mask.detach().cpu(), clear_cache=True)
            manager.register_hooks()
            try:
                _forward_for_trace(model, waveform=waveform, padding_mask=waveform_padding_mask, prepared=prepared)
                records = manager.pop_cache()
            finally:
                manager.clear_hooks()
                manager.clear_batch_context()

        observed_batch_keys: set[tuple[int, str, str]] = set()
        for record in records:
            if not isinstance(record, TokenTraceRecord):
                raise TypeError(f"TokenTraceHookManager returned unexpected record type: {type(record)!r}")
            position = _position_from_record(record)
            key = (int(record.layer_idx), str(record.module_name), str(position))
            if key in observed_batch_keys:
                raise RuntimeError(f"duplicate token trace oracle record in batch: {key}")
            observed_batch_keys.add(key)
            _append_record_rows(record=record, topk_ratio=topk_ratio, rows=rows_by_position[position])
        if observed_batch_keys != expected_batch_keys:
            missing = sorted(expected_batch_keys - observed_batch_keys)
            extra = sorted(observed_batch_keys - expected_batch_keys)
            raise RuntimeError(f"token trace oracle v2 record set mismatch; missing={missing[:8]}, extra={extra[:8]}")

        seen_sample_ids.update(batch_sample_ids)
        sample_id_order.extend(batch_sample_ids)

    if not sample_id_order:
        raise ValueError("eval_loader produced no samples for token trace oracle")
    for position, rows in rows_by_position.items():
        expected = len(sample_id_order) * len(layer_indices)
        if position == "input_activation":
            expected *= len(FFN_MODULE_NAMES)
        actual = len(rows["sample_id"])
        if actual != expected:
            raise RuntimeError(f"{position} token trace oracle row_count mismatch: {actual} vs {expected}")
    return {
        "rows_by_position": rows_by_position,
        "layer_indices": tuple(int(item) for item in layer_indices),
        "input_targets": tuple((int(layer_idx), str(module_name)) for layer_idx, module_name in input_targets),
        "sample_id_order": list(sample_id_order),
        "sample_id_set": sorted(seen_sample_ids),
        "topk_ratio": float(topk_ratio),
        "num_batches": int(num_batches),
        "used_frontend_shortcut": bool(used_frontend_shortcut),
    }


def _shared_identity(cfg: Any, *, sample_id_order: Sequence[str]) -> dict[str, Any]:
    return {
        "dataset_name": _dataset_name(cfg),
        "split": "eval",
        "eval_csv": str(_eval_csv(cfg).expanduser().resolve()),
        "eval_csv_sha256": _file_sha256(_eval_csv(cfg)),
        "checkpoint_path": _checkpoint_path_for_manifest(cfg),
        **_checkpoint_protocol_fields(cfg),
        "num_samples": len(sample_id_order),
        "sample_order_hash": _sample_order_hash(sample_id_order),
        "sample_order_hash_algorithm": "sha256-json-list",
    }


def _row_index(rows: Mapping[str, Sequence[Any]]) -> dict[str, int]:
    return {
        _row_key(sample_id, int(layer_idx), str(module_name), str(position)): idx
        for idx, (sample_id, layer_idx, module_name, position) in enumerate(
            zip(rows["sample_id"], rows["layer_idx"], rows["module_name"], rows["position"])
        )
    }


def _position_payload(*, rows: Mapping[str, Sequence[Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "oracle_version": ORACLE_VERSION,
        "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
        "cache_type": "fp32_token_trace_oracle_position_shard",
        "parent_cache_type": "fp32_token_trace_oracle",
        "manifest": dict(manifest),
        "sample_id": list(rows["sample_id"]),
        "layer_idx": torch.tensor([int(item) for item in rows["layer_idx"]], dtype=torch.long),
        "module_name": [str(item) for item in rows["module_name"]],
        "position": [str(item) for item in rows["position"]],
        "topk_token_indices": [torch.as_tensor(item, dtype=torch.long).detach().cpu().contiguous() for item in rows["topk_token_indices"]],
        "topk_token_energy": [torch.as_tensor(item, dtype=torch.float32).detach().cpu().contiguous() for item in rows["topk_token_energy"]],
        "topk_source": [str(item) for item in rows["topk_source"]],
        "topk_k": torch.tensor([int(item) for item in rows["topk_k"]], dtype=torch.long),
        "valid_token_count": torch.tensor([int(item) for item in rows["valid_token_count"]], dtype=torch.long),
        "layout_info": [dict(item) for item in rows["layout_info"]],
        "row_index": _row_index(rows),
        "sample_id_set": sorted(set(str(item) for item in rows["sample_id"])),
    }


def _subset_position_shard_payload(shard: Mapping[str, Any], keep_rows: Sequence[int]) -> dict[str, Any]:
    indices = [int(item) for item in keep_rows]
    result = dict(shard)
    row_fields = (
        "sample_id",
        "module_name",
        "position",
        "topk_token_indices",
        "topk_token_energy",
        "topk_source",
        "layout_info",
    )
    tensor_row_fields = ("layer_idx", "topk_k", "valid_token_count")
    for field in row_fields:
        values = list(shard.get(field, []))
        result[field] = [values[idx] for idx in indices]
    for field in tensor_row_fields:
        values = torch.as_tensor(shard[field]).detach().cpu()
        result[field] = values.index_select(0, torch.tensor(indices, dtype=torch.long)) if indices else values[:0]
    result["row_index"] = {
        _row_key(sample_id, int(layer_idx), str(module_name), str(position)): idx
        for idx, (sample_id, layer_idx, module_name, position) in enumerate(
            zip(
                result["sample_id"],
                torch.as_tensor(result["layer_idx"], dtype=torch.long).reshape(-1).tolist(),
                result["module_name"],
                result["position"],
            )
        )
    }
    result["sample_id_set"] = sorted(set(str(item) for item in result["sample_id"]))
    if isinstance(result.get("manifest"), Mapping):
        manifest = dict(result["manifest"])
        manifest["row_count"] = int(len(indices))
        result["manifest"] = manifest
    return result


def _position_coverage(position: str, rows: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    layer_indices = sorted({int(item) for item in rows["layer_idx"]})
    module_names = sorted({str(item) for item in rows["module_name"]})
    coverage: dict[str, Any] = {
        "position": position,
        "layer_indices": layer_indices,
        "module_names": module_names,
        "row_count": int(len(rows["sample_id"])),
    }
    if position == "input_activation":
        coverage["target_modules"] = [
            {"layer_idx": int(layer_idx), "module_name": str(module_name)}
            for layer_idx in layer_indices
            for module_name in module_names
        ]
    return coverage


def _materialize_header_from_shards(header: dict[str, Any], oracle_path: str | Path, *, positions: Sequence[str] | None = None) -> dict[str, Any]:
    path = Path(oracle_path)
    requested = list(header.get("available_positions", [])) if positions is None else [str(item) for item in positions]
    sample_id_rows: list[str] = []
    layer_idx_rows: list[int] = []
    module_name_rows: list[str] = []
    position_rows: list[str] = []
    topk_token_indices_rows: list[Tensor] = []
    topk_token_energy_rows: list[Tensor] = []
    topk_source_rows: list[str] = []
    topk_k_rows: list[int] = []
    valid_token_count_rows: list[int] = []
    layout_info_rows: list[dict[str, Any]] = []

    for position in requested:
        shard_rel = _position_shard_rel(header, position)
        shard_path = _resolve_oracle_relative_path(path, shard_rel)
        if not shard_path.exists() or not shard_path.is_file():
            raise FileNotFoundError(f"token trace oracle v2 header declares {position!r} but shard is missing: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard, Mapping):
            raise TypeError(f"token trace oracle shard must save a mapping, got {type(shard)!r}: {shard_path}")
        sample_ids = list(shard.get("sample_id", []))
        layer_tensor = torch.as_tensor(shard["layer_idx"], dtype=torch.long).reshape(-1)
        module_names = [str(item) for item in shard.get("module_name", [])]
        positions_for_rows = [str(item) for item in shard.get("position", [])]
        valid_tensor = torch.as_tensor(shard["valid_token_count"], dtype=torch.long).reshape(-1)
        topk_k_tensor = torch.as_tensor(shard["topk_k"], dtype=torch.long).reshape(-1) if "topk_k" in shard else None
        if len(module_names) != len(sample_ids) or len(positions_for_rows) != len(sample_ids):
            raise RuntimeError(f"token trace oracle shard row fields have inconsistent lengths: {shard_path}")
        for idx, sample_id in enumerate(sample_ids):
            sample_id_rows.append(str(sample_id))
            layer_idx_rows.append(int(layer_tensor[idx].item()))
            module_name_rows.append(str(module_names[idx]))
            position_rows.append(str(positions_for_rows[idx]))
            topk_token_indices_rows.append(torch.as_tensor(shard["topk_token_indices"][idx], dtype=torch.long).detach().cpu().contiguous())
            topk_token_energy_rows.append(torch.as_tensor(shard["topk_token_energy"][idx], dtype=torch.float32).detach().cpu().contiguous())
            topk_source_rows.append(str(shard["topk_source"][idx]))
            topk_k_rows.append(int(topk_k_tensor[idx].item()) if topk_k_tensor is not None else int(torch.as_tensor(shard["topk_token_indices"][idx]).numel()))
            valid_token_count_rows.append(int(valid_tensor[idx].item()))
            layout_info_rows.append(dict(shard["layout_info"][idx]))

    header.update(
        {
            "sample_id": sample_id_rows,
            "layer_idx": torch.tensor(layer_idx_rows, dtype=torch.long),
            "module_name": module_name_rows,
            "position": position_rows,
            "topk_token_indices": topk_token_indices_rows,
            "topk_token_energy": topk_token_energy_rows,
            "topk_source": topk_source_rows,
            "topk_k": torch.tensor(topk_k_rows, dtype=torch.long),
            "valid_token_count": torch.tensor(valid_token_count_rows, dtype=torch.long),
            "layout_info": layout_info_rows,
            "row_index": {
                _row_key(sample_id, layer_idx, module_name, position): idx
                for idx, (sample_id, layer_idx, module_name, position) in enumerate(
                    zip(sample_id_rows, layer_idx_rows, module_name_rows, position_rows)
                )
            },
            "sample_id_set": sorted(set(sample_id_rows)),
        }
    )
    return header


def _position_shard_rel(oracle: Mapping[str, Any], position: str) -> str:
    shards = oracle.get("shards")
    if isinstance(shards, Mapping):
        value = shards.get(position)
        if value is not None:
            return str(value)
    manifest = oracle.get("manifest", {})
    if isinstance(manifest, Mapping):
        shards = manifest.get("shards")
        if isinstance(shards, Mapping) and shards.get(position) is not None:
            return str(shards[position])
    raise KeyError(f"token trace oracle v2 missing shard path for requested position={position!r}")


def is_sharded_fp32_token_trace_oracle(oracle: Mapping[str, Any]) -> bool:
    manifest = oracle.get("manifest", {})
    manifest_format = manifest.get("storage_format") if isinstance(manifest, Mapping) else None
    return bool(oracle.get("sharded", False)) or str(oracle.get("storage_format", manifest_format)) in {
        ORACLE_STORAGE_FORMAT,
        LEGACY_ORACLE_STORAGE_FORMAT,
    }


def iter_fp32_token_trace_oracle_shards(
    oracle: Mapping[str, Any],
    oracle_path: str | Path,
    *,
    layer_idx: int | None = None,
    position: str | None = None,
    module_name: str | None = None,
):
    storage_format = str(oracle.get("storage_format", oracle.get("manifest", {}).get("storage_format", "")))
    if storage_format == ORACLE_STORAGE_FORMAT:
        requested = list(oracle.get("available_positions", [])) if position is None else [str(position)]
        for requested_position in requested:
            shard_rel = _position_shard_rel(oracle, requested_position)
            shard_path = _resolve_oracle_relative_path(oracle_path, shard_rel)
            if not shard_path.exists() or not shard_path.is_file():
                raise FileNotFoundError(f"token trace oracle v2 shard is missing: {shard_path}")
            shard = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard, Mapping):
                raise TypeError(f"token trace oracle shard must save a mapping, got {type(shard)!r}: {shard_path}")
            if layer_idx is None and module_name is None:
                yield dict(shard)
                continue
            keep_rows = []
            layer_tensor = torch.as_tensor(shard["layer_idx"], dtype=torch.long).reshape(-1)
            module_names = [str(item) for item in shard.get("module_name", [])]
            for idx in range(len(module_names)):
                if layer_idx is not None and int(layer_tensor[idx].item()) != int(layer_idx):
                    continue
                if module_name is not None and str(module_names[idx]) != str(module_name):
                    continue
                keep_rows.append(idx)
            if keep_rows:
                yield _subset_position_shard_payload(shard, keep_rows)
        return

    if not is_sharded_fp32_token_trace_oracle(oracle):
        yield dict(oracle)
        return
    shards = oracle.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise ValueError("legacy sharded token trace oracle manifest is missing shards")
    for entry in shards:
        if not isinstance(entry, Mapping):
            raise TypeError(f"token trace shard entry must be a mapping, got {type(entry)!r}")
        if layer_idx is not None and int(entry.get("layer_idx", -1)) != int(layer_idx):
            continue
        shard_rel = entry.get("path")
        if shard_rel is None:
            raise KeyError("token trace shard entry missing path")
        shard_path = _resolve_oracle_relative_path(oracle_path, shard_rel)
        shard = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard, Mapping):
            raise TypeError(f"token trace oracle shard must save a mapping, got {type(shard)!r}: {shard_path}")
        yield dict(shard)


def load_fp32_token_trace_oracle(
    oracle_path: str | Path,
    *,
    materialize: bool = False,
    positions: Sequence[str] | None = None,
) -> dict[str, Any]:
    path = Path(oracle_path)
    if not path.exists():
        raise FileNotFoundError(f"FP32 token trace oracle does not exist: {path}")
    oracle = torch.load(path, map_location="cpu")
    if not isinstance(oracle, Mapping):
        raise TypeError(f"FP32 token trace oracle must be a mapping, got {type(oracle)!r}")
    result = _normalise_oracle_header_paths(oracle)
    if not materialize:
        return result
    if str(result.get("storage_format", result.get("manifest", {}).get("storage_format", ""))) == ORACLE_STORAGE_FORMAT:
        return _materialize_header_from_shards(result, path, positions=positions)
    if is_sharded_fp32_token_trace_oracle(result):
        sample_id_rows: list[str] = []
        layer_idx_rows: list[int] = []
        position_rows: list[str] = []
        topk_token_indices_rows: list[Tensor] = []
        topk_token_energy_rows: list[Tensor] = []
        topk_source_rows: list[str] = []
        topk_k_rows: list[int] = []
        valid_token_count_rows: list[int] = []
        layout_info_rows: list[dict[str, Any]] = []
        for shard in iter_fp32_token_trace_oracle_shards(result, path):
            sample_ids = list(shard.get("sample_id", []))
            layer_tensor = torch.as_tensor(shard["layer_idx"], dtype=torch.long).reshape(-1)
            valid_tensor = torch.as_tensor(shard["valid_token_count"], dtype=torch.long).reshape(-1)
            topk_k_tensor = torch.as_tensor(shard["topk_k"], dtype=torch.long).reshape(-1) if "topk_k" in shard else None
            for idx, sample_id in enumerate(sample_ids):
                sample_id_rows.append(str(sample_id))
                layer_idx_rows.append(int(layer_tensor[idx].item()))
                position_rows.append(str(shard["position"][idx]))
                topk_token_indices_rows.append(torch.as_tensor(shard["topk_token_indices"][idx], dtype=torch.long).detach().cpu().contiguous())
                topk_token_energy_rows.append(torch.as_tensor(shard["topk_token_energy"][idx], dtype=torch.float32).detach().cpu().contiguous())
                topk_source_rows.append(str(shard["topk_source"][idx]))
                topk_k_rows.append(int(topk_k_tensor[idx].item()) if topk_k_tensor is not None else int(torch.as_tensor(shard["topk_token_indices"][idx]).numel()))
                valid_token_count_rows.append(int(valid_tensor[idx].item()))
                layout_info_rows.append(dict(shard["layout_info"][idx]))
        result.update(
            {
                "sample_id": sample_id_rows,
                "layer_idx": torch.tensor(layer_idx_rows, dtype=torch.long),
                "module_name": [BLOCK_MODULE_NAME] * len(sample_id_rows),
                "position": position_rows,
                "topk_token_indices": topk_token_indices_rows,
                "topk_token_energy": topk_token_energy_rows,
                "topk_source": topk_source_rows,
                "topk_k": torch.tensor(topk_k_rows, dtype=torch.long),
                "valid_token_count": torch.tensor(valid_token_count_rows, dtype=torch.long),
                "layout_info": layout_info_rows,
                "row_index": {
                    _row_key(sample_id, layer_idx, BLOCK_MODULE_NAME, "block_output"): idx
                    for idx, (sample_id, layer_idx) in enumerate(zip(sample_id_rows, layer_idx_rows))
                },
                "sample_id_set": sorted(set(sample_id_rows)),
            }
        )
    return result


def build_fp32_token_trace_oracle(
    cfg: Any,
    *,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Build an in-memory FP32 token-energy oracle for block outputs and FFN inputs."""

    owns_model = model is None
    if model is None:
        model = _restore_fp32_teacher(cfg)
    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)
    collected = _collect_oracle_rows(cfg, model=model, eval_loader=eval_loader)
    sample_id_order = collected["sample_id_order"]
    shared_identity = _shared_identity(cfg, sample_id_order=sample_id_order)
    topk_ratio = float(collected["topk_ratio"])
    manifest = {
        "oracle_version": ORACLE_VERSION,
        "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
        "cache_type": "fp32_token_trace_oracle",
        "oracle_source": "fp32_multi_position_token_energy",
        "stage": "quant_injection_pre_fp32_token_trace_oracle",
        "storage_format": ORACLE_STORAGE_FORMAT,
        "sharded": False,
        "positions": list(POSITIONS),
        "available_positions": list(POSITIONS),
        "shared_identity": dict(shared_identity),
        **shared_identity,
        "row_count": int(sum(len(rows["sample_id"]) for rows in collected["rows_by_position"].values())),
        "num_layers": len(collected["layer_indices"]),
        "layer_indices": [int(item) for item in collected["layer_indices"]],
        "input_activation_target_modules": [
            {"layer_idx": int(layer_idx), "module_name": str(module_name)}
            for layer_idx, module_name in collected["input_targets"]
        ],
        "position_coverage": {
            position: _position_coverage(position, rows)
            for position, rows in collected["rows_by_position"].items()
        },
        **_token_topk_protocol_fields(topk_ratio),
        "token_energy_definition": {
            "block_output": "sum_c(fp32_block_output[B,T,C] ** 2)",
            "input_activation": "sum_c(fp32_input_activation[B,T,C] ** 2)",
        },
        "layout_protocol": "canonical BTC via src.utils.layout_utils",
        "padding_policy": "padding token excluded before token top-k; padding_mask True means padding",
        "contains_full_token_tensor": False,
        "contains_oer_oracle": False,
        "required_for": ["block_output_token_survival", "input_activation_integer_grid_survival"],
        "forward_passes_per_batch": 1 if collected["used_frontend_shortcut"] else 2,
        "used_frontend_shortcut": bool(collected["used_frontend_shortcut"]),
        "num_batches": int(collected["num_batches"]),
    }
    result: dict[str, Any] = {
        "oracle_version": ORACLE_VERSION,
        "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
        "cache_type": "fp32_token_trace_oracle",
        "manifest": manifest,
        "storage_format": ORACLE_STORAGE_FORMAT,
        "sharded": False,
        "positions": list(POSITIONS),
        "available_positions": list(POSITIONS),
        "sample_id_order": list(sample_id_order),
        "sample_id_set": sorted(collected["sample_id_set"]),
    }
    for position, rows in collected["rows_by_position"].items():
        result[position] = _position_payload(rows=rows, manifest={"position": position, **manifest})
    if owns_model and _device(cfg).type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_and_save_fp32_token_trace_oracle_sharded(
    cfg: Any,
    oracle_path: str | Path,
    *,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Build a v2 multi-position FP32 token-energy oracle."""

    path = Path(oracle_path)
    shard_root = _oracle_shard_root(path)
    owns_model = model is None
    if model is None:
        model = _restore_fp32_teacher(cfg)
    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)
    collected = _collect_oracle_rows(cfg, model=model, eval_loader=eval_loader)
    sample_id_order = collected["sample_id_order"]
    shared_identity = _shared_identity(cfg, sample_id_order=sample_id_order)
    topk_ratio = float(collected["topk_ratio"])
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_records: dict[str, str] = {}
    shard_sha256: dict[str, str] = {}
    shard_entries: list[dict[str, Any]] = []
    row_count = 0
    position_coverage: dict[str, Any] = {}

    for position in POSITIONS:
        rows = collected["rows_by_position"][position]
        shard_path = _position_shard_path(shard_root, position)
        rel_path = _relative_to_parent(shard_path, path.parent)
        shard_manifest = {
            "oracle_version": ORACLE_VERSION,
            "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
            "cache_type": "fp32_token_trace_oracle_position_shard",
            "parent_cache_type": "fp32_token_trace_oracle",
            "storage_format": ORACLE_STORAGE_FORMAT,
            "position": position,
            "row_count": int(len(rows["sample_id"])),
            "shared_identity": dict(shared_identity),
            **_token_topk_protocol_fields(topk_ratio),
        }
        payload = _position_payload(rows=rows, manifest=shard_manifest)
        _atomic_torch_save(payload, shard_path)
        digest = _file_sha256(shard_path)
        shard_records[position] = rel_path
        shard_sha256[rel_path] = digest
        shard_entries.append(
            {
                "position": position,
                "path": rel_path,
                "sha256": digest,
                "row_count": int(len(rows["sample_id"])),
            }
        )
        position_coverage[position] = _position_coverage(position, rows)
        row_count += int(len(rows["sample_id"]))

    manifest_path = shard_root / "manifest.json"
    rel_manifest_path = _relative_to_parent(manifest_path, path.parent)
    manifest = {
        "oracle_version": ORACLE_VERSION,
        "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
        "cache_type": "fp32_token_trace_oracle",
        "oracle_source": "fp32_multi_position_token_energy",
        "stage": "quant_injection_pre_fp32_token_trace_oracle",
        "storage_format": ORACLE_STORAGE_FORMAT,
        "sharded": True,
        "shard_root": _relative_to_parent(shard_root, path.parent),
        "positions": list(POSITIONS),
        "available_positions": list(POSITIONS),
        "shards": dict(shard_records),
        "manifest_path": rel_manifest_path,
        "shared_identity": dict(shared_identity),
        **shared_identity,
        "row_count": int(row_count),
        "num_layers": len(collected["layer_indices"]),
        "layer_indices": [int(item) for item in collected["layer_indices"]],
        "input_activation_target_modules": [
            {"layer_idx": int(layer_idx), "module_name": str(module_name)}
            for layer_idx, module_name in collected["input_targets"]
        ],
        "position_coverage": position_coverage,
        **_token_topk_protocol_fields(topk_ratio),
        "token_energy_definition": {
            "block_output": "sum_c(fp32_block_output[B,T,C] ** 2)",
            "input_activation": "sum_c(fp32_input_activation[B,T,C] ** 2)",
        },
        "layout_protocol": "canonical BTC via src.utils.layout_utils",
        "padding_policy": "padding token excluded before token top-k; padding_mask True means padding",
        "contains_full_token_tensor": False,
        "contains_oer_oracle": False,
        "required_for": ["block_output_token_survival", "input_activation_integer_grid_survival"],
        "forward_passes_per_batch": 1 if collected["used_frontend_shortcut"] else 2,
        "used_frontend_shortcut": bool(collected["used_frontend_shortcut"]),
        "num_batches": int(collected["num_batches"]),
        "num_shards": len(shard_records),
        "shard_paths": [entry["path"] for entry in shard_entries],
        "shard_sha256": dict(shard_sha256),
        "shard_entries": list(shard_entries),
    }
    oracle = {
        "oracle_version": ORACLE_VERSION,
        "oracle_protocol_version": ORACLE_PROTOCOL_VERSION,
        "cache_type": "fp32_token_trace_oracle",
        "manifest": manifest,
        "storage_format": ORACLE_STORAGE_FORMAT,
        "sharded": True,
        "shard_root": _relative_to_parent(shard_root, path.parent),
        "positions": list(POSITIONS),
        "available_positions": list(POSITIONS),
        "shards": dict(shard_records),
        "manifest_path": rel_manifest_path,
        "shared_identity": dict(shared_identity),
        "sample_id_order": list(sample_id_order),
        "sample_id_set": sorted(collected["sample_id_set"]),
        "shard_paths": [entry["path"] for entry in shard_entries],
        "shard_sha256": dict(shard_sha256),
        "shard_entries": list(shard_entries),
    }
    _atomic_torch_save(oracle, path)
    _atomic_json_dump(manifest, manifest_path)
    _atomic_json_dump(manifest, _manifest_path(path))
    if owns_model and _device(cfg).type == "cuda":
        torch.cuda.empty_cache()
    return {"oracle": oracle, "oracle_path": path, "manifest_path": _manifest_path(path), "shard_manifest_path": manifest_path}


def save_fp32_token_trace_oracle(oracle: Mapping[str, Any], oracle_path: str | Path) -> Path:
    path = Path(oracle_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(oracle), path)
    manifest = oracle.get("manifest", {})
    with _manifest_path(path).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return path


def export_fp32_token_trace_oracle(
    cfg: Any | None = None,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    cfg = _default_cfg() if cfg is None else cfg
    path = _output_path(cfg, output_path)
    if path.exists() and not force:
        raise FileExistsError(f"FP32 token trace oracle already exists: {path}; pass force=True to rebuild")
    return build_and_save_fp32_token_trace_oracle_sharded(
        cfg,
        path,
        model=model,
        eval_loader=eval_loader,
    )


def main(cfg: Any | None = None) -> dict[str, Any]:
    return export_fp32_token_trace_oracle(cfg)


__all__ = [
    "MANIFEST_FILENAME",
    "ORACLE_FILENAME",
    "ORACLE_PROTOCOL_VERSION",
    "ORACLE_STORAGE_FORMAT",
    "ORACLE_VERSION",
    "POSITIONS",
    "TOPK_SOURCE",
    "build_and_save_fp32_token_trace_oracle_sharded",
    "build_fp32_token_trace_oracle",
    "export_fp32_token_trace_oracle",
    "is_sharded_fp32_token_trace_oracle",
    "iter_fp32_token_trace_oracle_shards",
    "load_fp32_token_trace_oracle",
    "main",
    "save_fp32_token_trace_oracle",
]


if __name__ == "__main__":
    main()
