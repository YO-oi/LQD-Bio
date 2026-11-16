from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.analysis.semantic_geometry_cache import *  # noqa: F401,F403
from src.analysis import semantic_geometry_cache as _legacy_cache


SEMANTIC_GEOMETRY_CACHE_PROTOCOL_VERSION = "semantic_geometry_cache.v2"
SEMANTIC_GEOMETRY_DONE_PROTOCOL_VERSION = "semantic_geometry_done.v1"
SEMANTIC_GEOMETRY_EXPERIMENT_PROTOCOL_VERSION = "semantic_geometry_projection.v2"
SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION = "semantic_geometry_embedding_identity.v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if int(tensor.numel()) == 1:
            return tensor.reshape(()).item()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def stable_json_hash(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    encoded = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_order_hash(sample_ids: Sequence[Any]) -> str:
    return stable_json_hash([str(item) for item in sample_ids])


def qparams_identity_manifest(qparams: Mapping[str, Any] | None, *, source_hint: str | None = None) -> dict[str, Any]:
    if qparams is None:
        return {
            "qparams_identity_type": "none",
            "qparams_identity": None,
            "qparams_source": source_hint or "none",
        }
    if not isinstance(qparams, Mapping):
        raise TypeError(f"qparams identity requires a mapping or None, got {type(qparams)!r}")
    for key in ("qparams_path", "qparams_cache_path", "path"):
        value = qparams.get(key)
        if value is not None:
            return {
                "qparams_identity_type": "file_identity",
                "qparams_identity": file_identity(value),
                "qparams_source": str(value),
            }
    return {
        "qparams_identity_type": "stable_json_hash",
        "qparams_identity": stable_json_hash(qparams),
        "qparams_source": source_hint or str(qparams.get("qparams_source", qparams.get("source", "in_memory_qparams"))),
    }


def build_semantic_embedding_identity(
    *,
    scope: str,
    quant_mode: str,
    granularity: str,
    target_modules: Sequence[Sequence[Any]],
    sample_ids: Sequence[Any],
    eval_csv: str | Path | None,
    calibration_csv: str | Path | None,
    checkpoint: Mapping[str, Any] | None,
    quant_config: Mapping[str, Any] | None,
    compute_logits_if_available: bool,
    embedding_dim: int | None,
    qparams_identity_type: str,
    qparams_identity: Any,
    qparams_source: Any,
) -> dict[str, Any]:
    return {
        "identity_version": SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION,
        "scope": str(scope),
        "quant_mode": str(quant_mode),
        "granularity": str(granularity),
        "target_modules": [[int(layer_idx), str(module_name)] for layer_idx, module_name in target_modules],
        "sample_order_hash": sample_order_hash(sample_ids),
        "eval_csv": file_identity(eval_csv),
        "calibration_csv": file_identity(calibration_csv),
        "checkpoint": None if checkpoint is None else dict(checkpoint),
        "quant_config_hash": stable_json_hash(quant_config or {}),
        "compute_logits_if_available": bool(compute_logits_if_available),
        "embedding_dim": None if embedding_dim is None else int(embedding_dim),
        "qparams_identity_type": str(qparams_identity_type),
        "qparams_identity": qparams_identity,
        "qparams_source": qparams_source,
    }


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    info: dict[str, Any] = {"path": str(resolved)}
    if resolved.exists() and resolved.is_file():
        stat = resolved.stat()
        info.update({
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
        hash_file = bool(stat.st_size <= 256 * 1024 * 1024)
        if hash_file:
            h = hashlib.sha256()
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            info["sha256"] = h.hexdigest()
        else:
            info["sha256"] = None
            info["sha256_skipped_reason"] = "file_larger_than_256MiB"
    else:
        info["exists"] = False
    return info


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def atomic_json_write(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def save_embedding_cache(
    cache: Mapping[str, Any],
    path: str | Path,
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,  # type: ignore[name-defined]
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        _legacy_cache.save_embedding_cache(cache, tmp_path, expected_dim=expected_dim)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def build_experiment_signature(payload: Mapping[str, Any]) -> str:
    return stable_json_hash({
        "protocol_version": SEMANTIC_GEOMETRY_EXPERIMENT_PROTOCOL_VERSION,
        **dict(payload),
    })


def done_marker_path(root: str | Path, shard_key: str) -> Path:
    return Path(root) / "done" / f"{shard_key}.done.json"


def write_done_marker(
    path: str | Path,
    *,
    experiment_signature: str,
    payload: Mapping[str, Any],
    required_paths: Sequence[str | Path],
) -> Path:
    marker = {
        "protocol_version": SEMANTIC_GEOMETRY_DONE_PROTOCOL_VERSION,
        "experiment_protocol_version": SEMANTIC_GEOMETRY_EXPERIMENT_PROTOCOL_VERSION,
        "status": "done",
        "experiment_signature": str(experiment_signature),
        "required_paths": [str(Path(item)) for item in required_paths],
        **dict(payload),
    }
    return atomic_json_write(path, marker)


def done_marker_valid(
    path: str | Path,
    *,
    experiment_signature: str,
    required_paths: Sequence[str | Path],
) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("protocol_version") != SEMANTIC_GEOMETRY_DONE_PROTOCOL_VERSION:
        return False
    if marker.get("experiment_signature") != str(experiment_signature):
        return False
    if marker.get("status") != "done":
        return False
    return all(Path(item).exists() for item in required_paths)


__all__ = list(getattr(_legacy_cache, "__all__", ())) + [
    "SEMANTIC_GEOMETRY_CACHE_PROTOCOL_VERSION",
    "SEMANTIC_GEOMETRY_DONE_PROTOCOL_VERSION",
    "SEMANTIC_GEOMETRY_EXPERIMENT_PROTOCOL_VERSION",
    "SEMANTIC_GEOMETRY_EMBEDDING_IDENTITY_VERSION",
    "atomic_json_write",
    "atomic_torch_save",
    "build_experiment_signature",
    "build_semantic_embedding_identity",
    "done_marker_path",
    "done_marker_valid",
    "file_identity",
    "qparams_identity_manifest",
    "sample_order_hash",
    "save_embedding_cache",
    "stable_json_hash",
    "write_done_marker",
]
