from __future__ import annotations

import hashlib
import json
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
    from src.quant.ffn_quant_injector import restore_clean_model_from_checkpoint
    from src.utils.group_utils import (
        ANALYSIS_GROUPS,
        BASE_BUCKETS,
        GroupValidationError,
        analysis_group_mask,
        bucket_from_train_label_n,
        group_protocol_metadata,
        read_canonical_bucket,
        tail_included,
    )
except ImportError:  # pragma: no cover - direct script debugging fallback.
    from ..data.audio_input import AudioInputConfig, AudioInputDataset, audio_input_collate_fn
    from ..quant.ffn_quant_injector import restore_clean_model_from_checkpoint
    from ..utils.group_utils import (
        ANALYSIS_GROUPS,
        BASE_BUCKETS,
        GroupValidationError,
        analysis_group_mask,
        bucket_from_train_label_n,
        group_protocol_metadata,
        read_canonical_bucket,
        tail_included,
    )


CACHE_VERSION = 1
CACHE_FILENAME = "fp32_core_cache.pt"
MANIFEST_FILENAME = "fp32_core_cache.manifest.json"
EXPECTED_EMBEDDING_DIM = 768
CORE_STORAGE_FORMAT = "sample_sharded_v1"
CORE_SHARD_DIR_SUFFIX = "_shards"
DEFAULT_SAMPLE_SHARD_SIZE = 512
LAYOUT_PROTOCOL_VERSION = 1
CHECKPOINT_HASH_TYPE = "sha256"
LABEL_COLUMNS = ("primary_label", "label", "ebird_code", "class_name")
BUCKET_COLUMNS = ("bucket",)
SHOT_GROUP_COLUMNS = ("shot_group",)
FILEPATH_COLUMNS = ("filepath", "audio_path", "path", "rel_path", "file_id", "file_name")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return default


def _export_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "FP32_CORE_CACHE", {})
    return value if isinstance(value, Mapping) else {}


def _dataset_cfg(cfg: Any) -> Mapping[str, Any]:
    value = _cfg_get(cfg, "CURRENT_DATASET", {})
    return value if isinstance(value, Mapping) else {}


def _setting_get(cfg: Any, key: str, default: Any = None) -> Any:
    export_cfg = _export_cfg(cfg)
    if key in export_cfg:
        return export_cfg[key]
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


def _device(cfg: Any) -> torch.device:
    configured = _setting_get(cfg, "DEVICE", _cfg_get(cfg, "DEVICE", None))
    if configured is None:
        raise AttributeError("export_fp32_core_cache requires DEVICE from config.py.")
    device = torch.device(str(configured))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"config.py DEVICE={configured!r}, but CUDA is not available.")
    return device


def _seed(cfg: Any) -> int:
    return int(_setting_get(cfg, "SEED", _cfg_get(cfg, "SPLIT_SEED", 42)))


def _batch_size(cfg: Any) -> int:
    return int(_setting_get(cfg, "EVAL_BATCH_SIZE", _setting_get(cfg, "BATCH_SIZE", 8)))


def _num_workers(cfg: Any) -> int:
    return int(_setting_get(cfg, "NUM_WORKERS", 0))


def _pin_memory(cfg: Any) -> bool:
    return bool(_setting_get(cfg, "PIN_MEMORY", _device(cfg).type == "cuda"))


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
    value = _setting_get(cfg, "EVAL_CSV", None)
    if value is not None:
        return Path(value)
    dataset_value = _nested_get(_dataset_cfg(cfg), "EVAL_CSV", None)
    if dataset_value is not None:
        return Path(dataset_value)
    raise AttributeError("Cannot find eval.csv: set CURRENT_DATASET['EVAL_CSV'] or FP32_CORE_CACHE['EVAL_CSV'].")


def _label_mapping_path(cfg: Any) -> Path:
    value = _setting_get(cfg, "LABEL_MAPPING", None)
    if value is not None:
        return Path(value)
    dataset_value = _nested_get(_dataset_cfg(cfg), "LABEL_MAPPING", None)
    if dataset_value is not None:
        return Path(dataset_value)
    raise AttributeError("无法找到 label_mapping.json")


def _output_dir(cfg: Any) -> Path:
    configured = _setting_get(cfg, "OUTPUT_DIR", None)
    if configured is not None:
        return Path(configured)
    return Path(_cfg_get(cfg, "BASE_DIR", Path.cwd())) / _dataset_name(cfg) / "fp32_core_cache"


def _cache_path(cfg: Any, output_path: str | Path | None = None) -> Path:
    if output_path is not None:
        return Path(output_path)
    configured = _setting_get(cfg, "CACHE_PATH", None)
    if configured is not None:
        return Path(configured)
    return _output_dir(cfg) / CACHE_FILENAME


def _manifest_path(cache_path: Path) -> Path:
    if cache_path.name == CACHE_FILENAME:
        return cache_path.with_name(MANIFEST_FILENAME)
    return cache_path.with_suffix(cache_path.suffix + ".manifest.json")


def _atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(target)
    return target


def _atomic_json_dump(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
    tmp.replace(target)
    return target


def _sample_shard_size(cfg: Any, *, default: int = DEFAULT_SAMPLE_SHARD_SIZE) -> int:
    value = _setting_get(cfg, "SAMPLE_SHARD_SIZE", _setting_get(cfg, "SHARD_SIZE", default))
    size = int(value)
    if size <= 0:
        raise ValueError(f"SAMPLE_SHARD_SIZE must be positive, got {size}")
    return size


def _core_shard_dir(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_name(path.stem + CORE_SHARD_DIR_SUFFIX)


def _relative_to_parent(path: str | Path, parent: str | Path) -> str:
    item = Path(path)
    root = Path(parent)
    try:
        return item.relative_to(root).as_posix()
    except ValueError:
        return item.as_posix()


def _portable_path_text(value: str | Path) -> str:
    text = str(value).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def _looks_like_windows_absolute(text: str) -> bool:
    return len(text) >= 3 and text[0].isalpha() and text[1] == ":" and text[2] == "/"


def _ancestor_named(path: Path, name: str) -> Path | None:
    for item in (path, *path.parents):
        if item.name == name:
            return item
    return None


def _resolve_cache_relative_path(cache_path: str | Path, value: str | Path) -> Path:
    text = _portable_path_text(value)
    path = Path(text)
    if path.is_absolute():
        return path
    cache_parent = Path(cache_path).parent
    if text.startswith("local_data/") or _looks_like_windows_absolute(text):
        local_root = _ancestor_named(cache_parent, "local_data")
        marker = "local_data/"
        if local_root is not None and marker in text:
            return local_root / text.split(marker, 1)[1]
    marker = f"/{cache_parent.name}/"
    if _looks_like_windows_absolute(text) and marker in text:
        return cache_parent / text.split(marker, 1)[1]
    return cache_parent / path


def is_sharded_fp32_core_cache(cache: Mapping[str, Any]) -> bool:
    manifest = cache.get("manifest", {})
    manifest_format = manifest.get("storage_format") if isinstance(manifest, Mapping) else None
    return bool(cache.get("sharded", False)) or str(cache.get("storage_format", manifest_format)) == CORE_STORAGE_FORMAT


def iter_fp32_core_cache_shards(cache: Mapping[str, Any], cache_path: str | Path):
    if not is_sharded_fp32_core_cache(cache):
        yield dict(cache)
        return
    shards = cache.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise ValueError("sharded core cache manifest is missing shards")
    for entry in shards:
        if not isinstance(entry, Mapping):
            raise TypeError(f"core cache shard entry must be a mapping, got {type(entry)!r}")
        shard_rel = entry.get("path")
        if shard_rel is None:
            raise KeyError("core cache shard entry missing path")
        shard_path = _resolve_cache_relative_path(cache_path, shard_rel)
        shard = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard, Mapping):
            raise TypeError(f"core cache shard must save a mapping, got {type(shard)!r}: {shard_path}")
        yield dict(shard)


def load_fp32_core_cache(
    cache_path: str | Path,
    *,
    materialize: bool = False,
    include_padding_mask: bool = True,
    include_logits: bool = True,
) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"FP32 core cache does not exist: {path}")
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, Mapping):
        raise TypeError(f"FP32 core cache must be a mapping, got {type(cache)!r}")
    result = dict(cache)
    if not materialize or not is_sharded_fp32_core_cache(result):
        return result

    embedding_chunks: list[Tensor] = []
    logits_chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []
    padding_chunks: list[Tensor] = []
    valid_count_chunks: list[Tensor] = []
    for shard in iter_fp32_core_cache_shards(result, path):
        embedding_chunks.append(torch.as_tensor(shard["embedding"], dtype=torch.float32).cpu().contiguous())
        if include_logits:
            logits_chunks.append(torch.as_tensor(shard["logits"], dtype=torch.float32).cpu().contiguous())
        label_chunks.append(torch.as_tensor(shard["label_id"], dtype=torch.long).cpu().reshape(-1).contiguous())
        if include_padding_mask:
            padding_chunks.append(torch.as_tensor(shard["padding_mask"]).cpu().bool().contiguous())
        valid_count_chunks.append(torch.as_tensor(shard["valid_token_count"], dtype=torch.long).cpu().reshape(-1).contiguous())
    result["embedding"] = torch.cat(embedding_chunks, dim=0) if embedding_chunks else torch.empty((0, 0), dtype=torch.float32)
    result["label_id"] = torch.cat(label_chunks, dim=0) if label_chunks else torch.empty((0,), dtype=torch.long)
    result["valid_token_count"] = torch.cat(valid_count_chunks, dim=0) if valid_count_chunks else torch.empty((0,), dtype=torch.long)
    if include_logits:
        result["logits"] = torch.cat(logits_chunks, dim=0) if logits_chunks else torch.empty((0, 0), dtype=torch.float32)
    if include_padding_mask:
        result["padding_mask"] = torch.cat(padding_chunks, dim=0) if padding_chunks else torch.empty((0, 0), dtype=torch.bool)
    return result

def _checkpoint_path_for_manifest(cfg: Any) -> str | None:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    if isinstance(beats_cfg, Mapping):
        for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT"):
            value = beats_cfg.get(key)
            if value is not None:
                return str(Path(value).expanduser().resolve())
    for key in ("CHECKPOINT", "CHECKPOINT_PATH", "CKPT"):
        value = _cfg_get(cfg, key, None)
        if value is not None:
            return str(Path(value).expanduser().resolve())
    return None


def _default_cfg() -> Any:
    try:
        import config as project_config
    except ModuleNotFoundError as exc:
        if exc.name != "config":
            raise
        raise AttributeError("cfg was not provided and config.py could not be imported.") from exc
    return project_config


def _load_label_mapping(path: str | Path) -> dict[str, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"label_mapping.json does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("label_mapping.json must be a non-empty mapping.")
    mapping: dict[str, int] = {}
    for label, idx in raw.items():
        if isinstance(idx, bool):
            raise ValueError(f"label_mapping contains a non-integer label_id: {label!r} -> {idx!r}")
        mapping[str(label)] = int(idx)
    ids = sorted(mapping.values())
    expected = list(range(len(ids)))
    if ids != expected:
        raise ValueError(f"label_mapping ids must be contiguous 0..C-1 values, got {ids[:10]}...")
    if len(set(ids)) != len(ids):
        raise ValueError("label_mapping id 存在重复")
    return mapping


def _reverse_label_mapping(mapping: Mapping[str, int]) -> dict[int, str]:
    return {int(idx): str(label) for label, idx in mapping.items()}


def _read_eval_metadata(eval_csv: str | Path) -> dict[str, dict[str, Any]]:
    import pandas as pd

    path = Path(eval_csv)
    if not path.exists():
        raise FileNotFoundError(f"eval.csv does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"eval.csv 为空: {path}")
    if "sample_id" not in frame.columns:
        raise KeyError("eval.csv 必须包含 sample_id")
    sample_ids = frame["sample_id"].astype(str)
    duplicated = sample_ids[sample_ids.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"eval.csv sample_id 存在重复: {duplicated[:8]}")
    result: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        result[sample_id] = {str(key): row[key] for key in frame.columns}
    return result


def _text_from_row(row: Mapping[str, Any], columns: Sequence[str], default: str = "") -> str:
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            return text
    return default


def _has_metadata_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(value != value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text and text.lower() not in {"nan", "none", "null", "<na>"})


def _canonical_bucket_from_row(row: Mapping[str, Any], *, sample_id: str) -> str:
    bucket_value = row.get("bucket")
    shot_group_value = row.get("shot_group")
    train_label_n_value = row.get("train_label_n")
    has_bucket = _has_metadata_value(bucket_value)
    has_shot_group = _has_metadata_value(shot_group_value)
    has_train_label_n = _has_metadata_value(train_label_n_value)

    if has_bucket:
        try:
            result = read_canonical_bucket(
                bucket=[bucket_value],
                shot_group=[shot_group_value] if has_shot_group else None,
                train_label_n=[train_label_n_value] if has_train_label_n else None,
                context=f"eval.csv sample_id={sample_id}",
                allow_shot_group_fallback=False,
            )
        except GroupValidationError as exc:
            raise ValueError(str(exc)) from exc
        return result.bucket[0]

    if has_train_label_n:
        try:
            bucket = bucket_from_train_label_n(train_label_n_value)
            result = read_canonical_bucket(
                bucket=[bucket],
                shot_group=[shot_group_value] if has_shot_group else None,
                train_label_n=[train_label_n_value],
                context=f"eval.csv sample_id={sample_id}",
                allow_shot_group_fallback=False,
            )
        except GroupValidationError as exc:
            raise ValueError(str(exc)) from exc
        return result.bucket[0]

    raise ValueError(f"eval.csv sample_id={sample_id}: bucket is required, or train_label_n must be present to derive it")

def _label_id_from_batch(batch: Mapping[str, Any]) -> Tensor:
    value = batch["label_id"]
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.long)
    return value.detach().cpu().long().reshape(-1)


def _sample_ids_from_batch(batch: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in batch["sample_id"]]


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    """Build a stable hash from sample_id order for fixed eval-order validation."""

    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256_for_manifest(cfg: Any) -> str | None:
    checkpoint_path = _checkpoint_path_for_manifest(cfg)
    return _file_sha256(checkpoint_path)


def _checkpoint_protocol_fields(cfg: Any) -> dict[str, str | None]:
    checkpoint_hash = _checkpoint_sha256_for_manifest(cfg)
    return {
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_hash_type": CHECKPOINT_HASH_TYPE,
    }


def _core_layout_protocol_fields() -> dict[str, Any]:
    return {
        "layout_protocol_version": LAYOUT_PROTOCOL_VERSION,
    }


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device=device, non_blocking=True) if isinstance(value, Tensor) else value
    return result


def _build_eval_loader(cfg: Any) -> DataLoader:
    override = _cfg_get(cfg, "build_eval_loader", None)
    if callable(override):
        return override(cfg)
    mode = str(_setting_get(cfg, "AUDIO_MODE", "eval"))
    if mode.lower() == "train":
        raise ValueError("FP32 core cache 禁止使用 train random crop")
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


def _restore_fp32_teacher(cfg: Any) -> nn.Module:
    for name in ("restore_fp32_teacher", "restore_clean_model", "restore_clean_model_from_checkpoint"):
        fn = _cfg_get(cfg, name, None)
        if callable(fn):
            model = fn(cfg)
            if not isinstance(model, nn.Module):
                raise TypeError(f"{name} 必须返回 nn.Module，当前为 {type(model)!r}")
            return model
    return restore_clean_model_from_checkpoint(cfg)


def _autocast_context(cfg: Any, device: torch.device) -> Any:
    beats_cfg = _cfg_get(cfg, "BEATS", {})
    use_amp = bool(beats_cfg.get("USE_AMP", False)) if isinstance(beats_cfg, Mapping) else False
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def _normalise_token_padding_mask(mask: Tensor | None, *, batch_size: int, num_tokens: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)
    mask = mask.to(device=device, dtype=torch.bool)
    expected = (batch_size, num_tokens)
    if tuple(mask.shape) == expected:
        return mask.contiguous()
    if tuple(mask.shape) == (num_tokens, batch_size):
        return mask.transpose(0, 1).contiguous()
    raise ValueError(f"token padding_mask shape must be [B,T], got {tuple(mask.shape)} vs {expected}")


def backbone_features_for_cache(model: nn.Module, waveform: Tensor, padding_mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Return final BEATs Transformer token features, token padding_mask, and logits."""

    if not all(hasattr(model, name) for name in ("preprocess", "patch_embedding", "encoder")):
        if hasattr(model, "extract_features"):
            features, feature_padding_mask = model.extract_features(waveform, padding_mask=padding_mask)
            if features.ndim != 3:
                raise ValueError(f"extract_features must return [B,T,D] features, got shape={tuple(features.shape)}")
            mask = _normalise_token_padding_mask(
                feature_padding_mask,
                batch_size=int(features.shape[0]),
                num_tokens=int(features.shape[1]),
                device=features.device,
            )
            return features, mask
        raise TypeError("model 无法提取 BEATs token feature")

    fbank = model.preprocess(waveform)
    feature_padding_mask = padding_mask
    if feature_padding_mask is not None:
        feature_padding_mask = model.forward_padding_mask(fbank, feature_padding_mask)

    fbank = fbank.unsqueeze(1)
    features = model.patch_embedding(fbank)
    features = features.reshape(features.shape[0], features.shape[1], -1)
    features = features.transpose(1, 2)
    features = model.layer_norm(features)

    if feature_padding_mask is not None:
        feature_padding_mask = model.forward_padding_mask(features, feature_padding_mask)

    if getattr(model, "post_extract_proj", None) is not None:
        features = model.post_extract_proj(features)

    x = model.dropout_input(features)
    x, _layer_results = model.encoder(x, padding_mask=feature_padding_mask)
    token_mask = _normalise_token_padding_mask(
        feature_padding_mask,
        batch_size=int(x.shape[0]),
        num_tokens=int(x.shape[1]),
        device=x.device,
    )
    return x, token_mask


def mask_aware_mean(features: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Mean-pool token features with True=padding token masks and return [B,D] embeddings."""

    if features.ndim != 3:
        raise ValueError(f"features must be [B,T,D], got shape={tuple(features.shape)}")
    if tuple(padding_mask.shape) != tuple(features.shape[:2]):
        raise ValueError(f"padding_mask must align with features [B,T], got {tuple(padding_mask.shape)} vs {tuple(features.shape[:2])}")
    valid = ~padding_mask.to(device=features.device, dtype=torch.bool)
    valid_count = valid.sum(dim=1)
    if bool((valid_count <= 0).any().item()):
        raise ValueError("存在有效 token 数为 0 的样本，不能构建 FP32 core cache")
    masked = features.masked_fill(~valid.unsqueeze(-1), 0.0)
    embedding = masked.sum(dim=1) / valid_count.unsqueeze(-1).to(dtype=features.dtype)
    return embedding, valid_count


def _pooled_logits(model: nn.Module, features: Tensor, padding_mask: Tensor) -> Tensor:
    if not hasattr(model, "predictor") or getattr(model, "predictor") is None:
        raise AttributeError("FP32 teacher must contain predictor to export logits.")
    x = model.predictor_dropout(features) if hasattr(model, "predictor_dropout") else features
    token_logits = model.predictor(x)
    if token_logits.ndim != 3:
        raise ValueError(f"predictor output must be [B,T,C], got shape={tuple(token_logits.shape)}")
    if tuple(token_logits.shape[:2]) != tuple(padding_mask.shape):
        raise ValueError("token logits and padding_mask [B,T] do not align.")
    valid = ~padding_mask.to(device=token_logits.device, dtype=torch.bool)
    if bool(valid.any().item()):
        masked = token_logits.masked_fill(~valid.unsqueeze(-1), 0.0)
        denom = valid.sum(dim=1).clamp_min(1).unsqueeze(-1).to(dtype=token_logits.dtype)
        return masked.sum(dim=1) / denom
    raise ValueError("所有样本都没有有效 token")


def forward_core_outputs(model: nn.Module, batch: Mapping[str, Any], cfg: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run one forward pass and return logits, embedding, token padding_mask, and valid_token_count."""

    device = _device(cfg)
    waveform = batch["waveform"]
    padding_mask = batch.get("padding_mask")
    with torch.no_grad(), _autocast_context(cfg, device):
        features, token_padding_mask = backbone_features_for_cache(model, waveform, padding_mask)
        embedding, valid_token_count = mask_aware_mean(features, token_padding_mask)
        logits = _pooled_logits(model, features, token_padding_mask)
    return logits, embedding, token_padding_mask, valid_token_count


def _validate_logits_and_embedding(logits: Tensor, embedding: Tensor, *, num_classes: int, expected_dim: int) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B,C], got shape={tuple(logits.shape)}")
    if int(logits.shape[1]) != int(num_classes):
        raise ValueError(f"logits dim must equal label_mapping class count {num_classes}, got {int(logits.shape[1])}")
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be [B,D], not [B,1,D] or [1,D], got shape={tuple(embedding.shape)}")
    if int(embedding.shape[1]) != int(expected_dim):
        raise ValueError(f"embedding dim must be {expected_dim}, got {int(embedding.shape[1])}")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits 存在 NaN/Inf")
    if not torch.isfinite(embedding).all().item():
        raise ValueError("embedding 存在 NaN/Inf")


def _bucket_summary(buckets: Sequence[str]) -> dict[str, Any]:
    counts = {bucket: 0 for bucket in BASE_BUCKETS}
    canonical_buckets = [str(item).strip().lower() for item in buckets]
    for bucket in canonical_buckets:
        if bucket not in counts:
            raise ValueError(f"invalid canonical bucket in cache metadata: {bucket!r}")
        counts[bucket] += 1
    analysis_counts = {
        str(group): int(sum(1 for keep in analysis_group_mask(canonical_buckets, str(group)) if bool(keep)))
        for group in ANALYSIS_GROUPS
    }
    protocol = group_protocol_metadata(
        tail_rule="tail is recomputed from raw sample rows with bucket in {medium, few}",
    )
    return {
        **protocol,
        "grouping_protocol": dict(protocol),
        "bucket_counts": dict(counts),
        "analysis_group_counts": analysis_counts,
        "shot_group_counts": dict(counts),
        "unknown_shot_group_count": 0,
        "has_unknown_shot_group": False,
    }


def _metadata_for_sample(
    sample_id: str,
    *,
    eval_metadata: Mapping[str, Mapping[str, Any]],
    reverse_label_mapping: Mapping[int, str],
    label_id: int,
    audio_path: str,
) -> dict[str, Any]:
    if sample_id not in eval_metadata:
        raise KeyError(f"sample_id={sample_id!r} 不在 eval.csv 元数据中")
    row = eval_metadata[sample_id]
    row_label = row.get("label_id")
    if row_label is not None and str(row_label).strip().lower() not in {"", "nan", "none", "<na>"}:
        if int(float(str(row_label))) != int(label_id):
            raise ValueError(f"sample_id={sample_id}: batch label_id={label_id} differs from eval.csv label_id={row_label}")
    primary_label = _text_from_row(row, LABEL_COLUMNS, default=reverse_label_mapping.get(int(label_id), str(label_id)))
    bucket = _canonical_bucket_from_row(row, sample_id=sample_id)
    split = _text_from_row(row, ("split", "source_split"), default="eval")
    filepath = _text_from_row(row, FILEPATH_COLUMNS, default=str(audio_path))
    if not filepath:
        filepath = str(audio_path)
    return {
        "filepath": filepath,
        "primary_label": primary_label,
        "bucket": bucket,
        "shot_group": bucket,
        "tail_included": bool(list(tail_included([bucket]))[0]),
        "split": split,
    }

def build_fp32_core_cache(
    cfg: Any,
    *,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Build the FP32 teacher core cache over eval.csv without Layer0~11 token traces."""

    label_mapping_path = _label_mapping_path(cfg)
    label_mapping = _load_label_mapping(label_mapping_path)
    reverse_mapping = _reverse_label_mapping(label_mapping)
    num_classes = len(label_mapping)
    expected_dim = int(_setting_get(cfg, "EXPECTED_EMBEDDING_DIM", EXPECTED_EMBEDDING_DIM))
    eval_csv = _eval_csv(cfg)
    eval_metadata = _read_eval_metadata(eval_csv)
    device = _device(cfg)

    owns_model = model is None
    if model is None:
        model = _restore_fp32_teacher(cfg)
    model = model.to(device)
    model.eval()

    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)

    sample_ids: list[str] = []
    filepaths: list[str] = []
    primary_labels: list[str] = []
    buckets: list[str] = []
    tail_flags: list[bool] = []
    shot_groups: list[str] = []
    splits: list[str] = []
    label_chunks: list[Tensor] = []
    logits_chunks: list[Tensor] = []
    embedding_chunks: list[Tensor] = []
    padding_mask_chunks: list[Tensor] = []
    valid_count_chunks: list[Tensor] = []

    for batch in eval_loader:
        batch_sample_ids = _sample_ids_from_batch(batch)
        labels = _label_id_from_batch(batch)
        if len(batch_sample_ids) != int(labels.numel()):
            raise ValueError("batch sample_id count differs from label_id count.")
        if set(batch_sample_ids) - set(eval_metadata):
            missing = sorted(set(batch_sample_ids) - set(eval_metadata))
            raise KeyError(f"batch contains sample_id values missing from eval.csv: {missing[:8]}")
        moved = _move_batch_to_device(batch, device)
        logits, embedding, token_padding_mask, valid_token_count = forward_core_outputs(model, moved, cfg)
        _validate_logits_and_embedding(logits, embedding, num_classes=num_classes, expected_dim=expected_dim)
        if int(logits.shape[0]) != len(batch_sample_ids) or int(embedding.shape[0]) != len(batch_sample_ids):
            raise ValueError("model output batch size does not match sample_id count.")

        audio_paths = [str(item) for item in batch.get("audio_path", [""] * len(batch_sample_ids))]
        for idx, sample_id in enumerate(batch_sample_ids):
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id encountered during export: {sample_id!r}")
            metadata = _metadata_for_sample(
                sample_id,
                eval_metadata=eval_metadata,
                reverse_label_mapping=reverse_mapping,
                label_id=int(labels[idx].item()),
                audio_path=audio_paths[idx] if idx < len(audio_paths) else "",
            )
            sample_ids.append(sample_id)
            filepaths.append(metadata["filepath"])
            primary_labels.append(metadata["primary_label"])
            buckets.append(metadata["bucket"])
            tail_flags.append(bool(metadata["tail_included"]))
            shot_groups.append(metadata["shot_group"])
            splits.append(metadata["split"])

        label_chunks.append(labels.cpu())
        logits_chunks.append(logits.detach().cpu().to(dtype=torch.float32).contiguous())
        embedding_chunks.append(embedding.detach().cpu().to(dtype=torch.float32).contiguous())
        padding_mask_chunks.append(token_padding_mask.detach().cpu().to(dtype=torch.bool).contiguous())
        valid_count_chunks.append(valid_token_count.detach().cpu().long().contiguous())

    if not sample_ids:
        raise ValueError("eval_loader 没有产生任何样本")
    if len(sample_ids) != len(eval_metadata):
        raise ValueError(f"core cache sample count {len(sample_ids)} differs from eval.csv sample count {len(eval_metadata)}")

    logits_all = torch.cat(logits_chunks, dim=0)
    embedding_all = torch.cat(embedding_chunks, dim=0)
    label_all = torch.cat(label_chunks, dim=0)
    padding_mask_all = torch.cat(padding_mask_chunks, dim=0)
    valid_count_all = torch.cat(valid_count_chunks, dim=0)
    _validate_logits_and_embedding(logits_all, embedding_all, num_classes=num_classes, expected_dim=expected_dim)
    if embedding_all.ndim != 2 or int(embedding_all.shape[1]) != expected_dim:
        raise ValueError("embedding must be saved as [N,768]; one sample should be [768].")

    sample_id_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    if len(sample_id_to_index) != len(sample_ids):
        raise ValueError("sample_id_to_index construction found duplicate sample_id values.")

    bucket_summary = _bucket_summary(buckets)
    sample_order_hash = _sample_order_hash(sample_ids)
    manifest = {
        "cache_version": CACHE_VERSION,
        "cache_type": "fp32_core_cache",
        "stage": "quant_injection_pre_fp32_reference",
        "eval_csv": str(eval_csv.expanduser().resolve()),
        "eval_csv_sha256": _file_sha256(eval_csv),
        "label_mapping": str(label_mapping_path.expanduser().resolve()),
        "label_mapping_sha256": _file_sha256(label_mapping_path),
        "checkpoint_path": _checkpoint_path_for_manifest(cfg),
        **_checkpoint_protocol_fields(cfg),
        **_core_layout_protocol_fields(),
        "dataset_name": _dataset_name(cfg),
        "num_samples": len(sample_ids),
        "sample_order_hash": sample_order_hash,
        "sample_order_hash_algorithm": "sha256-json-list",
        "num_classes": num_classes,
        "embedding_dim": expected_dim,
        "embedding_pooling": "mask_aware_mean",
        "padding_mask_level": "BEATs token padding mask; True means padding",
        "audio_input_mode": str(_setting_get(cfg, "AUDIO_MODE", "eval")),
        **bucket_summary,
    }

    cache = {
        "cache_version": CACHE_VERSION,
        "manifest": manifest,
        "sample_id": sample_ids,
        "sample_order_hash": sample_order_hash,
        "sample_id_to_index": sample_id_to_index,
        "filepath": filepaths,
        "label_id": label_all,
        "primary_label": primary_labels,
        "bucket": buckets,
        "shot_group": shot_groups,
        "tail_included": tail_flags,
        "split": splits,
        "logits": logits_all,
        "embedding": embedding_all,
        "padding_mask": padding_mask_all,
        "valid_token_count": valid_count_all,
        "label_mapping": dict(label_mapping),
    }
    if owns_model and device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


def build_and_save_fp32_core_cache_sharded(
    cfg: Any,
    cache_path: str | Path,
    *,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Build the FP32 core cache as sample shards and a lightweight top-level manifest."""

    path = Path(cache_path)
    shard_dir = _core_shard_dir(path)
    shard_size = _sample_shard_size(cfg)
    label_mapping_path = _label_mapping_path(cfg)
    label_mapping = _load_label_mapping(label_mapping_path)
    reverse_mapping = _reverse_label_mapping(label_mapping)
    num_classes = len(label_mapping)
    expected_dim = int(_setting_get(cfg, "EXPECTED_EMBEDDING_DIM", EXPECTED_EMBEDDING_DIM))
    eval_csv = _eval_csv(cfg)
    eval_metadata = _read_eval_metadata(eval_csv)
    device = _device(cfg)

    owns_model = model is None
    if model is None:
        model = _restore_fp32_teacher(cfg)
    model = model.to(device)
    model.eval()

    if eval_loader is None:
        eval_loader = _build_eval_loader(cfg)

    sample_ids: list[str] = []
    filepaths: list[str] = []
    primary_labels: list[str] = []
    buckets: list[str] = []
    tail_flags: list[bool] = []
    shot_groups: list[str] = []
    splits: list[str] = []
    label_rows: list[int] = []
    valid_count_rows: list[int] = []
    shard_records: list[dict[str, Any]] = []

    shard_sample_ids: list[str] = []
    shard_filepaths: list[str] = []
    shard_primary_labels: list[str] = []
    shard_buckets: list[str] = []
    shard_tail_flags: list[bool] = []
    shard_shot_groups: list[str] = []
    shard_splits: list[str] = []
    shard_label_chunks: list[Tensor] = []
    shard_logits_chunks: list[Tensor] = []
    shard_embedding_chunks: list[Tensor] = []
    shard_padding_mask_chunks: list[Tensor] = []
    shard_valid_count_chunks: list[Tensor] = []
    shard_start = 0

    def flush_shard() -> None:
        nonlocal shard_start
        if not shard_sample_ids:
            return
        shard_id = len(shard_records)
        shard_path = shard_dir / f"shard_{shard_id:05d}.pt"
        logits_shard = torch.cat(shard_logits_chunks, dim=0)
        embedding_shard = torch.cat(shard_embedding_chunks, dim=0)
        label_shard = torch.cat(shard_label_chunks, dim=0)
        padding_mask_shard = torch.cat(shard_padding_mask_chunks, dim=0)
        valid_count_shard = torch.cat(shard_valid_count_chunks, dim=0)
        _validate_logits_and_embedding(logits_shard, embedding_shard, num_classes=num_classes, expected_dim=expected_dim)
        sample_end = shard_start + len(shard_sample_ids)
        shard_manifest = {
            "cache_version": CACHE_VERSION,
            "cache_type": "fp32_core_cache_shard",
            "parent_cache_type": "fp32_core_cache",
            "storage_format": CORE_STORAGE_FORMAT,
            "shard_id": int(shard_id),
            "sample_start": int(shard_start),
            "sample_end": int(sample_end),
            "num_samples": int(len(shard_sample_ids)),
            "sample_order_hash": _sample_order_hash(shard_sample_ids),
            "embedding_shape": list(embedding_shard.shape),
            "logits_shape": list(logits_shard.shape),
            "padding_mask_shape": list(padding_mask_shard.shape),
        }
        payload = {
            "cache_version": CACHE_VERSION,
            "cache_type": "fp32_core_cache_shard",
            "manifest": shard_manifest,
            "sample_id": list(shard_sample_ids),
            "filepath": list(shard_filepaths),
            "label_id": label_shard,
            "primary_label": list(shard_primary_labels),
            "bucket": list(shard_buckets),
            "shot_group": list(shard_shot_groups),
            "tail_included": list(shard_tail_flags),
            "split": list(shard_splits),
            "logits": logits_shard,
            "embedding": embedding_shard,
            "padding_mask": padding_mask_shard,
            "valid_token_count": valid_count_shard,
        }
        _atomic_torch_save(payload, shard_path)
        shard_hash = _file_sha256(shard_path)
        shard_records.append(
            {
                "path": _relative_to_parent(shard_path, path.parent),
                "sha256": shard_hash,
                "shard_id": int(shard_id),
                "sample_start": int(shard_start),
                "sample_end": int(sample_end),
                "num_samples": int(len(shard_sample_ids)),
                "sample_order_hash": shard_manifest["sample_order_hash"],
                "embedding_shape": list(embedding_shard.shape),
                "logits_shape": list(logits_shard.shape),
                "padding_mask_shape": list(padding_mask_shard.shape),
            }
        )
        shard_start = sample_end
        shard_sample_ids.clear()
        shard_filepaths.clear()
        shard_primary_labels.clear()
        shard_buckets.clear()
        shard_tail_flags.clear()
        shard_shot_groups.clear()
        shard_splits.clear()
        shard_label_chunks.clear()
        shard_logits_chunks.clear()
        shard_embedding_chunks.clear()
        shard_padding_mask_chunks.clear()
        shard_valid_count_chunks.clear()

    for batch in eval_loader:
        batch_sample_ids = _sample_ids_from_batch(batch)
        labels = _label_id_from_batch(batch)
        if len(batch_sample_ids) != int(labels.numel()):
            raise ValueError("batch sample_id count differs from label_id count.")
        if set(batch_sample_ids) - set(eval_metadata):
            missing = sorted(set(batch_sample_ids) - set(eval_metadata))
            raise KeyError(f"batch contains sample_id values missing from eval.csv: {missing[:8]}")
        moved = _move_batch_to_device(batch, device)
        logits, embedding, token_padding_mask, valid_token_count = forward_core_outputs(model, moved, cfg)
        _validate_logits_and_embedding(logits, embedding, num_classes=num_classes, expected_dim=expected_dim)
        if int(logits.shape[0]) != len(batch_sample_ids) or int(embedding.shape[0]) != len(batch_sample_ids):
            raise ValueError("model output batch size does not match sample_id count.")

        audio_paths = [str(item) for item in batch.get("audio_path", [""] * len(batch_sample_ids))]
        batch_filepaths: list[str] = []
        batch_primary_labels: list[str] = []
        batch_buckets: list[str] = []
        batch_tail_flags: list[bool] = []
        batch_shot_groups: list[str] = []
        batch_splits: list[str] = []
        for idx, sample_id in enumerate(batch_sample_ids):
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id encountered during export: {sample_id!r}")
            metadata = _metadata_for_sample(
                sample_id,
                eval_metadata=eval_metadata,
                reverse_label_mapping=reverse_mapping,
                label_id=int(labels[idx].item()),
                audio_path=audio_paths[idx] if idx < len(audio_paths) else "",
            )
            sample_ids.append(sample_id)
            filepaths.append(metadata["filepath"])
            primary_labels.append(metadata["primary_label"])
            buckets.append(metadata["bucket"])
            tail_flags.append(bool(metadata["tail_included"]))
            shot_groups.append(metadata["shot_group"])
            splits.append(metadata["split"])
            label_rows.append(int(labels[idx].item()))
            batch_filepaths.append(metadata["filepath"])
            batch_primary_labels.append(metadata["primary_label"])
            batch_buckets.append(metadata["bucket"])
            batch_tail_flags.append(bool(metadata["tail_included"]))
            batch_shot_groups.append(metadata["shot_group"])
            batch_splits.append(metadata["split"])

        valid_counts_cpu = valid_token_count.detach().cpu().long().reshape(-1)
        valid_count_rows.extend(int(item) for item in valid_counts_cpu.tolist())
        shard_sample_ids.extend(batch_sample_ids)
        shard_filepaths.extend(batch_filepaths)
        shard_primary_labels.extend(batch_primary_labels)
        shard_buckets.extend(batch_buckets)
        shard_tail_flags.extend(batch_tail_flags)
        shard_shot_groups.extend(batch_shot_groups)
        shard_splits.extend(batch_splits)
        shard_label_chunks.append(labels.cpu())
        shard_logits_chunks.append(logits.detach().cpu().to(dtype=torch.float32).contiguous())
        shard_embedding_chunks.append(embedding.detach().cpu().to(dtype=torch.float32).contiguous())
        shard_padding_mask_chunks.append(token_padding_mask.detach().cpu().to(dtype=torch.bool).contiguous())
        shard_valid_count_chunks.append(valid_counts_cpu.contiguous())
        if len(shard_sample_ids) >= shard_size:
            flush_shard()

    flush_shard()
    if not sample_ids:
        raise ValueError("eval_loader 没有产生任何样本")
    if len(sample_ids) != len(eval_metadata):
        raise ValueError(f"core cache sample count {len(sample_ids)} differs from eval.csv sample count {len(eval_metadata)}")

    sample_id_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    if len(sample_id_to_index) != len(sample_ids):
        raise ValueError("sample_id_to_index construction found duplicate sample_id values.")

    bucket_summary = _bucket_summary(buckets)
    sample_order_hash = _sample_order_hash(sample_ids)
    manifest = {
        "cache_version": CACHE_VERSION,
        "cache_type": "fp32_core_cache",
        "stage": "quant_injection_pre_fp32_reference",
        "storage_format": CORE_STORAGE_FORMAT,
        "sharded": True,
        "sample_shard_size": int(shard_size),
        "eval_csv": str(eval_csv.expanduser().resolve()),
        "eval_csv_sha256": _file_sha256(eval_csv),
        "label_mapping": str(label_mapping_path.expanduser().resolve()),
        "label_mapping_sha256": _file_sha256(label_mapping_path),
        "checkpoint_path": _checkpoint_path_for_manifest(cfg),
        **_checkpoint_protocol_fields(cfg),
        **_core_layout_protocol_fields(),
        "dataset_name": _dataset_name(cfg),
        "num_samples": len(sample_ids),
        "sample_order_hash": sample_order_hash,
        "sample_order_hash_algorithm": "sha256-json-list",
        "num_classes": num_classes,
        "embedding_dim": expected_dim,
        "embedding_pooling": "mask_aware_mean",
        "padding_mask_level": "BEATs token padding mask; True means padding",
        "audio_input_mode": str(_setting_get(cfg, "AUDIO_MODE", "eval")),
        "num_shards": len(shard_records),
        "shard_dir": _relative_to_parent(shard_dir, path.parent),
        "shard_paths": [str(item["path"]) for item in shard_records],
        "shard_sha256": {str(item["path"]): item.get("sha256") for item in shard_records},
        "shards": list(shard_records),
        **bucket_summary,
    }
    cache = {
        "cache_version": CACHE_VERSION,
        "cache_type": "fp32_core_cache",
        "manifest": manifest,
        "storage_format": CORE_STORAGE_FORMAT,
        "sharded": True,
        "sample_id": sample_ids,
        "sample_order_hash": sample_order_hash,
        "sample_id_to_index": sample_id_to_index,
        "filepath": filepaths,
        "label_id": torch.as_tensor(label_rows, dtype=torch.long),
        "primary_label": primary_labels,
        "bucket": buckets,
        "shot_group": shot_groups,
        "tail_included": tail_flags,
        "split": splits,
        "valid_token_count": torch.as_tensor(valid_count_rows, dtype=torch.long),
        "label_mapping": dict(label_mapping),
        "shard_paths": [str(item["path"]) for item in shard_records],
        "shard_sha256": {str(item["path"]): item.get("sha256") for item in shard_records},
        "shards": list(shard_records),
    }
    _atomic_torch_save(cache, path)
    _atomic_json_dump(manifest, _manifest_path(path))
    if owns_model and device.type == "cuda":
        torch.cuda.empty_cache()
    return {"cache": cache, "cache_path": path, "manifest_path": _manifest_path(path)}

def save_fp32_core_cache(cache: Mapping[str, Any], cache_path: str | Path) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cache), path)
    manifest = cache.get("manifest", {})
    with _manifest_path(path).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return path


def export_fp32_core_cache(
    cfg: Any | None = None,
    *,
    output_path: str | Path | None = None,
    force: bool = False,
    model: nn.Module | None = None,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Build and save the FP32 core cache, returning both the cache and path."""

    cfg = _default_cfg() if cfg is None else cfg
    path = _cache_path(cfg, output_path)
    if path.exists() and not force:
        raise FileExistsError(f"core cache already exists: {path}; pass force=True to rebuild.")
    return build_and_save_fp32_core_cache_sharded(
        cfg,
        path,
        model=model,
        eval_loader=eval_loader,
    )

def main(cfg: Any | None = None) -> dict[str, Any]:
    return export_fp32_core_cache(cfg)


__all__ = [
    "CACHE_FILENAME",
    "CACHE_VERSION",
    "CORE_STORAGE_FORMAT",
    "EXPECTED_EMBEDDING_DIM",
    "_checkpoint_sha256_for_manifest",
    "_file_sha256",
    "backbone_features_for_cache",
    "build_and_save_fp32_core_cache_sharded",
    "build_fp32_core_cache",
    "export_fp32_core_cache",
    "forward_core_outputs",
    "is_sharded_fp32_core_cache",
    "iter_fp32_core_cache_shards",
    "load_fp32_core_cache",
    "main",
    "mask_aware_mean",
    "save_fp32_core_cache",
]
