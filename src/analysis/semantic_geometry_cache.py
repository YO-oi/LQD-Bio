from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

try:
    from src.export.export_fp32_core_cache import load_fp32_core_cache
    from src.utils.group_utils import read_canonical_bucket, tail_included
except ImportError:  # pragma: no cover - package-relative debugging.
    from ..export.export_fp32_core_cache import load_fp32_core_cache
    from ..utils.group_utils import read_canonical_bucket, tail_included


CACHE_VERSION = 1
CACHE_TYPE = "semantic_embedding_cache"
STAGE = "semantic_geometry_embedding_cache"
DEFAULT_EMBEDDING_DIM = 768
FP32_QUANT_MODE = "FP32"
FP32_GRANULARITY = "fp32"
CORE_CACHE_TYPE = "fp32_core_cache"
LABEL_FIELD = "label"
CORE_LABEL_CANDIDATES = ("label", "primary_label")
REQUIRED_CACHE_FIELDS = ("sample_id", LABEL_FIELD, "bucket", "embedding", "quant_mode", "granularity")


def _sample_order_hash(sample_ids: Sequence[str]) -> str:
    """用 sample_id 顺序生成稳定 hash，避免误把同集合不同顺序当成同一份 cache。"""

    payload = json.dumps([str(item) for item in sample_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_list(value: Sequence[Any] | Tensor, *, name: str, num_samples: int | None = None) -> list[Any]:
    if isinstance(value, Tensor):
        result = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (str, bytes)):
        raise TypeError(f"{name} 必须是样本级序列，不能是单个字符串")
    else:
        result = list(value)
    if num_samples is not None and len(result) != int(num_samples):
        raise ValueError(f"{name} 数量必须等于样本数，当前 {len(result)} vs {num_samples}")
    return result


def _sample_id_list(value: Sequence[Any] | Tensor, *, num_samples: int | None = None, context: str) -> list[str]:
    ids = [str(item) for item in _as_list(value, name=f"{context}.sample_id", num_samples=num_samples)]
    seen: set[str] = set()
    duplicated: list[str] = []
    for item in ids:
        if item in seen:
            duplicated.append(item)
        seen.add(item)
    if duplicated:
        preview = ", ".join(list(dict.fromkeys(duplicated))[:8])
        raise ValueError(f"{context} sample_id 必须唯一，发现重复: {preview}")
    return ids


def _canonical_bucket_lists(
    *,
    bucket: Sequence[Any] | Tensor | None,
    shot_group: Sequence[Any] | Tensor | None,
    num_samples: int,
    context: str,
) -> tuple[list[str], list[str], list[bool]]:
    bucket_values = None if bucket is None else _as_list(bucket, name=f"{context}.bucket", num_samples=num_samples)
    shot_values = None if shot_group is None else _as_list(shot_group, name=f"{context}.shot_group", num_samples=num_samples)
    result = read_canonical_bucket(
        bucket=bucket_values,
        shot_group=shot_values,
        context=context,
        allow_shot_group_fallback=False,
    )
    buckets = [str(item) for item in result.bucket]
    return buckets, list(buckets), [bool(item) for item in tail_included(buckets)]

def _embedding_matrix(
    embedding: Any,
    *,
    name: str,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
) -> Tensor:
    """校验并返回 [N,D] embedding 矩阵。

    这里故意不自动 squeeze [N,1,D] 或每样本 [1,D]，因为这会掩盖上游 pooling
    协议错误。缓存里统一保存 [N,D]，单个样本取出后自然是 [D]。
    """

    tensor = embedding.detach().cpu().float() if isinstance(embedding, Tensor) else torch.as_tensor(embedding, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"{name} 必须是 [N,D]，不能是 [N,1,D] 或每样本 [1,D]；当前 shape={tuple(tensor.shape)}")
    if int(tensor.shape[0]) <= 0:
        raise ValueError(f"{name} 至少要包含一个样本")
    if expected_dim is not None and int(tensor.shape[1]) != int(expected_dim):
        raise ValueError(f"{name} embedding 维度必须是 {int(expected_dim)}，当前为 {int(tensor.shape[1])}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    return tensor.contiguous()


def _optional_label_id(value: Any, *, num_samples: int, context: str) -> Tensor | None:
    if value is None:
        return None
    tensor = value.detach().cpu().long().reshape(-1) if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.long).reshape(-1)
    if int(tensor.numel()) != int(num_samples):
        raise ValueError(f"{context}.label_id 数量必须等于样本数，当前 {int(tensor.numel())} vs {num_samples}")
    if int(tensor.numel()) > 0 and int(tensor.min().item()) < 0:
        raise ValueError(f"{context}.label_id 不能为负数")
    return tensor.contiguous()


def _optional_logits(value: Any, *, num_samples: int, context: str) -> Tensor | None:
    if value is None:
        return None
    tensor = value.detach().cpu().float() if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"{context}.logits 必须是 [N,C]，当前 shape={tuple(tensor.shape)}")
    if int(tensor.shape[0]) != int(num_samples):
        raise ValueError(f"{context}.logits 样本数必须等于 embedding 样本数，当前 {int(tensor.shape[0])} vs {num_samples}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{context}.logits 包含 NaN 或 Inf")
    return tensor.contiguous()


def _manifest(
    *,
    sample_ids: Sequence[str],
    quant_mode: str,
    granularity: str,
    embedding_dim: int,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "cache_version": CACHE_VERSION,
        "cache_type": CACHE_TYPE,
        "stage": STAGE,
        "quant_mode": str(quant_mode),
        "granularity": str(granularity),
        "num_samples": int(len(sample_ids)),
        "embedding_dim": int(embedding_dim),
        "sample_order_hash": _sample_order_hash(sample_ids),
        "sample_order_hash_algorithm": "sha256-json-list",
        "required_fields": list(REQUIRED_CACHE_FIELDS),
    }
    if extra_manifest is not None:
        manifest.update(dict(extra_manifest))
    return manifest


def build_embedding_cache(
    *,
    sample_id: Sequence[Any] | Tensor,
    label: Sequence[Any] | Tensor,
    bucket: Sequence[Any] | Tensor | None = None,
    shot_group: Sequence[Any] | Tensor | None = None,
    embedding: Any,
    quant_mode: str,
    granularity: str,
    label_id: Any | None = None,
    logits: Any | None = None,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建语义几何 embedding cache，不做模型推理，也不计算任何几何指标。"""

    embedding_tensor = _embedding_matrix(embedding, name="embedding", expected_dim=expected_dim)
    num_samples = int(embedding_tensor.shape[0])
    sample_ids = _sample_id_list(sample_id, num_samples=num_samples, context="embedding_cache")
    labels = _as_list(label, name="label", num_samples=num_samples)
    buckets, shot_groups, tail_flags = _canonical_bucket_lists(
        bucket=bucket,
        shot_group=shot_group,
        num_samples=num_samples,
        context="embedding_cache",
    )
    label_id_tensor = _optional_label_id(label_id, num_samples=num_samples, context="embedding_cache")
    logits_tensor = _optional_logits(logits, num_samples=num_samples, context="embedding_cache")
    sample_id_to_index = {item: idx for idx, item in enumerate(sample_ids)}
    manifest = _manifest(
        sample_ids=sample_ids,
        quant_mode=str(quant_mode),
        granularity=str(granularity),
        embedding_dim=int(embedding_tensor.shape[1]),
        extra_manifest=extra_manifest,
    )
    cache: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "cache_type": CACHE_TYPE,
        "manifest": manifest,
        "sample_id": sample_ids,
        "sample_order_hash": manifest["sample_order_hash"],
        "sample_id_to_index": sample_id_to_index,
        "label": labels,
        "label_id": label_id_tensor,
        "bucket": buckets,
        "shot_group": shot_groups,
        "tail_included": tail_flags,
        "embedding": embedding_tensor,
        "quant_mode": str(quant_mode),
        "granularity": str(granularity),
    }
    if logits_tensor is not None:
        cache["logits"] = logits_tensor
    validate_embedding_cache(cache, expected_dim=expected_dim)
    return cache


def build_fp32_embedding_cache(
    *,
    sample_id: Sequence[Any] | Tensor,
    label: Sequence[Any] | Tensor,
    bucket: Sequence[Any] | Tensor | None = None,
    shot_group: Sequence[Any] | Tensor | None = None,
    embedding: Any,
    label_id: Any | None = None,
    logits: Any | None = None,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 FP32 embedding cache，quant_mode 固定为 FP32。"""

    return build_embedding_cache(
        sample_id=sample_id,
        label=label,
        bucket=bucket,
        shot_group=shot_group,
        embedding=embedding,
        quant_mode=FP32_QUANT_MODE,
        granularity=FP32_GRANULARITY,
        label_id=label_id,
        logits=logits,
        expected_dim=expected_dim,
        extra_manifest=extra_manifest,
    )


def _label_from_core_cache(core_cache: Mapping[str, Any], *, num_samples: int) -> list[Any]:
    """从 FP32 core cache 读取标签，并统一映射成 semantic cache 的 label 字段。"""

    for field in CORE_LABEL_CANDIDATES:
        if field in core_cache and core_cache[field] is not None:
            return _as_list(core_cache[field], name=f"core_cache.{field}", num_samples=num_samples)
    raise KeyError(f"FP32 core cache 必须包含 label 或 primary_label，用于生成标准字段 {LABEL_FIELD!r}")


def _core_manifest_summary(core_cache: Mapping[str, Any]) -> dict[str, Any]:
    manifest = core_cache.get("manifest", {})
    if manifest is None:
        manifest = {}
    if not isinstance(manifest, Mapping):
        raise TypeError("core_cache.manifest 必须是 mapping")
    return {
        "source_cache_type": manifest.get("cache_type", core_cache.get("cache_type")),
        "source_stage": manifest.get("stage"),
        "source_eval_csv": manifest.get("eval_csv"),
        "source_label_mapping": manifest.get("label_mapping"),
        "source_checkpoint_path": manifest.get("checkpoint_path"),
        "source_dataset_name": manifest.get("dataset_name"),
        "source_embedding_pooling": manifest.get("embedding_pooling"),
        "source_sample_order_hash": core_cache.get("sample_order_hash", manifest.get("sample_order_hash")),
    }


def build_fp32_embedding_cache_from_core_cache(
    core_cache: Mapping[str, Any],
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    require_core_cache_type: bool = True,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从 core_eval.pt 的内容构建 semantic geometry 标准 FP32 embedding cache。

    core cache 中的 primary_label 会被映射为标准字段 label；新 cache 不再输出
    primary_label。这里不做模型推理，也不计算任何几何指标。
    """

    if not isinstance(core_cache, Mapping):
        raise TypeError(f"core_cache 必须是 mapping，当前为 {type(core_cache)!r}")
    missing = [field for field in ("sample_id", "bucket", "embedding", "label_id") if field not in core_cache]
    if missing:
        raise KeyError(f"FP32 core cache 缺少必要字段: {missing}")

    manifest = core_cache.get("manifest", {})
    if require_core_cache_type:
        cache_type = manifest.get("cache_type") if isinstance(manifest, Mapping) else None
        if cache_type is None:
            cache_type = core_cache.get("cache_type")
        if str(cache_type) != CORE_CACHE_TYPE:
            raise ValueError(f"core_eval.pt cache_type 必须是 {CORE_CACHE_TYPE!r}，当前为 {cache_type!r}")

    embedding = _embedding_matrix(core_cache["embedding"], name="core_cache.embedding", expected_dim=expected_dim)
    num_samples = int(embedding.shape[0])
    sample_ids = _sample_id_list(core_cache["sample_id"], num_samples=num_samples, context="core_cache")
    labels = _label_from_core_cache(core_cache, num_samples=num_samples)
    buckets, shot_groups, _tail_flags = _canonical_bucket_lists(
        bucket=core_cache["bucket"],
        shot_group=core_cache.get("shot_group"),
        num_samples=num_samples,
        context="core_cache",
    )
    label_id = _optional_label_id(core_cache.get("label_id"), num_samples=num_samples, context="core_cache")
    logits = _optional_logits(core_cache.get("logits"), num_samples=num_samples, context="core_cache")

    core_hash = core_cache.get("sample_order_hash")
    if core_hash is None and isinstance(manifest, Mapping):
        core_hash = manifest.get("sample_order_hash")
    if core_hash is not None and str(core_hash) != _sample_order_hash(sample_ids):
        raise ValueError("core_eval.pt 的 sample_order_hash 与 sample_id 当前顺序不一致")

    manifest_extra = _core_manifest_summary(core_cache)
    if extra_manifest is not None:
        manifest_extra.update(dict(extra_manifest))
    return build_fp32_embedding_cache(
        sample_id=sample_ids,
        label=labels,
        bucket=buckets,
        shot_group=shot_groups,
        embedding=embedding,
        label_id=label_id,
        logits=logits,
        expected_dim=expected_dim,
        extra_manifest=manifest_extra,
    )


def load_fp32_embedding_cache_from_core_eval(
    core_eval_path: str | Path,
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    require_core_cache_type: bool = True,
) -> dict[str, Any]:
    """读取 core_eval.pt，并转换为 semantic geometry 标准 FP32 embedding cache。"""

    path = Path(core_eval_path)
    if not path.exists():
        raise FileNotFoundError(f"core_eval.pt 不存在: {path}")
    core_cache = load_fp32_core_cache(path, materialize=True, include_padding_mask=False, include_logits=True)
    if not isinstance(core_cache, Mapping):
        raise TypeError(f"core_eval.pt 必须保存 mapping，当前为 {type(core_cache)!r}")
    cache = build_fp32_embedding_cache_from_core_cache(
        core_cache,
        expected_dim=expected_dim,
        require_core_cache_type=require_core_cache_type,
        extra_manifest={"source_core_eval_path": str(path.expanduser().resolve())},
    )
    cache["source_core_eval_path"] = path
    return cache

def build_quant_embedding_cache(
    *,
    sample_id: Sequence[Any] | Tensor,
    label: Sequence[Any] | Tensor,
    bucket: Sequence[Any] | Tensor | None = None,
    shot_group: Sequence[Any] | Tensor | None = None,
    embedding: Any,
    quant_mode: str,
    granularity: str,
    label_id: Any | None = None,
    logits: Any | None = None,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建量化 embedding cache，例如 A4-only、W4A4、W4-only 或激活逐通道量化。"""

    if str(quant_mode) == FP32_QUANT_MODE:
        raise ValueError("量化 cache 的 quant_mode 不能是 FP32；FP32 请调用 build_fp32_embedding_cache()")
    return build_embedding_cache(
        sample_id=sample_id,
        label=label,
        bucket=bucket,
        shot_group=shot_group,
        embedding=embedding,
        quant_mode=quant_mode,
        granularity=granularity,
        label_id=label_id,
        logits=logits,
        expected_dim=expected_dim,
        extra_manifest=extra_manifest,
    )


def validate_embedding_cache(
    cache: Mapping[str, Any],
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    require_label_id: bool = False,
) -> dict[str, Any]:
    """检查单个 embedding cache 的字段、样本数、embedding 维度和 sample_id 唯一性。"""

    missing = [field for field in REQUIRED_CACHE_FIELDS if field not in cache]
    if missing:
        raise KeyError(f"embedding cache 缺少必要字段: {missing}")
    if int(cache.get("cache_version", CACHE_VERSION)) != CACHE_VERSION:
        raise ValueError(f"cache_version 不匹配: {cache.get('cache_version')} vs {CACHE_VERSION}")
    if str(cache.get("cache_type", CACHE_TYPE)) != CACHE_TYPE:
        raise ValueError(f"cache_type 必须是 {CACHE_TYPE!r}，当前为 {cache.get('cache_type')!r}")

    embedding = _embedding_matrix(cache["embedding"], name="cache.embedding", expected_dim=expected_dim)
    num_samples = int(embedding.shape[0])
    sample_ids = _sample_id_list(cache["sample_id"], num_samples=num_samples, context="cache")
    labels = _as_list(cache["label"], name="cache.label", num_samples=num_samples)
    buckets, shot_groups, tail_flags = _canonical_bucket_lists(
        bucket=cache["bucket"],
        shot_group=cache.get("shot_group"),
        num_samples=num_samples,
        context="cache",
    )
    label_id_tensor = _optional_label_id(cache.get("label_id"), num_samples=num_samples, context="cache")
    if require_label_id and label_id_tensor is None:
        raise KeyError("embedding cache 缺少 label_id，但当前流程要求 label_id")
    logits_tensor = _optional_logits(cache.get("logits"), num_samples=num_samples, context="cache")

    order_hash = _sample_order_hash(sample_ids)
    cached_hash = cache.get("sample_order_hash", cache.get("manifest", {}).get("sample_order_hash"))
    if cached_hash is not None and str(cached_hash) != order_hash:
        raise ValueError("embedding cache 的 sample_order_hash 与 sample_id 当前顺序不一致")

    manifest = cache.get("manifest", {})
    if manifest is not None and isinstance(manifest, Mapping):
        if manifest.get("quant_mode") is not None and str(manifest["quant_mode"]) != str(cache["quant_mode"]):
            raise ValueError("manifest.quant_mode 与 cache.quant_mode 不一致")
        if manifest.get("granularity") is not None and str(manifest["granularity"]) != str(cache["granularity"]):
            raise ValueError("manifest.granularity 与 cache.granularity 不一致")
        if manifest.get("embedding_dim") is not None and int(manifest["embedding_dim"]) != int(embedding.shape[1]):
            raise ValueError("manifest.embedding_dim 与 cache.embedding 维度不一致")
        if manifest.get("num_samples") is not None and int(manifest["num_samples"]) != num_samples:
            raise ValueError("manifest.num_samples 与 cache 样本数不一致")

    return {
        "sample_id": sample_ids,
        "label": labels,
        "label_id": label_id_tensor,
        "bucket": buckets,
        "shot_group": shot_groups,
        "tail_included": tail_flags,
        "embedding": embedding,
        "logits": logits_tensor,
        "quant_mode": str(cache["quant_mode"]),
        "granularity": str(cache["granularity"]),
        "sample_order_hash": order_hash,
        "embedding_dim": int(embedding.shape[1]),
        "num_samples": num_samples,
    }


def save_embedding_cache(
    cache: Mapping[str, Any],
    path: str | Path,
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
) -> Path:
    """保存 embedding cache。调用前会重新校验，避免坏 cache 落盘。"""

    info = validate_embedding_cache(cache, expected_dim=expected_dim)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(cache)
    payload["bucket"] = info["bucket"]
    payload["shot_group"] = info["shot_group"]
    payload["tail_included"] = info["tail_included"]
    payload["sample_order_hash"] = info["sample_order_hash"]
    payload["sample_id_to_index"] = {item: idx for idx, item in enumerate(info["sample_id"])}
    payload["manifest"] = dict(payload.get("manifest", {}))
    payload["manifest"].update(
        {
            "cache_version": CACHE_VERSION,
            "cache_type": CACHE_TYPE,
            "stage": STAGE,
            "quant_mode": info["quant_mode"],
            "granularity": info["granularity"],
            "num_samples": info["num_samples"],
            "embedding_dim": info["embedding_dim"],
            "sample_order_hash": info["sample_order_hash"],
            "sample_order_hash_algorithm": "sha256-json-list",
        }
    )
    torch.save(payload, path)
    return path


def load_embedding_cache(
    path: str | Path,
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    require_label_id: bool = False,
) -> dict[str, Any]:
    """读取并校验 embedding cache。"""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"embedding cache 不存在: {path}")
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, Mapping):
        raise TypeError(f"embedding cache 必须是 mapping，当前为 {type(cache)!r}")
    info = validate_embedding_cache(cache, expected_dim=expected_dim, require_label_id=require_label_id)
    result = dict(cache)
    result["bucket"] = info["bucket"]
    result["shot_group"] = info["shot_group"]
    result["tail_included"] = info["tail_included"]
    result["cache_path"] = path
    return result


def _index_for_alignment(source_ids: Sequence[str], target_ids: Sequence[str], *, context: str) -> Tensor:
    source_map = {sample_id: idx for idx, sample_id in enumerate(source_ids)}
    missing = [sample_id for sample_id in target_ids if sample_id not in source_map]
    extra = [sample_id for sample_id in source_ids if sample_id not in set(target_ids)]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"缺失样本: {missing[:8]}")
        if extra:
            details.append(f"额外样本: {extra[:8]}")
        raise ValueError(f"{context} evaluation set 与 FP32 reference 不一致；" + "；".join(details))
    return torch.tensor([source_map[sample_id] for sample_id in target_ids], dtype=torch.long)


def align_cache_to_reference(
    reference_cache: Mapping[str, Any],
    cache: Mapping[str, Any],
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    require_label_id: bool = False,
) -> dict[str, Any]:
    """按 FP32 reference 的 sample_id 顺序重排另一个 cache，并检查 metadata 一致性。"""

    ref = validate_embedding_cache(reference_cache, expected_dim=expected_dim, require_label_id=require_label_id)
    cur = validate_embedding_cache(cache, expected_dim=expected_dim, require_label_id=require_label_id)
    if ref["embedding_dim"] != cur["embedding_dim"]:
        raise ValueError(f"embedding 维度不一致: reference={ref['embedding_dim']} vs current={cur['embedding_dim']}")
    order = _index_for_alignment(cur["sample_id"], ref["sample_id"], context=str(cache.get("quant_mode", "cache")))

    aligned_embedding = cur["embedding"].index_select(0, order).contiguous()
    aligned_label = [cur["label"][int(idx)] for idx in order.tolist()]
    aligned_bucket = [cur["bucket"][int(idx)] for idx in order.tolist()]
    aligned_shot_group = [cur["shot_group"][int(idx)] for idx in order.tolist()]
    aligned_label_id = None if cur["label_id"] is None else cur["label_id"].index_select(0, order).contiguous()
    aligned_logits = None if cur["logits"] is None else cur["logits"].index_select(0, order).contiguous()

    if aligned_label != ref["label"]:
        raise ValueError("对齐后 label 与 FP32 reference 不一致，拒绝继续")
    if aligned_bucket != ref["bucket"]:
        raise ValueError("对齐后 bucket 与 FP32 reference 不一致，拒绝继续")
    if aligned_shot_group != ref["shot_group"]:
        raise ValueError("对齐后 shot_group 与 FP32 reference 不一致，拒绝继续")
    if require_label_id and ref["label_id"] is not None and aligned_label_id is not None:
        if not torch.equal(ref["label_id"], aligned_label_id):
            raise ValueError("对齐后 label_id 与 FP32 reference 不一致，拒绝继续")

    aligned = build_embedding_cache(
        sample_id=ref["sample_id"],
        label=aligned_label,
        bucket=aligned_bucket,
        shot_group=aligned_shot_group,
        embedding=aligned_embedding,
        quant_mode=cur["quant_mode"],
        granularity=cur["granularity"],
        label_id=aligned_label_id,
        logits=aligned_logits,
        expected_dim=ref["embedding_dim"],
        extra_manifest={
            "aligned_to_sample_order_hash": ref["sample_order_hash"],
            "source_sample_order_hash": cur["sample_order_hash"],
        },
    )
    if "cache_path" in cache:
        aligned["source_cache_path"] = cache["cache_path"]
    return aligned


def align_embedding_caches(
    fp32_cache: Mapping[str, Any],
    quant_caches: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    require_label_id: bool = False,
) -> dict[str, Any]:
    """把多个量化 cache 统一对齐到同一份 FP32 cache。"""

    fp32_info = validate_embedding_cache(fp32_cache, expected_dim=expected_dim, require_label_id=require_label_id)
    if fp32_info["quant_mode"] != FP32_QUANT_MODE:
        raise ValueError(f"fp32_cache.quant_mode 必须是 {FP32_QUANT_MODE!r}")

    if isinstance(quant_caches, Mapping):
        items = list(quant_caches.items())
    else:
        items = [(str(cache.get("quant_mode", idx)), cache) for idx, cache in enumerate(quant_caches)]

    aligned_quant: dict[str, dict[str, Any]] = {}
    for name, cache in items:
        aligned = align_cache_to_reference(
            fp32_cache,
            cache,
            expected_dim=expected_dim,
            require_label_id=require_label_id,
        )
        key = str(name)
        if key in aligned_quant:
            raise ValueError(f"量化 cache 名称重复: {key}")
        aligned_quant[key] = aligned

    return {
        "fp32_cache": dict(fp32_cache),
        "quant_caches": aligned_quant,
        "sample_id": fp32_info["sample_id"],
        "label": fp32_info["label"],
        "label_id": fp32_info["label_id"],
        "bucket": fp32_info["bucket"],
        "shot_group": fp32_info["shot_group"],
        "tail_included": fp32_info["tail_included"],
        "embedding_dim": fp32_info["embedding_dim"],
        "sample_order_hash": fp32_info["sample_order_hash"],
    }


def load_and_align_embedding_caches(
    fp32_cache_path: str | Path,
    quant_cache_paths: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    require_label_id: bool = False,
) -> dict[str, Any]:
    """读取 FP32 与多个量化 embedding cache，并在 cache 层完成 sample_id 对齐。"""

    fp32_cache = load_embedding_cache(fp32_cache_path, expected_dim=expected_dim, require_label_id=require_label_id)
    if isinstance(quant_cache_paths, Mapping):
        quant_caches = {
            str(name): load_embedding_cache(path, expected_dim=expected_dim, require_label_id=require_label_id)
            for name, path in quant_cache_paths.items()
        }
    else:
        quant_caches = [
            load_embedding_cache(path, expected_dim=expected_dim, require_label_id=require_label_id)
            for path in quant_cache_paths
        ]
    return align_embedding_caches(
        fp32_cache,
        quant_caches,
        expected_dim=expected_dim,
        require_label_id=require_label_id,
    )


def semantic_geometry_inputs(
    fp32_cache: Mapping[str, Any],
    quant_cache: Mapping[str, Any],
    *,
    expected_dim: int | None = DEFAULT_EMBEDDING_DIM,
    require_label_id: bool = True,
) -> dict[str, Any]:
    """为 semantic_geometry_metrics.py 准备已对齐输入，本函数仍不计算指标。"""

    fp32 = validate_embedding_cache(fp32_cache, expected_dim=expected_dim, require_label_id=require_label_id)
    aligned_quant = align_cache_to_reference(
        fp32_cache,
        quant_cache,
        expected_dim=expected_dim,
        require_label_id=require_label_id,
    )
    quant = validate_embedding_cache(aligned_quant, expected_dim=expected_dim, require_label_id=require_label_id)
    return {
        "sample_id": fp32["sample_id"],
        "label": fp32["label"],
        "label_id": fp32["label_id"],
        "bucket": fp32["bucket"],
        "shot_group": fp32["shot_group"],
        "tail_included": fp32["tail_included"],
        "fp32_embeddings": fp32["embedding"],
        "quant_embeddings": quant["embedding"],
        "fp32_logits": fp32["logits"],
        "quant_logits": quant["logits"],
        "quant_mode": quant["quant_mode"],
        "granularity": quant["granularity"],
        "sample_order_hash": fp32["sample_order_hash"],
    }


__all__ = [
    "CACHE_TYPE",
    "CORE_CACHE_TYPE",
    "CORE_LABEL_CANDIDATES",
    "CACHE_VERSION",
    "DEFAULT_EMBEDDING_DIM",
    "FP32_GRANULARITY",
    "FP32_QUANT_MODE",
    "LABEL_FIELD",
    "REQUIRED_CACHE_FIELDS",
    "STAGE",
    "align_cache_to_reference",
    "align_embedding_caches",
    "build_embedding_cache",
    "build_fp32_embedding_cache",
    "build_fp32_embedding_cache_from_core_cache",
    "build_quant_embedding_cache",
    "load_and_align_embedding_caches",
    "load_fp32_embedding_cache_from_core_eval",
    "load_embedding_cache",
    "save_embedding_cache",
    "semantic_geometry_inputs",
    "validate_embedding_cache",
]
